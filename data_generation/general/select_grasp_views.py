"""Choose the grasp and camera for every view of every object, for make_data.py --grasp-views. Run in the grasp env.

Per object (one per worker):
  1. grasps: from grasp_pipeline's final set (tables/grasps_final.parquet). 8 per size bin by default, distinct types
     first, random within (bin, type); cycled when an object has fewer grasps than views. A grasp whose required
     fingers are farther than --max-render-gap from the rendered (original) mesh is replaced.
  2. hand: by default the settled pose after Dexonomy's closing motion (pregrasp -> grasp -> squeeze in MuJoCo,
     before validation's forces), mapped into render_blender.py's normalized object frame; its contacts are saved
     for the touch step.
  3. cameras: --candidates per view, evenly spread directions, FOV from the TRELLIS distribution, distance solved so
     the object and the whole hand fit (camera aimed at the object centre).
  4. low-res ray cast of every candidate (Warp): object share, hand share, occluded fraction, contacts visible.
  5. --view-stats-only: write the candidate stats (and low-res masks) and stop. Otherwise apply the limits, pick one
     candidate per view and write views.json.

Outputs in DATA_ROOT/grasp_views/: hand_template.npz, <object_id>/candidates.npz, <object_id>/views.json,
object_ids.json (for make_data.py --object-ids), settings.json, summary.json.
"""
import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
import warp as wp

wp.config.quiet = True
import mujoco  # noqa: E402
import dexonomy  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from dexonomy.sim import HandCfg, MuJoCo_EvalCfg, MuJoCo_EvalEnv, MuJoCo_VisEnv  # noqa: E402
from dexonomy.util.file_util import load_json, load_scene_cfg  # noqa: E402
from dexonomy.util.traj_util import get_planner  # noqa: E402

DEX_ASSETS = Path(dexonomy.__path__[0]).parent / "assets" / "hand"
CUBE_RADIUS = np.sqrt(3) / 2  # render_blender.py: radius of the normalized unit cube's bounding sphere
CONTACT_THRE = 0.002  # Dexonomy's contact distance, as in grasp_pipeline build_grasp_table.py
SETTINGS = ("num_views", "candidates", "stats_res", "margin", "visibility_tolerance", "seed", "hand", "hand_pose")


# ---------- frames ----------

def object_id(path, root):
    """make_data.get_objects naming: <sanitized stem>-<sha256(relative path)[:12]>."""
    import hashlib
    relative = path.relative_to(root).as_posix()
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")[:64] or "object"
    return f"{name}-{hashlib.sha256(relative.encode()).hexdigest()[:12]}", relative


def render_transform(loaded_vertices):
    """render_blender.normalize_scene: largest extent 1, bbox centred. x_render = x_loaded * s + offset."""
    lo, hi = loaded_vertices.min(0), loaded_vertices.max(0)
    s = 1.0 / (hi - lo).max()
    return s, -(lo + hi) * 0.5 * s


def hand_affine(stage1_normalize, s_r, off_r, obj_scale):
    """Dexonomy world (m, object at the origin scaled by obj_scale) -> render frame: x = a * p + b."""
    c1, k1 = np.asarray(stage1_normalize["center"]), stage1_normalize["scale"]
    return k1 * s_r / obj_scale, c1 * s_r + off_r


# ---------- cameras ----------

def radical_inverse(base, n):
    v, d = 0.0, 1.0 / base
    while n > 0:
        v += (n % base) * d
        n //= base
        d /= base
    return v


def camera_angles(index, num, offset):  # render_blender.camera_angles
    u = index / num + offset[0] / num
    v = radical_inverse(2, index) + offset[1]
    u = 2.0 * u if u < 0.25 else (2.0 / 3.0) * u + 1.0 / 3.0
    return (v % 1.0) * 2 * np.pi, np.arccos(1.0 - 2.0 * u) - np.pi / 2


def sample_fovs(rng, n):
    """render_blender trellis2 camera mode: inverse-squared distance, FOV fitting the unit cube at that distance."""
    r_min = CUBE_RADIUS / np.sin(np.deg2rad(70 / 2))
    r_max = CUBE_RADIUS / np.sin(np.deg2rad(10 / 2))
    r = 1 / np.sqrt(rng.uniform(1 / r_max**2, 1 / r_min**2, n))
    return np.rad2deg(2 * np.arcsin(CUBE_RADIUS / r)), r


def camera_basis(direction):
    """Camera on +direction looking at the origin, world z up. Returns right, up, forward."""
    f = -direction / np.linalg.norm(direction)
    up_ref = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up_ref)
    r /= np.linalg.norm(r)
    return r, np.cross(r, f), f


def fit_distance(points, direction, fov_deg, margin):
    """Smallest distance along direction at which every point projects inside the frame shrunk by margin."""
    r, u, f = camera_basis(direction)
    t = np.tan(np.deg2rad(fov_deg) / 2) * (1 - margin)
    return float((points @ -f + np.maximum(np.abs(points @ r), np.abs(points @ u)) / t).max())


def blender_matrix_world(direction, distance):
    r, u, f = camera_basis(direction)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = r, u, -f
    M[:3, 3] = direction / np.linalg.norm(direction) * distance
    return M


# ---------- Warp queries ----------

@wp.kernel
def ray_kernel(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), dirs: wp.array(dtype=wp.vec3),
               out: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, origins[i], dirs[i], 1.0e6)
    if q.result:
        out[i] = q.t
    else:
        out[i] = 1.0e10


@wp.kernel
def closest_kernel(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), out_p: wp.array(dtype=wp.vec3),
                   out_d: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh, pts[i], 1.0e6)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        out_p[i] = cp
        out_d[i] = wp.length(cp - pts[i])
    else:
        out_p[i] = pts[i]
        out_d[i] = 1.0e10


def wp_mesh(v, f, device):
    return wp.Mesh(points=wp.array(np.asarray(v, np.float32), dtype=wp.vec3, device=device),
                   indices=wp.array(np.asarray(f, np.int32).ravel(), dtype=int, device=device))


def cast(mesh, origins, dirs, device):
    out = wp.zeros(len(dirs), dtype=float, device=device)
    wp.launch(ray_kernel, dim=len(dirs), inputs=[mesh.id, wp.array(origins, dtype=wp.vec3, device=device),
                                                 wp.array(dirs, dtype=wp.vec3, device=device), out], device=device)
    return out.numpy()


def closest(mesh, pts, device):
    out_p = wp.zeros(len(pts), dtype=wp.vec3, device=device)
    out_d = wp.zeros(len(pts), dtype=float, device=device)
    wp.launch(closest_kernel, dim=len(pts), inputs=[mesh.id, wp.array(np.asarray(pts, np.float32), dtype=wp.vec3,
                                                                       device=device), out_p, out_d], device=device)
    return out_p.numpy(), out_d.numpy()


def pixel_rays(cams, res):
    """Pixel-centre rays for each (matrix_world, fov); row 0 at the image top, as in the rendered images."""
    c = (res - 1) / 2
    jj, ii = np.meshgrid(np.arange(res), np.arange(res))
    origins, dirs = [], []
    for M, fov in cams:
        fl = 0.5 * res / np.tan(np.deg2rad(fov) / 2)
        d = ((jj - c) / fl)[..., None] * M[:3, 0] - ((ii - c) / fl)[..., None] * M[:3, 1] - M[:3, 2]
        dirs.append((d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3))
        origins.append(np.broadcast_to(M[:3, 3], (res * res, 3)))
    return np.concatenate(origins).astype(np.float32), np.concatenate(dirs).astype(np.float32)


def point_visibility(obj_mesh, hand_mesh, cams, pts, device, tol):
    """Per camera: fraction of pts visible against the object only, and against object + hand."""
    if len(pts) == 0:
        return np.full(len(cams), np.nan), np.full(len(cams), np.nan)
    origins = np.concatenate([np.broadcast_to(M[:3, 3], pts.shape) for M, _ in cams]).astype(np.float32)
    vec = np.concatenate([pts] * len(cams)) - origins
    dist = np.linalg.norm(vec, axis=1)
    dirs = (vec / dist[:, None]).astype(np.float32)
    vis_obj = (cast(obj_mesh, origins, dirs, device) > dist - tol).reshape(len(cams), -1)
    vis_all = vis_obj & (cast(hand_mesh, origins, dirs, device) > dist - tol).reshape(len(cams), -1)
    return vis_obj.mean(1), vis_all.mean(1)


# ---------- per object ----------

WORKER = {}


def init_worker(hand):
    wp.config.quiet = True  # one Warp banner per worker otherwise
    wp.init()
    WORKER["env"] = MuJoCo_VisEnv(hand_cfg=HandCfg(xml_path=str(DEX_ASSETS / hand / "right.xml"), freejoint=True),
                                  vis_mode="visual")
    WORKER["groups"] = {}
    WORKER["eval"] = dexonomy_eval_settings()


def required_groups(grasp_dir, hand, t):
    if t not in WORKER["groups"]:
        tmpl = np.load(Path(grasp_dir) / "templates" / hand / f"{t}.npy", allow_pickle=True).item()
        WORKER["groups"][t] = tmpl["required_cbody"]
    return WORKER["groups"][t]


def dexonomy_eval_settings():
    """Validation settings from Dexonomy's configs, as grasp_pipeline's validate_grasps.py reads them."""
    cfg_dir = Path(dexonomy.__path__[0]) / "config"
    base, ev = OmegaConf.load(cfg_dir / "base.yaml"), OmegaConf.load(cfg_dir / "op/eval.yaml")
    return dict(obj_mass=float(base.obj_mass), miu_coef=list(ev.miu_coef), hard=bool(ev.hard),
                skip_pregrasp=bool(ev.skip_pregrasp))


def eval_scene(scene_path, hand, st):
    """The validation scene exactly as op/eval_data.py builds it (density from the info file, friction)."""
    scene = load_scene_cfg(str(scene_path))
    for o in scene["scene"].values():
        if o["type"] == "rigid_object":
            info = load_json(o["info_path"])
            o["density"] = st["obj_mass"] / (info["mass"] / (info["density"] * info["scale"] ** 3) * np.prod(o["scale"]))
    env = MuJoCo_EvalEnv(hand_cfg=HandCfg(str(DEX_ASSETS / hand / "right.xml"), freejoint=True), scene_cfg=scene,
                         sim_cfg=MuJoCo_EvalCfg(miu_coef=st["miu_coef"]), debug_render=False, debug_view=False,
                         hard=st["hard"])
    return env, scene, np.copy(env.get_active_obj_pose())


def settle_grasp(z, g, env, scene, init_pose, st):
    """Hand state after Dexonomy's closing motion (pregrasp -> grasp -> squeeze, no external force), the moment
    before validation applies its forces: the fingers rest on the object. Returns the hand qpos and the
    hand-object contacts in the object frame (object at the origin, as the grasp files), and how far the object moved."""
    pre = z["grasp_qpos"][g] if st["skip_pregrasp"] else np.asarray(z["pregrasp_qpos"][g])
    ctrl, ctype, ext, _ = get_planner(scene["task"]).plan_trajectory(
        np.copy(init_pose), pre, z["grasp_qpos"][g], z["squeeze_qpos"][g], None)
    env.reset_qpos(ctrl[0])
    env.run_trajectory(ctrl, ctype, ext)
    hand, obj = env.get_hand_qpos().astype(np.float64).copy(), env.get_active_obj_pose().astype(np.float64)
    ho, _ = env.get_contacts(obj_margin=CONTACT_THRE)  # contacts within Dexonomy's 2 mm, as the grasp files
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, obj[3:])
    R = R.reshape(3, 3)
    conj, q = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(conj, obj[3:])
    mujoco.mju_mulQuat(q, conj, hand[3:7])
    hand[:3], hand[3:7] = R.T @ (hand[:3] - obj[:3]), q
    contacts = dict(pos=(np.asarray(ho["pos"]).reshape(-1, 3) - obj[:3]) @ R,
                    normal=np.asarray(ho["normal"]).reshape(-1, 3) @ R,
                    dist=np.asarray(ho["dist"]).reshape(-1), bn1=list(ho["bn1"]))
    d = np.zeros(4)
    mujoco.mju_mulQuat(d, conj, init_pose[3:])
    motion = dict(object_shift_m=float(np.linalg.norm(obj[:3] - init_pose[:3])),
                  object_turn_deg=float(np.rad2deg(2 * np.arccos(min(1.0, abs(d[0]))))))
    return hand, contacts, motion


def grasp_contacts(c, a, b, obj_mesh, groups, device):
    """Required-finger contacts in the render frame: finger points, the surface points under them, normals, and
    the worst required group's distance to the rendered mesh in metres (same contact rule as build_grasp_table)."""
    pos, dist, normal = np.asarray(c["pos"]), np.asarray(c["dist"]), np.asarray(c["normal"])
    bodies = np.asarray(c["bn1"])
    near = dist < CONTACT_THRE
    finger = (pos + 0.5 * dist[:, None] * normal) * a + b  # point on the finger surface, render frame
    surface, d = closest(obj_mesh, finger, device)
    gaps = [float(d[near & np.isin(bodies, grp)].min()) / a if (near & np.isin(bodies, grp)).any() else np.inf
            for grp in groups]
    return dict(finger=finger[near], surface=surface[near], normal=normal[near], dist_m=dist[near],
                bodies=bodies[near].tolist(), render_gap_m=max(gaps) if gaps else np.inf)


def select_grasps(table_rows, n_views, rng, check):
    """table_rows: (row, size_bin, type) of the object's final grasps. check(row) -> True to accept."""
    pools = {}
    for row, b, t in table_rows:
        pools.setdefault((b, t), []).append(row)
    for k in pools:
        rng.shuffle(pools[k])
    bins = sorted({k[0] for k in pools})
    used, rejected, accepted = set(), [], []

    def next_free(k):
        while pools[k]:
            row = pools[k].pop(0)
            if row in used:
                continue
            used.add(row)
            if check(row):
                return row
            rejected.append(row)
        return None

    def take(keys, n):
        out, keys = [], list(keys)
        rng.shuffle(keys)
        while len(out) < n and keys:
            for k in list(keys):
                if len(out) >= n:
                    break
                row = next_free(k)
                if row is None:
                    keys.remove(k)
                else:
                    out.append(row)
        return out

    for b in bins:
        accepted += take([k for k in pools if k[0] == b], n_views // len(bins))
    accepted += take(list(pools), n_views - len(accepted))  # bins short of grasps: fill from any bin
    distinct = list(accepted)
    reused = [False] * len(accepted)
    i = 0
    while distinct and len(accepted) < n_views:  # fewer grasps than views: cycle evenly
        accepted.append(distinct[i % len(distinct)])
        reused.append(True)
        i += 1
    return accepted, reused, rejected


def limits_broken(st, args):
    """Per candidate: how many of the limits it breaks (0 = passes)."""
    broken = np.zeros(len(st["view"]), int)
    for key, value, too_low in (("object_share", args.min_object_share, True),
                                ("object_share", args.max_object_share, False),
                                ("hand_share", args.min_hand_share, True),
                                ("occluded", args.max_occluded, False),
                                ("occluded", args.min_occluded, True)):
        if value is not None:
            broken += (st[key] < value) if too_low else (st[key] > value)
    return broken


def passes_limits(st, args):
    return limits_broken(st, args) == 0


def select_views(st, args, rng):
    """One candidate per view. Returns candidate indices and whether each view had to fall back."""
    broken = limits_broken(st, args)
    ok = broken == 0
    edges = np.array(args.occlusion_bins)
    nb = len(edges) - 1
    bin_of = np.clip(np.digitize(st[args.balance_on], edges) - 1, 0, nb - 1)
    n_views = int(st["view"].max()) + 1
    targets = np.tile(np.arange(nb), n_views // nb + 1)[:n_views]
    rng.shuffle(targets)
    sel, fallback = [], []
    for v in range(n_views):
        m = st["view"] == v
        cand = np.where(m & ok)[0]
        if len(cand) == 0:  # nothing passes the limits: a candidate breaking the fewest of them
            cand = np.where(m)[0]
            cand = cand[broken[cand] == broken[cand].min()]
            sel.append(int(st["candidate"][rng.choice(cand)]))
            fallback.append(True)
            continue
        if args.selection == "balanced":
            bins = bin_of[cand]
            avail = np.unique(bins)
            cand = cand[bins == avail[np.argmin(np.abs(avail - targets[v]))]]
        sel.append(int(st["candidate"][rng.choice(cand)]))
        fallback.append(False)
    return sel, fallback


def process_object(task):
    uid, table_rows, args = task
    t0 = time.perf_counter()
    env, device = WORKER["env"], args.device
    out_root = Path(args.data_root) / "grasp_views"
    rec = json.loads((Path(args.grasp_dir) / "objects" / uid / "object_record.json").read_text())
    s1_dir = Path(rec["stage1"]["dir"])
    s1 = json.loads((s1_dir / "record.json").read_text())
    oid, relative = object_id(Path(s1["source"]).resolve(), Path(args.objects_root).resolve())
    od = out_root / oid
    if (od / ("candidates.npz" if args.view_stats_only else "views.json")).exists():
        return dict(uid=uid, object_id=oid, status="already done")
    rng = np.random.RandomState([args.seed, int(uid[:8], 16)])
    z = np.load(Path(args.grasp_dir) / "grasps" / f"{uid}.npz", allow_pickle=True)
    loaded = np.load(s1_dir / "loaded_mesh.npz")
    s_r, off_r = render_transform(loaded["vertices"])
    obj_v = loaded["vertices"] * s_r + off_r
    obj_mesh = wp_mesh(obj_v, loaded["faces"], device)
    obj_hull = trimesh.Trimesh(obj_v, loaded["faces"], process=False).convex_hull.vertices

    contacts, hand_qpos, motion, scenes = {}, {}, {}, {}

    def check(row):
        a, b = hand_affine(s1["clean"]["normalize"], s_r, off_r, float(z["scale"][row]))
        if args.hand_pose == "settled":
            size = int(z["size_bin"][row])
            if size not in scenes:  # one validation scene per object size, as validate_grasps.py
                scenes[size] = eval_scene(Path(args.grasp_dir) / "objects" / uid / "scene_cfg" / f"size{size}.npy",
                                          args.hand, WORKER["eval"])
            hand_qpos[row], c, motion[row] = settle_grasp(z, row, *scenes[size], WORKER["eval"])
        else:  # the stored grasp pose and its contacts, before the squeeze
            hand_qpos[row], c, motion[row] = z["grasp_qpos"][row].astype(np.float64), z["contacts"][row], {}
        contacts[row] = grasp_contacts(c, a, b, obj_mesh, required_groups(args.grasp_dir, args.hand,
                                                                           str(z["type"][row])), device)
        return args.max_render_gap is None or contacts[row]["render_gap_m"] <= args.max_render_gap

    picked, reused, rejected = select_grasps(table_rows, args.num_views, rng, check)
    if not picked:
        return dict(uid=uid, object_id=oid, error="no grasp passes the render-gap check",
                    rejected=len(rejected))
    R = args.stats_res
    rows, views, masks = [], [], []
    for v, (g, re_used) in enumerate(zip(picked, reused)):
        a, b = hand_affine(s1["clean"]["normalize"], s_r, off_r, float(z["scale"][g]))
        xmat, xpos = env.forward_kinematics(hand_qpos[g])
        hand_m = env.get_posed_mesh(xmat, xpos, "hand")
        hand_v = hand_m.vertices * a + b
        hand_mesh = wp_mesh(hand_v, hand_m.faces, device)
        _, hd = closest(obj_mesh, hand_v[:: max(1, len(hand_v) // 4000)], device)
        pts = np.concatenate([obj_hull, trimesh.Trimesh(hand_v, hand_m.faces, process=False).convex_hull.vertices])
        offset = rng.rand(2)
        fovs, _ = sample_fovs(rng, args.candidates)
        cams = []
        for c in range(args.candidates):
            yaw, pitch = camera_angles(c, args.candidates, offset)
            d = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
            cams.append((blender_matrix_world(d, fit_distance(pts, d, fovs[c], args.margin)), float(fovs[c])))
        origins, dirs = pixel_rays(cams, R)
        t_obj = cast(obj_mesh, origins, dirs, device).reshape(-1, R, R)
        t_hand = cast(hand_mesh, origins, dirs, device).reshape(-1, R, R)
        obj_hit, hand_hit = t_obj < 1e9, t_hand < 1e9
        hidden = obj_hit & hand_hit & (t_hand < t_obj)
        hand_seen = hand_hit & ~(obj_hit & (t_obj < t_hand))
        cvis_obj, cvis_all = point_visibility(obj_mesh, hand_mesh, cams, contacts[g]["surface"], device,
                                              args.visibility_tolerance)
        border = np.zeros((R, R), bool)
        border[0] = border[-1] = border[:, 0] = border[:, -1] = True
        n_obj = obj_hit.sum((1, 2))
        for c, (M, fov) in enumerate(cams):
            rows.append(dict(view=v, candidate=c, row=g, size_bin=int(z["size_bin"][g]), reused=re_used,
                             fov=fov, distance=float(np.linalg.norm(M[:3, 3])),
                             object_share=n_obj[c] / R**2,
                             object_visible_share=(obj_hit[c] & ~hidden[c]).sum() / R**2,
                             hand_share=hand_seen[c].sum() / R**2,
                             occluded=hidden[c].sum() / max(n_obj[c], 1),
                             contacts_visible_object_only=cvis_obj[c], contacts_visible=cvis_all[c],
                             object_on_border=bool((obj_hit[c] & border).any()),
                             hand_on_border=bool((hand_hit[c] & border).any()),
                             hand_gap=float(hd.min()), render_gap_m=contacts[g]["render_gap_m"]))
        views.append(dict(row=g, reused=re_used, a=a, b=b, xmat=xmat, xpos=xpos, cams=cams))
        if args.view_stats_only:
            masks.append(np.packbits(np.stack([obj_hit, hand_seen, hidden], 1), axis=-1))
    st = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    od.mkdir(parents=True, exist_ok=True)
    info = dict(uid=uid, object_id=oid, views=len(picked), distinct_grasps=len(set(picked)),
                reused=int(sum(reused)), rejected_render_gap=len(rejected),
                max_hand_gap=float(st["hand_gap"].max()),
                max_squeeze_shift_mm=round(1000 * max((m.get("object_shift_m", 0) for m in motion.values()), default=0), 2),
                seconds=round(time.perf_counter() - t0, 2))
    if args.view_stats_only:
        np.save(od / "masks_lowres.npy", np.stack(masks))  # (views, candidates, 3, R, R/8) packed bits
        np.savez_compressed(od / "candidates.npz", **st)
        return info

    sel, fallback = select_views(st, args, rng)
    out_views = []
    for v, c in enumerate(sel):
        i = int(np.where((st["view"] == v) & (st["candidate"] == c))[0][0])
        w, g = views[v], views[v]["row"]
        M, fov = w["cams"][c]
        out_views.append(dict(
            view=v, candidate=c, fallback=fallback[v], grasp_row=g, reused=w["reused"], type=str(z["type"][g]),
            size_bin=int(z["size_bin"][g]), scale=float(z["scale"][g]),
            hand_pose=args.hand_pose, hand_qpos=hand_qpos[g].tolist(), grasp_qpos=z["grasp_qpos"][g].tolist(),
            squeeze_motion=motion[g], hand_a=float(w["a"]), hand_b=w["b"].tolist(),
            body_xmat=w["xmat"].tolist(), body_xpos=w["xpos"].tolist(),
            camera_matrix_world=M.tolist(), fov=fov,
            contacts=dict(frame="normalized_object", finger=contacts[g]["finger"].tolist(),
                          surface=contacts[g]["surface"].tolist(), normal=contacts[g]["normal"].tolist(),
                          dist_m=contacts[g]["dist_m"].tolist(), bodies=contacts[g]["bodies"]),
            stats={k: float(st[k][i]) for k in ("object_share", "object_visible_share", "hand_share", "occluded",
                                               "contacts_visible_object_only", "contacts_visible", "distance",
                                               "hand_gap", "render_gap_m")}))
    np.savez_compressed(od / "candidates.npz", **st)
    selection = {k: getattr(args, k) for k in ("selection", "balance_on", "occlusion_bins", "min_object_share",
                                                "max_object_share", "min_hand_share",
                                                "max_occluded", "min_occluded", "max_render_gap")}
    doc = dict(uid=uid, object_id=oid, source_relative_path=relative, source_sha256=s1["source_sha256"],
               coordinate_frame="normalized_object", s_render=s_r, offset_render=off_r.tolist(),
               selection=selection, views=out_views)
    tmp = od / "views.json.tmp"
    tmp.write_text(json.dumps(doc))
    tmp.replace(od / "views.json")
    info["fallback_views"] = int(sum(fallback))
    return info


def opt_float(s):
    return None if s == "none" else float(s)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grasp-dir", required=True, help="grasp_pipeline grasp_generation output folder")
    p.add_argument("--objects-root", required=True, help="the --objects-root given to make_data.py")
    p.add_argument("--data-root", required=True, help="the --data-root given to make_data.py")
    p.add_argument("--hand", default="shadow")
    p.add_argument("--num-views", type=int, default=32)
    p.add_argument("--candidates", type=int, default=64, help="candidate cameras per view")
    p.add_argument("--stats-res", type=int, default=64, help="low-res ray-cast resolution (multiple of 8)")
    p.add_argument("--margin", type=float, default=0.05, help="fraction of the half-frame kept empty")
    p.add_argument("--visibility-tolerance", type=float, default=0.005)
    p.add_argument("--view-stats-only", action="store_true", help="write candidate stats and low-res masks only")
    p.add_argument("--selection", choices=["random", "balanced"], default="balanced",
                   help="balanced: views spread evenly over --occlusion-bins; random: any candidate within the limits")
    p.add_argument("--balance-on", default="occluded")
    p.add_argument("--occlusion-bins", type=float, nargs="+", default=[0, .1, .2, .3, .4, .5, .6],
                   help="bin edges for balanced selection (chosen from the 100-object stats run)")
    p.add_argument("--min-object-share", type=opt_float, default=None, help="none = off")
    p.add_argument("--max-object-share", type=opt_float, default=0.35,
                   help="largest fraction of the image the object may cover (from the 100-object stats run); none = off")
    p.add_argument("--min-hand-share", type=opt_float, default=0.05,
                   help="smallest fraction of the image the visible hand must cover (from the 100-object stats run); none = off")
    p.add_argument("--max-occluded", type=opt_float, default=0.6,
                   help="largest fraction of the object hidden by the hand (from the 100-object stats run); none = off")
    p.add_argument("--min-occluded", type=opt_float, default=None, help="none = off")
    p.add_argument("--max-render-gap", type=opt_float, default=0.005,
                   help="m, worst required finger to the rendered mesh; failing grasps are replaced (same 5 mm as "
                        "filter_grasps.py's real-gap limit, which is measured on stage 1's closed mesh); none = off")
    p.add_argument("--hand-pose", choices=["settled", "grasp"], default="settled",
                   help="settled: fingers resting on the object after the squeeze (what the tactile step uses); "
                        "grasp: the stored pre-squeeze grasp pose")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, help="first N objects only")
    p.add_argument("--uids", nargs="+", help="these objects only")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.stats_res % 8:
        p.error("--stats-res must be a multiple of 8")
    t0 = time.time()
    out_root = Path(args.data_root) / "grasp_views"
    out_root.mkdir(parents=True, exist_ok=True)
    settings = {k: getattr(args, k) for k in SETTINGS}
    if (out_root / "settings.json").exists() and json.loads((out_root / "settings.json").read_text()) != settings:
        raise ValueError(f"settings differ from {out_root / 'settings.json'}; use a new --data-root")
    (out_root / "settings.json").write_text(json.dumps(settings, indent=1))

    t = pq.read_table(Path(args.grasp_dir) / "tables" / "grasps_final.parquet",
                      columns=["uid", "row", "size_bin", "type"]).to_pydict()
    by_uid = {}
    for u, r, b, ty in zip(t["uid"], t["row"], t["size_bin"], t["type"]):
        by_uid.setdefault(u, []).append((int(r), int(b), str(ty)))
    uids = sorted(by_uid)
    if args.uids:
        uids = [u for u in uids if u in set(args.uids)]
    if args.limit:
        uids = uids[: args.limit]

    init_worker(args.hand)  # hand template, from the same model the workers pose
    env = WORKER["env"]
    tmpl = {"names": np.array(env.get_all_body_names())}
    for i, name in enumerate(env.get_all_body_names()):
        m = env.get_body_mesh(name)
        tmpl[f"{i}_vertices"], tmpl[f"{i}_faces"] = m.vertices.astype(np.float32), m.faces.astype(np.int32)
        tmpl[f"{i}_body_id"] = np.int64(env.get_body_id(name))
    np.savez_compressed(out_root / "hand_template.npz", **tmpl)

    print(f"Selecting views for {len(uids)} objects | workers={args.workers} | "
          f"{'stats only' if args.view_stats_only else 'selection=' + args.selection}", flush=True)
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=init_worker, initargs=(args.hand,)) as pool:
        for i, res in enumerate(pool.imap_unordered(process_object_safe, [(u, by_uid[u], args) for u in uids]), 1):
            results.append(res)
            el = time.time() - t0
            print(f"Object {i}/{len(uids)} | " + " | ".join(f"{k}={v}" for k, v in res.items())
                  + f" | elapsed={el / 60:.1f}m | eta={el / i * (len(uids) - i) / 60:.1f}m", flush=True)

    errors = [r for r in results if "error" in r]
    if not args.view_stats_only:
        # make_data.py --object-ids: the objects with views (folder name = make_data's object id)
        ids = sorted(d.name for d in out_root.iterdir() if (d / "views.json").exists())
        (out_root / "object_ids.json").write_text(json.dumps(ids, indent=1))
        print(f"Object ids | {len(ids)} objects | {out_root / 'object_ids.json'}", flush=True)
    done = [r for r in results if "seconds" in r]
    summary = dict(objects=len(uids), processed=len(done), errors=errors,
                   already_done=sum(r.get("status") == "already done" for r in results),
                   seconds_per_object=float(np.median([r["seconds"] for r in done])) if done else None,
                   rejected_render_gap=sum(r.get("rejected_render_gap", 0) for r in done),
                   fallback_views=sum(r.get("fallback_views", 0) for r in done),
                   max_hand_gap=max((r["max_hand_gap"] for r in done), default=None))
    (out_root / ("summary_stats_only.json" if args.view_stats_only else "summary.json")).write_text(
        json.dumps(summary, indent=1))
    print("Done | " + json.dumps({k: v for k, v in summary.items() if k != "errors"} | {"errors": len(errors)}),
          flush=True)


def process_object_safe(task):
    try:
        return process_object(task)
    except Exception as e:
        return dict(uid=task[0], error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
