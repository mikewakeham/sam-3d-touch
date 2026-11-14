"""Grasp-contact touch: the object surface the settled hand presses on, as 8192 points per view. Run in the grasp env
(with --joint-pointmap: the sam3d-objects env, with warp-lang and fpsample installed).

Per view (all lengths in render_blender.py's normalized object frame, where the object's longest side is 1):
  1. pose the object (mesh.npz) and the settled hand (grasp_views/hand_template.npz + views.json body poses).
  2. candidates: dense points on the hand parts that MuJoCo reports touching, each moved to the closest point on the
     object surface (Warp). A second, denser pass covers the hand faces that produced touched points.
  3. touched: a candidate within --tolerance of those hand parts' surface (the skin squish). Default: Dexonomy's
     contact distance (2 mm), converted per object. Sharp yes/no.
  4. shapes: touched points are joined into pieces; gaps narrower than --gap are filled (surface closing), so nearby
     contact becomes one connected shape.
  5. confirm: only pieces that contain a saved MuJoCo contact point are kept.
  6. --points points by farthest-point sampling, shared between hand parts by touched area with at least
     --min-part-share of the budget each, then reordered by one global farthest-point pass (any prefix is spread out).

--joint-pointmap also saves the joint pointmap + touch input, as make_touch_data.py --joint-pointmap: 1024
farthest-point pointmap points then the first 7168 touch points, in the pointmap's normalized frame. Views whose touch
file already exists only get the joint arrays added (when missing).

Writes <view>/touch_grasp_contact.npz next to full_surface.npz and generated_data/samples_touch_grasp_contact.jsonl
(+ _camframe.jsonl when samples_camframe.jsonl exists). Views with no confirmed contact are listed in
generated_data/touch_grasp_contact_failures.json and left out of the manifests.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import fpsample
import numpy as np
import trimesh
import warp as wp
from scipy.ndimage import binary_closing, label
from scipy.spatial import cKDTree

from sample_full_surface import classify_visibility, sam_camera_transform, transform_normals, transform_points

wp.config.quiet = True
CONTACT_THRE = 0.002  # Dexonomy's contact distance in metres, as select_grasp_views.py
NAME = "touch_grasp_contact"
FORMAT_VERSION = 1
JOINT_TOUCH_POINTS = 7168  # as make_touch_data.py --joint-pointmap: 1024 pointmap + 7168 touch points
JOINT_KEYS = ("joint_points_camera", "joint_points_pre_encoder", "joint_pointmap_candidates_camera",
              "joint_pointmap_indices", "joint_pointmap_scale", "joint_pointmap_shift", "joint_source_sha256")


@wp.kernel
def closest_kernel(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), out_p: wp.array(dtype=wp.vec3),
                   out_f: wp.array(dtype=int), out_d: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh, pts[i], 1.0e6)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        out_p[i] = cp
        out_f[i] = q.face
        out_d[i] = wp.length(cp - pts[i])
    else:
        out_p[i] = pts[i]
        out_f[i] = -1
        out_d[i] = 1.0e10


def wp_mesh(v, f, device):
    return wp.Mesh(points=wp.array(np.asarray(v, np.float32), dtype=wp.vec3, device=device),
                   indices=wp.array(np.asarray(f, np.int32).ravel(), dtype=int, device=device))


def closest(mesh, pts, device):
    """Closest surface point, its face and the distance, for each of pts."""
    n = len(pts)
    out_p = wp.zeros(n, dtype=wp.vec3, device=device)
    out_f = wp.zeros(n, dtype=int, device=device)
    out_d = wp.zeros(n, dtype=float, device=device)
    if n:
        wp.launch(closest_kernel, dim=n, inputs=[mesh.id, wp.array(np.asarray(pts, np.float32), dtype=wp.vec3,
                                                                   device=device), out_p, out_f, out_d], device=device)
    return out_p.numpy().astype(np.float64), out_f.numpy(), out_d.numpy().astype(np.float64)


def sample_faces(mesh, faces, count, rng):
    """count area-uniform points on the given faces (trimesh.sample.sample_surface); returns points, their face
    ids, and area per point."""
    weight = np.zeros(len(mesh.faces))
    weight[faces] = mesh.area_faces[faces]
    if count == 0 or weight.sum() <= 0:
        return np.zeros((0, 3)), np.zeros(0, np.int64), 0.0
    pts, face = trimesh.sample.sample_surface(mesh, count, face_weight=weight, seed=int(rng.integers(2**31)))
    return pts, face, weight.sum() / count


def fps(points, count, rng):
    """Farthest-point sampling with fpsample (bucket QuickFPS; h=7, its documented setting for medium workloads),
    from a seeded random start. count == len(points) returns all points in farthest-point order."""
    count = min(count, len(points))
    if count < 2:
        return np.arange(count)
    h = min(7, int(np.log2(len(points))))  # fpsample requires 2**h <= number of points
    return fpsample.bucket_fps_kdline_sampling(points, count, h=h, start_idx=int(rng.integers(len(points)))).astype(np.int64)


def close_and_label(points, touched, radius):
    """Gap fill and pieces on one voxel grid (voxel = radius / 2, or 0.25% of object size when radius is 0):
    morphological closing of the touched voxels with a ball of the given radius (scipy.ndimage.binary_closing),
    then connected pieces of the closed voxels (scipy.ndimage.label, 26-connected). Returns the region mask over
    the candidates and each candidate's piece label (0 outside the region)."""
    voxel = radius / 2 if radius > 0 else 0.0025
    r = int(round(radius / voxel)) if radius > 0 else 0
    idx = np.floor((points - points.min(0)) / voxel).astype(np.int64) + r + 1
    grid = np.zeros(idx.max(0) + r + 2, bool)
    grid[tuple(idx[touched].T)] = True
    if r:
        z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
        grid = binary_closing(grid, structure=x * x + y * y + z * z <= r * r) | grid
    labels, _ = label(grid, structure=np.ones((3, 3, 3), bool))
    point_label = labels[tuple(idx.T)]
    return point_label > 0, point_label


def allocate(areas, counts, total, min_share):
    """Points per hand part: by touched area, at least min_share of total each, never more than a part has."""
    areas = np.asarray(areas, float)
    want = np.maximum(areas / areas.sum() * total, min_share * total)
    quota = np.minimum(np.floor(want / want.sum() * total).astype(np.int64), counts)
    while quota.sum() < min(total, counts.sum()):
        room = counts - quota
        open_ = room > 0
        extra = total - quota.sum()
        share = np.floor(want * open_ / (want * open_).sum() * extra).astype(np.int64)
        share = np.minimum(np.maximum(share, open_.astype(np.int64)), room)
        if share.sum() > extra:  # hand out the last few one by one, largest areas first
            share[:] = 0
            for i in np.argsort(-want * open_)[:extra]:
                share[i] = 1 if room[i] > 0 else 0
        quota += share
    return quota


def view_contact_touch(record, root, view, tmpl, obj, args, device, debug=False):
    """Touch points for one view (object frame) plus per-part stats; raises NoContact when nothing is confirmed.
    debug=True also returns every intermediate step (for view_contact_touch.py)."""
    a, b = float(view["hand_a"]), np.asarray(view["hand_b"], float)
    xmat, xpos = np.asarray(view["body_xmat"], float).reshape(-1, 3, 3), np.asarray(view["body_xpos"], float)
    c = view["contacts"]
    bodies = np.asarray(c["bodies"])
    contact_pts = np.asarray(c["surface"], float).reshape(-1, 3)
    tol = args.tolerance if args.tolerance is not None else CONTACT_THRE * a
    names = [str(n) for n in tmpl["names"]]
    touching = sorted({i for i, n in enumerate(names) for body in set(bodies.tolist()) if n.endswith(body)})
    if not touching:
        raise NoContact("no hand part touches")
    hv, hf, part_of_face = [], [], []
    offset = 0
    for k, i in enumerate(touching):
        bid = int(tmpl[f"{i}_body_id"])
        v = tmpl[f"{i}_vertices"].astype(np.float64) @ xmat[bid].T * a + a * xpos[bid] + b
        hv.append(v)
        hf.append(tmpl[f"{i}_faces"] + offset)
        part_of_face.append(np.full(len(tmpl[f"{i}_faces"]), k))
        offset += len(v)
    hv, hf, part_of_face = np.concatenate(hv), np.concatenate(hf), np.concatenate(part_of_face)
    hand = wp_mesh(hv, hf, device)
    hand_tm = trimesh.Trimesh(hv, hf, process=False)
    obj_v, obj_f, obj_mesh, obj_normals = obj
    # Per contact: its hand part and how far that part's rendered surface is from the contact. MuJoCo collides
    # simplified shapes, so a touching part can sit a little off the rendered mesh; --part-offset starts that part's
    # squish from there (tolerance + its closest contact gap) so every MuJoCo contact produces touch.
    contact_part = np.array([next((k for k, i in enumerate(touching) if names[i].endswith(body)), -1) for body in bodies])
    contact_gap = np.full(len(contact_pts), np.nan)
    for k in range(len(touching)):
        sel = contact_part == k
        if sel.any():
            part_mesh = wp_mesh(hv, hf[part_of_face == k], device)
            contact_gap[sel] = closest(part_mesh, contact_pts[sel], device)[2]
    part_gap = np.array([np.nanmin(contact_gap[contact_part == k]) if (contact_part == k).any() else 0.0
                         for k in range(len(touching))])
    part_tol = tol + (part_gap if args.part_offset else np.zeros(len(touching)))

    seed = int(hashlib.sha256(record["sample_id"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([args.seed, seed])
    reach = part_tol.max() + args.gap
    centroids = hv[hf].mean(1)
    radius = np.linalg.norm(hv[hf] - centroids[:, None], axis=2).max(1)
    _, _, d = closest(obj_mesh, centroids, device)
    near_faces = np.flatnonzero(d <= reach + radius)
    if len(near_faces) == 0:
        raise NoContact("touching hand parts are not near the rendered surface")

    def candidates(faces, count):
        """Hand samples on faces, moved to the closest object point; returns object point, object face, distance
        to the touching hand parts, nearest part, area per point, and the hand face each came from."""
        hp, source, w = sample_faces(hand_tm, faces, count, rng)
        p, f, d = closest(obj_mesh, hp, device)
        keep = d <= reach
        p, f, source = p[keep], f[keep], source[keep]
        _, hface, dh = closest(hand, p, device)
        return p, f, dh, part_of_face[hface], np.full(len(p), w), source

    first = candidates(near_faces, args.hand_samples)
    # Second, denser pass on the hand faces whose samples landed on touched surface; first-pass points from those
    # faces are dropped so each pass's area-per-point weight covers only its own faces.
    dense = np.unique(first[5][first[2] <= part_tol[first[3]]])
    if len(dense):
        second = candidates(dense, args.hand_samples)
        first = tuple(x[~np.isin(first[5], dense)] for x in first)
        pts, faces, dist, part, weight, _ = (np.concatenate([x, y]) for x, y in zip(first, second))
    else:
        pts, faces, dist, part, weight, _ = first
    touched = dist <= part_tol[part]
    if not touched.any():
        raise NoContact("no surface within the tolerance of the touching hand parts")

    region, piece = close_and_label(pts, touched, args.gap / 2)
    near = cKDTree(pts[region]).query_ball_point(contact_pts, max(args.gap, 0.0025))
    confirmed = np.unique(piece[region][np.concatenate([np.asarray(idx, np.int64) for idx in near])])
    keep = np.flatnonzero(region & np.isin(piece, confirmed))
    if len(keep) == 0:
        raise NoContact("no touched piece contains a MuJoCo contact point")

    parts_here = np.unique(part[keep])
    counts = np.array([np.sum(part[keep] == p) for p in parts_here])
    areas = np.array([weight[keep][part[keep] == p].sum() for p in parts_here])
    quota = allocate(areas, counts, args.points, args.min_part_share)
    chosen = []
    for p, q in zip(parts_here, quota):
        idx = keep[part[keep] == p]
        chosen.append(idx[fps(pts[idx], int(q), rng)])
    chosen = np.concatenate(chosen)
    chosen = chosen[fps(pts[chosen], len(chosen), rng)]
    hits = sum(len(idx) > 0 for idx in near)
    if debug:
        covered = np.zeros(len(contact_pts), bool)
        if len(keep):
            covered = cKDTree(pts[keep]).query(contact_pts, distance_upper_bound=args.gap)[0] < np.inf
        steps = dict(points=pts, distance=dist, part=part, part_tol=part_tol, touched=touched, region=region,
                     kept=keep, chosen=chosen, contact_part=contact_part, contact_gap=contact_gap,
                     contact_covered=covered, touching=touching, hand_vertices=hv, hand_faces=hf,
                     part_of_face=part_of_face)
    stats = dict(tolerance=float(tol), part_offset=bool(args.part_offset), part_gap=part_gap.tolist(), candidates=int(len(pts)), touched=int(touched.sum()), region=int(region.sum()),
                 kept=int(len(keep)), pieces=int(len(confirmed)), contacts=int(len(contact_pts)),
                 contacts_in_region=int(hits), parts=[names[touching[p]] for p in parts_here],
                 part_area=areas.tolist(), part_points=quota.tolist())
    result = (pts[chosen], obj_normals[faces[chosen]], np.asarray([touching[p] for p in part[chosen]], np.int16),
              names, stats)
    return result + (steps,) if debug else result


class NoContact(ValueError):
    pass


def object_mesh(v, f, device):
    """The object for touch: (vertices, faces, Warp mesh, face normals). Degenerate triangles (zero area, so no normal)
    are dropped: they add no surface, and a closest-point query landing on one would give a touch point without a
    normal."""
    tm = trimesh.Trimesh(v, f, process=False)
    normals = np.asarray(tm.face_normals, float)
    keep = np.isfinite(normals).all(1) & (np.linalg.norm(normals, axis=1) > 0.5) & (np.asarray(tm.area_faces) > 0)
    f = f[keep]
    return v, f, wp_mesh(v, f, device), normals[keep]


def mesh_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_view(record, root, path, points, normals, part, names, stats, obj_hash, args):
    transform = sam_camera_transform(root / record["camera_path"])
    with np.load(root / record["camera_path"], allow_pickle=False) as camera:
        K, T = camera["K"], camera["T_camera_from_object"]
    with np.load(root / record["full_surface_path"], allow_pickle=False) as surface:
        vis_tol = float(surface["visibility_tolerance"])
    depth = np.load(root / record["depth_path"], allow_pickle=False)
    arrays = dict(
        format_version=np.int64(FORMAT_VERSION), data_kind="grasp_contact", coordinate_frame="sam_camera",
        points_camera=transform_points(points, transform).astype(np.float32),
        normals_camera=transform_normals(normals, transform),
        points_object=points.astype(np.float32), normals_object=normals.astype(np.float32),
        part_index=part, part_names=np.array(names),
        point_visibility=classify_visibility(points, K, T, depth, vis_tol), visibility_tolerance=vis_tol,
        tolerance=np.float64(stats["tolerance"]), part_offset=np.bool_(args.part_offset),
        part_gap=np.array(stats["part_gap"]), gap=np.float64(args.gap),
        min_part_share=np.float64(args.min_part_share), hand_samples=np.int64(args.hand_samples),
        seed=np.int64(args.seed), contacts=np.int64(stats["contacts"]),
        contacts_in_region=np.int64(stats["contacts_in_region"]), pieces=np.int64(stats["pieces"]),
        part_area=np.array(stats["part_area"]), part_points=np.array(stats["part_points"]),
        mesh_sha256=obj_hash, numpy_version=np.__version__, warp_version=wp.__version__)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def has_joint(path):
    with np.load(path, allow_pickle=False) as saved:
        return all(key in saved.files for key in JOINT_KEYS)


def save_joint(record, root, path, args):
    """make_touch_data.py's save_joint + joint_arrays for one touch set: the view's image and pointmap preprocessed as
    training does, 1024 farthest-point pointmap points, then the first 7168 touch points (saved in farthest-point
    order; repeated cyclically when a view has fewer), in the pointmap's normalized frame. Adds the joint arrays to
    the view's touch file."""
    import os
    import sys
    import torch
    from omegaconf import OmegaConf
    from torchvision.transforms import InterpolationMode, functional as TF
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dataloader import TouchDataset
    from evaluation.input_visualizations import joint_camera_points
    from generate_target_latents import checkpoint_sha256
    from make_touch_data import EmptyPointmapMask, joint_preprocessor
    from sam3d_objects.model.backbone.dit.embedder.surface_encoder_utils import farthest_point_indices
    from train import normalize_touch_to_pointmap_frame, preprocess_pointmap_batch

    config_path = args.pipeline_config.resolve()
    config = OmegaConf.load(config_path)
    sources = [root / record[key] for key in ("image_path", "camera_path", "depth_path")]
    for key in ("pointmap_path", "object_mask_path"):
        if record.get(key):
            sources.append(root / record[key])
    sources.append(config_path)
    if config.get("ss_preprocessor") is None:
        sources.append(config_path.parent / config.ss_generator_config_path)
    fingerprint = hashlib.sha256("".join(checkpoint_sha256(source) for source in sources).encode()).hexdigest()
    dataset = object.__new__(TouchDataset)
    dataset.root = root
    rgba = dataset.load_rgba(record)
    pointmap = dataset.load_pointmap(record)
    # Match the normalizer's alpha threshold and nearest-neighbor resize.
    mask = torch.from_numpy((rgba[..., 3] / 255).astype(np.float32))[None]
    mask = TF.resize(mask, pointmap.shape[:2], interpolation=InterpolationMode.NEAREST)
    if not (mask > .5).any():
        raise EmptyPointmapMask(f"{record['sample_id']}: Empty foreground mask at pointmap resolution")
    preprocessor = joint_preprocessor(str(config_path))
    inputs = preprocess_pointmap_batch(preprocessor, torch.from_numpy(rgba)[None], torch.from_numpy(pointmap)[None], "cpu")
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved)
    touch_camera = arrays["points_camera"][np.arange(JOINT_TOUCH_POINTS) % len(arrays["points_camera"])]
    with torch.no_grad():
        points = inputs["pointmap"][0].permute(1, 2, 0).reshape(-1, 3)
        valid = (inputs["mask"][0].reshape(-1) > .5) & points.isfinite().all(dim=-1)
        points = points[valid]
        if not len(points):
            raise ValueError("No valid foreground pointmap points")
        if torch.cuda.is_available():  # PyTorch3D's GPU FPS; workers share the job's GPUs
            device = f"cuda:{os.getpid() % torch.cuda.device_count()}"
            chosen = farthest_point_indices(points[None].to(device), min(1024, len(points)))[0].cpu()
        else:
            chosen = farthest_point_indices(points[None], min(1024, len(points)))[0]
        chosen = chosen[torch.arange(1024, device=chosen.device) % len(chosen)]
        touch = torch.as_tensor(touch_camera)[None]
        touch_mask = torch.ones(touch.shape[:2], dtype=torch.bool)
        touch = normalize_touch_to_pointmap_frame(touch, touch_mask, inputs, preprocessor)[0]
        cloud = torch.cat((points[chosen], touch))
        restored = joint_camera_points(cloud, inputs, preprocessor)
        if not np.allclose(restored[1024:], touch_camera, atol=1e-5, rtol=1e-5):
            raise ValueError("Joint points do not round-trip to the saved touch points")
        arrays.update(joint_points_camera=np.asarray(restored, dtype=np.float32),
                      joint_points_pre_encoder=cloud.float().cpu().numpy(),
                      joint_pointmap_candidates_camera=joint_camera_points(points, inputs, preprocessor),
                      joint_pointmap_indices=chosen.cpu().numpy(),
                      joint_pointmap_scale=inputs["pointmap_scale"][0].float().cpu().numpy(),
                      joint_pointmap_shift=inputs["pointmap_shift"][0].float().cpu().numpy(),
                      joint_source_sha256=fingerprint)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def make_object(task):
    args, records = task
    root = args.data_root
    device = WORKER["device"]
    tmpl = WORKER["template"]
    outputs, failures, timings = [], [], []
    obj = None
    views = None
    for record in records:
        path = (root / record["full_surface_path"]).with_name(f"{NAME}.npz")
        if path.exists() and not args.overwrite:
            if args.joint_pointmap and not has_joint(path):
                try:
                    save_joint(record, root, path, args)
                except Exception as error:  # listed and left out of the manifests, as for touch errors
                    failures.append(dict(sample_id=record["sample_id"], reason=f"joint: {type(error).__name__}: {error}"))
                    continue
            outputs.append(dict(record, touch_path=str(path.relative_to(root))))
            continue
        started = time.perf_counter()
        if obj is None:
            with np.load(root / record["mesh_path"], allow_pickle=False) as m:
                if str(m["coordinate_frame"]) != "normalized_object":
                    raise ValueError(f"Unexpected mesh frame: {record['mesh_path']}")
                v, f = m["vertices"].astype(np.float64), m["faces"].astype(np.int64)
            obj = object_mesh(v, f, device)
            obj_hash = mesh_sha256(root / record["mesh_path"])
            views = json.loads((root / record["grasp_views_path"]).read_text())["views"]
        view = views[int(record["grasp_view_index"])]
        try:
            points, normals, part, names, stats = view_contact_touch(record, root, view, tmpl, obj, args, device)
            save_view(record, root, path, points, normals, part, names, stats, obj_hash, args)
            if args.joint_pointmap:
                save_joint(record, root, path, args)
        except NoContact as error:
            failures.append(dict(sample_id=record["sample_id"], reason=str(error)))
            continue
        except Exception as error:  # one bad view must not stop the run; it is listed and left out of the manifests
            failures.append(dict(sample_id=record["sample_id"], reason=f"error: {type(error).__name__}: {error}"))
            continue
        outputs.append(dict(record, touch_path=str(path.relative_to(root))))
        timings.append(dict(sample_id=record["sample_id"], seconds=time.perf_counter() - started, **stats))
    return outputs, failures, timings


WORKER = {}


def format_time(seconds):
    """As make_data.format_time (not imported: make_data needs the render env)."""
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours:02d}:{rest // 60:02d}:{rest % 60:02d}"


def init_worker(template_path, device):
    wp.config.quiet = True
    wp.init()
    WORKER["device"] = device
    with np.load(template_path, allow_pickle=False) as z:
        WORKER["template"] = dict(z)


def write_manifest(source, destination, records):
    if not source.is_file():
        return
    by_id = {r["sample_id"]: r for r in records}
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    content = "".join(json.dumps(dict(r, touch_path=by_id[r["sample_id"]]["touch_path"])) + "\n"
                      for r in rows if r["sample_id"] in by_id)
    temporary = destination.with_suffix(".tmp.jsonl")
    temporary.write_text(content)
    temporary.replace(destination)
    print(f"Wrote {destination} ({content.count(chr(10))} views)", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", default="generated_data/samples.jsonl", help="Views to process")
    parser.add_argument("--points", type=int, default=8192, help="Points per view")
    parser.add_argument("--tolerance", type=float, help="Skin squish, as a fraction of object size "
                        "(default: Dexonomy's 2 mm contact distance, per object)")
    parser.add_argument("--part-offset", action="store_true", help="Start each touching hand part's squish at its "
                        "own MuJoCo contact gap (rendered vs collision shapes), so every MuJoCo contact gets touch")
    parser.add_argument("--gap", type=float, default=0.01, help="Fill gaps narrower than this fraction of object "
                        "size; pieces closer than this are one shape")
    parser.add_argument("--min-part-share", type=float, default=0.02, help="Minimum share of --points per touching hand part")
    parser.add_argument("--hand-samples", type=int, default=65536, help="Hand-surface samples per pass")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--object-id", action="append", help="Process only these objects (repeatable); no manifests")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--joint-pointmap", action="store_true", help="Also save fixed 1024-pointmap + 7168-touch "
                        "inputs (as make_touch_data.py); existing touch files only get these added")
    parser.add_argument("--pipeline-config", type=Path, default=Path("checkpoints/hf/pipeline.yaml"))
    args = parser.parse_args(argv)
    if args.points < 1 or args.hand_samples < 1 or args.gap < 0 or not 0 <= args.min_part_share <= 1:
        parser.error("Need positive --points/--hand-samples, --gap >= 0, 0 <= --min-part-share <= 1")
    if args.tolerance is not None and args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    args.data_root = args.data_root.resolve()
    generated = args.data_root / "generated_data"
    records = [json.loads(line) for line in (args.data_root / args.manifest).read_text().splitlines() if line.strip()]
    groups = {}
    for r in records:
        if args.object_id is None or r["object_id"] in args.object_id:
            groups.setdefault(r["object_id"], []).append(r)
    if not groups:
        raise ValueError("No views to process")
    template = args.data_root / "grasp_views" / "hand_template.npz"
    print(f"Processing {len(groups)} objects ({sum(map(len, groups.values()))} views) on {args.workers} workers", flush=True)
    outputs, failures, timings = [], [], []
    started = time.monotonic()
    tasks = [(args, rows) for rows in groups.values()]
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(template, args.device)) as pool:
        for i, (out, fail, timing) in enumerate(pool.imap_unordered(make_object, tasks), 1):
            outputs += out
            failures += fail
            timings += timing
            elapsed = time.monotonic() - started
            eta = elapsed / i * (len(tasks) - i)
            print(f"[{i}/{len(tasks)}] elapsed={format_time(elapsed)} eta={format_time(eta)} "
                  f"{out[0]['object_id'] if out else ''}: {len(out)} views, {len(fail)} without contact", flush=True)
    if timings:
        seconds = np.array([t["seconds"] for t in timings])
        print(f"Per view: median {np.median(seconds):.2f} s, max {seconds.max():.2f} s", flush=True)
    if args.object_id is not None:
        for t in timings:
            print(json.dumps(t), flush=True)
        for f in failures:
            print("no contact:", json.dumps(f), flush=True)
        return
    path = generated / f"{NAME}_failures.json"
    path.write_text(json.dumps(sorted(failures, key=lambda f: f["sample_id"]), indent=1) + "\n")
    print(f"{len(failures)} views without touch (no confirmed contact, or an error; reasons in {path})", flush=True)
    write_manifest(generated / "samples.jsonl", generated / f"samples_{NAME}.jsonl", outputs)
    write_manifest(generated / "samples_camframe.jsonl", generated / f"samples_{NAME}_camframe.jsonl", outputs)


if __name__ == "__main__":
    main()
