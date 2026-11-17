# use grasp-generation env for this because uses mujoco warp. also download viser into that env
import argparse
import json
import threading
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.transforms as tf
import warp as wp
from PIL import Image

import make_contact_touch as mct

FINGER = {"th": (230, 25, 75), "ff": (0, 130, 200), "mf": (40, 160, 60), "rf": (245, 130, 48), "lf": (145, 30, 180),
          "palm": (240, 50, 230)}
GREY = (170, 170, 170)
RENDER_HAND = (225, 225, 225)  # render_grasp_views.py HAND_COLOR 0.75 (linear) shown in sRGB


def piece_color(i):
    import colorsys
    return tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618034) % 1, 0.55, 0.9))


def load_convex_pieces(grasp_dir, uid, s_render, offset_render):
    """CoACD pieces Dexonomy grasped against (grasp_pipeline L4, stage-1 normalized frame), in the render frame:
    x = (center + scale * p) * s_render + offset_render, as select_grasp_views.hand_affine. Pieces that
    convert_objects.py skipped as too flat for MuJoCo are left out, so these are the object's collision geoms."""
    rec = json.loads((grasp_dir / "objects" / uid / "object_record.json").read_text())
    skipped = set(rec.get("pieces_skipped_flat", []))
    s1_dir = Path(rec["stage1"]["dir"])
    s1 = json.loads((s1_dir / "record.json").read_text())
    name = s1.get("files", {}).get("L4") or s1.get("files", {}).get("L4_is")
    if not name or not (s1_dir / name).is_dir():
        raise FileNotFoundError(f"no L4 pieces saved in {s1_dir}")
    n = s1["clean"]["normalize"]
    c, k = np.asarray(n["center"], float), float(n["scale"])
    pieces = []
    for p in sorted((s1_dir / name).glob("piece_*.npz")):
        if p.name in skipped:
            continue
        with np.load(p, allow_pickle=False) as z:
            v = (c + k * z["vertices"].astype(float)) * s_render + np.asarray(offset_render, float)
            pieces.append((v, z["faces"].astype(np.int64)))
    if not pieces:
        raise FileNotFoundError(f"no piece_*.npz in {s1_dir / name}")
    return pieces


def finger_color(name):
    n = name.replace("hand-", "").replace("rh_", "")
    return FINGER["palm"] if n.startswith("palm") else FINGER.get(n[:2], GREY)


def short(name):
    n = name.replace("hand-", "").replace("rh_", "")
    seg = {"distal": " tip", "middle": " middle", "proximal": " base", "knuckle": " knuckle", "metacarpal": " metacarpal",
           "hub": " hub"}
    finger = {"th": "thumb", "ff": "index", "mf": "middle", "rf": "ring", "lf": "little"}
    return "palm" if n == "palm" else finger.get(n[:2], n[:2]) + seg.get(n[2:], " " + n[2:])


_SPHERE = trimesh.creation.icosphere(subdivisions=1)
SPHERE_V, SPHERE_F = np.asarray(_SPHERE.vertices), np.asarray(_SPHERE.faces)


def collision_template(hand):
    """MuJoCo's collision shapes per hand body (as select_grasp_views.py builds the visual template)."""
    import dexonomy
    from dexonomy.sim import HandCfg, MuJoCo_VisEnv
    xml = Path(dexonomy.__path__[0]).parent / "assets" / "hand" / hand / "right.xml"
    env = MuJoCo_VisEnv(hand_cfg=HandCfg(xml_path=str(xml), freejoint=True), vis_mode="collision")
    out = []
    for name in env.get_all_body_names():
        m = env.get_body_mesh(name)
        if m is not None and len(m.faces):
            out.append((name, int(env.get_body_id(name)), np.asarray(m.vertices, float), np.asarray(m.faces)))
    return out


def patch_viser_camera():
    """Copied from grasp_pipeline/visualization/viser_processed_objects.py. viser bug: WASD flying jumps on the first key press after idling, because CameraControls.tsx moves by
    2 * frame delta and the first frame after idle has delta ~1 s. Clamp the step to 50 ms in viser's client source
    and rebuild the installed client once (viser's own build helper: sandboxed Node via nodeenv, npm, vite)."""
    import shutil
    import tempfile
    from viser import _client_autobuild as cb
    marker = cb.build_dir / "camera_dt_patched"
    if marker.exists():
        return
    src = cb.client_dir / "src" / "CameraControls.tsx"
    s = src.read_text()
    old = "const linear = 2.0 * delta;\n    const angular = 50.0 * THREE.MathUtils.DEG2RAD * delta;"
    if old not in s and "Math.min(delta, 0.05)" not in s:
        print("viser camera patch: CameraControls.tsx changed upstream, not patching")
        return
    src.write_text(s.replace(old, "const dt = Math.min(delta, 0.05);\n    const linear = 2.0 * dt;\n"
                                  "    const angular = 50.0 * THREE.MathUtils.DEG2RAD * dt;"))
    print("viser camera patch: rebuilding viser's client once (a minute or two)...")
    with tempfile.TemporaryDirectory() as tmp:   # build elsewhere so a failed build leaves the old client working
        cb._build_viser_client(out_dir=Path(tmp) / "build", cached=False)
        built = Path(tmp) / "build" / "index.html"
        if not built.exists():
            print("viser camera patch: build failed, keeping the unpatched client")
            return
        shutil.copy(built, cb.build_dir / "index.html")
    marker.touch()


class Viewer:
    def __init__(self, root, port, grasp_dir=None):
        self.root = root
        self.grasp_dir = grasp_dir
        self.pieces, self.pieces_error = None, "pass --grasp-dir to show them"
        self.records = [json.loads(line) for line in (root / "generated_data" / "samples.jsonl").read_text().splitlines()
                        if line.strip()]
        self.by_object = {}
        for r in self.records:
            self.by_object.setdefault(r["object_id"], []).append(r)
        with np.load(root / "grasp_views" / "hand_template.npz", allow_pickle=False) as z:
            self.tmpl = dict(z)
        settings = json.loads((root / "grasp_views" / "settings.json").read_text()) \
            if (root / "grasp_views" / "settings.json").is_file() else {}
        self.collision = collision_template(settings.get("hand", "shadow"))
        wp.init()
        self.device = "cpu" if not wp.is_cuda_available() else "cuda:0"
        self.lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.requested = None
        self.steps = None
        self.handles = {}
        self.server = viser.ViserServer(port=port, label="contact touch")
        self.server.scene.set_up_direction("+z")

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (1.1, -1.1, 0.8)
            client.camera.look_at = (0.0, 0.0, 0.0)
        self.build_gui()
        self.load_object(self.gui_object.value)

    # ---------- GUI ----------

    def build_gui(self):
        gui = self.server.gui
        ids = list(self.by_object)
        self.gui_object = gui.add_dropdown("Object", ids, initial_value=ids[0])
        self.gui_view = gui.add_dropdown("View", ["000"])
        self.gui_info = gui.add_markdown("")
        self.gui_image = gui.add_image(np.zeros((8, 8, 3), np.uint8), label="Rendered image")
        with gui.add_folder("Touch settings (% of object size)"):
            self.gui_default_tol = gui.add_checkbox("MuJoCo contact distance as tolerance", True,
                                                    hint="Dexonomy's 2 mm, converted for this object")
            self.gui_tol = gui.add_slider("Tolerance (skin squish) %", 0.1, 6.0, 0.05, 1.0, disabled=True)
            self.gui_gap = gui.add_slider("Gap fill %", 0.0, 3.0, 0.05, 1.0)
            self.gui_offset = gui.add_checkbox("Per-part offset", False,
                                               hint="Start each hand part's squish at its own MuJoCo contact gap")
            self.gui_source = gui.add_dropdown("Final points from", ["live recompute", "saved file"])
            self.gui_recompute = gui.add_button("Recompute")
        self.layers = {}
        with gui.add_folder("Layers"):
            for key, label, on, opacity, size in [
                ("object", "Object", True, 1.0, None),
                ("pieces", "Object convex pieces (used for grasping)", False, 1.0, None),
                ("hand", "Hand (rendered)", True, 0.25, None),
                ("collision", "Hand (MuJoCo collision shapes)", False, 0.35, None),
                ("contacts", "MuJoCo contact points", False, 1.0, 0.8),
                ("distance", "Candidates by distance to hand", False, 1.0, 0.15),
                ("touched", "Touched surface", False, 1.0, 0.15),
                ("filled", "Gap-filled surface", False, 1.0, 0.2),
                ("dropped", "Dropped pieces (no MuJoCo contact)", False, 1.0, 0.25),
                ("final", "Final touch points", True, 1.0, 0.2),
                ("camera", "Image camera", True, None, None),
            ]:
                with gui.add_folder(label, expand_by_default=False):
                    show = gui.add_checkbox("Show", on)
                    alpha = gui.add_slider("Opacity", 0.0, 1.0, 0.05, opacity) if opacity is not None else None
                    radius = gui.add_slider("Point size %", 0.05, 3.0, 0.05, size) if size is not None else None
                self.layers[key] = (show, alpha, radius)
                for h in (show, alpha, radius):
                    if h is not None:
                        h.on_update(lambda _, k=key: self.draw(k))
            self.gui_labels = gui.add_checkbox("Contact numbers", True)
            self.gui_labels.on_update(lambda _: self.draw("contacts"))
            self.gui_hand_color = gui.add_dropdown("Hand color", ["by finger", "as rendered"])
            self.gui_hand_color.on_update(lambda _: self.draw("hand"))
            self.gui_pieces_note = gui.add_markdown("")
            self.gui_wire = gui.add_checkbox("Collision shapes as wireframe", False)
            self.gui_wire.on_update(lambda _: self.draw("collision"))
        with gui.add_folder("Contacts"):
            self.gui_fly = gui.add_dropdown("Fly to contact", ["-"])
            self.gui_cam = gui.add_button("Look from image camera")
            self.gui_contacts = gui.add_markdown("")
        self.gui_object.on_update(lambda _: self.load_object(self.gui_object.value))
        self.gui_view.on_update(lambda _: self.load_view())
        def on_default_tol(_):
            self.gui_tol.disabled = self.gui_default_tol.value
            if self.gui_default_tol.value and hasattr(self, "default_tol"):
                self.gui_tol.value = round(100 * self.default_tol, 2)
            self.recompute()
        self.gui_default_tol.on_update(on_default_tol)
        for h in (self.gui_tol, self.gui_gap, self.gui_offset):
            h.on_update(lambda _: self.recompute())
        self.gui_recompute.on_click(lambda _: self.recompute(force=True))
        self.gui_source.on_update(lambda _: self.draw("final"))
        self.gui_fly.on_update(lambda _: self.fly())
        self.gui_cam.on_click(lambda _: self.look_from_camera())

    # ---------- data ----------

    def load_object(self, object_id):
        self.ready = False
        with self.lock:
            rows = sorted(self.by_object[object_id], key=lambda r: r["view_id"])
            self.rows = {r["view_id"]: r for r in rows}
            with np.load(self.root / rows[0]["mesh_path"], allow_pickle=False) as m:
                v, f = m["vertices"].astype(np.float64), m["faces"].astype(np.int64)
            self.obj = mct.object_mesh(v, f, self.device)
            doc = json.loads((self.root / rows[0]["grasp_views_path"]).read_text())
            self.views = doc["views"]
            if self.grasp_dir is not None:
                try:
                    self.pieces = load_convex_pieces(self.grasp_dir, doc["uid"], doc["s_render"],
                                                     doc["offset_render"])
                    self.pieces_error = None
                except Exception as error:
                    self.pieces, self.pieces_error = None, f"{type(error).__name__}: {error}"
        self.gui_pieces_note.content = (f"Convex pieces: {len(self.pieces)}" if self.pieces is not None
                                        else f"Convex pieces not available: {self.pieces_error}")
        self.gui_view.options = list(self.rows)
        self.gui_view.value = rows[0]["view_id"]
        self.load_view()

    def load_view(self):
        if self.gui_view.value not in self.rows:
            return
        self.ready = False  # callbacks fired by the value changes below wait for the explicit recompute at the end
        self.record = self.rows[self.gui_view.value]
        self.view = self.views[int(self.record["grasp_view_index"])]
        d = self.root / Path(self.record["camera_path"]).parent
        with np.load(d / "camera.npz", allow_pickle=False) as c:
            self.K, self.T = c["K"], c["T_camera_from_object"]
        self.image = np.asarray(Image.open(d / "image.png").convert("RGBA"))
        rgb = self.image[..., :3].astype(float) * (self.image[..., 3:] / 255) + 255 * (1 - self.image[..., 3:] / 255)
        self.gui_image.image = rgb.astype(np.uint8)
        self.saved = None
        if (d / f"{mct.NAME}.npz").is_file():
            with np.load(d / f"{mct.NAME}.npz", allow_pickle=False) as z:
                self.saved = dict(points=z["points_object"], part=z["part_index"])
        self.default_tol = mct.CONTACT_THRE * float(self.view["hand_a"])
        self.gui_default_tol.label = f"MuJoCo contact distance as tolerance ({100 * self.default_tol:.2f}%)"
        if self.gui_default_tol.value:
            self.gui_tol.value = round(100 * self.default_tol, 2)
        c = self.view["contacts"]
        self.contacts = np.asarray(c["surface"], float).reshape(-1, 3)
        self.contact_finger = np.asarray(c["finger"], float).reshape(-1, 3)
        self.contact_bodies = list(c["bodies"])
        self.gui_fly.options = ["-"] + [f"{i}: {short(b)}" for i, b in enumerate(self.contact_bodies)]
        self.gui_fly.value = "-"
        self.ready = True
        self.recompute()

    def recompute(self, force=False):
        """Settings changes, view loads and viser's own callbacks for programmatic value changes all land here.
        Same view + same settings as the last run is skipped (unless forced); while one run is busy, only the newest
        request runs next, so dragging a slider does not queue a run per step."""
        if not getattr(self, "ready", False):
            return
        tol = None if self.gui_default_tol.value else self.gui_tol.value / 100
        key = (self.record["sample_id"], tol, self.gui_gap.value, self.gui_offset.value)
        with self.request_lock:
            if not force and key == self.requested:
                return
            self.requested = key
        args = argparse.Namespace(tolerance=tol, gap=self.gui_gap.value / 100, part_offset=self.gui_offset.value,
                                  points=8192, min_part_share=0.02, hand_samples=65536, seed=29)
        record, view = self.record, self.view
        self.gui_info.content = "Computing..."
        with self.lock:
            if key != self.requested:  # a newer request came in while waiting; it runs instead
                return
            try:
                *_, stats, steps = mct.view_contact_touch(record, self.root, view, self.tmpl, self.obj, args,
                                                          self.device, debug=True)
                self.steps, self.stats, self.error = steps, stats, None
            except mct.NoContact as error:
                self.steps, self.stats, self.error = None, None, str(error)
        self.update_text()
        for key in self.layers:
            self.draw(key)

    def update_text(self):
        r = self.record
        head = (f"**{r['object_id'][:16]}… view {r['view_id']}** — {r.get('grasp_type', '')}, size bin "
                f"{r.get('grasp_size_bin', '')}\n\n")
        if self.steps is None:
            self.gui_info.content = head + f"No touch: {self.error}"
            self.gui_contacts.content = ""
            return
        s, st = self.stats, self.steps
        names = [str(n) for n in self.tmpl["names"]]
        covered = st["contact_covered"]
        tol_txt = ", ".join(f"{short(names[st['touching'][k]])} {100 * t:.2f}%" for k, t in enumerate(st["part_tol"])) \
            if s["part_offset"] else f"{100 * s['tolerance']:.2f}%"
        self.gui_info.content = head + (
            f"Tolerance: {tol_txt}  \nMuJoCo contacts with touch: **{covered.sum()} / {len(covered)}**  \n"
            f"Candidates {s['candidates']}, touched {s['touched']}, after gap fill {s['region']}, kept {s['kept']} "
            f"in {s['pieces']} pieces  \nPoints per part: "
            + ", ".join(f"{short(p)} {n}" for p, n in zip(s["parts"], s["part_points"])))
        rows = ["| # | hand part | rendered gap (× tolerance) | touch |", "|---|---|---|---|"]
        for i, b in enumerate(self.contact_bodies):
            k = st["contact_part"][i]
            gap = st["contact_gap"][i] / st["part_tol"][k] if k >= 0 and np.isfinite(st["contact_gap"][i]) else np.nan
            rows.append(f"| {i} | {short(b)} | {'no mesh' if np.isnan(gap) else f'{gap:.2f}'} | "
                        f"{'yes' if covered[i] else '**no**'} |")
        self.gui_contacts.content = "\n".join(rows)

    # ---------- drawing ----------

    def clear(self, key):
        for h in self.handles.pop(key, []):
            h.remove()

    def spheres(self, name, points, colors, radius, opacity):
        if len(points) == 0:
            return []
        colors = np.broadcast_to(np.asarray(colors, np.uint8), (len(points), 3))
        return [self.server.scene.add_batched_meshes_simple(
            name, SPHERE_V, SPHERE_F, batched_wxyzs=np.tile([1.0, 0, 0, 0], (len(points), 1)),
            batched_positions=np.asarray(points, np.float32), batched_scales=np.full(len(points), radius, np.float32),
            batched_colors=colors, opacity=opacity if opacity < 1 else None, lod="off")]

    def draw(self, key):
        if not getattr(self, "ready", False):
            return
        self.clear(key)
        show, alpha, size = self.layers[key]
        op = float(alpha.value) if alpha is not None else 1.0
        if not show.value or op == 0:
            return
        radius = size.value / 100 if size is not None else None
        names = [str(n) for n in self.tmpl["names"]]
        a, b = float(self.view["hand_a"]), np.asarray(self.view["hand_b"], float)
        xmat = np.asarray(self.view["body_xmat"], float).reshape(-1, 3, 3)
        xpos = np.asarray(self.view["body_xpos"], float)
        touching_names = {names[i] for i in self.steps["touching"]} if self.steps is not None else set()
        out = []
        sc = self.server.scene
        if key == "object":
            v, f = self.obj[0], self.obj[1]
            out.append(sc.add_mesh_simple("/object", v, f, color=(205, 200, 190), opacity=op if op < 1 else None,
                                          side="double", flat_shading=False))
        elif key == "pieces":
            for j, (v, f) in enumerate(self.pieces or []):
                out.append(sc.add_mesh_simple(f"/pieces/{j}", v, f, color=piece_color(j),
                                              opacity=op if op < 1 else None, side="double"))
        elif key == "hand":
            for i, n in enumerate(names):
                bid = int(self.tmpl[f"{i}_body_id"])
                v = self.tmpl[f"{i}_vertices"].astype(float) @ xmat[bid].T * a + a * xpos[bid] + b
                if self.gui_hand_color.value == "as rendered":
                    color = RENDER_HAND
                else:
                    color = finger_color(n) if n in touching_names else GREY
                out.append(sc.add_mesh_simple(f"/hand/{i}", v, self.tmpl[f"{i}_faces"], color=color,
                                              opacity=op if op < 1 else None, side="double"))
        elif key == "collision":
            for j, (n, bid, v0, f0) in enumerate(self.collision):
                v = v0 @ xmat[bid].T * a + a * xpos[bid] + b
                color = finger_color(n) if n in touching_names else GREY
                out.append(sc.add_mesh_simple(f"/collision/{j}", v, f0, color=color, wireframe=self.gui_wire.value,
                                              opacity=op if op < 1 else None, side="double"))
        elif key == "contacts":
            colors = np.array([finger_color(n) for n in self.contact_bodies], np.uint8)
            out += self.spheres("/contacts", self.contacts, colors, radius, op)
            covered = self.steps["contact_covered"] if self.steps is not None else np.zeros(len(self.contacts), bool)
            for i, (p, q) in enumerate(zip(self.contacts, self.contact_finger)):
                seg = np.array([[p, q]], np.float32)
                out.append(sc.add_line_segments(f"/contact_links/{i}", seg, colors=(20, 20, 20), thickness=1.5,
                                                  thickness_units="screen"))
                if self.gui_labels.value:
                    out.append(sc.add_label(f"/contact_labels/{i}", f"{i}" + ("" if covered[i] else " ✗"),
                                            position=tuple(p), font_screen_scale=0.7))
        elif self.steps is None:
            return
        elif key == "distance":
            st = self.steps
            ratio = st["distance"] / st["part_tol"][st["part"]]
            sel = ratio <= 3
            r = np.clip(ratio[sel], 0, 3)
            colors = np.zeros((sel.sum(), 3))
            inside = r <= 1
            colors[inside] = [220, 30, 30]
            t = (r[~inside] - 1) / 2
            colors[~inside] = (1 - t)[:, None] * np.array([250, 200, 40]) + t[:, None] * np.array([150, 160, 190])
            out += self.spheres("/distance", st["points"][sel], colors.astype(np.uint8), radius, op)
        elif key == "touched":
            st = self.steps
            sel = st["touched"]
            colors = np.array([finger_color(names[st["touching"][p]]) for p in st["part"][sel]], np.uint8)
            out += self.spheres("/touched", st["points"][sel], colors, radius, op)
        elif key == "filled":
            st = self.steps
            sel = st["region"] & ~st["touched"]
            out += self.spheres("/filled", st["points"][sel], (0, 190, 200), radius, op)
        elif key == "dropped":
            st = self.steps
            sel = st["region"].copy()
            sel[st["kept"]] = False
            out += self.spheres("/dropped", st["points"][sel], (40, 40, 40), radius, op)
        elif key == "final":
            if self.gui_source.value == "saved file":
                if self.saved is None:
                    return
                pts = self.saved["points"]
                colors = np.array([finger_color(names[p]) for p in self.saved["part"]], np.uint8)
            else:
                st = self.steps
                pts = st["points"][st["chosen"]]
                colors = np.array([finger_color(names[st["touching"][p]]) for p in st["part"][st["chosen"]]], np.uint8)
            out += self.spheres("/final", pts, colors, radius, op)
        elif key == "camera":
            R, t = self.T[:3, :3], self.T[:3, 3]
            h, w = self.image.shape[:2]
            fov = 2 * np.arctan(h / 2 / self.K[1, 1])
            img = np.asarray(Image.fromarray(self.image[..., :3]).resize((w // 4, h // 4)))
            out.append(sc.add_camera_frustum("/camera", fov=float(fov), aspect=w / h, scale=0.25, image=img,
                                             wxyz=tf.SO3.from_matrix(R.T).wxyz, position=-R.T @ t))
        self.handles[key] = out

    # ---------- camera ----------

    def fly(self):
        if self.gui_fly.value == "-":
            return
        i = int(self.gui_fly.value.split(":")[0])
        p, q = self.contacts[i], self.contact_finger[i]
        direction = q - p
        if np.linalg.norm(direction) < 1e-9:
            direction = p - self.obj[0].mean(0)
        direction /= np.linalg.norm(direction)
        for client in self.server.get_clients().values():
            client.camera.look_at = tuple(p)
            client.camera.position = tuple(p + 0.18 * direction)

    def look_from_camera(self):
        R, t = self.T[:3, :3], self.T[:3, 3]
        centre = -R.T @ t
        for client in self.server.get_clients().values():
            client.camera.up_direction = tuple(-R.T @ np.array([0, 1.0, 0]))
            client.camera.position = tuple(centre)
            client.camera.look_at = tuple(centre + R.T @ np.array([0, 0, 1.0]))
            client.camera.fov = float(2 * np.arctan(self.image.shape[0] / 2 / self.K[1, 1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--grasp-dir", type=Path,
                        help="grasp_pipeline grasp_generation folder (select_grasp_views.py --grasp-dir), "
                             "for the convex pieces layer")
    args = parser.parse_args()
    patch_viser_camera()
    Viewer(args.data_root.resolve(), args.port, args.grasp_dir.resolve() if args.grasp_dir else None)
    threading.Event().wait()


if __name__ == "__main__":
    main()
