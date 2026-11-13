"""Blender: render one object with the Shadow-hand grasp chosen for each view (make_data.py --grasp-views).

Same scene as render_blender.py (import, freeze, normalization, Cycles settings, TRELLIS or fixed lighting); the
cameras and hand poses come from select_grasp_views.py's views.json. Per view: image.png (object and hand),
object_mask.png and hand_mask.png (visible pixels, 0/255), depth.npy
(object + hand, camera-axis z) and camera.npz in render_blender.py's format.
"""
import argparse
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_blender import (camera_extrinsics, camera_intrinsics, clear_scene, freeze_geometry,  # noqa: E402
                            init_random_lighting, initialize_lighting, initialize_render, load_depth_exr,
                            load_object, normalize_scene)

HAND_COLOR = (0.75, 0.75, 0.75, 1)
HAND_ROUGHNESS = 0.6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--object", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True, help="views.json from select_grasp_views.py")
    p.add_argument("--hand", type=Path, required=True, help="hand_template.npz from select_grasp_views.py")
    p.add_argument("--num-views", type=int, required=True)
    p.add_argument("--resolution", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", choices=["CUDA", "OPTIX", "CPU", "METAL"], default="CUDA")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--lighting", choices=["trellis2", "fixed"], default="trellis2")
    p.add_argument("--frame", type=int, default=1)
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:])


def add_hand(path):
    """One Blender object per hand body, meshes in the body frame; posed per view through matrix_world."""
    z = np.load(path)
    material = bpy.data.materials.new("Hand")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = HAND_COLOR
    shader.inputs["Roughness"].default_value = HAND_ROUGHNESS
    bodies = []
    for i in range(len(z["names"])):
        mesh = bpy.data.meshes.new(f"hand_{i}")
        mesh.from_pydata(z[f"{i}_vertices"].tolist(), [], z[f"{i}_faces"].tolist())
        mesh.update()
        obj = bpy.data.objects.new(f"hand_{i}", mesh)
        obj.data.materials.append(material)
        obj.pass_index = 2
        bpy.context.scene.collection.objects.link(obj)
        bodies.append((obj, int(z[f"{i}_body_id"])))
    return bodies


def write_mask_png(path, mask):
    """8-bit grayscale PNG (0/255), rows top to bottom; plain zlib, no colour management involved."""
    h, w = mask.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in (mask.astype(np.uint8) * 255))
    chunk = lambda kind, data: (struct.pack(">I", len(data)) + kind + data
                                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def initialize_outputs():
    """Compositor: depth and object index (1 object, 2 hand) to one EXR each per render."""
    scene, layer = bpy.context.scene, bpy.context.view_layer
    scene.use_nodes = True
    layer.use_pass_z = True
    layer.use_pass_object_index = True
    nodes, links = scene.node_tree.nodes, scene.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    layers = nodes.new("CompositorNodeRLayers")
    layers.layer = layer.name
    output = nodes.new("CompositorNodeOutputFile")
    output.format.file_format = "OPEN_EXR"
    output.format.color_mode = "RGB"
    output.format.color_depth = "32"
    output.file_slots.clear()
    output.file_slots.new("depth_")
    output.file_slots.new("index_")
    links.new(layers.outputs["Depth"], output.inputs[0])
    links.new(layers.outputs["IndexOB"], output.inputs[1])
    return output


def main():
    args = parse_args()
    started = time.perf_counter()
    spec = json.loads(args.views.read_text())
    views = spec["views"]
    if len(views) != args.num_views:
        raise ValueError(f"{args.views} has {len(views)} views, expected {args.num_views}")
    # Same stream layout as render_blender.py; only lighting and cycles are used here.
    stream_seeds = {name: int(child.generate_state(1)[0]) for name, child in zip(
        ("angles", "distance", "lighting", "cycles", "object_rotation"), np.random.SeedSequence(args.seed).spawn(5))}
    stream_seeds["cycles"] &= 0x7fffffff
    args.output.mkdir(parents=True, exist_ok=True)

    bpy.context.scene.render.engine = "CYCLES"
    clear_scene()
    load_object(args.object)
    assets = []
    for image in bpy.data.images:
        if image.source == "FILE" and not image.packed_file and image.filepath:
            path = Path(bpy.path.abspath(image.filepath, library=image.library)).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing texture: {path}")
            assets.append({"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
    mesh_objects = freeze_geometry(args.frame, [0, 0, 0])
    normalize_scene(mesh_objects, args.output)
    for obj in mesh_objects:
        obj.pass_index = 1

    # The hand was placed with the normalization computed from stage 1's loaded mesh; it must match Blender's.
    T = np.load(args.output / "object_transform.npz")["T_normalized_from_source"]
    diff = max(abs(float(T[0, 0]) - spec["s_render"]), float(np.abs(T[:3, 3] - np.array(spec["offset_render"])).max()))
    if diff > 1e-5:
        raise ValueError(f"Blender normalization differs from the one used for the hand by {diff:.2e}")

    hand = add_hand(args.hand)
    initialize_render(args.resolution, args.device, args.samples)
    bpy.context.scene.cycles.seed = stream_seeds["cycles"]
    outputs = initialize_outputs()
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.sensor_width = camera_data.sensor_height = 32.0
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.clip_start, camera_data.clip_end = 0.01, 100.0
    initialize_lighting()
    light_rng = np.random.RandomState(stream_seeds["lighting"])

    with (args.output / "render_metadata.json").open("w") as file:
        json.dump({"blender_version": bpy.app.version_string, "source": str(args.object), "render_seed": args.seed,
                   "render_stream_scheme": "seedsequence_spawn_v1", "render_stream_seeds": stream_seeds,
                   "frame": args.frame, "rotation_degrees": [0, 0, 0], "random_object_rotation": False,
                   "camera_mode": "grasp_views", "lighting": args.lighting, "samples": args.samples,
                   "persistent_data": True, "external_assets": assets, "depth_convention": "camera_axis_z",
                   "color_transform": "Filmic", "pointmap_convention": "x_left_y_up_z_forward",
                   "grasp_views": str(args.views), "normalization_check_max_abs_diff": diff,
                   "hand_material": {"base_color": HAND_COLOR, "roughness": HAND_ROUGHNESS},
                   "masks": ["object_mask.png", "hand_mask.png"]}, file, indent=2)

    for v in views:
        t_view = time.perf_counter()
        a, b = v["hand_a"], np.array(v["hand_b"])
        xmat, xpos = np.array(v["body_xmat"]), np.array(v["body_xpos"])
        for obj, body in hand:
            M = np.eye(4)
            M[:3, :3] = a * xmat[body].reshape(3, 3)
            M[:3, 3] = a * xpos[body] + b
            obj.matrix_world = Matrix(M.tolist())
        M_cam = np.array(v["camera_matrix_world"])
        camera.matrix_world = Matrix(M_cam.tolist())
        camera_data.lens = (camera_data.sensor_width / 2) / np.tan(np.deg2rad(v["fov"]) / 2)
        if args.lighting == "trellis2":
            init_random_lighting(M_cam[:3, 3] / np.linalg.norm(M_cam[:3, 3]), light_rng)
        bpy.context.view_layer.update()

        view_dir = args.output / "views" / f"{v['view']:03d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(view_dir / "image.png")
        outputs.base_path = str(view_dir)
        for stale in view_dir.glob("*_*.exr"):
            stale.unlink()
        bpy.ops.render.render(write_still=True)
        if not (view_dir / "image.png").is_file():
            raise FileNotFoundError(view_dir / "image.png")
        depth_files, index_files = list(view_dir.glob("depth_*.exr")), list(view_dir.glob("index_*.exr"))
        if len(depth_files) != 1 or len(index_files) != 1:
            raise RuntimeError(f"Expected one depth and one index EXR in {view_dir}")
        depth = load_depth_exr(depth_files[0])  # camera-axis z, as render_blender.render_view
        depth[~(np.isfinite(depth) & (depth > 0) & (depth < camera_data.clip_end))] = np.nan
        np.save(view_dir / "depth.npy", depth)
        index = np.rint(load_depth_exr(index_files[0])).astype(np.uint8)  # 1 object, 2 hand (pass_index)
        write_mask_png(view_dir / "object_mask.png", index == 1)
        write_mask_png(view_dir / "hand_mask.png", index == 2)
        for f in depth_files + index_files:
            f.unlink()
        K = camera_intrinsics(args.resolution, v["fov"])
        T_cw = camera_extrinsics(camera)
        pose = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32)) @ T_cw  # object transform is the identity
        np.savez(view_dir / "camera.npz", K=K, T_camera_from_world=T_cw,
                 T_world_from_object=np.eye(4, dtype=np.float32), T_camera_from_object=T_cw,
                 rotation=pose[:3, :3], translation=pose[:3, 3], scale=np.ones(3, dtype=np.float32),
                 pose_source_frame="normalized_object", pose_coordinate_frame="sam_camera",
                 pose_rotation_convention="column_vectors")
        print(f"Rendered view {v['view'] + 1}/{len(views)} | {time.perf_counter() - t_view:.1f}s", flush=True)
    print(f"Rendered {len(views)} views in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
