import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

if sys.platform.startswith("linux"):
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import open3d as o3d
import trimesh
import yaml
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--touch-config", type=Path, default=Path("configs/data1.yaml"))
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=["mesh", "voxel"], default=["mesh", "voxel"])
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--gif-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument("--orbit-radius", type=float)
    parser.add_argument("--orbit-height", type=float, default=0.0)
    parser.add_argument("--light-strength", type=float, default=1.0)
    parser.add_argument("--mp4", action="store_true")
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def sample_output_dir(args):
    return (args.output_dir or args.evaluation_dir / "orbits") / args.sample_id


def load_sample(args):
    with open(args.evaluation_dir / "metrics.csv", newline="") as file:
        all_rows = [row for row in csv.DictReader(file) if not row["error"]]
    rows = [row for row in all_rows if row["sample_id"] == args.sample_id]
    if not rows:
        object_id = args.sample_id.rsplit("_", 1)[0]
        sample_ids = sorted({
            row["sample_id"] for row in all_rows
            if row["object_id"] in {args.sample_id, object_id}
        })
        if len(sample_ids) == 1:
            print(f"using selected sample {sample_ids[0]} for object {object_id}")
            args.sample_id = sample_ids[0]
            rows = [row for row in all_rows if row["sample_id"] == args.sample_id]
    rows = {row["condition"]: row for row in rows}
    if not rows:
        raise ValueError(f"No completed results found for {args.sample_id}")

    if args.conditions:
        conditions = args.conditions
    else:
        with open(args.evaluation_dir / "summary.yaml") as file:
            summary = yaml.safe_load(file)
        conditions = summary["primary_conditions"] + summary["diagnostic_conditions"]
    missing = [condition for condition in conditions if condition not in rows]
    if missing:
        raise ValueError(f"Missing results for: {', '.join(missing)}")
    return rows, conditions


def load_mesh(path):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Could not load mesh from {path}")
    return mesh


def mesh_geometry(mesh):
    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    geometry.compute_vertex_normals()
    return geometry


def voxel_geometry(points, size):
    cube = trimesh.creation.box(extents=(size, size, size))
    count = len(points)
    vertices_per_cube = len(cube.vertices)
    vertices = (points[:, None] + cube.vertices[None]).reshape(-1, 3)
    faces = cube.faces[None] + vertices_per_cube * np.arange(count)[:, None, None]
    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.reshape(-1, 3)),
    )
    geometry.compute_vertex_normals()
    return geometry


def material(color):
    result = o3d.visualization.rendering.MaterialRecord()
    result.shader = "defaultLit"
    result.base_color = (*color, 1.0)
    result.sRGB_color = True
    return result


def initialize_lighting(renderer, strength):
    scene = renderer.scene.scene
    white = np.ones(3, dtype=np.float32)
    scene.enable_sun_light(False)
    scene.enable_indirect_light(True)
    scene.set_indirect_light_intensity(50000.0 * strength)
    scene.add_point_light(
        "KeyLight", white, np.array([4.0, 1.0, 6.0], dtype=np.float32),
        5e5 * strength, 100.0, False,
    )
    scene.add_directional_light(
        "TopLight", white, np.array([0.0, -1.0, 0.0], dtype=np.float32),
        5e4 * strength, False,
    )
    scene.add_directional_light(
        "BottomLight", white, np.array([0.0, 1.0, 0.0], dtype=np.float32),
        1e4 * strength, False,
    )


def camera_fit(args, arrays):
    bounds = np.array([
        np.min([points.min(axis=0) for points in arrays], axis=0),
        np.max([points.max(axis=0) for points in arrays], axis=0),
    ])
    center = bounds.mean(axis=0)
    corners = trimesh.bounds.corners(bounds) - center
    vertical = np.radians(args.fov)
    horizontal = 2 * np.arctan(
        np.tan(vertical / 2) * args.width / args.height
    )
    required = 0.0
    for frame in range(args.frames):
        angle = 2 * np.pi * frame / args.frames
        outward = np.array([np.sin(angle), 0.0, -np.cos(angle)])
        forward = -outward
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        depth_offset = corners @ forward
        required = max(
            required,
            np.max(np.abs(corners @ right) / np.tan(horizontal / 2) - depth_offset),
            np.max(np.abs(corners[:, 1]) / np.tan(vertical / 2) - depth_offset),
        )
    return center, args.orbit_radius or max(3.2, 1.08 * required)


def render(args, geometry, center, radius, name, color):
    output_dir = sample_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f"{safe_name(name)}.mp4"
    gif_path = output_dir / f"{safe_name(name)}.gif"
    write_gif = not gif_path.exists()
    write_mp4 = args.mp4 and not mp4_path.exists()
    if not write_mp4 and not write_gif:
        print(f"skipping existing {gif_path}")
        return

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background((1.0, 1.0, 1.0, 1.0))
    renderer.scene.show_skybox(False)
    initialize_lighting(renderer, args.light_strength)
    renderer.scene.add_geometry("geometry", geometry, material(color))

    frames = []
    for frame in range(args.frames):
        angle = 2 * np.pi * frame / args.frames
        eye = center + np.array([
            radius * np.sin(angle),
            args.orbit_height,
            -radius * np.cos(angle),
        ])
        renderer.setup_camera(args.fov, center, eye, (0.0, 1.0, 0.0))
        image = np.asarray(renderer.render_to_image())[..., :3].copy()
        depth = np.asarray(renderer.render_to_depth_image())
        image[depth >= 1.0] = 255
        frames.append(image)
        print(f"{name}: frame {frame + 1}/{args.frames}", end="\r", flush=True)
    print()

    if write_mp4:
        try:
            from moviepy import ImageSequenceClip
        except ImportError:
            from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(frames, fps=args.fps)
        clip.write_videofile(
            str(mp4_path), codec="libx264", audio=False, logger=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        clip.close()
        print(f"saved {mp4_path}")

    if write_gif:
        gif_fps = min(args.gif_fps, args.fps)
        count = max(1, round(len(frames) * gif_fps / args.fps))
        indices = np.linspace(0, len(frames), count, endpoint=False, dtype=int)
        gif_frames = [
            Image.fromarray(frames[index]).resize(
                (args.gif_size, args.gif_size), Image.Resampling.LANCZOS
            )
            for index in indices
        ]
        gif_frames[0].save(
            gif_path, save_all=True, append_images=gif_frames[1:],
            duration=round(1000 / gif_fps), loop=0, disposal=2, optimize=True,
        )
        print(f"saved {gif_path}")
    del renderer


def save_inputs(args, row):
    output_dir = sample_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(row["image_path"])
    shutil.copy2(image_path, output_dir / "input_view.png")

    generated_dir = next(
        (parent for parent in image_path.parents if parent.name == "generated_data"),
        None,
    )
    if generated_dir is None:
        raise ValueError(f"Could not infer data root from {image_path}")

    with open(args.touch_config) as file:
        touch = yaml.safe_load(file)["touch"]
    command = [
        sys.executable,
        str(Path(__file__).parent / "data_generation/objaverse-dexonomy/make_orbit.py"),
        "--data-root", str(generated_dir.parent),
        "--object-id", row["object_id"],
        "--view-id", row["view_id"],
        "--output-dir", str(output_dir),
        "--variants", json.dumps([
            ["mesh", "pointmap"],
            ["mesh", "touch"],
            ["mesh", "full_surface"],
        ]),
        "--contacts", str(touch["contacts"]["count"]),
        "--radius", str(touch["neighborhood"]["max_geodesic_distance"]),
        "--visibility", touch["neighborhood"]["visibility"],
        "--points-per-contact", str(touch["point_sampling"]["points_per_contact"]),
        "--frames", str(args.frames),
        "--fps", str(args.fps),
        "--gif-fps", str(args.gif_fps),
        "--gif-size", str(args.gif_size),
        "--width", str(args.width),
        "--height", str(args.height),
        "--light-strength", str(args.light_strength),
        "--white-background",
        "--gif-only",
        "--flat-output",
    ]
    subprocess.run(command, check=True)


def aligned_stage1_voxels(args, row):
    with np.load(resolve(args.evaluation_dir, row["stage1_path"]), allow_pickle=False) as data:
        factor = int(data["downsample_factor"])
        if factor == 1:
            grid = data["prediction"]
            points = np.argwhere(grid).astype(np.float64) / np.asarray(grid.shape) - 0.5
        else:
            points = data["coords"][:, 1:].astype(np.float64) / 64 - 0.5
            print(
                f"{row['condition']}: Stage-1 support was downsampled by {factor}; "
                "rendering the exact support passed to Stage 2"
            )
    with np.load(resolve(args.evaluation_dir, row["alignment_path"]), allow_pickle=False) as data:
        transform = data["icp_transform"] @ data["prediction_normalization"]
    points = trimesh.transform_points(points, transform)
    scale = abs(np.linalg.det(transform[:3, :3])) ** (1 / 3)
    return points, 0.9 * scale / 64


def main():
    args = parse_args()
    if args.orbit_radius is not None and args.orbit_radius <= 0:
        raise ValueError("--orbit-radius must be positive")
    rows, conditions = load_sample(args)
    first = rows[conditions[0]]
    target_mesh = load_mesh(resolve(args.evaluation_dir, first["target_mesh_path"]))
    meshes = {
        condition: load_mesh(
            resolve(args.evaluation_dir, rows[condition]["mesh_aligned_path"])
        )
        for condition in conditions
    }
    voxels = {}
    if "voxel" in args.modes:
        if "decoded_gt" not in rows:
            raise ValueError("Voxel rendering requires the decoded_gt evaluation result")
        for condition in ["decoded_gt", *conditions]:
            if condition not in voxels:
                voxels[condition] = aligned_stage1_voxels(args, rows[condition])
    arrays = [target_mesh.vertices]
    arrays.extend(mesh.vertices for mesh in meshes.values())
    arrays.extend(points for points, _ in voxels.values())
    center, radius = camera_fit(args, arrays)

    print(f"sample: {args.sample_id}")
    print(f"conditions: {', '.join(conditions)}")
    save_inputs(args, first)

    if "mesh" in args.modes:
        render(
            args, mesh_geometry(target_mesh), center, radius,
            "mesh_ground_truth", (0.71, 0.71, 0.71),
        )
        for condition in conditions:
            render(
                args, mesh_geometry(meshes[condition]), center, radius,
                f"mesh_{condition}", (0.71, 0.71, 0.71),
            )

    if "voxel" in args.modes:
        points, size = voxels["decoded_gt"]
        render(
            args, voxel_geometry(points, size), center, radius,
            "voxel_ground_truth", (0.45, 0.65, 0.85),
        )
        for condition in conditions:
            if condition == "decoded_gt":
                continue
            points, size = voxels[condition]
            render(
                args, voxel_geometry(points, size), center, radius,
                f"voxel_{condition}", (0.45, 0.65, 0.85),
            )


if __name__ == "__main__":
    main()
