import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

if sys.platform.startswith("linux"):
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import open3d as o3d
import trimesh
import yaml
try:
    from moviepy import ImageSequenceClip
except ImportError:
    from moviepy.editor import ImageSequenceClip
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=["mesh", "voxel"], default=["mesh"])
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--gif-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument("--orbit-radius", type=float, default=3.2)
    parser.add_argument("--orbit-height", type=float, default=0.0)
    parser.add_argument("--light-strength", type=float, default=1.0)
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_sample(args):
    with open(args.evaluation_dir / "metrics.csv", newline="") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["sample_id"] == args.sample_id and not row["error"]
        ]
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


def render(args, geometry, center, name, color):
    output_dir = (args.output_dir or args.evaluation_dir / "orbits") / args.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f"{safe_name(name)}.mp4"
    gif_path = output_dir / f"{safe_name(name)}.gif"
    write_mp4 = not mp4_path.exists()
    write_gif = not gif_path.exists()
    if not write_mp4 and not write_gif:
        print(f"skipping existing {mp4_path} and {gif_path}")
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
            args.orbit_radius * np.sin(angle),
            args.orbit_height,
            -args.orbit_radius * np.cos(angle),
        ])
        renderer.setup_camera(args.fov, center, eye, (0.0, 1.0, 0.0))
        image = np.asarray(renderer.render_to_image())[..., :3].copy()
        depth = np.asarray(renderer.render_to_depth_image())
        image[depth >= 1.0] = 255
        frames.append(image)
        print(f"{name}: frame {frame + 1}/{args.frames}", end="\r", flush=True)
    print()

    if write_mp4:
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


def voxel_points(grid):
    resolution = np.asarray(grid.shape, dtype=np.float64)
    return np.argwhere(grid).astype(np.float64) / resolution - 0.5


def normalize_points(points):
    bounds = np.array([points.min(axis=0), points.max(axis=0)])
    center = bounds.mean(axis=0)
    scale = 2.0 / (bounds[1] - bounds[0]).max()
    return (points - center) * scale, scale


def transform_points(points, transform):
    return trimesh.transform_points(points, transform)


def load_target_voxels(args, row):
    with np.load(resolve(args.evaluation_dir, row["voxel_path"]), allow_pickle=False) as data:
        points = voxel_points(data["target"])
    points, scale = normalize_points(points)
    return points, 0.9 * scale / 64


def load_prediction_voxels(args, row):
    with np.load(resolve(args.evaluation_dir, row["voxel_path"]), allow_pickle=False) as data:
        points = voxel_points(data["prediction"])
    with np.load(resolve(args.evaluation_dir, row["alignment_path"]), allow_pickle=False) as data:
        transform = data["icp_transform"] @ data["prediction_normalization"]
    points = transform_points(points, transform)
    scale = abs(np.linalg.det(transform[:3, :3])) ** (1 / 3)
    return points, 0.9 * scale / 64


def main():
    args = parse_args()
    rows, conditions = load_sample(args)
    first = rows[conditions[0]]
    target_mesh = load_mesh(resolve(args.evaluation_dir, first["target_mesh_path"]))
    center = target_mesh.bounds.mean(axis=0)

    print(f"sample: {args.sample_id}")
    print(f"conditions: {', '.join(conditions)}")

    if "mesh" in args.modes:
        render(args, mesh_geometry(target_mesh), center, "mesh_ground_truth", (0.71, 0.71, 0.71))
        for condition in conditions:
            mesh = load_mesh(resolve(args.evaluation_dir, rows[condition]["mesh_aligned_path"]))
            render(
                args, mesh_geometry(mesh), center,
                f"mesh_{condition}", (0.71, 0.71, 0.71),
            )

    if "voxel" in args.modes:
        points, size = load_target_voxels(args, first)
        render(
            args, voxel_geometry(points, size), center,
            "voxel_ground_truth", (0.45, 0.65, 0.85),
        )
        for condition in conditions:
            points, size = load_prediction_voxels(args, rows[condition])
            render(
                args, voxel_geometry(points, size), center,
                f"voxel_{condition}", (0.45, 0.65, 0.85),
            )


if __name__ == "__main__":
    main()
