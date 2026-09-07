import argparse
import json
import os
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
from PIL import Image, ImageDraw


SOURCES = ("full_surface", "full_surface_unmasked", "touch", "joint")
INPUT_COLOR = np.array([45, 125, 210], dtype=np.uint8)
MESH_COLOR = (0.71, 0.71, 0.71)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("checkpoints/hf/pipeline.yaml"),
    )
    parser.add_argument("--touch-config", type=Path, default=Path("configs/data1.yaml"))
    parser.add_argument(
        "--full-surface-config",
        type=Path,
        default=Path("configs/data_full_surface.yaml"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/vecsetx/outputs/reconstruction_256"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/vecsetx/outputs/orbits"),
    )
    parser.add_argument(
        "--sources", nargs="+", choices=SOURCES,
        default=["full_surface", "touch", "joint"],
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-error-fraction", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--gif-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument("--orbit-radius", type=float)
    parser.add_argument("--orbit-height", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=0.02)
    parser.add_argument("--light-strength", type=float, default=1.0)
    parser.add_argument("--mp4", action="store_true")
    return parser.parse_args()


def source_file(source):
    return "full_surface" if source == "full_surface_unmasked" else source


def ensure_artifacts(args):
    settings_path = args.input_dir / f"{args.sample_id}_settings.json"
    paths = [
        settings_path,
        args.input_dir / f"{args.sample_id}_reference.obj",
        args.input_dir / f"{args.sample_id}_reference.glb",
    ]
    for source in args.sources:
        point_source = source_file(source)
        paths.extend([
            args.input_dir / f"{args.sample_id}_{source}.obj",
            args.input_dir / f"{args.sample_id}_{point_source}_points.npy",
            args.input_dir / f"{args.sample_id}_{point_source}_normalization.npz",
            args.input_dir / f"{args.sample_id}_{source}_input_sdf.npy",
        ])
    if all(path.exists() for path in paths):
        with settings_path.open() as file:
            if json.load(file).get("resolution") == args.resolution:
                return

    print(f"Preparing VecSetX artifacts for {args.sample_id}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.vecsetx.reconstruct",
            "--pipeline-config", str(args.pipeline_config),
            "--touch-config", str(args.touch_config),
            "--full-surface-config", str(args.full_surface_config),
            "--output-dir", str(args.input_dir),
            "--split", args.split,
            "--sample-id", args.sample_id,
            "--resolution", str(args.resolution),
            "--skip-report",
        ],
        check=True,
    )


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Could not load mesh from {path}")
    return mesh


def transform_mesh(mesh, center, scale):
    mesh = mesh.copy()
    mesh.vertices = (mesh.vertices - center) * scale
    return mesh


def load_scene(args):
    reference = load_mesh(args.input_dir / f"{args.sample_id}_reference.obj")
    bounds = reference.bounds
    center = bounds.mean(axis=0)
    extent = (bounds[1] - bounds[0]).max()
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Reference mesh must have a positive finite extent")
    display_scale = 2.0 / extent
    reference = transform_mesh(reference, center, display_scale)

    variants = {}
    for source in args.sources:
        point_source = source_file(source)
        points = np.load(
            args.input_dir / f"{args.sample_id}_{point_source}_points.npy",
            allow_pickle=False,
        )
        input_sdf = np.load(
            args.input_dir / f"{args.sample_id}_{source}_input_sdf.npy",
            allow_pickle=False,
        )
        with np.load(
            args.input_dir / f"{args.sample_id}_{point_source}_normalization.npz",
            allow_pickle=False,
        ) as data:
            shift = data["shift"]
            scale = float(data["scale"].reshape(-1)[0])

        points = (points / scale + shift - center) * display_scale
        reconstruction = load_mesh(
            args.input_dir / f"{args.sample_id}_{source}.obj"
        )
        reconstruction.vertices = (
            reconstruction.vertices / scale + shift - center
        ) * display_scale
        sdf_error = np.abs(input_sdf) / scale * display_scale
        variants[source] = {
            "points": points,
            "mesh": reconstruction,
            "sdf_error": sdf_error,
        }
    return reference, variants, center, display_scale


def load_record(args):
    with args.touch_config.open() as file:
        config = yaml.safe_load(file)
    root = Path(config["dataset"]["root"])
    manifest = Path(config["dataset"]["manifest"])
    manifest = manifest if manifest.is_absolute() else root / manifest
    with manifest.open() as file:
        for line in file:
            record = json.loads(line)
            if record["sample_id"] == args.sample_id:
                return root, record
    raise ValueError(f"Unknown sample {args.sample_id!r}")


def load_textured_reference(path, center, scale):
    model = o3d.io.read_triangle_model(str(path))
    if not model.meshes:
        raise ValueError(f"Could not load textured mesh from {path}")
    for part in model.meshes:
        vertices = np.asarray(part.mesh.vertices)
        part.mesh.vertices = o3d.utility.Vector3dVector(
            (vertices - center) * scale
        )
        part.mesh.compute_vertex_normals()

    materials = []
    for imported in model.materials:
        result = o3d.visualization.rendering.MaterialRecord()
        result.shader = "defaultLit"
        result.base_color = (1.0, 1.0, 1.0, 1.0)
        result.albedo_img = imported.albedo_img
        materials.append(result)
    model.materials = materials
    return model


def mesh_geometry(mesh):
    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    geometry.compute_vertex_normals()
    return geometry


def particles(points, colors, size):
    # Copied from data_generation/objaverse-dexonomy/make_orbit.py.
    sphere = trimesh.creation.icosphere(subdivisions=0, radius=size / 2)
    vertices_per_point = len(sphere.vertices)
    vertices = (points[:, None] + sphere.vertices[None]).reshape(-1, 3)
    faces = sphere.faces[None] + vertices_per_point * np.arange(len(points))[:, None, None]
    colors = np.broadcast_to(colors, points.shape).reshape(-1, 3)
    colors = np.repeat(colors, vertices_per_point, axis=0) / 255.0

    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.reshape(-1, 3)),
    )
    geometry.vertex_colors = o3d.utility.Vector3dVector(colors)
    return geometry


def material(color, unlit=False):
    result = o3d.visualization.rendering.MaterialRecord()
    result.shader = "defaultUnlit" if unlit else "defaultLit"
    result.base_color = (*color, 1.0)
    result.sRGB_color = True
    return result


def initialize_lighting(renderer, strength):
    # Copied from make_eval_orbit.py.
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


def error_colors(errors, maximum):
    values = np.clip(errors / maximum, 0.0, 1.0)
    green = np.array([44, 162, 95], dtype=np.float64)
    yellow = np.array([253, 216, 53], dtype=np.float64)
    red = np.array([215, 48, 39], dtype=np.float64)
    colors = np.where(
        values[:, None] <= 0.5,
        green + (yellow - green) * (2 * values[:, None]),
        yellow + (red - yellow) * (2 * values[:, None] - 1),
    )
    return colors.astype(np.uint8)


def camera_fit(args, reference, variants):
    arrays = [reference.vertices]
    for variant in variants.values():
        arrays.extend((variant["mesh"].vertices, variant["points"]))
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
    radius = args.orbit_radius or max(3.2, 1.08 * required)
    return center, radius


def render(args, items, center, radius, name):
    output_dir = args.output_dir / args.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    gif_path = output_dir / f"{name}.gif"
    mp4_path = output_dir / f"{name}.mp4"

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background((1.0, 1.0, 1.0, 1.0))
    renderer.scene.show_skybox(False)
    initialize_lighting(renderer, args.light_strength)
    for index, (geometry, material_record) in enumerate(items):
        if material_record is None:
            renderer.scene.add_model(f"geometry_{index}", geometry)
        else:
            renderer.scene.add_geometry(
                f"geometry_{index}", geometry, material_record
            )

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

    Image.fromarray(frames[0]).save(png_path)
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
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=round(1000 / gif_fps),
        loop=0,
        disposal=2,
        optimize=True,
    )

    if args.mp4:
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
    del renderer
    print(f"saved {gif_path}")


def save_color_scale(path, maximum_fraction):
    width, height = 512, 62
    colors = error_colors(np.linspace(0, 1, width - 32), 1)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for x, color in enumerate(colors, 16):
        draw.line((x, 10, x, 28), fill=tuple(color))
    draw.text((16, 35), "0", fill="black")
    draw.text(
        (width // 2, 35), f"{maximum_fraction * 50:g}% object width",
        fill="black", anchor="ma",
    )
    draw.text(
        (width - 16, 35), f">={maximum_fraction * 100:g}% object width",
        fill="black", anchor="ra",
    )
    image.save(path)


def main():
    args = parse_args()
    if args.max_error_fraction <= 0:
        raise ValueError("--max-error-fraction must be positive")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive")
    if args.resolution < 1:
        raise ValueError("--resolution must be positive")
    if args.orbit_radius is not None and args.orbit_radius <= 0:
        raise ValueError("--orbit-radius must be positive")
    if min(
        args.frames, args.fps, args.gif_fps, args.gif_size,
        args.width, args.height,
    ) < 1:
        raise ValueError("Frame, image, and FPS settings must be positive")

    ensure_artifacts(args)
    reference, variants, display_center, display_scale = load_scene(args)
    textured_reference = load_textured_reference(
        args.input_dir / f"{args.sample_id}_reference.glb",
        display_center,
        display_scale,
    )
    camera_center, camera_radius = camera_fit(args, reference, variants)
    output_dir = args.output_dir / args.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    root, record = load_record(args)
    image_path = Path(record["image_path"])
    image_path = image_path if image_path.is_absolute() else root / image_path
    shutil.copy2(image_path, output_dir / "input_view.png")

    render(
        args,
        [(mesh_geometry(reference), material(MESH_COLOR))],
        camera_center,
        camera_radius,
        "original_object",
    )
    render(
        args,
        [(textured_reference, None)],
        camera_center,
        camera_radius,
        "original_object_textured",
    )

    maximum = 2.0 * args.max_error_fraction
    error_report = {}
    for source, variant in variants.items():
        points = variant["points"]
        reconstruction = variant["mesh"]
        errors = variant["sdf_error"]
        if len(errors) != len(points):
            raise ValueError(f"SDF output size does not match {source} points")
        error_report[source] = {
            "mean_fraction_of_object_width": float(errors.mean() / 2.0),
            "median_fraction_of_object_width": float(np.median(errors) / 2.0),
            "p95_fraction_of_object_width": float(np.quantile(errors, 0.95) / 2.0),
        }

        render(
            args,
            [
                (mesh_geometry(reconstruction), material(MESH_COLOR)),
                (
                    particles(points, INPUT_COLOR, args.point_size),
                    material((1.0, 1.0, 1.0), unlit=True),
                ),
            ],
            camera_center,
            camera_radius,
            source,
        )
        render(
            args,
            [
                (mesh_geometry(reconstruction), material(MESH_COLOR)),
                (
                    particles(
                        points, error_colors(errors, maximum), args.point_size
                    ),
                    material((1.0, 1.0, 1.0), unlit=True),
                ),
            ],
            camera_center,
            camera_radius,
            f"{source}_sdf_error",
        )

    save_color_scale(output_dir / "input_error_color_scale.png", args.max_error_fraction)
    error_report["visualization"] = {
        "decoder_grid_resolution": args.resolution,
        "mesh_smoothing": False,
        "error": "absolute decoder SDF at each prepared input point",
        "camera_center": camera_center.tolist(),
        "camera_radius": camera_radius,
    }
    with (output_dir / "input_error.json").open("w") as file:
        json.dump(error_report, file, indent=2)


if __name__ == "__main__":
    main()
