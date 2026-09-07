import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

if sys.platform.startswith("linux"):
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import open3d as o3d
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


SOURCES = ("full_surface", "full_surface_unmasked", "touch", "joint")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/vecsetx/outputs/reconstruction"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--sources", nargs="+", choices=SOURCES,
        default=["full_surface", "touch", "joint"],
    )
    parser.add_argument(
        "--color-scale", choices=["shared", "per-variant"], default="shared"
    )
    parser.add_argument("--max-error", type=float)
    parser.add_argument("--error-percentile", type=float, default=95.0)
    parser.add_argument("--metric-points", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--webp-fps", type=int, default=10)
    parser.add_argument("--webp-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument("--orbit-radius", type=float, default=3.2)
    parser.add_argument("--orbit-height", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=0.012)
    parser.add_argument("--mesh-opacity", type=float, default=0.18)
    parser.add_argument("--light-strength", type=float, default=1.0)
    parser.add_argument("--mp4", action="store_true")
    return parser.parse_args()


def load_settings(args):
    path = args.input_dir / "metrics.json"
    settings = {}
    if path.exists():
        with path.open() as file:
            settings = json.load(file).get("settings", {})
    return (
        args.metric_points or settings.get("metric_points", 32768),
        args.seed if args.seed is not None else settings.get("seed", 29),
        settings.get("resolution", 128),
    )


def load_variants(args, metric_points, seed):
    variants = {}
    for source in args.sources:
        point_source = "full_surface" if source == "full_surface_unmasked" else source
        mesh_path = args.input_dir / f"{args.sample_id}_{source}.obj"
        points_path = args.input_dir / f"{args.sample_id}_{point_source}_points.npy"
        if not mesh_path.exists() or not points_path.exists():
            raise FileNotFoundError(
                f"Missing {mesh_path} or {points_path}. Re-run reconstruct.py for "
                f"sample {args.sample_id!r} to save its prepared input points."
            )
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        points = np.load(points_path, allow_pickle=False)
        reconstructed, _ = trimesh.sample.sample_surface(
            mesh, metric_points, seed=seed
        )
        variants[source] = {
            "mesh": mesh,
            "points": points,
            "errors": cKDTree(reconstructed).query(points)[0],
        }
    return variants


def error_colors(errors, maximum):
    maximum = max(maximum, np.finfo(np.float64).eps)
    values = np.clip(errors / maximum, 0.0, 1.0)
    green = np.array([44, 162, 95], dtype=np.float64)
    yellow = np.array([253, 216, 53], dtype=np.float64)
    red = np.array([215, 48, 39], dtype=np.float64)
    first = green + (yellow - green) * (2 * values[:, None])
    second = yellow + (red - yellow) * (2 * values[:, None] - 1)
    return np.where(values[:, None] <= 0.5, first, second).astype(np.uint8)


def particles(points, colors, size):
    sphere = trimesh.creation.icosphere(subdivisions=0, radius=size / 2)
    vertices_per_point = len(sphere.vertices)
    vertices = (points[:, None] + sphere.vertices[None]).reshape(-1, 3)
    faces = sphere.faces[None] + vertices_per_point * np.arange(len(points))[:, None, None]
    colors = np.repeat(colors, vertices_per_point, axis=0) / 255.0

    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.reshape(-1, 3)),
    )
    geometry.vertex_colors = o3d.utility.Vector3dVector(colors)
    geometry.compute_vertex_normals()
    return geometry


def mesh_geometry(mesh):
    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    geometry.compute_vertex_normals()
    return geometry


def mesh_material(opacity):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = (0.71, 0.71, 0.71, opacity)
    material.has_alpha = True
    return material


def particle_material():
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = (1.0, 1.0, 1.0, 1.0)
    material.sRGB_color = True
    return material


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


def reconstruct_alpha(black, white):
    black = black.astype(np.float32)
    white = white.astype(np.float32)
    black_border = np.concatenate((black[0], black[-1], black[:, 0], black[:, -1]))
    white_border = np.concatenate((white[0], white[-1], white[:, 0], white[:, -1]))
    black_background = np.median(black_border, axis=0)
    white_background = np.median(white_border, axis=0)
    background_range = np.maximum(white_background - black_background, 1.0)

    alpha = 1.0 - np.median((white - black) / background_range, axis=-1)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[alpha < 1 / 255] = 0.0

    safe_alpha = np.maximum(alpha[..., None], 1 / 255)
    rgb = (black - (1.0 - alpha[..., None]) * black_background) / safe_alpha
    rgb = np.clip(rgb, 0, 255)
    rgb[alpha == 0] = 255
    return np.dstack((rgb.astype(np.uint8), (alpha * 255).astype(np.uint8)))


def render(args, source, variant, maximum):
    output_dir = (args.output_dir or args.input_dir / "orbits") / args.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.show_skybox(False)
    initialize_lighting(renderer, args.light_strength)
    renderer.scene.add_geometry(
        "mesh", mesh_geometry(variant["mesh"]), mesh_material(args.mesh_opacity)
    )
    renderer.scene.add_geometry(
        "points",
        particles(
            variant["points"],
            error_colors(variant["errors"], maximum),
            args.point_size,
        ),
        particle_material(),
    )

    frames = []
    center = np.zeros(3)
    for frame in range(args.frames):
        angle = 2 * np.pi * frame / args.frames
        eye = center + np.array([
            args.orbit_radius * np.sin(angle),
            args.orbit_height,
            -args.orbit_radius * np.cos(angle),
        ])
        renderer.setup_camera(args.fov, center, eye, (0.0, 1.0, 0.0))
        renderer.scene.set_background((0.0, 0.0, 0.0, 1.0))
        black = np.asarray(renderer.render_to_image())[..., :3].copy()
        renderer.scene.set_background((1.0, 1.0, 1.0, 1.0))
        white = np.asarray(renderer.render_to_image())[..., :3].copy()
        frames.append(reconstruct_alpha(black, white))
        print(f"{source}: frame {frame + 1}/{args.frames}", end="\r", flush=True)
    print()

    images = [Image.fromarray(frame, "RGBA") for frame in frames]
    images[0].save(output_dir / f"{source}.png")
    webp_fps = min(args.webp_fps, args.fps)
    frame_count = max(1, round(len(images) * webp_fps / args.fps))
    indices = np.linspace(0, len(images), frame_count, endpoint=False, dtype=int)
    webp_frames = [
        images[index].convert("RGBa").resize(
            (args.webp_size, args.webp_size), Image.Resampling.LANCZOS
        ).convert("RGBA")
        for index in indices
    ]
    webp_frames[0].save(
        output_dir / f"{source}.webp",
        save_all=True,
        append_images=webp_frames[1:],
        duration=round(1000 / webp_fps),
        loop=0,
        lossless=True,
        method=6,
    )

    if args.mp4:
        try:
            from moviepy import ImageSequenceClip
        except ImportError:
            from moviepy.editor import ImageSequenceClip
        rgb_frames = []
        for frame in frames:
            alpha = frame[..., 3:4].astype(np.float32) / 255.0
            rgb_frames.append(
                (frame[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
            )
        clip = ImageSequenceClip(rgb_frames, fps=args.fps)
        clip.write_videofile(
            str(output_dir / f"{source}.mp4"), codec="libx264", audio=False,
            logger=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        clip.close()


def save_color_scale(path, maximum, resolution):
    width, height = 512, 62
    values = np.linspace(0, maximum, width - 32)
    colors = error_colors(values, maximum)
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    for x, color in enumerate(colors, 16):
        draw.line((x, 10, x, 28), fill=tuple(color) + (255,))
    draw.text((16, 35), "0", fill=(0, 0, 0, 255))
    label = f"{maximum:.4f}  ({maximum / (2 / resolution):.2f} grid cells)"
    draw.text((width - 16, 35), label, fill=(0, 0, 0, 255), anchor="ra")
    image.save(path)


def main():
    args = parse_args()
    if not 0 < args.error_percentile <= 100:
        raise ValueError("--error-percentile must be in (0, 100]")
    if args.max_error is not None and args.max_error <= 0:
        raise ValueError("--max-error must be positive")
    if not 0 < args.mesh_opacity < 1:
        raise ValueError("--mesh-opacity must be in (0, 1)")
    if min(
        args.frames, args.fps, args.webp_fps, args.webp_size,
        args.width, args.height
    ) < 1:
        raise ValueError("frame, image, and FPS settings must be positive")

    metric_points, seed, resolution = load_settings(args)
    variants = load_variants(args, metric_points, seed)
    all_errors = np.concatenate([variant["errors"] for variant in variants.values()])
    shared_maximum = args.max_error or float(
        np.percentile(all_errors, args.error_percentile)
    )

    output_dir = (args.output_dir or args.input_dir / "orbits") / args.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    scales = {}
    for source, variant in variants.items():
        maximum = shared_maximum
        if args.color_scale == "per-variant" and args.max_error is None:
            maximum = float(np.percentile(variant["errors"], args.error_percentile))
        scales[source] = {
            "maximum": maximum,
            "mean": float(variant["errors"].mean()),
            "median": float(np.median(variant["errors"])),
            "p95": float(np.percentile(variant["errors"], 95)),
        }
        render(args, source, variant, maximum)
        if args.color_scale == "per-variant":
            save_color_scale(output_dir / f"{source}_color_scale.png", maximum, resolution)

    if args.color_scale == "shared":
        save_color_scale(output_dir / "color_scale.png", shared_maximum, resolution)
    with (output_dir / "error_scale.json").open("w") as file:
        json.dump({"color_scale": args.color_scale, "sources": scales}, file, indent=2)
    print(f"Saved transparent orbits to {output_dir}")


if __name__ == "__main__":
    main()
