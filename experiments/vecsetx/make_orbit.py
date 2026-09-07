import argparse
import json
import os
import subprocess
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
VIEWS = ("input", "reconstruction", "adherence", "extrapolation")
INPUT_COLOR = np.array([45, 125, 210], dtype=np.uint8)


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
        default=Path("experiments/vecsetx/outputs/reconstruction"),
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
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    parser.add_argument(
        "--color-scale", choices=["shared", "per-variant"], default="shared"
    )
    parser.add_argument("--max-error", type=float)
    parser.add_argument("--error-percentile", type=float, default=95.0)
    parser.add_argument("--metric-points", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--gif-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument("--orbit-radius", type=float, default=3.2)
    parser.add_argument("--orbit-height", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=0.02)
    parser.add_argument("--display-points", type=int, default=2048)
    parser.add_argument("--error-anchor-fraction", type=float, default=0.1)
    parser.add_argument("--mesh-opacity", type=float, default=0.8)
    parser.add_argument("--depth-tolerance", type=float)
    parser.add_argument("--ring-depth", type=float)
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


def ensure_variants(args):
    missing = []
    for source in args.sources:
        point_source = "full_surface" if source == "full_surface_unmasked" else source
        mesh_path = args.input_dir / f"{args.sample_id}_{source}.obj"
        points_path = args.input_dir / f"{args.sample_id}_{point_source}_points.npy"
        if not mesh_path.exists() or not points_path.exists():
            missing.append(source)
    if not missing:
        return

    print(
        f"Preparing missing VecSetX artifacts for {args.sample_id}: "
        f"{', '.join(missing)}",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.vecsetx.reconstruct",
            "--pipeline-config",
            str(args.pipeline_config),
            "--touch-config",
            str(args.touch_config),
            "--full-surface-config",
            str(args.full_surface_config),
            "--output-dir",
            str(args.input_dir),
            "--split",
            args.split,
            "--sample-id",
            args.sample_id,
            "--skip-report",
        ],
        check=True,
    )


def load_variants(args, metric_points, seed):
    variants = {}
    for source in args.sources:
        point_source = "full_surface" if source == "full_surface_unmasked" else source
        mesh_path = args.input_dir / f"{args.sample_id}_{source}.obj"
        points_path = args.input_dir / f"{args.sample_id}_{point_source}_points.npy"
        if not mesh_path.exists() or not points_path.exists():
            raise FileNotFoundError(f"Missing {mesh_path} or {points_path}")
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        points = np.load(points_path, allow_pickle=False)
        reconstructed, _ = trimesh.sample.sample_surface(
            mesh, metric_points, seed=seed
        )
        variants[source] = {
            "mesh": mesh,
            "points": points,
            "reconstructed": reconstructed,
            "adherence_errors": cKDTree(reconstructed).query(points)[0],
            "extrapolation_errors": cKDTree(points).query(reconstructed)[0],
            "vertex_extrapolation_errors": cKDTree(points).query(mesh.vertices)[0],
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


def sample_display_points(points, errors, count, error_fraction):
    if count == 0 or len(points) <= count:
        return points, errors

    error_count = min(count, round(count * error_fraction))
    error_order = np.argsort(errors)
    ranks = np.linspace(0, len(points) - 1, error_count).round().astype(int)
    selected = list(error_order[ranks])
    selected_mask = np.zeros(len(points), dtype=bool)
    selected_mask[selected] = True

    distances = np.full(len(points), np.inf)
    if not selected:
        selected.append(0)
        selected_mask[0] = True
    for index in selected:
        distances = np.minimum(
            distances, np.square(points - points[index]).sum(axis=1)
        )
    distances[selected_mask] = -1

    while len(selected) < count:
        index = int(np.argmax(distances))
        selected.append(index)
        selected_mask[index] = True
        distances = np.minimum(
            distances, np.square(points - points[index]).sum(axis=1)
        )
        distances[selected_mask] = -1

    return points[selected], errors[selected]


def mesh_geometry(mesh, colors=None):
    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    if colors is not None:
        geometry.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)
    geometry.compute_vertex_normals()
    return geometry


def mesh_material(opacity):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit" if opacity == 1 else "defaultLitTransparency"
    material.base_color = (1.0, 1.0, 1.0, opacity)
    material.has_alpha = opacity < 1
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


def reconstruct_alpha(black, white, background):
    black = black.astype(np.float32)
    white = white.astype(np.float32)
    black_border = np.concatenate((black[0], black[-1], black[:, 0], black[:, -1]))
    white_border = np.concatenate((white[0], white[-1], white[:, 0], white[:, -1]))
    black_background = np.median(black_border, axis=0)
    white_background = np.median(white_border, axis=0)
    background_range = np.maximum(white_background - black_background, 1.0)

    alpha = 1.0 - np.median((white - black) / background_range, axis=-1)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[background] = 0.0

    safe_alpha = np.maximum(alpha[..., None], 1 / 255)
    rgb = (black - (1.0 - alpha[..., None]) * black_background) / safe_alpha
    rgb = np.clip(rgb, 0, 255)
    rgb[alpha == 0] = 255
    return np.dstack((rgb.astype(np.uint8), (alpha * 255).astype(np.uint8)))


def high_quality_gif_frame(frame, size):
    # Copied from data_generation/objaverse-dexonomy/make_orbit.py.
    image = Image.fromarray(frame, "RGBA").convert("RGBa")
    image = image.resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")
    rgba = np.asarray(image)
    foreground = rgba[..., 3] >= 128

    image = Image.fromarray(rgba[..., :3]).quantize(
        colors=255, dither=Image.Dither.NONE
    )
    palette = image.getpalette()
    palette.extend([0] * (768 - len(palette)))
    palette[255 * 3:255 * 3 + 3] = [255, 255, 255]
    image.putpalette(palette)
    image.paste(255, mask=Image.fromarray((~foreground).astype(np.uint8) * 255))
    return image


def add_view(renderer, args, view, variant, maximum):
    colors = None
    if view == "extrapolation":
        colors = error_colors(variant["vertex_extrapolation_errors"], maximum)
    opacity = 1.0 if view == "reconstruction" else args.mesh_opacity
    material = mesh_material(opacity)
    if colors is None:
        material.base_color = (0.71, 0.71, 0.71, opacity)
    renderer.scene.add_geometry(
        "mesh", mesh_geometry(variant["mesh"], colors), material
    )

    if view == "reconstruction":
        return None, None
    if view == "adherence":
        points, errors = sample_display_points(
            variant["points"], variant["adherence_errors"],
            args.display_points, args.error_anchor_fraction,
        )
        return points, error_colors(errors, maximum)

    points, _ = sample_display_points(
        variant["points"], variant["adherence_errors"],
        args.display_points, 0,
    )
    return points, np.broadcast_to(INPUT_COLOR, (len(points), 3))


def project_points(points, eye, center, fov, width, height):
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    relative = points - eye
    depth = relative @ forward
    focal = height / (2 * np.tan(np.radians(fov) / 2))
    x = width / 2 + (relative @ right) * focal / depth
    y = height / 2 - (relative @ up) * focal / depth
    pixels = np.stack((x, y), axis=1)
    inside = (
        (depth > 0)
        & (x >= 0) & (x < width)
        & (y >= 0) & (y < height)
    )
    return pixels, depth, inside, focal


def draw_points(frame, mesh_depth, points, colors, eye, center, args):
    pixels, point_depth, inside, focal = project_points(
        points, eye, center, args.fov, args.width, args.height
    )
    indices = np.flatnonzero(inside)
    if len(indices) == 0:
        return frame

    xy = np.rint(pixels[indices]).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, args.width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, args.height - 1)
    surface_depth = mesh_depth[xy[:, 1], xy[:, 0]]
    depth_difference = point_depth[indices] - surface_depth
    hidden = (
        np.isfinite(surface_depth)
        & (depth_difference > args.depth_tolerance)
    )
    radii = np.maximum(
        2, np.rint(args.point_size * focal / (2 * point_depth[indices])).astype(int)
    )

    image = Image.fromarray(frame, "RGBA")
    draw = ImageDraw.Draw(image)
    order = np.argsort(point_depth[indices])[::-1]
    for position in order:
        if hidden[position] and depth_difference[position] > args.ring_depth:
            continue
        x, y = xy[position]
        radius = int(radii[position])
        box = (x - radius, y - radius, x + radius, y + radius)
        color = tuple(int(value) for value in colors[indices[position]]) + (255,)
        if hidden[position]:
            draw.ellipse(box, outline=color, width=max(1, radius // 2))
        else:
            draw.ellipse(box, fill=color)
    return np.asarray(image)


def render(args, source, view, variant, maximum=None):
    output_dir = (args.output_dir or args.input_dir / "orbits") / args.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.show_skybox(False)
    initialize_lighting(renderer, args.light_strength)
    points, colors = add_view(renderer, args, view, variant, maximum)

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
        depth = np.asarray(renderer.render_to_depth_image())
        image = reconstruct_alpha(black, white, depth >= 1.0)
        if points is not None:
            mesh_depth = np.asarray(
                renderer.render_to_depth_image(z_in_view_space=True)
            )
            image = draw_points(
                image, mesh_depth, points, colors, eye, center, args
            )
        frames.append(image)
        print(
            f"{source}/{view}: frame {frame + 1}/{args.frames}",
            end="\r", flush=True,
        )
    print()

    images = [Image.fromarray(frame, "RGBA") for frame in frames]
    name = f"{source}_{view}"
    images[0].save(output_dir / f"{name}.png")
    gif_fps = min(args.gif_fps, args.fps)
    frame_count = max(1, round(len(images) * gif_fps / args.fps))
    indices = np.linspace(0, len(images), frame_count, endpoint=False, dtype=int)
    gif_frames = [
        high_quality_gif_frame(frames[index], args.gif_size)
        for index in indices
    ]
    gif_frames[0].save(
        output_dir / f"{name}.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=round(1000 / gif_fps),
        loop=0,
        disposal=2,
        optimize=False,
        transparency=255,
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
            str(output_dir / f"{name}.mp4"), codec="libx264", audio=False,
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


def save_point_legend(path):
    image = Image.new("RGBA", (500, 48), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    color = tuple(int(value) for value in INPUT_COLOR) + (255,)
    draw.ellipse((10, 14, 22, 26), fill=color)
    draw.text((30, 13), "in front of surface", fill=(0, 0, 0, 255))
    draw.ellipse((250, 14, 262, 26), outline=color, width=2)
    draw.text((270, 13), "behind surface", fill=(0, 0, 0, 255))
    image.save(path)


def main():
    args = parse_args()
    if not 0 < args.error_percentile <= 100:
        raise ValueError("--error-percentile must be in (0, 100]")
    if args.max_error is not None and args.max_error <= 0:
        raise ValueError("--max-error must be positive")
    if not 0 < args.mesh_opacity < 1:
        raise ValueError("--mesh-opacity must be in (0, 1)")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive")
    if args.depth_tolerance is not None and args.depth_tolerance < 0:
        raise ValueError("--depth-tolerance must be non-negative")
    if args.ring_depth is not None and args.ring_depth <= 0:
        raise ValueError("--ring-depth must be positive")
    if args.display_points < 0:
        raise ValueError("--display-points must be non-negative")
    if not 0 <= args.error_anchor_fraction <= 1:
        raise ValueError("--error-anchor-fraction must be in [0, 1]")
    if min(
        args.frames, args.fps, args.gif_fps, args.gif_size,
        args.width, args.height
    ) < 1:
        raise ValueError("frame, image, and FPS settings must be positive")
    ensure_variants(args)
    metric_points, seed, resolution = load_settings(args)
    if args.depth_tolerance is None:
        args.depth_tolerance = 2 / resolution
    if args.ring_depth is None:
        args.ring_depth = 8 / resolution
    if args.ring_depth <= args.depth_tolerance:
        raise ValueError("--ring-depth must be greater than --depth-tolerance")
    variants = load_variants(args, metric_points, seed)
    error_names = {
        "adherence": "adherence_errors",
        "extrapolation": "extrapolation_errors",
    }
    shared_maximums = {
        view: args.max_error or float(np.percentile(
            np.concatenate([variant[key] for variant in variants.values()]),
            args.error_percentile,
        ))
        for view, key in error_names.items()
        if view in args.views
    }

    output_dir = (args.output_dir or args.input_dir / "orbits") / args.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    save_point_legend(output_dir / "point_visibility_legend.png")
    scales = {}
    for source, variant in variants.items():
        scales[source] = {}
        for view in args.views:
            maximum = None
            if view in error_names:
                errors = variant[error_names[view]]
                maximum = shared_maximums[view]
                if args.color_scale == "per-variant" and args.max_error is None:
                    maximum = float(np.percentile(errors, args.error_percentile))
                scales[source][view] = {
                    "maximum": maximum,
                    "mean": float(errors.mean()),
                    "median": float(np.median(errors)),
                    "p95": float(np.percentile(errors, 95)),
                }
                if args.color_scale == "per-variant":
                    save_color_scale(
                        output_dir / f"{source}_{view}_color_scale.png",
                        maximum,
                        resolution,
                    )
            render(args, source, view, variant, maximum)

    if args.color_scale == "shared":
        for view, maximum in shared_maximums.items():
            save_color_scale(
                output_dir / f"{view}_color_scale.png", maximum, resolution
            )
    with (output_dir / "error_scale.json").open("w") as file:
        json.dump({"color_scale": args.color_scale, "sources": scales}, file, indent=2)
    print(f"Saved transparent orbits to {output_dir}")


if __name__ == "__main__":
    main()
