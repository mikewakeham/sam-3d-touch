import argparse
import colorsys
import json
import os
import sys
from pathlib import Path

import numpy as np

if sys.platform.startswith("linux"):
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import open3d as o3d
import trimesh
try:
    from moviepy import ImageSequenceClip
except ImportError:
    from moviepy.editor import ImageSequenceClip
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--view-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("orbit_outputs"))
    parser.add_argument("--name")
    parser.add_argument(
        "--show", nargs="+",
        choices=["mesh", "mesh_textured", "pointmap", "touch", "camera"],
    )
    parser.add_argument("--variants", type=json.loads)
    parser.add_argument("--touch-levels", type=json.loads)
    parser.add_argument("--contacts", type=int)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--density-fraction", type=float, default=1.0)
    parser.add_argument("--points-per-contact", type=int)
    parser.add_argument(
        "--visibility", choices=["all", "not-visible", "visible", "hidden", "unknown"],
        default="not-visible",
    )
    parser.add_argument("--pointmap-stride", type=int, default=2)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--gif-size", type=int, default=512)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--high-quality-transparency", action="store_true")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fov", type=float)
    parser.add_argument("--orbit-radius", type=float)
    parser.add_argument("--orbit-height", type=float)
    parser.add_argument("--point-size", type=float, default=0.01)
    parser.add_argument("--touch-point-size", type=float, default=0.01)
    parser.add_argument("--center-size", type=float, default=0.01)
    parser.add_argument("--light-strength", type=float, default=1.0)
    return parser.parse_args()


def load_view(args):
    object_dir = args.data_root / "objects" / args.object_id
    generated_dir = args.data_root / "generated_data" / args.object_id
    view_dir = generated_dir / "views" / f"{args.view_id:03d}"

    with np.load(generated_dir / "object_transform.npz") as data:
        T_object_from_source = data["T_normalized_from_source"]

    with np.load(view_dir / "camera.npz") as data:
        K = data["K"]
        T_camera_from_object = data["T_camera_from_object"]

    # Raw OBJ -> normalized object -> OpenCV camera -> SAM camera.
    T_sam_from_object = np.diag([-1.0, -1.0, 1.0, 1.0]) @ T_camera_from_object
    T_sam_from_source = T_sam_from_object @ T_object_from_source
    mesh = trimesh.load(str(object_dir / "model.obj"), force="mesh", process=False, skip_materials=True)
    mesh.apply_transform(T_sam_from_source)
    textured_mesh = None
    if "mesh_textured" in args.show:
        textured_mesh = o3d.io.read_triangle_model(str(object_dir / "model.obj"))
        if not textured_mesh.meshes:
            raise ValueError(f"Could not load textured mesh from {object_dir / 'model.obj'}")
        for part in textured_mesh.meshes:
            part.mesh.transform(T_sam_from_source)
            part.mesh.compute_vertex_normals()
        materials = []
        for imported_material in textured_mesh.materials:
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.base_color = (1.0, 1.0, 1.0, 1.0)
            material.albedo_img = imported_material.albedo_img
            materials.append(material)
        textured_mesh.materials = materials

    rgba = np.array(Image.open(view_dir / "image.png").convert("RGBA"))
    pointmap = np.load(view_dir / "pointmap.npy")

    with np.load(view_dir / "touches.npz", allow_pickle=False) as data:
        touch = dict(data)

    return mesh, textured_mesh, rgba, pointmap, touch, K, T_sam_from_object


def select_touches(touch, args):
    count = len(touch["centers_camera"])
    if args.contacts is not None:
        if args.contacts > count:
            raise ValueError(f"Requested {args.contacts} contacts but the file contains {count}")
        count = args.contacts
    if args.radius is not None and np.any(args.radius > touch["patch_radius"][:count] + 1e-6):
        raise ValueError("Requested radius is larger than the saved touch radius")

    visibility_labels = {"unknown": -1, "hidden": 0, "visible": 1}
    contacts = []

    for i in range(count):
        a, b = touch["offsets"][i:i + 2]
        indices = np.arange(a, b)

        if args.radius is not None:
            indices = indices[touch["geodesic_distance"][indices] <= args.radius]
        if args.visibility == "not-visible":
            indices = indices[touch["point_visibility"][indices] != 1]
        elif args.visibility != "all":
            label = visibility_labels[args.visibility]
            indices = indices[touch["point_visibility"][indices] == label]

        center_matches = np.flatnonzero(touch["point_ids"][a:b] == touch["center_point_ids"][i]) + a
        center_index = int(center_matches[0])
        others = indices[indices != center_index]
        others = others[touch["keep_priority"][others] < args.density_fraction]

        if args.points_per_contact is not None:
            order = np.argsort(touch["keep_priority"][others])
            others = others[order[:max(0, args.points_per_contact - 1)]]

        indices = np.concatenate(([center_index], others))
        R = touch["R_camera_from_local"][i]
        center = touch["centers_camera"][i]
        points = touch["points_local"][indices] @ R.T + center
        color = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i * 0.618 % 1, 0.75, 1))
        contacts.append((points, center, color))

    return contacts


def particles(points, colors, size):
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
    geometry.compute_vertex_normals()
    return geometry


def particle_material():
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = (1.0, 1.0, 1.0, 1.0)
    material.sRGB_color = True
    return material


def camera_frustum(K, width, height, scale=0.15):
    pixels = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
    corners = pixels @ np.linalg.inv(K).T
    corners *= scale / corners[:, 2:3]
    corners[:, :2] *= -1
    points = np.concatenate((np.zeros((1, 3)), corners))
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ])
    geometry = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    geometry.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([255, 180, 50]) / 255.0, (len(lines), 1))
    )
    return geometry


def initialize_lighting(renderer, T_sam_from_object, strength):
    scene = renderer.scene.scene
    rotation = T_sam_from_object[:3, :3]
    translation = T_sam_from_object[:3, 3]
    key_position = rotation @ np.array([4.0, 1.0, 6.0]) + translation
    top_direction = rotation @ np.array([0.0, 0.0, -1.0])
    bottom_direction = rotation @ np.array([0.0, 0.0, 1.0])
    white = np.ones(3, dtype=np.float32)

    scene.enable_sun_light(False)
    scene.enable_indirect_light(True)
    scene.set_indirect_light_intensity(50000.0 * strength)
    scene.add_point_light(
        "KeyLight", white, key_position.astype(np.float32), 5e5 * strength, 100.0, False
    )
    scene.add_directional_light(
        "TopLight", white, top_direction.astype(np.float32), 5e4 * strength, False
    )
    scene.add_directional_light(
        "BottomLight", white, bottom_direction.astype(np.float32), 1e4 * strength, False
    )


def add_geometry(renderer, args, mesh, textured_mesh, rgba, pointmap, contacts, K):
    if "mesh" in args.show:
        geometry = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces),
        )
        geometry.compute_vertex_normals()
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = (0.71, 0.71, 0.71, 1.0)
        renderer.scene.add_geometry("mesh", geometry, material)

    if "mesh_textured" in args.show:
        renderer.scene.add_model("mesh_textured", textured_mesh)

    if "pointmap" in args.show:
        stride = args.pointmap_stride
        points = pointmap[::stride, ::stride]
        pixels = rgba[::stride, ::stride]
        valid = np.isfinite(points).all(axis=-1) & (pixels[..., 3] > 0)
        geometry = particles(points[valid], pixels[..., :3][valid], args.point_size)
        renderer.scene.add_geometry("pointmap", geometry, particle_material())

    if "touch" in args.show:
        for i, (points, center, color) in enumerate(contacts):
            geometry = particles(points, color, args.touch_point_size)
            renderer.scene.add_geometry(f"touch_{i}", geometry, particle_material())

            marker = particles(center[None], color, args.center_size)
            renderer.scene.add_geometry(f"center_{i}", marker, particle_material())

    if "camera" in args.show:
        height, width = rgba.shape[:2]
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 2.0
        renderer.scene.add_geometry(
            "render_camera", camera_frustum(K, width, height), material
        )


def output_name(args):
    if args.name:
        return args.name

    parts = [f"view{args.view_id:03d}", "-".join(args.show)]
    if "touch" in args.show:
        if args.touch_level:
            parts.append(args.touch_level)
        parts.append(f"k{args.contacts if args.contacts is not None else 'all'}")
        parts.append(f"r{args.radius:g}" if args.radius is not None else "rall")
        if args.points_per_contact is not None:
            parts.append(f"p{args.points_per_contact}")
        if args.density_fraction != 1.0:
            parts.append(f"d{args.density_fraction:g}")
        if args.visibility != "all":
            parts.append(args.visibility)
    return "_".join(parts)


def reconstruct_alpha(black, white, background):
    black = black.astype(np.float32)
    white = white.astype(np.float32)
    black_background = np.median(black[background], axis=0)
    white_background = np.median(white[background], axis=0)
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


def render(args, mesh, textured_mesh, rgba, pointmap, contacts, K, T_sam_from_object):
    object_output_dir = args.output_dir / args.object_id
    object_output_dir.mkdir(parents=True, exist_ok=True)
    name = output_name(args)
    mp4_path = object_output_dir / f"{name}.mp4"
    gif_path = object_output_dir / f"{name}.gif"
    write_mp4 = not mp4_path.exists()
    write_gif = not gif_path.exists()
    if not write_mp4 and not write_gif:
        print(f"skipping existing {mp4_path} and {gif_path}")
        return mp4_path, gif_path

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background((1.0, 1.0, 1.0, 1.0))
    renderer.scene.show_skybox(False)
    initialize_lighting(renderer, T_sam_from_object, args.light_strength)
    add_geometry(renderer, args, mesh, textured_mesh, rgba, pointmap, contacts, K)

    center = T_sam_from_object[:3, 3]
    radius = args.orbit_radius if args.orbit_radius is not None else np.linalg.norm(center)
    height = args.orbit_height if args.orbit_height is not None else 0.0
    fov = args.fov
    if fov is None:
        fov = np.degrees(2.0 * np.arctan(rgba.shape[0] / (2.0 * K[1, 1])))

    frames = []
    transparent_frames = []

    for frame in range(args.frames):
        angle = 2 * np.pi * frame / args.frames
        eye = center + np.array([
            radius * np.sin(angle), height, -radius * np.cos(angle)
        ])
        renderer.setup_camera(fov, center, eye, (0.0, 1.0, 0.0))
        if args.high_quality_transparency:
            renderer.scene.set_background((0.0, 0.0, 0.0, 1.0))
            black = np.asarray(renderer.render_to_image())[..., :3].copy()
            renderer.scene.set_background((1.0, 1.0, 1.0, 1.0))
            white = np.asarray(renderer.render_to_image())[..., :3].copy()
            depth = np.asarray(renderer.render_to_depth_image())
            background = depth >= 1.0
            rgb = white.copy()
            rgb[background] = 255
            transparent = reconstruct_alpha(black, white, background)
        else:
            image = np.asarray(renderer.render_to_image())
            depth = np.asarray(renderer.render_to_depth_image())
            rgb = image[..., :3].copy()
            background = depth >= 1.0
            rgb[background] = 255
            alpha = np.where(background, 0, 255).astype(np.uint8)
            transparent = np.dstack((rgb, alpha))
        frames.append(rgb)
        transparent_frames.append(transparent)
        print(f"frame {frame + 1}/{args.frames}", end="\r", flush=True)

    if write_mp4:
        clip = ImageSequenceClip(frames, fps=args.fps)
        clip.write_videofile(
            str(mp4_path), codec="libx264", audio=False, logger=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        clip.close()
        print(f"saved {mp4_path}")
    else:
        print(f"skipping existing {mp4_path}")

    if write_gif:
        gif_frame_count = max(1, round(len(frames) * min(args.gif_fps, args.fps) / args.fps))
        gif_indices = np.linspace(0, len(frames), gif_frame_count, endpoint=False, dtype=int)
        if args.white_background:
            gif_frames = [
                Image.fromarray(frames[index]).resize(
                    (args.gif_size, args.gif_size), Image.Resampling.LANCZOS
                )
                for index in gif_indices
            ]
        elif args.high_quality_transparency:
            gif_frames = [
                high_quality_gif_frame(transparent_frames[index], args.gif_size)
                for index in gif_indices
            ]
        else:
            gif_frames = [
                Image.fromarray(transparent_frames[index], "RGBA").resize(
                    (args.gif_size, args.gif_size), Image.Resampling.LANCZOS
                )
                for index in gif_indices
            ]
        duration = round(1000 / min(args.gif_fps, args.fps))
        save_options = {"optimize": True}
        if args.high_quality_transparency:
            save_options = {"optimize": False, "transparency": 255}
        gif_frames[0].save(
            gif_path, save_all=True, append_images=gif_frames[1:],
            duration=duration, loop=0, disposal=2, **save_options,
        )
        print(f"saved {gif_path}")
    else:
        print(f"skipping existing {gif_path}")
    return mp4_path, gif_path


def main():
    args = parse_args()
    args.touch_level = None
    if args.white_background and args.high_quality_transparency:
        raise ValueError("Use either --white-background or --high-quality-transparency")
    if args.show is not None and args.variants is not None:
        raise ValueError("Use either --show or --variants, not both")
    variants = args.variants or [args.show or ["mesh", "pointmap", "touch"]]
    choices = {"mesh", "mesh_textured", "pointmap", "touch", "camera"}
    for variant in variants:
        if not isinstance(variant, list) or not variant or any(item not in choices for item in variant):
            raise ValueError("Each variant must be a nonempty list of show options")
        if "mesh" in variant and "mesh_textured" in variant:
            raise ValueError("A variant cannot include both mesh and mesh_textured")
    if args.name and (len(variants) > 1 or args.touch_levels):
        raise ValueError("--name can only be used with one variant")
    args.show = list(dict.fromkeys(item for variant in variants for item in variant))
    if args.contacts is not None and args.contacts < 1:
        raise ValueError("--contacts must be positive")
    if args.radius is not None and args.radius <= 0:
        raise ValueError("--radius must be positive")
    if not 0 < args.density_fraction <= 1:
        raise ValueError("--density-fraction must be in (0, 1]")
    if args.points_per_contact is not None and args.points_per_contact < 1:
        raise ValueError("--points-per-contact must be positive")
    if min(args.pointmap_stride, args.frames, args.fps, args.gif_fps,
           args.gif_size, args.width, args.height) < 1:
        raise ValueError("stride, frames, fps, GIF settings, width and height must be positive")
    if min(args.point_size, args.touch_point_size, args.center_size) <= 0:
        raise ValueError("point sizes must be positive")
    if args.light_strength <= 0:
        raise ValueError("--light-strength must be positive")

    touch_sets = []
    if args.touch_levels:
        for level in args.touch_levels:
            name = str(level["name"])
            contacts = int(level["contacts"])
            radius = float(level["radius"])
            points_per_contact = int(level["points_per_contact"])
            if contacts < 1 or radius <= 0 or points_per_contact < 1:
                raise ValueError("Touch level values must be positive")
            args.contacts = contacts
            args.radius = radius
            args.points_per_contact = points_per_contact
            touch_sets.append((name, contacts, radius, points_per_contact))

    mesh, textured_mesh, rgba, pointmap, touch, K, T_sam_from_object = load_view(args)
    contacts = []
    if args.touch_levels:
        selected_touch_sets = []
        for name, contact_count, radius, points_per_contact in touch_sets:
            args.contacts = contact_count
            args.radius = radius
            args.points_per_contact = points_per_contact
            selected_touch_sets.append((
                name, contact_count, radius, points_per_contact,
                select_touches(touch, args),
            ))
    else:
        contacts = select_touches(touch, args) if "touch" in args.show else []

    for variant in variants:
        args.show = variant
        if "touch" in variant and args.touch_levels:
            for name, contact_count, radius, points_per_contact, selected in selected_touch_sets:
                args.touch_level = name
                args.contacts = contact_count
                args.radius = radius
                args.points_per_contact = points_per_contact
                render(args, mesh, textured_mesh, rgba, pointmap, selected, K, T_sam_from_object)
        else:
            args.touch_level = None
            render(
                args, mesh, textured_mesh, rgba, pointmap, contacts, K, T_sam_from_object
            )


if __name__ == "__main__":
    main()
