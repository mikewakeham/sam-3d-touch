"""Render meshes, point clouds, dataset views or saved SAM3D evaluation results."""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if sys.platform.startswith('linux'):
    os.environ.setdefault('EGL_PLATFORM', 'surfaceless')

from evaluation.geometry import add_input_arguments, load_inputs, camera_fit, scene_bounds


def mesh_geometry(mesh):
    import open3d as o3d

    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    geometry.compute_vertex_normals()
    return geometry


def material(color):
    import open3d as o3d

    result = o3d.visualization.rendering.MaterialRecord()
    result.shader = "defaultLit"
    result.base_color = (*color, 1.0)
    result.sRGB_color = True
    return result


def initialize_lighting(renderer, strength, center, scale, up):
    direction = np.array([0., 0., -1.]) if up == "z" else np.array([0., -1., 0.])
    scene = renderer.scene.scene
    white = np.ones(3, dtype=np.float32)
    scene.enable_sun_light(False)
    scene.enable_indirect_light(True)
    scene.set_indirect_light_intensity(50000.0 * strength)
    scene.add_point_light(
        "KeyLight", white, center + scale * np.array([4.0, 1.0, 6.0], dtype=np.float32),
        5e5 * strength * scale ** 2, 100.0 * scale, False,
    )
    scene.add_directional_light(
        "TopLight", white, direction.astype(np.float32),
        5e4 * strength, False,
    )
    scene.add_directional_light(
        "BottomLight", white, -direction.astype(np.float32),
        1e4 * strength, False,
    )


def add_geometry(renderer, item, index, args):
    import open3d as o3d

    name = str(index)
    if 'mesh' in item:
        if args.textured and item.get('path') and Path(item['path']).suffix.lower() != '.npz':
            # Same Open3D material loader used in experiments/vecsetx/make_orbit.py.
            model = o3d.io.read_triangle_model(item['path'])
            if not model.meshes:
                raise ValueError(f"No textured mesh in {item['path']}")
            for part in model.meshes:
                part.mesh.compute_vertex_normals()
            renderer.scene.add_model(name, model)
        else:
            renderer.scene.add_geometry(name, mesh_geometry(item['mesh']), material((.71, .71, .71)))
    else:
        geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(item['points']))
        colors = item.get('colors')
        if args.normal_colors and 'normals' in item:
            colors = np.clip((item['normals'] + 1) * 127.5, 0, 255)
        if colors is not None:
            geometry.colors = o3d.utility.Vector3dVector(colors / 255.)
        point_material = material((1., 1., 1.) if colors is not None else (.25, .55, .85))
        point_material.shader = 'defaultUnlit'
        point_material.point_size = args.point_size
        renderer.scene.add_geometry(name, geometry, point_material)


def orbit_eye(center, radius, height, angle, up):
    if up == 'z':
        return center + np.array([radius * np.sin(angle), -radius * np.cos(angle), height])
    return center + np.array([radius * np.sin(angle), height, -radius * np.cos(angle)])


def save_frames(frames, output, args):
    # GIF/MP4 export copied from make_eval_orbit.py; preserve non-square aspect ratios.
    if args.mp4:
        try:
            from moviepy import ImageSequenceClip
        except ImportError:
            from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(frames, fps=args.fps)
        clip.write_videofile(str(output.with_suffix('.mp4')), codec='libx264', audio=False,
                             logger=None, ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        clip.close()
    gif_fps = min(args.gif_fps, args.fps)
    count = max(1, round(len(frames) * gif_fps / args.fps))
    indices = np.linspace(0, len(frames), count, endpoint=False, dtype=int)
    gif_frames = []
    for index in indices:
        image = Image.fromarray(frames[index])
        image.thumbnail((args.gif_size, args.gif_size), Image.Resampling.LANCZOS)
        gif_frames.append(image)
    gif_frames[0].save(output.with_suffix('.gif'), save_all=True, append_images=gif_frames[1:],
                       duration=round(1000 / gif_fps), loop=0, disposal=2, optimize=True)
    Image.fromarray(frames[0]).save(output.with_suffix('.png'))


def render(items, center, radius, args, output, scale):
    import open3d as o3d

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    try:
        renderer.scene.set_background((1., 1., 1., 1.))
        renderer.scene.show_skybox(False)
        initialize_lighting(renderer, args.light_strength, center, scale, args.up)
        for index, item in enumerate(items):
            add_geometry(renderer, item, index, args)
        frames = []
        up = (0., 0., 1.) if args.up == 'z' else (0., 1., 0.)
        for frame in range(args.frames):
            eye = orbit_eye(center, radius, args.orbit_height, 2 * np.pi * frame / args.frames, args.up)
            renderer.setup_camera(args.fov, center, eye, up)
            image = np.asarray(renderer.render_to_image())[..., :3].copy()
            image[np.asarray(renderer.render_to_depth_image()) >= 1.] = 255
            frames.append(image)
            print(f'{output.name}: [{frame + 1}/{args.frames}]', end='\r', flush=True)
        print()
        save_frames(frames, output, args)
    finally:
        del renderer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--gif-fps', type=int, default=10)
    parser.add_argument('--gif-size', type=int, default=512)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--fov', type=float, default=40.)
    parser.add_argument('--orbit-radius', type=float)
    parser.add_argument('--orbit-height', type=float, default=0.)
    parser.add_argument('--light-strength', type=float, default=1.)
    parser.add_argument('--point-size', type=float, default=4.)
    parser.add_argument('--mp4', action='store_true')
    parser.add_argument('--textured', action='store_true', help='Use original mesh materials where available')
    parser.add_argument('--normal-colors', action='store_true')
    parser.add_argument('--overlay', action='store_true', help='Render all inputs together instead of separate orbits')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.frames, args.fps, args.gif_fps, args.gif_size, args.width, args.height, args.point_size) <= 0:
        raise ValueError('Frame counts, rates, dimensions and point size must be positive')
    if not 0 < args.fov < 180 or (args.orbit_radius is not None and args.orbit_radius <= 0):
        raise ValueError('FOV must be in (0, 180) and orbit radius must be positive')
    if args.mp4 and (args.width % 2 or args.height % 2):
        raise ValueError('H264 output requires even width and height')
    items, images = load_inputs(args)
    output = args.output_dir or (args.evaluation_dir / 'orbits' if args.evaluation_dir else Path('outputs/orbits'))
    if args.evaluation_dir:
        output = output / args.sample_id
    scale = max(float(np.max(np.ptp(scene_bounds(items), axis=0))), 1e-3)
    center, radius = camera_fit(items, args.fov, args.width / args.height)
    radius = args.orbit_radius or radius
    groups = [('overlay', items)] if args.overlay else [(item['name'], [item]) for item in items]
    names = [re.sub(r'[^A-Za-z0-9_-]+', '_', name) for name, _ in groups]
    if len(set(names)) != len(names):
        names = [f'{index:02d}_{name}' for index, name in enumerate(names)]
    for name in names:
        for extension in ('.gif', '.png', '.mp4'):
            path = output / (name + extension)
            if path.exists() and not args.overwrite:
                raise FileExistsError(f'{path} already exists; use --overwrite to replace')
    output.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(image, output / 'input_view.png')
    for name, (_, group) in zip(names, groups):
        render(group, center, radius, args, output / name, scale)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings['inputs'] = [str(path) for path in args.inputs] if args.inputs else None
    settings.update(center=center.tolist(), radius=float(radius), geometry_names=[item['name'] for item in items])
    (output / 'orbit_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    print(f'Saved orbits to {output}')


if __name__ == '__main__':
    main()
