import argparse
import colorsys
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from pointmaps import depth_to_pointmap
from generate_target_latents import load_normalized_mesh
from sample_full_surface import sam_camera_transform, transform_points, transform_normals
from surface_pool import validate_surface_pool
from sample_touch_patches import select_patch_indices
from PIL import Image


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--object-id", help="Defaults to the first object with simulated touches, then full-surface data")
    parser.add_argument("--view-id", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pointmap-stride", type=int, default=2)
    parser.add_argument('--contacts', type=int, choices=[32, 16, 8], default=32)
    joint = parser.add_mutually_exclusive_group()
    joint.add_argument('--joint-input', type=Path, help='Evaluated joint run stage1.npz: show its input after encoder FPS')
    joint.add_argument('--joint-pointmap', action='store_true', help='Preview joint pointmap/touch FPS directly from this view')
    parser.add_argument('--pipeline-config', type=Path, default=Path('checkpoints/hf/pipeline.yaml'),
                        help='Training pipeline preprocessing for --joint-pointmap; model weights are not loaded')
    parser.add_argument('--input-device', default='cpu', choices=['cpu', 'cuda'])
    return parser.parse_args(argv)


def touch_file(view_dir):
    path = view_dir / 'simulated_touches.npz'
    # Keep already-generated pilot files readable under the corrected name.
    old_path = view_dir / 'touches_adaptive_v1.npz'
    return old_path if not path.is_file() and old_path.is_file() else path


def load_view(args):
    if args.object_id is None:
        generated = args.data_root / "generated_data"
        objects = json.loads((generated / "objects.json").read_text())
        views = [(obj['object_id'], generated / obj['object_id'] / 'views' / f'{args.view_id:03d}')
                 for obj in objects]
        args.object_id = next((oid for oid, view in views if touch_file(view).is_file()), None)
        if args.object_id is None:
            args.object_id = next((oid for oid, view in views if (view / 'full_surface.npz').is_file()), None)
        if args.object_id is None:
            raise ValueError('No objects with saved data for this view; generate data first or specify --object-id')
    generated_dir = args.data_root / "generated_data" / args.object_id
    view_dir = generated_dir / "views" / f"{args.view_id:03d}"
    if (args.joint_input or args.joint_pointmap) and not touch_file(view_dir).is_file():
        raise ValueError('Joint preview requires simulated touches for this object/view')

    with np.load(view_dir / "camera.npz") as data:
        K = data["K"]
        T_camera_from_object = data["T_camera_from_object"]

    mesh = load_normalized_mesh(generated_dir / "mesh.npz")
    mesh.apply_transform(np.diag([-1., -1., 1., 1.]) @ T_camera_from_object)

    rgba = np.array(Image.open(view_dir / "image.png"))
    pointmap = depth_to_pointmap(np.load(view_dir / "depth.npy", allow_pickle=False), K)

    with np.load(view_dir / "full_surface.npz") as data:
        surface = dict(data)

    pool = None
    pool_path = generated_dir / "surface_pool.npz"
    if pool_path.is_file():
        with np.load(pool_path, allow_pickle=False) as data:
            validate_surface_pool(data)
            transform = sam_camera_transform(view_dir / "camera.npz")
            pool = {"points_camera": transform_points(data["points_object"], transform).astype(np.float32),
                    "normals_camera": transform_normals(data["normals_object"], transform)}
    return mesh, rgba, pointmap, surface, K, pool


def add_toggle(server, label, handle):
    checkbox = server.gui.add_checkbox(label, initial_value=handle.visible)

    @checkbox.on_update
    def update(_):
        handle.visible = checkbox.value


def add_normals(server, name, label, points, normals, visible):
    # Thin only the display, leaving the saved cloud untouched.
    stride = max(1, int(np.ceil(len(points) / 512)))
    starts = points[::stride]
    ends = starts + .025 * normals[::stride]
    lines = server.scene.add_line_segments(
        name, points=np.stack([starts, ends], axis=1),
        colors=(240, 100, 50), line_width=1.5, visible=visible,
    )
    tips = server.scene.add_point_cloud(
        name + "_tips", points=ends, colors=(255, 230, 80),
        point_size=.004, point_shape="circle", visible=visible,
    )
    toggle = server.gui.add_checkbox(label, initial_value=visible)

    @toggle.on_update
    def update(_):
        lines.visible = tips.visible = toggle.value


def load_touch_view(args):
    view = args.data_root / 'generated_data' / args.object_id / 'views' / f'{args.view_id:03d}'
    with np.load(touch_file(view), allow_pickle=False) as saved:
        data = dict(saved)
    count = args.contacts
    per_contact = 8192 // count
    indices = select_patch_indices(data, count, per_contact)
    palette = np.array([colorsys.hsv_to_rgb((i * .61803398875) % 1, .8, 1)
                        for i in range(count)]) * 255
    colors = np.repeat(palette.astype(np.uint8), per_contact, axis=0)
    centers = data['points_camera'][data['offsets'][:count]]
    # Three principal great circles make the ellipsoid boundary visible without a solid shell.
    angles = np.linspace(0, 2 * np.pi, 65)
    lines = []
    for index in range(count):
        for first, second in [(0, 1), (0, 2), (1, 2)]:
            circle = np.zeros((len(angles), 3))
            circle[:, first] = np.cos(angles)
            circle[:, second] = np.sin(angles)
            circle = (circle * data['radii'][index]) @ data['bases_camera'][index].T + centers[index]
            lines.append(np.stack([circle[:-1], circle[1:]], axis=1))
    return data['points_camera'][indices], colors, centers, np.concatenate(lines)


def prepare_joint_preview(args, rgba, pointmap):
    """Use the training preprocessor without constructing or loading any model."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    from train import build_stage1_preprocessor, preprocess_pointmap_batch

    preprocessor = build_stage1_preprocessor(args.pipeline_config)
    with torch.no_grad():
        inputs = preprocess_pointmap_batch(
            preprocessor, torch.from_numpy(rgba.copy())[None],
            torch.from_numpy(np.ascontiguousarray(pointmap))[None], args.input_device,
        )
    return preprocessor, inputs


def load_joint_cloud(args, touch_points, prepared=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evaluation.input_visualizations import surface_indices
    if prepared is None:
        with np.load(args.joint_input, allow_pickle=False) as saved:
            if 'encoder_input_camera' not in saved:
                raise ValueError('--joint-input needs a newly evaluated joint stage1.npz')
            camera = saved['encoder_input_camera']
            pre_encoder = saved['touch_centers']
            count = int(saved['touch_count'])
    else:
        import torch
        from train import combine_pointmap_and_touch, normalize_touch_to_pointmap_frame
        from evaluation.input_visualizations import joint_camera_points

        preprocessor, inputs = prepared
        count = len(touch_points)
        with torch.no_grad():
            touch = torch.as_tensor(touch_points, device=args.input_device)[None]
            mask = torch.ones(touch.shape[:2], dtype=torch.bool, device=touch.device)
            touch = normalize_touch_to_pointmap_frame(touch, mask, inputs, preprocessor)
            cloud, valid = combine_pointmap_and_touch(inputs, touch, mask)
            cloud = cloud[0, valid[0]]
            camera = joint_camera_points(cloud, inputs, preprocessor)
            pre_encoder = cloud.float().cpu().numpy()
    if count != len(touch_points) or not np.allclose(camera[-count:], touch_points, atol=1e-5, rtol=1e-5):
        raise ValueError('Joint input does not match this object/view/contact selection')
    print(f'Computing joint FPS for {args.contacts} patches: {len(camera):,} candidates -> 8192 points '
          f'({args.input_device})', flush=True)
    indices = surface_indices(pre_encoder, 'vecsetx', args.input_device)
    pointmap_count = len(camera) - count
    return camera, indices, pointmap_count


def joint_counts(args, indices, pointmap_count):
    patch_ids = np.full(len(indices), -1, dtype=np.int64)
    touch = indices >= pointmap_count
    patch_ids[touch] = (indices[touch] - pointmap_count) // (8192 // args.contacts)
    retained = np.bincount(patch_ids[touch], minlength=args.contacts)
    print(f'Joint FPS: {touch.sum()} touch + {(~touch).sum()} pointmap = {len(indices)}', flush=True)
    print(f'Touch points per patch after FPS: {retained.tolist()}', flush=True)
    return (f'**Before FPS:** {pointmap_count:,} pointmap + 8,192 touch points.\n\n'
            f'**After FPS:** {(~touch).sum():,} pointmap + {touch.sum():,} touch = {len(indices):,}.\n\n'
            f'**Touch points retained per patch (FPS order):** {retained.tolist()}')


def load_joint_view(args, touch_points):
    camera, indices, pointmap_count = load_joint_cloud(args, touch_points)
    joint_counts(args, indices, pointmap_count)
    return camera[indices], indices - pointmap_count


def build_viewer(server, args, mesh, rgba, pointmap, surface, K, pool=None):
    view = args.data_root / 'generated_data' / args.object_id / 'views' / f'{args.view_id:03d}'
    has_touches = touch_file(view).is_file()
    joint_input = getattr(args, 'joint_input', None)
    joint_preview = getattr(args, 'joint_pointmap', False)
    show_joint = bool(joint_input or joint_preview)
    server.scene.set_up_direction("+y")
    server.gui.add_image(rgba, label=f"{args.object_id} / view {args.view_id:03d}")

    mesh_handle = server.scene.add_mesh_simple(
        "/mesh", vertices=mesh.vertices, faces=mesh.faces,
        color=(180, 180, 180), flat_shading=True, side="double",
    )
    add_toggle(server, "Mesh", mesh_handle)

    # Display-only thinning of the pointmap, keeping RGB aligned.
    stride = args.pointmap_stride
    points = pointmap[::stride, ::stride]
    pixels = rgba[::stride, ::stride]
    valid = np.isfinite(points).all(axis=-1) & (pixels[..., 3] > 0)
    pointmap_handle = server.scene.add_point_cloud(
        "/pointmap", points=points[valid], colors=pixels[..., :3][valid],
        point_size=0.003, point_shape="circle",
        visible=pool is None and not has_touches,
    )
    add_toggle(server, "Pointmap", pointmap_handle)

    height, width = rgba.shape[:2]
    fov = float(2 * np.arctan(height / (2 * K[1, 1])))
    # Viser frustums use OpenCV axes: rotate X/Y into our SAM scene.
    camera_handle = server.scene.add_camera_frustum(
        "/render_camera", fov=fov, aspect=width / height,
        wxyz=(0., 0., 0., 1.), scale=0.15, color=(255, 180, 50),
    )
    add_toggle(server, "Render camera", camera_handle)
    points = surface["points_camera"]
    labels = surface["point_visibility"]
    colors = np.full((len(points), 3), (150, 150, 150), dtype=np.uint8)
    colors[labels == 0] = (45, 125, 210)
    colors[labels == 1] = (60, 200, 90)
    handle = server.scene.add_point_cloud(
        "/full_surface", points=points, colors=colors, point_size=0.003,
        point_shape="circle",
        visible=pool is None and not has_touches,
    )
    add_toggle(server, "Full surface (green visible / blue hidden)", handle)
    if has_touches:
        budgets = {f'{count} x {8192 // count} = 8192': count for count in (32, 16, 8)}
        selection = server.gui.add_dropdown(
            'Touch patch budget', options=list(budgets),
            initial_value=f'{args.contacts} x {8192 // args.contacts} = 8192',
            disabled=bool(joint_input),
        )
        points, colors, centers, boundaries = load_touch_view(args)
        touch_handle = server.scene.add_point_cloud('/touch', points=points, colors=colors,
                                                    point_size=.003, point_shape='circle', visible=not show_joint)
        add_toggle(server, 'Simulated touches', touch_handle)
        centers_handle = server.scene.add_point_cloud('/touch_centers', points=centers,
                                                      colors=(255, 255, 255), point_size=.008,
                                                      point_shape='circle', visible=not show_joint)
        add_toggle(server, 'Patch centers (FPS order)', centers_handle)
        regions = server.scene.add_line_segments('/touch_regions', points=boundaries,
                                                 colors=(255, 220, 80), line_width=1., visible=False)
        add_toggle(server, 'Patch ellipsoid boundaries', regions)

        if show_joint:
            prepared = prepare_joint_preview(args, rgba, pointmap) if joint_preview else None
            cache = {}
            counts = server.gui.add_markdown('Computing joint FPS…')

            def joint_data(points, colors):
                if args.contacts not in cache:
                    cache[args.contacts] = load_joint_cloud(args, points, prepared)
                camera, indices, pointmap_count = cache[args.contacts]
                joint_colors = np.concatenate([
                    np.tile(np.array([60, 200, 90], np.uint8), (pointmap_count, 1)), colors,
                ])
                counts.content = joint_counts(args, indices, pointmap_count)
                return camera, joint_colors, indices

            camera, joint_colors, indices = joint_data(points, colors)
            before_handle = server.scene.add_point_cloud(
                '/joint_before', points=camera, colors=joint_colors, point_size=.003,
                point_shape='circle', visible=False,
            )
            add_toggle(server, 'Joint before FPS (green pointmap / colored touch)', before_handle)
            joint_handle = server.scene.add_point_cloud(
                '/joint_fps', points=camera[indices], colors=joint_colors[indices],
                point_size=.003, point_shape='circle',
            )
            add_toggle(server, 'Joint after FPS (8192 points)', joint_handle)

        budget_lock = threading.Lock()

        @selection.on_update
        def update_touch_budget(_):
            with budget_lock:
                previous = args.contacts
                args.contacts = budgets[selection.value]
                selection.disabled = True
                try:
                    points, colors, centers, boundaries = load_touch_view(args)
                    if show_joint:
                        counts.content = 'Computing joint FPS…'
                        camera, joint_colors, indices = joint_data(points, colors)
                    with server.atomic():
                        touch_handle.points = points
                        touch_handle.colors = colors
                        centers_handle.points = centers
                        regions.points = boundaries
                        if show_joint:
                            before_handle.points = camera
                            before_handle.colors = joint_colors
                            joint_handle.points = camera[indices]
                            joint_handle.colors = joint_colors[indices]
                except Exception as error:
                    args.contacts = previous
                    selection.value = f'{previous} x {8192 // previous} = 8192'
                    if show_joint:
                        counts.content = f'Joint preview failed: {error}'
                    raise
                finally:
                    selection.disabled = bool(joint_input)
    if "normals_camera" in surface:
        add_normals(server, "/surface_normals", "Original cloud normals", surface['points_camera'],
                    surface["normals_camera"], visible=False)
    if pool is not None:
        pool_points, normals = pool["points_camera"], pool["normals_camera"]
        pool_handle = server.scene.add_point_cloud(
            "/surface_pool", points=pool_points,
            colors=np.clip((normals + 1) * 127.5, 0, 255).astype(np.uint8),
            point_size=.003, point_shape="circle", visible=not has_touches,
        )
        add_toggle(server, f"Shared cloud ({len(pool_points):,} points; normal colors)", pool_handle)
        add_normals(server, "/pool_normals", "Shared cloud normals (yellow tips)",
                    pool_points, normals, visible=not has_touches)

    target = mesh.bounds.mean(axis=0)

    @server.on_client_connect
    def initialize_camera(client):
        client.camera.up_direction = (0., 1., 0.)
        client.camera.position = target + np.array([1.2, 0.8, -1.5])
        client.camera.look_at = target

    camera_button = server.gui.add_button("Look from render camera")

    @camera_button.on_click
    def look_from_camera(event):
        event.client.camera.up_direction = (0., 1., 0.)
        event.client.camera.position = (0., 0., 0.)
        event.client.camera.look_at = (0., 0., 1.)
        event.client.camera.fov = fov


def main():
    import viser

    args = parse_args()
    if args.pointmap_stride < 1:
        raise ValueError("--pointmap-stride must be at least 1")

    data = load_view(args)
    print(f"Object: {args.object_id}; view: {args.view_id:03d}", flush=True)
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    build_viewer(server, args, *data)

    print(f"Viewer: http://localhost:{server.get_port()}", flush=True)
    print(f"Forward port {server.get_port()} from this compute node in VS Code.", flush=True)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
