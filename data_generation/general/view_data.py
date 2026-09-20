import argparse
import json
import time
from pathlib import Path

import numpy as np
from pointmaps import depth_to_pointmap
from generate_target_latents import load_normalized_mesh
from sample_full_surface import sam_camera_transform, transform_points, transform_normals
from surface_pool import validate_surface_pool
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--object-id", help="Defaults to the first object with a saved surface pool")
    parser.add_argument("--view-id", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pointmap-stride", type=int, default=2)
    return parser.parse_args()


def load_view(args):
    if args.object_id is None:
        generated = args.data_root / "generated_data"
        objects = json.loads((generated / "objects.json").read_text())
        args.object_id = next((obj["object_id"] for obj in objects
                               if (generated / obj["object_id"] / "surface_pool.npz").is_file()), None)
        if args.object_id is None:
            raise ValueError("No surface pools found. Run backfill_normals.py on a few objects first.")
    generated_dir = args.data_root / "generated_data" / args.object_id
    view_dir = generated_dir / "views" / f"{args.view_id:03d}"

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


def build_viewer(server, args, mesh, rgba, pointmap, surface, K, pool=None):
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
        visible=pool is None,
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
        visible=pool is None,
    )
    add_toggle(server, "Full surface (green visible / blue hidden)", handle)
    if "normals_camera" in surface:
        add_normals(server, "/surface_normals", "Original cloud normals", points,
                    surface["normals_camera"], visible=False)
    if pool is not None:
        pool_points, normals = pool["points_camera"], pool["normals_camera"]
        pool_handle = server.scene.add_point_cloud(
            "/surface_pool", points=pool_points,
            colors=np.clip((normals + 1) * 127.5, 0, 255).astype(np.uint8),
            point_size=.003, point_shape="circle",
        )
        add_toggle(server, f"Shared cloud ({len(pool_points):,} points; normal colors)", pool_handle)
        add_normals(server, "/pool_normals", "Shared cloud normals (yellow tips)",
                    pool_points, normals, visible=True)

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
