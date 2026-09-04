import argparse
import colorsys
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--view-id", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pointmap-stride", type=int, default=2)
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
    T_sam_from_source = np.diag([-1., -1., 1., 1.]) @ T_camera_from_object @ T_object_from_source
    mesh = trimesh.load(str(object_dir / "model.obj"), force="mesh", process=False, skip_materials=True)
    mesh.apply_transform(T_sam_from_source)

    rgba = np.array(Image.open(view_dir / "image.png"))
    pointmap = np.load(view_dir / "pointmap.npy")

    with np.load(view_dir / "touches.npz") as data:
        touch = dict(data)

    return mesh, rgba, pointmap, touch, K


def add_toggle(server, label, handle):
    checkbox = server.gui.add_checkbox(label, initial_value=handle.visible)

    @checkbox.on_update
    def update(_):
        handle.visible = checkbox.value


def add_touches(server, touch):
    contacts = []
    count = len(touch["centers_camera"])

    for i in range(count):
        a, b = touch["offsets"][i:i + 2]
        R = touch["R_camera_from_local"][i]
        center = touch["centers_camera"][i]
        points = touch["points_local"][a:b] @ R.T + center
        normals = touch["normals_local"][a:b] @ R.T
        color = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i * 0.618 % 1, 0.75, 1))

        cloud = server.scene.add_point_cloud(
            f"/touch/{i}/points", points=points, colors=color,
            point_size=0.003, point_shape="circle", precision="float32",
        )
        marker = server.scene.add_point_cloud(
            f"/touch/{i}/center", points=center[None], colors=color,
            point_size=0.012, point_shape="circle", precision="float32",
        )
        label = server.scene.add_label(f"/touch/{i}/label", text=str(i), position=center)

        # Display every 50th normal as a short line; saved data is unchanged.
        starts = points[::50]
        ends = starts + 0.025 * normals[::50]
        lines = server.scene.add_line_segments(
            f"/touch/{i}/normals", points=np.stack([starts, ends], axis=1),
            colors=color, thickness=1.5, thickness_units="screen", visible=False,
        )
        contacts.append((cloud, marker, label, lines))

    selection = server.gui.add_dropdown("Neighborhood", options=["All"] + [str(i) for i in range(count)])
    show_points = server.gui.add_checkbox("Touch points", initial_value=True)
    show_centers = server.gui.add_checkbox("Touch centers", initial_value=True)
    show_normals = server.gui.add_checkbox("Touch normals", initial_value=False)

    def update(_):
        for i, (cloud, marker, label, lines) in enumerate(contacts):
            selected = selection.value == "All" or selection.value == str(i)
            cloud.visible = selected and show_points.value
            marker.visible = selected and show_centers.value
            label.visible = selected and show_centers.value
            lines.visible = selected and show_normals.value

    for control in [selection, show_points, show_centers, show_normals]:
        control.on_update(update)


def build_viewer(server, args, mesh, rgba, pointmap, touch, K):
    server.scene.set_up_direction("+y")
    server.gui.add_image(rgba, label=f"View {args.view_id:03d}")

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
        point_size=0.003, point_shape="circle", precision="float32",
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
    add_touches(server, touch)

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
    args = parse_args()
    if args.pointmap_stride < 1:
        raise ValueError("--pointmap-stride must be at least 1")

    data = load_view(args)
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    build_viewer(server, args, *data)

    print(f"Viewer: http://localhost:{server.get_port()}", flush=True)
    print(f"Forward port {server.get_port()} from this compute node in VS Code.", flush=True)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()