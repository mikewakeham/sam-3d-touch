"""Explore geometry or a saved SAM3D comparison in Viser."""
import argparse
import time

import numpy as np
from PIL import Image

from evaluation.geometry import add_input_arguments, load_inputs, camera_fit, scene_bounds
from evaluation.make_orbit import orbit_eye


def add_toggle(server, label, handle):
    # From data_generation/general/view_data.py.
    checkbox = server.gui.add_checkbox(label, initial_value=handle.visible)

    @checkbox.on_update
    def update(_):
        handle.visible = checkbox.value


def add_normals(server, name, points, normals, length):
    stride = max(1, int(np.ceil(len(points) / 512)))
    starts = points[::stride]
    ends = starts + length * normals[::stride]
    handle = server.scene.add_line_segments(
        name, points=np.stack([starts, ends], axis=1),
        colors=(240, 100, 50), line_width=1.5, visible=False,
    )
    add_toggle(server, 'Normals (display subset)', handle)


def build_viewer(server, args, items, images):
    server.scene.set_up_direction('+' + args.up)
    for path in images:
        server.gui.add_image(np.array(Image.open(path)), label=path.name)
    center, radius = camera_fit(items)
    scale = max(float(np.max(np.ptp(scene_bounds(items), axis=0))), 1e-3)
    for index, item in enumerate(items):
        name = f'/geometry_{index}'
        with server.gui.add_folder(item['name']):
            server.gui.add_text('Coordinate frame', initial_value=item.get('frame', 'unspecified'), disabled=True)
            if 'mesh' in item:
                if args.textured:
                    handle = server.scene.add_mesh_trimesh(name, item['mesh'])
                else:
                    handle = server.scene.add_mesh_simple(
                        name, vertices=item['mesh'].vertices, faces=item['mesh'].faces,
                        color=(180, 180, 180), flat_shading=True, side='double',
                    )
            else:
                colors = item.get('colors', (60, 140, 215))
                if args.normal_colors and 'normals' in item:
                    colors = np.clip((item['normals'] + 1) * 127.5, 0, 255).astype(np.uint8)
                handle = server.scene.add_point_cloud(
                    name, points=item['points'], colors=colors,
                    point_size=args.point_size * scale, point_shape='circle',
                )
                if 'normals' in item:
                    add_normals(server, name + '_normals', item['points'], item['normals'], .025 * scale)
            add_toggle(server, 'Visible', handle)

    def reset_camera(client):
        client.camera.up_direction = (0., 0., 1.) if args.up == 'z' else (0., 1., 0.)
        client.camera.position = orbit_eye(center, radius, .2 * radius, .4, args.up)
        client.camera.look_at = center
        client.camera.fov = np.radians(40.)

    @server.on_client_connect
    def initialize_camera(client):
        reset_camera(client)

    button = server.gui.add_button('Reset camera')

    @button.on_click
    def reset(event):
        if event.client is not None:
            reset_camera(event.client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--point-size', type=float, default=.003, help='Fraction of the scene extent')
    parser.add_argument('--textured', action='store_true')
    parser.add_argument('--normal-colors', action='store_true')
    args = parser.parse_args()
    if args.point_size <= 0:
        raise ValueError('--point-size must be positive')
    items, images = load_inputs(args)
    import viser

    server = viser.ViserServer(host=args.host, port=args.port)
    build_viewer(server, args, items, images)
    print(f'Viewer: http://localhost:{server.get_port()}', flush=True)
    print('On the cluster, forward this port from the compute node.', flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == '__main__':
    main()
