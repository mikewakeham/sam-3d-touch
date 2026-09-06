import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import viser


SOURCES = (
    "full_surface",
    "full_surface_unmasked",
    "touch",
    "joint",
)

COLORS = {
    "full_surface": (90, 180, 90),
    "full_surface_unmasked": (80, 150, 220),
    "touch": (230, 150, 70),
    "joint": (190, 100, 210),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/vecsetx/outputs/reconstruction"),
    )
    parser.add_argument("--sample-id")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def find_samples(output_dir):
    samples = {}
    for path in output_dir.glob("*.obj"):
        for source in sorted(SOURCES, key=len, reverse=True):
            suffix = f"_{source}.obj"
            if path.name.endswith(suffix):
                sample_id = path.name[: -len(suffix)]
                samples.setdefault(sample_id, {})[source] = path
                break
    return samples


def main():
    args = parse_args()
    samples = find_samples(args.output_dir)
    if not samples:
        raise ValueError(f"No reconstruction meshes found in {args.output_dir}")

    sample_id = args.sample_id or sorted(samples)[0]
    if sample_id not in samples:
        raise ValueError(f"Unknown sample {sample_id!r}")

    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    server.scene.set_up_direction("+y")
    displayed = samples[sample_id]
    spacing = 2.5

    for index, source in enumerate(SOURCES):
        if source not in displayed:
            continue
        mesh = trimesh.load(displayed[source], force="mesh", process=False)
        offset = np.array([index * spacing, 0.0, 0.0])
        server.scene.add_mesh_simple(
            f"/{source}/mesh",
            vertices=np.asarray(mesh.vertices) + offset,
            faces=np.asarray(mesh.faces),
            color=COLORS[source],
            flat_shading=True,
            side="double",
        )
        server.scene.add_label(
            f"/{source}/label",
            text=source,
            position=offset + np.array([0.0, 1.25, 0.0]),
        )

    center = np.array([(len(SOURCES) - 1) * spacing / 2, 0.0, 0.0])

    @server.on_client_connect
    def initialize_camera(client):
        client.camera.up_direction = (0.0, 1.0, 0.0)
        client.camera.position = center + np.array([0.0, 2.5, -7.0])
        client.camera.look_at = center

    print(f"Sample: {sample_id}", flush=True)
    print(f"Viewer: http://localhost:{server.get_port()}", flush=True)
    print(f"Forward port {server.get_port()} from this compute node in VS Code.", flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
