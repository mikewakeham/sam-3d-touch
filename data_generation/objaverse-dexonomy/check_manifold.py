import argparse
from pathlib import Path

import numpy as np
import potpourri3d as pp3d
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sample_touch import refine_mesh, split_vertex_nonmanifold


DEFAULT_DATA_ROOT = Path(
    "/n/holylabs/qianqian_lab/Lab/mwakeham/"
    "visuotactile-objects/sam-3d-touch-data/objaverse-dexonomy"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--max-edge", type=float, default=0.03)
    return parser.parse_args()


def clean_mesh(mesh, aggressive_merge):
    if aggressive_merge:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
    else:
        mesh.merge_vertices()

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)

    return mesh


def test_solvers(mesh):
    edges = mesh.edges_unique

    graph = coo_matrix(
        (np.ones(len(edges)), edges.T),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    )

    component_count, components = connected_components(
        graph,
        directed=False,
    )

    failed_components = []

    for component in range(component_count):
        face_ids = np.flatnonzero(
            components[mesh.faces[:, 0]] == component
        )

        if len(face_ids) == 0:
            continue

        vertex_ids = np.unique(mesh.faces[face_ids])
        local_faces = np.searchsorted(
            vertex_ids,
            mesh.faces[face_ids],
        )

        try:
            solver = pp3d.MeshFastMarchingDistanceSolver(
                mesh.vertices[vertex_ids],
                local_faces,
            )

            distances = solver.compute_distance(
                [[(0, [1 / 3, 1 / 3, 1 / 3])]],
                sign=False,
            )

            if not np.isfinite(distances).all():
                raise ValueError("Geodesic calculation returned non-finite distances")

        except Exception as error:
            failed_components.append(
                (
                    component,
                    len(vertex_ids),
                    len(face_ids),
                    str(error),
                )
            )

    return component_count, failed_components


def report(name, mesh):
    edge_counts = np.bincount(mesh.edges_unique_inverse)
    component_count, failed_components = test_solvers(mesh)

    print()
    print(name)
    print(f"  vertices: {len(mesh.vertices)}")
    print(f"  faces: {len(mesh.faces)}")
    print(f"  connected components: {component_count}")
    print(f"  boundary edges: {np.sum(edge_counts == 1)}")
    print(f"  edges with two faces: {np.sum(edge_counts == 2)}")
    print(f"  edges with more than two faces: {np.sum(edge_counts > 2)}")
    print(f"  watertight: {mesh.is_watertight}")
    print(f"  winding consistent: {mesh.is_winding_consistent}")
    print(f"  Potpourri3D failures: {len(failed_components)}")

    for component, vertices, faces, error in failed_components:
        print(
            f"    component {component}: "
            f"{vertices} vertices, {faces} faces: {error}"
        )


def main():
    args = parse_args()

    model_path = (
        args.data_root
        / "objects"
        / args.object_id
        / "model.obj"
    )

    raw_mesh = trimesh.load(
        str(model_path),
        force="mesh",
        process=False,
        skip_materials=True,
    )

    raw_mesh.fix_normals(multibody=True)
    report("1. Raw OBJ", raw_mesh)

    default_merged = clean_mesh(
        raw_mesh.copy(),
        aggressive_merge=False,
    )
    report("2. Default vertex merge", default_merged)

    aggressive_merged = clean_mesh(
        raw_mesh.copy(),
        aggressive_merge=True,
    )
    report("3. Current aggressive vertex merge", aggressive_merged)

    subdivided_mesh = refine_mesh(
        aggressive_merged.copy(),
        max_edge=args.max_edge,
        max_faces=250000,
    )
    subdivided_mesh.fix_normals(multibody=True)
    report("4. Current merge plus subdivision", subdivided_mesh)

    split_mesh = split_vertex_nonmanifold(
        aggressive_merged.copy()
    )
    report("5. Split vertex-nonmanifold connections", split_mesh)

    subdivided_mesh = refine_mesh(
        split_mesh.copy(),
        max_edge=args.max_edge,
        max_faces=250000,
    )
    report("6. Split plus subdivision", subdivided_mesh)


if __name__ == "__main__":
    main()