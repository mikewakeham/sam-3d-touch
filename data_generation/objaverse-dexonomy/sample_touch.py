import argparse
import json
from pathlib import Path

import numpy as np
import potpourri3d as pp3d
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


DEFAULT_DATA_ROOT = Path(
    "/n/holylabs/qianqian_lab/Lab/mwakeham/"
    "visuotactile-objects/sam-3d-touch-data/objaverse-dexonomy"
)


def split_vertex_nonmanifold(mesh, max_added_components=32):
    original_faces = len(mesh.faces)

    edges = mesh.edges_unique
    graph = coo_matrix(
        (np.ones(len(edges)), edges.T),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    )
    before = connected_components(
        graph,
        directed=False,
        return_labels=False,
    )

    pieces = mesh.split(
        only_watertight=False,
        repair=False,
    )

    after = len(pieces)
    added = after - before

    if (
        max_added_components is not None
        and added > max_added_components
    ):
        raise ValueError(
            f"Mesh fragments from {before} to {after} "
            f"touch components (+{added})"
        )

    mesh = trimesh.util.concatenate(pieces)

    if len(mesh.faces) != original_faces:
        raise ValueError(
            "Face count changed while splitting mesh components"
        )

    return mesh


def load_mesh(model_path, object_transform_path, max_added_components=32):
    mesh = trimesh.load(str(model_path), force="mesh", process=False, skip_materials=True)

    with np.load(object_transform_path) as data:
        transform = data["T_normalized_from_source"]

    mesh.apply_transform(transform)

    # Join duplicated geometry across UV and normal seams.
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    # Undo nonmanifold vertex connections accidentally present in the mesh
    # or created by aggressive vertex merging.
    mesh = split_vertex_nonmanifold(mesh, max_added_components=max_added_components)

    mesh.fix_normals(multibody=True)

    if not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("Mesh must have finite vertices and positive surface area")

    return mesh


def refine_mesh(mesh, max_edge, max_faces=250000):
    """Uniform subdivision that preserves mesh connectivity."""
    if not np.isfinite(max_edge) or max_edge <= 0:
        raise ValueError("max_edge must be finite and positive")

    original_faces = len(mesh.faces)

    while mesh.edges_unique_length.max() > max_edge:
        if 4 * len(mesh.faces) > max_faces:
            break

        mesh = mesh.subdivide()

    mesh.fix_normals(multibody=True)

    actual_max_edge = mesh.edges_unique_length.max()

    print(
        f"Touch mesh: {original_faces} -> {len(mesh.faces)} faces, "
        f"max edge {actual_max_edge:.4f}"
    )

    return mesh


def build_surface(mesh, density, seed_parts):
    """One area-weighted master pool, independent of view and contact count."""
    if not np.isfinite(density) or density <= 0:
        raise ValueError("density must be finite and positive")

    rng = np.random.default_rng(seed_parts)
    count = int(np.ceil(mesh.area * density))
    points, face_ids = trimesh.sample.sample_surface(mesh, count, seed=int(rng.integers(2**31)))
    barycentric = trimesh.triangles.points_to_barycentric(mesh.triangles[face_ids], points)

    edges = mesh.edges_unique
    graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(len(mesh.vertices), len(mesh.vertices)))
    _, components = connected_components(graph, directed=False)
    point_components = components[mesh.faces[face_ids, 0]]

    return {
        "mesh": mesh,
        "points": points,
        "face_ids": face_ids,
        "barycentric": barycentric,
        "keep_priority": rng.random(count),
        "components": components,
        "point_components": point_components,
        "solvers": {},
        "density": density,
        "seed_parts": np.asarray(seed_parts, dtype=np.int64),
    }


def prepare_surface(
    model_path, object_transform_path, density, seed=29, max_edge=0.03,
    max_added_components=32,
):
    model_path = Path(model_path)
    mesh = load_mesh(model_path, object_transform_path, max_added_components=max_added_components)
    mesh = refine_mesh(mesh, max_edge)
    seed_parts = [seed, int(model_path.parent.name[:8], 16)]
    surface = build_surface(mesh, density, seed_parts)
    surface["max_edge"] = max_edge
    return surface


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def classify_visibility(points, K, camera_transform, depth, tolerance):
    """-1: unknown/out of frame; 0: behind rendered depth; 1: visible."""
    camera_points = transform_points(points, camera_transform)
    height, width = depth.shape
    labels = np.full(len(points), -1, dtype=np.int8)
    indices = np.flatnonzero(np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0))
    projected = camera_points[indices] @ K.T
    pixels = np.rint(projected[:, :2] / projected[:, 2:3])
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    indices = indices[inside]
    u, v = pixels[inside].astype(np.int64).T
    observed = depth[v, u]
    valid = np.isfinite(observed) & (observed > 0)
    indices = indices[valid]
    difference = camera_points[indices, 2] - observed[valid]
    labels[indices[np.abs(difference) <= tolerance]] = 1
    labels[indices[difference > tolerance]] = 0
    return labels


def geodesic_distances(surface, source_point):
    """Distances from one master-pool point to every master-pool point."""
    mesh = surface["mesh"]
    source_face = int(surface["face_ids"][source_point])
    component = int(surface["point_components"][source_point])

    if component not in surface["solvers"]:
        face_ids = np.flatnonzero(surface["components"][mesh.faces[:, 0]] == component)
        vertex_ids = np.unique(mesh.faces[face_ids])
        local_faces = np.searchsorted(vertex_ids, mesh.faces[face_ids])
        point_ids = np.flatnonzero(surface["point_components"] == component)
        point_vertices = np.searchsorted(vertex_ids, mesh.faces[surface["face_ids"][point_ids]])
        solver = pp3d.MeshFastMarchingDistanceSolver(mesh.vertices[vertex_ids], local_faces)
        surface["solvers"][component] = (face_ids, point_ids, point_vertices, solver)

    face_ids, point_ids, point_vertices, solver = surface["solvers"][component]
    local_face = int(np.searchsorted(face_ids, source_face))
    barycentric = surface["barycentric"][source_point].tolist()
    vertex_distances = solver.compute_distance([[(local_face, barycentric)]], sign=False)
    if not np.isfinite(vertex_distances).all() or np.any(vertex_distances < 0):
        raise ValueError(f"Invalid geodesic distances in mesh component {component}")

    distances = np.full(len(surface["points"]), np.inf)
    distances[point_ids] = np.sum(
        vertex_distances[point_vertices] * surface["barycentric"][point_ids], axis=1
    )

    # Within one flat triangle, the straight segment is the exact surface path.
    same_face = np.flatnonzero(surface["face_ids"] == source_face)
    distances[same_face] = np.linalg.norm(
        surface["points"][same_face] - surface["points"][source_point], axis=1
    )
    distances[source_point] = 0
    return distances


def sample_neighborhood(distances, eligible_points, radius):
    point_ids = np.flatnonzero(eligible_points & (distances <= radius))
    return point_ids, distances[point_ids]


def geodesic_ball(
    surface, K, camera_transform, depth, rng,
    num_contacts=32, radius=0.10, center_method="geodesic_farthest", 
    tolerance=0.005, neighborhood_mode="all",
):
    if num_contacts < 1 or not np.isfinite(radius) or radius <= 0 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("num_contacts and radius must be positive; tolerance must be nonnegative")
    if center_method not in ("random", "geodesic_farthest"):
        raise ValueError(f"Unknown center_method: {center_method}")
    if neighborhood_mode not in ("all", "hidden"):
        raise ValueError(f"Unknown neighborhood_mode: {neighborhood_mode}")

    mesh = surface["mesh"]
    labels = classify_visibility(surface["points"], K, camera_transform, depth, tolerance)
    if neighborhood_mode == "all":
        eligible_points = np.ones(len(surface["points"]), dtype=bool)
    else:
        eligible_points = labels == 0
    candidates = np.flatnonzero(labels == 0)
    if len(candidates) < num_contacts:
        raise ValueError(f"Only {len(candidates)} hidden surface samples for {num_contacts} contacts")

    # The pool is already area-weighted, so hidden point counts approximate
    # each component's hidden surface area.
    candidates = rng.permutation(candidates)
    candidate_components = surface["point_components"][candidates]
    component_ids, component_counts = np.unique(candidate_components, return_counts=True)

    nearest = np.full(len(candidates), np.inf)
    selected = []
    neighborhoods = []
    neighborhood_distances = []

    for index in range(num_contacts):
        available = nearest != -np.inf

        # Choose among all components that still have unused candidates.
        component_available = np.array([
            np.any(available & (candidate_components == component))
            for component in component_ids
        ])

        weights = component_counts.astype(np.float64)
        weights[~component_available] = 0
        weights /= weights.sum()

        component = rng.choice(component_ids, p=weights)
        positions = np.flatnonzero(
            available & (candidate_components == component)
        )

        if center_method == "random":
            position = int(positions[0])
        else:
            position = int(positions[np.argmax(nearest[positions])])

        point = int(candidates[position])
        distances = geodesic_distances(surface, point)
        point_ids, point_distances = sample_neighborhood(distances, eligible_points, radius)

        selected.append(point)
        neighborhoods.append(point_ids)
        neighborhood_distances.append(point_distances)
        nearest = np.minimum(nearest, distances[candidates])
        nearest[position] = -np.inf

    selected = np.asarray(selected, dtype=np.int64)
    center_faces = surface["face_ids"][selected]
    return {
        "points": [surface["points"][ids] for ids in neighborhoods],
        "point_normals": [mesh.face_normals[surface["face_ids"][ids]] for ids in neighborhoods],
        "centers": surface["points"][selected],
        "center_normals": mesh.face_normals[center_faces],
        "center_point_ids": selected,
        "center_face_ids": center_faces,
        "center_barycentric": surface["barycentric"][selected],
        "visibility": labels[selected],
        "radius": np.full(num_contacts, radius, dtype=np.float32),
        "point_ids": neighborhoods,
        "point_visibility": [labels[ids] for ids in neighborhoods],
        "keep_priority": [surface["keep_priority"][ids] for ids in neighborhoods],
        "geodesic_distance": neighborhood_distances,
        
        "parameters": {
            "num_contacts": num_contacts,
            "radius": radius,
            "center_method": center_method,
            "tolerance": tolerance,
            "neighborhood_mode": neighborhood_mode,
        },
    }


SAMPLERS = {"geodesic_ball": geodesic_ball}


def contact_frame(normal):
    z = normal / np.linalg.norm(normal)
    helper = np.eye(3)[np.argmin(np.abs(z))]
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def pack_touch(data, camera_transform):
    # OpenCV camera axes -> SAM camera axes, matching pointmap.npy.
    T_sam_from_object = np.diag([-1.0, -1.0, 1.0, 1.0]) @ camera_transform
    R_sam_from_object = T_sam_from_object[:3, :3]
    points_local = []
    normals_local = []
    camera_frames = []

    for points, normals, center, normal in zip(
        data["points"], data["point_normals"], data["centers"], data["center_normals"]
    ):
        R_object_from_local = contact_frame(normal)
        points_local.append((points - center) @ R_object_from_local)
        normals_local.append(normals @ R_object_from_local)
        camera_frames.append(R_sam_from_object @ R_object_from_local)

    counts = [len(points) for points in points_local]
    return {
        "points_local": np.concatenate(points_local).astype(np.float32),
        "normals_local": np.concatenate(normals_local).astype(np.float32),
        "offsets": np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
        "centers_camera": transform_points(data["centers"], T_sam_from_object).astype(np.float32),
        "normals_camera": (data["center_normals"] @ R_sam_from_object.T).astype(np.float32),
        "R_camera_from_local": np.stack(camera_frames).astype(np.float32),
        "center_point_ids": data["center_point_ids"],
        "center_face_ids": data["center_face_ids"],
        "center_barycentric": data["center_barycentric"],
        "visibility": data["visibility"],
        "patch_radius": data["radius"],
        "point_ids": np.concatenate(data["point_ids"]).astype(np.int64),
        "point_visibility": np.concatenate(data["point_visibility"]).astype(np.int8),
        "keep_priority": np.concatenate(data["keep_priority"]),
        "geodesic_distance": np.concatenate(data["geodesic_distance"]).astype(np.float32),
    }


def sample_touch(
    model_path, object_transform_path, camera_path, depth_path, output_path,
    density, method="geodesic_ball", seed=29, surface=None, max_edge=0.03, **method_args,
):
    """Pass a prepared surface to share one pool and solver setup across views."""
    model_path = Path(model_path)
    camera_path = Path(camera_path)
    output_path = Path(output_path)
    if surface is None:
        surface = prepare_surface(model_path, object_transform_path, density, seed, max_edge)
    expected_seed = [seed, int(model_path.parent.name[:8], 16)]
    if surface["density"] != density or not np.array_equal(surface["seed_parts"], expected_seed):
        raise ValueError("Prepared surface must use the requested density and object seed")
    if surface["max_edge"] != max_edge:
        raise ValueError("Prepared surface must use the requested max_edge")

    seed_parts = [*expected_seed, int(camera_path.parent.name)]
    rng = np.random.default_rng(seed_parts)
    with np.load(camera_path) as camera:
        K = camera["K"]
        camera_transform = camera["T_camera_from_object"]
    depth = np.load(depth_path)

    data = SAMPLERS[method](surface, K, camera_transform, depth, rng, **method_args)
    arrays = pack_touch(data, camera_transform)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, **arrays, format_version=4, method=method,
        max_edge=max_edge, distance_method="fast_marching",
        method_args=json.dumps(data["parameters"]), density=density,
        surface_point_count=len(surface["points"]), surface_area=surface["mesh"].area,
        surface_seed_parts=surface["seed_parts"], seed_parts=np.asarray(seed_parts, dtype=np.int64),
    )
    # print(f"Saved {output_path}: points per contact {np.diff(arrays['offsets']).tolist()}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--view-id", default="000")
    parser.add_argument("--method", choices=SAMPLERS, default="geodesic_ball")
    parser.add_argument("--method-args", type=json.loads, default={})
    parser.add_argument("--density", type=float, default=200000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output-name", default="touches.npz")
    parser.add_argument("--max-edge", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    object_dir = args.data_root / "objects" / args.object_id
    generated_dir = args.data_root / "generated_data" / args.object_id
    view_dir = generated_dir / "views" / args.view_id
    sample_touch(
        model_path=object_dir / "model.obj",
        object_transform_path=generated_dir / "object_transform.npz",
        camera_path=view_dir / "camera.npz",
        depth_path=view_dir / "depth.npy",
        output_path=view_dir / args.output_name,
        method=args.method, seed=args.seed, density=args.density, max_edge=args.max_edge, **args.method_args,
    )


if __name__ == "__main__":
    main()
