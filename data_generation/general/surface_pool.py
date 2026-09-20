"""A shared area-weighted XYZ/normal pool; encoder-specific FPS stays in the encoder."""

import numpy as np

try:
    from .sample_full_surface import NORMAL_METHOD
except ImportError:
    from sample_full_surface import NORMAL_METHOD


# CraftsMan v1.5's released encoder input count.
# https://huggingface.co/craftsman3d/craftsman/blob/df4ddf7544cc2e75c5d24cb8605d8e91f0fa4abc/config.yaml
DEFAULT_POOL_POINTS = 16384


def make_surface_pool(mesh, count, sample_seed, mesh_hash):
    import trimesh
    if count < 1:
        raise ValueError("Surface pool count must be positive")
    # Sample at the requested count; leave the original per-view samples intact.
    pool_seed = sample_seed
    points, faces = trimesh.sample.sample_surface(mesh, count, seed=pool_seed)
    arrays = {
        "points_object": points.astype(np.float32),
        "normals_object": mesh.face_normals[faces].astype(np.float32),
        "face_indices": faces.astype(np.int64),
        "format_version": np.int64(1),
        "data_kind": "surface_pool",
        "coordinate_frame": "normalized_object",
        "sampling_method": "area_weighted_direct",
        "normal_method": NORMAL_METHOD,
        "sample_seed": np.int64(pool_seed),
        "source_sample_seed": np.int64(sample_seed),
        "requested_point_count": np.int64(count),
        "mesh_sha256": mesh_hash,
        "numpy_version": np.__version__,
        "trimesh_version": str(trimesh.__version__ or "unknown"),
    }
    validate_surface_pool(arrays)
    return arrays


def validate_surface_pool(arrays, mesh=None, mesh_hash=None):
    if (int(arrays["format_version"]) != 1 or str(arrays["data_kind"]) != "surface_pool"
            or str(arrays["coordinate_frame"]) != "normalized_object"
            or str(arrays["normal_method"]) != NORMAL_METHOD
            or str(arrays["sampling_method"]) != "area_weighted_direct"):
        raise ValueError("Unsupported surface pool format")
    points, normals, faces = (arrays[key] for key in ("points_object", "normals_object", "face_indices"))
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 1
            or len(points) != int(arrays["requested_point_count"])
            or points.dtype != np.float32 or not np.isfinite(points).all()
            or np.abs(points).max() > .50001):
        raise ValueError("Invalid normalized-object pool points")
    if (normals.shape != points.shape or normals.dtype != np.float32
            or not np.isfinite(normals).all()
            or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid surface pool normals")
    if faces.shape != (len(points),) or faces.dtype != np.int64 or np.any(faces < 0):
        raise ValueError("Invalid surface pool triangle indices")
    if mesh_hash is not None and str(arrays["mesh_sha256"]) != mesh_hash:
        raise ValueError("Surface pool belongs to a different mesh")
    if mesh is not None:
        import trimesh
        replayed, replayed_faces = trimesh.sample.sample_surface(mesh, len(points), seed=int(arrays["sample_seed"]))
        if (not np.array_equal(faces, replayed_faces)
                or not np.allclose(points, replayed.astype(np.float32), atol=1e-6, rtol=0)
                or not np.allclose(normals, mesh.face_normals[replayed_faces], atol=1e-6, rtol=0)):
            raise ValueError("Surface pool sampling/normal replay differs; check mesh and library versions")


def save_surface_pool(path, arrays):
    validate_surface_pool(arrays)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def select_surface_pool(arrays, count, seed):
    """Uniform random subset, as in VecSetX/CraftsMan/TripoSG dataset loaders."""
    validate_surface_pool(arrays)
    if not 0 < count <= len(arrays["points_object"]):
        raise ValueError(f"Requested {count} points from a pool of {len(arrays['points_object'])}")
    indices = np.random.default_rng(seed).choice(len(arrays["points_object"]), count, replace=False)
    return tuple(np.ascontiguousarray(arrays[key][indices])
                 for key in ("points_object", "normals_object", "face_indices"))
