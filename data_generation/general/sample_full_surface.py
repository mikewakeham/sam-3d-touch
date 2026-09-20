"""Area-weighted full surfaces in the same SAM camera frame as the pointmap."""


import numpy as np


NORMAL_METHOD = "triangle_face_winding"


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_normals(normals, transform):
    # Inverse transpose for column vectors; translation never affects normals.
    transformed = np.asarray(normals, dtype=np.float64) @ np.linalg.inv(transform[:3, :3])
    lengths = np.linalg.norm(transformed, axis=-1, keepdims=True)
    if not np.isfinite(transformed).all() or np.any(lengths <= 0):
        raise ValueError("Normals must be finite and nonzero")
    return (transformed / lengths).astype(np.float32)


def sam_camera_transform(camera_path):
    with np.load(camera_path, allow_pickle=False) as camera:
        transform = np.diag([-1., -1., 1., 1.]) @ camera["T_camera_from_object"]
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError(f"Expected a rigid camera transform: {camera_path}")
    return transform


def validate_surface(data, require_normals=False):
    version = int(data["format_version"])
    if (version not in (1, 2) or str(data["data_kind"]) != "full_surface"
            or str(data["coordinate_frame"]) != "sam_camera"):
        raise ValueError("Unsupported full-surface format or coordinate frame")
    points = data["points_camera"]
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
            or points.dtype != np.float32 or not np.isfinite(points).all()):
        raise ValueError("Expected finite float32 surface points [N,3]")
    count = len(points)
    for key in ("point_ids", "point_visibility"):
        if data[key].shape != (count,) or not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError(f"Invalid {key}")
    if (not np.isin(data["point_visibility"], [-1, 0, 1]).all()
            or len(np.unique(data["point_ids"])) != count or np.any(data["point_ids"] < 0)):
        raise ValueError("Invalid visibility labels or point IDs")
    tolerance = float(data["visibility_tolerance"])
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid visibility tolerance")
    if version == 1:
        if require_normals:
            raise ValueError("Surface has no normals; run backfill_normals.py first")
        return
    normals = data["normals_camera"]
    if (normals.shape != points.shape or normals.dtype != np.float32
            or not np.isfinite(normals).all()
            or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Expected finite float32 unit normals aligned with points [N,3]")
    faces = data["face_indices"]
    if faces.shape != (count,) or faces.dtype != np.int64 or np.any(faces < 0):
        raise ValueError("Invalid sampled face indices")
    if str(data["normal_method"]) != NORMAL_METHOD:
        raise ValueError("Unsupported normal method")


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


def sample_full_surface(points, camera_path, depth_path, tolerance, normals, face_indices):
    with np.load(camera_path, allow_pickle=False) as camera:
        K = camera["K"]
        camera_transform = camera["T_camera_from_object"]
    depth = np.load(depth_path, allow_pickle=False)
    # OpenCV -> SAM camera axes, matching depth_to_pointmap().
    T_sam_from_object = sam_camera_transform(camera_path)
    points_camera = transform_points(points, T_sam_from_object).astype(np.float32)
    if not len(points_camera) or not np.isfinite(points_camera).all():
        raise ValueError(f"Empty or non-finite surface for {camera_path}")
    arrays = {
        "points_camera": points_camera,
        "normals_camera": transform_normals(normals, T_sam_from_object),
        "face_indices": np.asarray(face_indices, dtype=np.int64),
        "normal_method": NORMAL_METHOD,
        "point_visibility": classify_visibility(
            points, K, camera_transform, depth, tolerance
        ),
        # IDs identify this object's new sample, not the historical touch pool.
        "point_ids": np.arange(len(points), dtype=np.int64),
        "format_version": np.int64(2),
        "data_kind": "full_surface",
        "coordinate_frame": "sam_camera",
        "sampling_method": "area_weighted_direct",
        "visibility_tolerance": tolerance,
    }
    validate_surface(arrays, require_normals=True)
    return arrays


def save_surface(path, arrays, overwrite):
    validate_surface(arrays)
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Surface files must not require pickle/object arrays")
    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[key], value) for key, value in arrays.items()
            ):
                raise ValueError(f"Existing surface differs: {path}; use --overwrite")
        return
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
