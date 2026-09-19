"""Area-weighted full surfaces in the same SAM camera frame as the pointmap."""


import numpy as np

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


def sample_full_surface(points, camera_path, depth_path, tolerance):
    with np.load(camera_path, allow_pickle=False) as camera:
        K = camera["K"]
        camera_transform = camera["T_camera_from_object"]
    depth = np.load(depth_path, allow_pickle=False)
    # OpenCV -> SAM camera axes, matching depth_to_pointmap().
    T_sam_from_object = np.diag([-1.0, -1.0, 1.0, 1.0]) @ camera_transform
    points_camera = transform_points(points, T_sam_from_object).astype(np.float32)
    if not len(points_camera) or not np.isfinite(points_camera).all():
        raise ValueError(f"Empty or non-finite surface for {camera_path}")
    return {
        "points_camera": points_camera,
        "point_visibility": classify_visibility(
            points, K, camera_transform, depth, tolerance
        ),
        # IDs identify this object's new sample, not the historical touch pool.
        "point_ids": np.arange(len(points), dtype=np.int64),
        "format_version": np.int64(1),
        "data_kind": "full_surface",
        "coordinate_frame": "sam_camera",
        "sampling_method": "area_weighted_direct",
        "visibility_tolerance": tolerance,
    }


def save_surface(path, arrays, overwrite):
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

