"""CPU geometry for the representation probe, independent of model imports."""
import numpy as np
from scipy.spatial import cKDTree


def point_grid(points, resolution=64):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid surface points")
    if not len(points) or np.max(np.abs(points)) > .50001:
        raise ValueError("Surface is outside the expected object cube")
    indices = np.floor((np.clip(points, -.5 + 1e-6, .5 - 1e-6) + .5) * resolution).astype(int)
    grid = np.zeros((resolution,) * 3, dtype=bool)
    grid[tuple(indices.T)] = True
    return grid


def overlap(pred, target):
    assert pred.shape == target.shape and pred.dtype == target.dtype == bool
    intersection = int((pred & target).sum())
    union = int((pred | target).sum())
    return {"iou": intersection / union if union else 1.,
            "precision": intersection / int(pred.sum()) if pred.any() else 0.,
            "recall": intersection / int(target.sum()) if target.any() else 0.,
            "predicted_occupied": int(pred.sum()), "target_occupied": int(target.sum())}


def sample_mesh(vertices, faces, count=20000, seed=29):
    triangles = np.asarray(vertices)[np.asarray(faces)]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("Invalid reconstructed mesh")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), count, p=area / area.sum())]
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, :1])
    weights = np.concatenate((1 - root, root * (1 - uv[:, 1:]), root * uv[:, 1:]), axis=1)
    return (selected * weights[:, :, None]).sum(axis=1)


def surface_agreement(predicted, reference, tolerance=1 / 64):
    """Sampling-based accuracy/completeness at one target voxel width; no ICP."""
    if not len(predicted) or not len(reference):
        return {"valid": False, "precision": 0., "recall": 0., "fscore": 0.}
    p = cKDTree(reference).query(predicted)[0]
    q = cKDTree(predicted).query(reference)[0]
    precision, recall = float(np.mean(p <= tolerance)), float(np.mean(q <= tolerance))
    return {"valid": True, "tolerance": tolerance, "precision": precision, "recall": recall,
            "fscore": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "prediction_to_reference_p95": float(np.quantile(p, .95)),
            "reference_to_prediction_p95": float(np.quantile(q, .95))}


def zero_surface(field, lower=-1.05, upper=1.05):
    from skimage.measure import marching_cubes
    if not np.isfinite(field).all():
        raise FloatingPointError("Nonfinite VecSetX field")
    if not field.min() < 0 < field.max():
        return None
    vertices, faces, _, _ = marching_cubes(field, level=0,
        spacing=((upper - lower) / (field.shape[0] - 1),) * 3)
    return vertices + lower, faces
