"""Rotation-only registration and voxelization for generated Stage-1 supports."""
import numpy as np
from scipy.spatial import cKDTree

from experiments.coordinate_system.scripts.shared.pose_shape_geometry import (
    metrics, points, proper_axes, subset,
)


def objective(source, target, rotation):
    moved = source @ rotation.T
    forward = cKDTree(target).query(moved, workers=1)[0]
    backward = cKDTree(moved).query(target, workers=1)[0]
    return float((np.mean(forward**2) + np.mean(backward**2))/2)


def kabsch_about_origin(source, target, weights):
    """Fit target ~= source @ R.T, with no centering or translation."""
    weights = weights/weights.sum()
    u, _, vt = np.linalg.svd((source*weights[:, None]).T @ target)
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ fix @ u.T
    assert np.linalg.det(rotation) > 0
    return rotation


def refine(source, target, initial, iterations=30):
    rotation = np.asarray(initial, dtype=np.float64)
    best = (objective(source, target, rotation), rotation.copy())
    target_tree = cKDTree(target)
    for _ in range(iterations):
        moved = source @ rotation.T
        forward = target_tree.query(moved, workers=1)[1]
        backward = cKDTree(moved).query(target, workers=1)[1]
        a = np.concatenate((source, source[backward]))
        b = np.concatenate((target[forward], target))
        weights = np.concatenate((np.full(len(source), .5/len(source)),
                                  np.full(len(target), .5/len(target))))
        candidate = kabsch_about_origin(a, b, weights)
        score = objective(source, target, candidate)
        if score < best[0]:
            best = score, candidate.copy()
        if np.max(np.abs(candidate-rotation)) < 1e-9:
            break
        rotation = candidate
    return best


def fit_rotation(source, target):
    """Multi-start proper rotation fit about the fixed grid origin."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if not len(source) or not len(target):
        raise ValueError('Cannot align an empty decoded support')
    coarse_source, coarse_target = subset(source, 1536), subset(target, 1536)
    source_basis = np.linalg.eigh(source.T @ source/len(source))[1]
    target_basis = np.linalg.eigh(target.T @ target/len(target))[1]
    if np.linalg.det(source_basis) < 0: source_basis[:, 0] *= -1
    if np.linalg.det(target_basis) < 0: target_basis[:, 0] *= -1
    starts = [('identity', np.eye(3))]
    starts += [(f'axes_{i}', r) for i, r in enumerate(proper_axes())]
    starts += [(f'pca_{i}', target_basis @ r @ source_basis.T)
               for i, r in enumerate(proper_axes())]
    # Rank all analytic starts cheaply, then refine only the best candidates.
    ranked = sorted((objective(coarse_source, coarse_target, r), label, r)
                    for label, r in starts)
    coarse = []
    for _, label, initial in ranked[:8]:
        score, rotation = refine(coarse_source, coarse_target, initial, 25)
        coarse.append((score, label, rotation))
    coarse.sort(key=lambda item: item[0])
    fine_source, fine_target = subset(source, 4096), subset(target, 4096)
    finalists = []
    for _, label, initial in coarse[:4]:
        score, rotation = refine(fine_source, fine_target, initial, 40)
        finalists.append((score, label, rotation))
    _, label, rotation = min(finalists)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(rotation), 1, atol=1e-8)
    aligned = source @ rotation.T
    cosine = np.clip((np.trace(rotation)-1)/2, -1, 1)
    return aligned, {'rotation': rotation.tolist(), 'angle_degrees': float(np.degrees(np.arccos(cosine))),
                     'winning_start': label, 'starts': len(starts), 'refined_starts': 8,
                     'raw': metrics(source, target), 'aligned_continuous': metrics(aligned, target)}


def voxelize_points(point_cloud, resolution=64):
    """Nearest-cell rasterization in the fixed [-.5,.5]^3 target grid."""
    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    indices = np.floor((point_cloud+.5)*resolution).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < resolution), axis=1)
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    grid[tuple(indices[valid].T)] = True
    return grid, {'input_points': len(point_cloud), 'in_bounds_points': int(valid.sum()),
                  'dropped_fraction': float(1-valid.mean()), 'occupied_voxels': int(grid.sum())}


def align_decoded_grid(prediction, target):
    source, reference = points(prediction), points(target)
    aligned_points, fit = fit_rotation(source, reference)
    original_grid, original_raster = voxelize_points(source)
    np.testing.assert_array_equal(original_grid, prediction)
    aligned_grid, aligned_raster = voxelize_points(aligned_points)
    fit['raw_rasterization'] = original_raster
    fit['aligned_rasterization'] = aligned_raster
    fit['aligned_voxelized'] = metrics(points(aligned_grid), reference)
    return aligned_points, aligned_grid, fit
