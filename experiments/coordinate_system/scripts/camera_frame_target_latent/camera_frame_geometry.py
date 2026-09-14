"""Explicit affine frames and deterministic splits for the camera-frame rerun.

No pretrained model imports. Target bounds and VecSetX units are deliberately
distinct; sharing a pointmap normalizer does not change the target cube.
"""
import hashlib
import json
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def affine(points, matrix):
    p = np.asarray(points)
    m = np.asarray(matrix, dtype=np.float64)
    if p.shape[-1] != 3 or m.shape != (4, 4):
        raise ValueError('Expected XYZ points and a 4x4 transform')
    return p @ m[:3, :3].T + m[:3, 3]


def normalization(points, kind):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected nonempty finite [N,3] geometry')
    center = (points.min(0) + points.max(0))/2
    centered = points-center
    if kind == 'radius':
        denominator = np.linalg.norm(centered, axis=1).max()
    elif kind == 'extent':
        denominator = np.ptp(points, axis=0).max()
    else:
        raise ValueError(kind)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Degenerate geometry')
    matrix = np.eye(4)
    matrix[:3, :3] /= denominator
    matrix[:3, 3] = -center/denominator
    return matrix, center, float(denominator)


def make_frames(object_vertices, camera_surface, camera_from_object):
    transform = np.asarray(camera_from_object, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Invalid camera matrix')
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError('Invalid homogeneous camera row')
    singular = np.linalg.svd(transform[:3, :3], compute_uv=False)
    if np.linalg.det(transform[:3, :3]) <= 0 or np.max(np.abs(singular-1)) > 32*np.finfo(np.float32).eps:
        raise ValueError('Expected approximately proper camera rotation')
    vertices_camera = affine(object_vertices, transform)
    target_from_camera, mesh_center, mesh_extent = normalization(vertices_camera, 'extent')
    vec_from_camera, sample_center, sample_radius = normalization(camera_surface, 'radius')
    target_vertices = affine(vertices_camera, target_from_camera)
    if np.abs(target_vertices).max() > .5+1e-9:
        raise ValueError('Target would exceed the fixed voxel cube')
    return dict(camera_from_object=transform.tolist(),
        target_from_camera=target_from_camera.tolist(), vec_from_camera=vec_from_camera.tolist(),
        target_from_object=(target_from_camera@transform).tolist(),
        target_from_vec=(target_from_camera@np.linalg.inv(vec_from_camera)).tolist(),
        mesh_camera_center=mesh_center.tolist(), mesh_camera_extent=mesh_extent,
        surface_camera_center=sample_center.tolist(), surface_camera_radius=sample_radius,
        target_max_abs=float(np.abs(target_vertices).max()),
        camera_singular_error=float(np.max(np.abs(singular-1))))


def select_records(records, splits, seed=29, train_objects=16, val_objects=0, held_views=4, train_views=8):
    """Deterministic view subsets; optional disjoint objects for later larger studies."""
    if train_objects < 1 or val_objects < 0 or held_views < 1 or train_views < 0:
        raise ValueError('Training/held counts must be positive; val objects and train views may be zero')
    if set(splits['train']) & set(splits['val']):
        raise ValueError('Original train/validation objects overlap')
    by_object = {}
    seen = set()
    for row in records:
        if row['sample_id'] in seen:
            raise ValueError('Duplicate sample_id')
        seen.add(row['sample_id'])
        by_object.setdefault(row['object_id'], []).append(row)
    selected = {}
    for split, count in [('train', train_objects), ('val', val_objects)]:
        # No filtering based on geometric quality or model performance.
        ids = sorted(set(splits[split]) & by_object.keys(), key=lambda oid: digest([seed, oid]))
        if len(ids) < count:
            raise ValueError(f'Need {count} {split} objects, found {len(ids)}')
        selected[split] = ids[:count]
    result = {'train': [], 'held_view': [], 'val': []}
    for split, ids in selected.items():
        for oid in ids:
            rows = sorted(by_object[oid], key=lambda r: digest([seed, r['sample_id']]))
            if split == 'train':
                if len(rows) < held_views+max(train_views, 2):
                    raise ValueError(f'{oid}: insufficient views; do not silently replace selected objects')
                result['held_view'].extend(rows[:held_views])
                result['train'].extend(rows[held_views:held_views+train_views] if train_views else rows[held_views:])
            else:
                result['val'].extend(rows)
    return result, selected
