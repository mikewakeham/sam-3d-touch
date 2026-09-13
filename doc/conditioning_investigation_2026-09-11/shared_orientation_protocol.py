"""Alternative target convention only; never used to construct condition tokens."""
import numpy as np

# Bound accumulated float32 camera-matrix arithmetic by geometric distortion.
# On the original unit cube this permits <0.00022 of a 64-grid voxel relative
# to the nearest proper rotation. Keep the recorded matrix itself unchanged.
CAMERA_SINGULAR_VALUE_TOLERANCE = 32 * np.finfo(np.float32).eps


def validate_camera_transform(transform):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Invalid camera transform')
    rotation = transform[:3, :3]
    singular_values = np.linalg.svd(rotation, compute_uv=False)
    error = float(np.max(np.abs(singular_values - 1)))
    determinant = float(np.linalg.det(rotation))
    row_error = float(np.max(np.abs(transform[3] - [0, 0, 0, 1])))
    if (determinant <= 0 or error > CAMERA_SINGULAR_VALUE_TOLERANCE
            or row_error > 1e-8):
        raise ValueError('Expected approximately proper rigid camera transform: '
                         f'det={determinant:.9g}, max_singular_error={error:.9g} '
                         f'(limit={CAMERA_SINGULAR_VALUE_TOLERANCE:.9g}), '
                         f'homogeneous_row_error={row_error:.9g}')
    return dict(determinant=determinant, singular_values=singular_values.tolist(),
        orthogonality_max_error=float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        max_singular_value_error=error, singular_value_tolerance=float(CAMERA_SINGULAR_VALUE_TOLERANCE),
        unit_cube_displacement_bound=error*np.sqrt(3)/2,
        unit_cube_displacement_bound_in_voxels=error*np.sqrt(3)/2*64,
        matrix_policy='Recorded matrix retained; no SVD projection or input change')


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def camera_oriented_target(vertices, camera_from_object):
    """Retain camera axes, using the target generator's centered unit-box convention.

    Camera translation cancels under object centering. This is normalized shape
    supervision, not metric camera placement. Complete GT defines labels only.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    transform = np.asarray(camera_from_object, dtype=np.float64)
    camera_check = validate_camera_transform(transform)
    rotation = transform[:3, :3]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid vertices')
    rotated = vertices @ rotation.T
    low, high = rotated.min(0), rotated.max(0)
    center = (low + high) / 2
    extent = float((high - low).max())
    if extent <= 0:
        raise ValueError('Degenerate target mesh')
    output_from_object = np.eye(4)
    output_from_object[:3, :3] = rotation / extent
    output_from_object[:3, 3] = -center / extent
    normalized = transform_points(vertices, output_from_object)
    assert np.abs(normalized).max() <= .5 + 1e-10
    np.testing.assert_allclose((normalized.min(0) + normalized.max(0))/2, 0, atol=1e-10)
    np.testing.assert_allclose((normalized.max(0)-normalized.min(0)).max(), 1, atol=1e-10)
    inverse = np.linalg.inv(output_from_object)
    np.testing.assert_allclose(transform_points(normalized, inverse), vertices, atol=1e-10)
    metadata = dict(output_from_object=output_from_object.tolist(), object_from_output=inverse.tolist(),
        camera_validation=camera_check,
        rotated_bbox_center=center.tolist(), rotated_max_extent=extent,
        naive_rotation_outside_cube_fraction=float(np.mean(np.any(np.abs(rotated)>.5+1e-6, axis=1))),
        normalized_max_absolute_coordinate=float(np.abs(normalized).max()))
    return normalized, metadata
