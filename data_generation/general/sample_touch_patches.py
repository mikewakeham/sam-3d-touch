"""Dense patches on original mesh triangles; no mesh processing or geodesics."""

import numpy as np


def farthest_centers(points, count, seed):
    # NumPy version of the minimum-distance loop used by the existing FPS helpers.
    if count < 1 or len(points) < count:
        raise ValueError(f"Need {count} hidden candidates, got {len(points)}")
    selected = np.empty(count, dtype=np.int64)
    distance = np.full(len(points), np.inf)
    current = int(np.random.default_rng(seed).integers(len(points)))
    for index in range(count):
        selected[index] = current
        distance = np.minimum(distance, np.sum((points - points[current]) ** 2, axis=1))
        distance[selected[:index + 1]] = -1
        current = int(distance.argmax())
    return selected


def adaptive_region(points, normals, center, radius, thickness):
    # Local covariance plane fit, as in PCL's normal-estimation tutorial.
    # All nearby sides contribute: disagreement rounds the patch instead of deleting faces.
    # Estimate orientation over twice the patch radius: the original 8K cloud can
    # contain only a handful of samples inside a small patch. Sampling stays at radius.
    nearby = np.sum((points - center) ** 2, axis=1) <= (2 * radius) ** 2
    local = points[nearby]
    basis = np.eye(3)
    flatness = 0.
    if len(local) >= 12:
        delta = local - local.mean(axis=0)
        values, basis = np.linalg.eigh(delta.T @ delta / len(local))
        if values[2] > 0 and values[1] / values[2] >= .1:
            plane_confidence = np.clip((.15 - values[0] / values[1]) / (.15 - .02), 0, 1)
            local_normals = normals[nearby]
            concentration = np.linalg.eigvalsh(local_normals.T @ local_normals / len(local))[-1]
            normal_confidence = np.clip((concentration - .85) / (.98 - .85), 0, 1)
            flatness = float(min(plane_confidence, normal_confidence))
    # eigh orders axes from smallest to largest variance: axis 0 is the plane normal.
    radii = radius * np.array([1 - flatness * (1 - thickness), 1., 1.])
    return basis, radii, len(local)


def sample_region(mesh, center, basis, radii, count, seed):
    """Area-uniform samples on triangle/ellipsoid intersections, with original face IDs."""
    import trimesh

    radii = np.asarray(radii, dtype=np.float64)
    if count < 1 or np.any(radii <= 0) or not np.isfinite(radii).all():
        raise ValueError("Positive sample count and finite region radii required")
    # AABB overlap includes triangles crossing the region with no vertices inside it.
    triangles = mesh.triangles
    radius = float(max(radii))
    candidates = np.flatnonzero(
        (triangles.max(axis=1) >= center - radius).all(axis=1)
        & (triangles.min(axis=1) <= center + radius).all(axis=1)
        & (mesh.area_faces > 0)
    )
    if not len(candidates):
        raise ValueError('Region has no positive-area mesh intersection')
    triangles = (triangles[candidates] - center) @ basis / radii
    closest = trimesh.triangles.closest_point(triangles, np.zeros((len(triangles), 3)))
    keep = np.sum(closest ** 2, axis=1) < 1
    candidates, triangles = candidates[keep], triangles[keep]
    if not len(candidates):
        raise ValueError("Region has no positive-area mesh intersection")

    # In ellipsoid coordinates, each triangle plane intersects a unit sphere in a disk.
    # Bound that disk in the triangle's barycentric coordinates. This avoids sampling
    # an entire large wall triangle to fill a tiny patch, without clipping the mesh.
    edges = triangles[:, 1:] - triangles[:, :1]
    gradients = np.linalg.pinv(edges)
    normals = np.cross(edges[:, 0], edges[:, 1])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    plane_distance = np.sum(normals * triangles[:, 0], axis=1)
    disk_center = normals * plane_distance[:, None]
    uv_center = np.einsum('ni,nij->nj', disk_center - triangles[:, 0], gradients)
    halfwidth = np.sqrt(np.maximum(0, 1 - plane_distance ** 2))[:, None] * np.linalg.norm(gradients, axis=1)
    lower = np.maximum(0, uv_center - halfwidth)
    upper = np.minimum(1, uv_center + halfwidth)
    width = np.maximum(0, upper - lower)
    # Choose the smaller enclosing proposal: the original triangle or this parallelogram.
    rectangle_ratio = 2 * np.prod(width, axis=1)
    use_rectangle = rectangle_ratio < 1
    weights = mesh.area_faces[candidates] * np.minimum(1, rectangle_ratio)
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        raise ValueError("Region has no sampleable surface area")
    rng = np.random.default_rng(seed)
    samples, faces = [], []
    remaining = count
    for _ in range(200):
        size = max(1024, remaining * 4)
        chosen = np.searchsorted(cumulative, rng.random(size) * cumulative[-1], side='right')
        uv = rng.random((size, 2))
        rectangle = use_rectangle[chosen]
        # Triangle-point picking is the reflection method from trimesh.sample.sample_surface.
        flip = (~rectangle) & (uv.sum(axis=1) > 1)
        uv[flip] = 1 - uv[flip]
        uv[rectangle] = lower[chosen[rectangle]] + uv[rectangle] * width[chosen[rectangle]]
        local = triangles[chosen, 0] + np.einsum('ni,nij->nj', uv, edges[chosen])
        accept = (uv.sum(axis=1) <= 1) & (np.sum(local ** 2, axis=1) <= 1)
        chosen, uv = chosen[accept][:remaining], uv[accept][:remaining]
        face_ids = candidates[chosen]
        original = mesh.triangles[face_ids]
        samples.append(original[:, 0] + np.einsum('ni,nij->nj', uv, original[:, 1:] - original[:, :1]))
        faces.append(face_ids)
        remaining -= len(face_ids)
        if remaining == 0:
            return np.concatenate(samples), np.concatenate(faces).astype(np.int64)
    raise ValueError(f"Region rejection sampling could not fill {count} points; {remaining} missing")


def make_patch_bank(mesh, points, face_indices, visibility, seed, radius=.04,
                    thickness=.2, contacts=32, points_per_contact=1024):
    if not np.isfinite(radius) or radius <= 0 or not 0 < thickness <= 1:
        raise ValueError("Require positive finite radius and 0 < thickness <= 1")
    if points_per_contact < 2:
        raise ValueError("Need a center and at least one sampled point per contact")
    hidden = np.flatnonzero(visibility == 0)
    centers = hidden[farthest_centers(points[hidden], contacts, seed)]
    normals = mesh.face_normals[face_indices]
    patches, patch_faces, bases, sizes, support = [], [], [], [], []
    for index, center_index in enumerate(centers):
        center = points[center_index]
        basis, radii, count = adaptive_region(points, normals, center, radius, thickness)
        patch_seed = np.random.SeedSequence(seed, spawn_key=(index,))
        sampled, faces = sample_region(mesh, center, basis, radii, points_per_contact - 1, patch_seed)
        patches.append(np.vstack([center, sampled]))
        patch_faces.append(np.r_[face_indices[center_index], faces])
        bases.append(basis)
        sizes.append(radii)
        support.append(count)
    return dict(points_object=np.concatenate(patches), face_indices=np.concatenate(patch_faces),
                center_indices=centers, bases_object=np.array(bases), radii=np.array(sizes),
                support_counts=np.array(support),
                offsets=np.arange(contacts + 1, dtype=np.int64) * points_per_contact)


def validate_patches(data):
    if (int(data['format_version']) != 1 or str(data['data_kind']) != 'touch_patches'
            or str(data['coordinate_frame']) != 'sam_camera'):
        raise ValueError('Unsupported touch-patch format or frame')
    points, offsets = data['points_camera'], data['offsets']
    if (points.ndim != 2 or points.shape[1] != 3 or points.dtype != np.float32
            or not np.isfinite(points).all()):
        raise ValueError('Expected finite float32 patch points [N,3]')
    if (offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0
            or offsets[-1] != len(points) or np.any(np.diff(offsets) <= 0)
            or offsets.dtype != np.int64):
        raise ValueError('Invalid patch offsets')
    contacts = len(offsets) - 1
    for key, shape in [('face_indices', (len(points),)), ('point_visibility', (len(points),)),
                       ('center_point_ids', (contacts,))]:
        if data[key].shape != shape or not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError(f'Invalid {key}')
    if (np.any(data['face_indices'] < 0) or not np.isin(data['point_visibility'], [-1, 0, 1]).all()
            or np.any(data['point_visibility'][offsets[:-1]] != 0)):
        raise ValueError('Invalid faces, visibility, or non-hidden centers')
    basis, radii = data['bases_camera'], data['radii']
    if (basis.shape != (contacts, 3, 3) or radii.shape != (contacts, 3)
            or not np.isfinite(basis).all() or not np.isfinite(radii).all() or np.any(radii <= 0)
            or not np.allclose(basis.transpose(0, 2, 1) @ basis, np.eye(3), atol=1e-5)):
        raise ValueError('Invalid patch regions')
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        local = (points[start:end] - points[start]) @ basis[index] / radii[index]
        if np.any(np.sum(local ** 2, axis=1) > 1 + 1e-3):
            raise ValueError('Points extend outside their saved region')


def select_patch_indices(data, contacts, points_per_contact):
    validate_patches(data)
    offsets = data['offsets']
    if (contacts < 1 or contacts >= len(offsets) or points_per_contact < 1
            or np.any(np.diff(offsets)[:contacts] < points_per_contact)):
        raise ValueError('Requested more contacts/points than saved in the patch bank')
    # Nested prefixes retain the center and the same samples across all three budgets.
    return (offsets[:contacts, None] + np.arange(points_per_contact)).ravel()


def save_patches(path, arrays, overwrite=False):
    validate_patches(arrays)
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError('Patch files must not require pickle')
    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], value)
                                                     for key, value in arrays.items()):
                raise ValueError(f'Existing patches differ: {path}; use --overwrite')
        return
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
