"""Presentation layout: grounded meshes spinning beneath a fixed camera."""
import itertools

import numpy as np
import trimesh


SPACING = 1.35
ELEVATION = np.radians(25.)


def prepare_row(items, up):
    """Copy and normalize meshes to a unit rotation envelope, keeping materials."""
    vertical = 2 if up == 'z' else 1
    horizontal = [axis for axis in range(3) if axis != vertical]
    result = []
    for index, item in enumerate(items):
        if 'mesh' not in item:
            raise ValueError('--row requires mesh inputs, not point clouds')
        mesh = item['mesh'].copy()
        bounds = mesh.bounds
        origin = bounds.mean(axis=0)
        origin[vertical] = bounds[0, vertical]
        centered = mesh.vertices - origin
        # Fit the whole swept cylinder, not just the initial bounding box.
        diameter = 2 * np.linalg.norm(centered[:, horizontal], axis=1).max()
        size = max(diameter, np.ptp(mesh.vertices[:, vertical]))
        pivot = np.zeros(3)
        pivot[0] = (index - (len(items) - 1) / 2) * SPACING
        transform = np.eye(4)
        transform[:3, :3] /= size
        transform[:3, 3] = pivot - origin / size
        mesh.apply_transform(transform)
        result.append(dict(item, mesh=mesh, pivot=pivot,
                           transform=transform @ item.get('transform', np.eye(4))))
    return result


def spin_transform(pivot, angle, up):
    axis = [0., 0., 1.] if up == 'z' else [0., 1., 0.]
    return trimesh.transformations.rotation_matrix(angle, axis, pivot)


def row_camera(count, up, fov, aspect):
    """Fit the floor and every rotation to a fixed elevated orthographic camera."""
    vertical = 2 if up == 'z' else 1
    depth = 1 if up == 'z' else 2
    center = np.zeros(3)
    center[vertical] = .35
    toward_eye = row_eye(np.zeros(3), 1., up)
    camera_up = np.zeros(3)
    camera_up[vertical] = np.cos(ELEVATION)
    camera_up[depth] = np.sin(ELEVATION) * (1 if up == 'z' else -1)
    corners = np.array(list(itertools.product(
        [-count * SPACING / 2 - .2, count * SPACING / 2 + .2], [-1.3, 1.3], [-.027, 1.]
    )))
    if up == 'y':
        corners = corners[:, [0, 2, 1]]
    relative = corners - center
    tan_v = np.tan(np.radians(fov) / 2)
    projected = np.maximum(np.abs(relative[:, 0]) / (tan_v * aspect),
                           np.abs(relative @ camera_up) / tan_v)
    distance = float(max(np.max(1.1 * projected), np.max(relative @ toward_eye) + 1.))
    return center, distance


def row_eye(center, distance, up):
    # Both up conventions show the input order from left to right.
    direction = np.array([0., -np.cos(ELEVATION), np.sin(ELEVATION)])
    if up == 'y':
        direction = direction[[0, 2, 1]] * [1., 1., -1.]
    return center + distance * direction


def add_row_floor(renderer, count, up):
    import open3d as o3d

    vertical = 2 if up == 'z' else 1
    depth = 1 if up == 'z' else 2
    width = count * SPACING

    def add_box(name, x, length, floor_depth, top, thickness, color):
        extent = np.zeros(3)
        extent[0], extent[depth], extent[vertical] = length, floor_depth, thickness
        mesh = o3d.geometry.TriangleMesh.create_box(*extent)
        offset = -extent / 2
        offset[0] = x - length / 2
        offset[vertical] = top - thickness
        mesh.translate(offset)
        mesh.compute_vertex_normals()
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = 'defaultLit'
        material.base_color = (*color, 1.)
        material.base_roughness = 1.
        renderer.scene.add_geometry(name, mesh, material)
        renderer.scene.scene.geometry_shadows(name, False, True)

    add_box('floor', 0., width + .4, 2.6, -.002, .025, (.87, .88, .90))
    for index in range(count + 1):
        add_box(f'divider_{index}', (index - count / 2) * SPACING,
                .009, 2.2, -.001, .002, (.65, .67, .70))


def initialize_row_lighting(renderer, strength, up):
    scene = renderer.scene.scene
    direction = np.array([-.35, .45, -1.])
    if up == 'y':
        direction = direction[[0, 2, 1]]
    scene.enable_sun_light(False)
    scene.enable_indirect_light(True)
    scene.set_indirect_light_intensity(50000. * strength)
    scene.add_directional_light('RowKey', np.ones(3, dtype=np.float32),
                                direction.astype(np.float32), 80000. * strength, True)
