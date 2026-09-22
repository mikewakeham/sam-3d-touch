"""Geometry loading shared by SAM3D evaluation, orbits and the browser viewer."""
import csv
import json
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path):
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            mesh = trimesh.Trimesh(vertices=data['vertices'], faces=data['faces'], process=False)
    else:
        # Trimesh applies scene graph transforms when flattening a GLB/GLTF scene.
        mesh = trimesh.load(str(path), force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty or not len(mesh.faces):
        raise ValueError(f'No triangle mesh in {path}')
    if not np.isfinite(mesh.vertices).all() or np.max(mesh.extents) <= 0:
        raise ValueError(f'Invalid mesh coordinates in {path}')
    return mesh


def load_dataset_mesh(record, dataset):
    """Return a mesh in normalized-object coordinates, without applying transforms twice."""
    path = dataset.resolve_path(record['mesh_path']) if record.get('mesh_path') else (
        dataset.root / 'objects' / record['object_id'] / 'model.obj'
    )
    mesh = load_mesh(path)
    transform = np.eye(4)
    if Path(path).suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            if str(data['coordinate_frame']) != 'normalized_object':
                raise ValueError(f'Expected normalized_object mesh: {path}')
        return mesh, transform
    # The original Objaverse manifests refer to the source mesh.
    with np.load(dataset.resolve_path(record['object_transform_path']), allow_pickle=False) as data:
        transform = data['T_normalized_from_source'].astype(np.float64)
    mesh.apply_transform(transform)
    return mesh, transform


def voxel_mesh(points, size):
    # Cube construction from the original make_eval_orbit.py.
    cube = trimesh.creation.box(extents=(size, size, size))
    vertices = (points[:, None] + cube.vertices[None]).reshape(-1, 3)
    faces = cube.faces[None] + len(cube.vertices) * np.arange(len(points))[:, None, None]
    return trimesh.Trimesh(vertices=vertices, faces=faces.reshape(-1, 3), process=False)


def load_geometry(path, points_key=None, voxel_key=None):
    path = Path(path)
    item = {'name': path.stem, 'path': str(path), 'frame': 'unspecified'}
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            if 'coordinate_frame' in data:
                item['frame'] = str(data['coordinate_frame'])
            if voxel_key:
                grid = data[voxel_key]
                if grid.ndim != 3 or grid.shape != (64, 64, 64):
                    raise ValueError('SAM3D occupancy must be a 64 x 64 x 64 grid')
                if not np.isin(grid, [0, 1]).all():
                    raise ValueError('Expected decoded binary occupancy, not logits or a latent')
                points = np.argwhere(grid).astype(np.float64) / 64 - .5
                if not len(points):
                    raise ValueError(f'Empty occupancy in {path}')
                item['mesh'] = voxel_mesh(points, .9 / 64)
                item['frame'] = 'sam3d_decoder'
                return item
            if 'vertices' in data and 'faces' in data and not points_key:
                item['mesh'] = load_mesh(path)
                return item
            keys = [key for key in ('points', 'points_object', 'points_camera') if key in data]
            if points_key is None and len(keys) != 1:
                raise ValueError(f'Select --points-key or --voxel-key for {path}; keys: {list(data)}')
            key = points_key or keys[0]
            item['points'] = data[key].copy()
            normal_key = key.replace('points', 'normals', 1)
            if normal_key in data:
                item['normals'] = data[normal_key].copy()
            if 'colors' in data:
                item['colors'] = data['colors'].copy()
    elif path.suffix.lower() == '.npy':
        item['points'] = np.load(path, allow_pickle=False)
    else:
        loaded = trimesh.load(str(path), process=False)
        if isinstance(loaded, trimesh.points.PointCloud):
            item['points'] = loaded.vertices.copy()
            if loaded.colors is not None and len(loaded.colors):
                item['colors'] = loaded.colors[:, :3].copy()
        else:
            item['mesh'] = load_mesh(path)
            return item
    points = np.asarray(item['points'])
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError(f'Expected finite nonempty Nx3 points in {path}')
    for key in ('normals', 'colors'):
        if key in item and (item[key].shape != points.shape or not np.isfinite(item[key]).all()):
            raise ValueError(f'{key} must match Nx3 points in {path}')
    if 'colors' in item:
        colors = item['colors']
        if np.issubdtype(colors.dtype, np.floating) and colors.max() <= 1:
            colors = colors * 255
        item['colors'] = np.clip(colors, 0, 255).astype(np.uint8)
    return item


def scene_bounds(items):
    bounds = []
    for item in items:
        points = item['mesh'].vertices if 'mesh' in item else item['points']
        bounds.extend([np.min(points, axis=0), np.max(points, axis=0)])
    bounds = np.asarray(bounds)
    return np.stack([bounds.min(axis=0), bounds.max(axis=0)])


def camera_fit(items, fov=40., aspect=1.):
    bounds = scene_bounds(items)
    center = bounds.mean(axis=0)
    radius = np.linalg.norm(bounds[1] - bounds[0]) / 2
    half_fov = min(np.radians(fov) / 2, np.arctan(np.tan(np.radians(fov) / 2) * aspect))
    distance = max(radius, 1e-3) / np.sin(half_fov)
    return center, distance


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def find_record(config, sample_id):
    root = Path(config['dataset']['root'])
    with resolve(root, config['dataset']['manifest']).open() as file:
        for line in file:
            record = json.loads(line)
            if record['sample_id'] == sample_id:
                return root, record
    raise ValueError(f'Sample {sample_id} is missing from the saved dataset manifest')


def load_reference_camera(args):
    """Saved OpenCV camera expressed in the same frame as the displayed geometry."""
    from PIL import Image

    if args.inputs:
        return None
    if args.evaluation_dir:
        import yaml

        root = args.evaluation_dir
        with (root / 'config.yaml').open() as file:
            config = yaml.safe_load(file)
        dataset_root, record = find_record(config['selection_data_config'], args.sample_id)
        camera_path = resolve(dataset_root, record['camera_path'])
        image_path = resolve(dataset_root, record['image_path'])
        with (root / 'metrics.csv').open(newline='') as file:
            row = next(row for row in csv.DictReader(file)
                       if row['sample_id'] == args.sample_id and row.get('error') == ''
                       and (not args.conditions or row['condition'] in args.conditions))
        with np.load(resolve(root, row['target_points_path']), allow_pickle=False) as data:
            world_from_object = data['evaluation_normalization'].astype(np.float64)
    else:
        view = args.data_root / 'generated_data' / args.object_id / 'views' / f'{args.view_id:03d}'
        camera_path, image_path = view / 'camera.npz', view / 'image.png'
        with np.load(camera_path, allow_pickle=False) as data:
            # load_dataset_view displays geometry in SAM camera coordinates.
            world_from_object = np.diag([-1., -1., 1., 1.]) @ data['T_camera_from_object']
    with np.load(camera_path, allow_pickle=False) as data:
        K = data['K'].astype(np.float64)
        pose = world_from_object @ np.linalg.inv(data['T_camera_from_object'])
    # Evaluation normalization scales both camera position and geometry. Camera
    # axes must remain orthonormal for Open3D's rigid world-to-camera extrinsic.
    scale = np.cbrt(np.linalg.det(pose[:3, :3]))
    if not np.isfinite(pose).all() or scale <= 0:
        raise ValueError(f'Invalid input camera transform: {camera_path}')
    pose[:3, :3] /= scale
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError('Input camera requires a rigid pose and uniform geometry normalization')
    with Image.open(image_path) as image:
        width, height = image.size
    K[0] *= args.width / width
    K[1] *= args.height / height
    # OpenCV camera +Y points down the image. Orbit around screen-up so even
    # a top-down input travels around the object instead of circling its pole.
    up = -pose[:3, 1]
    up = up / np.linalg.norm(up)
    return dict(intrinsics=K.tolist(), camera_to_world=pose.tolist(),
                pivot=world_from_object[:3, 3].tolist(), up=up.tolist(),
                source=str(camera_path), first_frame='input camera', orbit_axis='input_camera_up')


def load_evaluation(evaluation_dir, sample_id, conditions=None, modes=('mesh',)):
    root = Path(evaluation_dir)
    with open(root / 'metrics.csv', newline='') as file:
        rows = [row for row in csv.DictReader(file) if row['sample_id'] == sample_id and not row['error']]
    rows = {row['condition']: row for row in rows}
    conditions = conditions or list(rows)
    if not conditions or any(name not in rows for name in conditions):
        raise ValueError(f'No completed results for requested conditions/sample {sample_id}')
    first = rows[conditions[0]]
    items = []
    if 'mesh' in modes:
        items.append({'name': 'mesh_ground_truth', 'mesh': load_mesh(resolve(root, first['target_mesh_path']))})
        for name in conditions:
            items.append({'name': f'mesh_{name}', 'mesh': load_mesh(resolve(root, rows[name]['mesh_aligned_path']))})
    if 'voxel' in modes:
        for name in conditions:
            row = rows[name]
            # Show the actual decoded Stage-1 output, before Stage-2 pruning/downsampling.
            with np.load(resolve(root, row['stage1_path']), allow_pickle=False) as data:
                grid = data['prediction']
                if grid.shape != (64, 64, 64):
                    raise ValueError(f"Invalid Stage-1 occupancy shape: {grid.shape}")
                points = np.argwhere(grid).astype(np.float64) / 64 - .5
                if not len(points):
                    raise ValueError(f"Empty Stage-1 occupancy: {row['condition']}")
            with np.load(resolve(root, row['alignment_path']), allow_pickle=False) as data:
                transform = data['icp_transform'] @ data['prediction_normalization']
            points = trimesh.transform_points(points, transform)
            size = .9 * abs(np.linalg.det(transform[:3, :3])) ** (1 / 3) / 64
            items.append({'name': f'voxel_{name}', 'mesh': voxel_mesh(points, size)})
    for item in items:
        item['frame'] = 'evaluation_aligned'
    # The copied image keeps the report usable after moving the dataset.
    image = root / 'selected_views' / f'{sample_id}.png'
    if not image.exists():
        image = Path(first['image_path'])
    return items, [image] if image.exists() else []


def load_dataset_view(root, object_id, view_id):
    # Current general/view_data.py layout and transforms; no training imports.
    from PIL import Image
    from data_generation.general.pointmaps import depth_to_pointmap

    directory = Path(root) / 'generated_data' / object_id
    view = directory / 'views' / f'{view_id:03d}'
    mesh = load_mesh(directory / 'mesh.npz')
    with np.load(view / 'camera.npz', allow_pickle=False) as data:
        K = data['K']
        transform = np.diag([-1., -1., 1., 1.]) @ data['T_camera_from_object']
    mesh.apply_transform(transform)
    items = [{'name': 'mesh', 'mesh': mesh}]
    rgba = np.array(Image.open(view / 'image.png').convert('RGBA'))
    points = depth_to_pointmap(np.load(view / 'depth.npy', allow_pickle=False), K)
    valid = np.isfinite(points).all(axis=-1) & (rgba[..., 3] > 0)
    items.append({'name': 'pointmap', 'points': points[valid], 'colors': rgba[..., :3][valid]})
    surface = view / 'full_surface.npz'
    if surface.exists():
        items.append(load_geometry(surface, points_key='points_camera'))
    pool = directory / 'surface_pool.npz'
    if pool.exists():
        item = load_geometry(pool, points_key='points_object')
        item['points'] = trimesh.transform_points(item['points'], transform)
        item['normals'] = item['normals'] @ transform[:3, :3].T
        items.append(item)
    for item in items:
        item['frame'] = 'sam_camera'
    return items, [view / 'image.png']


def add_input_arguments(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--inputs', type=Path, nargs='+', help='Meshes or point clouds; coordinates are preserved')
    source.add_argument('--evaluation-dir', type=Path)
    source.add_argument('--data-root', type=Path, help='Current general data-generation output')
    parser.add_argument('--sample-id', help='Required with --evaluation-dir')
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--modes', nargs='+', choices=['mesh', 'voxel'], default=['mesh'])
    parser.add_argument('--object-id', help='Required with --data-root')
    parser.add_argument('--view-id', type=int, default=0)
    parser.add_argument('--points-key', help='NPZ point array; otherwise recognizes points/points_object/points_camera')
    parser.add_argument('--voxel-key', help='Explicit binary 64-cubed SAM3D occupancy key, e.g. prediction or target')
    parser.add_argument('--labels', nargs='+', help='One display label per loaded geometry')
    parser.add_argument('--up', choices=['y', 'z'], default='y')


def load_inputs(args):
    if args.inputs:
        items = [load_geometry(path, args.points_key, args.voxel_key) for path in args.inputs]
        images = []
    elif args.evaluation_dir:
        if not args.sample_id:
            raise ValueError('--evaluation-dir requires --sample-id')
        items, images = load_evaluation(args.evaluation_dir, args.sample_id, args.conditions, args.modes)
    else:
        if not args.object_id:
            raise ValueError('--data-root requires --object-id')
        items, images = load_dataset_view(args.data_root, args.object_id, args.view_id)
    if args.labels:
        if len(args.labels) != len(items):
            raise ValueError('--labels must have one entry per geometry')
        for item, name in zip(items, args.labels):
            item['name'] = name
    return items, images
