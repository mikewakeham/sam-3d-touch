"""Input orbit groups, using the dataset and encoder configuration saved by evaluation."""
import copy
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import trimesh
import yaml
from PIL import Image

from evaluation.geometry import load_mesh, resolve


def surface_indices(pre_encoder_points, encoder_name, device='cuda'):
    if encoder_name in ('craftsman', 'triposg'):
        # All input points are the attention key/value cloud; query FPS does not replace it.
        return np.arange(len(pre_encoder_points))
    if encoder_name != 'vecsetx':
        raise ValueError(f'Unknown surface encoder: {encoder_name}')
    # make_vecsetx() -> learnable_vec1024x32_dim1024_depth24_nb(pc_size=8192).
    count = 8192
    if len(pre_encoder_points) == count:
        return np.arange(count)
    import torch
    from pytorch3d.ops import sample_farthest_points

    # Copied from TouchEncoder.normalize_points_for_vecsetx / prepare_points.
    # Replay on the SAVED pre-encoder coordinates: FPS in camera space can differ after SSI.
    points = torch.as_tensor(pre_encoder_points, dtype=torch.float32, device=device)[None]
    shift = (points.amax(dim=1) + points.amin(dim=1)) / 2
    points = points - shift[:, None]
    radius = torch.linalg.vector_norm(points, dim=2).amax(dim=1)
    if not torch.isfinite(radius).all() or (radius <= 0).any():
        raise ValueError('Invalid saved pre-encoder point cloud')
    points = points * (1 / radius)[:, None, None]
    _, indices = sample_farthest_points(points, lengths=torch.tensor([points.shape[1]], device=device),
                                        K=count, random_start_point=False)
    return indices[0][indices[0] >= 0].cpu().numpy()


def textured_mesh_path(object_dir, cache_dir, blender):
    metadata_path = object_dir / 'render_metadata.json'
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Textured export needs the general pipeline metadata: {metadata_path}')
    metadata = json.loads(metadata_path.read_text())
    sources = [object_dir / 'mesh.npz', metadata_path, Path(metadata['source'])]
    sources.extend(Path(asset['path']) for asset in metadata.get('external_assets', []))
    identity = [{'path': str(path.resolve()), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                for path in sources]
    output = cache_dir / 'mesh_textured.glb'
    stamp = cache_dir / 'mesh_textured_sources.json'
    if output.is_file() and output.stat().st_size and stamp.is_file() and json.loads(stamp.read_text()) == identity:
        return output
    cache_dir.mkdir(parents=True, exist_ok=True)
    command = [str(blender), '--background', '--factory-startup', '--disable-autoexec', '--threads', '1',
               '--python-exit-code', '1',
               '--python', str(Path(__file__).with_name('export_textured_mesh.py')), '--',
               '--object-dir', str(object_dir.resolve()), '--output', str(output.resolve())]
    log = cache_dir / 'textured_export.log'
    print(f'Exporting textured source; log: {log}', flush=True)
    with log.open('w') as file:
        result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT)
    if result.returncode or not output.is_file():
        raise RuntimeError(f'Textured export failed; see {log}')
    stamp.write_text(json.dumps(identity, indent=2) + '\n')
    return output


def find_record(config, sample_id):
    root = Path(config['dataset']['root'])
    with resolve(root, config['dataset']['manifest']).open() as file:
        for line in file:
            record = json.loads(line)
            if record['sample_id'] == sample_id:
                return root, record
    raise ValueError(f'Sample {sample_id} is missing from the saved dataset manifest')


def input_groups(mesh, textured, pointmap, surfaces):
    groups = [('input_mesh_textured', [textured]), ('input_mesh', [mesh]),
              ('input_mesh_pointmap', [mesh, pointmap]), ('input_pointmap', [pointmap])]
    for surface, conditions in surfaces:
        suffix = '' if len(surfaces) == 1 else '_' + conditions[0]
        groups.extend([
            ('input_mesh_surface' + suffix, [mesh, surface]),
            ('input_mesh_pointmap_surface' + suffix, [mesh, pointmap, surface]),
            ('input_surface' + suffix, [surface]),
            ('input_pointmap_surface' + suffix, [pointmap, surface]),
        ])
    return groups


def load_input_visualizations(args):
    root = args.evaluation_dir
    with (root / 'config.yaml').open() as file:
        config = yaml.safe_load(file)
    with (root / 'metrics.csv').open(newline='') as file:
        rows = {row['condition']: row for row in csv.DictReader(file)
                if row['sample_id'] == args.sample_id and row.get('error') == ''
                and (not args.conditions or row['condition'] in args.conditions)}
    if not rows:
        raise ValueError(f'No completed results for {args.sample_id}')
    first = next(iter(rows.values()))
    dataset_root, record = find_record(config['selection_data_config'], args.sample_id)
    source_mesh = resolve(dataset_root, record.get('mesh_path', f"objects/{record['object_id']}/model.obj"))
    object_dir = source_mesh.parent
    with np.load(resolve(dataset_root, record['camera_path']), allow_pickle=False) as camera:
        K = camera['K']
        sam_from_object = np.diag([-1., -1., 1., 1.]) @ camera['T_camera_from_object']
    with np.load(resolve(root, first['target_points_path']), allow_pickle=False) as data:
        evaluation_from_object = data['evaluation_normalization']
    evaluation_from_sam = evaluation_from_object @ np.linalg.inv(sam_from_object)
    image_path = resolve(dataset_root, record['image_path'])
    rgba = np.asarray(Image.open(image_path).convert('RGBA'))
    if record.get('pointmap_path'):
        points = np.load(resolve(dataset_root, record['pointmap_path']), allow_pickle=False)
    else:
        from data_generation.general.pointmaps import depth_to_pointmap
        points = depth_to_pointmap(np.load(resolve(dataset_root, record['depth_path']), allow_pickle=False), K)
    valid = np.isfinite(points).all(axis=-1) & (rgba[..., 3] > 0)
    if not valid.any():
        raise ValueError('Input pointmap has no valid foreground points')
    pointmap = {'name': 'input_pointmap', 'points': trimesh.transform_points(points[valid], evaluation_from_sam),
                'colors': rgba[..., :3][valid], 'spheres': True}
    mesh = {'name': 'input_mesh', 'mesh': load_mesh(resolve(root, first['target_mesh_path']))}
    surfaces = []
    descriptions = {}
    from dataloader import TouchDataset

    for condition, row in rows.items():
        settings = config['runs'].get(condition)
        if not settings or not settings.get('touch_config'):
            continue
        if settings['mode'] == 'image_touch_joint':
            raise ValueError('Joint pointmap/surface encoders need separate saved input indices; cannot label that cloud full surface')
        run_data = copy.deepcopy(settings['data'])
        if run_data.get('touch', {}).get('source') != 'full_surface':
            raise ValueError(f'{condition} is not a full-surface encoder run')
        dataset = TouchDataset(run_data, include_touch=True)
        run_record = next((item for item in dataset.records if item['sample_id'] == args.sample_id), None)
        if run_record is None:
            raise ValueError(f'{args.sample_id} missing from {condition} dataset')
        if dataset.surface_pool_count is not None:
            camera_points, _ = dataset.load_surface_pool(run_record)
        else:
            camera_points = dataset.load_full_surface(dataset.resolve_path(run_record['full_surface_path']))
        with np.load(resolve(root, row['stage1_path']), allow_pickle=False) as stage1:
            pre_encoder = stage1['touch_centers']
        if pre_encoder.shape != camera_points.shape:
            raise ValueError(f'{condition}: saved encoder input {pre_encoder.shape} differs from dataset {camera_points.shape}')
        encoder_name = settings['touch_config']['encoder_name']
        indices = surface_indices(pre_encoder, encoder_name, args.input_device)
        if not len(indices):
            raise ValueError(f'{condition}: empty surface input')
        print(f'{condition}: {encoder_name}, {len(indices):,} surface input points from {len(camera_points):,}', flush=True)
        # Physical coordinates for overlays, selected by the encoder's own sampling space.
        with np.load(dataset.resolve_path(run_record['camera_path']), allow_pickle=False) as camera:
            run_sam_from_object = np.diag([-1., -1., 1., 1.]) @ camera['T_camera_from_object']
        display = trimesh.transform_points(camera_points[indices], evaluation_from_object @ np.linalg.inv(run_sam_from_object))
        descriptions[condition] = {'encoder': encoder_name, 'source_points': len(camera_points),
                                   'encoder_valid_points': len(indices), 'selection': 'all' if len(indices) == len(pre_encoder) else 'encoder FPS',
                                   'coordinate_frame': 'evaluation_aligned'}
        cache_dir = root / 'inputs' / args.sample_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_dir / f'{condition}_surface.npz', points=display, source_indices=indices,
                            pre_encoder_points=pre_encoder[indices])
        for surface, members in surfaces:
            if np.array_equal(surface['points'], display):
                members.append(condition)
                break
        else:
            surfaces.append(({'name': 'input_surface', 'points': display,
                              'colors': np.broadcast_to(np.array([45, 125, 210], dtype=np.uint8), display.shape),
                              'spheres': True}, [condition]))
    if not surfaces:
        raise ValueError('Input surface variants require a completed full-surface encoder condition')
    if source_mesh.suffix.lower() == '.npz':
        textured_path = textured_mesh_path(object_dir, root / 'inputs' / record['object_id'], args.blender)
        textured_transform = evaluation_from_object
    else:
        # Original Objaverse orbit path: source OBJ -> normalized object -> evaluation frame.
        textured_path = source_mesh
        with np.load(resolve(dataset_root, record['object_transform_path']), allow_pickle=False) as data:
            textured_transform = evaluation_from_object @ data['T_normalized_from_source']
    textured = {'name': 'input_mesh_textured', 'mesh': mesh['mesh'], 'path': str(textured_path),
                'textured': True, 'transform': textured_transform}
    descriptions['pointmap'] = {'points': int(valid.sum()), 'source': 'foreground depth back-projected in SAM camera coordinates'}
    descriptions['surface_groups'] = [members for _, members in surfaces]
    descriptions['textured_mesh'] = str(textured_path)
    return input_groups(mesh, textured, pointmap, surfaces), [image_path], descriptions
