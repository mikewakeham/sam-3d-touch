"""Add adaptive touch banks to an existing rendered/encoded dataset."""

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import lru_cache

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import numpy as np
import trimesh

from generate_target_latents import checkpoint_sha256, load_normalized_mesh
from sample_full_surface import classify_visibility, sam_camera_transform, transform_points, validate_surface
from sample_touch_patches import make_patch_bank, save_patches, validate_patches, select_patch_indices


def initialize_worker():
    # The parent handles Ctrl+C and terminates the pool, including idle workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', default='generated_data/samples.jsonl')
    parser.add_argument('--name', default='simulated_touches', help='Name for the saved touches and manifest')
    parser.add_argument('--radius', type=float, default=.04, help='Maximum radius in normalized-object units')
    parser.add_argument('--thickness', type=float, default=.2, help='Normal/tangent radius ratio on flat surfaces')
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--object-id', action='append', help='Process only these objects (repeatable)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verify-existing', action='store_true', help='Fully validate saved arrays and source hashes when resuming')
    parser.add_argument('--joint-pointmap', action='store_true', help='Also save fixed 1024-pointmap + 7168-touch inputs for all three budgets')
    parser.add_argument('--pipeline-config', type=Path, default=Path('checkpoints/hf/pipeline.yaml'))
    args = parser.parse_args(argv)
    if (not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name)
            or args.workers < 1):
        parser.error('Use a simple alphanumeric/underscore/hyphen name and positive worker count')
    return args


def joint_arrays(bank, preprocessor, inputs):
    """Compute once during generation; training and the viewer only read these arrays."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    from train import normalize_touch_to_pointmap_frame
    from evaluation.input_visualizations import joint_camera_points
    from sam3d_objects.model.backbone.dit.embedder.surface_encoder_utils import farthest_point_indices

    with torch.no_grad():
        pointmap = inputs['pointmap'][0].permute(1, 2, 0).reshape(-1, 3)
        valid = (inputs['mask'][0].reshape(-1) > .5) & pointmap.isfinite().all(dim=-1)
        pointmap = pointmap[valid]
        if not len(pointmap):
            raise ValueError('No valid foreground pointmap points')
        chosen = farthest_point_indices(pointmap[None], min(1024, len(pointmap)))[0]
        chosen = chosen[torch.arange(1024, device=chosen.device) % len(chosen)]
        camera, normalized = [], []
        for contacts in (32, 16, 8):
            indices = select_patch_indices(bank, contacts, 7168 // contacts)
            touch = torch.as_tensor(bank['points_camera'][indices], device=pointmap.device)[None]
            mask = torch.ones(touch.shape[:2], dtype=torch.bool, device=pointmap.device)
            touch = normalize_touch_to_pointmap_frame(touch, mask, inputs, preprocessor)[0]
            cloud = torch.cat((pointmap[chosen], touch))
            restored = joint_camera_points(cloud, inputs, preprocessor)
            if not np.allclose(restored[1024:], bank['points_camera'][indices], atol=1e-5, rtol=1e-5):
                raise ValueError('Joint points do not round-trip to the saved touch points')
            camera.append(restored)
            normalized.append(cloud.float().cpu().numpy())
        return dict(joint_points_camera=np.array(camera, dtype=np.float32),
                    joint_points_pre_encoder=np.array(normalized, dtype=np.float32),
                    joint_pointmap_candidates_camera=joint_camera_points(pointmap, inputs, preprocessor),
                    joint_pointmap_indices=chosen.cpu().numpy(),
                    joint_pointmap_scale=inputs['pointmap_scale'][0].float().cpu().numpy(),
                    joint_pointmap_shift=inputs['pointmap_shift'][0].float().cpu().numpy())


@lru_cache(maxsize=1)
def joint_preprocessor(path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    from train import build_stage1_preprocessor
    torch.set_num_threads(1)
    return build_stage1_preprocessor(path)


def save_joint(args, record, arrays, path):
    if not args.joint_pointmap:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    from PIL import Image
    from dataloader import TouchDataset
    from train import preprocess_pointmap_batch
    from omegaconf import OmegaConf

    config_path = args.pipeline_config.resolve()
    config = OmegaConf.load(config_path)
    sources = [args.data_root / record[key] for key in ('image_path', 'camera_path', 'depth_path')]
    if record.get('pointmap_path'):
        sources.append(args.data_root / record['pointmap_path'])
    sources.append(config_path)
    if config.get('ss_preprocessor') is None:
        sources.append(config_path.parent / config.ss_generator_config_path)
    fingerprint = hashlib.sha256(''.join(checkpoint_sha256(source) for source in sources).encode()).hexdigest()
    if all(key in arrays for key in (
        'joint_points_camera', 'joint_points_pre_encoder', 'joint_pointmap_candidates_camera',
        'joint_pointmap_indices', 'joint_pointmap_scale', 'joint_pointmap_shift',
    )) and str(arrays.get('joint_source_sha256')) == fingerprint:
        return
    print(f"{record['sample_id']}: computing joint pointmap; keeping existing touch patches", flush=True)
    dataset = object.__new__(TouchDataset)
    dataset.root = args.data_root
    with Image.open(args.data_root / record['image_path']) as image:
        rgba = np.array(image.convert('RGBA'))
    pointmap = dataset.load_pointmap(record)
    preprocessor = joint_preprocessor(str(config_path))
    inputs = preprocess_pointmap_batch(preprocessor, torch.from_numpy(rgba)[None],
                                      torch.from_numpy(pointmap)[None], 'cpu')
    arrays.update(joint_arrays(arrays, preprocessor, inputs), joint_source_sha256=fingerprint)
    # Upgrade this file atomically while preserving the existing patch bank.
    save_patches(path, arrays, overwrite=True)


def make_object(task):
    args, records = task
    root = args.data_root
    mesh_paths = {record['mesh_path'] for record in records}
    if len(mesh_paths) != 1:
        raise ValueError('Views of an object must share one mesh')
    mesh_path = root / records[0]['mesh_path']
    mesh = None
    mesh_hash = None
    outputs = []
    reused = 0
    generated = 0
    for record in records:
        surface_path = root / record['full_surface_path']
        path = surface_path.with_name(f'{args.name}.npz')
        identity = int(hashlib.sha256(record['sample_id'].encode()).hexdigest()[:8], 16)
        seed = [args.seed, identity]
        if path.exists() and not args.overwrite:
            with np.load(path, allow_pickle=False) as saved:
                if (float(saved['radius']) != args.radius or float(saved['thickness']) != args.thickness
                        or not np.array_equal(saved['sample_seed_parts'], seed)):
                    raise ValueError(f'Existing touch settings differ: {path}; use --overwrite')
                if not args.verify_existing and (not args.joint_pointmap or all(key in saved.files for key in (
                    'joint_points_camera', 'joint_points_pre_encoder', 'joint_pointmap_candidates_camera',
                    'joint_pointmap_indices', 'joint_pointmap_scale', 'joint_pointmap_shift', 'joint_source_sha256',
                ))):
                    outputs.append(dict(record, touch_path=str(path.relative_to(root))))
                    reused += 1
                    continue
                arrays = dict(saved)
            validate_patches(arrays)
            if mesh_hash is None:
                mesh_hash = checkpoint_sha256(mesh_path)
            if (str(arrays['mesh_sha256']) != mesh_hash
                    or str(arrays['source_surface_sha256']) != checkpoint_sha256(surface_path)):
                raise ValueError(f'Existing touch settings or source differ: {path}; use --overwrite')
            save_joint(args, record, arrays, path)
            outputs.append(dict(record, touch_path=str(path.relative_to(root))))
            reused += 1
            continue
        print(f"{record['sample_id']}: generating touch patches", flush=True)
        if mesh_hash is None:
            mesh_hash = checkpoint_sha256(mesh_path)
        if mesh is None:
            print(f"{record['object_id']}: loading mesh", flush=True)
            mesh = load_normalized_mesh(mesh_path)
            print(f"{record['object_id']}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces", flush=True)
        with np.load(surface_path, allow_pickle=False) as saved:
            surface = dict(saved)
        validate_surface(surface)
        transform = sam_camera_transform(root / record['camera_path'])
        # Reuse backfill_normals' exact replay. No normals are guessed or written back.
        points, faces = trimesh.sample.sample_surface(mesh, int(surface['requested_point_count']),
                                                      seed=int(surface['sample_seed']))
        replayed = transform_points(points, transform).astype(np.float32)
        if replayed.shape != surface['points_camera'].shape or not np.allclose(
                replayed, surface['points_camera'], atol=1e-6, rtol=0):
            raise ValueError(f'Sampling replay differs: {surface_path}; use the original NumPy/Trimesh versions')
        if ('mesh_sha256' in surface and str(surface['mesh_sha256']) != mesh_hash
                or 'face_indices' in surface and not np.array_equal(surface['face_indices'], faces)):
            raise ValueError(f'Saved surface does not match original mesh: {surface_path}')
        with np.load(root / record['camera_path'], allow_pickle=False) as saved:
            camera = dict(saved)
        depth = np.load(root / record['depth_path'], allow_pickle=False)
        tolerance = float(surface['visibility_tolerance'])
        visibility = classify_visibility(points, camera['K'], camera['T_camera_from_object'], depth, tolerance)
        if not np.array_equal(visibility, surface['point_visibility']):
            raise ValueError(f'Saved surface visibility differs: {surface_path}')
        identity = int(hashlib.sha256(record['sample_id'].encode()).hexdigest()[:8], 16)
        seed = [args.seed, identity]
        bank = make_patch_bank(mesh, points, faces, visibility, seed, args.radius, args.thickness)
        centers = bank['center_indices']
        arrays = dict(
            format_version=np.int64(1), data_kind='touch_patches', coordinate_frame='sam_camera',
            sampling_method='area_weighted_triangle_ellipsoid', center_method='hidden_euclidean_fps',
            points_camera=transform_points(bank['points_object'], transform).astype(np.float32),
            offsets=bank['offsets'], face_indices=bank['face_indices'],
            center_point_ids=surface['point_ids'][centers],
            bases_camera=np.einsum('ij,njk->nik', transform[:3, :3], bank['bases_object']).astype(np.float32),
            radii=bank['radii'].astype(np.float32), support_counts=bank['support_counts'],
            point_visibility=classify_visibility(bank['points_object'], camera['K'], camera['T_camera_from_object'], depth, tolerance),
            visibility_tolerance=tolerance, radius=args.radius, thickness=args.thickness,
            sample_seed_parts=np.array(seed, dtype=np.int64), mesh_sha256=mesh_hash,
            source_surface_sha256=checkpoint_sha256(surface_path),
            numpy_version=np.__version__, trimesh_version=trimesh.__version__,
        )
        path = surface_path.with_name(f'{args.name}.npz')
        save_patches(path, arrays, args.overwrite)
        save_joint(args, record, arrays, path)
        outputs.append(dict(record, touch_path=str(path.relative_to(root))))
        generated += 1
    return outputs, reused, generated


def main(argv=None):
    args = parse_args(argv)
    args.data_root = args.data_root.resolve()
    generated = args.data_root / 'generated_data'
    manifest = args.data_root / args.manifest
    output = generated / f'samples_{args.name}.jsonl'
    if output.resolve() == manifest.resolve():
        raise ValueError('Input and output manifests must differ')
    with (generated / '.build.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'Dataset is locked by another generation process: {generated}. '
                             'Stop that process before restarting; do not delete .build.lock.')
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        groups = {}
        for record in records:
            if args.object_id is None or record['object_id'] in args.object_id:
                groups.setdefault(record['object_id'], []).append(record)
        if not groups or args.object_id and set(args.object_id) - set(groups):
            raise ValueError('Requested objects are absent from the input manifest')
        tasks = [(args, rows) for rows in groups.values()]
        print(f'Processing {len(tasks)} objects with {args.workers} workers.', flush=True)
        if args.overwrite:
            print('Overwrite enabled: regenerating existing touch files.', flush=True)
        else:
            print('Reusing existing touch files and generating missing ones. Progress counts both.', flush=True)
            if not args.verify_existing:
                print('Resume checks saved file headers and settings only; --verify-existing enables full source checks.', flush=True)
        completed = []
        if args.workers == 1:
            for index, task in enumerate(tasks, 1):
                rows, reused, new_views = make_object(task)
                completed.extend(rows)
                print(f"[{index}/{len(tasks)}] {rows[0]['object_id']}: {reused} views reused, {new_views} generated", flush=True)
        else:
            # Match make_data.py's spawn context: workers must not inherit the dataset lock.
            pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                       initializer=initialize_worker)
            remaining = set(groups)
            try:
                futures = {pool.submit(make_object, task): task[1][0]['object_id'] for task in tasks}
                for index, future in enumerate(as_completed(futures), 1):
                    object_id = futures[future]
                    try:
                        rows, reused, new_views = future.result()
                    except BrokenProcessPool as error:
                        raise RuntimeError(
                            f'Touch worker exited unexpectedly. {len(remaining)} objects have no returned result; '
                            f'first IDs: {sorted(remaining)[:20]}. Completed touch files are preserved; '
                            'rerun with --workers 1 to locate the failing object.'
                        ) from error
                    except Exception as error:
                        raise RuntimeError(f'Touch generation failed for {object_id}') from error
                    completed.extend(rows)
                    remaining.remove(object_id)
                    print(f"[{index}/{len(tasks)}] {rows[0]['object_id']}: {reused} views reused, {new_views} generated", flush=True)
            finally:
                previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    if sys.exc_info()[0] is not None:
                        # Python 3.11 has no public executor method to terminate running workers.
                        for process in list((pool._processes or {}).values()):
                            if process.is_alive():
                                process.terminate()
                    pool.shutdown(wait=True, cancel_futures=True)
                finally:
                    signal.signal(signal.SIGINT, previous_handler)
        # A pilot subset must never replace a previously complete training manifest.
        if args.object_id is not None:
            print(f'Saved simulated touches for {len(completed)} pilot views.', flush=True)
            return
        by_id = {row['sample_id']: row for row in completed}
        if len(by_id) != len(records):
            raise ValueError('Duplicate or missing sample IDs')
        content = ''.join(json.dumps(by_id[row['sample_id']]) + '\n' for row in records)
        if output.exists() and output.read_text() != content and not args.overwrite:
            raise ValueError(f'Existing manifest differs: {output}; use --overwrite')
        temporary = output.with_suffix('.tmp.jsonl')
        temporary.write_text(content)
        temporary.replace(output)
        print(f'Wrote {output} ({len(records)} views)', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Finished touch files are preserved; rerun the same command to continue.', flush=True)
        raise SystemExit(130)
