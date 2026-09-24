"""Add adaptive touch banks to an existing rendered/encoded dataset."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from generate_target_latents import checkpoint_sha256, load_normalized_mesh
from sample_full_surface import classify_visibility, sam_camera_transform, transform_points, validate_surface
from sample_touch_patches import make_patch_bank, save_patches


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', default='generated_data/samples.jsonl')
    parser.add_argument('--name', default='adaptive_v1', help='Suffix for the new NPZ files and manifest')
    parser.add_argument('--radius', type=float, default=.04, help='Maximum radius in normalized-object units')
    parser.add_argument('--thickness', type=float, default=.2, help='Normal/tangent radius ratio on flat surfaces')
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--object-id', action='append', help='Process only these objects (repeatable)')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if (not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name)
            or args.workers < 1):
        parser.error('Use a simple alphanumeric/underscore/hyphen name and positive worker count')
    return args


def make_object(task):
    args, records = task
    root = args.data_root
    mesh_paths = {record['mesh_path'] for record in records}
    if len(mesh_paths) != 1:
        raise ValueError('Views of an object must share one mesh')
    mesh_path = root / records[0]['mesh_path']
    mesh = load_normalized_mesh(mesh_path)
    mesh_hash = checkpoint_sha256(mesh_path)
    outputs = []
    for record in records:
        surface_path = root / record['full_surface_path']
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
        path = surface_path.with_name(f'touches_{args.name}.npz')
        save_patches(path, arrays, args.overwrite)
        outputs.append(dict(record, touch_path=str(path.relative_to(root))))
    return outputs


def main(argv=None):
    args = parse_args(argv)
    args.data_root = args.data_root.resolve()
    generated = args.data_root / 'generated_data'
    manifest = args.data_root / args.manifest
    output = generated / f'samples_touch_{args.name}.jsonl'
    if output.resolve() == manifest.resolve():
        raise ValueError('Input and output manifests must differ')
    with (generated / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        groups = {}
        for record in records:
            if args.object_id is None or record['object_id'] in args.object_id:
                groups.setdefault(record['object_id'], []).append(record)
        if not groups or args.object_id and set(args.object_id) - set(groups):
            raise ValueError('Requested objects are absent from the input manifest')
        tasks = [(args, rows) for rows in groups.values()]
        completed = []
        if args.workers == 1:
            for task in tasks:
                rows = make_object(task)
                completed.extend(rows)
                print(f"{rows[0]['object_id']}: {len(rows)} views", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for rows in pool.map(make_object, tasks):
                    completed.extend(rows)
                    print(f"{rows[0]['object_id']}: {len(rows)} views", flush=True)
        # A pilot subset must never replace a previously complete training manifest.
        if args.object_id is not None:
            print(f'Wrote {len(completed)} pilot views; full manifest is published by an unfiltered run.', flush=True)
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
    main()
