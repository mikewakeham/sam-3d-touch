"""Reconstruct the same Zeroverse training patches with progressively fewer points."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from dataloader import TouchDataset, load_data_config
from data_generation.general.sample_full_surface import sam_camera_transform, transform_points
from data_generation.general.sample_touch_patches import select_patch_indices
from evaluation.geometry import load_dataset_mesh
from evaluation.input_visualizations import joint_camera_points
from experiments.vecsetx.reconstruct import make_grid, encode_and_decode, make_mesh
from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
from train import amp, build_stage1_preprocessor, preprocess_pointmap_batch, normalize_touch_to_pointmap_frame


POINTS_PER_PATCH = (256, 128, 64, 32)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_records(records, count, seed):
    grouped = {}
    for record in records:
        grouped.setdefault(record['object_id'], []).append(record)
    objects = sorted(grouped)
    random.Random(seed).shuffle(objects)
    if not 0 < count <= len(objects):
        raise ValueError(f'Requested {count} objects; found {len(objects)}')
    selected = []
    for object_id in sorted(objects[:count]):
        views = sorted(grouped[object_id], key=lambda row: row['sample_id'])
        selected.append(random.Random(f'{seed}:{object_id}').choice(views))
    return selected


def subset_indices(points_per_patch, patch_count=32, baseline_points_per_patch=256, prefix_points=0):
    # Keep a fixed prefix (the joint pointmap, when present), then nested patches.
    patches = (np.arange(patch_count)[:, None] * baseline_points_per_patch
               + np.arange(points_per_patch)).ravel()
    return np.concatenate((np.arange(prefix_points), prefix_points + patches))


def metrics(mesh, reference_points, patch_points, count, seed):
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    tree = cKDTree(points)
    forward = cKDTree(reference_points).query(points)[0]
    backward = tree.query(reference_points)[0]
    precision, recall = float(np.mean(forward < .01)), float(np.mean(backward < .01))
    local = tree.query(patch_points)[0]
    return {
        'chamfer': float((forward.mean() + backward.mean()) / 2),
        'fscore_0.01': 2 * precision * recall / max(precision + recall, 1e-12),
        'patch_distance_mean': float(local.mean()),
        'patch_distance_p95': float(np.quantile(local, .95)),
    }


def save_summary(rows, output):
    names = ('chamfer', 'fscore_0.01', 'patch_distance_mean', 'patch_distance_p95')
    baseline = {row['sample_id']: row for row in rows
                if row['points_per_patch'] == 256 and row['status'] == 'ok'}
    summary = {}
    for count in POINTS_PER_PATCH:
        attempted = [row for row in rows if row['points_per_patch'] == count]
        valid = [row for row in attempted if row['status'] == 'ok']
        paired = [row for row in valid if row['sample_id'] in baseline]
        entry = {'attempted': len(attempted), 'successful': len(valid),
                 'failed': len(attempted) - len(valid), 'paired_with_baseline': len(paired)}
        for name in names:
            values = [row[name] for row in valid]
            delta = np.array([row[name] - baseline[row['sample_id']][name] for row in paired])
            entry[name] = {
                'mean': float(np.mean(values)) if values else None,
                'median': float(np.median(values)) if values else None,
                'mean_change_from_256': float(delta.mean()) if len(delta) else None,
                'fraction_worse_than_256': float(np.mean(delta < 0 if name == 'fscore_0.01' else delta > 0)) if len(delta) else None,
            }
        summary[count] = entry
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', type=Path, default=Path('configs/data_zeroverse_5000_8views_touch_32x256.yaml'))
    parser.add_argument('--pipeline-config', type=Path, default=Path('checkpoints/hf/pipeline.yaml'))
    parser.add_argument('--encoder-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/vecsetx/outputs/point_count'))
    parser.add_argument('--objects', type=int, default=100)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--metric-points', type=int, default=32768)
    parser.add_argument('--block-size', type=int, default=100000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    args = parser.parse_args()
    if min(args.objects, args.resolution, args.metric_points, args.block_size) < 1:
        raise ValueError('Counts must be positive')
    config = load_data_config(args.data_config)
    touch = config['touch']
    if (config['dataset']['split'] != 'train' or touch['source'] != 'touch_patches'
            or touch['contacts']['count'] != 32 or touch['point_sampling']['points_per_contact'] != 256):
        raise ValueError('Use the training config with 32 patches and 256 points per patch')
    dataset = TouchDataset(config)
    dataset.records = select_records(dataset.records, args.objects, args.seed)
    # Each submission gets a separate output directory; do not mix runs.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'selection.json').write_text(json.dumps(dataset.records, indent=2) + '\n')
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(data=config, points_per_patch=POINTS_PER_PATCH,
                    encoder_sha256=sha256(args.encoder_checkpoint),
                    pipeline_sha256=sha256(args.pipeline_config),
                    manifest_sha256=sha256(dataset.resolve_path(config['dataset']['manifest'])),
                    split_sha256=sha256(dataset.resolve_path(config['dataset']['split_file'])),
                    normalization='fixed from the exact 32x256 training input, before subsetting',
                    metric_frame='normalized_object; no registration or independent mesh normalization',
                    chamfer='symmetric mean unsquared Euclidean distance',
                    fscore_threshold=.01, local_reference='all 8192 baseline patch points at every level',
                    torch_version=torch.__version__, numpy_version=np.__version__)
    (args.output_dir / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # SSI uses matrix products for coordinates. TF32 ('high') can break the
    # forward/inverse round trip at ~1e-3; keep geometry in full float32.
    # The encoder/decoder still use the requested bf16 autocast below.
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    preprocessor = build_stage1_preprocessor(args.pipeline_config)
    encoder = TouchEncoder(encoder_checkpoint=args.encoder_checkpoint, use_position=False).to(device).eval()
    encoder.requires_grad_(False)
    model = encoder.encoder
    assert model.query_type == 'learnable' and model.num_latents == 1024
    grid = make_grid(args.resolution, device)
    rows = []

    with torch.inference_mode(), (args.output_dir / 'metrics.csv').open('w', newline='') as file:
        writer = None
        for index, record in enumerate(dataset.records):
            sample = dataset[index]
            output = args.output_dir / record['sample_id']
            output.mkdir()
            raw = sample['touch_xyz'].numpy()
            assert raw.shape == (8192, 3)
            with np.load(dataset.resolve_path(record['touch_path']), allow_pickle=False) as bank:
                bank_indices = select_patch_indices(bank, 32, 256)
                np.testing.assert_array_equal(raw, bank['points_camera'][bank_indices])
            inputs = preprocess_pointmap_batch(preprocessor, sample['image'][None], sample['pointmap'][None], device)
            mask = torch.ones((1, 8192), dtype=torch.bool, device=device)
            points = normalize_touch_to_pointmap_frame(sample['touch_xyz'][None].to(device), mask, inputs, preprocessor)
            # Normalize exactly once, as in training; no FPS at this 8192-point baseline.
            prepared, prepared_mask, shift, scale = encoder.prepare_points(points, mask)
            assert prepared.shape == (1, 8192, 3) and prepared_mask.all()
            object_from_camera = np.linalg.inv(sam_camera_transform(dataset.resolve_path(record['camera_path'])))

            def to_object(vertices):
                vertices = torch.as_tensor(np.asarray(vertices), dtype=torch.float32, device=device)
                vertices = vertices / scale[0] + shift[0]
                return transform_points(joint_camera_points(vertices, inputs, preprocessor), object_from_camera)

            patch_points = transform_points(raw, object_from_camera)
            np.testing.assert_allclose(to_object(prepared[0].cpu().numpy()), patch_points, atol=1e-5, rtol=1e-5)
            np.savez_compressed(output / 'inputs.npz', points_camera=raw,
                                points_encoder=prepared[0].cpu().numpy(), points_object=patch_points,
                                bank_indices=bank_indices, shift=shift.cpu().numpy(), scale=scale.cpu().numpy(),
                                pointmap_scale=inputs['pointmap_scale'].cpu().numpy(),
                                pointmap_shift=inputs['pointmap_shift'].cpu().numpy(),
                                object_from_camera=object_from_camera)
            source_hashes = {key: sha256(dataset.resolve_path(record[key])) for key in
                             ('touch_path', 'mesh_path', 'camera_path', 'image_path', 'depth_path')}
            (output / 'sources.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
            reference, _ = load_dataset_mesh(record, dataset)
            if not np.isclose(reference.extents.max(), 1, atol=1e-4):
                raise ValueError('Expected a unit-extent source mesh for the fixed .01 threshold')
            reference.export(output / 'reference.ply')
            metric_seed = int(hashlib.sha256(f'{args.seed}:{record["sample_id"]}'.encode()).hexdigest()[:8], 16)
            reference_points, _ = trimesh.sample.sample_surface(reference, args.metric_points, seed=metric_seed)
            np.save(output / 'reference_points.npy', reference_points)
            previous = np.arange(8192)
            for count in POINTS_PER_PATCH:
                selected = subset_indices(count)
                assert np.isin(selected, previous).all()
                assert np.array_equal(selected.reshape(32, count)[:, 0], np.arange(32) * 256)
                previous = selected
                np.save(output / f'indices_{count}.npy', selected)
                cloud = prepared[:, selected]
                # num_inputs is only an input-length assertion, not a learned parameter.
                model.num_inputs = cloud.shape[1]
                with amp(device, args.precision):
                    _, _, sdf = encode_and_decode(model, cloud, torch.ones_like(cloud[..., 0], dtype=torch.bool), grid, args.block_size)
                row = dict(sample_id=record['sample_id'], object_id=record['object_id'], view_id=record['view_id'],
                           points_per_patch=count, total_points=cloud.shape[1], status='ok',
                           chamfer=None, **{'fscore_0.01': None}, patch_distance_mean=None, patch_distance_p95=None)
                if not torch.isfinite(sdf).all():
                    row['status'] = 'nonfinite_sdf'
                else:
                    mesh, _, _ = make_mesh(sdf[0], args.resolution)
                    if mesh is None or mesh.is_empty:
                        row['status'] = 'no_surface'
                    else:
                        mesh.vertices = to_object(mesh.vertices)
                        mesh.export(output / f'reconstruction_{count}.ply')
                        row.update(metrics(mesh, reference_points, patch_points, args.metric_points, metric_seed))
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                file.flush()
                print(f'[{index + 1}/{len(dataset)}] {record["sample_id"]} 32x{count}: {row}', flush=True)
            save_summary(rows, args.output_dir)
    print(f'Saved {len(rows)} reconstructions/attempts to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
