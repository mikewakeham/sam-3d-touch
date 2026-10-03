"""Thin fixed Zeroverse patches at inference through one trained SAM-3D-Touch model."""
import argparse
import copy
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from dataloader import TouchDataset, collate_touch_batch
from data_generation.general.sample_full_surface import sam_camera_transform, transform_points
from data_generation.general.sample_touch_patches import select_patch_indices
from evaluation.evaluate import (
    read_run, restore_run, build_pipeline, preprocess_stage2, sample_shape,
    decode_voxels, trimesh_from_result, load_target_mesh, stable_seed,
)
from evaluation.metrics import normalize_mesh, sample_surface, align_mesh, mesh_metrics
from experiments.vecsetx.point_count_sweep import POINTS_PER_PATCH, sha256, select_records, subset_indices
from sam3d_objects.pipeline.inference_utils import prune_sparse_structure, downsample_sparse_structure
from train import prepare_batch


METRICS = ('stage1_iou', 'chamfer', 'fscore_0.01', 'patch_distance_mean', 'patch_distance_p95')


def seed_generation(seed, generator):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    generator.random_generator.manual_seed(seed)


def touch_tokens(encoder, points, shift, scale):
    # Same encode/projection path as TouchEncoder.forward, without re-normalizing
    # or padding a smaller cloud. All conditions keep the baseline position term.
    previous_count = encoder.encoder.num_inputs
    encoder.encoder.num_inputs = points.shape[1]
    try:
        mask = torch.ones_like(points[..., 0], dtype=torch.bool)
        tokens = encoder.encoder.encode(points, mask)['x']
        if encoder.use_learn:
            tokens = encoder.encoder.learn(tokens)
        tokens = encoder.output_projection(tokens)
        if encoder.use_position:
            position_scale = scale.log() if encoder.position_scale == 'log' else scale
            tokens = tokens + encoder.position_projection(torch.cat((shift, position_scale), dim=-1))[:, None]
        return tokens + encoder.touch_embedding
    finally:
        encoder.encoder.num_inputs = previous_count


def summary(rows, seed_count, seed):
    # Average seeds within each object first. Missing seeds are never replaced
    # with successful seeds from another condition or silently averaged away.
    grouped = {}
    for row in rows:
        grouped.setdefault((row['points_per_patch'], row['sample_id']), {})[row['draw']] = row
    means = {}
    for key, draws in grouped.items():
        means[key] = {}
        if set(draws) != set(range(seed_count)):
            continue
        for metric in METRICS:
            values = [row.get(metric) for row in draws.values()]
            if all(value is not None and np.isfinite(value) for value in values):
                means[key][metric] = float(np.mean(values))
    result = {}
    for count in POINTS_PER_PATCH:
        attempted = [row for row in rows if row['points_per_patch'] == count]
        entry = dict(attempted=len(attempted), successful=sum(row['status'] == 'ok' for row in attempted),
                     failed=sum(row['status'] != 'ok' for row in attempted))
        for metric in METRICS:
            values, deltas = [], []
            for (points, sample_id), measurements in sorted(means.items()):
                if points != count or metric not in measurements:
                    continue
                values.append(measurements[metric])
                baseline = means.get((256, sample_id), {})
                if metric in baseline:
                    deltas.append(measurements[metric] - baseline[metric])
            delta = np.asarray(deltas)
            ci = None
            if len(delta):
                rng = np.random.default_rng(seed)
                bootstrap = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
                ci = np.quantile(bootstrap, [.025, .975]).tolist()
            entry[metric] = dict(
                complete_objects=len(values), paired_objects=len(delta),
                mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                mean_change_from_256=float(delta.mean()) if len(delta) else None,
                median_change_from_256=float(np.median(delta)) if len(delta) else None,
                mean_change_95ci=ci,
                fraction_worse_than_256=float(np.mean(delta < 0 if metric in ('stage1_iou', 'fscore_0.01') else delta > 0)) if len(delta) else None,
            )
        result[count] = entry
    return result


def reconstruct(pipeline, condition_args, condition_kwargs, tokens, stage2_inputs,
                target_voxels, target, patch_points, seed, metric_seed, output, args, row):
    device = torch.device(args.device)
    seed_generation(seed, pipeline.ss_generator)
    with torch.autocast(device.type, dtype=pipeline.shape_model_dtype, enabled=device.type == 'cuda' and not args.no_amp):
        prediction = sample_shape(pipeline, condition_args, condition_kwargs, tokens, args.inference_steps, args.device)
        voxels = decode_voxels(pipeline.models['ss_decoder'], prediction)
    intersection = (voxels & target_voxels).sum().item()
    union = (voxels | target_voxels).sum().item()
    row['stage1_iou'] = intersection / max(union, 1)
    # Save Stage 1 even when Stage 2 fails, and retain its independent metric.
    np.savez_compressed(output / 'stage1.npz', latent=prediction.float().cpu().numpy(),
                        prediction=voxels[0].cpu().numpy(), stage1_seed=np.int64(seed))
    coords = torch.argwhere(voxels).int()
    if not len(coords):
        raise ValueError('Stage 1 generated an empty support')
    if pipeline.downsample_ss_dist > 0:
        coords = prune_sparse_structure(coords, pipeline.downsample_ss_dist)
    coords, factor = downsample_sparse_structure(coords)
    seed_generation(seed + 1_000_000, pipeline.models['slat_generator'])
    slat = pipeline.sample_slat(stage2_inputs, coords, inference_steps=args.stage2_inference_steps, use_distillation=False)
    result = pipeline.decode_slat(slat, ['mesh'])['mesh'][0]
    if not result.success:
        raise ValueError('Stage-2 mesh decoder returned an empty mesh')
    raw = trimesh_from_result(result)
    raw.export(output / 'mesh_raw.ply')
    normalized, normalization = normalize_mesh(raw)
    icp_points, _ = sample_surface(normalized, args.icp_points, metric_seed + 1)
    aligned, alignment, error, fitness, rmse = align_mesh(
        normalized, target['mesh'], icp_points, target['icp_points'], args.metric_workers)
    aligned.export(output / 'mesh_aligned.ply')
    points, normals = sample_surface(aligned, args.surface_points, metric_seed + 2)
    measured = mesh_metrics(points, normals, target['points'], target['normals'],
                            0, args.device, args.metric_workers)
    local = cKDTree(points).query(patch_points, workers=args.metric_workers)[0]
    np.savez_compressed(output / 'alignment.npz', prediction_normalization=normalization,
                        icp_transform=alignment, icp_error=error, icp_fitness=fitness, icp_rmse=rmse,
                        stage2_seed=np.int64(seed + 1_000_000), metric_seed=np.int64(metric_seed),
                        downsample_factor=int(factor))
    row.update(chamfer=measured['chamfer'], **{'fscore_0.01': measured['fscore_0.01']},
               patch_distance_mean=float(local.mean()), patch_distance_p95=float(np.quantile(local, .95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/zeroverse/zeroverse_pointmap_touch_32x256/best.pt'))
    parser.add_argument('--pipeline-config', type=Path, default=Path('checkpoints/hf/pipeline.yaml'))
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/vecsetx/outputs/sam3d_point_count'))
    parser.add_argument('--objects', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--inference-steps', type=int, default=25)
    parser.add_argument('--stage2-inference-steps', type=int, default=25)
    parser.add_argument('--surface-points', type=int, default=1_000_000)
    parser.add_argument('--icp-points', type=int, default=20_000)
    parser.add_argument('--metric-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()
    if min(args.objects, args.seeds, args.inference_steps, args.stage2_inference_steps,
           args.surface_points, args.icp_points, args.metric_workers) < 1:
        raise ValueError('Counts must be positive')
    checkpoint, run_config, _ = read_run(args.checkpoint, split='train')
    data = copy.deepcopy(run_config['data'])
    data['dataset']['split'] = 'val'
    touch = data['touch']
    conditioning = checkpoint.get('conditioning_config', {})
    if (checkpoint['mode'] != 'image_touch' or checkpoint['touch_config']['encoder_name'] != 'vecsetx'
            or checkpoint.get('training_config', {}).get('constant_touch', False)
            or conditioning.get('oracle_point_frame', False) or conditioning.get('shared_pointmap_normalization', False)
            or touch.get('source') != 'touch_patches' or touch['contacts']['count'] != 32
            or touch['point_sampling']['points_per_contact'] != 256):
        raise ValueError('Use a separate-branch VecSetX checkpoint trained on 32x256 camera-frame patches')
    dataset = TouchDataset(data)
    splits = json.loads(dataset.resolve_path(data['dataset']['split_file']).read_text())
    if set(splits['train']) & set(splits['val']):
        raise ValueError('Training and validation object IDs overlap')
    dataset.records = select_records(dataset.records, args.objects, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'selection.json').write_text(json.dumps(dataset.records, indent=2) + '\n')
    for count in POINTS_PER_PATCH:
        np.save(args.output_dir / f'indices_{count}.npy', subset_indices(count))
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(data=data, run_config=run_config, points_per_patch=POINTS_PER_PATCH,
                    checkpoint_sha256=sha256(args.checkpoint),
                    run_config_sha256=sha256(args.checkpoint.parent / 'config.yaml'),
                    pipeline_sha256=sha256(args.pipeline_config),
                    manifest_sha256=sha256(dataset.resolve_path(data['dataset']['manifest'])),
                    split_sha256=sha256(dataset.resolve_path(data['dataset']['split_file'])),
                    normalization='fixed from 32x256 before subsetting; fixed position term if enabled',
                    stage1_metric='native occupancy IoU against decoded target latent, before pruning/registration',
                    mesh_metrics='source/prediction independently centered, longest extent 2; PCA similarity + ICP',
                    chamfer='symmetric mean unsquared Euclidean distance after registration', fscore_threshold=.01,
                    patch_metric='same 8192 original points in normalized source frame vs registered prediction',
                    summary='average all seeds per object, paired deltas, 2000 object bootstrap draws; missing seeds excluded per metric',
                    torch_version=torch.__version__, numpy_version=np.__version__)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    pipeline, pipeline_config = build_pipeline(args.pipeline_config, args.device)
    pipeline.ss_generator.no_shortcut = True
    model = restore_run(pipeline, checkpoint, args.device)
    model.requires_grad_(False)
    model.eval()
    pipeline.models.requires_grad_(False)
    for embedder in pipeline.condition_embedders.values():
        if embedder is not None:
            embedder.requires_grad_(False)
    encoder = model.touch_encoder
    assert encoder.encoder.query_type == 'learnable' and encoder.encoder.num_latents == 1024
    settings['pipeline'] = OmegaConf.to_container(pipeline_config, resolve=True)
    settings['shape_guidance'] = {key: getattr(pipeline.ss_generator.reverse_fn, key, None)
                                  for key in ('strength', 'interval')}
    (args.output_dir / 'settings.json').write_text(json.dumps(settings, indent=2, default=str) + '\n')
    del checkpoint
    rows = []

    with torch.inference_mode(), (args.output_dir / 'metrics.csv').open('w', newline='') as file:
        writer = None
        for index, record in enumerate(dataset.records):
            sample_id = record['sample_id']
            output = args.output_dir / sample_id
            output.mkdir()
            batch = collate_touch_batch([dataset[index]])
            raw = batch['touch_xyz'][0].numpy()
            with np.load(dataset.resolve_path(record['touch_path']), allow_pickle=False) as bank:
                bank_indices = select_patch_indices(bank, 32, 256)
                np.testing.assert_array_equal(raw, bank['points_camera'][bank_indices])
            _, condition_args, condition_kwargs, touch_xyz, touch_mask, inputs = prepare_batch(
                pipeline, batch, device, 'fp32' if args.no_amp else 'bf16', True, return_inputs=True)
            prepared, mask, shift, scale = encoder.prepare_points(touch_xyz, touch_mask)
            assert prepared.shape == (1, 8192, 3) and mask.all()
            tokens = {}
            with torch.autocast(device.type, dtype=pipeline.shape_model_dtype, enabled=device.type == 'cuda' and not args.no_amp):
                for count in POINTS_PER_PATCH:
                    tokens[count] = touch_tokens(encoder, prepared[:, subset_indices(count)], shift, scale)
                # Confirms the unthinned path retains the checkpoint's actual conditioning.
                torch.testing.assert_close(tokens[256], model.get_touch_tokens(touch_xyz, touch_mask), atol=1e-4, rtol=1e-4)
                target_voxels = decode_voxels(pipeline.models['ss_decoder'], batch['target_shape'].to(device))
            stage2_inputs = preprocess_stage2(pipeline, batch['image'])
            metric_seed = stable_seed(args.seed, sample_id) % (2**32 - 3)
            target = load_target_mesh(record, dataset, args.output_dir, args.surface_points,
                                      args.icp_points, args.surface_points, metric_seed)
            object_from_camera = np.linalg.inv(sam_camera_transform(dataset.resolve_path(record['camera_path'])))
            with np.load(target['points_path'], allow_pickle=False) as saved:
                target_normalization = saved['evaluation_normalization']
            patch_points = transform_points(raw, target_normalization @ object_from_camera)
            np.savez_compressed(output / 'inputs.npz', points_camera=raw, points_encoder=prepared[0].cpu().numpy(),
                                points_evaluation=patch_points, bank_indices=bank_indices,
                                shift=shift.cpu().numpy(), scale=scale.cpu().numpy(),
                                pointmap_scale=inputs['pointmap_scale'].cpu().numpy(),
                                pointmap_shift=inputs['pointmap_shift'].cpu().numpy(),
                                object_from_camera=object_from_camera, target_normalization=target_normalization)
            np.save(output / 'target_voxels.npy', target_voxels[0].cpu().numpy())
            sources = {key: sha256(dataset.resolve_path(record[key])) for key in
                       ('touch_path', 'mesh_path', 'camera_path', 'image_path', 'depth_path', 'target_path')}
            (output / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
            for draw in range(args.seeds):
                seed = stable_seed(args.seed, f'{sample_id}:{draw}')
                for count in POINTS_PER_PATCH:
                    destination = output / f'seed_{draw}' / str(count)
                    destination.mkdir(parents=True)
                    row = dict(sample_id=sample_id, object_id=record['object_id'], view_id=record['view_id'],
                               draw=draw, stage1_seed=seed, stage2_seed=seed + 1_000_000,
                               points_per_patch=count, total_points=32 * count, status='ok', error='',
                               **{metric: None for metric in METRICS})
                    try:
                        reconstruct(pipeline, condition_args, condition_kwargs, tokens[count], stage2_inputs,
                                    target_voxels, target, patch_points, seed, metric_seed, destination, args, row)
                    except Exception as error:
                        row.update(status='error', error=f'{type(error).__name__}: {error}')
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    file.flush()
                    print(f'[{index + 1}/{len(dataset)}] {sample_id} seed {draw} 32x{count}: {row}', flush=True)
            (args.output_dir / 'summary.json').write_text(json.dumps(summary(rows, args.seeds, args.seed), indent=2) + '\n')
    print(f'Saved {len(rows)} reconstruction attempts to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
