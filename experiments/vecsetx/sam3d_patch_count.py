"""Remove whole patches from 32-patch inputs using joint or full-surface checkpoints."""
import argparse
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from dataloader import TouchDataset, build_dataloader, load_data_config
from evaluation.evaluate import (
    read_run, restore_run, build_pipeline, select_records, stable_seed,
    load_target_mesh, evaluate_condition, file_digest, file_identity, write_metrics, override_data_root,
)


PATCH_COUNTS = (32, 24, 16, 12, 8, 4, 1)
METRICS = ('stage1_iou', 'chamfer', 'fscore_0.01', 'normal_consistency', 'voxel_iou_64', 'emd',
           'pose_box_iou_3d', 'pose_add_s', 'pose_icp_rot_deg')
LOWER_IS_BETTER = ('chamfer', 'emd', 'pose_add_s', 'pose_icp_rot_deg')


def subset_indices(points_per_patch, patch_count=32, *, joint_pointmap=True):
    stride = 224 if joint_pointmap else 256
    if not 1 <= patch_count <= 32 or not 1 <= points_per_patch <= stride:
        raise ValueError(f'Expected 1–32 patches with 1–{stride} points per patch')
    patches = (np.arange(patch_count)[:, None] * stride + np.arange(points_per_patch)).ravel()
    return np.concatenate((np.arange(1024), 1024 + patches)) if joint_pointmap else patches


def touch_tokens(encoder, points, shift, scale):
    """Use the checkpoint's encoding path without padding or re-normalizing."""
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


def encode_patches(model, count, points, mask, normalization='full', joint_pointmap=True):
    # Keep the exact ordinary evaluation path for the full-input baseline.
    if count == 32:
        return model.get_touch_tokens(points, mask)
    encoder = model.touch_encoder
    indices = subset_indices(224 if joint_pointmap else 256, count, joint_pointmap=joint_pointmap)
    if normalization == 'retained':
        # Normalize only the actual input, without prepare_points padding/FPS.
        prepared, shift, scale = encoder.normalize_points_for_vecsetx(points[:, indices], mask[:, indices])
        return touch_tokens(encoder, prepared, shift, scale)
    prepared, valid, shift, scale = encoder.prepare_points(points, mask)
    assert prepared.shape == (1, 8192, 3) and valid.all()
    # Normalize all 32 patches (plus pointmap for joint models) once, then subset.
    # Do not pad or resample the retained points.
    return touch_tokens(encoder, prepared[:, indices], shift, scale)


def check_reference(args, selected, selection_data):
    reference = args.reference_evaluation
    if reference is None:
        return
    selection_path = reference / 'selected_samples.json'
    saved = (json.loads(selection_path.read_text()) if selection_path.exists()
             else yaml.safe_load((reference / 'selected_samples.yaml').read_text()))
    keys = ('sample_id', 'object_id', 'view_id')
    if [tuple(row[key] for key in keys) for row in saved] != [tuple(row[key] for key in keys) for row in selected]:
        raise ValueError('Selected objects/views differ from the reference evaluation')
    settings = json.loads((reference / 'evaluation_settings.json').read_text())
    for key in ('split', 'selection', 'max_samples', 'selection_seed', 'seed', 'inference_steps',
                'stage2_inference_steps', 'surface_points', 'icp_points', 'emd_points', 'save_points', 'no_amp'):
        if hasattr(args, key) and settings[key] != getattr(args, key):
            raise ValueError(f'{key} differs from the reference evaluation: {settings[key]!r}')
    if settings['pipeline'] != file_identity(args.pipeline_config):
        raise ValueError('Pipeline config differs from the reference evaluation')
    if settings['selection_data'] != selection_data:
        raise ValueError('Selection data config differs from the reference evaluation')
    for identity in settings['dataset_files']:
        if file_identity(identity['path']) != identity:
            raise ValueError(f'Reference dataset file changed: {identity["path"]}')
    return settings


def check_reference_run(settings, checkpoint_hash, config_hash, data, surface_patches=False):
    if settings is None:
        return None
    matches = [(name, entry) for name, entry in settings.get('entries', {}).items()
               if entry['checkpoint_sha256'] == checkpoint_hash and entry['config_sha256'] == config_hash]
    if len(matches) != 1:
        raise ValueError('Checkpoint/config is not uniquely registered in the reference evaluation; '
                         'evaluate this checkpoint normally first')
    name, entry = matches[0]
    reference_data = deepcopy((entry.get('metadata') or {}).get('data'))
    if surface_patches and reference_data is not None:
        if reference_data['touch']['source'] != 'full_surface':
            raise ValueError('Surface checkpoint reference must use full_surface data')
        # The sweep replaces the surface observation with the saved patch bank.
        # Keep all other evaluation dataset settings, including split and frame.
        reference_data['touch'] = data['touch']
        reference_data['dataset']['manifest'] = data['dataset']['manifest']
    if reference_data != data:
        raise ValueError('Touch data config differs from this checkpoint\'s reference evaluation')
    return name


def check_selected_inputs(selected, selection_dataset, records, dataset, selection_data, data):
    # The reference can select canonical targets while this model uses camera-frame
    # targets. Images, geometry and camera/depth observations must still match.
    keys = ['mesh_path', 'image_path', 'camera_path', 'depth_path']
    if selection_data['dataset'].get('target_frame', 'canonical') == data['dataset'].get('target_frame', 'canonical'):
        keys.append('target_path')
    for record in selected:
        if record['sample_id'] not in records:
            raise ValueError(f'Touch manifest is missing selected sample {record["sample_id"]}')
        if any(record[key] != records[record['sample_id']][key] for key in ('object_id', 'view_id')):
            raise ValueError('Selection and touch manifests disagree on object/view identity')
        for key in keys:
            if selection_dataset.resolve_path(record[key]) != dataset.resolve_path(records[record['sample_id']][key]):
                raise ValueError(f'Selection and touch manifests disagree on {key}')


def summarize(rows, seed, group_key='patch_count', groups=PATCH_COUNTS, reference=32):
    baseline = {row['sample_id']: row for row in rows if row[group_key] == reference}
    result = {}
    for count in groups:
        attempted = [row for row in rows if row[group_key] == count]
        valid = [row for row in attempted if not row['error']]
        entry = dict(attempted=len(attempted), successful=len(valid), failed=len(attempted) - len(valid))
        entry['pose_failed'] = sum(bool(row.get('pose_error')) for row in attempted)
        # Report failures explicitly rather than letting success-only means hide them.
        entry['fscore_0.01_failures_as_zero'] = (
            float(np.mean([row.get('fscore_0.01', 0.) if not row['error'] else 0. for row in attempted]))
            if attempted else None)
        for metric in METRICS:
            measured = attempted if metric == 'stage1_iou' else valid
            if metric.startswith('pose_'):
                measured = [row for row in measured if not row.get('pose_error')]
            values = [row[metric] for row in measured if np.isfinite(row.get(metric, np.nan))]
            delta = np.asarray([row[metric] - baseline[row['sample_id']][metric] for row in measured
                                if row['sample_id'] in baseline and np.isfinite(row.get(metric, np.nan))
                                and (metric == 'stage1_iou' or not baseline[row['sample_id']]['error'])
                                and (not metric.startswith('pose_') or not baseline[row['sample_id']].get('pose_error'))
                                and np.isfinite(baseline[row['sample_id']].get(metric, np.nan))])
            ci = None
            if len(delta):
                bootstrap = np.random.default_rng(seed).choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
                ci = np.quantile(bootstrap, [.025, .975]).tolist()
            entry[metric] = dict(
                complete_objects=len(values), paired_objects=len(delta),
                mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                mean_change_95ci=ci,
                **{f'mean_change_from_{reference}': float(delta.mean()) if len(delta) else None,
                   f'median_change_from_{reference}': float(np.median(delta)) if len(delta) else None,
                   f'fraction_worse_than_{reference}': float(np.mean(delta > 0 if metric in LOWER_IS_BETTER else delta < 0)) if len(delta) else None},
            )
        result[count] = entry
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/zeroverse/zeroverse_pointmap_touch_32x256_joint_sampled_camframe/best.pt'))
    parser.add_argument('--data-config', type=Path)
    parser.add_argument('--data-root', type=Path, help='Override both evaluation and selection dataset roots')
    parser.add_argument('--selection-data-config', type=Path, required=True)
    parser.add_argument('--reference-evaluation', type=Path, help='Verify selection, protocol and checkpoint against an existing main evaluation')
    parser.add_argument('--pipeline-config', type=Path, default=Path('checkpoints/hf/pipeline.yaml'))
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/vecsetx/outputs/sam3d_patch_count_sampled_camframe_full'))
    parser.add_argument('--split', default='val')
    parser.add_argument('--max-samples', type=int, default=100, help='Objects; 0 selects all, as in the main evaluator')
    parser.add_argument('--selection', choices=['random', 'hidden'], default='random')
    parser.add_argument('--selection-seed', type=int, default=29)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--normalization-probe', type=int, default=0, metavar='N',
                        help='Evaluate full vs retained normalization at 16 patches on N random objects from the reference selection')
    parser.add_argument('--preflight-only', action='store_true',
                        help='Validate checkpoint, data and reference selection without loading the GPU pipeline or writing outputs')
    parser.add_argument('--smoke-samples', type=int, default=0, metavar='N',
                        help='After validating the full reference selection, run all patch counts on its first N samples; use a separate output directory')
    parser.add_argument('--inference-steps', type=int, default=25)
    parser.add_argument('--stage2-inference-steps', type=int, default=25)
    parser.add_argument('--surface-points', type=int, default=1_000_000)
    parser.add_argument('--icp-points', type=int, default=20_000)
    parser.add_argument('--emd-points', type=int, default=2048)
    parser.add_argument('--save-points', type=int, default=8192)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--metric-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()
    if min(args.surface_points, args.icp_points, args.save_points, args.inference_steps,
           args.stage2_inference_steps, args.metric_workers) < 1 or min(args.max_samples, args.emd_points, args.workers, args.normalization_probe, args.smoke_samples) < 0:
        raise ValueError('Invalid sample counts or worker count')
    if args.smoke_samples and args.normalization_probe:
        raise ValueError('Choose either smoke samples or a normalization probe')
    return args


def main():
    args = parse_args()
    checkpoint, run_config, data = read_run(args.checkpoint, args.data_config, args.split)
    data = override_data_root(data, args.data_root)
    if run_config['data']['dataset'].get('target_frame', 'canonical') != data['dataset'].get('target_frame', 'canonical'):
        raise ValueError('Training and evaluation target frames must match')
    conditioning = checkpoint.get('conditioning_config', {})
    joint_pointmap = checkpoint['mode'] == 'image_touch_joint'
    if (checkpoint['mode'] not in ('image_touch_joint', 'image_touch')
            or (joint_pointmap and conditioning.get('joint_pointmap_points') != 1024)
            or checkpoint['touch_config']['encoder_name'] != 'vecsetx'
            or checkpoint.get('training_config', {}).get('constant_touch', False)
            or conditioning.get('oracle_point_frame', False) or conditioning.get('shared_pointmap_normalization', False)):
        raise ValueError('Use a 32-patch joint or full-surface VecSetX checkpoint')
    training_touch = run_config['data']['touch']
    if not joint_pointmap and training_touch['source'] != 'full_surface':
        raise ValueError('Non-joint checkpoints must be trained on full_surface')
    for config in ((run_config['data'], data) if joint_pointmap else (data,)):
        touch = config['touch']
        if (touch['source'] != 'touch_patches' or touch['contacts']['count'] != 32
                or touch['point_sampling']['points_per_contact'] != 256):
            raise ValueError('Patch evaluation data must use 32x256 patches')
    points_per_patch = 224 if joint_pointmap else 256
    pointmap_points = 1024 if joint_pointmap else 0

    selection_data = override_data_root(load_data_config(args.selection_data_config), args.data_root)
    selection_data['dataset']['split'] = args.split
    selection_dataset = TouchDataset(selection_data, include_touch=False)
    selected, details = select_records(selection_dataset, selection_data, args.max_samples, args.selection_seed, args.selection)
    if not selected:
        raise ValueError('No samples selected')
    checkpoint_hash = file_digest(args.checkpoint)
    config_hash = file_digest(args.checkpoint.parent / 'config.yaml')
    reference_settings = check_reference(args, details, selection_data)
    reference_condition = check_reference_run(
        reference_settings, checkpoint_hash, config_hash, data, surface_patches=not joint_pointmap)
    if args.normalization_probe:
        if args.normalization_probe > len(selected):
            raise ValueError('Normalization probe exceeds the reference sample count')
        # Choose before measuring either arm, keeping the same reference objects/views.
        indices = sorted(random.Random(args.selection_seed).sample(range(len(selected)), args.normalization_probe))
        selected, details = [selected[i] for i in indices], [details[i] for i in indices]
    elif args.smoke_samples:
        if args.smoke_samples > len(selected):
            raise ValueError('Smoke sample count exceeds the reference sample count')
        selected, details = selected[:args.smoke_samples], details[:args.smoke_samples]
    conditions = ([(16, 'full'), (16, 'retained')] if args.normalization_probe
                  else [(count, 'full') for count in PATCH_COUNTS])
    loader = build_dataloader(data, 1, args.workers, shuffle=False, joint_pointmap=joint_pointmap)
    records = {record['sample_id']: record for record in loader.dataset.records}
    if len(records) != len(loader.dataset.records):
        raise ValueError('Duplicate sample IDs in touch manifest')
    check_selected_inputs(selected, selection_dataset, records, loader.dataset, selection_data, data)
    loader.dataset.records = [records[record['sample_id']] for record in selected]
    print(f'Validated {len(selected)} objects/views; reference condition: {reference_condition}; '
          f'patch counts: {sorted({count for count, _ in conditions}, reverse=True)}', flush=True)
    if args.preflight_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'selected_samples.yaml').write_text(yaml.safe_dump(details, sort_keys=False))
    (args.output_dir / 'selected_samples.json').write_text(json.dumps(details, indent=2) + '\n')
    (args.output_dir / 'selection.json').write_text(json.dumps(loader.dataset.records, indent=2) + '\n')
    for count in sorted({count for count, _ in conditions}):
        np.save(args.output_dir / f'indices_{count}.npy', subset_indices(points_per_patch, count, joint_pointmap=joint_pointmap))
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(data=data, selection_data=selection_data, run_config=run_config,
                    checkpoint_sha256=checkpoint_hash, run_config_sha256=config_hash,
                    reference_condition=reference_condition, conditioning_config=conditioning,
                    patch_counts=sorted({count for count, _ in conditions}), points_per_patch=points_per_patch, pointmap_points=pointmap_points,
                    script_sha256=file_digest(__file__), pipeline_sha256=file_digest(args.pipeline_config),
                    manifest_sha256=file_digest(loader.dataset.resolve_path(data['dataset']['manifest'])),
                    split_sha256=file_digest(loader.dataset.resolve_path(data['dataset']['split_file'])),
                    patch_selection='first K patches in the saved farthest-center order; nested, no resampling',
                    normalization=('paired full-32 vs retained-16 normalization, including position embedding'
                                   if args.normalization_probe else f'fixed from saved {pointmap_points} pointmap + 32x{points_per_patch} touch points before removing patches, including position embedding'),
                    evaluation='evaluation.evaluate.evaluate_condition, including its seeds and mesh metrics',
                    torch_version=torch.__version__, numpy_version=np.__version__)
    (args.output_dir / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    target_cache = {record['object_id']: load_target_mesh(
        record, selection_dataset, args.output_dir, args.surface_points, args.icp_points,
        args.save_points, stable_seed(args.seed, f"target:{record['object_id']}")) for record in selected}
    pipeline, pipeline_config = build_pipeline(args.pipeline_config, args.device)
    pipeline.ss_generator.no_shortcut = True
    model = restore_run(pipeline, checkpoint, args.device)
    model.requires_grad_(False)
    model.eval()
    encoder = model.touch_encoder
    assert encoder.encoder.query_type == 'learnable' and encoder.encoder.num_latents == 1024
    settings['pipeline'] = OmegaConf.to_container(pipeline_config, resolve=True)
    (args.output_dir / 'settings.json').write_text(json.dumps(settings, indent=2, default=str) + '\n')
    del checkpoint
    rows, completed = [], set()
    for count, normalization in conditions:
        name = f'patches_{count}' + (f'_{normalization}' if args.normalization_probe else '')
        measured = evaluate_condition(
            name, pipeline, encoder, loader, records, target_cache,
            completed, args.output_dir / 'metrics.csv', args,
            touch_token_fn=partial(encode_patches, model, count, normalization=normalization, joint_pointmap=joint_pointmap),
            joint_pointmap=joint_pointmap, fixed_joint_patches=joint_pointmap)
        for row in measured:
            row['normalization'] = normalization
            row.update(patch_count=count, points_per_patch=points_per_patch, pointmap_points=pointmap_points,
                       touch_points=count * points_per_patch, total_points=pointmap_points + count * points_per_patch, stage1_iou=np.nan)
            if row['stage1_path']:
                with np.load(args.output_dir / row['stage1_path'], allow_pickle=False) as saved:
                    row['stage1_iou'] = float(np.logical_and(saved['prediction'], saved['target']).sum()
                                             / max(np.logical_or(saved['prediction'], saved['target']).sum(), 1))
                    stage1 = dict(saved)
                indices = subset_indices(points_per_patch, count, joint_pointmap=joint_pointmap)
                stage1['touch_centers'] = stage1['touch_centers'][indices]
                if joint_pointmap:
                    stage1['encoder_input_camera'] = stage1['encoder_input_camera'][indices]
                    stage1['touch_source_indices'] = stage1['touch_source_indices'][:count * points_per_patch]
                stage1['touch_count'] = np.int64(count * points_per_patch)
                stage1['encoder_input_indices'] = indices
                stage1['patch_count'] = np.int64(count)
                stage1['encoder_normalization'] = np.asarray(normalization)
                np.savez_compressed(args.output_dir / row['stage1_path'], **stage1)
        rows.extend(measured)
        write_metrics(args.output_dir / 'metrics.csv', rows)
        summary = (summarize(rows, args.seed, group_key='condition',
                             groups=['patches_16_full', 'patches_16_retained'], reference='patches_16_full')
                   if args.normalization_probe else summarize(rows, args.seed))
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Saved {len(rows)} reconstruction attempts to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
