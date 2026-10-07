"""Remove whole patches from the saved 32-patch joint input using the main evaluator."""
import argparse
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
    load_target_mesh, evaluate_condition, file_digest, file_identity, write_metrics,
)
from experiments.vecsetx.sam3d_point_count import touch_tokens, subset_indices


PATCH_COUNTS = (32, 28, 24, 16, 12, 8)
METRICS = ('stage1_iou', 'chamfer', 'fscore_0.01', 'normal_consistency', 'voxel_iou_64', 'emd')


def encode_patches(model, count, points, mask, normalization='full'):
    # Keep the exact ordinary evaluation path for the full-input baseline.
    if count == 32:
        return model.get_touch_tokens(points, mask)
    encoder = model.touch_encoder
    indices = subset_indices(224, count)
    if normalization == 'retained':
        # Normalize only the actual input, without prepare_points padding/FPS.
        prepared, shift, scale = encoder.normalize_points_for_vecsetx(points[:, indices], mask[:, indices])
        return touch_tokens(encoder, prepared, shift, scale)
    prepared, valid, shift, scale = encoder.prepare_points(points, mask)
    assert prepared.shape == (1, 8192, 3) and valid.all()
    # Keep all 1024 saved pointmap points and 224 points per retained patch.
    # Normalize the full joint cloud once, then subset without padding or FPS.
    return touch_tokens(encoder, prepared[:, indices], shift, scale)


def check_reference(args, selected, selection_data):
    reference = args.reference_evaluation
    if reference is None:
        return
    saved = yaml.safe_load((reference / 'selected_samples.yaml').read_text())
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


def summarize(rows, seed, group_key='patch_count', groups=PATCH_COUNTS, reference=32):
    baseline = {row['sample_id']: row for row in rows if row[group_key] == reference}
    result = {}
    for count in groups:
        attempted = [row for row in rows if row[group_key] == count]
        valid = [row for row in attempted if not row['error']]
        entry = dict(attempted=len(attempted), successful=len(valid), failed=len(attempted) - len(valid))
        for metric in METRICS:
            measured = attempted if metric == 'stage1_iou' else valid
            values = [row[metric] for row in measured if np.isfinite(row.get(metric, np.nan))]
            delta = np.asarray([row[metric] - baseline[row['sample_id']][metric] for row in measured
                                if row['sample_id'] in baseline and np.isfinite(row.get(metric, np.nan))
                                and (metric == 'stage1_iou' or not baseline[row['sample_id']]['error'])
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
                   f'fraction_worse_than_{reference}': float(np.mean(delta > 0 if metric in ('chamfer', 'emd') else delta < 0)) if len(delta) else None},
            )
        result[count] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/zeroverse/zeroverse_pointmap_touch_32x256_joint/best.pt'))
    parser.add_argument('--data-config', type=Path)
    parser.add_argument('--selection-data-config', type=Path, required=True)
    parser.add_argument('--reference-evaluation', type=Path, help='Verify selection and protocol against an existing main evaluation')
    parser.add_argument('--pipeline-config', type=Path, default=Path('checkpoints/hf/pipeline.yaml'))
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/vecsetx/outputs/sam3d_patch_count'))
    parser.add_argument('--split', default='val')
    parser.add_argument('--max-samples', type=int, default=100, help='Objects; 0 selects all, as in the main evaluator')
    parser.add_argument('--selection', choices=['random', 'hidden'], default='random')
    parser.add_argument('--selection-seed', type=int, default=29)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--normalization-probe', type=int, default=0, metavar='N',
                        help='Evaluate full vs retained normalization at 16 patches on N random objects from the reference selection')
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
           args.stage2_inference_steps, args.metric_workers) < 1 or min(args.max_samples, args.emd_points, args.workers, args.normalization_probe) < 0:
        raise ValueError('Invalid sample counts or worker count')
    checkpoint, run_config, data = read_run(args.checkpoint, args.data_config, args.split)
    conditioning = checkpoint.get('conditioning_config', {})
    if (checkpoint['mode'] != 'image_touch_joint' or conditioning.get('joint_pointmap_points') != 1024
            or checkpoint['touch_config']['encoder_name'] != 'vecsetx'
            or checkpoint.get('training_config', {}).get('constant_touch', False)
            or conditioning.get('oracle_point_frame', False) or conditioning.get('shared_pointmap_normalization', False)):
        raise ValueError('Use the 32-patch joint VecSetX checkpoint with saved 1024-pointmap + 7168-touch inputs')
    for config in (run_config['data'], data):
        touch = config['touch']
        if (touch['source'] != 'touch_patches' or touch['contacts']['count'] != 32
                or touch['point_sampling']['points_per_contact'] != 256):
            raise ValueError('Training and evaluation data must use 32x256 patches')

    selection_data = load_data_config(args.selection_data_config)
    selection_data['dataset']['split'] = args.split
    selection_dataset = TouchDataset(selection_data, include_touch=False)
    selected, details = select_records(selection_dataset, selection_data, args.max_samples, args.selection_seed, args.selection)
    if not selected:
        raise ValueError('No samples selected')
    checkpoint_hash = file_digest(args.checkpoint)
    check_reference(args, details, selection_data)
    if args.normalization_probe:
        # Choose before measuring either arm, keeping the same reference objects/views.
        indices = sorted(random.Random(args.selection_seed).sample(range(len(selected)), args.normalization_probe))
        selected, details = [selected[i] for i in indices], [details[i] for i in indices]
    conditions = ([(16, 'full'), (16, 'retained')] if args.normalization_probe
                  else [(count, 'full') for count in PATCH_COUNTS])
    loader = build_dataloader(data, 1, args.workers, shuffle=False, joint_pointmap=True)
    records = {record['sample_id']: record for record in loader.dataset.records}
    loader.dataset.records = [records[record['sample_id']] for record in selected]
    # Match the inputs and target used by the main evaluator's selection dataset.
    for record in selected:
        for key in ('mesh_path', 'image_path', 'camera_path', 'depth_path', 'target_path'):
            if selection_dataset.resolve_path(record[key]) != loader.dataset.resolve_path(records[record['sample_id']][key]):
                raise ValueError(f'Selection and touch manifests disagree on {key}')

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'selected_samples.yaml').write_text(yaml.safe_dump(details, sort_keys=False))
    (args.output_dir / 'selection.json').write_text(json.dumps(loader.dataset.records, indent=2) + '\n')
    for count in sorted({count for count, _ in conditions}):
        np.save(args.output_dir / f'indices_{count}.npy', subset_indices(224, count))
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(data=data, selection_data=selection_data, run_config=run_config,
                    checkpoint_sha256=checkpoint_hash, conditioning_config=conditioning,
                    patch_counts=sorted({count for count, _ in conditions}), points_per_patch=224, pointmap_points=1024,
                    script_sha256=file_digest(__file__), pipeline_sha256=file_digest(args.pipeline_config),
                    manifest_sha256=file_digest(loader.dataset.resolve_path(data['dataset']['manifest'])),
                    split_sha256=file_digest(loader.dataset.resolve_path(data['dataset']['split_file'])),
                    patch_selection='first K patches in the saved farthest-center order; nested, no resampling',
                    normalization=('paired full-32 vs retained-16 joint normalization, including position embedding'
                                   if args.normalization_probe else 'fixed from saved 1024 pointmap + 32x224 touch points before removing patches'),
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
            touch_token_fn=partial(encode_patches, model, count, normalization=normalization),
            joint_pointmap=True, fixed_joint_patches=True)
        for row in measured:
            if args.normalization_probe:
                row['normalization'] = normalization
            row.update(patch_count=count, points_per_patch=224, pointmap_points=1024,
                       touch_points=count * 224, total_points=1024 + count * 224, stage1_iou=np.nan)
            if row['stage1_path']:
                with np.load(args.output_dir / row['stage1_path'], allow_pickle=False) as saved:
                    row['stage1_iou'] = float(np.logical_and(saved['prediction'], saved['target']).sum()
                                             / max(np.logical_or(saved['prediction'], saved['target']).sum(), 1))
                    stage1 = dict(saved)
                # The main evaluator sees the dense cloud for normalization; save
                # only the contacts actually encoded as this condition's touches.
                indices = subset_indices(224, count)
                stage1['touch_centers'] = stage1['touch_centers'][indices]
                stage1['encoder_input_camera'] = stage1['encoder_input_camera'][indices]
                stage1['touch_source_indices'] = stage1['touch_source_indices'][:count * 224]
                stage1['touch_count'] = np.int64(count * 224)
                stage1['encoder_input_indices'] = indices
                stage1['patch_count'] = np.int64(count)
                if args.normalization_probe:
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
