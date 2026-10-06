"""Remove whole patches from a frozen 32x256 model, using the main evaluation loop."""
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
from experiments.vecsetx.sam3d_point_count import touch_tokens


PATCH_COUNTS = (32, 28, 24, 16, 12, 8)
METRICS = ('stage1_iou', 'chamfer', 'fscore_0.01', 'normal_consistency', 'voxel_iou_64', 'emd')


def encode_patches(model, count, points, mask):
    # Keep the exact ordinary evaluation path for the full-input baseline.
    if count == 32:
        return model.get_touch_tokens(points, mask)
    encoder = model.touch_encoder
    prepared, valid, shift, scale = encoder.prepare_points(points, mask)
    assert prepared.shape == (1, 8192, 3) and valid.all()
    # The loader concatenates 256 points per patch in saved FPS-center order.
    # Normalize the full cloud once, then remove whole patches without padding.
    return touch_tokens(encoder, prepared[:, :count * 256], shift, scale)


def check_reference(args, selected, checkpoint_hash, selection_data, data):
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
        if settings[key] != getattr(args, key):
            raise ValueError(f'{key} differs from the reference evaluation: {settings[key]!r}')
    if settings['pipeline'] != file_identity(args.pipeline_config):
        raise ValueError('Pipeline config differs from the reference evaluation')
    if settings['selection_data'] != selection_data:
        raise ValueError('Selection data config differs from the reference evaluation')
    for identity in settings['dataset_files']:
        if file_identity(identity['path']) != identity:
            raise ValueError(f'Reference dataset file changed: {identity["path"]}')
    matching = [entry for entry in settings.get('entries', {}).values()
                if entry.get('checkpoint_sha256') == checkpoint_hash]
    if not matching:
        raise ValueError('This checkpoint is not recorded in the reference evaluation; use the same weights')
    if not any((entry.get('metadata') or {}).get('data') == data for entry in matching):
        raise ValueError('Checkpoint evaluation data differs from the reference evaluation')


def summarize(rows, seed):
    baseline = {row['sample_id']: row for row in rows if row['patch_count'] == 32 and not row['error']}
    result = {}
    for count in PATCH_COUNTS:
        attempted = [row for row in rows if row['patch_count'] == count]
        valid = [row for row in attempted if not row['error']]
        entry = dict(attempted=len(attempted), successful=len(valid), failed=len(attempted) - len(valid))
        for metric in METRICS:
            values = [row[metric] for row in valid if np.isfinite(row.get(metric, np.nan))]
            delta = np.asarray([row[metric] - baseline[row['sample_id']][metric] for row in valid
                                if row['sample_id'] in baseline and np.isfinite(row.get(metric, np.nan))
                                and np.isfinite(baseline[row['sample_id']].get(metric, np.nan))])
            ci = None
            if len(delta):
                bootstrap = np.random.default_rng(seed).choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
                ci = np.quantile(bootstrap, [.025, .975]).tolist()
            entry[metric] = dict(
                complete_objects=len(values), paired_objects=len(delta),
                mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                mean_change_from_32=float(delta.mean()) if len(delta) else None,
                median_change_from_32=float(np.median(delta)) if len(delta) else None,
                mean_change_95ci=ci,
                fraction_worse_than_32=float(np.mean(delta > 0 if metric in ('chamfer', 'emd') else delta < 0)) if len(delta) else None,
            )
        result[count] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/zeroverse/zeroverse_pointmap_touch_32x256/best.pt'))
    parser.add_argument('--data-config', type=Path)
    parser.add_argument('--selection-data-config', type=Path, required=True)
    parser.add_argument('--reference-evaluation', type=Path, help='Verify selection, protocol and weights against an existing main evaluation')
    parser.add_argument('--pipeline-config', type=Path, default=Path('checkpoints/hf/pipeline.yaml'))
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/vecsetx/outputs/sam3d_patch_count'))
    parser.add_argument('--split', default='val')
    parser.add_argument('--max-samples', type=int, default=100, help='Objects; 0 selects all, as in the main evaluator')
    parser.add_argument('--selection', choices=['random', 'hidden'], default='random')
    parser.add_argument('--selection-seed', type=int, default=29)
    parser.add_argument('--seed', type=int, default=29)
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
           args.stage2_inference_steps, args.metric_workers) < 1 or min(args.max_samples, args.emd_points, args.workers) < 0:
        raise ValueError('Invalid sample counts or worker count')
    checkpoint, run_config, data = read_run(args.checkpoint, args.data_config, args.split)
    conditioning = checkpoint.get('conditioning_config', {})
    if (checkpoint['mode'] != 'image_touch' or checkpoint['touch_config']['encoder_name'] != 'vecsetx'
            or checkpoint.get('training_config', {}).get('constant_touch', False)
            or conditioning.get('oracle_point_frame', False) or conditioning.get('shared_pointmap_normalization', False)):
        raise ValueError('Use the separate-branch camera-frame VecSetX model from the point-count sweep')
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
    check_reference(args, details, checkpoint_hash, selection_data, data)
    loader = build_dataloader(data, 1, args.workers, shuffle=False)
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
    for count in PATCH_COUNTS:
        np.save(args.output_dir / f'indices_{count}.npy', np.arange(count * 256))
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(data=data, selection_data=selection_data, run_config=run_config,
                    checkpoint_sha256=checkpoint_hash, patch_counts=PATCH_COUNTS, points_per_patch=256,
                    script_sha256=file_digest(__file__), pipeline_sha256=file_digest(args.pipeline_config),
                    manifest_sha256=file_digest(loader.dataset.resolve_path(data['dataset']['manifest'])),
                    split_sha256=file_digest(loader.dataset.resolve_path(data['dataset']['split_file'])),
                    patch_selection='first K patches in the saved farthest-center order; nested, no resampling',
                    normalization='fixed from all 32x256 points before removing patches',
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
    for count in PATCH_COUNTS:
        measured = evaluate_condition(
            f'patches_{count}', pipeline, encoder, loader, records, target_cache,
            completed, args.output_dir / 'metrics.csv', args,
            touch_token_fn=partial(encode_patches, model, count))
        for row in measured:
            row.update(patch_count=count, points_per_patch=256, total_points=count * 256, stage1_iou=np.nan)
            if not row['error']:
                with np.load(args.output_dir / row['stage1_path'], allow_pickle=False) as saved:
                    row['stage1_iou'] = float(np.logical_and(saved['prediction'], saved['target']).sum()
                                             / max(np.logical_or(saved['prediction'], saved['target']).sum(), 1))
                    stage1 = dict(saved)
                # The main evaluator sees the dense cloud for normalization; save
                # only the contacts actually encoded as this condition's touches.
                stage1['touch_centers'] = stage1['touch_centers'][:count * 256]
                stage1['patch_count'] = np.int64(count)
                np.savez_compressed(args.output_dir / row['stage1_path'], **stage1)
        rows.extend(measured)
        write_metrics(args.output_dir / 'metrics.csv', rows)
        (args.output_dir / 'summary.json').write_text(json.dumps(summarize(rows, args.seed), indent=2) + '\n')
    print(f'Saved {len(rows)} reconstruction attempts to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
