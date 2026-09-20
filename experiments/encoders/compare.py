"""Paired Stage-1 loss and voxel previews, using the existing training/evaluation code."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import torch
from experiments.encoders.prepare_data import experiment_path


def replace_surface(batch, donor):
    # Move XYZ, normals, and mask together; keep recipient image, depth, and target.
    return {**batch, **{key: donor[key] for key in ('touch_xyz', 'touch_normals', 'touch_mask') if key in donor}}


def seed_sample(seed, generator):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    generator.random_generator.manual_seed(seed)


def plot_voxels(path, image, target, predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(4 * (2 + len(predictions)), 4))
    axis = figure.add_subplot(1, 2 + len(predictions), 1)
    axis.imshow(image)
    axis.set_title('Input')
    axis.axis('off')
    for index, (name, voxels) in enumerate([('Target latent decoded', target), *predictions.items()], 2):
        axis = figure.add_subplot(1, 2 + len(predictions), index, projection='3d')
        points = np.argwhere(voxels)
        # Thin only the plot; metrics and NPZ retain the complete occupancy grid.
        points = points[::max(1, len(points) // 12000)]
        if len(points):
            axis.scatter(*points.T, s=1, c=points[:, 2], cmap='viridis')
        axis.set(xlim=(0, voxels.shape[0]), ylim=(0, voxels.shape[1]), zlim=(0, voxels.shape[2]), title=name)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=20, azim=35)
    figure.subplots_adjust(left=.02, right=.98, bottom=.1, top=.84, wspace=.15)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_losses(run_dirs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run_dir in dict.fromkeys(run_dirs):
        path = run_dir / 'metrics.jsonl'
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for axis, key in zip(axes, ['loss/train', 'loss/val']):
            values = [row for row in rows if key in row]
            axis.plot([row['global_step'] for row in values], [row[key] for row in values], label=run_dir.name)
            axis.set(xlabel='Optimizer step', ylabel=key)
    for axis in axes:
        if axis.lines:
            axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / 'losses.png', dpi=140)
    plt.close(figure)


def main():
    from omegaconf import OmegaConf
    from dataloader import build_dataloader, collate_touch_batch
    from evaluate import read_run, restore_run, select_records, sample_shape, decode_voxels, stable_seed, safe_name
    from train import build_stage1_pipeline, prepare_batch, configure_encoder_data, amp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--pipeline-config', type=Path, default=HERE.parents[1] / 'checkpoints/hf/pipeline.yaml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--objects', type=int, default=8)
    parser.add_argument('--inference-steps', type=int, default=25)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.objects < 2 or args.inference_steps < 1:
        parser.error('Need at least two objects for the shuffled-surface control and positive inference steps')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Use python on one GPU, not distributed torchrun')
    output = experiment_path(args.output_dir)
    names = [f'{path.parent.name}_{path.stem}' for path in args.checkpoints]
    if len(set(names)) != len(names):
        parser.error('Checkpoint run names/stems must be unique')
    output.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(output / '.matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(output / '.cache')
    if (output / 'metrics.csv').exists():
        raise FileExistsError('Choose a new comparison output directory')
    device = torch.device(args.device)
    selected_ids = None
    summaries = {}
    torch.set_float32_matmul_precision('high')

    for name, path in zip(names, args.checkpoints):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        checkpoint, config, data = read_run(path)
        use_touch = checkpoint['touch_config'] is not None
        if use_touch:
            configure_encoder_data(data, checkpoint['touch_config']['encoder_name'])
        loader = build_dataloader(data, 1, 0, shuffle=False, include_touch=use_touch)
        if selected_ids is None:
            selected, details = select_records(loader.dataset, data, args.objects, args.seed, 'random')
            selected_ids = [row['sample_id'] for row in selected]
            if len(selected_ids) < 2:
                raise ValueError('Need at least two validation objects')
            (output / 'selected_samples.json').write_text(json.dumps(details, indent=2) + '\n')
        records = {row['sample_id']: row for row in loader.dataset.records}
        loader.dataset.records = [records[sample_id] for sample_id in selected_ids]
        pipeline = build_stage1_pipeline(args.pipeline_config, device)
        model = restore_run(pipeline, checkpoint, device)
        pipeline.ss_generator.no_shortcut = True
        # Conditional-only, like training, for every encoder and the image baseline.
        # Keep CFG's training mode with p_unconditional=0; backbone/encoder stay in eval.
        pipeline_config = OmegaConf.load(args.pipeline_config)
        decoder = pipeline.init_ss_decoder(pipeline_config.ss_decoder_config_path,
                                           pipeline_config.ss_decoder_ckpt_path).eval().requires_grad_(False)
        run_dir = output / name
        run_dir.mkdir(exist_ok=True)
        rows = []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for index, batch in enumerate(loader):
                seed = stable_seed(args.seed, batch['sample_id'][0])
                target = decode_voxels(decoder, batch['target_shape'].to(device))[0].cpu().numpy()
                predictions = {}
                donor_index = (index + 1) % len(loader.dataset)
                donor = collate_touch_batch([loader.dataset[donor_index]]) if use_touch else None
                for treatment in (['matched', 'shuffled'] if use_touch else ['matched']):
                    inputs = replace_surface(batch, donor) if treatment == 'shuffled' else batch
                    prepared = prepare_batch(pipeline, inputs, device, 'bf16', use_touch,
                                             checkpoint['mode'] == 'image_touch_joint',
                                             model.conditioning_config['oracle_point_frame'],
                                             model.conditioning_config.get('shared_pointmap_normalization', False),
                                             use_normals=use_touch and model.touch_encoder.requires_normals)
                    seed_sample(seed, pipeline.ss_generator)
                    with amp(device, 'bf16'):
                        loss = model(*prepared).item()
                        tokens = model.get_touch_tokens(prepared[3], prepared[4]) if use_touch else None
                        seed_sample(seed, pipeline.ss_generator)
                        latent = sample_shape(pipeline, prepared[1], prepared[2], tokens,
                                              args.inference_steps, device)
                    predicted = decode_voxels(decoder, latent.float())[0].cpu().numpy()
                    predictions[treatment] = predicted
                    union = np.logical_or(target, predicted).sum()
                    row = {'run': name, 'step': checkpoint['step'], 'sample_id': batch['sample_id'][0],
                           'treatment': treatment, 'loss': loss,
                           'voxel_iou': float(np.logical_and(target, predicted).sum() / max(union, 1)),
                           'donor_sample_id': selected_ids[donor_index] if treatment == 'shuffled' else ''}
                    rows.append(row)
                    with (output / 'metrics.csv').open('a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=list(row))
                        if file.tell() == 0:
                            writer.writeheader()
                        writer.writerow(row)
                filename = safe_name(batch['sample_id'][0])
                np.savez_compressed(run_dir / f'{filename}.npz', target=target, **predictions)
                plot_voxels(run_dir / f'{filename}.png', batch['image'][0].numpy(), target, predictions)
                print(f'{name} [{index + 1}/{len(loader)}] {filename}', flush=True)
        summaries[name] = {
            'checkpoint': str(path.resolve()), 'step': checkpoint['step'],
            'peak_gpu_gib': torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else None,
            'treatments': {treatment: {key: float(np.mean([row[key] for row in rows if row['treatment'] == treatment]))
                                      for key in ['loss', 'voxel_iou']}
                           for treatment in sorted({row['treatment'] for row in rows})},
        }
        (output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
        del model, pipeline, decoder, checkpoint, prepared, tokens, latent
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (output / 'settings.json').write_text(json.dumps({'seed': args.seed, 'inference_steps': args.inference_steps,
                                                     'guidance': 'conditional-only', 'objects': len(selected_ids)}, indent=2) + '\n')
    plot_losses([path.parent for path in args.checkpoints], output)


if __name__ == '__main__':
    main()
