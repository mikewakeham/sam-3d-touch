"""Actual predictions: decode, fit rotation only, and re-encode before/after."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image


def save_inputs(dataset, record, output):
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dataset.resolve_path(record['image_path']), output/'input_view.png')
    rgba = np.asarray(Image.open(output/'input_view.png').convert('RGBA'))
    pointmap = np.load(dataset.resolve_path(record['pointmap_path']))
    valid = np.isfinite(pointmap).all(-1) & (rgba[..., 3] > 0) & (pointmap[..., 2] > 0)
    ids = np.flatnonzero(valid)
    ids = ids[np.linspace(0, len(ids)-1, min(5000, len(ids)), dtype=int)] if len(ids) else ids
    surface = dataset.load_full_surface(dataset.resolve_path(record['full_surface_path']))
    if len(surface) > 5000:
        surface = surface[np.linspace(0, len(surface)-1, 5000, dtype=int)]
    np.savez_compressed(output/'input_clouds.npz', surface=surface,
        pointmap=pointmap.reshape(-1, 3)[ids], pointmap_rgb=rgba.reshape(-1, 4)[ids, :3]/255.)


def mse(a, b):
    return float(np.mean((np.asarray(a, dtype=np.float64)-b)**2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='best.pt')
    parser.add_argument('--pipeline-config', type=Path, default=REPO/'checkpoints/hf/pipeline.yaml')
    parser.add_argument('--encoder-checkpoint', type=Path, default=REPO/'checkpoints/hf/ss_encoder.ckpt')
    parser.add_argument('--val-objects', type=int, default=32)
    parser.add_argument('--views', type=int, default=4)
    parser.add_argument('--draws', type=int, default=2)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--inference-steps', type=int, default=25)
    parser.add_argument('--cfg-strength', type=float, default=0.)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from dataloader import TouchDataset, collate_touch_batch
    from evaluation.evaluate import read_run, restore_run, stable_seed
    from train import amp, build_stage1_pipeline, prepare_batch
    from sam3d_objects.model.io import load_model_from_checkpoint
    from experiments.coordinate_system.scripts.full_checkpoints.protocol import evaluation_groups
    from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_rollout_core import sample_from_noise
    from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_rollout_gpu import decode_support
    from experiments.coordinate_system.scripts.rotation_loss.rotation_utils import load_encoder, encode
    from experiments.coordinate_system.scripts.generated_rotation_examples.rotation_fit import align_decoded_grid

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    checkpoint, _, data = read_run(args.run_dir/args.checkpoint)
    conditioning = checkpoint.get('conditioning_config', {})
    pipeline_path = args.pipeline_config.resolve()
    pipeline = build_stage1_pipeline(pipeline_path, device)
    model = restore_run(pipeline, checkpoint, device).requires_grad_(False).eval()
    step, mode = checkpoint['step'], checkpoint['mode']
    del checkpoint
    data = copy.deepcopy(data); data['dataset']['split'] = 'val'
    use_touch = model.touch_encoder is not None
    dataset = TouchDataset(data, include_touch=use_touch, oracle_point_frame=conditioning.get('oracle_point_frame', False))
    groups = evaluation_groups(dataset.records, args.seed, args.val_objects, args.views)
    lookup = {r['sample_id']: i for i, r in enumerate(dataset.records)}
    cfg = OmegaConf.load(pipeline_path)
    generator = pipeline.ss_generator
    generator.fm_eps_max = 0.; generator.self_consistency_prob = 0.; generator.reverse_fn.p_unconditional = 0.
    pipeline.override_ss_generator_cfg_config(generator, cfg_strength=args.cfg_strength,
        inference_steps=args.inference_steps, rescale_t=float(cfg.get('ss_rescale_t', 3)),
        cfg_interval=list(cfg.get('ss_cfg_interval', [0, 500])), cfg_strength_pm=0.)
    generator.no_shortcut = True; generator.reverse_fn.training = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'complete': False,
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'checkpoint': {'step': step, 'mode': mode, 'conditioning': conditioning},
        'selection': 'Fixed validation identities/views selected from metadata before generation; every prediction measured.',
        'examples': []}
    generated = []
    with torch.no_grad():
        for group_index, records in enumerate(groups):
            batch = collate_touch_batch([dataset[lookup[r['sample_id']]] for r in records])
            targets, visual, keyword, xyz, mask = prepare_batch(pipeline, batch, device, args.precision,
                use_touch, joint_pointmap=mode == 'image_touch_joint',
                oracle_point_frame=conditioning.get('oracle_point_frame', False),
                shared_pointmap_normalization=conditioning.get('shared_pointmap_normalization', False))
            if len(visual) != 1 or keyword:
                raise ValueError('Expected one precomputed visual context and no keyword context')
            with amp(device, args.precision):
                tokens = model.get_touch_tokens(xyz, mask) if use_touch else None
            seed = stable_seed(args.seed, '|'.join(batch['sample_id']))
            for record in records:
                save_inputs(dataset, record, args.output_dir/'inputs'/record['sample_id'])
            for draw in range(args.draws):
                sample_seed = seed+500000+draw
                torch.manual_seed(sample_seed)
                noise = generator._generate_noise({k: tuple(v.shape) for k, v in targets.items()}, device)
                with amp(device, args.precision):
                    prediction = sample_from_noise(generator, noise, visual[0], tokens)
                for i, record in enumerate(records):
                    filename = f"{record['sample_id']}_draw{draw}.npz"
                    predicted = prediction[i].float().cpu().numpy()
                    target = targets['shape'][i].float().cpu().numpy()
                    np.savez_compressed(args.output_dir/filename, prediction=predicted, target=target)
                    generated.append({'object_id': record['object_id'], 'sample_id': record['sample_id'],
                        'draw': draw, 'seed': sample_seed, 'latent_file': filename,
                        'input_dir': f"inputs/{record['sample_id']}",
                        'generated_to_target_mse': mse(predicted, target)})
            print(f'Generated group {group_index+1}/{len(groups)}', flush=True)
    del model, generator, pipeline, targets, visual, keyword, xyz, mask, tokens, noise, prediction
    gc.collect(); torch.cuda.empty_cache()

    decoder_config = OmegaConf.load(pipeline_path.parent/cfg.ss_decoder_config_path)
    if 'pretrained_ckpt_path' in decoder_config: del decoder_config['pretrained_ckpt_path']
    decoder = load_model_from_checkpoint(instantiate(decoder_config),
        str(pipeline_path.parent/cfg.ss_decoder_ckpt_path), device='cpu', strict=True,
        freeze=True, eval=True, state_dict_key=None).to(device)
    encoder = load_encoder(args.encoder_checkpoint, device)
    with torch.no_grad():
        for index, row in enumerate(generated):
            path = args.output_dir/row['latent_file']
            with np.load(path) as data_file:
                prediction, target = data_file['prediction'], data_file['target']
            prediction_grid = decode_support(decoder, torch.from_numpy(prediction)[None].to(device))
            target_grid = decode_support(decoder, torch.from_numpy(target)[None].to(device))
            aligned_points, aligned_grid, fit = align_decoded_grid(prediction_grid, target_grid)
            prediction_roundtrip = encode(encoder, prediction_grid, device)
            aligned_roundtrip = encode(encoder, aligned_grid, device)
            target_roundtrip = encode(encoder, target_grid, device)
            row['rotation_fit'] = fit
            row['latent_mse'] = {
                'generated_to_stored_target': row.pop('generated_to_target_mse'),
                'decoded_unaligned_reencoded_to_stored_target': mse(prediction_roundtrip, target),
                'decoded_aligned_reencoded_to_stored_target': mse(aligned_roundtrip, target),
                'decoded_target_reencoded_to_stored_target_floor': mse(target_roundtrip, target),
                'generated_to_decoded_unaligned_reencoded_roundtrip_change': mse(prediction, prediction_roundtrip)}
            np.savez_compressed(path, prediction=prediction, target=target,
                prediction_occupancy=prediction_grid, target_occupancy=target_grid,
                aligned_occupancy=aligned_grid, aligned_points=aligned_points,
                fitted_rotation=np.asarray(fit['rotation']))
            report['examples'].append(row)
            (args.output_dir/'results.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f"Measured {index+1}/{len(generated)}: {row['sample_id']} draw {row['draw']}", flush=True)
    report['complete'] = True
    report['definitions'] = {
        'fit': 'Post-generation continuous proper rotation fit about fixed origin; no translation, scaling, reflection or camera transform.',
        'latent_comparison': 'Both decoded unaligned and fitted-aligned prediction geometries are rasterized and passed through the same frozen target encoder, then compared with the unchanged stored target.',
        'target_floor': 'Frozen encoder applied to decoded target occupancy, compared with its originating stored target; measures decoder/encoder round-trip floor.',
        'rasterization': 'Nearest fixed 64-cube cell; identity round-trip asserted exact. Out-of-bounds point fraction reported.',
        'scope': 'All selected validation predictions reported. Rotation fitting diagnoses shape agreement after orientation correction; it does not alter training or generation.'}
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Saved:', args.output_dir.resolve())


if __name__ == '__main__':
    main()
