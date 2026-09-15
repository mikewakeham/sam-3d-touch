"""Decode shortlisted Stage-1 predictions and plot the corresponding rotation comparison."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import numpy as np
from experiments.coordinate_system.scripts.coordinate_audit.frame_contract import rotate_grid
from experiments.coordinate_system.scripts.alignment_robustness.alignment_tolerance_protocol import rotation
from experiments.coordinate_system.scripts.shared.pose_shape_geometry import points, metrics
from experiments.coordinate_system.scripts.presentation.render_conditioning import plot
from experiments.coordinate_system.scripts.presentation.render_oracle import strip


def comparison(prediction, target, rotated_target, name):
    r = np.eye(3, dtype=int) if name == 'identity' else np.rint(rotation(name[0], int(name[1:]))).astype(int)
    aligned = rotate_grid(prediction, r.T)
    restored_reference = rotate_grid(rotated_target, r.T)
    return aligned, dict(
        raw=metrics(points(prediction), points(target)),
        aligned=metrics(points(aligned), points(target)),
        prediction_vs_rotated_reference=metrics(points(prediction), points(rotated_target)),
        decoder_rotation_reference=metrics(points(restored_reference), points(target)),
        prediction_to_target_rotation=r.T.tolist())


def decode_candidates(args, report):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam3d_objects.model.io import load_model_from_checkpoint
    from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_rollout_gpu import decode_support

    probe = json.loads((args.find_dir/'results.json').read_text())
    pipeline_path = args.pipeline_config or Path(probe['settings']['pipeline_config'])
    config = OmegaConf.load(pipeline_path)
    # Same loading as InferencePipeline.init_ss_decoder, without loading DINO
    # or the generator again. Only the frozen Stage-1 decoder is needed here.
    decoder_config = OmegaConf.load(pipeline_path.parent/config.ss_decoder_config_path)
    if 'pretrained_ckpt_path' in decoder_config:
        del decoder_config['pretrained_ckpt_path']
    decoder = instantiate(decoder_config)
    decoder = load_model_from_checkpoint(decoder,
        str(pipeline_path.parent/config.ss_decoder_ckpt_path), device='cpu',
        strict=True, freeze=True, eval=True, state_dict_key=None).to(args.device)
    decoded = args.output_dir/'decoded'
    decoded.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for row in report['examples']:
            with np.load(args.find_dir/'candidates'/row['latent_file']) as data:
                grids = {key: decode_support(decoder,
                    torch.from_numpy(data[key]).float()[None].to(args.device))
                    for key in ['prediction', 'target', 'rotated_target']}
            np.savez_compressed(decoded/row['latent_file'], **grids)
            print(f"Decoded candidate {row['rank']}: {row['sample_id']}", flush=True)


def render(args, report):
    rows = []
    for row in report['examples']:
        with np.load(args.output_dir/'decoded'/row['latent_file']) as data:
            predicted, target, rotated_target = [data[key] for key in ['prediction', 'target', 'rotated_target']]
        aligned, scores = comparison(predicted, target, rotated_target, row['best_rotation'])
        folder = args.output_dir/Path(row['latent_file']).stem
        folder.mkdir(parents=True, exist_ok=True)
        paths = [folder/f'{name}.png' for name in ['target', 'prediction', 'aligned_prediction']]
        target_layer = (points(target), '#219b52', .9)
        plot(paths[0], [target_layer], np.zeros(3), .55, marker_size=.8)
        plot(paths[1], [target_layer, (points(predicted), '#168de0', .9)], np.zeros(3), .55, marker_size=.8)
        plot(paths[2], [target_layer, (points(aligned), '#168de0', .9)], np.zeros(3), .55, marker_size=.8)
        raw_f = scores['raw']['fscore_2v']
        aligned_f = scores['aligned']['fscore_2v']
        strip(paths, ['Decoded training target', 'Actual generated prediction', f"Inverse of {row['best_rotation'].upper()}"],
              ['Green: target   Blue: prediction',
               f"Original-target latent MSE: {row['identity_mse']:.4f}\nGeometry F@2 voxels: {raw_f:.1%}",
               f"Rotated-target latent MSE: {row['best_mse']:.4f}\nGeometry F@2 voxels: {aligned_f:.1%}"],
              folder/'comparison',
              'Latent prediction unchanged in both MSE comparisons. Only decoded prediction is rotated in panel 3. No translation, rescaling or Stage 2.')
        rows.append(dict(row, geometry=scores))
    summary = dict(total_predictions=report['total_predictions'],
                   nonidentity_preferred=report['nonidentity_preferred'],
                   selection=report['selection'], examples=rows,
                   definitions={
                       'geometry': 'Frozen Stage-1 decoded occupied-voxel centers; distances in target units. F@2 uses tolerance 2/64.',
                       'aligned': 'Inverse of the latent-selected exact rotation, applied to decoded prediction; no extra geometric optimization.',
                       'decoder_rotation_reference': 'Inverse-rotated decoded rotated-target reference vs decoded original target. Measures decoder non-equivariance for this control.',
                       'limits': 'Post-hoc latent shortlist, not exhaustive geometric search. Seven orientations only. No automatic claim of correct shape or coordinate-system causality.'})
    (args.output_dir/'geometry_results.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Figures and geometry:', args.output_dir.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--find-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--pipeline-config', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--render-only', action='store_true', help='CPU: reuse already saved decoded grids')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.find_dir/'candidates'/'candidates.json').read_text())
    if not args.render_only and report['examples']:
        decode_candidates(args, report)
    render(args, report)


if __name__ == '__main__':
    main()
