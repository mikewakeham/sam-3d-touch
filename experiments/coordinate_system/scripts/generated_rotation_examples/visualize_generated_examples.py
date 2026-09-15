"""Actual validation inputs and predictions: test their known inverse camera rotations."""
import argparse
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
from experiments.coordinate_system.scripts.shared.pose_shape_geometry import points, metrics
from experiments.coordinate_system.scripts.presentation.render_conditioning import plot
import matplotlib.pyplot as plt


def comparison(prediction, target, camera_from_object):
    inverse_rotation = np.linalg.inv(camera_from_object[:3, :3])
    p, q = points(prediction), points(target)
    aligned = p @ inverse_rotation.T
    return aligned, dict(raw=metrics(p, q), oracle_rotation=metrics(aligned, q),
        inverse_camera_rotation=inverse_rotation.tolist(),
        camera_rotation_orthogonality_error=float(np.max(np.abs(
            camera_from_object[:3, :3].T @ camera_from_object[:3, :3]-np.eye(3)))))


def decode_examples(args, report):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam3d_objects.model.io import load_model_from_checkpoint
    from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_rollout_gpu import decode_support

    pipeline_path = args.pipeline_config or Path(report['settings']['pipeline_config'])
    config = OmegaConf.load(pipeline_path)
    decoder_config = OmegaConf.load(pipeline_path.parent/config.ss_decoder_config_path)
    if 'pretrained_ckpt_path' in decoder_config:
        del decoder_config['pretrained_ckpt_path']
    decoder = load_model_from_checkpoint(instantiate(decoder_config),
        str(pipeline_path.parent/config.ss_decoder_ckpt_path), device='cpu',
        strict=True, freeze=True, eval=True, state_dict_key=None).to(args.device)
    decoded = args.output_dir/'decoded'
    decoded.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for row in report['examples']:
            with np.load(args.generation_dir/row['latent_file']) as data:
                grids = {key: decode_support(decoder,
                    torch.from_numpy(data[key]).float()[None].to(args.device))
                    for key in ['prediction', 'target']}
                grids['camera_from_object'] = data['camera_from_object']
            np.savez_compressed(decoded/row['latent_file'], **grids)
            print(f"Decoded {row['sample_id']} draw {row['draw']}", flush=True)


def save_panel_figure(paths, titles, captions, output, footnote):
    fig, axes = plt.subplots(1, len(paths), figsize=(5*len(paths), 5.6))
    for ax, path, title, caption in zip(axes, paths, titles, captions):
        ax.imshow(Image.open(path)); ax.axis('off')
        ax.set_title(title, fontsize=14, pad=10)
        ax.text(.5, -.03, caption, transform=ax.transAxes, ha='center', va='top', fontsize=12)
    fig.subplots_adjust(left=.01, right=.99, top=.88, bottom=.2, wspace=.08)
    fig.text(.5, .035, footnote, ha='center', fontsize=10, color='#555555')
    for suffix in ['.png', '.pdf']:
        fig.savefig(output.with_suffix(suffix), dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def render_inputs(args, row, conditioning):
    source = args.generation_dir/row['input_dir']
    output = args.output_dir/'inputs'/row['sample_id']
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source/'input_view.png', output/'input_view.png')
    with np.load(source/'input_clouds.npz') as data:
        surface, pm, rgb = data['surface'], data['pointmap'], data['pointmap_rgb']
    clouds = [x for x in [surface, pm] if len(x)]
    low = np.min([x.min(0) for x in clouds], axis=0)
    high = np.max([x.max(0) for x in clouds], axis=0)
    center = (low+high)/2
    limit = float(np.max(high-low)*.55)
    spacing = .5 if limit <= 1.5 else 1.
    plot(output/'surface.png', [(surface, '#168de0', .9)], center, limit,
         coordinate_order=(2, 0, 1), grid_spacing=spacing, marker_size=.8)
    plot(output/'pointmap.png', [(pm, rgb, .9)], center, limit,
         coordinate_order=(2, 0, 1), grid_spacing=spacing, marker_size=.8)
    plot(output/'surface_and_pointmap.png', [(surface, '#168de0', .9), (pm, '#f18b32', .9)],
         center, limit, coordinate_order=(2, 0, 1), grid_spacing=spacing, marker_size=.8)
    pm_caption = 'Reference only: disabled for this checkpoint' if conditioning.get('no_pointmap', False) else 'Pointmap input'
    save_panel_figure([output/name for name in ['input_view.png', 'surface.png', 'pointmap.png', 'surface_and_pointmap.png']],
        ['Actual input view', 'Full surface', 'Pointmap', 'Surface + pointmap'],
        [row['sample_id'], 'Saved raw camera points', pm_caption, 'Blue: surface   Orange: pointmap'],
        output/'inputs', 'Raw inputs before model preprocessing. Camera (Z,X,Y) displayed as plot (X,Y,Z); model arrays unchanged.')
    return output/'input_view.png'


def render(args, report):
    rows = []
    input_paths = {}
    for row in report['examples']:
        sid = row['sample_id']
        if sid not in input_paths:
            input_paths[sid] = render_inputs(args, row, report['conditioning'])
        with np.load(args.output_dir/'decoded'/row['latent_file']) as data:
            predicted, target = data['prediction'], data['target']
            aligned, scores = comparison(predicted, target, data['camera_from_object'])
        folder = args.output_dir/Path(row['latent_file']).stem
        folder.mkdir(parents=True, exist_ok=True)
        paths = [folder/f'{name}.png' for name in ['target', 'prediction', 'oracle_rotated_prediction']]
        target_layer = (points(target), '#219b52', .9)
        p = points(predicted)
        limit = max(.55, float(np.max(np.abs(np.concatenate([p, points(target), aligned])))*1.05))
        plot(paths[0], [target_layer], np.zeros(3), limit, marker_size=.8)
        plot(paths[1], [target_layer, (p, '#168de0', .9)], np.zeros(3), limit, marker_size=.8)
        plot(paths[2], [target_layer, (aligned, '#168de0', .9)], np.zeros(3), limit, marker_size=.8)
        save_panel_figure([input_paths[sid]]+paths,
            ['Actual validation view', 'Decoded training target', 'Actual generated prediction', 'Known inverse camera rotation'],
            [f"{sid}\nDraw {row['draw']}", 'Green: target   Blue: prediction',
             f"Latent MSE: {row['latent_mse']:.4f}\nGeometry F@2: {scores['raw']['fscore_2v']:.1%}",
             f"Geometry F@2: {scores['oracle_rotation']['fscore_2v']:.1%}"],
            folder/'comparison',
            'No rotated-target encoding or rotation search. Known camera inverse applied only to decoded prediction directions; no translation fitting or rescaling.')
        rows.append(dict(row, geometry=scores))
    summary = dict(selection=report['selection'], total_predictions=len(rows),
        conditioning=report['conditioning'], examples=rows,
        definitions={'latent_mse': 'Actual generated latent vs stored training target only; unchanged by the geometric display transform.',
            'oracle_rotation': 'Inverse of the saved camera-from-object 3x3 linear block, applied about origin to decoded prediction points. Camera translation is not applied to centered shape output.',
            'geometry': 'Decoded Stage-1 occupied voxel centers; F@2 tolerance 2/64. No Stage 2, ICP, scale fitting, target re-encoding or outcome-based selection.',
            'scope': 'Tests whether actual predictions retain this input camera orientation. Improvement is not guaranteed; all metadata-selected examples are reported.'})
    (args.output_dir/'geometry_results.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Figures and geometry:', args.output_dir.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--pipeline-config', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--render-only', action='store_true', help='CPU: reuse saved decoded grids')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.generation_dir/'examples.json').read_text())
    if not args.render_only:
        decode_examples(args, report)
    render(args, report)


if __name__ == '__main__':
    main()
