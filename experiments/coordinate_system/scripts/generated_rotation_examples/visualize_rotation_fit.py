"""Render selected actual predictions before/after fitted rotation and re-encoding."""
import argparse
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiments.coordinate_system.scripts.presentation.render_conditioning import plot
from experiments.coordinate_system.scripts.shared.pose_shape_geometry import points


def panel_figure(paths, titles, captions, output, footnote):
    fig, axes = plt.subplots(1, len(paths), figsize=(5*len(paths), 5.9))
    for axis, path, title, caption in zip(axes, paths, titles, captions):
        axis.imshow(Image.open(path)); axis.axis('off')
        axis.set_title(title, fontsize=14, pad=10)
        axis.text(.5, -.025, caption, transform=axis.transAxes,
                  ha='center', va='top', fontsize=11)
    fig.subplots_adjust(left=.01, right=.99, top=.88, bottom=.22, wspace=.07)
    fig.text(.5, .035, footnote, ha='center', fontsize=9.5, color='#555555')
    for suffix in ['.png', '.pdf']:
        fig.savefig(output.with_suffix(suffix), dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def render_inputs(measurement_dir, output, row, conditioning):
    source = measurement_dir/row['input_dir']
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source/'input_view.png', output/'input_view.png')
    with np.load(source/'input_clouds.npz') as data:
        surface, pointmap, colors = data['surface'], data['pointmap'], data['pointmap_rgb']
    low = np.minimum(surface.min(0), pointmap.min(0)); high = np.maximum(surface.max(0), pointmap.max(0))
    center, limit = (low+high)/2, float(np.max(high-low)*.55)
    spacing = .5 if limit <= 1.5 else 1.
    plot(output/'surface.png', [(surface, '#168de0', .9)], center, limit,
         coordinate_order=(2,0,1), grid_spacing=spacing, marker_size=.8)
    plot(output/'pointmap.png', [(pointmap, colors, .9)], center, limit,
         coordinate_order=(2,0,1), grid_spacing=spacing, marker_size=.8)
    plot(output/'combined.png', [(surface, '#168de0', .9), (pointmap, '#f18b32', .9)],
         center, limit, coordinate_order=(2,0,1), grid_spacing=spacing, marker_size=.8)
    pointmap_caption = ('Saved reference; disabled by checkpoint' if conditioning.get('no_pointmap', False)
                        else 'Conditioning pointmap')
    panel_figure([output/name for name in ['input_view.png','surface.png','pointmap.png','combined.png']],
        ['Input view','Full surface','Pointmap','Surface + pointmap'],
        [row['sample_id'],'Camera-frame conditioning surface',pointmap_caption,
         'Blue: surface   Orange: pointmap'], output/'inputs',
        'Raw saved data before preprocessing. Camera (Z,X,Y) displayed as plot (X,Y,Z); model arrays unchanged.')
    return output/'input_view.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement-dir', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--all', action='store_true', help='Render every measured example instead of selected examples')
    args = parser.parse_args()
    measured = json.loads((args.measurement_dir/'results.json').read_text())
    selection = json.loads(args.selection.read_text())
    rows = selection['all_examples'] if args.all else selection['selected']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = {}; rendered = []
    for rank, row in enumerate(rows, 1):
        sid = row['sample_id']
        if sid not in input_paths:
            input_paths[sid] = render_inputs(args.measurement_dir,
                args.output_dir/'inputs'/sid, row, measured['checkpoint']['conditioning'])
        with np.load(args.measurement_dir/row['latent_file']) as data:
            prediction, target, aligned = (data['prediction_occupancy'],
                data['target_occupancy'], data['aligned_occupancy'])
        folder = args.output_dir/f"{rank:02d}_{sid}_draw{row['draw']}"
        folder.mkdir(parents=True, exist_ok=True)
        target_points, prediction_points, aligned_points = points(target), points(prediction), points(aligned)
        limit = max(.55, float(np.max(np.abs(np.concatenate(
            [target_points,prediction_points,aligned_points])))*1.05))
        target_layer = (target_points, '#219b52', .9)
        paths = [folder/f'{name}.png' for name in ['target','prediction','aligned']]
        plot(paths[0], [target_layer], np.zeros(3), limit, marker_size=.8)
        plot(paths[1], [target_layer,(prediction_points,'#168de0',.9)], np.zeros(3), limit, marker_size=.8)
        plot(paths[2], [target_layer,(aligned_points,'#168de0',.9)], np.zeros(3), limit, marker_size=.8)
        latent = row['latent_mse']; fit = row['rotation_fit']
        raw = fit['raw']; aligned_score = fit['aligned_voxelized']
        panel_figure([input_paths[sid]]+paths,
            ['Actual validation view','Decoded stored target','Actual generated prediction','Rotation-fitted prediction'],
            [f"{sid}\nDraw {row['draw']}",
             f"Target round-trip floor: {latent['decoded_target_reencoded_to_stored_target_floor']:.4f}",
             f"Original latent MSE: {latent['generated_to_stored_target']:.4f}\nRe-encoded MSE: {latent['decoded_unaligned_reencoded_to_stored_target']:.4f}\nF@2: {raw['fscore_2v']:.1%}",
             f"Fit angle: {fit['angle_degrees']:.1f}°\nRe-encoded MSE: {latent['decoded_aligned_reencoded_to_stored_target']:.4f}\nF@2: {aligned_score['fscore_2v']:.1%}"],
            folder/'comparison',
            'Prediction generated normally. Rotation fitted after decoding about fixed origin; no translation or scale. Both decoded variants voxelized and passed through the same frozen target encoder.')
        rendered.append({'rank':rank,'sample_id':sid,'draw':row['draw'],
                         'comparison':str((folder/'comparison.png').relative_to(args.output_dir))})
    (args.output_dir/'rendered.json').write_text(json.dumps({'selection':selection['selection'],
        'all':args.all,'examples':rendered}, indent=2)+'\n')
    print(f'Rendered {len(rendered)} examples:', args.output_dir.resolve())


if __name__ == '__main__':
    main()
