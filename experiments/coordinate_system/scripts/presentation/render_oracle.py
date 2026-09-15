"""Octopus inverse-transform figures; reuse measured rotation-probe scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

if __package__:
    from .render_conditioning import affine, plot
else:
    from render_conditioning import affine, plot


def strip(paths, titles, captions, output, footnote):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, path, title, caption in zip(axes, paths, titles, captions):
        ax.imshow(Image.open(path))
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=12)
        ax.text(.5, -.025, caption, transform=ax.transAxes,
                ha='center', va='top', fontsize=14)
    fig.subplots_adjust(left=.01, right=.99, top=.89, bottom=.19, wspace=.07)
    fig.text(.5, .035, footnote, ha='center', fontsize=10, color='#555555')
    for suffix in ['.png', '.pdf']:
        fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-dir', type=Path, required=True)
    parser.add_argument('--rotation-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--view', default='005')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Controlled rotation: the scores were measured with the frozen SS encoder
    # on the cluster. Render saved occupancy arrays, without local encoding.
    example = args.rotation_dir/'examples'/f'{args.object_dir.name}.npz'
    with np.load(example) as data:
        np.testing.assert_array_equal(data['original'], data['restored'])
        scores = [0., float(data['latent_mse']), float(data['restored_latent_mse'])]
        paths = []
        for name in ['original', 'rotated', 'restored']:
            path = out/f'rotation_{name}.png'
            points = (np.argwhere(data[name])+.5)/64-.5
            plot(path, [(points, '#168de0', .9)], np.zeros(3), .55,
                 grid_spacing=.5, marker_size=1.1)
            paths.append(path)
    strip(paths, ['Target orientation', 'Rotate +90° about Z', 'Apply inverse: −90° about Z'],
          [f'Latent MSE = {score:.6f}' if score else 'Latent MSE = 0' for score in scores],
          out/'inverse_rotation_latent_mse',
          'MSE against the original frozen Stage-1 encoder mean. Exact voxel rotation; inverse restores identical encoder input. No decoding.')

    # Actual oracle operation: invert the saved SAM-camera-from-object matrix.
    # This is a different transform from the controlled 90-degree example.
    view = args.object_dir/'views'/args.view
    camera_points = np.load(view/'full_surface.npz')['points_camera']
    transform = np.diag([-1., -1., 1., 1.]) @ np.load(view/'camera.npz')['T_camera_from_object']
    object_points = affine(camera_points, np.linalg.inv(transform))
    mesh = trimesh.load(args.object_dir/'model.obj', force='mesh', process=False)
    mesh.apply_transform(np.load(args.object_dir/'object_transform.npz')['T_normalized_from_source'])
    surface = np.load(view/'full_surface.npz')
    sampled, _ = trimesh.sample.sample_surface(mesh, len(camera_points), seed=int(surface['sample_seed']))
    error = float(np.max(np.abs(sampled-object_points)))
    assert error < 2e-5
    # Independent target samples keep both colors visible after exact alignment.
    target_points, _ = trimesh.sample.sample_surface(mesh, 3000, seed=43)
    camera_centered = camera_points-transform[:3, 3]
    paths = [out/'camera_target_overlay.png', out/'oracle_target_overlay.png', out/'target_surface.png']
    target_layer = (target_points, '#219b52', .9)
    plot(paths[0], [target_layer, (camera_centered, '#168de0', .9)], np.zeros(3), .55,
         marker_size=.8)
    plot(paths[1], [target_layer, (object_points, '#168de0', .9)], np.zeros(3), .55,
         marker_size=.8)
    plot(paths[2], [target_layer], np.zeros(3), .55, marker_size=.8)
    strip(paths, [f'Before inverse: view {args.view}', 'After inverse camera transform', 'Target surface reference'],
          ['Camera translation removed only', 'Blue: transformed surface   Green: target', 'Green: points sampled from target mesh'],
          out/'camera_to_object_oracle',
          'Same native XYZ plotting axes, viewpoint and scale in all panels. Before VecSetX normalization. No camera-view latent MSE measured.')
    (out/'figure_metadata.json').write_text(json.dumps({
        'object_id': args.object_dir.name, 'view': args.view,
        'rotation_example': str(example.resolve()), 'rotation_latent_mse': scores,
        'camera_from_object': transform.tolist(),
        'oracle_surface_vs_replayed_object_points_max_error': error,
        'camera_latent_mse': None,
        'overlay_before': 'Camera translation subtracted; camera rotation retained. Target remains in native object XYZ.',
        'overlay_display_coordinate_order': [0, 1, 2],
        'note': 'Camera alignment and the measured exact-rotation control are separate examples.'
    }, indent=2)+'\n')
    print(out.resolve())


if __name__ == '__main__':
    main()
