"""Summarize complete representation reports; no best-preprocessing selection."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median


def summarize(report):
    if report.get('complete') is not True:
        raise ValueError('Incomplete report; inspect partial output without ranking it')
    sam = report['sam_rows']; vec = report['vecset_rows']
    n = report['settings']['objects']
    ids = [r['sample_id'] for r in report['records']]
    assert len(ids) == len(set(ids)) == len(sam) == n
    assert len({r['object_id'] for r in report['records']}) == n
    assert [r['sample_id'] for r in sam] == ids
    assert len(vec) == 2 * n
    lookup = {(r['sample_id'], r['frame']): r for r in vec}
    assert len(lookup) == len(vec)
    assert set(lookup) == {(sid, frame) for sid in ids for frame in ['oracle', 'camera']}
    assert all(math.isfinite(r['surface_latent_mse']) and r['surface_latent_mse'] >= 0 for r in sam)
    for r in sam:
        assert 0 <= r['decoded_surface_vs_target']['iou'] <= 1
    controls = report['sam_latent_controls']
    assert controls['sample_ids'] == ids
    assert len(controls['surface_vs_each_target_mse']) == n
    rows = []
    for i, r in enumerate(sam):
        sid = r['sample_id']
        distances = controls['surface_vs_each_target_mse'][i]
        assert len(distances) == n and all(math.isfinite(x) and x >= 0 for x in distances)
        row = {'sample_id': sid, 'surface_latent_mse': r['surface_latent_mse'],
               'surface_latent_mse_over_target_energy': r['surface_latent_mse'] / r['target_latent_mean_square'],
               'raw_point_iou': r['point_grid_vs_target']['iou'],
               'sam_decoded_iou': r['decoded_surface_vs_target']['iou'],
               'target_decode_iou': r['target_decode_vs_mesh']['iou'],
               'input_reference_fscore': r['input_points_vs_reference_surface']['fscore'],
               'reference_sampling_fscore': r['mesh_sampling_calibration']['fscore']}
        row['own_target_is_nearest_latent'] = distances[i] == min(distances)
        row['mean_other_target_latent_mse'] = mean(x for j, x in enumerate(distances) if j != i)
        row['leave_one_object_out_target_mean_mse'] = controls['leave_one_object_out_target_mean_mse'][i]
        for frame in ['oracle', 'camera']:
            v = lookup[sid, frame]
            row[f'vecset_{frame}_status'] = v['status']
            row[f'vecset_{frame}_fscore'] = v['surface_agreement']['fscore']
            row[f'vecset_{frame}_near_boundary_fraction'] = v.get('near_query_boundary_vertex_fraction')
        rows.append(row)
    keys = ['surface_latent_mse', 'surface_latent_mse_over_target_energy', 'raw_point_iou',
            'sam_decoded_iou', 'target_decode_iou', 'vecset_oracle_fscore', 'vecset_camera_fscore']
    return {'objects': rows, 'summary': {k: {'mean': mean(r[k] for r in rows),
               'median': median(r[k] for r in rows), 'min': min(r[k] for r in rows),
               'max': max(r[k] for r in rows)} for k in keys},
            'vecset_failures': [dict(sample_id=r['sample_id'], frame=r['frame'], status=r['status'])
                               for r in vec if r['status'] != 'ok'],
            'limits': 'SAM latent error and decoded IoU are one representation test; native VecSetX surface F-score is another, not a directly comparable model ranking. Oracle frame is privileged. A 64-cell isosurface can miss thin details; inspect fields and increase resolution on prespecified ambiguous cases before calling encoder failure. No training or conditioning improvement is established.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(json.loads(args.results.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))
