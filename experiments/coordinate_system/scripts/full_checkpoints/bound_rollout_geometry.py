"""Conservative all-rigid proximity bounds for correct-surface oracle outputs."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist
from experiments.coordinate_system.scripts.shared.pose_shape_geometry import points


def bound(source, target):
    if not len(source):
        return {'upper_bound': 0., 'diameter_upper_bound': 0., 'diameter_method': 'empty'}
    try:
        vertices = source[ConvexHull(source).vertices]
        diameter = float(pdist(vertices).max())
        method = 'convex_hull_diameter'
    except QhullError:
        diameter = float(np.linalg.norm(np.ptp(source, axis=0)))
        method = 'bounding_box_diagonal_upper_bound'
    axes = np.concatenate((np.eye(3), np.linalg.eigh(np.cov(target.T))[1].T))
    caps = []
    for axis in axes:
        projection = np.sort(target @ axis)
        # Any covered target subset has diameter <= D+2*epsilon, hence fits
        # in an interval of this width on every target projection axis.
        counts = np.searchsorted(projection, projection + diameter + 4/64 + 1e-10, side='right') - np.arange(len(target))
        caps.append(float(counts.max() / len(target)))
    return {'upper_bound': min(caps), 'diameter_upper_bound': diameter,
            'diameter_method': method, 'axis_bounds': caps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.root / 'oracle/results.json').read_text())
    rows, controls, target_seen = [], [], set()
    for sample in report['samples']:
        if sample['surface_shift'] != 0:
            continue
        with np.load(args.root / 'oracle' / sample['prediction_file'], allow_pickle=False) as pfile, \
             np.load(args.root / 'oracle' / sample['target_file'], allow_pickle=False) as qfile:
            index = sample['array_index']
            p, q = points(pfile['predicted_occupancy'][index]), points(qfile['target_occupancy'][index])
        recall, precision = bound(p, q), bound(q, p)
        rows.append({k: sample[k] for k in ('sample_id', 'object_id', 'split', 'visual', 'draw')} |
                    {'recall': recall, 'precision': precision,
                     'certified_cannot_reach_both_95': min(recall['upper_bound'], precision['upper_bound']) < .95})
        if sample['object_id'] not in target_seen:
            same = bound(q, q)
            rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            moved = bound(q @ rotation.T + [.11, -.23, .07], q)
            assert same['upper_bound'] == moved['upper_bound'] == 1
            assert abs(same['diameter_upper_bound'] - moved['diameter_upper_bound']) < 1e-10
            controls.append({'object_id': sample['object_id'], 'reference_bound': same['upper_bound'], 'rigid_reference_bound': moved['upper_bound']})
            target_seen.add(sample['object_id'])
    rejected = [r for r in rows if r['certified_cannot_reach_both_95']]
    result = {'rows': rows, 'certified': rejected, 'controls': controls, 'tolerance': 2/64,
              'scope': 'Finite decoded support centers, all rigid rotations/translations; no scaling. '
                       'Necessary diameter/projection condition; upper bound 1 is inconclusive. '
                       'Rejecting 95% P/R does not quantify optimal shape similarity or model impossibility.'}
    (args.root / 'rigid_bounds.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Certified failures', len(rejected), '/', len(rows), 'positive controls', len(controls), flush=True)
    for split in ('train', 'val'):
        for visual in ('present', 'zero'):
            selection = [r for r in rejected if (r['split'], r['visual']) == (split, visual)]
            print(split, visual, len(selection), flush=True)


if __name__ == '__main__':
    main()
