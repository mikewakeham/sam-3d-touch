"""Validate pairing, then summarize the no-update oracle/constant probe."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean


def analyze(reports):
    oracle, constant = reports['oracle'], reports['constant']
    for arm, report in reports.items():
        assert report['complete'] and report['adapted_parameters_unchanged']
        assert report['settings']['arm'] == arm
        assert report['checkpoint_metadata']['step'] == 14660
    for name in ('source_sha256', 'pipeline_sha256', 'dataset_sha256', 'data_config', 'banks'):
        assert oracle[name] == constant[name], f'Unmatched {name}; do not pool'
    for key in ('seed', 'precision', 'objects_per_split'):
        assert oracle['settings'].get(key) == constant['settings'].get(key), key
    matched_keys = ('split', 'group', 'sample_ids', 'object_ids', 'image_sha256',
                    'pointmap_sha256', 'visual_sha256', 'points_sha256', 'mask_sha256', 'target_sha256')
    assert len(oracle['inputs']) == len(constant['inputs'])
    for left, right in zip(oracle['inputs'], constant['inputs']):
        assert all(left[k] == right[k] for k in matched_keys), 'Different observation/target tensors'
    cells = {}
    for arm, report in reports.items():
        inputs = {(r['split'], r['group']): r for r in report['inputs']}
        assert len(inputs) == len(report['inputs'])
        actual_keys = set()
        object_values = defaultdict(list)
        for row in report['rows']:
            key = tuple(row[k] for k in ('split', 'group', 'visual', 'surface_shift', 'time_kind', 'draw'))
            assert key not in actual_keys, 'Duplicate probe row'
            actual_keys.add(key)
            ids = inputs[row['split'], row['group']]['object_ids']
            assert len(ids) == len(row['losses']) == 4 and len(set(ids)) == 4
            for oid, loss in zip(ids, row['losses']):
                assert math.isfinite(loss) and loss >= 0, 'Invalid loss'
                object_values[(row['split'], row['visual'], row['surface_shift'], row['time_kind'], oid)].append(loss)
        expected_keys = {
            (bank['split'], bank['group'], visual, shift, bank['time_kind'], bank['draw'])
            for bank in report['banks'] for visual in ('present', 'zero')
            for shift in (range(4) if arm == 'oracle' else range(1))}
        assert actual_keys == expected_keys, 'Missing or unexpected probe cells'
        # Aggregate repeated views/draws within each object before comparing arms.
        cells[arm] = {key: mean(values) for key, values in object_values.items()}
    summary = {}
    for split in ('train', 'val'):
        summary[split] = {}
        for visual in ('present', 'zero'):
            summary[split][visual] = {}
            for time_kind in ('native', 't_0.05', 't_0.5', 't_0.95'):
                ids = sorted(k[-1] for k in cells['oracle'] if k[:4] == (split, visual, 0, time_kind))
                assert len(ids) == oracle['settings']['objects_per_split']
                objects = []
                for oid in ids:
                    correct = cells['oracle'][split, visual, 0, time_kind, oid]
                    wrong = mean(cells['oracle'][split, visual, shift, time_kind, oid] for shift in (1, 2, 3))
                    reference = cells['constant'][split, visual, 0, time_kind, oid]
                    objects.append({'object_id': oid, 'oracle': correct, 'constant': reference,
                                    'oracle_wrong_mean': wrong, 'wrong_minus_correct': wrong - correct,
                                    'constant_minus_oracle': reference - correct})
                summary[split][visual][time_kind] = {
                    'objects': objects,
                    **{f'mean_{k}': mean(row[k] for row in objects) for k in
                       ('oracle', 'constant', 'oracle_wrong_mean', 'wrong_minus_correct', 'constant_minus_oracle')},
                    'objects_oracle_better_than_constant': sum(row['constant_minus_oracle'] > 0 for row in objects),
                    'objects_correct_better_than_wrong_mean': sum(row['wrong_minus_correct'] > 0 for row in objects),
                }
    return {'pairing_verified': True, 'summary': summary,
            'scope': 'Paired losses on selected identities, not reconstructed-shape fidelity or proof of geometry understanding. '
                     'Train and validation are separate identities. Fixed time panels are not the native training distribution. '
                     'No p-values or equivalence claims; one checkpoint per arm.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    reports = {arm: json.loads((args.root / arm / 'results.json').read_text()) for arm in ('oracle', 'constant')}
    output = analyze(reports)
    (args.root / 'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    for split, visuals in output['summary'].items():
        for visual, times in visuals.items():
            for time, values in times.items():
                print(split, visual, time,
                      f"oracle={values['mean_oracle']:.6f}",
                      f"constant={values['mean_constant']:.6f}",
                      f"wrong-minus-correct={values['mean_wrong_minus_correct']:+.6f}", flush=True)
    print('Return:', args.root / 'summary.json')


if __name__ == '__main__':
    main()
