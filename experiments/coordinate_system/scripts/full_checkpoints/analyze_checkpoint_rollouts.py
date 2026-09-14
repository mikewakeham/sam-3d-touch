"""Validate complete paired rollout reports and summarize raw Stage-1 fidelity."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def analyze(reports):
    oracle, constant = reports['oracle'], reports['constant']
    for key in ('source_sha256', 'additional_source_sha256', 'dataset_sha256', 'pipeline_sha256',
                'decoder_sha256', 'sampler', 'banks'):
        assert oracle[key] == constant[key], f'Unpaired {key}'
    assert len(oracle['inputs']) == len(constant['inputs'])
    for left, right in zip(oracle['inputs'], constant['inputs']):
        assert {k: v for k, v in left.items() if k != 'tokens_sha256'} == {
            k: v for k, v in right.items() if k != 'tokens_sha256'}, 'Unpaired input tensors'
    metrics = ('iou', 'precision_2v', 'recall_2v', 'fscore_2v')
    values = {}
    for arm, report in reports.items():
        assert report['complete'] and report['adapted_parameters_unchanged']
        assert report['settings']['arm'] == arm
        assert report['checkpoint_metadata']['step'] == 14660
        actual, per_object = set(), defaultdict(list)
        for sample in report['samples']:
            key = tuple(sample[k] for k in ('split', 'group', 'sample_id', 'visual', 'surface_shift', 'draw'))
            assert key not in actual, 'Duplicate rollout sample'
            actual.add(key)
            for metric in metrics:
                value = sample[metric]
                assert math.isfinite(value) and 0 <= value <= 1, 'Invalid metric'
                per_object[sample['split'], sample['visual'], sample['surface_shift'], sample['object_id'], metric].append(value)
        expected = {(item['split'], item['group'], sid, visual, shift, draw)
                    for item in report['inputs'] for sid in item['sample_ids']
                    for visual in ('present', 'zero') for shift in ((0, 1) if arm == 'oracle' else (0,))
                    for draw in range(2)}
        assert actual == expected, 'Missing/unexpected rollout sample'
        ids = {(item['split'], sid): oid for item in report['inputs']
               for sid, oid in zip(item['sample_ids'], item['object_ids'])}
        assert all(ids[s['split'], s['sample_id']] == s['object_id'] for s in report['samples'])
        values[arm] = {key: mean(v) for key, v in per_object.items()}
    summary = {}
    for split in ('train', 'val'):
        summary[split] = {}
        for visual in ('present', 'zero'):
            cells = {}
            ids = sorted({k[3] for k in values['oracle'] if k[:3] == (split, visual, 0)})
            for label, arm, shift in (('oracle', 'oracle', 0), ('oracle_wrong', 'oracle', 1), ('constant', 'constant', 0)):
                objects = [{'object_id': oid, **{metric: values[arm][split, visual, shift, oid, metric]
                                                for metric in metrics}} for oid in ids]
                ious = sorted(o['iou'] for o in objects)
                p10 = ious[int(.1 * len(ious))]
                cells[label] = {'objects': objects, **{f'mean_{metric}': mean(o[metric] for o in objects) for metric in metrics},
                    'object_iou_p10_order_statistic': p10,
                    'raw_endpoint_met': mean(ious) >= .95 and p10 >= .90,
                    'objects_95_precision_and_recall': sum(o['precision_2v'] >= .95 and o['recall_2v'] >= .95 for o in objects)}
            for label in ('constant', 'oracle_wrong'):
                changes = [{'object_id': a['object_id'], 'fscore_difference': a['fscore_2v'] - b['fscore_2v'],
                            'iou_difference': a['iou'] - b['iou']}
                           for a, b in zip(cells['oracle']['objects'], cells[label]['objects'])]
                cells['oracle_minus_' + label] = {
                    'per_object': changes, 'mean_fscore_difference': mean(c['fscore_difference'] for c in changes),
                    'mean_iou_difference': mean(c['iou_difference'] for c in changes),
                    'objects_positive_fscore_difference': sum(c['fscore_difference'] > 0 for c in changes)}
            summary[split][visual] = cells
    return {'pairing_and_completeness_verified': True, 'summary': summary,
            'scope': 'Raw pose-sensitive Stage1 support fidelity. The historical full-data raw endpoint is mean object IoU>=.95 '
                     'and the floor(0.1*N) sorted object IoU>=.90. Failure is not a pose-independent shape failure; '
                     'saved supports permit rigid analysis. No significance or universal sampler/architecture claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    reports = {arm: json.loads((args.root / arm / 'results.json').read_text()) for arm in ('oracle', 'constant')}
    for arm, report in reports.items():
        for name, digest in report['files_sha256'].items():
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert sha(args.root / arm / relative) == digest, f'Damaged output {arm}/{name}'
    result = analyze(reports)
    (args.root / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for split, visuals in result['summary'].items():
        for visual, cells in visuals.items():
            for arm in ('oracle', 'oracle_wrong', 'constant'):
                print(split, visual, arm, f"IoU={cells[arm]['mean_iou']:.4f}",
                      f"F2v={cells[arm]['mean_fscore_2v']:.4f}", flush=True)
    print('Return summary and NPZ supports together for independent geometry verification.')


if __name__ == '__main__':
    main()
