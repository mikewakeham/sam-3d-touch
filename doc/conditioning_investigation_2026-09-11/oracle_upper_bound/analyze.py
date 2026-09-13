"""Validate paired full-data assessments and summarize by independent object."""
import argparse
import json
import random
from pathlib import Path
from statistics import mean
from protocol import ARMS


def keyed_inputs(report):
    return {(r['split'], r['group']): {k: v for k, v in r.items() if k != 'tokens_sha256'}
            for r in report['inputs']}


def object_values(report, split, condition, metric):
    groups = {}
    for row in report['samples']:
        if row['split'] == split and row['condition'] == condition:
            groups.setdefault(row['object_id'], []).append(row[metric])
    return {key: mean(values) for key, values in groups.items()}


def paired_difference(a, b):
    if set(a) != set(b) or not a:
        raise ValueError('Paired object identities differ or empty')
    values = [a[k] - b[k] for k in sorted(a)]
    rng = random.Random(29)
    boot = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return dict(mean_difference=mean(values), object_bootstrap_95=[boot[49], boot[1949]],
                objects=len(values), positive_objects=sum(v > 0 for v in values),
                scope='Objects resampled, views/draws averaged first. One training seed; development split.')


def analyze(root, epoch):
    reports, configs = {}, {}
    for arm in ARMS:
        configs[arm] = json.loads((root / arm / 'config.json').read_text())
        reports[arm] = json.loads((root / arm / f'assessment_{epoch}/results.json').read_text())
        if not reports[arm]['complete'] or reports[arm]['epoch'] != epoch:
            raise ValueError('Incomplete/wrong endpoint')
    base, config = reports[ARMS[0]], configs[ARMS[0]]
    for arm in ARMS:
        report, c = reports[arm], configs[arm]
        for field in ('seed', 'batch_size', 'evidence', 'cross_attention_scope',
                      'learning_rate', 'cross_attention_learning_rate'):
            if c['contract'][field] != config['contract'][field]:
                raise ValueError(f'Unmatched contract: {arm}/{field}')
        if c['initial_ca_sha256'] != config['initial_ca_sha256']:
            raise ValueError('Different initial shape cross-attention weights')
        if arm != 'image' and c['initial_adapter_sha256'] != config['initial_adapter_sha256']:
            raise ValueError('Different initial surface adapter weights')
        if report['step'] != base['step'] or keyed_inputs(report) != keyed_inputs(base):
            raise ValueError('Unmatched endpoint or evaluation observations/targets')
        expected = {(r['split'], r['sample_id'], r['draw']): r['noise_sha256']
                    for r in base['samples'] if r['condition'] == 'correct'}
        for row in report['samples']:
            if row['noise_sha256'] != expected[(row['split'], row['sample_id'], row['draw'])]:
                raise ValueError('Unpaired sampling noise')
        for condition in {r['condition'] for r in report['samples']}:
            keys = [(r['split'], r['sample_id'], r['draw']) for r in report['samples'] if r['condition'] == condition]
            if len(keys) != len(set(keys)) or set(keys) != set(expected):
                raise ValueError('Missing/duplicate evaluation sample')
    result = dict(epoch=epoch, step=base['step'], raw={}, paired={}, native={},
                  caution='Raw pose-specific metrics. Failure to pass is not a pose-independent shape failure. '
                          'Saved NPZ supports permit rigid analysis. No Stage-2 measurements.')
    for split in ('train', 'val'):
        result['raw'][split] = {}; result['paired'][split] = {}
        result['native'][split] = {}
        for arm, report in reports.items():
            native = {}
            for row in report['native']:
                if row['split'] == split:
                    for oid, value in zip(row['object_ids'], row['losses']):
                        native.setdefault(row['condition'], {}).setdefault(oid, []).append(value)
            result['native'][split][arm] = {condition: mean(mean(v) for v in objects.values())
                                           for condition, objects in native.items()}
            iou = object_values(report, split, 'correct', 'iou')
            precision = object_values(report, split, 'correct', 'precision_2v')
            recall = object_values(report, split, 'correct', 'recall_2v')
            fs = object_values(report, split, 'correct', 'fscore_2v')
            values = sorted(iou.values())
            result['raw'][split][arm] = dict(mean_object_iou=mean(values),
                object_iou_p10=values[max(0, int(.1 * len(values)))], mean_object_fscore_2v=mean(fs.values()),
                objects_95_precision_and_recall=sum(precision[o] >= .95 and recall[o] >= .95 for o in iou),
                objects=len(iou), per_object_iou=iou,
                raw_endpoint_met=mean(values) >= .95 and values[max(0, int(.1 * len(values)))] >= .90)
        oracle = object_values(reports['oracle_dropout'], split, 'correct', 'fscore_2v')
        for comparator in ('image', 'constant_dropout', 'camera_dropout'):
            result['paired'][split]['oracle_minus_' + comparator] = paired_difference(
                oracle, object_values(reports[comparator], split, 'correct', 'fscore_2v'))
        result['paired'][split]['oracle_correct_minus_wrong'] = paired_difference(
            oracle, object_values(reports['oracle_dropout'], split, 'wrong_1', 'fscore_2v'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--epoch', type=int, default=20)
    a = p.parse_args()
    result = analyze(a.root, a.epoch)
    output = a.root / f'comparison_{a.epoch}.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
