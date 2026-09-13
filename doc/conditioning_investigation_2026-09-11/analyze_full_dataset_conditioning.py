"""Validate native-loss interventions on existing dataset-wide checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_checkpoint_metadata(meta, config, expected):
    assert config['cross_attention_scope'] == meta['cross_attention_scope'] == 'full'
    assert meta['mode'] == config['mode']
    assert meta['touch_config'] == config['touch_config']
    assert meta['conditioning_config'] == {'no_pointmap': False, 'oracle_point_frame': False}
    assert not config.get('no_pointmap', False) and not config.get('oracle_point_frame', False)
    assert not config.get('joint_pointmap', False) and not config.get('train_vecsetx', False)
    assert config['precision'] == 'bf16'
    assert math.isfinite(meta['best_loss']) and abs(meta['best_loss']-expected['best_loss']) <= 1e-7
    assert meta['step'] == expected['step'], 'Checkpoint does not match exported best step'
    if expected['epoch'] is not None:
        assert meta['epoch'] == expected['epoch']


def validate(report, run, expected, frame):
    key = report['settings']['model_key']
    assert key in ('image', 'surface') and report['complete'] and report['parameters_unchanged']
    assert report['settings'] == dict(model_key=key, seed=37, precision='bf16', draws=8, bank_base=900000)
    validate_checkpoint_metadata(report['checkpoint_metadata'], run['config'], expected)
    conditions = ('removed',) if key == 'image' else ('correct', 'wrong_1', 'wrong_2', 'wrong_3', 'removed')
    assert report['context_checks'] == {c: 7528 if c == 'removed' else 8552 for c in conditions}
    assert len(report['input_batches']) == 16
    values = {'train': {}, 'val': {}}
    rows = {(r['batch'], r['condition']): r for r in report['rows']}
    assert len(rows) == len(report['rows']) and set(rows) == {(b,c) for b in range(16) for c in conditions}
    for index, (meta, old) in enumerate(zip(report['input_batches'], frame['input_batches'][16:32])):
        split = 'train' if index < 8 else 'val'
        assert meta['batch'] == index and meta['dataset_split'] == split
        for name, value in meta.items():
            if name not in ('batch', 'dataset_split'):
                assert old[name] == value, name
        assert len(set(meta['object_ids'])) == 4
        for c in conditions:
            row = rows[index, c]; matrix, scalar = row['per_object_losses'], row['scalar_losses']
            assert len(matrix) == len(scalar) == 8
            for vector, loss in zip(matrix, scalar):
                assert len(vector) == 4 and all(math.isfinite(v) and v >= 0 for v in vector)
                assert math.isfinite(loss) and abs(mean(vector)-loss) <= 1e-7+1e-5*abs(loss)
            for i, oid in enumerate(meta['object_ids']):
                values[split].setdefault(oid, {}).setdefault(c, []).extend(v[i] for v in matrix)
    assert not set(values['train']) & set(values['val'])
    for split, objects in values.items():
        assert len(objects) == 16
        for oid, cells in objects.items():
            assert set(cells) == set(conditions) and all(len(v) == 16 for v in cells.values())
            objects[oid] = {c: mean(v) for c, v in cells.items()}
            if key == 'surface':
                objects[oid]['wrong_mean'] = mean(objects[oid][f'wrong_{i}'] for i in (1,2,3))
    return values


def paired(delta):
    return dict(mean_delta=mean(delta), median_delta=median(delta), objects_negative=sum(d < 0 for d in delta),
                objects_positive=sum(d > 0 for d in delta), objects=len(delta))


def analyze(reports, runs, expected, frames):
    assert set(reports) == {'image', 'surface'}
    values = {key: validate(r, runs[key], expected[key], frames[key]) for key, r in reports.items()}
    for a, b in zip(reports['surface']['input_batches'], reports['image']['input_batches']):
        assert {k:v for k,v in a.items() if k != 'features_sha256'} == b
    assert reports['image']['source_sha256'] == reports['surface']['source_sha256']
    summary = {}
    for split in ('train', 'val'):
        image, surface = values['image'][split], values['surface'][split]
        ids = sorted(image); assert set(ids) == set(surface)
        summary[split] = dict(cells={'image': mean(v['removed'] for v in image.values()),
            **{c:mean(v[c] for v in surface.values()) for c in ('correct','wrong_1','wrong_2','wrong_3','wrong_mean','removed')}},
            paired_comparisons={
                'surface_correct_minus_image': paired([surface[o]['correct']-image[o]['removed'] for o in ids]),
                'surface_wrong_mean_minus_image': paired([surface[o]['wrong_mean']-image[o]['removed'] for o in ids]),
                'surface_removed_minus_image': paired([surface[o]['removed']-image[o]['removed'] for o in ids]),
                'wrong_minus_correct': paired([surface[o]['wrong_mean']-surface[o]['correct'] for o in ids]),
                'removed_minus_correct': paired([surface[o]['removed']-surface[o]['correct'] for o in ids])})
    return dict(summary=summary, per_object=values,
                limits='Two existing full-dataset checkpoints selected by their own historical best validation losses; '
                       'different training steps, not a matched causal estimate of adding surfaces. Within-checkpoint '
                       'condition interventions use paired noise/time and unchanged weights. 16 current train/16 val '
                       'identities, two views each; already inspected diagnostic pool, not population validation. '
                       'Three distractors per object and removed-stream distribution shift limit interpretation. '
                       'Checkpoint metadata matches exported mode/config/best loss/step; historical data/source '
                       'fingerprints were not stored. Current inputs match earlier diagnostics. No training or Stage 2.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports, runs, expected, frames, hashes = {}, {}, {}, {}, {}
    for key in ('image', 'surface'):
        p = args.probe_root/f'{key}.json';reports[key] = json.loads(p.read_text());hashes[key] = sha(p)
        rp = HERE/'full_dataset_probe_references'/f'{key}.json';runs[key] = json.loads(rp.read_text())
        ep = rp.with_name(f'{key}_expected.json');expected[key] = json.loads(ep.read_text())
        fp = HERE/'object_transfer_returned_manual'/('camera' if key == 'surface' else 'image')/'results.json'
        frames[key] = json.loads(fp.read_text())
        for field, path in [('run_reference_sha256',rp),('expected_reference_sha256',ep),('frame_reference_sha256',fp)]:
            assert reports[key][field] == sha(path)
        for name,digest in reports[key]['source_sha256'].items():
            assert sha(REPO/name) == digest, name
    result = analyze(reports, runs, expected, frames);result['report_sha256'] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
