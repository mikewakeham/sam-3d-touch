"""Validate frozen checkpoint interventions and aggregate by held identity."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODELS = ('image', 'camera', 'oracle')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditions(key):
    return ('removed',) if key == 'image' else ('correct', 'wrong_1', 'wrong_2', 'wrong_3', 'removed')


def validate(report, fit):
    key = report['settings']['model_key']
    assert key in MODELS and report['complete']
    assert report['settings']['seed'] == 37 and report['settings']['precision'] == 'bf16'
    assert report['settings']['steps'] == [256, 1024]
    assert report['settings']['fresh_bank_base'] == 800000 and report['settings']['draws'] == 8
    assert report['input_batches'] == fit['input_batches'][24:]
    assert [c['step'] for c in report['checkpoints']] == [256, 1024]
    for checkpoint in report['checkpoints']:
        step = checkpoint['step']
        assert checkpoint['replay_passed'] and checkpoint['parameters_unchanged']
        if step == 1024:
            assert checkpoint['model_sha256'] == fit['final_model_sha256']
        expected = fit['assessments'][1]['rows'] if step == 256 else fit['final_rows']
        expected = [r for r in expected if r['split'] == 'held_object']
        assert len(expected) == len(checkpoint['replay'])
        for actual, old in zip(checkpoint['replay'], expected):
            condition = 'removed' if key == 'image' else ('wrong_1' if old['swapped_surface'] else 'correct')
            assert actual == dict(step=step, batch=old['batch'], condition=condition,
                                  scalar_losses=old['scalar_losses'], per_object_losses=old['per_object_losses'])
    contexts = {(r['step'], r['condition']): r for r in report['context_checks']}
    assert len(contexts) == len(report['context_checks'])
    assert set(contexts) == {(s, c) for s in (256, 1024) for c in conditions(key)}
    for (_, condition), row in contexts.items():
        assert row['visual_tokens'] == 7528
        assert row['surface_tokens'] == (0 if condition == 'removed' else 1024)
    mapping = {(r['step'], r['batch'], r['condition']): r for r in report['rows']}
    assert len(mapping) == len(report['rows'])
    assert set(mapping) == {(s, b, c) for s in (256, 1024) for b in range(24, 32) for c in conditions(key)}
    values = {step: {} for step in (256, 1024)}
    for (step, batch, condition), row in mapping.items():
        matrix, scalar = row['per_object_losses'], row['scalar_losses']
        assert len(matrix) == len(scalar) == 8
        for vector, total in zip(matrix, scalar):
            assert len(vector) == 4 and all(math.isfinite(x) and x >= 0 for x in vector)
            assert math.isfinite(total) and abs(mean(vector)-total) <= 1e-7+1e-5*abs(total)
        for i, oid in enumerate(fit['input_batches'][batch]['object_ids']):
            values[step].setdefault(oid, {}).setdefault(condition, []).extend(v[i] for v in matrix)
    for step, objects in values.items():
        assert set(objects) == set(fit['plan']['held_object_ids']) and len(objects) == 16
        for oid, cells in objects.items():
            assert set(cells) == set(conditions(key)) and all(len(v) == 16 for v in cells.values())
            objects[oid] = {c: mean(v) for c, v in cells.items()}
            if key != 'image':
                objects[oid]['wrong_mean'] = mean(objects[oid][f'wrong_{i}'] for i in (1, 2, 3))
    return values


def paired(values):
    return dict(mean_delta=mean(values), median_delta=median(values),
                objects_negative=sum(d < 0 for d in values), objects_positive=sum(d > 0 for d in values), objects=len(values))


def analyze(reports, fits):
    assert set(reports) == set(fits) == set(MODELS)
    values = {key: validate(reports[key], fits[key]) for key in MODELS}
    for key in MODELS:
        assert reports[key]['settings']['model_key'] == key
        assert reports[key]['source_sha256'] == reports['image']['source_sha256']
        for a, b in zip(reports[key]['input_batches'], reports['image']['input_batches']):
            assert {k: v for k, v in a.items() if k != 'features_sha256'} == b
    summary = {}
    for step in (256, 1024):
        ids = sorted(values['image'][step])
        cells, comparisons = {}, {}
        for key in MODELS:
            objects = values[key][step]
            cells[key] = {c: mean(v[c] for v in objects.values()) for c in next(iter(objects.values()))}
            if key != 'image':
                for name, positive, negative in [('identity_benefit', 'wrong_mean', 'correct'),
                                                   ('surface_presence_benefit', 'removed', 'correct')]:
                    comparisons[f'{key}_{name}'] = paired([objects[o][positive]-objects[o][negative] for o in ids])
                for condition in ('correct', 'removed'):
                    comparisons[f'{key}_{condition}_minus_image'] = paired([
                        objects[o][condition]-values['image'][step][o]['removed'] for o in ids])
        comparisons['oracle_minus_camera_correct'] = paired([
            values['oracle'][step][o]['correct']-values['camera'][step][o]['correct'] for o in ids])
        summary[step] = dict(cells=cells, paired_comparisons=comparisons)
    return dict(summary=summary, per_object=values,
                limits='16 previously inspected held identities, two correlated views each, one training seed. '
                       'Fresh common bank validates a checkpoint contrast chosen using earlier monitors. '
                       'Three wrong identities per object are a restricted control, not a universal test. '
                       'Removed-surface inputs are outside surface-training support and alter token count. '
                       'A removed-surface residual against image locates differences in trained shared CA/norm2; '
                       'it does not establish which alternative training architecture will fix them. '
                       'No training, target reorientation, rollout or Stage 2.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports, fits, hashes = {}, {}, {}
    for key in MODELS:
        path = args.probe_root/f'{key}.json'
        reports[key] = json.loads(path.read_text()); hashes[key] = sha(path)
        fit_path = HERE/'object_transfer_returned_manual'/key/'results.json'
        fits[key] = json.loads(fit_path.read_text())
        assert reports[key]['fit_report_sha256'] == sha(fit_path)
        for name, digest in reports[key]['source_sha256'].items():
            assert sha(REPO/name) == digest, name
    result = analyze(reports, fits); result['report_sha256'] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2))
