"""Validate the bounded object-disjoint training comparison; no torch required."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from object_transfer_protocol import MODEL_KEYS, digest_json, make_schedule, validate_plan

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPLITS = ('fit', 'held_view', 'held_object')


def object_losses(report, rows, draws, swaps):
    expected = {(b, swap) for b in range(32) for swap in
                ((False, True) if swaps and report['settings']['model_key'] != 'image' else (False,))}
    mapping = {(r['batch'], r['swapped_surface']): r for r in rows}
    assert len(mapping) == len(rows) and set(mapping) == expected
    values = {split: {} for split in SPLITS}
    for (batch, swap), row in mapping.items():
        meta = report['input_batches'][batch]
        assert row['split'] == meta['split']
        matrix = row['per_object_losses']; scalar = row['scalar_losses']
        assert len(matrix) == len(scalar) == draws
        for vector, total in zip(matrix, scalar):
            assert len(vector) == 4 and all(math.isfinite(x) and x >= 0 for x in vector)
            assert math.isfinite(total) and abs(mean(vector)-total) <= 1e-7+1e-5*abs(total)
        for i, oid in enumerate(meta['object_ids']):
            entry = values[row['split']].setdefault(oid, {'correct': [], 'wrong': []})
            entry['wrong' if swap else 'correct'].extend(vector[i] for vector in matrix)
    return {split: {oid: {k: mean(v) for k, v in entry.items() if v} for oid, entry in objects.items()}
            for split, objects in values.items()}


def validate(report):
    assert report['complete']
    validate_plan(report['plan'])
    settings = report['settings']; key = settings['model_key']
    assert key in MODEL_KEYS and settings['seed'] == 37 and settings['steps'] == 1024
    assert settings['precision'] == 'bf16'
    assert settings['visual_dropout_fraction'] == (.5 if key == 'oracle_dropout' else 0)
    assert (settings['monitor_bank_base'], settings['monitor_draws'], settings['final_bank_base'], settings['final_draws']) == (600000, 4, 700000, 8)
    schedule = make_schedule(37)
    assert report['schedule'] == schedule and report['schedule_sha256'] == digest_json(schedule)
    assert len(report['input_batches']) == 32 and len(report['training']) == 1024
    target_hashes = {}
    for actual, expected in zip(report['input_batches'], report['plan']['batches']):
        for k, v in expected.items():
            assert actual[k] == v
        assert len(actual['object_target_sha256']) == 4
        for oid, digest in zip(actual['object_ids'], actual['object_target_sha256']):
            assert target_hashes.setdefault(oid, digest) == digest
    for step, (row, scheduled) in enumerate(zip(report['training'], schedule), 1):
        assert row['step'] == step and row['batch'] == scheduled['batch']
        assert row['visual_dropped'] == (key == 'oracle_dropout' and scheduled['drop_visual'])
        assert math.isfinite(row['loss']) and row['loss'] >= 0
        assert math.isfinite(row['preclip_gradient_norm']) and row['preclip_gradient_norm'] >= 0
    assert [r['step'] for r in report['assessments']] == [0, 256, 512, 1024]
    for assessment in report['assessments']:
        object_losses(report, assessment['rows'], 4, False)
    assert report['final_model_sha256'] != report['initial_model_sha256']
    assert key != 'oracle_dropout' or report['dropout_context_verified']
    return object_losses(report, report['final_rows'], 8, True)


def analyze(reports):
    assert set(reports) == set(MODEL_KEYS)
    result = {'summary': {}, 'curves': {}, 'per_object': {key: validate(r) for key, r in reports.items()}}
    image = reports['image']
    for key, r in reports.items():
        assert r['settings']['model_key'] == key
        for field in ('plan', 'plan_sha256', 'schedule_sha256', 'source_sha256', 'reference_sha256',
                      'initial_generator_sha256', 'initial_trainable_ca_sha256'):
            assert r[field] == image[field], field
        assert r['trainable_parameters']['cross_attention'] == image['trainable_parameters']['cross_attention']
        for a, b in zip(r['input_batches'], image['input_batches']):
            assert {k: v for k, v in a.items() if k != 'features_sha256'} == b
    assert len({reports[k]['initial_encoder_sha256'] for k in ('camera', 'oracle', 'oracle_dropout')}) == 1
    assert reports['oracle']['input_batches'] == reports['oracle_dropout']['input_batches']
    for split in SPLITS:
        ids = sorted(result['per_object']['image'][split])
        assert len(ids) == 16
        cells = {}
        for key in MODEL_KEYS:
            values = result['per_object'][key][split]
            cells[key] = {'mean_correct_loss': mean(v['correct'] for v in values.values()),
                          'median_correct_loss': median(v['correct'] for v in values.values())}
            if key != 'image':
                deltas = [v['wrong']-v['correct'] for v in values.values()]
                cells[key].update(mean_surface_benefit=mean(deltas),
                                  objects_preferring_correct_surface=sum(d > 0 for d in deltas))
        comparisons = {}
        for treatment, baseline in [('camera', 'image'), ('oracle', 'image'), ('oracle_dropout', 'image'),
                                     ('oracle', 'camera'), ('oracle_dropout', 'oracle')]:
            delta = [result['per_object'][treatment][split][oid]['correct']-
                     result['per_object'][baseline][split][oid]['correct'] for oid in ids]
            comparisons[f'{treatment}_minus_{baseline}'] = {'mean_delta': mean(delta), 'median_delta': median(delta),
                                                           'objects_improved': sum(d < 0 for d in delta), 'objects': 16}
        result['summary'][split] = {'cells': cells, 'paired_comparisons': comparisons}
    for key, report in reports.items():
        result['curves'][key] = []
        for assessment in report['assessments']:
            values = object_losses(report, assessment['rows'], 4, False)
            result['curves'][key].append({'step': assessment['step'], **{
                split: mean(v['correct'] for v in values[split].values()) for split in SPLITS}})
    result['limits'] = ('One matched seed, 16 fitted and 16 held identities, finite 1024-update budget. '
                        'Previously inspected development pool. Distinct held-view and held-object outcomes; '
                        'not an isolated causal estimate of dataset size versus older four-object runs. '
                        'Primary: coherent all-input losses. Mixed-objective training losses are not directly '
                        'comparable. No generated geometry/rollout or inherent-limitation conclusion.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports, hashes = {}, {}
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for key in MODEL_KEYS:
        path = args.fit_root/key/'results.json'
        reports[key] = json.loads(path.read_text()); hashes[key] = sha(path)
        for source, digest in reports[key]['source_sha256'].items():
            assert sha(REPO/source) == digest, source
        assert reports[key]['plan_sha256'] == sha(HERE/'object_transfer_plan.json')
        assert reports[key]['reference_sha256'] == sha(HERE/'tiny_fit_returned_46083371/oracle/results.json')
    result = analyze(reports); result['report_sha256'] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2))
