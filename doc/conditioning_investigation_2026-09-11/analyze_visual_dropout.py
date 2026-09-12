"""Validate and compare the matched camera/oracle visual-dropout experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from visual_dropout_protocol import dropout_schedule, schedule_digest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOSS_KEYS = ('fresh_noise_native_losses', 'swapped_surface_native_losses')


def validate_rows(rows):
    assert len(rows) == 7
    for group, row in enumerate(rows):
        assert row['group'] == group
        assert row['split'] == ('fit' if group < 4 else 'reserved_view')
        for key in LOSS_KEYS:
            assert len(row[key]) == 8
            assert all(math.isfinite(x) and x >= 0 for x in row[key])


def validate(report, old):
    assert report['complete'] and report['baseline_preflight_passed'] and report['initial_parameters_restored']
    settings = report['settings']
    for key, value in old['settings'].items():
        assert settings[key] == value, key
    assert settings['visual_dropout_fraction'] == .5 and settings['dropout_unit'] == 'whole_batch'
    assert settings['fresh_bank_base'] == 400000
    schedule = dropout_schedule(settings['seed'])
    assert report['dropout_schedule'] == schedule
    assert report['dropout_schedule_sha256'] == schedule_digest(schedule)
    assert report['initial_all_parameters_sha256'] == old['initial_all_parameters_sha256']
    assert report['initial_trainable_ca_sha256'] == old['initial_trainable_ca_sha256']
    assert report['baseline_final_parameters_sha256'] == old['final_all_parameters_sha256']
    assert report['final_all_parameters_sha256'] != report['initial_all_parameters_sha256']
    assert report['input_batches'] == old['input_batches']
    assert report['baseline_preflight']['initial'] == old['assessments'][0]['rows']
    assert report['baseline_preflight']['final'] == old['assessments'][-1]['rows']
    assert set(report['interface_checks']) == {'visual_zero_surface_present', 'visual_present_surface_present'}
    for check in report['interface_checks'].values():
        assert check['passed']
        assert check['context_shape'][1] == check['visual_tokens'] + check['surface_tokens']
        assert check['surface_tokens'] == 1024
    assert len(report['training']) == 1000
    for step, row in enumerate(report['training'], 1):
        assert row['step'] == step and row['group'] == (step - 1) % 4
        assert row['visual_dropped'] == schedule[step - 1]
        assert math.isfinite(row['loss']) and row['loss'] >= 0
        assert math.isfinite(row['preclip_gradient_norm']) and row['preclip_gradient_norm'] >= 0
    assert [a['step'] for a in report['assessments']] == [0, 100, 300, 1000]
    for assessment in report['assessments']:
        validate_rows(assessment['rows'])
    assert report['assessments'][0]['rows'] == old['assessments'][0]['rows']
    for phase in ('baseline_fresh', 'final_fresh'):
        assert set(report[phase]) == {'visual_present', 'visual_zero'}
        for rows in report[phase].values():
            validate_rows(rows)


def summarize(rows, groups):
    correct = mean(v for g in groups for v in rows[g][LOSS_KEYS[0]])
    wrong = mean(v for g in groups for v in rows[g][LOSS_KEYS[1]])
    return {'correct_surface_loss': correct, 'wrong_surface_loss': wrong,
            'surface_benefit_wrong_minus_correct': wrong - correct}


def compare(reports, references):
    for arm in ('camera', 'oracle'):
        validate(reports[arm], references[arm])
    assert reports['camera']['dropout_schedule_sha256'] == reports['oracle']['dropout_schedule_sha256']
    assert reports['camera']['initial_all_parameters_sha256'] == reports['oracle']['initial_all_parameters_sha256']
    assert reports['camera']['source_sha256'] == reports['oracle']['source_sha256']
    for camera, oracle in zip(reports['camera']['input_batches'], reports['oracle']['input_batches']):
        assert {k: v for k, v in camera.items() if k != 'features_sha256'} == {
            k: v for k, v in oracle.items() if k != 'features_sha256'}
    result = {'fresh_bank': {}, 'historical_bank_curves': {}, 'per_group_all_inputs': {}}
    for mode in ('visual_present', 'visual_zero'):
        result['fresh_bank'][mode] = {}
        for split, groups in [('fit', range(4)), ('reserved_view', range(4, 7))]:
            cells = {f'{arm}_{policy}': summarize(reports[arm][phase][mode], groups)
                     for arm in ('camera', 'oracle')
                     for policy, phase in [('original', 'baseline_fresh'), ('dropout', 'final_fresh')]}
            c0, c1, o0, o1 = [cells[k]['correct_surface_loss'] for k in
                              ('camera_original', 'camera_dropout', 'oracle_original', 'oracle_dropout')]
            result['fresh_bank'][mode][split] = {
                'cells': cells,
                'dropout_minus_original': {'camera': c1-c0, 'oracle': o1-o0},
                'camera_minus_oracle': {'original': c0-o0, 'dropout': c1-o1},
                'interaction_change_in_camera_oracle_gap': (c1-o1)-(c0-o0)}
    for arm in ('camera', 'oracle'):
        result['historical_bank_curves'][arm] = {
            policy: [{'step': a['step'], 'fit': summarize(a['rows'], range(4)),
                      'reserved_view': summarize(a['rows'], range(4, 7))} for a in assessments]
            for policy, assessments in [('original', references[arm]['assessments']),
                                         ('dropout', reports[arm]['assessments'])]}
        result['per_group_all_inputs'][arm] = [
            {'group': g, 'original': summarize(reports[arm]['baseline_fresh']['visual_present'], [g]),
             'dropout': summarize(reports[arm]['final_fresh']['visual_present'], [g])} for g in range(7)]
    result['limits'] = ('Primary: coherent all-input visual_present assessment on fresh paired draws. '
                        'visual_zero is a diagnostic, not the deployment criterion. Four fitted objects, '
                        'one training seed, finite budget. Mixed-objective training losses are not directly '
                        'comparable with original visual-present training losses. No per-object loss, '
                        'unseen-object result, rollout result or inherent-limit conclusion.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit_root', type=Path)
    parser.add_argument('--reference-root', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports, refs, hashes = {}, {}, {}
    for arm in ('camera', 'oracle'):
        path = args.fit_root / arm / 'results.json'
        reports[arm] = json.loads(path.read_text())
        hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name, expected in reports[arm]['source_sha256'].items():
            assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected, name
        for key, relative in [('single', f'tiny_fit_returned_46083371/{arm}/results.json'),
                              ('multiple', f'multiple_view_returned_46083371/{arm}.json')]:
            refpath = args.reference_root / relative
            assert hashlib.sha256(refpath.read_bytes()).hexdigest() == reports[arm]['reference_sha256'][key]
        refs[arm] = json.loads((args.reference_root / f'multiple_view_returned_46083371/{arm}.json').read_text())
    result = compare(reports, refs)
    result['report_sha256'] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['fresh_bank']['visual_present'], indent=2))
