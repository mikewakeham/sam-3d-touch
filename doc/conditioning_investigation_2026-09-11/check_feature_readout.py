"""Validate paired readout reports and summarize fit versus separate objects."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def analyze(reports):
    assert set(reports) == {'raw', 'raw_slot', 'processed'}
    reference = reports['raw']
    paired = ['sample_ids', 'fit_sample_ids', 'reserved_object_ids', 'target_sha256',
              'raw_features_sha256', 'source_report_sha256', 'initial_common_parameters_sha256', 'driver_sha256']
    summary = {}
    for arm, r in reports.items():
        assert r.get('complete') and r['settings']['arm'] == arm
        for key in paired:
            assert r[key] == reference[key], (arm, key)
        for key in ['steps', 'learning_rate', 'seed']:
            assert r['settings'][key] == reference['settings'][key], (arm, key)
        assert r['start_step'] == reference['start_step']
        ids = r['sample_ids']
        assert len(ids) == len(set(ids)) == 32
        assert r['fit_sample_ids'] == ids[:24] and r['reserved_object_ids'] == ids[24:]
        assert r['assessments'][-1]['step'] == r['settings']['steps']
        curves = []
        for assessment in r['assessments']:
            rows = assessment['objects']
            assert [x['sample_id'] for x in rows] == ids
            assert all(x['split'] == ('fit' if i < 24 else 'reserved_object') for i, x in enumerate(rows))
            assert all(math.isfinite(x[k]) and x[k] >= 0 for x in rows for k in ['latent_mse', 'wrong_object_latent_mse'])
            item = {'step': assessment['step']}
            for split, selection in [('fit', rows[:24]), ('reserved_object', rows[24:])]:
                item[split] = {k: mean(x[k] for x in selection) for k in ['latent_mse', 'wrong_object_latent_mse']}
                if all('decoded_vs_target' in x for x in selection):
                    assert all(0 <= x['decoded_vs_target']['iou'] <= 1 for x in selection)
                    item[split]['decoded_iou'] = mean(x['decoded_vs_target']['iou'] for x in selection)
            curves.append(item)
        final = r['assessments'][-1]['objects']
        assert all('decoded_vs_target' in x for x in final)
        summary[arm] = {'curves': curves, 'final_objects': final,
                        'trainable_parameters': r['trainable_parameters'],
                        'fit_mean_control_fit_mse': mean(r['fit_mean_target_control_mse'][:24]),
                        'fit_mean_control_reserved_mse': mean(r['fit_mean_target_control_mse'][24:])}
    assert reports['raw']['readout_features_sha256'] == reports['raw_slot']['readout_features_sha256']
    return {'arms': summary, 'limits': 'One seed, 24 fitted / 8 reserved previously screened objects, oracle frame. A finite failed readout fit does not establish information loss. Processed arm has a larger input projection and native pretrained computation; it is not a matched-capacity isolation of slot identity. Raw versus raw_slot is the narrower comparison.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fit_root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = analyze({arm: json.loads((a.fit_root / arm / 'results.json').read_text()) for arm in ['raw', 'raw_slot', 'processed']})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for arm, r in result['arms'].items():
        print(arm, r['curves'][-1])
