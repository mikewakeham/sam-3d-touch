"""Check cross-arm pairing and summarize native fitting losses (stdlib only)."""
import json
import sys
from pathlib import Path
from statistics import mean


def check(root):
    reports = {arm: json.loads((root / arm / 'results.json').read_text())
               for arm in ['image', 'camera', 'oracle']}
    reference = reports['image']
    keys = ['sample_ids', 'initial_trainable_ca_sha256', 'image_sha256',
            'pointmap_sha256', 'target_sha256', 'source_sha256',
            'pipeline_yaml', 'generator_yaml', 'data_yaml']
    for arm, report in reports.items():
        for key in keys:
            assert report[key] == reference[key], (arm, key, 'Pairing failed')
        for key in ['steps', 'objects', 'seed', 'precision']:
            assert report['settings'][key] == reference['settings'][key], (arm, key)
        assert report['final_trainable_ca_sha256'] != report['initial_trainable_ca_sha256']
        assert len(report['assessments']) == len(reference['assessments'])
        for assessment, other in zip(report['assessments'], reference['assessments']):
            assert assessment['step'] == other['step']
            assert assessment['sample_noise_sha256'] == other['sample_noise_sha256']
    assert reports['camera']['initial_all_parameters_sha256'] == reports['oracle']['initial_all_parameters_sha256']
    print('Cross-arm pairing passed. Fresh-noise loss on the same training objects:')
    for arm, report in reports.items():
        for assessment in report['assessments']:
            wrong = assessment['swapped_surface_native_losses']
            print(arm, assessment['step'], 'correct', mean(assessment['fresh_noise_native_losses']),
                  'swapped', mean(wrong) if wrong else None)
    print('Small-set fitting and swapped-condition diagnostics; no generalization or impossibility claim.')


if __name__ == '__main__':
    check(Path(sys.argv[1]))
