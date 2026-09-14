"""Check cross-arm pairing and summarize native fitting losses (stdlib only)."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

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
