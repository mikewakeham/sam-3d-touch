"""Validate four coordinate cells and report conditional effects, not attribution percentages."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def effects(loss):
    o, r, n, c = [loss[k] for k in ['oracle', 'rotation_only', 'normalization_only', 'camera']]
    return {'rotation_with_object_normalization': r - o,
            'rotation_with_camera_normalization': c - n,
            'normalization_in_object_axes': n - o,
            'normalization_in_camera_axes': c - r,
            'interaction': c - r - n + o,
            'total_camera_minus_oracle': c - o}


def analyze(reports):
    names = ['oracle', 'rotation_only', 'normalization_only', 'camera']
    assert set(reports) == set(names)
    reference = reports['oracle']
    curves = {}
    for arm in names:
        r = reports[arm]
        assert r['settings']['arm'] == arm
        for key in ['steps', 'seed', 'precision', 'objects', 'fit_views_per_object', 'reserved_views_per_object']:
            assert r['settings'][key] == reference['settings'][key], (arm, key)
        for key in ['initial_all_parameters_sha256', 'initial_trainable_ca_sha256', 'source_sha256', 'sample_noise_sha256']:
            assert r[key] == reference[key], (arm, key)
        assert r['final_all_parameters_sha256'] != r['initial_all_parameters_sha256']
        assert len(r['input_batches']) == 7
        for group, batch in enumerate(r['input_batches']):
            old = reference['input_batches'][group]
            for key in ['group', 'split', 'sample_ids', 'image_sha256', 'pointmap_sha256', 'target_sha256']:
                assert batch[key] == old[key], (arm, group, key)
            if arm in ['rotation_only', 'normalization_only']:
                for base in ['camera', 'oracle']:
                    assert batch['features_sha256'][base] == reports[base]['input_batches'][group]['features_sha256']
        if arm in ['rotation_only', 'normalization_only']:
            assert r['complete'] and r['baseline_preflight_passed']
            for base in ['camera', 'oracle']:
                assert r['baseline_preflight'][base] == reports[base]['assessments'][0]['rows']
        train = r['training']
        assert [t['step'] for t in train] == list(range(1, 1001))
        assert all(t['group'] == (t['step'] - 1) % 4 for t in train)
        assert all(math.isfinite(t[k]) for t in train for k in ['loss', 'preclip_gradient_norm'])
        assert all(t.get('gradients/vecsetx_encoder', 0) == 0 for t in train)
        assert [x['step'] for x in r['assessments']] == [0, 100, 300, 1000]
        curves[arm] = []
        for a in r['assessments']:
            assert [x['group'] for x in a['rows']] == list(range(7))
            for row in a['rows']:
                assert row['split'] == ('fit' if row['group'] < 4 else 'reserved_view')
                for key in ['fresh_noise_native_losses', 'swapped_surface_native_losses']:
                    assert len(row[key]) == 8 and all(math.isfinite(v) and v >= 0 for v in row[key])
            item = {'step': a['step']}
            for split in ['fit', 'reserved_view']:
                rows = [x for x in a['rows'] if x['split'] == split]
                item[split] = {'loss': mean(v for x in rows for v in x['fresh_noise_native_losses']),
                    'swapped_loss': mean(v for x in rows for v in x['swapped_surface_native_losses']),
                    'group_losses': [mean(x['fresh_noise_native_losses']) for x in rows]}
            curves[arm].append(item)
        expected_samples = {(b['group'], sid, draw) for b in r['input_batches'] for sid in b['sample_ids'] for draw in [0, 1]}
        assert len(r['sampled']) == 56
        assert {(s['group'], s['sample_id'], s['noise_draw']) for s in r['sampled']} == expected_samples
        assert all(s['cfg'] == 0 and math.isfinite(s['latent_mse']) and 0 <= s['voxel_iou'] <= 1 for s in r['sampled'])
    contrasts = []
    for i, step in enumerate([0, 100, 300, 1000]):
        contrasts.append({'step': step, **{split: effects({arm: curves[arm][i][split]['loss'] for arm in names})
                                          for split in ['fit', 'reserved_view']}})
    return {'curves': curves, 'contrasts': contrasts,
            'limits': 'Four objects, one training seed, same-object reserved views. Positive differences mean higher loss. '
                      'Report conditional effects and interaction separately; input coordinate magnitudes are not causal contribution percentages. '
                      'A falling finite-budget curve is not an architectural ceiling. Baseline replay checks do not independently rerun historical optimization.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fit_root', type=Path)
    p.add_argument('--reference-root', type=Path, default=HERE)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    paths = {arm: a.reference_root / 'multiple_view_returned_46083371' / f'{arm}.json' for arm in ['oracle', 'camera']}
    paths.update({arm: a.fit_root / arm / 'results.json' for arm in ['rotation_only', 'normalization_only']})
    reports = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    for arm in ['rotation_only', 'normalization_only']:
        assert reports[arm]['driver_sha256'] == hashlib.sha256((HERE / 'fit_frame_factors_gpu.py').read_bytes()).hexdigest()
        for base in ['camera', 'oracle']:
            assert reports[arm]['reference_sha256'][base]['multiple'] == hashlib.sha256(paths[base].read_bytes()).hexdigest()
    result = analyze(reports)
    result['report_sha256'] = {arm: hashlib.sha256(path.read_bytes()).hexdigest() for arm, path in paths.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['contrasts'][-1], indent=2))


if __name__ == '__main__':
    main()
