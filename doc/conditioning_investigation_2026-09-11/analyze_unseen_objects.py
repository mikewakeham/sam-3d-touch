"""Paired, object-level summary of the frozen unseen-object probe."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_KEYS = ('image_original', 'camera_original', 'camera_dropout', 'oracle_original', 'oracle_dropout')


def validate(report):
    assert report['complete'] and report['anchor_replay_passed']
    key = report['settings']['model_key']
    assert key in MODEL_KEYS and report['settings']['draws'] == 8
    assert report['settings']['seed'] == 29 and report['settings']['unseen_bank_base'] == 500000
    selected = set(report['selection']['sample_ids'])
    assert len(selected) == 64
    modes = (False,) if key == 'image_original' else (False, True)
    rows = {(r['sample_id'], r['swapped_surface']): r for r in report['rows']}
    assert len(rows) == len(report['rows']) == 64 * len(modes)
    assert set(rows) == {(sid, mode) for sid in selected for mode in modes}
    objects = {}
    for sid in selected:
        row = rows[sid, False]
        objects.setdefault(row['object_id'], []).append(sid)
    assert len(objects) == 32 and all(len(sids) == 2 for sids in objects.values())
    assert not set(objects) & set(report['fitted_object_ids'])
    assert len(report['input_batches']) == 16
    for index, batch in enumerate(report['input_batches']):
        assert batch['batch'] == index and len(batch['sample_ids']) == len(batch['object_ids']) == 4
        assert len(set(batch['object_ids'])) == 4
        for i, sid in enumerate(batch['sample_ids']):
            for mode in modes:
                r = rows[sid, mode]
                assert r['batch'] == index and r['view_index'] == index // 8
                assert r['object_id'] == batch['object_ids'][i]
                expected_surface = None if key == 'image_original' else batch['object_ids'][(i-1) % 4 if mode else i]
                assert r['surface_object_id'] == expected_surface
                assert r['dataset_split'] in ('train', 'val')
                assert len(r['losses']) == 8 and all(math.isfinite(x) and x >= 0 for x in r['losses'])
    scalar = {(r['batch'], r['swapped_surface']): r['losses'] for r in report['batch_scalar_losses']}
    assert len(scalar) == len(report['batch_scalar_losses']) == 16 * len(modes)
    for index, batch in enumerate(report['input_batches']):
        for mode in modes:
            values = scalar[index, mode]
            assert len(values) == 8
            for draw, value in enumerate(values):
                recovered = mean(rows[sid, mode]['losses'][draw] for sid in batch['sample_ids'])
                assert math.isfinite(value) and abs(value-recovered) <= 1e-7+1e-5*abs(value)
    per_object = {}
    for oid, sids in objects.items():
        splits = {rows[sid, False]['dataset_split'] for sid in sids}
        assert len(splits) == 1 and {rows[sid, False]['view_index'] for sid in sids} == {0, 1}
        per_object[oid] = {'dataset_split': splits.pop(),
                          'correct_loss': mean(x for sid in sids for x in rows[sid, False]['losses'])}
        if True in modes:
            wrong = mean(x for sid in sids for x in rows[sid, True]['losses'])
            per_object[oid]['wrong_loss'] = wrong
            per_object[oid]['surface_benefit'] = wrong-per_object[oid]['correct_loss']
    return per_object


def analyze(reports):
    assert set(reports) == set(MODEL_KEYS)
    per_object = {key: validate(reports[key]) for key in MODEL_KEYS}
    image = reports['image_original']
    for key, report in reports.items():
        assert report['settings']['model_key'] == key
        assert report['selection_sha256'] == image['selection_sha256'] and report['selection'] == image['selection']
        for a, b in zip(report['input_batches'], image['input_batches']):
            assert {k: v for k, v in a.items() if k != 'features_sha256'} == b
        for oid in per_object[key]:
            assert per_object[key][oid]['dataset_split'] == per_object['image_original'][oid]['dataset_split']
    for arm in ('camera', 'oracle'):
        assert reports[f'{arm}_original']['input_batches'] == reports[f'{arm}_dropout']['input_batches']
    summaries = {}
    for split in ('all', 'train', 'val'):
        ids = sorted(oid for oid, row in per_object['image_original'].items()
                     if split == 'all' or row['dataset_split'] == split)
        assert ids
        cells = {}
        for key in MODEL_KEYS:
            rows = [per_object[key][oid] for oid in ids]
            cells[key] = {'mean_correct_loss': mean(r['correct_loss'] for r in rows),
                          'median_correct_loss': median(r['correct_loss'] for r in rows)}
            if key != 'image_original':
                cells[key].update(mean_surface_benefit=mean(r['surface_benefit'] for r in rows),
                                  objects_preferring_correct_surface=sum(r['surface_benefit'] > 0 for r in rows))
        comparisons = {}
        pairs = [(f'{arm}_dropout', f'{arm}_original') for arm in ('camera', 'oracle')]
        pairs += [(key, 'image_original') for key in MODEL_KEYS if key != 'image_original']
        pairs += [(f'camera_{policy}', f'oracle_{policy}') for policy in ('original', 'dropout')]
        for treatment, baseline in pairs:
            differences = [per_object[treatment][oid]['correct_loss']-per_object[baseline][oid]['correct_loss'] for oid in ids]
            comparisons[f'{treatment}_minus_{baseline}'] = {
                'mean_delta': mean(differences), 'median_delta': median(differences),
                'objects_improved': sum(x < 0 for x in differences), 'objects': len(ids)}
        summaries[split] = {'objects': len(ids), 'cells': cells, 'paired_comparisons': comparisons}
    return {'summary': summaries, 'per_object': per_object,
            'limits': 'No training on these 32 identities by the compared tiny finetunes; pretraining '
                      'exposure unknown. Diagnostic selection with prior use in other analyses, not '
                      'a pristine population benchmark. Object averages combine two views and eight '
                      'paired draws. Original/dropout policies each have one four-object training seed. '
                      'Failure to transfer does not imply failure after training on more identities. '
                      'Native flow losses only, no generation-quality conclusion.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe_root', type=Path)
    parser.add_argument('--reference-root', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports, hashes = {}, {}
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for key in MODEL_KEYS:
        path = args.probe_root / f'{key}.json'
        report = json.loads(path.read_text())
        reports[key] = report; hashes[key] = sha(path)
        for source, expected in report['source_sha256'].items():
            assert sha(REPO / source) == expected, source
        assert sha(args.reference_root / 'unseen_object_selection.json') == report['selection_sha256']
        arm, policy = key.split('_')
        paths = {'single': args.reference_root / f'tiny_fit_returned_46083371/{arm}/results.json',
                 'multiple': args.reference_root / f'multiple_view_returned_46083371/{arm}.json'}
        paths['trained'] = (args.reference_root / f'visual_dropout_returned_manual/{arm}/results.json'
                            if policy == 'dropout' else paths['multiple'])
        assert report['reference_sha256'] == {n: sha(p) for n, p in paths.items()}
        trained = json.loads(paths['trained'].read_text())
        assert report['final_parameters_sha256'] == trained['final_all_parameters_sha256']
        expected = (trained['final_fresh']['visual_present'][0] if policy == 'dropout'
                    else trained['assessments'][-1]['rows'][0])
        replay_keys = ['fresh_noise_native_losses'] + ([] if arm == 'image' else ['swapped_surface_native_losses'])
        assert report['anchor_replay'] == {k: expected[k] for k in replay_keys}
    result = analyze(reports); result['report_sha256'] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary']['all'], indent=2))
