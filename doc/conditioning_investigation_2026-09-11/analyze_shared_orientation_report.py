"""Validate/aggregate the returned report; raw-array geometry verification is separate."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'shared_orientation_returned_manual'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, native_key, common_key):
    result = {}
    for split in ('fit', 'reserved_view'):
        selected = [r for r in rows if r['split'] == split]
        cells = {}
        for oid in ['all'] + sorted({r['object_id'] for r in selected}):
            rr = [r for r in selected if oid == 'all' or r['object_id'] == oid]
            cells[oid] = dict(n=len(rr), native_iou=mean(r[native_key] for r in rr),
                **{k: mean(r[common_key][k] for r in rr)
                   for k in ('precision_2v', 'recall_2v', 'fscore_2v')})
        result[split] = cells
    return result


def main():
    path = OUT / 'results.json'
    d = json.loads(path.read_text())
    assert d['complete'] and d['target_quality_gate_passed']
    assert d['parameters_unchanged_during_sampling']
    assert d['initial_all_parameters_sha256'] != d['final_all_parameters_sha256']
    source = {k: sha(REPO / k) == v for k, v in d['source_sha256'].items()}
    references = {k: sha(HERE / k) == v for k, v in d['reference_sha256'].items()}
    assert all(source.values()) and all(references.values())
    old = json.loads((HERE / 'visual_dropout_returned_manual/camera/results.json').read_text())
    paired = json.loads((HERE / 'camera_dropout_sampling_reference.json').read_text())
    assert d['initial_all_parameters_sha256'] == old['initial_all_parameters_sha256']
    assert d['dropout_schedule_sha256'] == old['dropout_schedule_sha256']
    for current, historical in zip(d['input_batches'], old['input_batches'], strict=True):
        assert {k: v for k, v in current.items() if k != 'shared_target_sha256'} == historical
    assert len(d['training']) == 1000
    for current, historical in zip(d['training'], old['training'], strict=True):
        for key in ('step', 'group', 'visual_dropped'):
            assert current[key] == historical[key]
        assert all(math.isfinite(current[k]) for k in ('loss', 'preclip_gradient_norm'))
    assert d['interface_checks'] == {'visual_present': True, 'visual_zero': True}
    assert len(d['target_frames']) == len(d['target_quality']) == 28
    assert len(d['rows']) == 112 and len(d['artifacts']) == 28
    assert len({(r['condition'], r['group'], r['draw'], r['sample_id']) for r in d['rows']}) == 112
    for row in d['rows']:
        assert row['sample_id'] in d['input_batches'][row['group']]['sample_ids']
        assert row['split'] == ('fit' if row['group'] < 4 else 'reserved_view')
        assert row['noise_sha256'] == paired['noise'][row['object_id'] + '/' + str(row['draw'])]
        assert row['predicted_occupied'] > 0 and row['target_occupied'] > 0
        assert math.isfinite(row['latent_mse']) and 0 <= row['voxel_iou'] <= 1
    cells = {}
    for condition in ('correct_surface', 'wrong_surface'):
        cells['shared/' + condition] = summarize(
            [r for r in d['rows'] if r['condition'] == condition], 'voxel_iou', 'common_object_units')
    historical_paths = ['camera_dropout_geometry_analysis/predictions.jsonl',
                        'pose_shape_analysis/predictions.jsonl']
    selections = [('camera_dropout', 'correct_surface'), ('dropout', 'aligned')]
    for file, (model, condition) in zip(historical_paths, selections, strict=True):
        rr = [json.loads(line) for line in (HERE / file).read_text().splitlines()]
        rr = [r for r in rr if r['model'] == model and r['condition'] == condition]
        assert len(rr) == 56
        name = 'camera_dropout' if model == 'camera_dropout' else 'oracle_dropout'
        cells[name] = summarize(rr, 'raw_iou', 'raw')
    losses = {str(a['step']): {
        split + '/' + ('wrong' if wrong else 'correct'): mean(
            x for row in a['rows'] if (row['group'] < 4) == (split == 'fit')
            and row['wrong'] == wrong for x in row['losses'])
        for split in ('fit', 'reserved_view') for wrong in (False, True)} for a in d['assessments']}
    delta = {split: {oid: cells['shared/correct_surface'][split][oid]['fscore_2v']
        - cells['shared/wrong_surface'][split][oid]['fscore_2v']
        for oid in cells['shared/correct_surface'][split]} for split in ('fit', 'reserved_view')}
    summary = dict(cells=cells, native_losses=losses, raw_surface_advantage_2v=delta,
        scope='Report aggregates only for shared arm. No independent raw-array replay or rigid alignment yet. '
              'Historical controls are from previously verified arrays. Native IoU uses each arm\'s own target; '
              'two-voxel proximity uses common original units for all arms. Four fitted identities, one training seed.')
    validation = dict(report_sha256=sha(path), source_matches=source, reference_matches=references,
        historical_analysis_sha256={p: sha(HERE / p) for p in historical_paths},
        complete=True, training_steps=1000, sampled_rows=112, exact_initialization_match=True,
        exact_original_inputs_all_seven_groups=True, exact_dropout_schedule=True,
        exact_sampling_noise_all_112_rows=True, parameters_unchanged_during_sampling=True,
        target_roundtrip_iou_min=min(q['roundtrip']['voxel_iou'] for q in d['target_quality']),
        all_target_common_two_voxel_precision_recall_one=all(
            q['common_object_units'][k] == 1 for q in d['target_quality']
            for k in ('precision_2v', 'recall_2v')), raw_geometry_bundle_received=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps({k: {sp: v[sp]['all'] for sp in v} for k, v in cells.items()}, indent=2))


if __name__ == '__main__':
    main()
