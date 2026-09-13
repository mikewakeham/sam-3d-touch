"""Validate supplied shards without manufacturing missing data.

Copied from the original analyzer; changes select existing files, retain actual shard
IDs, and replace full aggregation with comparison of available cells to the supplied
aggregate plus arithmetic checks. Original source files remain unchanged for replay.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import statistics
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_tolerance_protocol import cases, rotation

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root):
    paths = [root/str(i)/'results.json' for i in range(4) if (root/str(i)/'results.json').is_file()]
    assert paths and paths[0].parent.name == '0'
    reports = [json.loads(p.read_text()) for p in paths]
    multi = json.loads((HERE/'multiple_view_returned_46083371/oracle.json').read_text())
    drop = json.loads((HERE/'visual_dropout_returned_manual/oracle/results.json').read_text())
    expected_refs = {name: sha(HERE/path) for name, path in {
        'tiny_fit_returned_46083371/oracle/results.json': 'tiny_fit_returned_46083371/oracle/results.json',
        'oracle.json': 'multiple_view_returned_46083371/oracle.json',
        'visual_dropout_returned_manual/oracle/results.json': 'visual_dropout_returned_manual/oracle/results.json'}.items()}
    rows, paired_noise = [], {}
    for path, report in zip(paths, reports):
        shard = int(path.parent.name)
        assert report['complete'] and report['parameters_unchanged']
        assert report['settings'] == {'shard': shard, 'seed': 29, 'precision': 'bf16', 'steps': 25,
                                     'cfg': 0, 'sampling_bank': 200000, 'sampling_draws': 2,
                                     'cases': [list(c) for c in cases(shard)]}
        assert report['reference_sha256'] == expected_refs
        assert report['input_batches'] == multi['input_batches'] == drop['input_batches']
        assert report['source_sha256'] == reports[0]['source_sha256']
        for name, value in report['source_sha256'].items():
            assert sha(REPO/name) == value, name
        policies = {c[0] for c in cases(shard)}
        assert set(report['checkpoint_checks']) == policies
        for policy in policies:
            expected = multi if policy == 'original' else drop
            assert report['checkpoint_checks'][policy]['parameter_sha256'] == expected['final_all_parameters_sha256']
            if policy == 'dropout':
                assert report['checkpoint_checks'][policy] == reports[0]['checkpoint_checks'][policy]
        replay_keys = {(p, g, w) for p in policies for g in range(7) for w in (False, True)}
        assert len(report['replay']) == len(replay_keys)
        for replay in report['replay']:
            key = replay['policy'], replay['group'], replay['wrong']
            assert key in replay_keys; replay_keys.remove(key)
            old = (drop['baseline_fresh'] if replay['policy'] == 'original' else drop['final_fresh'])['visual_present'][replay['group']]
            expected = old['swapped_surface_native_losses' if replay['wrong'] else 'fresh_noise_native_losses']
            assert len(replay['losses']) == len(expected) == 8
            assert all(math.isfinite(a) and abs(a-b) <= 1e-7+1e-5*abs(b) for a,b in zip(replay['losses'], expected))
        coordinate_keys = {(p,c,g) for p,c,_,_ in cases(shard) for g in range(7)}
        assert len(report['coordinate_checks']) == len(coordinate_keys)
        expected_angles = {(p,c): abs(d) for p,c,_,d in cases(shard)}
        treatments = {(p,c): (axis,d) for p,c,axis,d in cases(shard)}
        for record in report['coordinate_checks']:
            key = record['policy'], record['condition'], record['group']
            assert key in coordinate_keys; coordinate_keys.remove(key)
            assert abs(record['actual_angle_degrees'] - expected_angles[key[:2]]) < .002
            assert record['rotation'] == rotation(*treatments[key[:2]])
            assert len(record['point_displacement_rms']) == 4
            assert all(math.isfinite(v) and v >= 0 for v in record['point_displacement_rms'])
            if expected_angles[key[:2]] == 0:
                assert record['point_displacement_rms'] == [0.,0.,0.,0.]
                assert record['features_sha256'] == multi['input_batches'][record['group']]['features_sha256']
        expected_keys = {(p,c,g,d,sid) for p,c,_,_ in cases(shard) for g in range(7) for d in range(2)
                         for sid in multi['input_batches'][g]['sample_ids']}
        assert len(report['rows']) == len(expected_keys)
        for row in report['rows']:
            key = row['policy'], row['condition'], row['group'], row['draw'], row['sample_id']
            assert key in expected_keys; expected_keys.remove(key)
            assert row['split'] == ('fit' if row['group'] < 4 else 'reserved_view')
            assert (row['axis'], row['degrees']) == treatments[key[:2]]
            assert row['object_id'] == row['sample_id'].rsplit('_',1)[0]
            assert math.isfinite(row['voxel_iou']) and 0 <= row['voxel_iou'] <= 1
            assert math.isfinite(row['latent_mse']) and row['latent_mse'] >= 0
            assert row['target_occupied'] > 0 and row['predicted_occupied'] >= 0
            noise_key = row['object_id'], row['draw']
            assert paired_noise.setdefault(noise_key, row['noise_sha256']) == row['noise_sha256']
        rows.extend(report['rows'])
    # Same sampler/seeds/inputs as the original multi-view geometry assessment.
    old = {(r['group'], r['noise_draw'], r['sample_id']): r for r in multi['sampled']}
    for row in rows:
        if row['policy'] != 'original':
            continue
        previous = old[row['group'], row['draw'], row['sample_id']]
        assert abs(row['voxel_iou']-previous['voxel_iou']) <= 1e-4, 'Original sampled geometry did not replay'
        assert abs(row['latent_mse']-previous['latent_mse']) <= 1e-6+1e-4*abs(previous['latent_mse'])
    supplied = json.loads((root/'analysis.json').read_text())
    grouped = {}
    for row in rows:
        key = row['split'], row['policy']+'/'+row['condition'], row['object_id']
        grouped.setdefault(key, []).append(row['voxel_iou'])
    cells = {}
    for (split, condition, obj), values in grouped.items():
        cells.setdefault(split, {}).setdefault(condition, {})[obj] = statistics.mean(values)
    for split, conditions in cells.items():
        for condition, objects in conditions.items():
            actual = {'mean_object_iou': statistics.mean(objects.values()),
                      'minimum_object_mean_iou': min(objects.values()), 'object_iou': objects}
            assert supplied['cells'][split][condition] == actual
    for path in paths:
        assert supplied['report_sha256'][path.parent.name] == sha(path)
    baseline = supplied['cells']['reserved_view']['dropout/aligned']['object_iou']
    reference = statistics.mean(baseline.values()) >= .95 and min(baseline.values()) >= .90
    assert supplied['aligned_reference_meets_95_mean_90_each'] == reference
    for angle in ('5', '15', '30'):
        checks = []
        for name, entry in supplied['rotation_tolerance'][angle]['directions'].items():
            objects = supplied['cells']['reserved_view']['dropout/'+name]['object_iou']
            delta = {obj: objects[obj]-baseline[obj] for obj in baseline}
            passed = min(delta.values()) >= -.02 and min(objects.values()) >= .90
            assert entry == {'mean_iou_change': statistics.mean(delta.values()),
                             'minimum_object_mean_iou': min(objects.values()),
                             'object_iou_change': delta, 'passes_sampled_direction': passed}
            checks.append(passed)
        assert supplied['rotation_tolerance'][angle]['passes_all_sampled_directions'] == (reference and all(checks))
    result = {'available_shards': [int(p.parent.name) for p in paths],
              'missing_shards': [i for i in range(4) if not (root/str(i)/'results.json').is_file()],
              'scope': 'Reuses original analyzer validation logic for supplied shards only. '
                       'All supplied aggregate gate arithmetic checked; missing shard raw rows cannot be verified.'}
    result['report_sha256'] = {p.parent.name: sha(p) for p in paths}
    result['validation'] = 'Available shards: complete unique coverage, source/reference hashes, historical native replay, '
    result['validation'] += 'original sampled replay, correct input grouping and common sampling noise passed.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
