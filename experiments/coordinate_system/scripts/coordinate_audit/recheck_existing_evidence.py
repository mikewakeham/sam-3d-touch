"""Read-only checks of existing returned evidence; no model runs or registration."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
INV = HERE.parent
REPO = next(p for p in Path(__file__).resolve().parents if (_source_path(p, 'train.py')).is_file() and (p / 'sam3d_objects').is_dir())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    frame = _source_path(REPO, 'experiments/coordinate_system/outputs/full_frame_probe_20260913_172925')
    old = _source_path(INV, 'oracle_upper_bound')
    roots = {'camera': frame / 'camera', **{
        arm: old / 'rollouts_returned_20260913_143515' / arm
        for arm in ('oracle', 'constant')}}
    reports = {arm: read(root / 'results.json') for arm, root in roots.items()}
    rows = []
    verified_payloads = 0
    for arm, report in reports.items():
        assert report['complete'] and report['adapted_parameters_unchanged']
        for name, digest in report['files_sha256'].items():
            assert Path(name).name == name
            assert sha(roots[arm] / name) == digest, (arm, name)
            verified_payloads += 1
        # Recompute IoU/counts from occupancy bytes, not the old summary.
        cache = {}
        for row in report['samples']:
            for field, key in [('prediction_file', 'predicted_occupancy'),
                               ('target_file', 'target_occupancy')]:
                name = row[field]
                if name not in cache:
                    with np.load(roots[arm] / name, allow_pickle=False) as data:
                        cache[name] = (data[key].copy(), list(data['sample_ids']))
            i = row['array_index']
            pred, ids = cache[row['prediction_file']]
            target, tids = cache[row['target_file']]
            assert ids[i] == tids[i] == row['sample_id']
            a, b = pred[i], target[i]
            value = float(np.logical_and(a, b).sum() / np.logical_or(a, b).sum())
            assert abs(value - row['iou']) < 1e-12
            assert int(a.sum()) == row['predicted_count'] and int(b.sum()) == row['target_count']
            rows.append({'arm': arm, **row})
    # Reaggregate prior pose-search outputs. Do not pretend this reruns the search.
    geometry = [json.loads(line) for path in [
        old / 'rollouts_geometry_20260913_143515/predictions.jsonl',
        frame / 'local_analysis/camera_predictions.jsonl']
        for line in path.read_text().splitlines()]
    key = lambda r: tuple(r[x] for x in ['arm', 'split', 'group', 'sample_id', 'visual', 'surface_shift', 'draw'])
    source = {key(r): r for r in rows}
    assert len(source) == len(rows) == len(geometry) == 1280
    assert len({key(r) for r in geometry}) == len(geometry)
    for row in geometry:
        assert all(row[k] == v for k, v in source[key(row)].items())
    cells = defaultdict(lambda: defaultdict(list))
    for row in geometry:
        k = '/'.join(map(str, [row[x] for x in ['split', 'visual', 'arm', 'surface_shift']]))
        cells[k][row['object_id']].append(row)
    measured = {}
    for cell, objects in cells.items():
        assert len(objects) == 16 and all(len(r) == 4 for r in objects.values())
        measured[cell] = {'objects': 16, 'draws_per_object': 4,
            'raw_iou': mean(mean(r['iou'] for r in rs) for rs in objects.values()),
            **{kind + '_fscore_2v': mean(mean(r[kind]['fscore_2v'] for r in rs)
                                       for rs in objects.values()) for kind in ('raw', 'witness')}}
    normalization = []
    for batch in reports['camera']['inputs']:
        with np.load(roots['camera'] / f"{batch['split']}_g{batch['group']}_coordinate_inputs.npz") as data:
            for i, sid in enumerate(data['sample_ids']):
                assert data['mask'][i].all()
                p = data['oracle_pre_encoder_points'][i].astype(float)
                q = data['oracle_vecsetx_points'][i].astype(float)
                c = (p.max(0) + p.min(0))/2
                r = np.linalg.norm(p-c, axis=1).max()
                q_expected = (p-c)/r
                canonical = q / np.ptp(q, axis=0).max()
                # Explicitly distinguish numeric replay from missing-extrema displacement.
                normalization.append({'sample_id': str(sid),
                    'numeric_max_error': float(np.abs(q-q_expected).max()),
                    'geometry_recovery_max_displacement_voxels': float(np.linalg.norm(canonical-p, axis=1).max()*64)})
    assert len(normalization) == 64
    assert max(r['numeric_max_error'] for r in normalization) < 5e-6
    ff = read(_source_path(INV, 'frame_factors_returned_manual/analysis.json'))
    factors = {arm: curve[-1]['reserved_view']['loss'] for arm, curve in ff['curves'].items()}
    tiny = read(_source_path(INV, 'tiny_fit_summary.json'))
    tiny_means = {}
    for arm, data in tiny['arms'].items():
        for name, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict) and 'voxel_iou' in value[0]:
                selected = [r['voxel_iou'] for r in value if r['cfg'] == 0]
                tiny_means[arm] = mean(selected)
    original = read(_source_path(REPO, 'experiments/coordinate_system/outputs/full_frame_probe_20260913_170408/local_reference_audit.json'))
    assert len(original['objects']) == 32
    assert all(r['target_max_error'] == 0 for r in original['objects'])
    inventory = []
    for path in sorted((REPO.parent / 'wandb-results').glob('*/run.json')):
        d = read(path)
        inventory.append({'id': d['id'], 'name': d['name'], 'state': d['state'],
            'settings': {k: d['config'].get(k, 'not recorded') for k in [
                'oracle_point_frame', 'no_pointmap', 'no_visual', 'visual_dropout', 'constant_touch']},
            'step': d['summary'].get('global_step'), 'val_loss': d['summary'].get('loss/val')})
    source_files = [_source_path(REPO, p) for p in ['train.py', 'dataloader.py', 'evaluation/evaluate.py',
        'sam3d_objects/model/backbone/dit/embedder/touch.py',
        'sam3d_objects/model/backbone/dit/embedder/vecsetx/autoencoder.py',
        'sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py',
        'sam3d_objects/model/backbone/tdfy_dit/modules/attention/modules.py',
        'data_generation/objaverse-dexonomy/generate_target_latents.py',
        'data_generation/objaverse-dexonomy/sample_full_surface.py']]
    out = {'status': 'existing_artifacts_rechecked_not_gpu_reproduced',
        'payload_hashes_checked': verified_payloads, 'occupancy_iou_and_counts_recomputed': len(rows),
        'prior_registration_rows_reaggregated_not_rerun': len(geometry), 'cells': measured,
        'normalization': {'observations': len(normalization),
            'numeric_max_error': max(r['numeric_max_error'] for r in normalization),
            'recovery_max_displacement_voxels': max(r['geometry_recovery_max_displacement_voxels'] for r in normalization)},
        'four_object_factorial_reserved_losses': factors, 'tiny_fit_cfg0_iou': tiny_means,
        'target_regeneration': {'previous_report_objects': 32, 'all_exact': True,
            'scope': 'Read existing source-reference report; did not run target encoder.'},
        'run_inventory': inventory,
        'current_source_sha256': {str(p.relative_to(REPO)): sha(p) for p in source_files},
        'raw_report_sha256': {arm: sha(root/'results.json') for arm, root in roots.items()},
        'script_sha256': sha(Path(__file__)),
        'limits': 'No model rerun, new registration, fresh objects, training seeds, cluster inspection, or independence claim.'}
    (_source_path(HERE, 'evidence_recheck.json')).write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('cells', 'run_inventory', 'current_source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
