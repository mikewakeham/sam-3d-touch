"""Reproduce reference metrics from a returned bundle without meshes or CUDA."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.coordinate_system.scripts.full_frame_comparison.frame_probe_core import triangle_witness


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    path = root/'results.partial.json'
    report = json.loads(path.read_text())
    for name, digest in report['files_sha256'].items():
        assert Path(name).name == name and sha(root/name) == digest
    objects = []
    for row in report['geometry_objects']:
        with np.load(root/f"object_{row['object_id']}.npz", allow_pickle=False) as f:
            delta = float(np.abs(f['regenerated_target_shape']-f['saved_target_shape']).max())
            assert delta == row['target_regeneration_max_error']
            objects.append({'object_id': row['object_id'], 'target_max_error': delta})
    samples = []
    for row in report['geometry_samples']:
        with np.load(root/f"geometry_{row['sample_id']}.npz", allow_pickle=False) as f:
            inv = np.linalg.inv(f['camera_from_object'])
            p = f['points_camera'].astype(float) @ inv[:3, :3].T + inv[:3, 3]
            assert np.array_equal(p, f['oracle_points_float64'])
            old = np.linalg.norm(p-f['closest_points_object'], axis=1)
            witness = triangle_witness(p, f['closest_triangles_object'])
            witness_distance = np.linalg.norm(p-witness, axis=1)
            with np.load(root/f"object_{row['object_id']}.npz", allow_pickle=False) as o:
                reference = o['mesh_sample_object'][f['point_ids']]
            replay = float(np.abs(p-reference).max())
            assert replay == row['seed_replay_max_error']
            assert float(old.max()) == row['point_to_mesh_max_distance']
            samples.append({'sample_id': row['sample_id'],
                            'seed_replay_max_error': replay,
                            'seed_replay_max_euclidean': float(np.linalg.norm(p-reference, axis=1).max()),
                            'nearest_query_max_distance': float(old.max()),
                            'selected_triangle_witness_max_distance': float(witness_distance.max()),
                            'nearest_query_passed': row['point_to_mesh_passed']})
    return {'partial_report_sha256': sha(path), 'artifact_hashes_verified': len(report['files_sha256']),
            'objects': objects, 'samples': samples,
            'summary': {'objects': len(objects), 'observations': len(samples),
                        'target_regeneration_max_error': max(x['target_max_error'] for x in objects),
                        'seed_replay_max_error': max(x['seed_replay_max_error'] for x in samples),
                        'seed_replay_max_euclidean': max(x['seed_replay_max_euclidean'] for x in samples),
                        'nearest_query_failed_observations': sum(not x['nearest_query_passed'] for x in samples),
                        'nearest_query_max_distance': max(x['nearest_query_max_distance'] for x in samples)},
            'scope': 'Saved seeded samples and inverse arithmetic verified. Original source meshes unavailable locally; fresh source-face witnesses still require cluster mesh access. No camera generator inference has completed.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.camera_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2))
