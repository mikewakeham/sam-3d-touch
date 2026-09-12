"""Audit saved camera metadata, not learned pose accuracy or identifiability.

Uses the existing 64-sample selection. In particular, a near-constant camera
translation is NOT a substitute for object orientation in this orbit renderer.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    root = REPO / 'data_generation/objaverse-dexonomy'
    selection = REPO / 'outputs/diagnostics/sample_selection.json'
    manifest = root / 'generated_data/samples_full_surface.jsonl'
    ids = json.loads(selection.read_text())['sample_ids']
    records = {r['sample_id']: r for r in map(json.loads, manifest.read_text().splitlines())}
    rows = []
    rotations = []
    for sid in ids:
        record = records[sid]
        path = root / record['camera_path']
        with np.load(path, allow_pickle=False) as data:
            transform = np.diag([-1., -1., 1., 1.]) @ data['T_camera_from_object'].astype(np.float64)
        rotation, translation = transform[:3, :3], transform[:3, 3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-6)
        rotations.append(rotation)
        rows.append({'sample_id': sid, 'object_id': record['object_id'],
                     'translation_camera': translation.tolist(),
                     'camera_position_object': (-np.linalg.solve(rotation, translation)).tolist(),
                     'camera_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    translations = np.array([r['translation_camera'] for r in rows])
    rotation_angles = []
    for i, rotation in enumerate(rotations):
        for j in range(i):
            if rows[i]['object_id'] == rows[j]['object_id']:
                cosine = (np.trace(rotation @ rotations[j].T) - 1) / 2
                rotation_angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    summary = {
        'samples': len(rows), 'objects': len({r['object_id'] for r in rows}),
        'camera_translation_min': translations.min(0).tolist(),
        'camera_translation_max': translations.max(0).tolist(),
        'within_object_relative_rotation_degrees': {
            'min': min(rotation_angles), 'median': float(np.median(rotation_angles)),
            'max': max(rotation_angles), 'pairs': len(rotation_angles)},
    }
    sources = [Path(__file__), selection, manifest, REPO / 'train.py', REPO / 'dataloader.py',
               root / 'render_blender.py', root / 'make_data.py',
               REPO / 'sam3d_objects/model/backbone/tdfy_dit/modules/attention/modules.py',
               REPO / 'sam3d_objects/model/backbone/tdfy_dit/modules/transformer/modulated.py']
    report = {'summary': summary, 'rows': rows,
              'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sources},
              'limits': 'Camera metadata audit only. Camera translation is the location of the '
                        'object origin in camera axes; it is not camera position in object axes. '
                        'These statistics do not prove object orientation is unidentifiable from '
                        'RGB or geometry, nor measure the pretrained layout prediction.'}
    (HERE / 'pose_observability_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
