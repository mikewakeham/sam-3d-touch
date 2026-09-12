"""Separate orientation from bbox/radius normalization on existing full surfaces.

NumPy input-coordinate audit only; no claims about encoder or training effects.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def normalize(points, centroid=False):
    center = points.mean(0) if centroid else (points.min(0) + points.max(0)) / 2
    radius = np.linalg.norm(points - center, axis=1).max()
    assert np.isfinite(radius) and radius > 0
    return (points - center) / radius, center, radius


def factorize(camera_points, camera_from_object):
    rotation = camera_from_object[:3, :3]
    translation = camera_from_object[:3, 3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    # Use the exact numerical inverse for inversion, and the stored rotation
    # for forward transport. Rounded camera matrices need not be exactly SO(3).
    inverse = np.linalg.inv(rotation)
    obj = (camera_points - translation) @ inverse.T
    oracle, center_o, radius_o = normalize(obj)
    camera, center_c, radius_c = normalize(camera_points)
    rotation_only = oracle @ rotation.T
    normalization_only = camera @ inverse.T
    scale_ratio = radius_o / radius_c
    camera_offset = (center_o @ rotation.T + translation - center_c) / radius_c
    reconstructed = scale_ratio * rotation_only + camera_offset
    residual = float(np.max(np.abs(reconstructed - camera)))
    assert residual < 1e-10
    object_offset = camera_offset @ inverse.T
    np.testing.assert_allclose(normalization_only, scale_ratio * oracle + object_offset, atol=1e-10)
    centered_o, _, _ = normalize(obj, centroid=True)
    centered_c, _, _ = normalize(camera_points, centroid=True)
    centroid_error = float(np.max(np.abs(centered_c @ inverse.T - centered_o)))
    # With slightly rounded rotations this is approximate, not algebraically zero.
    assert centroid_error < 1e-6
    return {'rotation_only_rms_change': float(np.sqrt(np.mean((rotation_only - oracle) ** 2))),
            'normalization_only_rms_change': float(np.sqrt(np.mean((normalization_only - oracle) ** 2))),
            'normalization_only_max_change': float(np.max(np.abs(normalization_only - oracle))),
            'camera_to_oracle_radius_ratio': radius_c / radius_o,
            'normalization_offset_object_units': float(np.linalg.norm(object_offset * radius_o)),
            'factorization_max_error': residual, 'centroid_commutation_max_error': centroid_error}


def main():
    root = REPO / 'data_generation/objaverse-dexonomy'
    ids = json.loads((REPO / 'outputs/diagnostics/sample_selection.json').read_text())['sample_ids']
    manifest = root / 'generated_data/samples_full_surface.jsonl'
    records = {r['sample_id']: r for r in map(json.loads, manifest.read_text().splitlines())}
    rows = []
    for sid in ids:
        rec = records[sid]
        surface_path, camera_path = root / rec['full_surface_path'], root / rec['camera_path']
        with np.load(surface_path, allow_pickle=False) as data:
            points = data['points_camera'].astype(np.float64)
        with np.load(camera_path, allow_pickle=False) as data:
            transform = np.diag([-1., -1., 1., 1.]) @ data['T_camera_from_object'].astype(np.float64)
        row = {'sample_id': sid, 'object_id': rec['object_id'], **factorize(points, transform),
               'inputs_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [surface_path, camera_path]}}
        rows.append(row)
    summary = {'samples': len(rows), 'objects': len({r['object_id'] for r in rows})}
    for key in rows[0]:
        if isinstance(rows[0][key], float):
            values = [r[key] for r in rows]
            summary[key] = {'min': float(np.min(values)), 'median': float(np.median(values)), 'max': float(np.max(values))}
    report = {'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), REPO / 'train.py', REPO / 'dataloader.py',
                                          REPO / 'sam3d_objects/model/backbone/dit/embedder/touch.py']},
              'rows': rows, 'summary': summary,
              'limits': 'Input-space factorization, no neural tests. RMS magnitudes do not determine training impact. '
                        'Centroid normalization is an algebraic control, not a validated replacement for pretrained VecSetX preprocessing.'}
    (HERE / 'frame_factorization_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
