"""Measure normalization recoverability on available saved full surfaces (CPU)."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / 'data_generation/objaverse-dexonomy'
INV = HERE.parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest = DATA / 'generated_data/samples_full_surface.jsonl'
    records = [json.loads(x) for x in manifest.read_text().splitlines()]
    available = [r for r in records if (DATA / r['full_surface_path']).exists()]
    # Execute the actual normalization method without importing CUDA-only modules.
    src = REPO / 'sam3d_objects/model/backbone/dit/embedder/touch.py'
    cls = next(n for n in ast.parse(src.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'TouchEncoder')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'normalize_points_for_vecsetx')
    scope = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(src), 'exec'), scope)
    normalize = scope[method.name]
    rollout_root = INV / 'oracle_upper_bound/rollouts_returned_20260913_143515/oracle'
    report = json.loads((rollout_root / 'results.json').read_text())
    latest = {}
    for batch in report['inputs']:
        with np.load(rollout_root / f"{batch['split']}_g{batch['group']}_targets.npz", allow_pickle=False) as z:
            for i, oid in enumerate(batch['object_ids']):
                latest[oid] = {'support': z['target_occupancy'][i].copy(),
                               'latent': z['target_shape'][i].copy(),
                               'file': f"{batch['split']}_g{batch['group']}_targets.npz", 'index': i}
    rows = []; clouds = {}
    for rec in available:
        pp, cp = DATA / rec['full_surface_path'], DATA / rec['camera_path']
        with np.load(pp, allow_pickle=False) as z:
            camera = z['points_camera'].copy(); ids = z['point_ids'].copy()
        with np.load(cp, allow_pickle=False) as z:
            matrix = np.diag([-1., -1., 1., 1.]) @ z['T_camera_from_object']
        inverse = np.linalg.inv(matrix).astype(np.float32)
        p = (torch.from_numpy(camera) @ torch.from_numpy(inverse[:3, :3]).T
             + torch.from_numpy(inverse[:3, 3])).numpy()
        q, center, scale = normalize(None, torch.from_numpy(p)[None], torch.ones((1, len(p)), dtype=torch.bool))
        q, center, scale = q[0].numpy().astype(float), center[0].numpy().astype(float), float(scale[0, 0])
        pd = p.astype(float)
        c = (pd.min(0) + pd.max(0)) / 2
        radius = np.linalg.norm(pd - c, axis=1).max()
        expected = (pd - c) / radius
        recovered = q / scale + center
        canonical = (q - (q.min(0)+q.max(0))/2) / np.ptp(q, axis=0).max()
        displacement = np.linalg.norm(canonical-pd, axis=1)
        row = {'sample_id': rec['sample_id'], 'object_id': rec['object_id'], 'points': len(p),
               'source_hashes': {str(x.relative_to(DATA)): sha(x) for x in [pp, cp, DATA/rec['target_path']]},
               'center': c.tolist(), 'radius': float(radius), 'surface_max_extent': float(np.ptp(pd, axis=0).max()),
               'normalization_float32_vs_independent_float64_max': float(np.abs(q-expected).max()),
               'inverse_normalization_max_error': float(np.abs(recovered-pd).max()),
               'canonicalization_rms_displacement_voxels': float(np.sqrt(np.mean(displacement**2))*64),
               'canonicalization_max_displacement_voxels': float(displacement.max()*64),
               'strict_target_q_over_2_scale_relative_to_current': float(1/(2*radius))}
        if rec['object_id'] in latest:
            target = latest[rec['object_id']]
            centers = (np.argwhere(target['support'])+.5)/64-.5
            tree = cKDTree(centers)
            row['latest_target_support_reference'] = {k:v for k,v in target.items() if k not in ('support', 'latent')}
            with np.load(DATA / rec['target_path'], allow_pickle=False) as z:
                local_target = z['mean'].transpose(1, 2, 3, 0).reshape(4096, 8)
            row['latest_and_local_target_latents_equal'] = bool(np.array_equal(local_target, target['latent']))
            assert row['latest_and_local_target_latents_equal']
            for label, pts in [('original_oracle', pd), ('canonicalized', canonical)]:
                distance = tree.query(pts)[0]
                row[label+'_point_precision_1voxel'] = float(np.mean(distance <= 1/64))
                row[label+'_point_precision_2voxels'] = float(np.mean(distance <= 2/64))
        rows.append(row)
        clouds.setdefault(rec['object_id'], []).append((rec['sample_id'], ids, pd, q))
    pairs = []
    for oid, samples in clouds.items():
        first = samples[0]
        for other in samples[1:]:
            ai, bi = np.argsort(first[1]), np.argsort(other[1])
            equal = np.array_equal(first[1][ai], other[1][bi])
            pairs.append({'object_id': oid, 'samples': [first[0],other[0]], 'point_ids_equal': equal,
                          'oracle_xyz_max_error': float(np.abs(first[2][ai]-other[2][bi]).max()) if equal else None,
                          'normalized_xyz_max_error': float(np.abs(first[3][ai]-other[3][bi]).max()) if equal else None})
    numeric = [k for k,v in rows[0].items() if isinstance(v,float)]
    summary = {k: {'min': min(r[k] for r in rows), 'median': float(np.median([r[k] for r in rows])),
                   'max': max(r[k] for r in rows)} for k in numeric}
    summary.update(samples=len(rows), objects=len(clouds), latest_probe_objects_overlap=len(set(clouds)&set(latest)),
                   rows_with_latest_support=sum('latest_target_support_reference' in r for r in rows))
    output = {'summary': summary, 'rows': rows, 'same_object_view_pairs': pairs,
              'source_sha256': {str(p.relative_to(REPO)):sha(p) for p in [Path(__file__), src, manifest, REPO/'train.py', REPO/'dataloader.py']},
              'scope': '64 locally available historical surfaces, before FPS (8192 points). CPU float32 production-normalizer replay and independent float64 arithmetic. '
                       'Not a CUDA/FPS replay or an independent mesh-provenance check. Meshes unavailable locally; latest probe has different views and only three overlapping identities. '
                       'Canonicalization uses the known centered max-extent-one dataset convention, not target geometry. Latest target proximity is descriptive, with target-latent equality checked separately.'}
    (HERE/'normalization_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
