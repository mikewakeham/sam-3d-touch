"""Pairing and coordinate checks independent of SAM3D/CUDA imports."""
import numpy as np


def triangle_witness(points, triangles):
    """Float64 witnesses on supplied triangles, not a global nearest-face query.

    Every candidate is a convex combination of triangle vertices. Clamping
    barycentrics can overestimate distance but cannot create an off-mesh witness.
    Edge candidates also handle collapsed and very thin triangles.
    """
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.float64)
    assert p.ndim == 2 and p.shape[1] == 3 and t.shape == (len(p), 3, 3)
    assert np.isfinite(p).all() and np.isfinite(t).all()
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    u, v = b-a, c-a
    n = np.cross(u, v); nn = np.sum(n*n, axis=1)
    s = np.divide(np.sum(np.cross(p-a, v)*n, axis=1), nn,
                  out=np.zeros(len(p)), where=nn > 0)
    w = np.divide(np.sum(np.cross(u, p-a)*n, axis=1), nn,
                  out=np.zeros(len(p)), where=nn > 0)
    weights = np.maximum(np.stack([1-s-w, s, w], axis=1), 0)
    weights /= weights.sum(axis=1, keepdims=True)
    candidates = [np.sum(weights[:, :, None]*t, axis=1)]
    for x, y in ((a, b), (b, c), (c, a)):
        e = y-x; ee = np.sum(e*e, axis=1)
        alpha = np.divide(np.sum((p-x)*e, axis=1), ee,
                          out=np.zeros(len(p)), where=ee > 0).clip(0, 1)
        candidates.append(x+alpha[:, None]*e)
    candidates = np.stack(candidates, axis=1)
    distances = np.linalg.norm(p[:, None]-candidates, axis=2)
    best = distances.argmin(axis=1)
    return candidates[np.arange(len(p)), best]


def sampled_mesh_check(points, sampled, triangles, limit=5e-5):
    """Require both exact seeded correspondence and a source-triangle witness."""
    p = np.asarray(points, dtype=np.float64)
    sampled = np.asarray(sampled, dtype=np.float64)
    assert p.shape == sampled.shape and np.isfinite(sampled).all()
    witness = triangle_witness(p, triangles)
    distance = np.linalg.norm(p-witness, axis=1)
    replay = float(np.abs(p-sampled).max())
    return {'point_to_mesh_max_distance': float(distance.max()),
            'point_to_mesh_passed': bool(distance.max() < limit and replay < limit),
            'distance_limit': limit, 'seed_replay_max_error': replay,
            'mesh_reference_method': 'seeded_source_triangle_float64'}, witness


def check_observation(actual, reference):
    # Point frames and learned surface tokens are the intended differences.
    keys = ('split', 'group', 'sample_ids', 'object_ids', 'image_sha256',
            'pointmap_sha256', 'visual_sha256', 'mask_sha256', 'target_sha256')
    for key in keys:
        assert actual[key] == reference[key], f'Unpaired observation: {key}'


def normalization_check(points, prepared, center, scale):
    """No-FPS case used by this fixed8192-point bank; reject changed counts."""
    p = np.asarray(points, dtype=np.float64)
    q = np.asarray(prepared, dtype=np.float64)
    assert p.shape == q.shape and p.ndim == 2 and p.shape[1] == 3
    c = (p.min(0) + p.max(0)) / 2
    radius = np.linalg.norm(p-c, axis=1).max()
    expected = (p-c)/radius
    error = float(np.abs(q-expected).max())
    inverse_error = float(np.abs(q/float(scale)+np.asarray(center)-p).max())
    assert error < 5e-6 and inverse_error < 5e-6, 'Pre-encoder normalization mismatch'
    canonical = (q-(q.min(0)+q.max(0))/2) / np.ptp(q,axis=0).max()
    displacement = np.linalg.norm(canonical-p,axis=1)
    return {'normalization_max_error': error, 'inverse_max_error': inverse_error,
            'center': c.tolist(), 'radius': float(radius),
            'canonicalization_max_displacement_voxels': float(displacement.max()*64)}


def check_report_pair(camera, oracle):
    assert camera['complete'] and camera['adapted_parameters_unchanged']
    assert camera['settings']['arm'] == 'camera'
    for key in ('dataset_sha256', 'pipeline_sha256', 'decoder_sha256', 'sampler', 'banks'):
        assert camera[key] == oracle[key], f'Unpaired report: {key}'
    assert camera['source_sha256'] == oracle['source_sha256']
    assert len(camera['inputs']) == len(oracle['inputs'])
    for actual, reference in zip(camera['inputs'], oracle['inputs']):
        check_observation(actual, reference)
        assert actual['oracle_points_sha256'] == reference['points_sha256']
        assert actual['target_support_sha256'] == reference['target_support_sha256']
    expected = {(b['split'], b['group'], sid, visual, shift, draw)
                for b in oracle['inputs'] for sid in b['sample_ids']
                for visual in ('present','zero') for shift in (0,1) for draw in range(2)}
    actual = [tuple(s[k] for k in ('split','group','sample_id','visual','surface_shift','draw'))
              for s in camera['samples']]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    identities = {(b['split'],sid):oid for b in oracle['inputs']
                  for sid,oid in zip(b['sample_ids'],b['object_ids'])}
    for s in camera['samples']:
        assert identities[s['split'],s['sample_id']] == s['object_id']
        assert all(np.isfinite(s[k]) and 0 <= s[k] <= 1 for k in ('iou','precision_2v','recall_2v','fscore_2v'))
    assert camera['checkpoint_metadata']['epoch'] == 20
    assert camera['checkpoint_metadata']['step'] == 14660
    assert camera['checkpoint_metadata']['conditioning_config'] == {'no_pointmap':False,'oracle_point_frame':False}
    assert camera['checkpoint_metadata']['training_config'] == {'visual_dropout':.5,'constant_touch':False}
    assert camera['coordinate_contract_passed']


def check_loss_pair(camera, reference):
    assert camera['loss_banks']==reference['banks'], 'Unpaired denoising times/noise'
    expected={tuple(r[k] for k in ('split','group','visual','surface_shift','time_kind','draw'))
              for r in reference['rows']}
    actual=[tuple(r[k] for k in ('split','group','visual','surface_shift','time_kind','draw'))
            for r in camera['rows']]
    assert len(actual)==len(set(actual)) and set(actual)==expected, 'Missing/duplicate loss rows'
    assert all(len(r['losses'])==4 and all(np.isfinite(v) and v>=0 for v in r['losses']) for r in camera['rows'])
