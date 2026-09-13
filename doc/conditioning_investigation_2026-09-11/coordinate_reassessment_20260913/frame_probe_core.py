"""Pairing and coordinate checks independent of SAM3D/CUDA imports."""
import numpy as np


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
