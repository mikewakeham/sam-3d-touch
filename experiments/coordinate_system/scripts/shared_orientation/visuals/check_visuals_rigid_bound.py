"""Apply the established all-rigid coverage bound, including collapsed supports."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.distance import pdist
from experiments.coordinate_system.scripts.shared.pose_shape_geometry import points
from experiments.coordinate_system.scripts.shared_orientation.shared_orientation_protocol import transform_points

HERE=Path(__file__).resolve().parent


def diameter_bound(p):
    assert len(p)>0
    if len(p)==1:return 0.,'exact_singleton'
    if len(p)<=2048:return float(pdist(p).max()),'exact_all_pairs'
    try:return float(pdist(p[ConvexHull(p).vertices]).max()),'exact_hull_pairs'
    except QhullError:
        # A larger D only loosens the coverage upper bound, never creates a
        # false impossibility certificate. Do not perturb/joggle the supports.
        return float(np.linalg.norm(np.ptp(p,axis=0))),'bounding_box_upper_bound'


def bound(source,target,epsilon=2/64):
    d,method=diameter_bound(source)
    axes=np.eye(3)
    if len(target)>1:axes=np.concatenate((axes,np.linalg.eigh(np.cov(target.T))[1].T))
    caps=[]
    for axis in axes:
        x=np.sort(target@axis)
        counts=np.searchsorted(x,x+d+2*epsilon+1e-10,side='right')-np.arange(len(x))
        caps.append(float(counts.max()/len(x)))
    return dict(source_diameter_upper_bound=d,diameter_method=method,coverage_upper_bound=min(caps))


def main():
    out=_source_path(HERE, 'shared_orientation_visuals_analysis');root=out/'bundle'
    d=json.loads((root/'visual_probe/results.json').read_text())
    frames={f['sample_id']:f for f in d['target_frames']}
    original=np.load(root/'reference/original_target_occupancy.npy',allow_pickle=False)
    # Exact small/degenerate diameter controls and rigid-invariance check.
    plane=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
    assert diameter_bound(plane)[0]==np.sqrt(2) and diameter_bound(plane[:1])[0]==0
    angle=.731;c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    np.testing.assert_allclose(diameter_bound(plane@rot.T+[.11,-.23,.07])[0],np.sqrt(2),atol=1e-12)
    rows=[];controls=[]
    for g in range(7):
        for visual in ('present','zero'):
            for draw in range(2):
                pref=root/f'visual_probe/{visual}_correct_surface_g{g}_d{draw}'
                p,q=[np.load(pref/f'{k}_occupancy.npy',allow_pickle=False) for k in ('predicted','target')]
                for i,sid in enumerate(d['input_batches'][g]['sample_ids']):
                    inv=np.array(frames[sid]['object_from_output']);pp=transform_points(points(p[i]),inv);qq=points(original[i])
                    rows.append(dict(visual_mode=visual,group=g,draw=draw,sample_id=sid,split='fit' if g<4 else 'reserved_view',
                        occupied=int(p[i].sum()),recall=bound(pp,qq),precision=bound(qq,pp)))
                    if visual=='present' and draw==0:
                        target=transform_points(points(q[i]),inv);a,b=bound(target,qq),bound(qq,target)
                        assert a['coverage_upper_bound']==b['coverage_upper_bound']==1
                        moved=bound(target@rot.T+[.11,-.23,.07],qq)
                        np.testing.assert_allclose(moved['source_diameter_upper_bound'],a['source_diameter_upper_bound'],atol=1e-12)
                        assert moved['coverage_upper_bound']==a['coverage_upper_bound']
                        controls.append(dict(sample_id=sid,recall=a,precision=b))
    reject=[r for r in rows if min(r['recall']['coverage_upper_bound'],r['precision']['coverage_upper_bound'])<.95]
    result=dict(rows=rows,certified_below_95_percent=reject,positive_controls=controls,tolerance=2/64,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Same conservative diameter/projection-interval argument over every rigid transform. '
        'Exact small-support diameter handles collapsed planar predictions. No fitted scale. Non-rejection inconclusive.')
    (out/'rigid_bounds.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Certified failures',len(reject))
    for r in reject:print(r['visual_mode'],r['split'],r['sample_id'],r['draw'],r['occupied'],r['recall']['coverage_upper_bound'],r['precision']['coverage_upper_bound'])


if __name__=='__main__':main()
