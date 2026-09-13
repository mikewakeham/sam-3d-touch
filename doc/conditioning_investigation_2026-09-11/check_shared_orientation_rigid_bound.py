"""Conservative all-rigid recall/precision bounds independent of ICP convergence.

If source diameter is D, any target centers within epsilon of a rigidly moved
source have pairwise separation <= D+2epsilon. Their projections along every
fixed target axis must therefore lie in an interval of that width. The largest
target fraction in such an interval upper-bounds recall for every rigid motion.
Swapping source/target supplies a precision upper bound. A bound of 1 is
inconclusive. No fitted scale or claim about the optimal F-score is involved.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from pose_shape_geometry import points
from shared_orientation_protocol import transform_points

HERE=Path(__file__).resolve().parent
OUT=HERE/'shared_orientation_geometry_analysis'


def diameter(p):
    # Maximum distance is attained between convex-hull vertices.
    return float(pdist(p[ConvexHull(p).vertices]).max())


def coverage_bound(source, target, epsilon=2/64):
    d=diameter(source)
    axes=np.concatenate((np.eye(3),np.linalg.eigh(np.cov(target.T))[1].T))
    caps=[]
    for axis in axes:
        x=np.sort(target@axis)
        counts=np.searchsorted(x,x+d+2*epsilon+1e-10,side='right')-np.arange(len(x))
        caps.append(float(counts.max()/len(x)))
    return dict(source_diameter=d,coverage_upper_bound=min(caps),axis_upper_bounds=caps)


def main():
    root=OUT/'bundle';d=json.loads((root/'shared_orientation/results.json').read_text())
    frames={r['sample_id']:r for r in d['target_frames']}
    originals=np.load(root/'reference/original_target_occupancy.npy',allow_pickle=False)
    rows=[];controls=[]
    for g in range(7):
        for draw in range(2):
            path=root/f'shared_orientation/correct_surface_g{g}_d{draw}'
            pp=np.load(path/'predicted_occupancy.npy',allow_pickle=False)
            qq=np.load(path/'target_occupancy.npy',allow_pickle=False)
            for i,sid in enumerate(d['input_batches'][g]['sample_ids']):
                inv=np.array(frames[sid]['object_from_output']);p=transform_points(points(pp[i]),inv);q=points(originals[i])
                r=dict(sample_id=sid,group=g,draw=draw,split='fit' if g<4 else 'reserved_view',
                       recall=coverage_bound(p,q),precision=coverage_bound(q,p))
                rows.append(r)
                if draw==0:
                    target=transform_points(points(qq[i]),inv)
                    a,b=coverage_bound(target,q),coverage_bound(q,target)
                    assert a['coverage_upper_bound']==b['coverage_upper_bound']==1
                    # Translate/rotate source; the coverage bound must not move.
                    angle=.731;c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                    rotated=coverage_bound(target@rot.T+[.11,-.23,.07],q)
                    np.testing.assert_allclose(rotated['source_diameter'],a['source_diameter'],atol=1e-12)
                    assert rotated['coverage_upper_bound']==a['coverage_upper_bound']
                    controls.append(dict(sample_id=sid,recall_upper=a['coverage_upper_bound'],precision_upper=b['coverage_upper_bound']))
    rejected=[r for r in rows if min(r['recall']['coverage_upper_bound'],r['precision']['coverage_upper_bound'])<.95]
    result=dict(rows=rows,positive_controls=controls,certified_below_95_percent=rejected,
        tolerance=2/64,scope='Finite returned voxel centers, every proper rigid transform (also valid for reflections); '
        'no fitted scale. 1e-10 numerical slack. A non-rejection proves nothing. All 28 faithful-target controls pass.')
    (OUT/'rigid_bounds.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Bound rejects',len(rejected),'of 56 correct-surface predictions')
    for r in rejected: print(r['sample_id'],r['draw'],r['split'],r['recall']['coverage_upper_bound'],r['precision']['coverage_upper_bound'])


if __name__=='__main__':main()
