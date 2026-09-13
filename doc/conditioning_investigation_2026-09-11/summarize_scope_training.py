"""Aggregate paired native-loss trajectories and inspect fitted-target resemblance."""
import json
from pathlib import Path
from statistics import mean
import numpy as np
from pose_shape_geometry import points,metrics

HERE=Path(__file__).resolve().parent
OUT=HERE/'shared_orientation_scope_analysis'


def losses(d):
    out={}
    for a in d['assessments']:
        groups={r['group'] for r in a['rows']}
        if groups!=set(range(7)):continue  # Continuation replay is group0 only.
        out[str(a['step'])]={split+'/'+('wrong' if wrong else 'correct'):mean(
            v for r in a['rows'] if (r['group']<4)==(split=='fit') and r['wrong']==wrong for v in r['losses'])
            for split in ('fit','reserved_view') for wrong in (False,True)}
    return out


def main():
    base=HERE/'shared_orientation_geometry_analysis/bundle/shared_orientation'
    parent=json.loads((base/'results.json').read_text());frames={r['sample_id']:r for r in parent['target_frames']}
    bank=[]
    for g in range(4):
        aa=np.load(base/f'correct_surface_g{g}_d0/target_occupancy.npy',allow_pickle=False)
        for i,sid in enumerate(parent['input_batches'][g]['sample_ids']):bank.append((sid,aa[i]))
    trajectories={'parent':losses(parent)}
    for arm in ('current','shape_path'):
        root=OUT/'bundles'/arm/('scope_'+arm);d=json.loads((root/'results.json').read_text())
        trajectories[arm]=losses(d);rows=[]
        for g in range(4,7):
            for draw in range(2):
                pp=np.load(root/f'correct_surface_g{g}_d{draw}/predicted_occupancy.npy',allow_pickle=False)
                qq=np.load(root/f'correct_surface_g{g}_d{draw}/target_occupancy.npy',allow_pickle=False)
                for i,sid in enumerate(parent['input_batches'][g]['sample_ids']):
                    p,q=pp[i],qq[i]
                    scores=[dict(sample_id=name,iou=float((p&t).sum()/(p|t).sum())) for name,t in bank]
                    best=max(scores,key=lambda r:r['iou'])
                    ownbest=max([r for r in scores if r['sample_id'].rsplit('_',1)[0]==sid.rsplit('_',1)[0]],key=lambda r:r['iou'])
                    r=dict(sample_id=sid,draw=draw,own_target_iou=float((p&q).sum()/(p|q).sum()),
                        best_fitted_target=best,best_same_object_fitted_target=ownbest,all_fitted_target_ious=scores)
                    if ownbest['iou']>=.9:
                        other=next(t for name,t in bank if name==ownbest['sample_id'])
                        rotations=[np.array(frames[name]['output_from_object'])[:3,:3]*frames[name]['rotated_max_extent'] for name in (sid,ownbest['sample_id'])]
                        r['target_relative_rotation_degrees']=float(np.degrees(np.arccos(np.clip((np.trace(rotations[0]@rotations[1].T)-1)/2,-1,1))))
                        r['true_vs_fitted_target_fscore_2v_native_cube']=metrics(points(q),points(other))['fscore_2v']
                        r['prediction_vs_true_fscore_2v_native_cube']=metrics(points(p),points(q))['fscore_2v']
                        r['caution']='Direct overlap in normalized camera cubes; approximate shape symmetry can make large rotations geometrically similar. Resemblance is not mechanism proof.'
                    rows.append(r)
        (OUT/f'fitted_target_comparison_{arm}.json').write_text(json.dumps(dict(rows=rows,
            scope='Exploratory direct overlap with all16 fitted targets. No registration; no identity-retrieval or pose-mechanism claim.'),indent=2)+'\n')
    (OUT/'native_trajectories.json').write_text(json.dumps(trajectories,indent=2)+'\n')


if __name__=='__main__':main()
