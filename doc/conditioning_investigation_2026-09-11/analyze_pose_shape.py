"""Analyze existing Stage-1 supports; GT registration is diagnostic only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from alignment_tolerance_protocol import rotation
from pose_shape_geometry import points,metrics,register

HERE=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def array_pair(root,case,index):
    p=np.load(root/case['prefix']/'predicted_occupancy.npy',allow_pickle=False)
    q=np.load(root/case['prefix']/'target_occupancy.npy',allow_pickle=False)
    expected=(4,64,64,64) if case['kind']=='alignment' else (64,64,64)
    assert p.shape==q.shape==expected and p.dtype==q.dtype==np.bool_
    return (p[index],q[index]) if p.ndim==4 else (p,q)


def tasks_and_targets(root):
    manifest=json.loads((root/'bundle_manifest.json').read_text())
    for entry in manifest['files']:
        path=root/entry['archive_path']
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256']
    reports={i:json.loads((root/f'alignment/{i}/results.json').read_text()) for i in range(4)}
    maps={i:{(r['policy'],r['condition'],r['group'],r['draw'],r['sample_id']):r for r in rep['rows']} for i,rep in reports.items()}
    camera=json.loads((root/'camera/results.json').read_text()) if (root/'camera/results.json').exists() else None
    cm={(r['group'],r['noise_draw'],r['sample_id']):r for r in camera['sampled']} if camera else {}
    targets,tasks,seen={},[],set()
    for case in manifest['cases']:
        for i,sid in enumerate(case['sample_ids']):
            p,q=array_pair(root,case,i);obj=sid.rsplit('_',1)[0]
            assert q.any()
            if obj in targets:assert np.array_equal(q,targets[obj])
            else:targets[obj]=q
            if case['kind']=='alignment':
                row=maps[case['shard']][case['policy'],case['condition'],case['group'],case['draw'],sid]
                model=case['policy'];condition=case['condition'];axis=case['axis'];degrees=case['degrees']
            else:
                row=cm[case['group'],case['draw'],sid];model='camera';condition='natural';axis=None;degrees=0
            key=(model,condition,case['group'],case['draw'],sid);assert key not in seen;seen.add(key)
            iou=float(np.count_nonzero(p&q)/np.count_nonzero(p|q))
            assert iou==row['voxel_iou'] and int(p.sum())==row['predicted_occupied'] and int(q.sum())==row['target_occupied']
            tasks.append(dict(root=str(root),case=case,index=i,sample_id=sid,object_id=obj,
                model=model,condition=condition,axis=axis,degrees=degrees,
                split='fit' if case['group']<4 else 'reserved_view',group=case['group'],draw=case['draw'],
                raw_iou=iou,expected_cd=row['unaligned_voxel_center_chamfer']))
    assert len(tasks)==1232 if camera else len(tasks)==1176
    return tasks,targets


def run_control(job):
    kind,source,target,label=job
    result=register(source,target)
    return {'kind':kind,'label':label,'raw':metrics(source,target),'rigid':result}


def controls(targets,workers):
    jobs=[]
    matrices=[('x5',np.array(rotation('x',5))),('z30',np.array(rotation('z',30))),
              ('mixed',np.array(rotation('z',23))@np.array(rotation('y',-17))@np.array(rotation('x',11)))]
    for obj,o in targets.items():
        q=points(o)
        for name,r in matrices:
            p=q@r.T+np.array([.03,-.02,.01])
            jobs.append(('known_rigid_continuous',p,q,obj+'/'+name))
        # Revoxelization control has a changed lattice. Registration cannot
        # promise exact IoU after rotation even for a known correct shape.
        r=matrices[-1][1];p=q@r.T
        ijk=np.floor((p+.5)*64).astype(int)
        # Quantize on the same-spacing infinite lattice: some rotations extend
        # outside the original cube. Clipping would inject artificial damage.
        p=(np.unique(ijk,axis=0)+.5)/64-.5
        jobs.append(('known_rigid_revoxelized',p,q,obj+'/mixed_revoxelized'))
    for a,oa in targets.items():
        for b,ob in targets.items():
            if a!=b:jobs.append(('wrong_object',points(oa),points(ob),a+'/'+b))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        result=list(pool.map(run_control,jobs))
    return result


def run_prediction(task):
    t=time.monotonic();p,q=array_pair(Path(task['root']),task['case'],task['index']);p,q=points(p),points(q)
    raw=metrics(p,q)
    expected=task['expected_cd']
    assert (raw['mean_distance'] is None and expected is None) or abs(raw['mean_distance']-expected)<1e-12
    known=np.array(rotation(task['axis'],task['degrees'])).T if task['degrees'] else None
    inverse=metrics(p@known.T,q) if known is not None else raw
    rigid=register(p,q,known)
    return {k:v for k,v in task.items() if k not in ('root','case','index','expected_cd')}|{
        'raw':raw,'known_inverse':inverse,'rigid':rigid,'seconds':time.monotonic()-t,
        'predicted_count':len(p),'target_count':len(q)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--phase',choices=['controls','predictions'],required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    config={'source_sha256':{n:digest(HERE/n) for n in ['pose_shape_geometry.py','analyze_pose_shape.py']},
            'bundle_manifest_sha256':digest(args.bundle_dir/'bundle_manifest.json'),
            'registration':'SE(3) only; 24 proper axes +24 PCA axes +identity/known inverse; '
            '1024 coarse/4096 fine samples, 35 iterations each, 6 prescreened starts plus analytic starts, '
            '3 fine candidates, full-cloud symmetric squared-distance selection; no trimming or scaling.',
            'thresholds':[1/64,2/64],'scope':'GT-assisted pose/shape diagnosis on fitted identities, not deployment or new-object validation.'}
    tasks,targets=tasks_and_targets(args.bundle_dir)
    print('Validated occupancy counts/IoU and shared targets:',len(tasks),'predictions',flush=True)
    if args.phase=='controls':
        path=args.output_dir/'controls.json'
        if path.exists():raise FileExistsError(path)
        result=controls(targets,args.workers)
        path.write_text(json.dumps({'config':config,'controls':result},indent=2)+'\n')
        for r in result:print(r['kind'],r['label'],r['rigid']['metrics'],flush=True)
    else:
        saved=json.loads((args.output_dir/'controls.json').read_text());assert saved['config']==config
        # Control failures must be reviewed; never silently proceed after them.
        positives=[r for r in saved['controls'] if r['kind'].startswith('known_rigid')]
        assert all(min(r['rigid']['metrics']['precision_1v'],r['rigid']['metrics']['recall_1v'])>=.95 for r in positives)
        wrong=[r for r in saved['controls'] if r['kind']=='wrong_object']
        assert all(min(r['rigid']['metrics']['precision_1v'],r['rigid']['metrics']['recall_1v'])<.95 for r in wrong)
        path=args.output_dir/'predictions.jsonl'
        with path.open('x') as f,ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,r in enumerate(pool.map(run_prediction,tasks),1):
                f.write(json.dumps(r)+'\n');f.flush()
                if i%20==0:print('Processed',i,'/',len(tasks),flush=True)
        (args.output_dir/'complete.json').write_text(json.dumps({'config':config,'rows':len(tasks),'predictions_sha256':digest(path)},indent=2)+'\n')


if __name__=='__main__':main()
