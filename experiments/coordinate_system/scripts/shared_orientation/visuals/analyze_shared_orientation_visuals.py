"""Verify the visual factorial and retain the established Stage-1 geometry endpoints."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from experiments.coordinate_system.scripts.shared_orientation.analyze_shared_orientation_geometry import analyze, aggregate, same
from experiments.coordinate_system.scripts.shared.pose_shape_geometry import points, metrics
from experiments.coordinate_system.scripts.shared_orientation.shared_orientation_protocol import transform_points

HERE=Path(__file__).resolve().parent
REPO=next(p for p in Path(__file__).resolve().parents if (_source_path(p, 'train.py')).is_file() and (p / 'sam3d_objects').is_dir())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    root=out/'bundle';oldroot=_source_path(HERE, 'shared_orientation_scope_analysis')
    oldbundle=oldroot/'bundles/shape_path'
    oldfit=oldbundle/'scope_shape_path/results.json'
    completed=json.loads((oldroot/'complete.json').read_text())
    assert sha(oldroot/'predictions.jsonl')==completed['predictions_sha256']
    controls=json.loads((_source_path(HERE, 'pose_shape_analysis/controls.json')).read_text())
    assert sha(_source_path(HERE, 'pose_shape_geometry.py'))==controls['config']['source_sha256']['pose_shape_geometry.py']
    assert all(min(c['rigid']['metrics']['precision_1v'],c['rigid']['metrics']['recall_1v'])>=.95
               for c in controls['controls'] if c['kind'].startswith('known_rigid'))
    with zipfile.ZipFile(args.bundle) as z:
        assert z.testzip() is None
        m=json.loads(z.read('bundle_manifest.json'))
        assert m['format_version']==5 and len(m['cases'])==56
        names=[r['archive_path'] for r in m['files']]
        assert len(names)==len(set(names))==144
        assert len(z.namelist())==145 and set(z.namelist())==set(names)|{'bundle_manifest.json'}
        for e in m['files']:
            b=z.read(e['archive_path']);assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
        for name in z.namelist():
            dest=(root/name).resolve();assert dest.is_relative_to(root.resolve())
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
    d=json.loads((root/'visual_probe/results.json').read_text())
    fit=json.loads((root/'reference/fit_results.json').read_text())
    assert fit==json.loads(oldfit.read_text()) and sha(root/'reference/fit_results.json')==d['fit_report_sha256']
    assert d['complete'] and d['parameters_unchanged'] and d['settings']['training_updates']==0
    assert d['settings']['model_key']=='shape_path_2000' and d['settings']['sampling_steps']==25 and d['settings']['cfg']==0
    assert d['settings']['visual_modes']==['present','zero'] and d['settings']['conditions']==['correct_surface','wrong_surface']
    assert d['settings']['seed']==29 and d['settings']['sampling_bank']==200000 and d['settings']['sampling_draws']==2
    assert len(d['interface_checks'])==8 and all(d['interface_checks'].values())
    assert d['checkpoint_parameters_sha256']==fit['final_all_parameters_sha256']
    for name,value in d['source_sha256'].items():assert sha(_source_path(REPO, name))==value,name
    for k in ('input_batches','target_frames','target_quality'):assert d[k]==fit[k],k
    for path in ['reference/parent_results.json','reference/original_target_occupancy.npy']+[f'physical/{f["sample_id"]}.npy' for f in d['target_frames']]:
        assert sha(root/path)==sha(oldbundle/path),path
    native={(r['visual_mode'],r['group'],r['wrong']):r['losses'] for r in d['native']}
    assert len(native)==len(d['native'])==28
    assert set(native)=={(v,g,w) for v in ('present','zero') for g in range(7) for w in (False,True)}
    assert all(len(v)==8 and np.isfinite(v).all() for v in native.values())
    a=np.array([native['present',r['group'],r['wrong']] for r in fit['assessments'][-1]['rows']])
    b=np.array([r['losses'] for r in fit['assessments'][-1]['rows']])
    np.testing.assert_allclose(a,b,rtol=.001,atol=.000001)
    assert float(np.max(np.abs(a-b)))==d['native_replay']['max_absolute_error']
    key=lambda r:(r['condition'],r['group'],r['draw'],r['sample_id'])
    prior={key(r):r for r in map(json.loads,(oldroot/'predictions.jsonl').read_text().splitlines()) if r['model']=='scope_shape_path'}
    fitrows={key(r):r for r in fit['rows']}
    rows={(r['visual_mode'],)+key(r):r for r in d['rows']}
    assert len(rows)==len(d['rows'])==224 and len(d['artifacts'])==56
    frames={f['sample_id']:f for f in d['target_frames']}
    replay={(r['group'],r['draw'],r['condition']):r for r in d['historical_sample_replay']}
    assert len(replay)==28
    original_path=root/'reference/original_target_occupancy.npy'
    original=np.load(original_path,allow_pickle=False)
    tasks=[];result=[];seen=set();equal=0
    for case in m['cases']:
        assert case['kind']=='visual_probe'
        assert {k:v for k,v in case.items() if k not in ('kind','prefix')} in d['artifacts']
        v,g,c,draw=(case[k] for k in ('visual_mode','group','condition','draw'))
        pp,qp=[root/case['prefix']/f'{k}_occupancy.npy' for k in ('predicted','target')]
        p,q=[np.load(path,allow_pickle=False) for path in (pp,qp)]
        assert p.shape==q.shape==(4,64,64,64) and p.dtype==q.dtype==np.bool_
        oldp,oldq=[np.load(oldbundle/f'scope_shape_path/{c}_g{g}_d{draw}/{k}_occupancy.npy',allow_pickle=False) for k in ('predicted','target')]
        np.testing.assert_array_equal(q,oldq)
        assert case['sample_ids']==d['input_batches'][g]['sample_ids']
        if v=='present':
            rr=replay[g,draw,c]
            assert rr['bitwise_support_equal']==np.array_equal(p,oldp)
            np.testing.assert_array_equal(rr['per_object_iou_to_previous'],[np.count_nonzero(x&y)/max(1,np.count_nonzero(x|y)) for x,y in zip(p,oldp)])
        for i,sid in enumerate(case['sample_ids']):
            k=(c,g,draw,sid);full=(v,)+k;assert full not in seen;seen.add(full)
            r=rows[full];old=fitrows[k]
            for field in ('noise_sha256','split','object_id'):assert r[field]==old[field]
            assert r['voxel_iou']==np.count_nonzero(p[i]&q[i])/np.count_nonzero(p[i]|q[i])
            assert r['predicted_occupied']==int(p[i].sum()) and r['target_occupied']==int(q[i].sum())
            inv=frames[sid]['object_from_output']
            task=dict(p=str(pp),q=str(qp),original=str(original_path),index=i,row=r,inverse=inv,visual_mode=v)
            if v=='present' and np.array_equal(p[i],oldp[i]):
                # Raw/native scores reproduce; registration reuses the exact same
                # support arrays, frame and verified analysis implementation.
                raw=metrics(transform_points(points(p[i]),np.array(inv)),points(original[i]))
                same(raw,r['common_object_units']);same(raw,prior[k]['raw'])
                nm=metrics(points(p[i]),points(q[i]));same(nm,prior[k]['native'])
                np.testing.assert_allclose(nm['mean_distance'],r['unaligned_voxel_center_chamfer'],rtol=1e-12,atol=1e-12)
                result.append({**prior[k],'model':'visual_present'});equal+=1
            else:tasks.append(task)
    assert seen==set(rows)
    validation=dict(bundle_sha256=sha(args.bundle),payload_members=144,rows=224,
        native_replay_exact=bool(np.array_equal(a,b)),historical_identical_predictions=equal,
        all_target_reference_input_noise_matches=True,all_native_ious_reproduced=True,
        source_sha256={n:sha(_source_path(HERE, n)) for n in ('analyze_shared_orientation_visuals.py','analyze_shared_orientation_geometry.py','pose_shape_geometry.py')},
        scope='Parameter/checkpoint hashes are source-audited GPU reports, not independent local checkpoint loading. '
              'All supports verified locally; prior registration reused only for identical supports/frames/source.')
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('Verified224 rows; reused',equal,'exact historical registrations. Remaining',len(tasks),flush=True)
    with (out/'predictions.jsonl').open('x') as f,ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in result:f.write(json.dumps(r)+'\n')
        f.flush()
        for i,(task,r) in enumerate(zip(tasks,pool.map(analyze,tasks),strict=True),1):
            r['model']='visual_'+task['visual_mode'];result.append(r);f.write(json.dumps(r)+'\n');f.flush()
            if i%8==0:print('Registered',i,'/',len(tasks),flush=True)
    cells={}
    for split in ('fit','reserved_view'):
        cc={}
        for v in ('present','zero'):
            for c in ('correct_surface','wrong_surface'):
                selected=[r for r in result if r['split']==split and r['model']=='visual_'+v and r['condition']==c]
                cell=aggregate(selected)
                cell['objects']={oid:aggregate([r for r in selected if r['object_id']==oid]) for oid in sorted({r['object_id'] for r in selected})}
                groups=range(4) if split=='fit' else range(4,7)
                cell['native_loss']=float(np.mean([native[v,g,c=='wrong_surface'] for g in groups]))
                cc[v+'/'+c]=cell
            correct,wrong=[cc[v+'/'+c] for c in ('correct_surface','wrong_surface')]
            for oid,item in correct['objects'].items():
                item['common_reference_pass']=min(item['witness']['precision_2v'],item['witness']['recall_2v'])>=.98
                item['surface_advantage_2v']=item['witness']['fscore_2v']-wrong['objects'][oid]['witness']['fscore_2v']
            correct['native_endpoint']=correct['iou']>=.95 and all(x['iou']>=.9 for x in correct['objects'].values())
            correct['common_geometry_endpoint']=all(x['common_reference_pass'] for x in correct['objects'].values())
            correct['dependence_endpoint']=all(x['surface_advantage_2v']>=.1 for x in correct['objects'].values())
        cells[split]=cc
    summary=dict(cells=cells,scope='Four fitted identities only. Same checkpoint/noise. Common tolerance two original voxels. '
        'Rigid witness is a measured candidate, not a certified optimum. No new-object or architecture-impossibility claim.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps(dict(validation=validation,rows=len(result),
        predictions_sha256=sha(out/'predictions.jsonl'),summary_sha256=sha(out/'summary.json')),indent=2)+'\n')
    for split,cc in cells.items():
        for name,c in cc.items():print(split,name,'IoU',round(c['iou'],5),'raw/witnessF2',round(c['raw']['fscore_2v'],5),round(c['witness']['fscore_2v'],5),'native',round(c['native_loss'],5),flush=True)


if __name__=='__main__':main()
