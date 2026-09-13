"""Verify the matched continuation bundles and reuse the fixed geometry analysis."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from analyze_shared_orientation_geometry import analyze,aggregate,witness,same
from pose_shape_geometry import points,metrics
from shared_orientation_protocol import transform_points
from shared_orientation_scope_protocol import continuation_schedule,additional_shape_parameter
from visual_dropout_protocol import schedule_digest

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--current-bundle',type=Path,required=True)
    ap.add_argument('--shape-path-bundle',type=Path,required=True)
    ap.add_argument('--current-report',type=Path,required=True)
    ap.add_argument('--shape-path-report',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    oldroot=HERE/'shared_orientation_geometry_analysis'
    parent=json.loads((oldroot/'bundle/shared_orientation/results.json').read_text())
    oldcomplete=json.loads((oldroot/'complete.json').read_text())
    assert sha(oldroot/'predictions.jsonl')==oldcomplete['predictions_sha256']
    parent_original=np.load(oldroot/'bundle/reference/original_target_occupancy.npy',allow_pickle=False)
    reports={};validations={};tasks=[]
    for arm,bundle,pasted in [('current',args.current_bundle,args.current_report),
                              ('shape_path',args.shape_path_bundle,args.shape_path_report)]:
        root=out/'bundles'/arm;prefix='scope_'+arm
        with zipfile.ZipFile(bundle) as z:
            assert z.testzip() is None
            m=json.loads(z.read('bundle_manifest.json'));assert m['format_version']==4 and len(m['cases'])==28
            names=[f['archive_path'] for f in m['files']]
            assert len(names)==len(set(names))==87
            assert len(z.namelist())==88 and set(z.namelist())==set(names)|{'bundle_manifest.json'}
            for entry in m['files']:
                b=z.read(entry['archive_path'])
                assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256']
            for name in z.namelist():
                dest=(root/name).resolve();assert dest.is_relative_to(root.resolve())
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
        d=json.loads((root/prefix/'results.json').read_text());reports[arm]=d
        assert d==json.loads(pasted.read_text())
        assert d['complete'] and d['frozen_parameters_unchanged'] and d['parameters_unchanged_during_sampling']
        assert d['additional_parameters_changed']==(arm=='shape_path')
        assert d['settings']['arm']==arm
        assert json.loads((root/'reference/parent_results.json').read_text())==parent
        assert sha(root/'reference/parent_results.json')==d['parent_report_sha256']
        assert d['initial_all_parameters_sha256']==parent['final_all_parameters_sha256']
        assert d['final_all_parameters_sha256']!=d['initial_all_parameters_sha256']
        for name,value in d['source_sha256'].items():assert sha(REPO/name)==value,name
        assert d['input_batches']==parent['input_batches'] and d['target_frames']==parent['target_frames']
        assert d['target_quality']==parent['target_quality']
        flags=continuation_schedule();assert schedule_digest(flags)==d['dropout_schedule_sha256']
        assert len(d['training'])==1000
        for i,r in enumerate(d['training']):
            assert (r['step'],r['group'],r['visual_dropped'])==(1001+i,i%4,flags[i])
            assert math.isfinite(r['loss']) and math.isfinite(r['preclip_gradient_norm'])
        if arm=='shape_path':
            assert d['training'][0]['additional_gradient_norm']>0
            assert all(additional_shape_parameter(n) for n in d['additional_parameter_names'])
        else:assert not d['additional_parameter_names']
        replay=d['assessments'][0];assert replay['step']==1000
        expected=[r for r in parent['assessments'][-1]['rows'] if r['group']==0]
        for r,e in zip(replay['rows'],expected,strict=True):
            assert r['group']==e['group'] and r['wrong']==e['wrong']
            np.testing.assert_allclose(r['losses'],e['losses'],rtol=1e-3,atol=1e-6)
        original_path=root/'reference/original_target_occupancy.npy'
        original=np.load(original_path,allow_pickle=False);np.testing.assert_array_equal(original,parent_original)
        for f in d['target_frames']:
            sid=f['sample_id'];np.testing.assert_array_equal(
                np.load(root/f'physical/{sid}.npy',allow_pickle=False),
                np.load(oldroot/f'bundle/physical/{sid}.npy',allow_pickle=False))
        frames={r['sample_id']:r for r in d['target_frames']}
        rows={(r['condition'],r['group'],r['draw'],r['sample_id']):r for r in d['rows']}
        oldrows={(r['condition'],r['group'],r['draw'],r['sample_id']):r for r in parent['rows']}
        assert len(rows)==len(d['rows'])==112 and len(d['artifacts'])==28
        seen=set()
        for case in m['cases']:
            assert case['kind']==prefix
            assert {k:v for k,v in case.items() if k not in ('kind','prefix')} in d['artifacts']
            pp,qp=[root/case['prefix']/f'{k}_occupancy.npy' for k in ('predicted','target')]
            p,q=[np.load(path,allow_pickle=False) for path in (pp,qp)]
            assert p.shape==q.shape==(4,64,64,64) and p.dtype==q.dtype==np.bool_
            np.testing.assert_array_equal(q,np.load(oldroot/f'bundle/shared_orientation/correct_surface_g{case["group"]}_d0/target_occupancy.npy',allow_pickle=False))
            assert case['sample_ids']==d['input_batches'][case['group']]['sample_ids']
            for i,sid in enumerate(case['sample_ids']):
                key=(case['condition'],case['group'],case['draw'],sid);assert key not in seen;seen.add(key)
                r=rows[key];assert r['noise_sha256']==oldrows[key]['noise_sha256']
                assert r['split']==oldrows[key]['split'] and r['object_id']==oldrows[key]['object_id']
                iou=float(np.count_nonzero(p[i]&q[i])/np.count_nonzero(p[i]|q[i]))
                assert iou==r['voxel_iou'] and int(p[i].sum())==r['predicted_occupied'] and int(q[i].sum())==r['target_occupied']
                tasks.append(dict(p=str(pp),q=str(qp),original=str(original_path),index=i,row=r,
                                  inverse=frames[sid]['object_from_output'],arm=arm))
        assert seen==set(rows)
        validations[arm]=dict(bundle_sha256=sha(bundle),report_sha256=sha(root/prefix/'results.json'),
            payload_members=87,rows=112,target_arrays_match_parent=True,native_ious_reproduced=True,
            replay_max_relative=d['resume_replay']['max_relative_error'])
    a,b=reports['current'],reports['shape_path']
    for key in ('parent_report_sha256','parent_checkpoint_sha256','source_sha256','initial_all_parameters_sha256',
                'restored_optimizer_sha256','common_parameter_names','input_batches','target_cache_sha256',
                'target_frames','target_quality','dropout_schedule_sha256','resume_replay'):
        assert a[key]==b[key],key
    assert {k:v for k,v in a['settings'].items() if k!='arm'}=={k:v for k,v in b['settings'].items() if k!='arm'}
    assert a['assessments'][0]==b['assessments'][0]
    assert a['training'][0]['loss']==b['training'][0]['loss']
    for folder in ('camera_dropout_geometry_analysis','pose_shape_analysis'):
        c=json.loads((HERE/folder/'complete.json').read_text());assert sha(HERE/folder/'predictions.jsonl')==c['predictions_sha256']
    controls=json.loads((HERE/'pose_shape_analysis/controls.json').read_text())
    assert sha(HERE/'pose_shape_geometry.py')==controls['config']['source_sha256']['pose_shape_geometry.py']
    assert all(min(c['rigid']['metrics']['precision_1v'],c['rigid']['metrics']['recall_1v'])>=.95
               for c in controls['controls'] if c['kind'].startswith('known_rigid'))
    validation=dict(arms=validations,matched_start_and_exposure=True,first_update_loss_identical=True,
        scope='Weight/Adam content digests matched from independently returned reports; full checkpoint tensors remain on cluster. '
              'CPU independently verifies returned arrays. One training seed/four fitted identities.',
        source_sha256={n:sha(HERE/n) for n in ('analyze_shared_orientation_scope.py','analyze_shared_orientation_geometry.py','pose_shape_geometry.py')})
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('Both bundles, 224 rows, matched inputs/start/optimizer/exposure verified. Registering.',flush=True)
    result=[]
    with (out/'predictions.jsonl').open('x') as f,ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(task,row) in enumerate(zip(tasks,pool.map(analyze,tasks),strict=True),1):
            row['model']='scope_'+task['arm'];result.append(row);f.write(json.dumps(row)+'\n');f.flush()
            if i%16==0:print('Registered',i,'/224',flush=True)
    for folder in ('shared_orientation_geometry_analysis','camera_dropout_geometry_analysis','pose_shape_analysis'):
        rr=[json.loads(s) for s in (HERE/folder/'predictions.jsonl').read_text().splitlines()]
        result.extend(r for r in rr if r['model'] in ('shared_orientation','camera_dropout') or
                      r['model']=='dropout' and r['condition'] in ('aligned','wrong_surface'))
    cells={}
    for split in ('fit','reserved_view'):
        rr=[r for r in result if r['split']==split];cc={}
        for model,condition in sorted({(r['model'],r['condition']) for r in rr}):
            selected=[r for r in rr if (r['model'],r['condition'])==(model,condition)]
            c=aggregate(selected);c['objects']={oid:aggregate([r for r in selected if r['object_id']==oid]) for oid in sorted({r['object_id'] for r in selected})}
            cc[model+'/'+condition]=c
        for model in ('scope_current','scope_shape_path'):
            c,w=[cc[model+'/'+cond] for cond in ('correct_surface','wrong_surface')]
            for oid,v in c['objects'].items():
                v['surface_advantage_2v']=v['witness']['fscore_2v']-w['objects'][oid]['witness']['fscore_2v']
                v['common_reference_pass']=min(v['witness']['precision_2v'],v['witness']['recall_2v'])>=.98
            c['native_endpoint']=c['iou']>=.95 and all(v['iou']>=.9 for v in c['objects'].values())
            c['common_geometry_endpoint']=all(v['common_reference_pass'] for v in c['objects'].values())
            c['dependence_endpoint']=all(v['surface_advantage_2v']>=.1 for v in c['objects'].values())
        cells[split]=cc
    summary=dict(cells=cells,scope='Identical two-voxel original-unit tolerance; each native IoU against own decoded target. '
        'Raw and fixed proper-rigid candidates reported; no global optimum, fitted scale, or new-object claim.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps(dict(validation=validation,rows=224,
        predictions_sha256=sha(out/'predictions.jsonl'),summary_sha256=sha(out/'summary.json')),indent=2)+'\n')
    for split,cc in cells.items():
        for name,c in cc.items():print(split,name,'IoU',round(c['iou'],5),'F2 raw/witness',round(c['raw']['fscore_2v'],5),round(c['witness']['fscore_2v'],5),flush=True)


if __name__=='__main__':main()
