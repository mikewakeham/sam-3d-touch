"""Independently verify coverage arms, all rotated labels and Stage-1 geometry."""

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
from experiments.coordinate_system.scripts.shared_orientation.analyze_shared_orientation_geometry import analyze, aggregate as old_aggregate, same, witness
from statistics import mean
from experiments.coordinate_system.scripts.rotation_coverage.orientation_coverage_protocol import rotations, rotate_grid, inverse_frame, schedule
from experiments.coordinate_system.scripts.shared_orientation.shared_orientation_protocol import transform_points
from experiments.coordinate_system.scripts.shared.pose_shape_geometry import points, metrics, register

HERE=Path(__file__).resolve().parent
REPO=next(p for p in Path(__file__).resolve().parents if (_source_path(p, 'train.py')).is_file() and (p / 'sam3d_objects').is_dir())


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def analyze_case(task):
    support=np.load(task['p'],allow_pickle=False)[task['index']]
    if support.any():return analyze(task)
    # Empty predictions are failures, not missing observations. The established
    # metrics assign zero P/R/F-score and unavailable distance; preserve both.
    row=task['row'];q=np.load(task['q'],allow_pickle=False)[task['index']]
    original=np.load(task['original'],allow_pickle=False)[task['index']]
    p=points(support);target=points(original);native=metrics(p,points(q));raw=metrics(p,target)
    assert row['unaligned_voxel_center_chamfer'] is None
    same(raw,row['common_object_units'])
    return {k:row[k] for k in ('condition','group','draw','sample_id','object_id','split')}|dict(
        raw_iou=row['voxel_iou'],raw=raw,rigid=register(p,target),native=native,
        covariance_axis_std=dict(prediction=None,target=np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(target.T)),0)).tolist()))


def aggregate(rows):
    empty=sum(r['raw']['empty'] for r in rows)
    if not empty:return dict(**old_aggregate(rows),empty_predictions=0)
    result=dict(n=len(rows),iou=mean(r['raw_iou'] for r in rows),empty_predictions=empty)
    for name,get in [('raw',lambda r:r['raw']),('rigid',lambda r:r['rigid']['metrics']),('witness',witness)]:
        mm=[get(r) for r in rows]
        result[name]={k:(None if any(m[k] is None for m in mm) else mean(m[k] for m in mm)) for k in
            ('precision_1v','recall_1v','fscore_1v','precision_2v','recall_2v','fscore_2v','p95_distance')}
        result[name]['individual_passes_2v']=sum(min(m['precision_2v'],m['recall_2v'])>=.95 for m in mm)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control-bundle',type=Path,required=True)
    ap.add_argument('--augmented-bundle',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.output_dir
    if args.resume:assert out.is_dir() and not (out/'complete.json').exists()
    else:out.mkdir(parents=True,exist_ok=False)
    oldroot=_source_path(HERE, 'shared_orientation_visuals_analysis/bundle')
    oldprobe=json.loads((oldroot/'visual_probe/results.json').read_text())
    oldfit=json.loads((oldroot/'reference/fit_results.json').read_text())
    rr=rotations();ss=schedule();reports={};validations={};tasks=[];bank_files={}
    controls=json.loads((_source_path(HERE, 'pose_shape_analysis/controls.json')).read_text())
    assert sha(_source_path(HERE, 'pose_shape_geometry.py'))==controls['config']['source_sha256']['pose_shape_geometry.py']
    assert all(min(c['rigid']['metrics']['precision_1v'],c['rigid']['metrics']['recall_1v'])>=.95
               for c in controls['controls'] if c['kind'].startswith('known_rigid'))
    for arm,bundle in [('control',args.control_bundle),('augmented',args.augmented_bundle)]:
        root=out/'bundles'/arm
        with zipfile.ZipFile(bundle) as z:
            assert z.testzip() is None
            m=json.loads(z.read('bundle_manifest.json'));assert m['format_version']==6
            assert len(m['cases'])==64 and len(m['target_bank_cases'])==96
            names=[e['archive_path'] for e in m['files']]
            assert len(names)==len(set(names))==354
            assert len(z.namelist())==355 and set(z.namelist())==set(names)|{'bundle_manifest.json'}
            for e in m['files']:
                b=z.read(e['archive_path']);assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
            for name in z.namelist():
                dest=(root/name).resolve();assert dest.is_relative_to(root.resolve())
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
        d=json.loads((root/'coverage/results.json').read_text());reports[arm]=d
        bank=json.loads((root/'reference/bank_results.json').read_text())
        assert json.loads((root/'reference/fit_results.json').read_text())==oldfit
        assert json.loads((root/'reference/visual_probe_results.json').read_text())==oldprobe
        for field,file in [('fit','fit_results'),('probe','visual_probe_results'),('bank','bank_results')]:
            assert sha(root/f'reference/{file}.json')==d[field+'_report_sha256']
        assert bank['complete'] and not bank['failed_quality']
        assert bank['fit_report_sha256']==d['fit_report_sha256'] and bank['probe_report_sha256']==d['probe_report_sha256']
        assert bank['source_sha256']==d['source_sha256']
        for name,value in d['source_sha256'].items():assert sha(_source_path(REPO, name))==value,name
        assert d['complete'] and d['frozen_parameters_unchanged'] and d['parameters_unchanged_during_sampling']
        assert d['settings']['arm']==arm and d['settings']['base_step']==2000 and d['settings']['additional_steps']==1000
        assert d['initial_all_parameters_sha256']==oldfit['final_all_parameters_sha256']
        assert d['final_all_parameters_sha256']!=d['initial_all_parameters_sha256']
        assert d['trainable_counts']==oldfit['trainable_counts']
        for key in ('input_batches','target_frames','target_quality'):assert d[key]==oldfit[key]
        assert d['schedule']==ss and d['schedule_sha256']==hashlib.sha256(json.dumps(ss,sort_keys=True).encode()).hexdigest()
        assert len(d['training'])==1000
        for r,s in zip(d['training'],ss,strict=True):
            assert {k:r[k] for k in s}==s
            assert r['applied_rotation_index']==(s['rotation_index'] if arm=='augmented' else 0)
            assert np.isfinite([r['loss'],r['preclip_gradient_norm']]).all()
        assert len(d['interface_checks'])==(8 if arm=='augmented' else 7) and all(d['interface_checks'].values())
        for a in d['native']:
            assert len(a['rows'])==28 and {(r['visual_mode'],r['group'],r['wrong']) for r in a['rows']}=={(v,g,w) for v in ('present','zero') for g in range(7) for w in (False,True)}
            assert all(len(r['losses'])==8 and np.isfinite(r['losses']).all() for r in a['rows'])
        assert [a['step'] for a in d['native']]==[2000,2300,3000]
        assert d['native'][0]['rows']==oldprobe['native']
        assert d['resume_replay']['max_absolute_error']==d['resume_replay']['max_relative_error']==0
        for name in ['reference/parent_results.json','reference/original_target_occupancy.npy']+[f'physical/{f["sample_id"]}.npy' for f in d['target_frames']]:assert sha(root/name)==sha(oldroot/name)
        original_path=root/'reference/original_target_occupancy.npy';original=np.load(original_path,allow_pickle=False)
        frames={f['sample_id']:f for f in d['target_frames']}
        bankquality={(q['group'],q['rotation_index'],q['sample_id']):q for q in bank['quality']}
        assert len(bankquality)==384 and bank['rotations']==[r.tolist() for r in rr]
        bankentries={e['key']:e for e in bank['entries']};assert len(bankentries)==96
        archived_npz={e['prefix']:e for e in m['source_npz']}
        bank_seen=set()
        for case in m['target_bank_cases']:
            g,ri=case['group'],case['rotation_index'];assert g<4 and (g,ri) not in bank_seen;bank_seen.add((g,ri))
            assert {k:v for k,v in case.items() if k!='prefix'} in bank['decoded_artifacts']
            assert archived_npz[case['prefix']]['sha256']==case['sha256']
            pp,qp=[root/case['prefix']/f'{k}_occupancy.npy' for k in ('predicted','target')]
            p,q=[np.load(f,allow_pickle=False) for f in (pp,qp)]
            assert p.shape==q.shape==(4,64,64,64) and p.dtype==q.dtype==np.bool_
            e=bankentries[f'{g}/{ri}']
            assert hashlib.sha256(p.astype(np.float32).tobytes()).hexdigest()==e['support_sha256']
            assert hashlib.sha256(q.astype(np.float32).tobytes()).hexdigest()==e['physical_sha256']
            assert case['sample_ids']==d['input_batches'][g]['sample_ids']
            for file in (pp,qp):
                name=str(file.relative_to(root))
                if arm=='control':bank_files[name]=sha(file)
                else:assert bank_files[name]==sha(file)
            # Both arms contain the exact same bank; validate its384 geometries
            # once, then require every corresponding payload hash to match.
            if arm=='control':
                for i,sid in enumerate(case['sample_ids']):
                    physical=np.load(root/f'physical/{sid}.npy',allow_pickle=False)
                    np.testing.assert_array_equal(q[i],rotate_grid(physical,rr[ri]))
                    qi=bankquality[g,ri,sid]
                    assert np.count_nonzero(p[i]&q[i])/np.count_nonzero(p[i]|q[i])==qi['roundtrip']['voxel_iou']
                    assert int(p[i].sum())==qi['roundtrip']['predicted_occupied'] and int(q[i].sum())==qi['roundtrip']['target_occupied']
                    np.testing.assert_allclose(metrics(points(p[i]),points(q[i]))['mean_distance'],qi['roundtrip']['unaligned_voxel_center_chamfer'],rtol=1e-12,atol=1e-12)
                    inv=inverse_frame(frames[sid]['object_from_output'],rr[ri])
                    same(metrics(transform_points(points(p[i]),inv),points(original[i])),qi['common_object_units'])
            if ri==0:np.testing.assert_array_equal(p,np.load(oldroot/f'visual_probe/present_correct_surface_g{g}_d0/target_occupancy.npy',allow_pickle=False))
        assert bank_seen=={(g,ri) for g in range(4) for ri in range(24)}
        rows={(r['visual_mode'],r['condition'],r['group'],r['rotation_index'],r['draw'],r['sample_id']):r for r in d['rows']}
        oldrows={(r['visual_mode'],r['condition'],r['group'],r['draw'],r['sample_id']):r for r in oldprobe['rows']}
        assert len(rows)==len(d['rows'])==256 and len(d['artifacts'])==64
        seen=set()
        for case in m['cases']:
            assert case['kind']=='coverage' and {k:v for k,v in case.items() if k not in ('kind','prefix')} in d['artifacts']
            g,ri,v,c,draw=(case[k] for k in ('group','rotation_index','visual_mode','condition','draw'))
            pp,qp=[root/case['prefix']/f'{k}_occupancy.npy' for k in ('predicted','target')]
            p,q=[np.load(f,allow_pickle=False) for f in (pp,qp)]
            assert p.shape==q.shape==(4,64,64,64) and p.dtype==q.dtype==np.bool_
            expected=root/f'target_bank/target_bank_g{g}_r{ri}/predicted_occupancy.npy' if ri else oldroot/f'visual_probe/present_correct_surface_g{g}_d0/target_occupancy.npy'
            np.testing.assert_array_equal(q,np.load(expected,allow_pickle=False))
            assert case['sample_ids']==d['input_batches'][g]['sample_ids']
            for i,sid in enumerate(case['sample_ids']):
                key=(v,c,g,ri,draw,sid);assert key not in seen;seen.add(key);r=rows[key]
                assert r['noise_sha256']==oldrows[v,c,g,draw,sid]['noise_sha256']
                assert r['split']==('augmented_fit_diagnostic' if ri else 'fit' if g<4 else 'reserved_view')
                inv=inverse_frame(frames[sid]['object_from_output'],rr[ri]);np.testing.assert_array_equal(inv,np.array(r['object_from_output']))
                assert r['voxel_iou']==np.count_nonzero(p[i]&q[i])/np.count_nonzero(p[i]|q[i])
                assert r['predicted_occupied']==int(p[i].sum()) and r['target_occupied']==int(q[i].sum())
                tasks.append(dict(p=str(pp),q=str(qp),original=str(original_path),index=i,row=r,inverse=inv.tolist(),arm=arm,visual_mode=v))
        assert seen==set(rows)
        validations[arm]=dict(bundle_sha256=sha(bundle),payload_members=354,rows=256,rotated_labels=384,
            native_replay_exact=True,bank_quality_min_iou=min(q['roundtrip']['voxel_iou'] for q in bank['quality']),
            bank_quality_min_common_pr=min(min(q['common_object_units']['precision_2v'],q['common_object_units']['recall_2v']) for q in bank['quality']))
        print('Verified',arm,'bundle, schedules, labels and native IoUs.',flush=True)
    a,b=reports['control'],reports['augmented']
    for k in ('fit_report_sha256','probe_report_sha256','bank_report_sha256','source_sha256','initial_all_parameters_sha256','restored_optimizer_sha256','schedule_sha256','schedule','input_batches','target_frames','target_quality','trainable_counts','resume_replay'):assert a[k]==b[k],k
    assert {k:v for k,v in a['settings'].items() if k!='arm'}=={k:v for k,v in b['settings'].items() if k!='arm'}
    validation=dict(arms=validations,matched_start_optimizer_exposure=True,all384_target_bank_geometries_reproduced=True,
        source_sha256={n:sha(_source_path(HERE, n)) for n in ('analyze_orientation_coverage.py','analyze_shared_orientation_geometry.py','pose_shape_geometry.py','orientation_coverage_protocol.py')},
        scope='Checkpoint/Adam/feature hashes source-audited from GPU reports; raw model/feature tensors remain on cluster. All supplied supports checked locally.')
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    result=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()] if args.resume else []
    key=lambda r:(r['model'],r['condition'],r['group'],r['rotation_index'],r['draw'],r['sample_id'])
    done={key(r) for r in result};assert len(done)==len(result)
    expected={key(dict(t['row'],model=t['arm']+'_'+t['visual_mode'])) for t in tasks};assert done<=expected
    tasks=[t for t in tasks if key(dict(t['row'],model=t['arm']+'_'+t['visual_mode'])) not in done]
    print('Reusing',len(result),'completed rows; analyzing',len(tasks),flush=True)
    with (out/'predictions.jsonl').open('a' if args.resume else 'x') as f,ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(task,r) in enumerate(zip(tasks,pool.map(analyze_case,tasks),strict=True),1):
            r.update(model=task['arm']+'_'+task['visual_mode'],rotation_index=task['row']['rotation_index'])
            result.append(r);f.write(json.dumps(r)+'\n');f.flush()
            if i%16==0:print('Registered',len(result),'/512',flush=True)
    assert len(result)==512 and {key(r) for r in result}==expected
    cells={}
    for split in ('fit','reserved_view','augmented_fit_diagnostic'):
        cc={};selected=[r for r in result if r['split']==split]
        for model,c in sorted({(r['model'],r['condition']) for r in selected}):
            rows=[r for r in selected if (r['model'],r['condition'])==(model,c)];cell=aggregate(rows)
            cell['objects']={oid:aggregate([r for r in rows if r['object_id']==oid]) for oid in sorted({r['object_id'] for r in rows})}
            cell['native_endpoint']=cell['iou']>=.95 and all(x['iou']>=.9 for x in cell['objects'].values())
            cell['common_geometry_endpoint']=all(min(x['witness']['precision_2v'],x['witness']['recall_2v'])>=.98 for x in cell['objects'].values())
            cc[model+'/'+c]=cell
        if split!='augmented_fit_diagnostic':
            for model in ('control_present','control_zero','augmented_present','augmented_zero'):
                c,w=[cc[model+'/'+condition] for condition in ('correct_surface','wrong_surface')]
                for oid,x in c['objects'].items():x['surface_advantage_2v']=x['witness']['fscore_2v']-w['objects'][oid]['witness']['fscore_2v']
                c['dependence_endpoint']=all(x['surface_advantage_2v']>=.1 for x in c['objects'].values())
        cells[split]=cc
    summary=dict(cells=cells,scope='Same two-original-voxel tolerance and fixed proper-rigid candidate search. Four fitted identities/development views. Augmented-fit diagnostic covers16 of368 new conditions, not the full bank.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps(dict(validation=validation,rows=512,predictions_sha256=sha(out/'predictions.jsonl'),summary_sha256=sha(out/'summary.json')),indent=2)+'\n')
    for split,cc in cells.items():
        for name,c in cc.items():print(split,name,'IoU',round(c['iou'],5),'raw/witnessF2',round(c['raw']['fscore_2v'],5),round(c['witness']['fscore_2v'],5),flush=True)


if __name__=='__main__':main()
