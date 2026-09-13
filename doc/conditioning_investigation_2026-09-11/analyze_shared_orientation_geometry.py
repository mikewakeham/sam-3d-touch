"""Replay format-3 shared-target geometry and the predeclared proper-rigid screen."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from statistics import mean
import zipfile
import numpy as np
from pose_shape_geometry import points, metrics, register
from shared_orientation_protocol import transform_points
from analyze_shared_orientation_report import main as validate_report

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same(actual, expected):
    assert actual.keys() == expected.keys()
    for k, v in expected.items():
        if isinstance(v, (float, int)):
            np.testing.assert_allclose(actual[k], v, atol=1e-12, rtol=1e-12, err_msg=k)
        else:
            assert actual[k] == v, k


def analyze(task):
    p = np.load(task['p'], allow_pickle=False)[task['index']]
    q = np.load(task['q'], allow_pickle=False)[task['index']]
    original = np.load(task['original'], allow_pickle=False)[task['index']]
    row = task['row']
    native = metrics(points(p), points(q))
    np.testing.assert_allclose(native['mean_distance'], row['unaligned_voxel_center_chamfer'], atol=1e-12, rtol=1e-12)
    mapped = transform_points(points(p), np.array(task['inverse']))
    target = points(original)
    raw = metrics(mapped, target)
    same(raw, row['common_object_units'])
    rigid = register(mapped, target)
    # Eigenvalues are invariant under rigid motion, but these support-weighted
    # moments are descriptive; they do not certify a precision/recall bound.
    moments = {name: np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(x.T)), 0)).tolist()
               for name, x in [('prediction', mapped), ('target', target)]}
    return {k: row[k] for k in ('condition','group','draw','sample_id','object_id','split')} | dict(
        model='shared_orientation', raw_iou=row['voxel_iou'], raw=raw, rigid=rigid,
        native=native, covariance_axis_std=moments)


def witness(row):
    return max((row['raw'], row['rigid']['metrics']),
               key=lambda m: min(m['precision_2v'], m['recall_2v']))


def aggregate(rows):
    result = dict(n=len(rows), iou=mean(r['raw_iou'] for r in rows))
    for name, get in [('raw', lambda r:r['raw']), ('rigid', lambda r:r['rigid']['metrics']), ('witness', witness)]:
        mm = [get(r) for r in rows]
        result[name] = {k:mean(m[k] for m in mm) for k in
            ('precision_1v','recall_1v','fscore_1v','precision_2v','recall_2v','fscore_2v','p95_distance')}
        result[name]['individual_passes_2v'] = sum(min(m['precision_2v'],m['recall_2v']) >= .95 for m in mm)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    root=out/'bundle'
    with zipfile.ZipFile(args.bundle) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('bundle_manifest.json'))
        assert manifest['format_version']==3 and len(manifest['cases'])==28
        names=[f['archive_path'] for f in manifest['files']]
        assert len(names)==len(set(names))==86
        assert len(z.namelist())==87 and set(z.namelist())==set(names)|{'bundle_manifest.json'}
        for entry in manifest['files']:
            payload=z.read(entry['archive_path'])
            assert len(payload)==entry['bytes'] and hashlib.sha256(payload).hexdigest()==entry['sha256']
        for name in z.namelist():
            dest=(root/name).resolve();assert dest.is_relative_to(root.resolve())
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
    report=json.loads((root/'shared_orientation/results.json').read_text())
    assert report==json.loads((HERE/'shared_orientation_returned_manual/results.json').read_text())
    validate_report()
    original_path=root/'reference/original_target_occupancy.npy'
    original=np.load(original_path,allow_pickle=False)
    assert original.dtype==np.bool_ and original.shape==(4,64,64,64)
    paired=json.loads((HERE/'camera_dropout_sampling_reference.json').read_text())
    assert hashlib.sha256(original.tobytes()).hexdigest()==paired['target_support_content_sha256']
    frames={r['sample_id']:r for r in report['target_frames']}
    quality={r['sample_id']:r for r in report['target_quality']}
    recorded={(r['condition'],r['group'],r['draw'],r['sample_id']):r for r in report['rows']}
    tasks=[];seen=set();target_arrays={}
    for case in manifest['cases']:
        assert case['kind']=='shared_orientation'
        assert {k:v for k,v in case.items() if k not in ('kind','prefix')} in report['artifacts']
        pp,qp=[root/case['prefix']/f'{k}_occupancy.npy' for k in ('predicted','target')]
        p,q=[np.load(path,allow_pickle=False) for path in (pp,qp)]
        assert p.shape==q.shape==(4,64,64,64) and p.dtype==q.dtype==np.bool_
        assert case['sample_ids']==report['input_batches'][case['group']]['sample_ids']
        for i,sid in enumerate(case['sample_ids']):
            key=(case['condition'],case['group'],case['draw'],sid);assert key not in seen;seen.add(key)
            row=recorded[key]
            iou=float(np.count_nonzero(p[i]&q[i])/np.count_nonzero(p[i]|q[i]))
            assert iou==row['voxel_iou'] and int(p[i].sum())==row['predicted_occupied'] and int(q[i].sum())==row['target_occupied']
            inverse=np.array(frames[sid]['object_from_output'])
            np.testing.assert_allclose(inverse@np.array(frames[sid]['output_from_object']),np.eye(4),atol=1e-12)
            if sid in target_arrays:
                np.testing.assert_array_equal(target_arrays[sid],q[i])
            else:
                target_arrays[sid]=q[i].copy()
                physical=np.load(root/f'physical/{sid}.npy',allow_pickle=False)
                assert physical.dtype==np.bool_ and physical.shape==(64,64,64)
                qi=quality[sid]
                assert np.count_nonzero(q[i]&physical)/np.count_nonzero(q[i]|physical)==qi['roundtrip']['voxel_iou']
                same(metrics(transform_points(points(q[i]),inverse),points(original[i])),qi['common_object_units'])
                np.testing.assert_allclose(metrics(points(q[i]),points(physical))['mean_distance'],qi['roundtrip']['unaligned_voxel_center_chamfer'],atol=1e-12)
            tasks.append(dict(p=str(pp),q=str(qp),original=str(original_path),index=i,row=row,inverse=inverse.tolist()))
    assert seen==set(recorded) and len(target_arrays)==28
    prior=json.loads((HERE/'pose_shape_analysis/complete.json').read_text())
    assert sha(HERE/'pose_shape_geometry.py')==prior['config']['source_sha256']['pose_shape_geometry.py']
    assert sha(HERE/'pose_shape_analysis/predictions.jsonl')==prior['predictions_sha256']
    controls=json.loads((HERE/'pose_shape_analysis/controls.json').read_text())
    assert controls['config']==prior['config']
    assert all(min(c['rigid']['metrics']['precision_1v'],c['rigid']['metrics']['recall_1v'])>=.95
               for c in controls['controls'] if c['kind'].startswith('known_rigid'))
    validation=dict(bundle_sha256=sha(args.bundle),members=86,samples=112,target_labels=28,
        report_matches=True,target_quality_arrays_reproduced=True,all_native_iou_counts_reproduced=True,
        registration_source_matches_prior_controls=True,source_sha256=sha(Path(__file__)))
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('Bundle, report, all target arrays and native IoUs verified; registering 112 predictions.',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool,(out/'predictions.jsonl').open('x') as f:
        for i,row in enumerate(pool.map(analyze,tasks),1):
            rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
            if i%16==0:print('Registered',i,'/112',flush=True)
    for filename in ('camera_dropout_geometry_analysis/predictions.jsonl','pose_shape_analysis/predictions.jsonl'):
        older=[json.loads(s) for s in (HERE/filename).read_text().splitlines()]
        rows.extend(r for r in older if r['model']=='camera_dropout' or
                    (r['model']=='dropout' and r['condition'] in ('aligned','wrong_surface')))
    cells={}
    for split in ('fit','reserved_view'):
        selected=[r for r in rows if r['split']==split];cc={}
        for model,condition in sorted({(r['model'],r['condition']) for r in selected}):
            rr=[r for r in selected if (r['model'],r['condition'])==(model,condition)]
            cell=aggregate(rr)
            cell['objects']={obj:aggregate([r for r in rr if r['object_id']==obj]) for obj in sorted({r['object_id'] for r in rr})}
            cc[model+'/'+condition]=cell
        correct,wrong=[cc['shared_orientation/'+c+'_surface'] for c in ('correct','wrong')]
        for obj,c in correct['objects'].items():
            c['meets_common_geometry_reference']=min(c['witness']['precision_2v'],c['witness']['recall_2v'])>=.98
            c['surface_advantage_2v']=c['witness']['fscore_2v']-wrong['objects'][obj]['witness']['fscore_2v']
        correct['native_endpoint']=correct['iou']>=.95 and all(c['iou']>=.9 for c in correct['objects'].values())
        correct['common_geometry_endpoint']=all(c['meets_common_geometry_reference'] for c in correct['objects'].values())
        correct['dependence_endpoint']=all(c['surface_advantage_2v']>=.1 for c in correct['objects'].values())
        cells[split]=cc
    summary=dict(cells=cells,scope='Four fitted identities. Two-voxel witness maximizes min(P,R) between raw and fixed RMS proper-rigid candidate. '
        'Known label inverse first; no fitted scale/reflection/warping. Registration not globally certified.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps(dict(validation=validation,rows=112,
        predictions_sha256=sha(out/'predictions.jsonl'),summary_sha256=sha(out/'summary.json')),indent=2)+'\n')
    for split,cc in cells.items():
        for name,c in cc.items(): print(split,name,'IoU',c['iou'],'F2 raw/rigid/witness',*[c[k]['fscore_2v'] for k in ('raw','rigid','witness')],flush=True)


if __name__=='__main__':main()
