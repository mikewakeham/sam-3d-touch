"""Validate returned camera/dropout supports and reuse the frozen pose analysis."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics as st
import zipfile
import numpy as np
from pose_shape_geometry import points, metrics, register

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def analyze_one(task):
    p = np.load(task.pop('prediction_path'), allow_pickle=False)[task['index']]
    q = np.load(task.pop('target_path'), allow_pickle=False)[task.pop('index')]
    p, q = points(p), points(q)
    raw = metrics(p, q)
    assert abs(raw['mean_distance'] - task.pop('expected_distance')) < 1e-12
    return task | {'raw': raw, 'rigid': register(p, q),
                   'predicted_count': len(p), 'target_count': len(q)}


def witness(row):
    # Fixed before these results: maximize the weaker of precision and recall,
    # among identity and the previously fixed RMS-registration candidate only.
    return max((row['raw'], row['rigid']['metrics']),
               key=lambda m: min(m['precision_1v'], m['recall_1v']))


def aggregate(rows):
    result = {'n': len(rows), 'iou': st.mean(r['raw_iou'] for r in rows)}
    for name, get in [('raw', lambda r:r['raw']),
                      ('rigid', lambda r:r['rigid']['metrics']), ('witness', witness)]:
        mm = [get(r) for r in rows]
        result[name] = {k: st.mean(m[k] for m in mm) for k in
                       ('precision_1v','recall_1v','fscore_1v','fscore_2v','p95_distance')}
        result[name]['individual_absolute_passes'] = sum(
            min(m['precision_1v'], m['recall_1v']) >= .95 for m in mm)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--pasted-report', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    root = out/'bundle'
    with zipfile.ZipFile(args.bundle) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('bundle_manifest.json'))
        assert manifest['format_version'] == 2 and len(manifest['cases']) == 28
        names = [f['archive_path'] for f in manifest['files']]
        assert len(names) == len(set(names)) == 57
        assert set(archive.namelist()) == set(names) | {'bundle_manifest.json'}
        assert len(archive.namelist()) == 58
        for entry in manifest['files']:
            payload = archive.read(entry['archive_path'])
            assert len(payload) == entry['bytes']
            assert hashlib.sha256(payload).hexdigest() == entry['sha256']
        for name in archive.namelist():
            dest = (root/name).resolve()
            assert dest.is_relative_to(root.resolve())
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
    report = read(root/'camera_dropout/results.json')
    assert report == read(args.pasted_report)
    assert report['complete'] and report['parameters_unchanged']
    assert len(report['rows']) == 112 and len(report['artifacts']) == 28
    for name, value in report['reference_sha256'].items(): assert sha(HERE/name) == value
    for name, value in report['source_sha256'].items(): assert sha(REPO/name) == value
    paired = read(HERE/'camera_dropout_sampling_reference.json')
    drop = read(HERE/'visual_dropout_returned_manual/camera/results.json')
    assert report['checkpoint']['parameters_sha256'] == drop['final_all_parameters_sha256']
    assert report['target_support_content_sha256'] == paired['target_support_content_sha256']
    assert report['settings'] == dict(model_key='camera_dropout', seed=29, precision='bf16',
        steps=25, cfg=0, sampling_bank=200000, sampling_draws=2,
        conditions=['correct_surface','wrong_surface'])
    assert report['input_batches'] == drop['input_batches']
    for a,b in zip(report['input_batches'], paired['input_batches']):
        assert {k:v for k,v in a.items() if k!='features_sha256'} == {k:v for k,v in b.items() if k!='features_sha256'}
    assert len(report['replay']) == 14
    assert {(r['group'],r['wrong']) for r in report['replay']} == {(g,w) for g in range(7) for w in (False,True)}
    replay_max_error = 0.
    for row in report['replay']:
        key = 'swapped_surface_native_losses' if row['wrong'] else 'fresh_noise_native_losses'
        expected = drop['final_fresh']['visual_present'][row['group']][key]
        np.testing.assert_allclose(row['losses'], expected, rtol=1e-5, atol=1e-7)
        replay_max_error = max(replay_max_error, float(np.max(np.abs(np.array(row['losses'])-expected))))
    oldroot = HERE/'pose_shape_analysis'
    complete = read(oldroot/'complete.json')
    assert sha(oldroot/'predictions.jsonl') == complete['predictions_sha256']
    assert sha(HERE/'pose_shape_geometry.py') == complete['config']['source_sha256']['pose_shape_geometry.py']
    controls = read(oldroot/'controls.json')
    assert controls['config'] == complete['config']
    assert all(min(x['rigid']['metrics']['precision_1v'],x['rigid']['metrics']['recall_1v']) >= .95
               for x in controls['controls'] if x['kind'].startswith('known_rigid'))
    rows = {(r['condition'],r['group'],r['draw'],r['sample_id']):r for r in report['rows']}
    assert len(rows) == 112
    tasks, seen = [], set()
    for case in manifest['cases']:
        assert case['kind'] == 'camera_dropout'
        assert {k:v for k,v in case.items() if k not in ('kind','prefix')} in report['artifacts']
        ppath, qpath = [root/case['prefix']/f'{kind}_occupancy.npy' for kind in ('predicted','target')]
        p, q = [np.load(path, allow_pickle=False) for path in (ppath,qpath)]
        assert p.shape == q.shape == (4,64,64,64) and p.dtype == q.dtype == np.bool_
        assert hashlib.sha256(q.tobytes()).hexdigest() == paired['target_support_content_sha256']
        assert case['sample_ids'] == report['input_batches'][case['group']]['sample_ids']
        for i,sid in enumerate(case['sample_ids']):
            key = (case['condition'],case['group'],case['draw'],sid)
            assert key not in seen; seen.add(key)
            row = rows[key]
            assert row['noise_sha256'] == paired['noise'][row['object_id']+'/'+str(row['draw'])]
            iou = float(np.count_nonzero(p[i]&q[i])/np.count_nonzero(p[i]|q[i]))
            assert iou == row['voxel_iou']
            assert int(p[i].sum()) == row['predicted_occupied'] and int(q[i].sum()) == row['target_occupied']
            assert row['split'] == ('fit' if case['group']<4 else 'reserved_view')
            tasks.append({k:row[k] for k in ('condition','group','draw','sample_id','object_id','split')} |
                dict(model='camera_dropout',raw_iou=iou,prediction_path=str(ppath),target_path=str(qpath),
                     index=i,expected_distance=row['unaligned_voxel_center_chamfer']))
    assert seen == set(rows)
    validation = dict(bundle_sha256=sha(args.bundle),members=57,rows=112,
        report_sha256=sha(root/'camera_dropout/results.json'),replay_max_absolute_error=replay_max_error,
        source_sha256={n:sha(HERE/n) for n in ('analyze_camera_dropout_geometry.py','pose_shape_geometry.py')},
        reference_analysis_sha256=sha(oldroot/'complete.json'),
        checks='CRC/member hashes; pasted report equality; source/reference hashes; unchanged checkpoint parameter hash; '
        'all native replay losses; inputs, GT support and sampling noise; array dtypes/shapes/counts/IoUs; prior controls.')
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('Validated bundle/provenance and all 112 predictions.',flush=True)
    with (out/'predictions.jsonl').open('x') as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        new = []
        for i,row in enumerate(pool.map(analyze_one,tasks),1):
            new.append(row);f.write(json.dumps(row)+'\n');f.flush()
            if i%16 == 0: print('Registered',i,'/112',flush=True)
    old = [json.loads(s) for s in (oldroot/'predictions.jsonl').read_text().splitlines()]
    comparison = [r for r in old if (r['model'] in ('original','dropout') and r['condition']=='aligned') or r['model']=='camera']
    assert len(comparison) == 168
    allrows = new + comparison
    cells = {}
    for split in ('fit','reserved_view'):
        selected = [r for r in allrows if r['split']==split]
        cells[split] = {}
        for model, condition in sorted({(r['model'],r['condition']) for r in selected}):
            group = [r for r in selected if (r['model'],r['condition'])==(model,condition)]
            value = aggregate(group)
            value['objects'] = {obj:aggregate([r for r in group if r['object_id']==obj])
                                for obj in sorted({r['object_id'] for r in group})}
            cells[split][model+'/'+condition] = value
        camera = cells[split]['camera_dropout/correct_surface']
        wrong = cells[split]['camera_dropout/wrong_surface']
        oracle = cells[split]['dropout/aligned']
        for obj,v in camera['objects'].items():
            ref = oracle['objects'][obj]['witness']
            v['meets_pose_reference'] = all(v['witness'][k] >= max(.95,ref[k]-.02) for k in ('precision_1v','recall_1v'))
            v['wrong_surface_fscore_penalty'] = v['witness']['fscore_1v']-wrong['objects'][obj]['witness']['fscore_1v']
        camera['fixed_frame_gate'] = camera['iou']>=.95 and all(v['iou']>=.9 for v in camera['objects'].values())
        camera['pose_reference_gate'] = all(v['meets_pose_reference'] for v in camera['objects'].values())
        camera['surface_dependence_gate'] = all(v['wrong_surface_fscore_penalty']>=.1 for v in camera['objects'].values())
    summary = dict(cells=cells,validation=validation,
        scope='Four fitted identities, four fitted/three reserved views, two sampling seeds. No new-identity claim. '
        'Pose witness maximizes min(P,R) over identity and measured proper-rigid registration; no global optimum claim.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps(dict(rows=112,predictions_sha256=sha(out/'predictions.jsonl'),
        summary_sha256=sha(out/'summary.json'),validation=validation),indent=2)+'\n')
    for split, cc in cells.items():
        for name,v in cc.items():
            print(split,name,'IoU',round(v['iou'],5),'raw/rigid/witness F1',
                  *[round(v[m]['fscore_1v'],5) for m in ('raw','rigid','witness')],flush=True)


if __name__=='__main__':main()
