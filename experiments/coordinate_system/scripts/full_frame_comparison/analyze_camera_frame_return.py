"""Finish the camera/oracle comparison using unchanged prior rigid registration."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'oracle_upper_bound'
sys.path[:0] = [str(OLD), str(HERE.parent)]
from experiments.coordinate_system.scripts.full_checkpoints.analyze_rollout_geometry_local import work
from experiments.coordinate_system.scripts.full_frame_comparison.analyze_frame_probe import summarize as raw_summary
from experiments.coordinate_system.scripts.full_frame_comparison.frame_probe_core import normalization_check


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def key(r):
    return tuple(r[k] for k in ('arm','split','group','sample_id','visual','surface_shift','draw'))


def summarize(rows):
    out = {}
    for split in ('train','val'):
        out[split] = {}
        for visual in ('present','zero'):
            cells = {}
            for label,arm,shift in [('camera','camera',0),('camera_wrong','camera',1),('oracle','oracle',0),('oracle_wrong','oracle',1),('constant','constant',0)]:
                selected = [r for r in rows if (r['arm'],r['split'],r['visual'],r['surface_shift']) == (arm,split,visual,shift)]
                objects = {}
                for oid in sorted({r['object_id'] for r in selected}):
                    group = [r for r in selected if r['object_id']==oid]
                    assert len(group)==4
                    objects[oid] = {kind:{m:mean(r[kind][m] for r in group) for m in ('precision_2v','recall_2v','fscore_2v')} for kind in ('raw','witness')}
                assert len(objects)==16
                cells[label] = {'objects':objects}
                for kind in ('raw','witness'):
                    cells[label][kind] = {m:mean(v[kind][m] for v in objects.values()) for m in ('precision_2v','recall_2v','fscore_2v')}
                    cells[label][kind]['objects_95_precision_and_recall'] = sum(min(v[kind]['precision_2v'],v[kind]['recall_2v'])>=.95 for v in objects.values())
            for name,a,b in [('oracle_minus_camera','oracle','camera'),('camera_minus_constant','camera','constant'),('camera_minus_wrong','camera','camera_wrong'),('oracle_minus_wrong','oracle','oracle_wrong')]:
                cells[name] = {}
                for kind in ('raw','witness'):
                    delta = {oid:v[kind]['fscore_2v']-cells[b]['objects'][oid][kind]['fscore_2v'] for oid,v in cells[a]['objects'].items()}
                    cells[name][kind] = {'mean_difference':mean(delta.values()),'objects_positive':sum(v>0 for v in delta.values()),'per_object':delta}
            out[split][visual] = cells
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    camera_path=a.root/'camera/results.json';camera=json.loads(camera_path.read_text())
    rollouts={arm:json.loads((_source_path(OLD, 'rollouts_returned_20260913_143515')/arm/'results.json').read_text()) for arm in ('oracle','constant')}
    losses={arm:json.loads((_source_path(OLD, 'returned_20260913_133239')/arm/'results.json').read_text()) for arm in ('oracle','constant')}
    for name,digest in camera['files_sha256'].items():
        assert Path(name).name==name and sha(a.root/'camera'/name)==digest,name
    for name,digest in camera['source_sha256'].items():assert sha(next(p for p in Path(__file__).resolve().parents if (_source_path(p, 'train.py')).is_file() and (p / 'sam3d_objects').is_dir())/name)==digest,name
    for name,digest in camera['additional_source_sha256'].items():
        assert sha(a.root/'source'/Path(name).name)==digest,name
    assert raw_summary(camera,rollouts,losses)==json.loads((a.root/'summary.json').read_text())
    old_geo=_source_path(OLD, 'rollouts_geometry_20260913_143515')
    old_summary=json.loads((old_geo/'summary.json').read_text())
    assert sha(_source_path(OLD, 'analyze_rollout_geometry_local.py'))==old_summary['source_sha256']['analyze_rollout_geometry_local.py']
    assert sha(_source_path(HERE.parent, 'pose_shape_geometry.py'))==old_summary['source_sha256']['pose_shape_geometry.py']
    for arm in ('oracle','constant'):
        assert sha(_source_path(OLD, 'rollouts_returned_20260913_143515')/arm/'results.json')==old_summary['report_sha256'][arm]
    # Independently replay the saved actual pre-encoder normalization arithmetic.
    normalizations=[]
    for b in camera['inputs']:
        with np.load(a.root/'camera'/f"{b['split']}_g{b['group']}_coordinate_inputs.npz",allow_pickle=False) as f:
            assert list(f['sample_ids'])==b['sample_ids']
            for i,sid in enumerate(b['sample_ids']):
                row={'sample_id':sid}
                for frame in ('camera','oracle'):
                    row[frame]=normalization_check(f[f'{frame}_pre_encoder_points'][i],f[f'{frame}_vecsetx_points'][i],f[f'{frame}_center'][i],float(f[f'{frame}_inverse_radius'][i,0]))
                recorded=next(x for x in camera['normalization_checks'] if x['sample_id']==sid)
                assert row['camera']==recorded['camera'] and row['oracle']==recorded['oracle']
                normalizations.append(row)
    verification={'camera_report_sha256':sha(camera_path),'payload_hashes_verified':len(camera['files_sha256']),
                  'raw_summary_reproduced':True,'actual_normalization_arrays_reproduced':len(normalizations),
                  'registration_helper_sha256':old_summary['source_sha256'],
                  'normalization':{frame:{k:max(r[frame][k] for r in normalizations) for k in ('normalization_max_error','inverse_max_error')} for frame in ('camera','oracle')}}
    (a.output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    old_rows=[json.loads(line) for line in (old_geo/'predictions.jsonl').read_text().splitlines()]
    expected_old={key({'arm':arm,**s}):s for arm,r in rollouts.items() for s in r['samples']}
    assert len(old_rows)==len(expected_old)==768 and len({key(r) for r in old_rows})==768
    for r in old_rows:assert all(r[k]==v for k,v in expected_old[key(r)].items())
    progress=a.output/'camera_predictions.jsonl'
    rows=[json.loads(line) for line in progress.read_text().splitlines()] if a.resume and progress.exists() else []
    expected={key({'arm':'camera',**s}):s for s in camera['samples']}
    assert len({key(r) for r in rows})==len(rows)
    for r in rows:assert all(r[k]==v for k,v in expected[key(r)].items())
    done={key(r) for r in rows}
    pending=[(str(a.root),'camera',s) for k,s in expected.items() if k not in done]
    pending.sort(key=lambda t:(t[2]['surface_shift']!=0,t[2]['visual']!='present',t[2]['group']))
    print('Verified returned bank; registering',len(pending),'camera predictions',flush=True)
    start=time.monotonic()
    with progress.open('a' if a.resume else 'x') as f, ThreadPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(work,pending):
            rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
            if len(rows)%16==0:print(len(rows),'/512 registered;',round(time.monotonic()-start,1),'seconds',flush=True)
    assert len(rows)==512
    result={'complete':True,'summary':summarize(old_rows+rows),'verification':verification,
            'source_sha256':sha(Path(__file__)),
            'scope':'Fixed common proper-rigid search; no scale/reflection. Four observations averaged within each of16 objects per split. One training seed; repeated development bank. Poor registration is not an impossibility certificate.'}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for split,vs in result['summary'].items():
        for visual,c in vs.items():
            print(split,visual,'rigid camera/oracle/constant',*[round(c[x]['witness']['fscore_2v'],5) for x in ('camera','oracle','constant')],
                  'oracle minus camera',c['oracle_minus_camera']['witness']['mean_difference'],flush=True)


if __name__=='__main__':main()
