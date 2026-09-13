"""Verify paired camera extension and report object-level frame differences."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'oracle_upper_bound'))
from analyze_checkpoint_rollouts import analyze as check_old_rollouts
from analyze_checkpoint_probe import analyze as check_old_losses
from frame_probe_core import check_loss_pair, check_report_pair


def summarize(camera, rollouts, losses):
    check_old_rollouts(rollouts);check_old_losses(losses)
    check_report_pair(camera,rollouts['oracle']);check_loss_pair(camera,losses['oracle'])
    shapes={};denoising={}
    for arm,r in {**rollouts,'camera':camera}.items():
        values=defaultdict(list)
        for s in r['samples']:
            for metric in ('iou','precision_2v','recall_2v','fscore_2v'):
                values[s['split'],s['visual'],s['surface_shift'],s['object_id'],metric].append(s[metric])
        shapes[arm]={k:mean(v) for k,v in values.items()}
    for arm,r in {**losses,'camera':camera}.items():
        inputs={(b['split'],b['group']):b for b in r['inputs']};values=defaultdict(list)
        for s in r['rows']:
            for oid,value in zip(inputs[s['split'],s['group']]['object_ids'],s['losses']):
                values[s['split'],s['visual'],s['surface_shift'],s['time_kind'],oid].append(value)
        denoising[arm]={k:mean(v) for k,v in values.items()}
    summary={}
    for split in ('train','val'):
        summary[split]={}
        for visual in ('present','zero'):
            objects=sorted({k[3] for k in shapes['camera'] if k[:3]==(split,visual,0)})
            cells={}
            for label,arm,shift in [('camera','camera',0),('camera_wrong','camera',1),('oracle','oracle',0),('oracle_wrong','oracle',1),('constant','constant',0)]:
                per_object=[{'object_id':oid,**{metric:shapes[arm][split,visual,shift,oid,metric]
                    for metric in ('iou','precision_2v','recall_2v','fscore_2v')}} for oid in objects]
                cells[label]={'objects':per_object,**{f'mean_{m}':mean(x[m] for x in per_object)
                    for m in ('iou','precision_2v','recall_2v','fscore_2v')}}
            for label,a,b in [('oracle_minus_camera','oracle','camera'),('camera_minus_constant','camera','constant'),('camera_minus_wrong','camera','camera_wrong')]:
                diffs=[{'object_id':x['object_id'],'fscore_difference':x['fscore_2v']-y['fscore_2v']}
                       for x,y in zip(cells[a]['objects'],cells[b]['objects'])]
                cells[label]={'objects':diffs,'mean_difference':mean(x['fscore_difference'] for x in diffs),
                              'objects_positive':sum(x['fscore_difference']>0 for x in diffs)}
            time_cells={}
            for kind in ('native','t_0.05','t_0.5','t_0.95'):
                per_object=[]
                for oid in objects:
                    row={'object_id':oid,**{arm:denoising[arm][split,visual,0,kind,oid] for arm in denoising}}
                    row['camera_wrong_mean']=mean(denoising['camera'][split,visual,s,kind,oid] for s in (1,2,3))
                    row['camera_minus_oracle']=row['camera']-row['oracle']
                    row['camera_wrong_minus_correct']=row['camera_wrong_mean']-row['camera']
                    per_object.append(row)
                time_cells[kind]={'objects':per_object,**{f'mean_{k}':mean(o[k] for o in per_object)
                    for k in ('camera','oracle','constant','camera_wrong_mean','camera_minus_oracle','camera_wrong_minus_correct')}}
            summary[split][visual]={'shape':cells,'loss':time_cells}
    return {'pairing_and_completeness_verified':True,'coordinate_contract_passed':camera['coordinate_contract_passed'],
            'summary':summary,'scope':'Raw pose-sensitive support scores; one seed per trained arm, views/draws averaged within objects. '
            'Proper-rigid shape analysis is a separate local follow-up on saved arrays. No new training or causal attribution to pure rotation alone.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera-dir',type=Path,required=True);p.add_argument('--probe-root',type=Path,required=True)
    p.add_argument('--rollout-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    camera=json.loads((a.camera_dir/'results.json').read_text())
    rollouts={arm:json.loads((a.rollout_root/arm/'results.json').read_text()) for arm in ('oracle','constant')}
    losses={arm:json.loads((a.probe_root/arm/'results.json').read_text()) for arm in ('oracle','constant')}
    for name,digest in camera['files_sha256'].items():
        relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
        assert hashlib.sha256((a.camera_dir/relative).read_bytes()).hexdigest()==digest, name
    assert hashlib.sha256((a.rollout_root/'oracle/results.json').read_bytes()).hexdigest()==camera['rollout_reference_sha256']
    assert hashlib.sha256((a.probe_root/'oracle/results.json').read_bytes()).hexdigest()==camera['reference_sha256']
    result=summarize(camera,rollouts,losses)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    reference_dir=a.output.parent/'references';reference_dir.mkdir(exist_ok=True)
    for arm in ('oracle','constant'):
        (reference_dir/f'{arm}_rollouts.json').write_text(json.dumps(rollouts[arm],indent=2)+'\n')
        (reference_dir/f'{arm}_losses.json').write_text(json.dumps(losses[arm],indent=2)+'\n')
    for split,vs in result['summary'].items():
        for visual,c in vs.items():
            print(split,visual,'camera/oracle/constant F2v:',
                  ' / '.join(f"{c['shape'][arm]['mean_fscore_2v']:.4f}" for arm in ('camera','oracle','constant')),flush=True)
    print('Return summary with the camera coordinate/support bundle for independent analysis.')


if __name__=='__main__':main()
