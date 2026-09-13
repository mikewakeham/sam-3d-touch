"""Exploratory: compare every reserved correct prediction to all four fitted shapes."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from pose_shape_geometry import points, metrics, register

HERE=Path(__file__).resolve().parent
ROOT=HERE/'camera_dropout_geometry_analysis'


def run(job):
    ppath,qpath,index,target_index,meta=job
    p=points(np.load(ppath,allow_pickle=False)[index])
    q=points(np.load(qpath,allow_pickle=False)[target_index])
    raw=metrics(p,q);rigid=register(p,q)
    choice=max((raw,rigid['metrics']),key=lambda m:min(m['precision_1v'],m['recall_1v']))
    return meta|dict(raw=raw,rigid=rigid,witness=choice)


def main():
    report=json.loads((ROOT/'bundle/camera_dropout/results.json').read_text())
    previous=[json.loads(s) for s in (ROOT/'predictions.jsonl').read_text().splitlines()]
    ids=[s.rsplit('_',1)[0] for s in report['input_batches'][0]['sample_ids']]
    jobs=[];rows=[]
    for r in previous:
        if r['split']!='reserved_view' or r['condition']!='correct_surface':continue
        root=ROOT/f'bundle/camera_dropout/correct_surface_g{r["group"]}_d{r["draw"]}'
        index=ids.index(r['object_id'])
        for i,obj in enumerate(ids):
            meta={k:r[k] for k in ('sample_id','object_id','group','draw')}
            meta['comparison_object']=obj
            if i==index:
                choice=max((r['raw'],r['rigid']['metrics']),key=lambda m:min(m['precision_1v'],m['recall_1v']))
                rows.append(meta|dict(raw=r['raw'],rigid=r['rigid'],witness=choice))
            else:jobs.append((root/'predicted_occupancy.npy',root/'target_occupancy.npy',index,i,meta))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(run,jobs),1):
            rows.append(r)
            if i%24==0:print('Cross-object comparisons',i,'/72',flush=True)
    assert len(rows)==96
    summary=[]
    for sid,draw in sorted({(r['sample_id'],r['draw']) for r in rows}):
        group=[r for r in rows if (r['sample_id'],r['draw'])==(sid,draw)]
        correct=next(r for r in group if r['comparison_object']==r['object_id'])
        best=max(group,key=lambda r:min(r['witness']['precision_1v'],r['witness']['recall_1v']))
        summary.append(dict(sample_id=sid,draw=draw,best_object=best['comparison_object'],
            correct_object=correct['object_id'],correct_fscore=correct['witness']['fscore_1v'],
            best_fscore=best['witness']['fscore_1v'],best_precision=best['witness']['precision_1v'],
            best_recall=best['witness']['recall_1v']))
    result=dict(scope='Exploratory post-result diagnostic, all 24 reserved correct-surface predictions, '
        'all four fitted target supports. Same proper-rigid procedure, no scale. Candidate identity selected '
        'by maximum weaker P/R. Not an independent classification benchmark or proof of latent mechanism.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,summary=summary)
    (ROOT/'cross_identity.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in summary:
        if r['best_object']!=r['correct_object']:print(r,flush=True)


if __name__=='__main__':main()
