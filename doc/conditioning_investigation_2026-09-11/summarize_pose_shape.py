"""Report raw and pose-adjusted shape on identical support metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st


def mean_metrics(rows, method):
    values=[r['rigid']['metrics'] if method=='rigid' else r[method] for r in rows]
    return {k:st.mean(v[k] for v in values if v[k] is not None)
            if any(v[k] is not None for v in values) else None for k in values[0] if k!='empty'}


def meets(value,reference):
    return all(value[k]>=max(.95,reference[k]-.02) for k in ('precision_1v','recall_1v'))


def summarize(root):
    completed=json.loads((root/'complete.json').read_text())
    path=root/'predictions.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==completed['predictions_sha256']
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    assert len(rows)==completed['rows']
    by={}
    for r in rows:by.setdefault((r['split'],r['model'],r['condition'],r['object_id']),[]).append(r)
    per_object={}
    for key,group in by.items():
        per_object[key]={'n':len(group),'raw_iou':st.mean(r['raw_iou'] for r in group),
                         **{m:mean_metrics(group,m) for m in ('raw','known_inverse','rigid')},
                         'max_registered_p95':max(r['rigid']['metrics']['p95_distance'] or 0 for r in group),
                         'individual':[]}
        reference_rows=by[key[0],'dropout','aligned',key[3]]
        references={m:mean_metrics(reference_rows,m) for m in ('raw','known_inverse','rigid')}
        result=per_object[key]
        for method in ('raw','known_inverse','rigid'):
            # Same metric and method for the positive reference.
            reference=references['raw' if method=='known_inverse' else method]
            result[method+'_meets_reference']=meets(result[method],reference)
        for row in group:
            reference=next(r for r in reference_rows if r['group']==row['group'] and r['draw']==row['draw'])
            result['individual'].append({'sample_id':row['sample_id'],'draw':row['draw'],
                'raw_meets_reference':meets(row['raw'],reference['raw']),
                'rigid_meets_reference':meets(row['rigid']['metrics'],reference['rigid']['metrics']),
                'known_inverse_meets_reference':meets(row['known_inverse'],reference['raw'])})
    cells={}
    for (split,model,condition,obj),value in per_object.items():
        cells.setdefault(split,{}).setdefault(model+'/'+condition,{})[obj]=value
    summary={}
    for split,conditions in cells.items():
        summary[split]={}
        for condition,objects in conditions.items():
            summary[split][condition]={'objects':objects,'mean_iou':st.mean(v['raw_iou'] for v in objects.values())}
            for method in ('raw','known_inverse','rigid'):
                summary[split][condition][method]={k:st.mean(v[method][k] for v in objects.values()) for k in ('precision_1v','recall_1v','fscore_1v','mean_distance','p95_distance')}
                summary[split][condition][method]['objects_meeting_reference']=sum(v[method+'_meets_reference'] for v in objects.values())
                summary[split][condition][method]['individual_meeting_reference']=sum(x[method+'_meets_reference'] for v in objects.values() for x in v['individual'])
    return {'cells':summary,'source':completed,'limits':'Four fitted identities. Proper rigid GT-assisted registration is diagnostic only; '
            'same-metric aligned-dropout reference and >=95% precision/recall within one voxel with <=2pp loss. '
            'Registration is not globally certified. No new-object or deployment claim. Empty predictions count as failures.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    result=summarize(args.root)
    (args.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for condition,value in result['cells']['reserved_view'].items():
        print(condition,'IoU',round(value['mean_iou'],4),
              'raw/inverse/rigid F1',*[round(value[m]['fscore_1v'],4) for m in ('raw','known_inverse','rigid')],
              'passing objects',*[value[m]['objects_meeting_reference'] for m in ('raw','known_inverse','rigid')])


if __name__=='__main__':main()
