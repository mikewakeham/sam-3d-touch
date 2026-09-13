"""Static figures for complete pose/shape diagnostics; no metric registration changes."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/sam3d-geometry-mpl')
import json
from pathlib import Path
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pose_shape_geometry import points

HERE=Path(__file__).resolve().parent


def main():
    output=HERE/'pose_shape_analysis';root=HERE/'pose_shape_bundle'
    assert (output/'complete.json').is_file()
    rows=[json.loads(x) for x in (output/'predictions.jsonl').read_text().splitlines()]
    reserved=[r for r in rows if r['split']=='reserved_view']
    objects=sorted({r['object_id'] for r in rows})
    colors=['#167a83','#8a51a0','#d5801f','#c34b50']
    labels={'479bbf9062de4b3196f08e52d1c12821':'Shield','e846574df9334973955c80286af47ba7':'Bathtub',
            'fe200ce00a5d4c298eee89de0fc15f01':'Drill','146b6c1f10bd445da977424c64dbb051':'146b6c1f'}
    fig,axs=plt.subplots(1,3,figsize=(13,4.3),sharey=True)
    for obj,color in zip(objects,colors):
        values=[[],[],[]]
        for angle in (0,5,15,30):
            group=[r for r in reserved if r['object_id']==obj and r['model']=='dropout'
                   and r['condition']!='wrong_surface' and abs(r['degrees'])==angle]
            values[0].append(100*st.mean(r['raw_iou'] for r in group))
            values[1].append(100*st.mean(r['raw']['fscore_1v'] for r in group))
            values[2].append(100*st.mean(r['rigid']['metrics']['fscore_1v'] for r in group))
        for ax,v in zip(axs,values):ax.plot([0,5,15,30],v,'o-',color=color,label=labels[obj],lw=2)
    for ax,title in zip(axs,['Exact voxel overlap: raw IoU','Shape proximity: raw F-score','Same F-score after rigid registration']):
        ax.set_title(title,fontsize=11);ax.set_xlabel('Residual rotation magnitude (degrees)');ax.set_xticks([0,5,15,30]);ax.set_ylim(0,103);ax.grid(alpha=.2)
    axs[0].set_ylabel('Percent');axs[2].legend(loc='lower left',fontsize=8)
    fig.suptitle('Four fitted identities, reserved views — shape and pose are different measurements',fontsize=12)
    fig.text(.5,.015,'F-score uses bidirectional one-voxel proximity (1/64). Nonzero angles average six signed axes. Registration uses GT; it is not deployment performance.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(output/'rotation_shape_comparison.png',dpi=170);plt.close(fig)
    # Explicit post-result examples, not an unbiased evaluation subset.
    camera=[r for r in reserved if r['model']=='camera']
    selections=[('Natural camera: largest registration gain',max(camera,key=lambda r:r['rigid']['metrics']['fscore_1v']-r['raw']['fscore_1v'])),
                ('30-degree perturbation: largest residual',min([r for r in reserved if abs(r['degrees'])==30],key=lambda r:r['rigid']['metrics']['fscore_1v'])),
                ('5-degree perturbation: lowest raw IoU',min([r for r in reserved if abs(r['degrees'])==5],key=lambda r:r['raw_iou']))]
    manifest=json.loads((root/'bundle_manifest.json').read_text())
    fig,axs=plt.subplots(3,4,figsize=(13,10))
    selected=[]
    for row,(label,result) in enumerate(selections):
        case=next(c for c in manifest['cases'] if c['group']==result['group'] and c['draw']==result['draw'] and result['sample_id'] in c['sample_ids']
                  and ((result['model']=='camera' and c['kind']=='natural_camera') or
                       (c['kind']=='alignment' and c['policy']==result['model'] and c['condition']==result['condition'])))
        po=np.load(root/case['prefix']/'predicted_occupancy.npy');qo=np.load(root/case['prefix']/'target_occupancy.npy')
        if po.ndim==4:
            index=case['sample_ids'].index(result['sample_id']);po,qo=po[index],qo[index]
        p,q=points(po),points(qo);reg=result['rigid'];aligned=p@np.array(reg['rotation']).T+np.array(reg['translation'])
        bound=max(.55,float(max(np.abs(x).max() for x in [p,q,aligned]))+.02)
        for col,(cloud,pair,title) in enumerate([(p,(0,1),'Raw XY'),(p,(0,2),'Raw XZ'),(aligned,(0,1),'Registered XY'),(aligned,(0,2),'Registered XZ')]):
            ax=axs[row,col];a,b=pair
            ax.scatter(q[:,a],q[:,b],s=.6,c='#202020',alpha=.22,rasterized=True)
            ax.scatter(cloud[:,a],cloud[:,b],s=.6,c='#d56032',alpha=.28,rasterized=True)
            ax.set(xlim=(-bound,bound),ylim=(-bound,bound),aspect='equal');ax.set_title(title,fontsize=10);ax.grid(alpha=.15)
            if col==0:ax.set_ylabel(label+'\n'+result['sample_id'][:8]+' '+result['condition'],fontsize=9)
        selected.append({'selection':label,**result})
        axs[row,1].set_xlabel(f"Raw IoU {result['raw_iou']:.1%}; F-score {result['raw']['fscore_1v']:.1%}")
        axs[row,3].set_xlabel(f"Registered F-score {reg['metrics']['fscore_1v']:.1%}")
    fig.suptitle('Diagnostic examples selected after results: black = GT support; orange = prediction',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(output/'pose_shape_examples.png',dpi=150);plt.close(fig)
    (output/'figure_selection.json').write_text(json.dumps(selected,indent=2)+'\n')


if __name__=='__main__':main()
