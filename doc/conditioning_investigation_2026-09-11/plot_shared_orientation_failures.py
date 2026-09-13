"""Post-result examples; aggregate conclusions use all predictions."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/sam3d-geometry-mpl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pose_shape_geometry import points,subset
from shared_orientation_protocol import transform_points

HERE=Path(__file__).resolve().parent
out=HERE/'shared_orientation_geometry_analysis';root=out/'bundle'
d=json.loads((root/'shared_orientation/results.json').read_text())
frames={r['sample_id']:r for r in d['target_frames']}
rows=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
fig=plt.figure(figsize=(12,8))
for i,(group,draw,label) in enumerate([(2,1,'Fitted drill view 015'),(5,0,'Reserved drill view 000')]):
    sid=d['input_batches'][group]['sample_ids'][2]
    row=next(r for r in rows if r['condition']=='correct_surface' and r['sample_id']==sid and r['draw']==draw)
    p=points(np.load(root/f'shared_orientation/correct_surface_g{group}_d{draw}/predicted_occupancy.npy',allow_pickle=False)[2])
    p=transform_points(p,np.array(frames[sid]['object_from_output']))
    q=points(np.load(root/'reference/original_target_occupancy.npy',allow_pickle=False)[2])
    aligned=p@np.array(row['rigid']['rotation']).T+np.array(row['rigid']['translation'])
    titles=['Decoded target',f'Prediction before pose adjustment\nF-score {row["raw"]["fscore_2v"]:.1%}',
            f'Prediction after rigid adjustment\nF-score {row["rigid"]["metrics"]["fscore_2v"]:.1%}']
    for j,(cloud,title) in enumerate(zip([q,p,aligned],titles)):
        ax=fig.add_subplot(2,3,i*3+j+1,projection='3d');cloud=subset(cloud,5000)
        ax.scatter(*cloud.T,s=1.2,alpha=.45,color='#247f91' if j==0 else '#c45f34',rasterized=True)
        ax.set(xlim=(-.55,.55),ylim=(-.55,.55),zlim=(-.55,.55),xlabel='X',ylabel='Y',zlabel='Z')
        ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-60)
        ax.set_title((label+'\n' if j==0 else '')+title,fontsize=10)
fig.suptitle('Camera-oriented targets: pose correction leaves residual geometry errors',fontsize=14)
fig.text(.5,.025,'Selected diagnostic examples, correct full surfaces. Known label inverse applied first; identical axes/scale.\n'
         'Two-voxel tolerance in original units. Reserved draw 0 cannot attain 95% recall under any rigid motion (bound: 82.71%).',
         ha='center',fontsize=9)
fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(out/'drill_failures.png',dpi=160)
