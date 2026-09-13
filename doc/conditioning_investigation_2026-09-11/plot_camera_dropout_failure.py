"""Post-result diagnostic illustration of the bathtub-to-shield failure."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/sam3d-geometry-mpl')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pose_shape_geometry import points

root=Path(__file__).resolve().parent/'camera_dropout_geometry_analysis'
base=root/'bundle/camera_dropout'
target=np.load(base/'correct_surface_g5_d0/target_occupancy.npy',allow_pickle=False)
pred=np.load(base/'correct_surface_g5_d0/predicted_occupancy.npy',allow_pickle=False)
wrong=np.load(base/'wrong_surface_g5_d0/predicted_occupancy.npy',allow_pickle=False)
clouds=[points(target[1]),points(pred[1]),points(target[0]),points(wrong[1])]
titles=['Bathtub target','Correct bathtub surface → output','Shield target','Wrong shield surface → output']
fig,axes=plt.subplots(2,4,figsize=(13,6.5))
for j,(p,title) in enumerate(zip(clouds,titles)):
    for i,(a,b) in enumerate([(0,1),(0,2)]):
        ax=axes[i,j]
        ax.scatter(p[:,a],p[:,b],s=.7,c='#237c88' if j in (0,2) else '#c15e34',alpha=.3,rasterized=True)
        ax.set(xlim=(-.55,.55),ylim=(-.55,.55),aspect='equal',xlabel='X')
        ax.set_ylabel('Y' if i==0 else 'Z');ax.grid(alpha=.15)
        if i==0:ax.set_title(title,fontsize=10)
fig.suptitle('Camera + dropout: a reserved bathtub view produces the fitted shield\n'
             'Same image and sampling noise; only the supplied surface changes',fontsize=12)
fig.text(.5,.015,'Diagnostic example selected after results: bathtub view 000, draw 0. '
         'Raw coordinates, no output alignment. Full analysis retains all 24 reserved correct predictions.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,.92))
fig.savefig(root/'bathtub_shield_failure.png',dpi=160)
