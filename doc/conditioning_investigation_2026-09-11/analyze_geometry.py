"""CPU geometric evidence; never changes model predictions or training data."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/sam3d-geometry-mpl')
import hashlib,itertools,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
HERE=Path(__file__).resolve().parent
B=HERE/'geometry_bundle'
OUT=HERE/'geometry_analysis'
OUT.mkdir(exist_ok=True)
meta=json.loads((B/'bundle_manifest.json').read_text())
results=json.loads((B/'results.json').read_text())

def cloud(o):return (np.argwhere(o)+.5)/64-.5

def cd(p,q):
 a=cKDTree(q).query(p)[0];b=cKDTree(p).query(q)[0]
 return float((a.mean()+b.mean())/2)

def desc(p):return {'count':len(p),'mean':p.mean(0).tolist(),'min':p.min(0).tolist(),'max':p.max(0).tolist()}

def transforms():
 for perm in itertools.permutations(range(3)):
  for signs in itertools.product([-1,1],repeat=3):
   r=np.eye(3)[list(perm)]*np.array(signs)[:,None]
   yield f'perm{perm}_sign{signs}',r
axes=list(transforms())
rows=[];targets=[]
for sid,rec in meta['records'].items():
 with np.load(B/'data'/rec['camera_path']) as c:T=np.diag([-1.,-1.,1.,1.])@c['T_camera_from_object'];K=c['K']
 with np.load(B/'data'/rec['full_surface_path']) as f:p_cam=f['points_camera'].astype(float)
 surface=(p_cam-T[:3,3])@np.linalg.inv(T[:3,:3]).T
 with np.load(B/'data'/rec['target_path']) as f:z=f['mean'].transpose(1,2,3,0).reshape(4096,8)
 case_rows=[];supports=[];target=None
 for run in results['runs']:
  for row in run['rows']:
   if row['sample_id']!=sid or row['strength']!=7:continue
   with np.load(B/'rollouts'/row['artifact']) as f:
    pred=f['prediction'];targ=f['target'];o=f['predicted_occupancy'];to=f['target_occupancy']
   assert np.array_equal(z,targ)
   if target is not None:assert np.array_equal(target,to)
   target=to;p=cloud(o);q=cloud(to)
   raw=cd(p,q);iou=float(np.count_nonzero(o&to)/np.count_nonzero(o|to));mse=float(np.mean((pred-targ)**2))
   assert np.isclose(raw,row['unaligned_voxel_center_chamfer'],atol=1e-12)
   assert np.isclose(iou,row['voxel_iou'],atol=1e-12)
   assert np.isclose(mse,row['latent_mse'],rtol=1e-5)
   scores=[(cd(p@r.T,q),name,int(round(np.linalg.det(r))),r.tolist()) for name,r in axes]
   scores.sort()
   proper=next(x for x in scores if x[2]==1)
   center=lambda x:(x.max(0)+x.min(0))/2
   candidates={'raw':raw,'centroid_translation':cd(p-p.mean(0)+q.mean(0),q),'bbox_translation':cd(p-center(p)+center(q),q),
    'camera_rotation':cd(p@T[:3,:3].T,q),'inverse_camera_rotation':cd(p@T[:3,:3],q),
    'inverse_full_camera_transform':cd((p-T[:3,3])@np.linalg.inv(T[:3,:3]).T,q)}
   rr={'sample_id':sid,'checkpoint':run['checkpoint'],'seed':row['seed'],'artifact':row['artifact'],'raw_cd':raw,'iou':iou,'latent_mse':mse,
    'prediction':desc(p),'target':desc(q),'transform_cd':candidates,'best_signed_axis':scores[0],'best_proper_axis':proper}
   rows.append(rr);case_rows.append(rr);supports.append(p)
 q=cloud(target)
 scores=sorted((cd(surface@r.T,q),name,int(round(np.linalg.det(r)))) for name,r in axes)
 targets.append({'sample_id':sid,'surface_to_target_cd':cd(surface,q),'surface':desc(surface),'target':desc(q),'best_axis_surface_to_target':scores[0]})
 fig,axs=plt.subplots(3,6,figsize=(18,9))
 img=Image.open(B/'data'/rec['image_path'])
 axs[0,0].imshow(img);axs[0,0].set_title(sid[:8]+' input');axs[0,0].axis('off')
 for ax in axs[1:,0]:ax.axis('off')
 clouds=[surface]+supports
 labels=['Surface vs target']+[('Image' if 'stage1_image_' in r['checkpoint'] else 'Surface')+f" seed {r['seed']}\nCD {r['raw_cd']:.4f}" for r in case_rows]
 for col,(p,label) in enumerate(zip(clouds,labels),1):
  for row,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
   ax=axs[row,col];ax.scatter(q[:,a],q[:,b],s=.5,c='black',alpha=.22,rasterized=True);ax.scatter(p[:,a],p[:,b],s=.5,c='#e85924',alpha=.3,rasterized=True)
   ax.set(xlim=(-.55,.55),ylim=(-.55,.55),aspect='equal',xlabel='XYZ'[a],ylabel='XYZ'[b]);ax.grid(alpha=.2)
   if row==0:ax.set_title(label)
 fig.suptitle('Black = decoded target; orange = observation/prediction. Original fixed frame.');fig.tight_layout();fig.savefig(OUT/(sid+'.png'),dpi=130);plt.close(fig)
 print(sid,'target surface CD',targets[-1]['surface_to_target_cd'],flush=True)
 for r in case_rows:print(r['checkpoint'].split('/')[1],r['seed'],'raw',round(r['raw_cd'],5),'best axis',r['best_signed_axis'][:3],'translations',r['transform_cd']['centroid_translation'],r['transform_cd']['bbox_translation'],flush=True)
(OUT/'metrics.json').write_text(json.dumps({'targets':targets,'predictions':rows,'limits':'Post-selected diagnostic subset. Best axes use GT and are not deployment results. Mesh not supplied; no new VAE execution.'},indent=2)+'\n')
