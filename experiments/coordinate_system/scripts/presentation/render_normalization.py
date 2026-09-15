"""Same display points, XYZ axes and numeric bounds across normalization stages."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.color import uv_to_color
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_conditioning import plot


def main():
    p=argparse.ArgumentParser();p.add_argument('object_dir',type=Path);p.add_argument('output_dir',type=Path)
    a=p.parse_args();root=a.object_dir;out=a.output_dir
    mesh=trimesh.load(root/'model.obj',force='mesh',process=False)
    fc=(uv_to_color(mesh.visual.uv,mesh.visual.material.image)/255.)[mesh.faces].mean(1)
    mesh.apply_transform(np.load(root/'object_transform.npz')['T_normalized_from_source'])
    views=['004','005','006'];data={};bounds=[]
    for view in views:
        src=root/'views'/view
        rgba=np.asarray(Image.open(src/'image.png').convert('RGBA'))
        pm=np.load(src/'pointmap.npy');valid=np.isfinite(pm).all(-1)&(rgba[...,3]>0)&(pm[...,2]>0)
        ids=np.flatnonzero(valid);ids=ids[np.linspace(0,len(ids)-1,min(5000,len(ids)),dtype=int)]
        q=pm.reshape(-1,3)[ids];colors=rgba.reshape(-1,4)[ids,:3]/255.
        surface=np.load(src/'full_surface.npz');s=surface['points_camera']
        _,faces=trimesh.sample.sample_surface(mesh,len(s),seed=int(surface['sample_seed']))
        norm=np.load(out/(view+'.npz'))
        pm_sam=(q-norm['shift'])/norm['scale']
        stages={'raw':(s,q),'sam3d':(norm['surface_sam'],pm_sam),'vecsetx':(norm['surface_vec'],pm_sam)}
        data[view]=(stages,fc[faces],colors)
        for stage in ('sam3d','vecsetx'):bounds.extend(stages[stage])
    lower=np.min([x.min(0) for x in bounds],axis=0)
    upper=np.max([x.max(0) for x in bounds],axis=0)
    span=np.max(upper-lower);center=(upper+lower)/2;limit=float(span/2+.005)
    names=['rgb','surface_colored','pointmap_colored','surface_and_pointmap']
    titles={'raw':'Raw saved camera coordinates','sam3d':'After SAM3D normalization: both clouds',
            'vecsetx':'After VecSetX: surface only; pointmap stays SAM3D-normalized'}
    labels=['RGB reference','Full surface: color','Pointmap: color','Surface blue / pointmap orange']
    for stage,title in titles.items():
        stage_center = np.array([0.,0.,2.]) if stage == 'raw' else center
        stage_limit = .55 if stage == 'raw' else limit
        tick_spacing = .5 if stage == 'raw' else 2.
        for view in views:
            dest=out/stage/view;dest.mkdir(parents=True,exist_ok=True)
            rotation=(np.diag([-1.,-1.,1.,1.])@np.load(root/'views'/view/'camera.npz')['T_camera_from_object'])[:3,:3]
            (s,q),sc,qc=data[view][0][stage],data[view][1],data[view][2]
            shutil.copyfile(root/'views'/view/'image.png',dest/'rgb.png')
            plot(dest/'surface_colored.png',[(s,sc,.9)],stage_center,stage_limit,grid_spacing=tick_spacing,object_axes=None,frame='Camera XYZ directions — '+stage)
            plot(dest/'pointmap_colored.png',[(q,qc,.9)],stage_center,stage_limit,grid_spacing=tick_spacing,object_axes=None,frame='Camera XYZ directions — '+('sam3d' if stage=='vecsetx' else stage))
            plot(dest/'surface_and_pointmap.png',[(s,'#337bc4',.9),(q,'#f18b32',.9)],stage_center,stage_limit,grid_spacing=tick_spacing,object_axes=None,frame='Camera XYZ directions — '+stage)
        fig,axs=plt.subplots(3,4,figsize=(20,15))
        for row,view in enumerate(views):
            for col,name in enumerate(names):
                axs[row,col].imshow(Image.open(out/stage/view/(name+'.png')));axs[row,col].axis('off')
        fig.tight_layout(rect=(0,0,1,.955));fig.savefig(out/stage/'contact_sheet.png',dpi=300);plt.close(fig)
    # Stage comparison makes the fixed scale/translation immediately visible.
    fig,axs=plt.subplots(3,3,figsize=(18,18))
    for row,view in enumerate(views):
        for col,stage in enumerate(titles):
            axs[row,col].imshow(Image.open(out/stage/view/'surface_and_pointmap.png'));axs[row,col].axis('off')
    fig.tight_layout(rect=(0,0,1,.955));fig.savefig(out/'normalization_comparison.png',dpi=300);plt.close(fig)
    (out/'display.json').write_text(json.dumps(dict(raw_center=[0,0,2],raw_limit=.55,normalized_tick_spacing=2,center=center.tolist(),limit=limit,axis_limits=np.stack([center-limit,center+limit],axis=1).tolist(),points_per_cloud=5000),indent=2)+'\n')
    (out/'README.md').write_text('''# Original non-oracle normalization path

- `raw/contact_sheet.png`: saved camera XYZ.
- `sam3d/contact_sheet.png`: pointmap and full surface both transformed by (p - SAM shift) / SAM scale.
- `vecsetx/contact_sheet.png`: surface additionally bbox-centered and divided by maximum Euclidean radius; pointmap remains SAM-normalized.
- `normalization_comparison.png`: the overlay for all three views at all three stages.

All plots share the same viewpoint, XYZ directions, opaque markers, and sampled point identities. Raw inputs use a closer view (X/Y [-0.55,0.55], Z [1.45,2.55]). The two normalized stages share identical numeric bounds and scale. XYZ scale is equal within every plot. Tick spacing is 0.5 for raw and 2 for normalized plots. XYZ values remain in each stage's units; fixed plotting scale exposes rather than removes normalization. These are coordinate-transform visualizations, NOT final cropped/resized encoder input grids. RGB is the original reference image at every stage. No oracle rotation is applied.

The normalization helper executes the actual repository ObjectCentricSSI moment method and TouchEncoder normalization method on CPU, using use_scene_scale=True, scale_factor=1, as configured in the locally available sam-3d-objects/checkpoints/hf/pipeline.yaml. Same-size mask resizing is identity. The SSI inverse affine is evaluated explicitly as (p-shift)/scale. Normalization uses complete clouds, before display subsampling. Saved normalization.json records every shift and scale. Colors are for display only.
''')
    print('Common XYZ limits:',np.stack([center-limit,center+limit],axis=1).tolist())

if __name__=='__main__':main()
