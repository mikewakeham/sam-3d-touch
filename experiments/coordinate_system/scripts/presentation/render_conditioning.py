"""Individual plots of saved raw camera-frame conditioning; no model inference."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trimesh.visual.color import uv_to_color
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyBboxPatch


def affine(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def plot(path, layers, center, limit, mesh=None, face_colors=None, frame='SAM camera frame', grid_spacing=.5, object_axes=None, vertical_axis='z', coordinate_order=(0,1,2), marker_size=.9):
    fig = plt.figure(figsize=(6,6), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    ax.view_init(elev=12, azim=-65, vertical_axis=vertical_axis)
    center=np.asarray(center)[list(coordinate_order)]
    for points, colors, size in layers:
        points=np.asarray(points)[:,list(coordinate_order)]
        # Display-only subset; keep point/color correspondence.
        indices = np.random.default_rng(42).choice(len(points), min(5000,len(points)), replace=False)
        points = points[indices]
        if not isinstance(colors,str):
            colors = np.asarray(colors)[indices]
        ax.scatter(*points.T, c=colors, s=marker_size, linewidths=0, depthshade=False, alpha=1.)
    if mesh is not None:
        ax.add_collection3d(Poly3DCollection(mesh.triangles[:,:,list(coordinate_order)],facecolors=face_colors,
                                           edgecolors='none',linewidths=0))
    for dim, (setter, axis, label, color) in enumerate(zip(
            [ax.set_xlim,ax.set_ylim,ax.set_zlim], [ax.xaxis,ax.yaxis,ax.zaxis],
            'XYZ', ['#bc3434','#26804b','#3265b0'])):
        setter(center[dim]-limit,center[dim]+limit)
        axis.set_major_locator(MultipleLocator(grid_spacing))
        axis.set_major_formatter(FormatStrFormatter('%.1f'))
        axis.set_pane_color((.95,.96,.98,1.))
        axis._axinfo['grid'].update(color=(.72,.76,.81,1.),linewidth=.7)
        axis.label.set_color(color)
    ax.set_xlabel('X',labelpad=1,fontsize=12)
    ax.set_ylabel('Y',labelpad=1,fontsize=12)
    ax.set_zlabel('Z',labelpad=1,fontsize=12)
    ax.tick_params(labelsize=8,pad=-1)
    ax.set_box_aspect((1,1,1));ax.grid(True)
    fig.subplots_adjust(left=.04,right=.90,bottom=.10,top=.91)
    if object_axes is not None:
        # Presentation convention: green Y points toward the octopus's face
        # (target -Y), while the plot grid retains native positive Y.
        basis=np.asarray(object_axes)[list(coordinate_order),:]@np.diag([1.,-1.,1.])
        inset=fig.add_axes([.14,.56,.24,.23],projection='3d',facecolor='none',zorder=5)
        inset.set_proj_type('ortho')
        inset.view_init(elev=12,azim=-65,vertical_axis=vertical_axis)
        inset.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),zlim=(-1.2,1.2))
        inset.set_box_aspect((1,1,1),zoom=1.15);inset.set_axis_off()
        light=np.array([.4,-.5,.8]);light/=np.linalg.norm(light)
        directions=[]
        for i,(label,color) in enumerate(zip('XYZ',[[.8,.12,.1],[.12,.62,.23],[.12,.36,.85]])):
            direction=basis[:,i];direction=direction/np.linalg.norm(direction)
            directions.append((label,color,direction))
            shaft=trimesh.creation.cylinder(radius=.018,height=.80,sections=32)
            shaft.apply_translation([0,0,.40])
            tip=trimesh.creation.cone(radius=.062,height=.20,sections=32)
            tip.apply_translation([0,0,.80])
            arrow=trimesh.util.concatenate([shaft,tip])
            arrow.apply_transform(trimesh.geometry.align_vectors([0,0,1],direction))
            shade=.86+.14*np.maximum(arrow.face_normals@light,0)
            rgba=np.ones((len(arrow.faces),4));rgba[:,:3]=np.array(color)*shade[:,None]
            inset.add_collection3d(Poly3DCollection(arrow.triangles,facecolors=rgba,edgecolors='none'))
        # Matplotlib's 3D text rotates/occludes badly for arrows aimed at the
        # camera. Project each cone tip, then place its label in figure 2D.
        fig.canvas.draw()
        projection=inset.get_proj()
        origin=np.array(proj3d.proj_transform(0,0,0,projection)[:2])
        arrow_pixels=[inset.transData.transform(origin)]
        labels=[]
        for i,(label,color,direction) in enumerate(directions):
            tip=np.array(proj3d.proj_transform(*direction,projection)[:2])
            arrow_pixels.append(inset.transData.transform(tip))
            delta=inset.transData.transform(tip)-inset.transData.transform(origin)
            if np.linalg.norm(delta)<8:  # nearly pointing into/out of screen
                delta=np.array([[1.,1.],[-1.,-1.],[1.,-1.]][i])
            delta=delta/np.linalg.norm(delta)*12
            pixel=inset.transData.transform(tip)+delta
            x,y=fig.transFigure.inverted().transform(pixel)
            labels.append(fig.text(x,y,label,color=color,fontsize=10,ha='center',va='center',zorder=6))
        # Fit a subtle card to the rendered arrows AND labels. The inset size
        # stays unchanged; only its whole position moves over the plot.
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        pixels=np.asarray(arrow_pixels)
        x0,y0=pixels.min(axis=0);x1,y1=pixels.max(axis=0)
        for label in labels:
            bounds=label.get_window_extent(renderer)
            x0=min(x0,bounds.x0);y0=min(y0,bounds.y0)
            x1=max(x1,bounds.x1);y1=max(y1,bounds.y1)
        (x0,y0),(x1,y1)=fig.transFigure.inverted().transform([(x0-9,y0-9),(x1+9,y1+9)])
        fig.add_artist(FancyBboxPatch((x0,y0),x1-x0,y1-y0,
                       boxstyle='round,pad=.004,rounding_size=.009',
                       transform=fig.transFigure,facecolor='#f9fbfd',
                       edgecolor='#dfe6ee',linewidth=.6,zorder=4))
    fig.savefig(path,dpi=200,facecolor='white',bbox_inches='tight',pad_inches=.15);plt.close(fig)
    image=Image.open(path).convert('RGB')
    pixels=np.asarray(image)
    ink=np.any(pixels<248,axis=2)
    rows,cols=np.where(ink)
    margin=18
    image.crop((max(0,int(cols.min())-margin),max(0,int(rows.min())-margin),
                min(image.width,int(cols.max())+margin+1),
                min(image.height,int(rows.max())+margin+1))).save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('object_dir',type=Path);p.add_argument('output_dir',type=Path)
    args=p.parse_args();root=args.object_dir;out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.load(root/'model.obj',force='mesh',process=False)
    fc=(uv_to_color(mesh.visual.uv,mesh.visual.material.image)/255.)[mesh.faces].mean(1)
    mesh.apply_transform(np.load(root/'object_transform.npz')['T_normalized_from_source'])
    records=[]
    for view in ['004','005','006']:
        src=root/'views'/view;dest=out/view;dest.mkdir(exist_ok=True)
        surface=np.load(src/'full_surface.npz');xyz=surface['points_camera']
        obj,faces=trimesh.sample.sample_surface(mesh,len(xyz),seed=int(surface['sample_seed']))
        sam=np.diag([-1.,-1.,1.,1.])@np.load(src/'camera.npz')['T_camera_from_object']
        replay_error=float(np.abs(affine(obj,sam)-xyz).max())
        assert replay_error<2e-5
        rgba=np.asarray(Image.open(src/'image.png').convert('RGBA'))
        pm=np.load(src/'pointmap.npy');valid=np.isfinite(pm).all(-1)&(rgba[...,3]>0)&(pm[...,2]>0)
        indices=np.flatnonzero(valid)
        # Deterministic thinning solely for readable plots; no coordinate changes.
        indices=indices[np.linspace(0,len(indices)-1,min(5000,len(indices)),dtype=int)]
        q=pm.reshape(-1,3)[indices];rgb=rgba.reshape(-1,4)[indices,:3]/255.
        center=np.array([0.,0.,2.]);limit=.55
        assert np.all(np.abs(xyz-center)<=limit) and np.all(np.abs(q-center)<=limit)
        shutil.copyfile(src/'image.png',dest/'rgb.png')
        plot(dest/'surface_colored.png',[(xyz,fc[faces],.65)],center,limit,object_axes=None,coordinate_order=(2,0,1))
        plot(dest/'surface_uncolored.png',[(xyz,'#657385',.65)],center,limit,object_axes=None,coordinate_order=(2,0,1))
        plot(dest/'pointmap_colored.png',[(q,rgb,.35)],center,limit,object_axes=None,coordinate_order=(2,0,1))
        plot(dest/'pointmap_uncolored.png',[(q,'#657385',.35)],center,limit,object_axes=None,coordinate_order=(2,0,1))
        plot(dest/'surface_and_pointmap.png',[(xyz,'#337bc4',.65),(q,'#f18b32',.45)],center,limit,object_axes=None,coordinate_order=(2,0,1))
        # One actual sampled surface point expressed in two coordinate systems.
        records.append(dict(view=view,raw_surface_frame='sam_camera',pointmap_frame='sam_camera',display_coordinate_order=[2,0,1],
            replay_max_error=replay_error,object_point=obj[0].tolist(),camera_point=xyz[0].tolist(),
            plotted_surface_count=min(5000,len(xyz)),plotted_pointmap_count=len(q),valid_pointmap_count=int(valid.sum()),
            display_center=center.tolist(),display_limit=limit))
    plot(out/'target_object_mesh.png',[],np.zeros(3),.55,mesh=mesh,face_colors=fc,frame='Object frame: mesh used for target encoding',object_axes=None)
    plot(out/'target_object_surface.png',[(obj,fc[faces],.65)],np.zeros(3),.55,frame='Object frame: full surface',object_axes=None)
    target=np.load(root/'target_latent.npz')['mean']
    fig,ax=plt.subplots(figsize=(3,3))
    ax.imshow(target[0,:,:,8].T,origin='lower',cmap='PuOr',interpolation='nearest')
    ax.axis('off');fig.tight_layout(pad=0)
    fig.savefig(out/'target_latent_slice.png',dpi=180);plt.close(fig)
    (out/'coordinates.json').write_text(json.dumps(records,indent=2)+'\n')
    (out/'README.md').write_text('''# Individual conditioning figures

Views 004, 005, 006 are above the object equator. Each view has RGB, colored/uncolored full surface, colored/uncolored pointmap, and an overlay (blue = full surface; orange = pointmap).

All geometry is plotted directly in the saved SAM camera frame, BEFORE SAM preprocessing or VecSetX normalization. Across ALL views, camera plots share X/Y limits [-0.55, 0.55], Z limits [1.45, 2.55], the same Z-up display viewpoint, equal XYZ scale, and 0.5-unit grid spacing. Target plots use the same viewpoint, axis span and grid spacing, with all axes [-0.55, 0.55] because their origin is different. For plotting only, camera (Z,X,Y) is displayed as (X,Y,Z). This preserves the former upright view with target-style grid labels; model input arrays are unchanged. Units are dataset coordinate units, not a claim of meters. The overlay demonstrates the raw inputs share a frame; it does not show their separately normalized encoder inputs.

Surface colors come from source material face colors, used only for visualization. Pointmap colors come from corresponding RGB pixels. Each plotted cloud contains at most 5000 deterministically selected points, with fully opaque markers. Full surface data still contains 8192 points; subsampling is for display only. No geometry is synthesized.

Target encoding assets: target_object_mesh.png is the mesh supplied for target generation, NOT a decoded target; target_latent_slice.png is one scalar feature slice of the saved target, NOT an occupancy map. Present these as mesh → fixed-grid voxelization → frozen Stage-1 encoder → stored latent. No decoded target for this object is available here.
''')
    # Contact sheet only to inspect the individual assets; labels distinguish inputs.
    fig,axs=plt.subplots(3,4,figsize=(20,15))
    names=['rgb','surface_colored','pointmap_colored','surface_and_pointmap']
    labels=['RGB','Full surface: color','Pointmap: color','Surface blue / pointmap orange']
    for row,view in enumerate(['004','005','006']):
        for col,name in enumerate(names):
            axs[row,col].imshow(Image.open(out/view/(name+'.png')));axs[row,col].axis('off')
    fig.tight_layout();fig.savefig(out/'contact_sheet.png',dpi=300);plt.close(fig)
    print(out.resolve())


if __name__=='__main__':main()
