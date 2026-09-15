"""Export standalone object-orientation markers for slide placement."""
import argparse
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render(path, rotation, vertical_axis='y'):
    fig=plt.figure(figsize=(3,3),facecolor='none')
    ax=fig.add_axes([.08,.08,.84,.84],projection='3d',facecolor='none',zorder=2)
    ax.set_proj_type('ortho')
    ax.view_init(elev=12,azim=-65,vertical_axis=vertical_axis)
    ax.set(xlim=(-1.25,1.25),ylim=(-1.25,1.25),zlim=(-1.25,1.25))
    ax.set_box_aspect((1,1,1),zoom=1.18);ax.set_axis_off()
    basis=np.asarray(rotation)@np.diag([1.,-1.,1.])
    light=np.array([.4,-.5,.8]);light/=np.linalg.norm(light)
    directions=[]
    for i,(label,color) in enumerate(zip('XYZ',[[.8,.12,.1],[.12,.62,.23],[.12,.36,.85]])):
        direction=basis[:,i]/np.linalg.norm(basis[:,i])
        directions.append((label,color,direction))
        shaft=trimesh.creation.cylinder(radius=.018,height=.80,sections=32)
        shaft.apply_translation([0,0,.40])
        tip=trimesh.creation.cone(radius=.062,height=.20,sections=32)
        tip.apply_translation([0,0,.80])
        arrow=trimesh.util.concatenate([shaft,tip])
        arrow.apply_transform(trimesh.geometry.align_vectors([0,0,1],direction))
        shade=.86+.14*np.maximum(arrow.face_normals@light,0)
        rgba=np.ones((len(arrow.faces),4));rgba[:,:3]=np.array(color)*shade[:,None]
        ax.add_collection3d(Poly3DCollection(arrow.triangles,facecolors=rgba,edgecolors='none'))
    fig.canvas.draw()
    projection=ax.get_proj()
    origin=np.array(proj3d.proj_transform(0,0,0,projection)[:2])
    pixels=[ax.transData.transform(origin)]
    labels=[]
    for i,(label,color,direction) in enumerate(directions):
        tip=np.array(proj3d.proj_transform(*direction,projection)[:2])
        pixels.append(ax.transData.transform(tip))
        delta=ax.transData.transform(tip)-ax.transData.transform(origin)
        if np.linalg.norm(delta)<8:
            delta=np.array([[1.,1.],[-1.,-1.],[1.,-1.]][i])
        pixel=ax.transData.transform(tip)+delta/np.linalg.norm(delta)*16
        x,y=fig.transFigure.inverted().transform(pixel)
        labels.append(fig.text(x,y,label,color=color,fontsize=18,ha='center',va='center',zorder=3))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    pixels=np.asarray(pixels)
    x0,y0=pixels.min(axis=0);x1,y1=pixels.max(axis=0)
    for label in labels:
        box=label.get_window_extent(renderer)
        x0=min(x0,box.x0);y0=min(y0,box.y0)
        x1=max(x1,box.x1);y1=max(y1,box.y1)
    (x0,y0),(x1,y1)=fig.transFigure.inverted().transform([(x0-14,y0-14),(x1+14,y1+14)])
    # Keep the card square and center the entire rendered triad, including labels.
    side=max(x1-x0,y1-y0)
    cx,cy=(x0+x1)/2,(y0+y1)/2
    x0,x1=cx-side/2,cx+side/2
    y0,y1=cy-side/2,cy+side/2
    fig.add_artist(FancyBboxPatch((x0,y0),x1-x0,y1-y0,
                   boxstyle='round,pad=.004,rounding_size=.025',
                   transform=fig.transFigure,facecolor='#f9fbfd',
                   edgecolor='#dfe6ee',linewidth=.8,zorder=1))
    fig.savefig(path,dpi=200,transparent=True)
    plt.close(fig)
    image=Image.open(path)
    pad=.02
    image.crop((int((x0-pad)*image.width),int((1-y1-pad)*image.height),
                int((x1+pad)*image.width),int((1-y0+pad)*image.height))).save(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('object_dir',type=Path);p.add_argument('output_dir',type=Path)
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    render(out/'target.png',np.eye(3))
    # Cancel render()'s presentation-only Y flip to show the target's native +Y.
    render(out/'target_plus_y.png',np.diag([1.,-1.,1.]))
    # Same native target axes, viewed with object +Z as the screen-up direction.
    render(out/'target_plus_y_z_up.png',np.diag([1.,-1.,1.]),vertical_axis='z')
    for view in ['004','005','006']:
        cv=np.load(args.object_dir/'views'/view/'camera.npz')['T_camera_from_object']
        sam=np.diag([-1.,-1.,1.,1.])@cv
        render(out/(view+'.png'),sam[:3,:3])
        # Cancel render()'s presentation-only Y flip to show native target +Y.
        render(out/(view+'_plus_y.png'),sam[:3,:3]@np.diag([1.,-1.,1.]))
    (out/'README.md').write_text('''Standalone boxed object-orientation markers. target.png uses the presentation convention where green Y points toward the octopus face (target -Y). target_plus_y.png shows the target frame's native +Y with the old Y-up plotting view. target_plus_y_z_up.png shows the same native axes with object +Z displayed upward and matches the upright target mesh/voxel figures. Red X and blue Z are target +X and +Z. 004.png, 005.png, and 006.png use the old -Y presentation convention. 004_plus_y.png, 005_plus_y.png, and 006_plus_y.png show the native target basis transformed into each saved SAM camera frame and match the camera-coordinate plots. All PNGs use square boxes. Drag each onto slides independently.\n''')
    print(out.resolve())

if __name__=='__main__':main()
