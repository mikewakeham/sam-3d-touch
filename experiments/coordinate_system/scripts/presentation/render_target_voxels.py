"""Reproduce target-generation surface occupancy and render its real grid cells."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh
import open3d as o3d
from trimesh.visual.color import uv_to_color
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from render_conditioning import plot


def main():
    p=argparse.ArgumentParser();p.add_argument('object_dir',type=Path);p.add_argument('output_dir',type=Path)
    a=p.parse_args();root=a.object_dir;out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.load(root/'model.obj',force='mesh',process=False)
    fc=(uv_to_color(mesh.visual.uv,mesh.visual.material.image)/255.)[mesh.faces].mean(1)
    mesh.apply_transform(np.load(root/'object_transform.npz')['T_normalized_from_source'])
    workspace=next(p for p in Path(__file__).resolve().parents if (p/'sam-3d-touch-data').is_dir())
    source=workspace/'sam-3d-touch-data/objaverse-dexonomy/generate_target_latents.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='voxelize_mesh')
    # Actual voxelization function; only replace Torch's output container with NumPy.
    namespace=dict(np=np,o3d=o3d,torch=SimpleNamespace(zeros=lambda *shape,dtype:np.zeros(shape,dtype=dtype),float32=np.float32))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    occupied=namespace['voxelize_mesh'](mesh)[0].astype(bool)
    np.savez_compressed(out/'target_occupancy.npz',occupancy=occupied)
    # Draw actual unit voxel faces, not enlarged point markers or filled interiors.
    indices=np.argwhere(occupied);faces=[];facet_centers=[];facet_normals=[]
    for axis in range(3):
        other=[k for k in range(3) if k!=axis]
        for sign in [-1,1]:
            neighbor=indices.copy();neighbor[:,axis]+=sign
            valid=((neighbor>=0)&(neighbor<64)).all(1)
            exposed=np.ones(len(indices),dtype=bool)
            exposed[valid]=~occupied[tuple(neighbor[valid].T)]
            cells=indices[exposed]
            corners=np.zeros((4,3));corners[:,axis]=int(sign>0)
            corners[:,other]=[[0,0],[1,0],[1,1],[0,1]]
            quads=(cells[:,None,:]+corners[None,:,:])/64-.5
            centers=quads.mean(axis=1)
            faces.extend(quads[:,[0,1,2],:]);faces.extend(quads[:,[0,2,3],:])
            facet_centers.extend(centers);facet_centers.extend(centers)
            normal=np.zeros(3);normal[axis]=sign
            facet_normals.extend(np.repeat(normal[None,:],2*len(cells),axis=0))
    triangles=np.asarray(faces)
    vox=trimesh.Trimesh(vertices=triangles.reshape(-1,3),faces=np.arange(triangles.size//3).reshape(-1,3),process=False)
    # Preserve an untextured version as a clear view of the occupancy geometry.
    shade=.85+.15*np.abs(vox.face_normals@np.array([.3,.5,.8]))
    colors=np.ones((len(vox.faces),4));colors[:,:3]=np.array([.18,.51,.76])*shade[:,None]
    # Assign each exposed voxel facet the source texture at the nearest point
    # on the mesh surface. Sampling is deterministic and changes appearance only:
    # the occupied cells and visible facets above remain exactly the same.
    samples, sample_faces=trimesh.sample.sample_surface(mesh,200000,seed=41)
    bary=trimesh.triangles.points_to_barycentric(mesh.triangles[sample_faces],samples)
    uv=np.einsum('ij,ijk->ik',bary,mesh.visual.uv[mesh.faces[sample_faces]])
    sample_rgb=uv_to_color(uv,mesh.visual.material.image)[:,:3]/255.
    distances, nearest=cKDTree(samples).query(np.asarray(facet_centers),k=8,workers=-1)
    weights=1/np.maximum(distances,1e-5)**2
    weights/=weights.sum(axis=1,keepdims=True)
    facet_rgb=(sample_rgb[nearest]*weights[...,None]).sum(axis=1)
    light=np.array([.35,-.45,.8]);light/=np.linalg.norm(light)
    fill=np.array([-.55,.25,.45]);fill/=np.linalg.norm(fill)
    voxel_normals=np.asarray(facet_normals)
    voxel_shade=(.82+.30*np.maximum(voxel_normals@light,0)
                     +.08*np.maximum(voxel_normals@fill,0))
    textured=np.ones((len(vox.faces),4))
    # Both triangles in one square must have exactly the same color.
    textured[:,:3]=np.clip(facet_rgb*(1.18*voxel_shade[:,None]),0,1)
    # Soft directional shading for the textured mesh. The geometry and source
    # texture colors stay unchanged; only display brightness varies by normal.
    normals=mesh.vertex_normals[mesh.faces].mean(axis=1)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    mesh_shade=.82+.30*np.maximum(normals@light,0)+.08*np.maximum(normals@fill,0)
    shaded_fc=fc.copy()
    shaded_fc[:,:3]=np.clip(fc[:,:3]*(1.18*mesh_shade[:,None]),0,1)
    plot(out/'target_mesh.png',[],np.zeros(3),.55,mesh=mesh,face_colors=shaded_fc,object_axes=None,frame='Target mesh — object coordinates')
    plot(out/'target_voxels.png',[],np.zeros(3),.55,mesh=vox,face_colors=colors,object_axes=None,frame='Target encoder input — 64³ surface voxels')
    plot(out/'target_voxels_textured.png',[],np.zeros(3),.55,mesh=vox,face_colors=textured,object_axes=None,frame='Textured target voxels')
    plot(out/'target_mesh_upright.png',[],np.zeros(3),.55,mesh=mesh,face_colors=shaded_fc,object_axes=None,vertical_axis='z',frame='Upright reference — +Z up, face toward −Y')
    plot(out/'target_voxels_upright.png',[],np.zeros(3),.55,mesh=vox,face_colors=colors,object_axes=None,vertical_axis='z',frame='Target voxels — upright viewing reference')
    plot(out/'target_voxels_textured_upright.png',[],np.zeros(3),.55,mesh=vox,face_colors=textured,object_axes=None,vertical_axis='z',frame='Textured target voxels')
    fig,axs=plt.subplots(1,2,figsize=(12,6))
    for ax,name in zip(axs,['target_mesh','target_voxels_textured']):
        ax.imshow(Image.open(out/(name+'.png')));ax.axis('off')
    fig.tight_layout(pad=0);fig.savefig(out/'target_encoding.png',dpi=300);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,6))
    for ax,name in zip(axs,['target_mesh_upright','target_voxels_textured_upright']):
        ax.imshow(Image.open(out/(name+'.png')));ax.axis('off')
    fig.tight_layout(pad=0);fig.savefig(out/'target_encoding_upright.png',dpi=300);plt.close(fig)
    (out/'README.md').write_text('''# Target orientation and voxelization

`target_encoding.png` shows mesh and actual 64³ surface voxelization, using identical object coordinates and the same display viewpoint as the camera-input figures. `target_mesh_upright.png` is a separately labeled Z-up viewing reference; it changes only the viewing camera, not mesh coordinates.

The voxelization reruns the existing target-generation function with Open3D on the mesh transformed by the saved T_normalized_from_source. It is the encoder-input construction, not a decoded latent or predicted output; the saved latent was not re-encoded or compared on this CPU run. Surface voxels are not solid-filled. The textured rendering assigns each complete exposed voxel face a texture color from nearby sampled mesh points, then applies moderate directional facet shading. Color is a visualization approximation, not input to the occupancy encoder. The original untextured images remain available. By presentation convention, the inset green Y arrow points toward the face (target -Y), while the plot grid uses native +Y. The default target views use the same plotting camera as the input plots. Upright reference views change only the viewing camera and are kept separate. No semantic canonicalization is applied.
''')
    (out/'voxelization.json').write_text(json.dumps(dict(occupied_cells=int(occupied.sum()),resolution=64,cell_width=1/64,source=str(source),open3d=o3d.__version__,decoded_latent=False),indent=2)+'\n')
    print('Occupied cells:',occupied.sum())

if __name__=='__main__':main()
