"""CPU Torch: use repository normalization methods without loading model weights."""
import ast
import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from PIL import Image


def method(path, class_name, method_name, namespace):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method_name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[method_name]


def main():
    p=argparse.ArgumentParser();p.add_argument('object_dir',type=Path);p.add_argument('output_dir',type=Path)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    repo=next(p for p in Path(__file__).resolve().parents if (p/'train.py').exists())
    # The supplied masks already match pointmap resolution; the actual method's
    # nearest resize is identity here. Assert that rather than substituting resizing.
    def same_size(mask,size,interpolation):
        assert tuple(mask.shape[-2:])==tuple(size)
        return mask
    tv=SimpleNamespace(transforms=SimpleNamespace(functional=SimpleNamespace(resize=same_size),InterpolationMode=SimpleNamespace(NEAREST=0)))
    ns=dict(torch=torch,torchvision=tv)
    moments=method(repo/'sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py','ObjectCentricSSI','_compute_scale_and_shift',ns)
    vec=method(repo/'sam3d_objects/model/backbone/dit/embedder/touch.py','TouchEncoder','normalize_points_for_vecsetx',dict(torch=torch))
    config=SimpleNamespace(use_scene_scale=True,scale_factor=1.,raise_on_no_valid_points=True)
    report={}
    for view in ['004','005','006']:
        folder=a.object_dir/'views'/view
        pm=torch.from_numpy(np.load(folder/'pointmap.npy')).permute(2,0,1)
        mask=torch.from_numpy(np.asarray(Image.open(folder/'image.png').convert('RGBA')).copy()[...,3]).float()[None]/255
        scale,shift=moments(config,pm,mask)
        surface=torch.from_numpy(np.load(folder/'full_surface.npz')['points_camera'])
        # ScaleShiftInvariant.ssi_to_metric is scale followed by translation;
        # the inverse used in train.py is exactly (p - shift) / scale.
        sam_surface=(surface-shift)/scale
        vec_surface,center,multiplier=vec(None,sam_surface[None],torch.ones((1,len(surface)),dtype=torch.bool))
        direct,_,_=vec(None,surface[None],torch.ones((1,len(surface)),dtype=torch.bool))
        torch.testing.assert_close(vec_surface,direct,atol=2e-6,rtol=2e-6)
        np.savez(a.output_dir/(view+'.npz'),scale=scale.numpy(),shift=shift.numpy(),surface_sam=sam_surface.numpy(),surface_vec=vec_surface[0].numpy())
        report[view]=dict(sam_shift=shift.tolist(),sam_scale=scale.tolist(),vec_center_in_sam_units=center[0].tolist(),vec_radius_in_sam_units=float(1/multiplier[0,0]),direct_vec_max_difference=float((direct-vec_surface).abs().max()))
    (a.output_dir/'normalization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
