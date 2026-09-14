"""Exact grid-preserving proper rotations and a balanced coverage intervention."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import itertools
import random
import numpy as np
from experiments.coordinate_system.scripts.shared.visual_dropout_protocol import dropout_schedule


def rotations():
    result=[]
    for cols in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            r=np.zeros((3,3),dtype=np.int64);r[np.arange(3),cols]=signs
            if round(np.linalg.det(r))==1:result.append(r)
    identity=np.eye(3,dtype=np.int64)
    return [identity]+[r for r in result if not np.array_equal(r,identity)]


def indices(rotation):
    r=np.asarray(rotation)
    if r.shape!=(3,3) or not np.isin(r,[-1,0,1]).all() or not np.array_equal(r@r.T,np.eye(3)) or round(np.linalg.det(r))!=1:
        raise ValueError('Expected a proper signed-permutation rotation')
    cols=np.argmax(np.abs(r),axis=1)
    return cols,r[np.arange(3),cols]


def rotate_points(p,rotation):
    cols,signs=indices(rotation)
    return np.asarray(p)[...,cols]*signs


def rotate_grid(grid,rotation):
    a=np.asarray(grid)
    if a.ndim!=3 or len(set(a.shape))!=1:raise ValueError('Expected a cubic 3D array')
    cols,signs=indices(rotation);out=np.transpose(a,cols)
    for axis,sign in enumerate(signs):
        if sign<0:out=np.flip(out,axis=axis)
    return np.ascontiguousarray(out)


def inverse_frame(object_from_output,rotation):
    indices(rotation)
    h=np.eye(4);h[:3,:3]=np.asarray(rotation).T
    return np.asarray(object_from_output)@h


def schedule():
    # Each group has 125 visual-present +125 dropped updates. Among dropped
    # updates, each of the23 nonidentity rotations occurs5 or6 times per group.
    flags=dropout_schedule(31);banks=[];offset=[0]*4
    for g in range(4):
        rng=random.Random(831000+g);bank=[]
        while len(bank)<125:
            chunk=list(range(1,24));rng.shuffle(chunk);bank.extend(chunk)
        banks.append(bank[:125])
    out=[]
    for i,drop in enumerate(flags):
        g=i%4;ri=banks[g][offset[g]] if drop else 0
        if drop:offset[g]+=1
        out.append(dict(step=2001+i,group=g,visual_dropped=drop,rotation_index=ri,noise_seed=29+2001+i))
    return out
