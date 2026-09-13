"""Supplement GT controls with known rigid transforms of aligned predictions."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
from pose_shape_geometry import points,register
from alignment_tolerance_protocol import rotation

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'pose_shape_bundle';output=HERE/'pose_shape_analysis/aligned_prediction_controls.json'
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((root/'bundle_manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['kind']=='alignment' and c['policy']=='dropout'
              and c['condition']=='aligned' and c['group']==4 and c['draw']==0)
    occupancy=np.load(root/case['prefix']/'predicted_occupancy.npy',allow_pickle=False)
    r=np.array(rotation('z',23))@np.array(rotation('y',-17))@np.array(rotation('x',11))
    rows=[]
    for i,sid in enumerate(case['sample_ids']):
        print('Checking',sid,flush=True)
        p=points(occupancy[i]);moved=p@r.T+np.array([.03,-.02,.01]);result=register(moved,p)
        assert result['metrics']['precision_1v']==result['metrics']['recall_1v']==1
        rows.append({'sample_id':sid,'rigid':result})
    output.write_text(json.dumps({'selection':'All four objects at first reserved group 4, draw 0, aligned dropout; '
        'known mixed rotation and translation applied to each prediction and registered to itself.',
        'rows':rows},indent=2)+'\n')
    print('All four aligned-prediction controls passed.',flush=True)


if __name__=='__main__':main()
