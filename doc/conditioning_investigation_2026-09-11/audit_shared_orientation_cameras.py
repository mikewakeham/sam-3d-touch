"""CPU-only audit of the exact 28 camera matrices; no training or tolerance change."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    single=json.loads((HERE/'tiny_fit_returned_46083371/camera/results.json').read_text())
    drop=json.loads((HERE/'visual_dropout_returned_manual/camera/results.json').read_text())
    config=Path(single['settings']['data_config'])
    if not config.is_absolute():config=REPO/config
    data=yaml.safe_load(config.read_text());root=Path(data['dataset']['root'])
    manifest=Path(data['dataset']['manifest'])
    if not manifest.is_absolute():manifest=root/manifest
    selected={sid for b in drop['input_batches'] for sid in b['sample_ids']}
    records={r['sample_id']:r for r in map(json.loads,manifest.read_text().splitlines()) if r['sample_id'] in selected}
    assert set(records)==selected
    rows=[]
    for group,batch in enumerate(drop['input_batches']):
        for sid in batch['sample_ids']:
            path=Path(records[sid]['camera_path'])
            if not path.is_absolute():path=root/path
            with np.load(path,allow_pickle=False) as archive:
                saved=archive['T_camera_from_object']
                transform=np.diag([-1.,-1.,1.,1.])@saved
                dtype=str(saved.dtype)
            assert transform.shape==(4,4) and np.isfinite(transform).all(),sid
            rotation=transform[:3,:3];gram=rotation.T@rotation;det=float(np.linalg.det(rotation))
            checks=dict(orthogonality=bool(np.allclose(gram,np.eye(3),atol=1e-6)),
                determinant=bool(np.isclose(det,1.,atol=1e-6)),
                homogeneous_row=bool(np.allclose(transform[3],[0,0,0,1])))
            rows.append(dict(sample_id=sid,group=group,path=str(path),source_dtype=dtype,
                camera_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                passes=all(checks.values()),checks=checks,
                orthogonality_max_error=float(np.max(np.abs(gram-np.eye(3)))),
                determinant=det,singular_values=np.linalg.svd(rotation,compute_uv=False).tolist(),
                homogeneous_row_max_error=float(np.max(np.abs(transform[3]-[0,0,0,1]))),
                transform=transform.tolist()))
    result=dict(samples=len(rows),failures=[r for r in rows if not r['passes']],rows=rows,
        numpy_version=np.__version__,scope='Exact current validation checks, CPU only. No matrix repair or tolerance modification.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    print('Saved',args.output)


if __name__=='__main__':main()
