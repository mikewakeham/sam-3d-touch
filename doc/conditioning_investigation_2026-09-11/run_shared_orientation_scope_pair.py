"""Run the two matched arms on up to two allocated GPUs, sequentially on one."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def allocation_devices(visible, count):
    if count < 1:
        raise ValueError('No visible CUDA GPU. Run inside the interactive GPU allocation.')
    if visible is None:
        return [str(i) for i in range(min(count, 2))]
    devices=[x.strip() for x in visible.split(',') if x.strip()]
    if len(devices)<count:
        raise ValueError('CUDA visibility count disagrees with the allocation mask')
    return devices[:min(count,2)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    script=Path(__file__).resolve().with_name('continue_shared_orientation_gpu.py')
    for name in ('results.json','checkpoint_1000.pt'):
        if not (args.fit_dir/name).is_file():raise FileNotFoundError(args.fit_dir/name)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    import torch
    devices=allocation_devices(os.environ.get('CUDA_VISIBLE_DEVICES'),torch.cuda.device_count())
    args.output_dir.mkdir(parents=True)
    jobs=[];status=[]
    def launch(arm,device):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device
        env['OMP_NUM_THREADS']='1';env['PYTHONUNBUFFERED']='1';env['LIDRA_SKIP_INIT']='true'
        cmd=[sys.executable,str(script),'--arm',arm,'--fit-dir',str(args.fit_dir),
             '--output-dir',str(args.output_dir/arm)]
        print('Starting',arm,'on allocated device',device,flush=True)
        return subprocess.Popen(cmd,env=env)
    if len(devices)==1:
        for arm in ('current','shape_path'):
            code=launch(arm,devices[0]).wait();status.append(dict(arm=arm,returncode=code))
            if code:break
    else:
        for arm,device in zip(('current','shape_path'),devices):jobs.append((arm,launch(arm,device)))
        for arm,job in jobs:status.append(dict(arm=arm,returncode=job.wait()))
    (args.output_dir/'pair_status.json').write_text(json.dumps(status,indent=2)+'\n')
    if len(status)!=2 or any(r['returncode'] for r in status):
        raise SystemExit('An arm failed; return its traceback and results.partial.json. See pair_status.json.')
    print('Both arms finished. Return current/scope_current_bundle.zip and shape_path/scope_shape_path_bundle.zip under',args.output_dir,flush=True)


if __name__=='__main__':main()
