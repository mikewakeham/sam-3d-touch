"""Prepare once, then run matched arms on one or two allocated GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_shared_orientation_scope_pair import allocation_devices


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit-dir',type=Path,required=True)
    ap.add_argument('--visual-probe-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    for p in [args.fit_dir/'checkpoint_2000.pt',args.fit_dir/'scope_shape_path_bundle.zip',args.visual_probe_dir/'results.json']:
        if not p.is_file():raise FileNotFoundError(p)
    import torch
    devices=allocation_devices(os.environ.get('CUDA_VISIBLE_DEVICES'),torch.cuda.device_count())
    args.output_dir.mkdir(parents=True);script=Path(__file__).with_name('orientation_coverage_gpu.py').resolve()
    common=[sys.executable,str(script),'--fit-dir',str(args.fit_dir),'--visual-probe-dir',str(args.visual_probe_dir)]
    status=[]
    def record():(args.output_dir/'pair_status.json').write_text(json.dumps(status,indent=2)+'\n')
    def launch(extra,device):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',LIDRA_SKIP_INIT='true')
        return subprocess.Popen(common+extra,env=env)
    print('Preparing shared rotation bank on allocated device',devices[0],flush=True)
    bank=args.output_dir/'bank';code=launch(['--phase','prepare','--output-dir',str(bank)],devices[0]).wait()
    status.append(dict(phase='prepare',returncode=code));record()
    if code:raise SystemExit('Bank preparation failed before training; return traceback and bank/results.partial.json if present.')
    def arm(name,device):
        print('Starting',name,'on allocated device',device,flush=True)
        return launch(['--phase','fit','--arm',name,'--bank-dir',str(bank),'--output-dir',str(args.output_dir/name)],device)
    if len(devices)==1:
        for name in ('control','augmented'):
            code=arm(name,devices[0]).wait();status.append(dict(phase=name,returncode=code));record()
            if code:break
    else:
        jobs=[(name,arm(name,device)) for name,device in zip(('control','augmented'),devices)]
        for name,job in jobs:status.append(dict(phase=name,returncode=job.wait()));record()
    if len(status)!=3 or any(s['returncode'] for s in status):raise SystemExit('An arm failed; return its traceback and results.partial.json. See pair_status.json.')
    print('Return control/coverage_control_bundle.zip and augmented/coverage_augmented_bundle.zip under',args.output_dir,flush=True)


if __name__=='__main__':main()
