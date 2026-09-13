"""Run independent arms on 1, 2 or 4 visible GPUs, preserving batch/exposure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from protocol import ARMS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight-report', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpus', type=int, choices=(1, 2, 4), required=True)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if len(a.arms) != len(set(a.arms)):
        p.error('Duplicate arm')
    if not a.dry_run:
        import torch
        if torch.cuda.device_count() < a.gpus:
            p.error('Fewer visible GPUs than requested')
        if not json.loads(a.preflight_report.read_text()).get('complete'):
            p.error('Preflight incomplete')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible is not None else [str(i) for i in range(a.gpus)]
    if len(devices) < a.gpus:
        p.error('CUDA_VISIBLE_DEVICES has too few entries')
    jobs = []
    for i, arm in enumerate(a.arms):
        command = [sys.executable, str(Path(__file__).with_name('run_gpu.py')), '--arm', arm,
                   '--preflight-report', str(a.preflight_report), '--output-dir', str(a.output_dir / arm),
                   '--epochs', str(a.epochs), '--workers', str(a.workers), '--batch-size', '4']
        jobs.append((arm, command))
    if a.dry_run:
        print(json.dumps({'gpus': devices[:a.gpus], 'waves': [jobs[i:i+a.gpus] for i in range(0, len(jobs), a.gpus)]}, indent=2))
        return
    a.output_dir.mkdir(parents=True, exist_ok=False)
    # Fixed waves: two GPUs prioritize the oracle and constant comparison first.
    for offset in range(0, len(jobs), a.gpus):
        active = []
        for device, (arm, command) in zip(devices, jobs[offset:offset + a.gpus]):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            log = (a.output_dir / f'{arm}.log').open('w')
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append((arm, process, log)); print('Started', arm, 'GPU', device, flush=True)
        pending = list(active); failed = []
        while pending:
            for item in list(pending):
                arm, process, log = item
                status = process.poll()
                if status is not None:
                    log.close(); pending.remove(item)
                    print('Finished', arm, 'exit', status, flush=True)
                    if status:
                        failed.append((arm, status))
            if pending:
                time.sleep(2)
        if failed:
            raise RuntimeError(f'Failed {failed}; no automatic retry or next wave. See arm logs.')


if __name__ == '__main__':
    main()
