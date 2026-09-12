#!/bin/bash
#SBATCH --job-name=conditioning_rollout
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python \
  doc/conditioning_investigation_2026-09-11/rollout_gpu.py \
  --probe-json doc/conditioning_investigation_2026-09-11/gpu_probe_train_46083371.json \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --output-dir "outputs/conditioning_investigation/conditioning_rollout/train-${SLURM_JOB_ID:-manual}" \
  --strengths 0 1 7 --seeds 29 30 --steps 25
