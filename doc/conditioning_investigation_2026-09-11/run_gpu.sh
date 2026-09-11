#!/bin/bash
#SBATCH --job-name=conditioning_probe
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python \
  doc/conditioning_investigation_2026-09-11/probe_gpu.py \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --checkpoints outputs/stage1_image_full_cross_attention/best.pt \
                outputs/stage1_full_surface_full_cross_attention/best.pt \
  --split train --objects 8 --batch-size 4 --draws 4 \
  --output "outputs/conditioning_probe/train-${SLURM_JOB_ID:-manual}.json"
