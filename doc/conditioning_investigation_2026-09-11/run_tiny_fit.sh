#!/bin/bash
#SBATCH --job-name=frame_tiny_fit
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
for arm in image camera oracle; do
 /n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python \
  doc/conditioning_investigation_2026-09-11/tiny_fit_gpu.py \
  --arm "$arm" --objects 4 --steps 1000 \
  --output-dir "outputs/conditioning_investigation/frame_tiny_fit/${SLURM_JOB_ID:-manual}/$arm"
done
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python \
 doc/conditioning_investigation_2026-09-11/check_tiny_fit.py \
 "outputs/conditioning_investigation/frame_tiny_fit/${SLURM_JOB_ID:-manual}"
