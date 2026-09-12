#!/bin/bash
#SBATCH --job-name=tiny_fit_views
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
fit_root="${1:-outputs/conditioning_investigation/frame_tiny_fit/46083371}"
for arm in image camera oracle; do
 /n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python \
  doc/conditioning_investigation_2026-09-11/probe_tiny_fit_views.py \
  --fit-dir "$fit_root/$arm" \
  --output "outputs/conditioning_investigation/tiny_fit_view_probe/${SLURM_JOB_ID:-manual}/$arm.json"
done
