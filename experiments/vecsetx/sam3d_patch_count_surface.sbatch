#!/bin/bash
#SBATCH --job-name=sam3d_patch_count_surface_camframe
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/experiments/vecsetx/logs/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/experiments/vecsetx/logs/%x-%j.err
#SBATCH --open-mode=truncate

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Create experiments/vecsetx/logs before submitting.
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python -m experiments.vecsetx.sam3d_patch_count \
  --checkpoint outputs/zeroverse/zeroverse_pointmap_surface_frozen_lr1e4_sampled_camframe/best.pt \
  --data-config configs/data_zeroverse_5000_8views_sampled_touch_32x256_camframe.yaml \
  --data-root /n/lab_storage/qianqian_lab/Lab/mwakeham/zeroverse_5000_sam3d_8views_sampled \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --output-dir experiments/vecsetx/outputs/sam3d_patch_count_surface_sampled_camframe_full \
  --selection-data-config configs/data_zeroverse_5000_8views_full_surface.yaml \
  --reference-evaluation outputs/zeroverse/evaluation_lr1e4_sampled \
  --max-samples 100 \
  --selection random \
  --selection-seed 29 \
  --seed 29 \
  --inference-steps 25 \
  --stage2-inference-steps 25 "$@"
