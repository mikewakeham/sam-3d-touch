#!/bin/bash
#SBATCH --job-name=sam3d_patch_count_toys4k
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/experiments/vecsetx/logs/sam3d_patch_count_toys4k.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/experiments/vecsetx/logs/sam3d_patch_count_toys4k.err
#SBATCH --open-mode=truncate

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Create experiments/vecsetx/logs before submitting.
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python -m experiments.vecsetx.sam3d_patch_count \
  --checkpoint outputs/zeroverse/zeroverse_pointmap_touch_32x256/best.pt \
  --data-config configs/data_toys4k_100_1view_touch_32x256.yaml \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --output-dir experiments/vecsetx/outputs/sam3d_patch_count_toys4k \
  --selection-data-config configs/data_toys4k_100_1view_full_surface.yaml \
  --reference-evaluation outputs/zeroverse/evaluation_toys4k \
  --max-samples 0 \
  --selection random \
  --selection-seed 29 \
  --seed 29 \
  --inference-steps 25 \
  --stage2-inference-steps 25 "$@"
