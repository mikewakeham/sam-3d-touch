#!/bin/bash
#SBATCH --job-name=zv_evaluation_toys4k
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/zeroverse/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/zeroverse/%x-%j.err

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/torchrun --standalone --nproc_per_node=4 -m evaluation.evaluate \
  --data-config configs/data_toys4k_100_1view_full_surface.yaml \
  --run-dirs \
  outputs/zeroverse/zeroverse_pointmap_lr1e4 \
  outputs/zeroverse/zeroverse_pointmap_surface_frozen_lr1e4 \
  --data-config configs/data_toys4k_100_1view_touch_32x256.yaml \
  --run-dirs \
  outputs/zeroverse/zeroverse_pointmap_touch_32x256_joint \
  outputs/zeroverse/zeroverse_pointmap_touch_32x256 \
  --data-config configs/data_toys4k_100_1view_touch_16x512.yaml \
  --run-dirs outputs/zeroverse/zeroverse_pointmap_touch_16x512_joint \
  --data-config configs/data_toys4k_100_1view_touch_8x1024.yaml \
  --run-dirs outputs/zeroverse/zeroverse_pointmap_touch_8x1024_joint \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --selection-data-config configs/data_toys4k_100_1view_full_surface.yaml \
  --output-dir outputs/zeroverse/evaluation_toys4k \
  --split val \
  --max-samples 0 \
  --selection random \
  --workers 4 \
  --inference-steps 25 \
  --stage2-inference-steps 25 "$@"
