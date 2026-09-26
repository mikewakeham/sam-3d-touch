#!/bin/bash
#SBATCH --job-name=zv_evaluate_lr1e4
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
  --run-dirs \
  outputs/zeroverse/zeroverse_pointmap_lr1e4 \
  outputs/zeroverse/zeroverse_pointmap_surface_frozen_lr1e4 \
  outputs/zeroverse/zeroverse_pointmap_surface_scratch_lr1e4 \
  outputs/zeroverse/zeroverse_pointmap_touch_32x256_joint \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --selection-data-config configs/data_zeroverse_5000_8views_full_surface.yaml \
  --output-dir outputs/zeroverse/evaluation_lr1e4 \
  --split val \
  --max-samples 100 \
  --selection random \
  --workers 4 \
  --inference-steps 25 \
  --stage2-inference-steps 25 "$@"
