#!/bin/bash
#SBATCH --job-name=zv_pointmap_surface_scratch_lr1e4
#SBATCH --partition=kempner_h200
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
export WANDB_RUN_GROUP=zeroverse

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/torchrun --standalone --nproc_per_node=4 train.py \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --data-config configs/data_zeroverse_5000_8views_full_surface.yaml \
  --output-dir outputs/zeroverse/zeroverse_pointmap_surface_scratch_lr1e4 \
  --batch-size 4 \
  --workers 8 \
  --val-workers 2 \
  --cross-attention-learning-rate 1e-4 \
  --max-steps 20000 \
  --train-scope shape_cross_attention \
  --train-vecsetx \
  --vecsetx-from-scratch "$@"
