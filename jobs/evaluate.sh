#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/%x-%j.err

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python evaluate.py \
  --run-dirs \
  outputs/stage1_image_full_cross_attention \
  outputs/stage1_full_surface_full_cross_attention \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --selection-data-config configs/data_full_surface.yaml \
  --output-dir outputs/evaluation \
  --max-samples 0 \
  --selection hidden \
  --workers 4 \
  --inference-steps 25 \
  --stage2-inference-steps 25
