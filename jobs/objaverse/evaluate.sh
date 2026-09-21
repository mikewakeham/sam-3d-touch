#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/objaverse/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/objaverse/%x-%j.err

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python -m evaluation.evaluate \
  --run-dirs \
  outputs/objaverse/stage1_image_full_cross_attention \
  outputs/objaverse/stage1_full_surface_full_cross_attention \
  outputs/objaverse/stage1_image_shared_normalization \
  outputs/objaverse/stage1_full_surface_shared_normalization \
  outputs/objaverse/stage1_image_no_pointmap_full_cross_attention \
  outputs/objaverse/stage1_full_surface_no_pointmap_full_cross_attention \
  outputs/objaverse/stage1_full_surface_oracle_no_pointmap \
  outputs/objaverse/stage1_full_surface_oracle \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --selection-data-config configs/data_full_surface.yaml \
  --output-dir outputs/objaverse/evaluation_coordinate_system \
  --split val \
  --max-samples 0 \
  --selection hidden \
  --workers 4 \
  --inference-steps 25 \
  --stage2-inference-steps 25
