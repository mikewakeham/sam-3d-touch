#!/bin/bash
#SBATCH --job-name=vecsetx_experiments
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

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python -m experiments.vecsetx.latent_space \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --touch-config configs/data1.yaml \
  --full-surface-config configs/data_full_surface.yaml \
  --output-dir experiments/vecsetx/outputs/latent_space \
  --objects 0 \
  --views-per-object 0 \
  --batch-size 4

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python -m experiments.vecsetx.reconstruct \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --touch-config configs/data1.yaml \
  --full-surface-config configs/data_full_surface.yaml \
  --output-dir experiments/vecsetx/outputs/reconstruction \
  --objects 0 \
  --views-per-object 1
