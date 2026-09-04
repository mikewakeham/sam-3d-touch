#!/bin/bash
#SBATCH --job-name=diagnostics
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

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python diagnostics.py \
  --checkpoint outputs/stage1_image/best.pt \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --data-config configs/data1.yaml

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python diagnostics.py \
  --checkpoint outputs/stage1_touch/best.pt \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --data-config configs/data1.yaml

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python diagnostics.py \
  --checkpoint outputs/stage1_touch_train_vecsetx/best.pt \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --data-config configs/data1.yaml
