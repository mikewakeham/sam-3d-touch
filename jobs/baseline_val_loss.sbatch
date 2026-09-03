#!/bin/bash
#SBATCH --job-name=s1_official_baseline
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/%x-%j.err

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/torchrun --standalone --nproc_per_node=4 baseline_val_loss.py \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --data-config configs/data1.yaml \
  --output-dir outputs/s1_official_baseline_gbs16_s29 \
  --batch-size 4 \
  --workers 2 \
  --epochs 20
