#!/bin/bash
#SBATCH --job-name=s1_touch_kv3e6
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/jobs/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/jobs/%x-%j.err

set -e
source activate sam3d-objects
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

torchrun --standalone --nproc_per_node=4 train.py \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --data-config configs/data1.yaml \
  --output-dir outputs/s1_touch_kv3e-6_gbs16_s29 \
  --batch-size 4 \
  --workers 8 \
  --val-workers 2 \
  --epochs 10 \
  --cross-attention-learning-rate 3e-6
