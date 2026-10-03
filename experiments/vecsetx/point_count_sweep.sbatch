#!/bin/bash
#SBATCH --job-name=vecsetx_point_count
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/experiments/vecsetx/point_count.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/experiments/vecsetx/point_count.err
#SBATCH --open-mode=truncate

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python -m experiments.vecsetx.point_count_sweep \
  --data-config configs/data_zeroverse_5000_8views_touch_32x256.yaml \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --encoder-checkpoint /n/home12/mwakeham/.cache/huggingface/hub/models--Zbalpha--VecSetX/snapshots/5fb84917189d2bee8392404f833f42ca5c067e0b/learnable_vec1024x32_dim1024_depth24_sdf_nb/checkpoint-125.pth \
  --output-dir experiments/vecsetx/outputs/point_count \
  --objects 100 \
  --seed 29 \
  --resolution 128 "$@"
