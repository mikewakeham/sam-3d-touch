#!/bin/bash
#SBATCH --job-name=oa-32views-grasp-redo
#SBATCH --account=kempner_qianqian_lab
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/oa-32views-grasp-redo-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/oa-32views-grasp-redo-%j.err

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-data/bin/python -u data_generation/general/redo_small_object_views_once.py \
    --data-root /n/lab_storage/qianqian_lab/Lab/mwakeham/datasets/objaverse_oa/sam3d_32views_grasp \
    --finish \
    "$@"
