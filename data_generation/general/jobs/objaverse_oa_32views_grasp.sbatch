#!/bin/bash
#SBATCH --job-name=oa-32views-grasp
#SBATCH --account=kempner_qianqian_lab
#SBATCH --partition=kempner_rtx
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/oa-32views-grasp-%A_%a.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/oa-32views-grasp-%A_%a.err

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-data/bin/python -u data_generation/general/make_data.py \
    --objects-root /n/lab_storage/qianqian_lab/Lab/mwakeham/datasets/objaverse_oa/meshes \
    --objects-manifest /n/lab_storage/qianqian_lab/Lab/mwakeham/datasets/objaverse_oa/sam3d_32views_grasp/grasp_views/objects_manifest.json \
    --grasp-views /n/lab_storage/qianqian_lab/Lab/mwakeham/datasets/objaverse_oa/sam3d_32views_grasp/grasp_views \
    --data-root /n/lab_storage/qianqian_lab/Lab/mwakeham/datasets/objaverse_oa/sam3d_32views_grasp \
    --shard "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT" \
    --stage all \
    --num-views 32 \
    --num-points 8192 \
    --resolution 768 \
    --samples 32 \
    --camera-mode trellis2 \
    --lighting trellis2 \
    --seed 29 \
    --train-fraction 0.9 \
    --val-fraction 0.05 \
    --blender /n/holylabs/qianqian_lab/Lab/mwakeham/blender/blender-4.5.9-linux-x64/blender \
    --render-device OPTIX \
    --workers 24 \
    --render-workers-per-gpu 2 \
    --blender-threads 6 \
    "$@"
