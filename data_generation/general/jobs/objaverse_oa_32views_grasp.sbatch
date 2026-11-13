#!/bin/bash
#SBATCH --job-name=oa-32views-grasp
#SBATCH --account=kempner_qianqian_lab
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=16:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/oa-32views-grasp-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/oa-32views-grasp-%j.err

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Run jobs/select_grasp_views_objaverse_oa.sbatch (same --data-root) first.
# Create data_generation/general/logs before submitting.
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-data/bin/python -u data_generation/general/make_data.py \
    --objects-root /n/lab_storage/qianqian_lab/Lab/mwakeham/datasets/objaverse_oa/meshes \
    --objects-manifest /n/lab_storage/qianqian_lab/Lab/mwakeham/datasets/objaverse_oa/sam3d_32views_grasp/grasp_views/objects_manifest.json \
    --grasp-views /n/lab_storage/qianqian_lab/Lab/mwakeham/datasets/objaverse_oa/sam3d_32views_grasp/grasp_views \
    --data-root /n/lab_storage/qianqian_lab/Lab/mwakeham/datasets/objaverse_oa/sam3d_32views_grasp \
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
