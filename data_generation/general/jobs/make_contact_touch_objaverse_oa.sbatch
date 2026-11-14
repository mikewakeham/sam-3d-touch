#!/bin/bash
#SBATCH --job-name=oa-contact-touch
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/oa-contact-touch-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/oa-contact-touch-%j.err

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

OA=/n/lab_storage/qianqian_lab/Lab/mwakeham/datasets/objaverse_oa

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python -u data_generation/general/make_contact_touch.py \
    --data-root "$OA/sam3d_32views_grasp" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --joint-pointmap \
    "$@"
