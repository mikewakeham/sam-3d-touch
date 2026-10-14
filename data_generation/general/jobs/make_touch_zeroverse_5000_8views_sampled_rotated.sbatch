#!/bin/bash
#SBATCH --job-name=make_touch_zeroverse_5000_8views_sampled_rotated
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/make_touch_zeroverse_5000_8views_sampled_rotated-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/make_touch_zeroverse_5000_8views_sampled_rotated-%j.err

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Create data_generation/general/logs before submitting.
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python -u data_generation/general/make_touch_data.py \
    --data-root /n/netscratch/qianqian_lab/Lab/mwakeham/zeroverse_5000_sam3d_8views_sampled_rotated \
    --workers 16 \
    --joint-pointmap
