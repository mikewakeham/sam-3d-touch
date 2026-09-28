#!/bin/bash
#SBATCH --job-name=toys4k-100-1view
#SBATCH --account=kempner_qianqian_lab
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/toys4k-100-1view-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/toys4k-100-1view-%j.err

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

toys4k_assets=/n/netscratch/qianqian_lab/Lab/mwakeham/toys4k/raw/extracted
if [[ -d "$toys4k_assets/toys4k_blend_files" ]]; then
    toys4k_assets="$toys4k_assets/toys4k_blend_files"
fi

# Create data_generation/general/logs before submitting.
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-data/bin/python -u data_generation/general/make_data.py \
    --objects-root "$toys4k_assets" \
    --objects-manifest data_generation/toys4k/selection.json \
    --data-root /n/netscratch/qianqian_lab/Lab/mwakeham/toys4k_100_sam3d_1view \
    --stage all \
    --num-views 1 \
    --num-points 8192 \
    --resolution 768 \
    --samples 32 \
    --camera-mode trellis2 \
    --lighting trellis2 \
    --seed 29 \
    --train-fraction 0 \
    --val-fraction 1 \
    --blender /n/holylabs/qianqian_lab/Lab/mwakeham/blender/blender-4.5.9-linux-x64/blender \
    --render-device OPTIX \
    --workers 2 \
    --render-workers-per-gpu 2 \
    "$@"
