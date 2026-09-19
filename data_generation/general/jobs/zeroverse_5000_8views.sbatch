#!/bin/bash
#SBATCH --job-name=zeroverse-5000-8views
#SBATCH --account=kempner_qianqian_lab
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/zeroverse-5000-8views-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/zeroverse-5000-8views-%j.err

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Create data_generation/general/logs before submitting; Slurm opens logs before running this script.
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-data/bin/python -u data_generation/general/make_data.py \
    --objects-root /n/netscratch/qianqian_lab/Lab/mwakeham/zeroverse_5000 \
    --data-root /n/netscratch/qianqian_lab/Lab/mwakeham/zeroverse_5000_sam3d_8views \
    --num-views 8 \
    --pattern '**/*.glb' \
    --blender /n/holylabs/qianqian_lab/Lab/mwakeham/blender/blender-4.5.9-linux-x64/blender \
    --render-device OPTIX \
    --ready-marker '{stem}_generation.json' \
    --workers 12 \
    --render-workers-per-gpu 2 \
    "$@"
