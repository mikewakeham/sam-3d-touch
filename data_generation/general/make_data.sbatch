#!/bin/bash
#SBATCH --job-name=sam3d-data
#SBATCH --account=kempner_qianqian_lab
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=sam3d-data-%j.out
#SBATCH --error=sam3d-data-%j.err

set -euo pipefail
module purge
source activate sam3d-data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Submit from the sam-3d-touch checkout.
python -u data_generation/general/make_data.py \
    --objects-root /n/netscratch/qianqian_lab/Lab/mwakeham/zeroverse_5000 \
    --data-root /n/netscratch/qianqian_lab/Lab/mwakeham/zeroverse_5000_sam3d_trellis2 \
    --pattern '**/*.glb' \
    --blender /n/holylabs/qianqian_lab/Lab/mwakeham/blender/blender-4.5.9-linux-x64/blender \
    --render-device OPTIX \
    --ready-marker '{stem}_generation.json' \
    --workers 12 \
    --render-workers-per-gpu 2 \
    "$@"
