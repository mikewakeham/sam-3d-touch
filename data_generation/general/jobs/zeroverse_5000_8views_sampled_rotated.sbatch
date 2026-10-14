#!/bin/bash
#SBATCH --job-name=zeroverse-5000-8views-sampled-rotated
#SBATCH --account=kempner_qianqian_lab
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/zeroverse-5000-8views-sampled-rotated-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/zeroverse-5000-8views-sampled-rotated-%j.err

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

data_root=/n/netscratch/qianqian_lab/Lab/mwakeham/zeroverse_5000_sam3d_8views_sampled_rotated
original_split=/n/netscratch/qianqian_lab/Lab/mwakeham/zeroverse_5000_sam3d_8views/generated_data/splits_train_val.json
mkdir -p "$data_root/generated_data"
for split in splits.json splits_train_val.json; do
    if [[ -f "$data_root/generated_data/$split" ]]; then
        cmp "$original_split" "$data_root/generated_data/$split"
    else
        cp "$original_split" "$data_root/generated_data/$split"
    fi
done

# Create data_generation/general/logs before submitting; Slurm opens logs before running this script.
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-data/bin/python -u data_generation/general/make_data.py \
    --objects-root /n/netscratch/qianqian_lab/Lab/mwakeham/zeroverse_5000 \
    --data-root "$data_root" \
    --stage all \
    --num-views 8 \
    --random-object-rotation \
    --num-points 8192 \
    --resolution 768 \
    --samples 32 \
    --camera-mode trellis2 \
    --lighting trellis2 \
    --seed 29 \
    --pattern '**/*.glb' \
    --blender /n/holylabs/qianqian_lab/Lab/mwakeham/blender/blender-4.5.9-linux-x64/blender \
    --render-device OPTIX \
    --ready-marker '{stem}_generation.json' \
    --workers 12 \
    --render-workers-per-gpu 2
