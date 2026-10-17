#!/bin/bash
#SBATCH --job-name=toys4k_100_1view_sampled_camframe
#SBATCH --account=kempner_qianqian_lab
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/toys4k_100_1view_sampled_camframe-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/data_generation/general/logs/toys4k_100_1view_sampled_camframe-%j.err

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-data/bin/python -u data_generation/general/generate_target_latents.py \
    --data-root /n/lab_storage/qianqian_lab/Lab/mwakeham/toys4k_100_sam3d_1view_sampled \
    --target-frame camera "$@"
