#!/bin/bash
#SBATCH --job-name=sam3d-make-data
#SBATCH --account=kempner_qianqian_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --gres=gpu:3
#SBATCH --mem=512G
#SBATCH --exclude=holygpu8a19102
#SBATCH --time=08:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch-data/objaverse-dexonomy/logs/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch-data/objaverse-dexonomy/logs/%x-%j.err

module purge
module load cuda/12.2.0-fasrc01

source activate sam3d-objects

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch-data/objaverse-dexonomy

python make_data.py \
    --workers 14 \
    --render-workers-per-gpu 2