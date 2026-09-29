#!/bin/bash
#SBATCH --job-name=zv_orbits_lr1e4
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/zeroverse/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/zeroverse/%x-%j.err

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python -m evaluation.make_orbit \
  --evaluation-dir outputs/zeroverse/evaluation_lr1e4 \
  --all-samples \
  --max-samples 10 \
  --with-inputs \
  --modes mesh voxel \
  --blender /n/holylabs/qianqian_lab/Lab/mwakeham/blender/blender-4.5.9-linux-x64/blender "$@"
