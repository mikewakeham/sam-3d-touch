#!/bin/bash
#SBATCH --job-name=encoder_compare
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/experiments/encoders/logs/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/experiments/encoders/logs/%x-%j.err

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# First argument names a new output directory; remaining arguments are checkpoints.
name=${1:?Specify a comparison name}
shift
if [ "$#" -eq 0 ]; then echo "Provide checkpoints to compare" >&2; exit 1; fi
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python experiments/encoders/compare.py \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --output-dir "experiments/encoders/outputs/$name" \
  --checkpoints "$@"
