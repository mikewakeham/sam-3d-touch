#!/bin/bash
#SBATCH --job-name=feature_readout
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --time=01:00:00

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
readout_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
readout_arms=(raw raw_slot processed)
readout_arm=${readout_arms[${SLURM_ARRAY_TASK_ID:-0}]}
readout_source=${1:-outputs/conditioning_investigation/representation_probe/46083371}
"$readout_python" doc/conditioning_investigation_2026-09-11/test_feature_readout.py
"$readout_python" doc/conditioning_investigation_2026-09-11/fit_feature_readout.py \
  --representation-dir "$readout_source" --arm "$readout_arm" --steps 2000 \
  --output-dir "outputs/conditioning_investigation/feature_readout/${SLURM_ARRAY_JOB_ID:-manual}/$readout_arm"
