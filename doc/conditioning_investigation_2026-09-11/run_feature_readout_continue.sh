#!/bin/bash
# Run on an already allocated interactive GPU node. No new allocation is made.
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
readout_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
readout_fit_root=${1:-outputs/conditioning_investigation/feature_readout/manual}
readout_output_root=${2:-outputs/conditioning_investigation/feature_readout_continued/manual}
readout_source=${3:-outputs/conditioning_investigation/representation_probe/46083371}
for readout_arm in raw raw_slot processed; do
  test -f "$readout_fit_root/$readout_arm/checkpoint_2000.pt" || { echo "Missing checkpoint: $readout_fit_root/$readout_arm/checkpoint_2000.pt" >&2; exit 1; }
  test ! -e "$readout_output_root/$readout_arm" || { echo "Output already exists: $readout_output_root/$readout_arm" >&2; exit 1; }
done
for readout_arm in raw raw_slot processed; do
  "$readout_python" doc/conditioning_investigation_2026-09-11/fit_feature_readout.py \
    --representation-dir "$readout_source" --arm "$readout_arm" --steps 8000 \
    --resume "$readout_fit_root/$readout_arm/checkpoint_2000.pt" \
    --output-dir "$readout_output_root/$readout_arm"
done
