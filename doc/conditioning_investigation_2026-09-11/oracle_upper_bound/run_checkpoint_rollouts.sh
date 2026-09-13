#!/usr/bin/env bash
# Existing interactive allocation, sam3d-objects environment, repository root.
set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
ROLLOUT_CODE=doc/conditioning_investigation_2026-09-11/oracle_upper_bound
ROLLOUT_REFERENCE=outputs/conditioning_investigation/full_checkpoint_probe_20260913_133239
ROLLOUT_ROOT=outputs/conditioning_investigation/full_checkpoint_rollouts_$(date +%Y%m%d_%H%M%S)
ROLLOUT_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
if (( ROLLOUT_GPUS < 1 )); then
  echo 'This diagnostic needs an allocated GPU.' >&2
  exit 1
fi
if (( ROLLOUT_GPUS >= 2 )); then
  python "$ROLLOUT_CODE/checkpoint_rollout_gpu.py" --probe-root "$ROLLOUT_REFERENCE" --arm oracle --device-index 0 --output "$ROLLOUT_ROOT/oracle" &
  ORACLE_PID=$!
  python "$ROLLOUT_CODE/checkpoint_rollout_gpu.py" --probe-root "$ROLLOUT_REFERENCE" --arm constant --device-index 1 --output "$ROLLOUT_ROOT/constant" &
  CONSTANT_PID=$!
  ROLLOUT_STATUS=0
  wait "$ORACLE_PID" || ROLLOUT_STATUS=1
  wait "$CONSTANT_PID" || ROLLOUT_STATUS=1
  if (( ROLLOUT_STATUS != 0 )); then exit "$ROLLOUT_STATUS"; fi
else
  python "$ROLLOUT_CODE/checkpoint_rollout_gpu.py" --probe-root "$ROLLOUT_REFERENCE" --arm oracle --output "$ROLLOUT_ROOT/oracle"
  python "$ROLLOUT_CODE/checkpoint_rollout_gpu.py" --probe-root "$ROLLOUT_REFERENCE" --arm constant --output "$ROLLOUT_ROOT/constant"
fi
python "$ROLLOUT_CODE/analyze_checkpoint_rollouts.py" --root "$ROLLOUT_ROOT"
python -m zipfile -c "$ROLLOUT_ROOT.zip" "$ROLLOUT_ROOT"
echo "Return this bundle: $ROLLOUT_ROOT.zip"
