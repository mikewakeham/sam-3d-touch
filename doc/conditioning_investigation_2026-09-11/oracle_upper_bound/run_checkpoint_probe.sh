#!/usr/bin/env bash
# Run inside an existing GPU allocation, with the sam3d-objects environment active.
set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PROBE_CODE=doc/conditioning_investigation_2026-09-11/oracle_upper_bound
PROBE_ROOT=outputs/conditioning_investigation/full_checkpoint_probe_$(date +%Y%m%d_%H%M%S)
PROBE_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
if (( PROBE_GPUS < 1 )); then
  echo 'This probe needs an allocated GPU.' >&2
  exit 1
fi
if (( PROBE_GPUS >= 2 )); then
  python "$PROBE_CODE/checkpoint_probe_gpu.py" --arm oracle --device-index 0 --output "$PROBE_ROOT/oracle" &
  ORACLE_PID=$!
  python "$PROBE_CODE/checkpoint_probe_gpu.py" --arm constant --device-index 1 --output "$PROBE_ROOT/constant" &
  CONSTANT_PID=$!
  PROBE_STATUS=0
  wait "$ORACLE_PID" || PROBE_STATUS=1
  wait "$CONSTANT_PID" || PROBE_STATUS=1
  if (( PROBE_STATUS != 0 )); then exit "$PROBE_STATUS"; fi
else
  python "$PROBE_CODE/checkpoint_probe_gpu.py" --arm oracle --output "$PROBE_ROOT/oracle"
  python "$PROBE_CODE/checkpoint_probe_gpu.py" --arm constant --output "$PROBE_ROOT/constant"
fi
python "$PROBE_CODE/analyze_checkpoint_probe.py" --root "$PROBE_ROOT"
python -m zipfile -c "$PROBE_ROOT.zip" "$PROBE_ROOT"
echo "Return this bundle: $PROBE_ROOT.zip"
