#!/usr/bin/env bash
# Existing interactive allocation; one visible GPU, no training.
set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
FRAME_CODE=doc/conditioning_investigation_2026-09-11/coordinate_reassessment_20260913
FRAME_LOSS=outputs/conditioning_investigation/full_checkpoint_probe_20260913_133239
FRAME_ROLLOUT=outputs/conditioning_investigation/full_checkpoint_rollouts_20260913_143515
FRAME_REUSE=outputs/conditioning_investigation/full_frame_probe_20260913_170408/camera
FRAME_ROOT=outputs/conditioning_investigation/full_frame_probe_$(date +%Y%m%d_%H%M%S)
mkdir -p "$FRAME_ROOT/source"
cp "$FRAME_CODE/frame_probe_gpu.py" "$FRAME_CODE/frame_probe_core.py" "$FRAME_CODE/analyze_frame_probe.py" "$FRAME_ROOT/source/"
FRAME_STATUS=0
if python "$FRAME_CODE/frame_probe_gpu.py" \
    --probe-root "$FRAME_LOSS" --rollout-root "$FRAME_ROLLOUT" \
    --resume-geometry-from "$FRAME_REUSE" \
    --output "$FRAME_ROOT/camera" 2>&1 | tee "$FRAME_ROOT/run.log"; then
  python "$FRAME_CODE/analyze_frame_probe.py" \
    --camera-dir "$FRAME_ROOT/camera" --probe-root "$FRAME_LOSS" \
    --rollout-root "$FRAME_ROLLOUT" --output "$FRAME_ROOT/summary.json" \
    2>&1 | tee "$FRAME_ROOT/analysis.log" || FRAME_STATUS=$?
else
  FRAME_STATUS=$?
fi
python -m zipfile -c "$FRAME_ROOT.zip" "$FRAME_ROOT"
echo "Return this bundle (including if a check failed): $FRAME_ROOT.zip"
exit "$FRAME_STATUS"
