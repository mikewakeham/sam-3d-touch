# Run the two missing coordinate-factor treatments

This implements the next experiment in `COORDINATE_DIAGNOSIS_RESYNC.md`. It keeps frozen VecSetX, the original trainable projector/full shape cross-attention, 8192 points, targets, visuals, noise schedule, four training views and three reserved views of the original four objects.

The historical camera/oracle arms already exist. `rotation_only` rotates object-normalized points into camera axes. `normalization_only` rotates camera-normalized points back into object axes. Both enter `encoder.encode` directly: a second normalization would invalidate the intervention.

Before any training the new driver requires:

- Archived single-view and multi-view reports, source hashes, configuration files and historical driver/view-selector hashes to agree.
- The same initialized model and trainable CA hashes.
- Every group's image, pointmap, target and both baseline feature hashes to match the historical reports exactly.
- All seven groups of initial baseline losses (eight noise/time draws, correct and swapped surfaces) to reproduce exactly for both camera and oracle.
- Rigid coordinate inversion checks and exactly 8192 valid points, preventing a hidden FPS/density change.

It fails rather than relaxing a failed reproduction check. If preflight fails, return the traceback and any partial report; do not rerun old training or lower tolerances without identifying the difference. A successful initial replay validates comparability, not an independent reproduction of the historical optimizer trajectory.

Each mixed treatment then runs exactly the historical 1000 updates, with assessments at 0/100/300/1000 and optimizer checkpoints at 300/1000. Correct/swap losses use the original monitoring draws. Final conditional Stage-1 samples reuse the historical two noise draws and CFG 0; these are supporting checks. No Stage 2, broader finetuning, new encoder, bridge, sparse sampling, or full-dataset training is used.

`analyze_frame_factors.py` reports rotation effects under both normalizations, normalization effects under both orientations, and the interaction. No percentage attribution is inferred from coordinate distances. One training seed and four objects make this a mechanism screen. A supported correction still needs a modest multi-object confirmation and independent draws before adoption.

## Paste into an allocated interactive terminal

Sync the investigation folder first, including the archived `tiny_fit_returned_46083371/{camera,oracle}/results.json` and `multiple_view_returned_46083371/{camera,oracle}.json`. The new driver uses these local reference copies; it does not need the historical cluster output directory names. Dataset/config paths and pretrained weights must match those reports.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
frame_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
for arm in rotation_only normalization_only; do
  "$frame_python" doc/conditioning_investigation_2026-09-11/fit_frame_factors_gpu.py \
    --arm "$arm" \
    --output-dir "outputs/conditioning_investigation/frame_factors/manual/$arm"
done
"$frame_python" doc/conditioning_investigation_2026-09-11/analyze_frame_factors.py \
  outputs/conditioning_investigation/frame_factors/manual \
  --output outputs/conditioning_investigation/frame_factors/manual/analysis.json
)
```

The two treatments run sequentially on the allocated GPU. Return their `results.json` files and the combined `analysis.json`. Preserve checkpoints/NPZs on the cluster. Output directories must be new; nothing is silently overwritten. No runtime estimate is claimed before running this driver. Python/shell syntax and analyzer logic are checked locally; the torch/GPU preflight must execute on the cluster.
