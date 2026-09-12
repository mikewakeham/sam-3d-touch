# Frozen early/late checkpoint interventions

The completed 16-object fits initially beat image+pointmap on held objects, then lose that advantage. We do not yet know whether the early advantage uses the correct surface, or whether late degradation remains in the trained shared visual attention when the surface is removed. This probe resolves that specific fork before another training intervention.

Use existing step **256 and 1024** checkpoints for **image, camera and oracle**. No retraining or optimizer steps. Oracle dropout is omitted because it did not supply a held-identity rescue; the two original point arms retain the coordinate comparison. Evaluate only the 16 held identities × two views already reserved in `object_transfer_plan.json`.

Conditions for each point checkpoint: correct surface, wrong surface from each of the other three objects in its batch, and surface tokens removed. Image has its normal visual-only context (including pointmap), labelled `removed` in the report. All visual tokens stay present. Removing touch does not zero the visuals or activate CFG. The first shape attention's actual input is checked for every condition/checkpoint. Wrong surfaces preserve the number of tokens; removal changes it and is a diagnostic out-of-distribution intervention, not an inference recommendation.

Require the original run's report byte hash, source/configuration checks, exact actual visual/point-feature/target hashes and checkpoint metadata. Reproduce all held-object scalar and per-object values at step 256 on the original monitor bank; reproduce both correct and original wrong-surface conditions at 1024 on its final bank. Refuse mismatches. The late full-parameter digest must match the original report. Verify all parameters remain unchanged during each assessment and the frozen parameters remain unchanged across both checkpoint loads.

Then use eight fresh common draws, bank `800000 + 37 + 100*(original_batch_index % 4) + draw`. These are shared across models, checkpoints, conditions and the two views of each identity. Average per identity; do not count views/draws/distractors as independent objects. Three distractors are still a restricted set, not an exhaustive wrong-shape distribution. Early checkpoint selection is based on an already inspected monitor; fresh draws reduce noise selection effects, not object-selection bias.

The analyzer reports correct, three-distractor mean and removed loss, with paired object differences: wrong-minus-correct (identity benefit), removed-minus-correct (net surface-presence benefit), correct-minus-image and removed-minus-image. Read the outcome branches in `OBJECT_TRANSFER_RETURNED_FINDINGS.md` before proposing training. In particular, an early same-step advantage alone is insufficient: the later image checkpoint already achieves lower absolute held loss.

This is 1,408 fresh batch forwards plus 416 historical replay forwards across all three models, with no backward passes. GPU wall time is unmeasured. Three visible allocated GPUs run concurrently; fewer run sequentially on one visible device. A fourth GPU is not needed. Local Python syntax, synthetic paired arithmetic, complete condition coverage and negative checks for missing cells, replay/input drift, incorrect scalar reduction and failed token removal pass. GPU execution remains pending.

Sync the new scripts and `object_transfer_returned_manual/` alongside the existing investigation folder. Keep the original fit checkpoints in place. Paste the actual contents below into the allocated interactive terminal:

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
probe_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
probe_root=outputs/conditioning_investigation/transfer_checkpoints/manual
fit_root=outputs/conditioning_investigation/object_transfer/manual
visible_gpus=$("$probe_python" -c 'import torch; print(torch.cuda.device_count())')
test "$visible_gpus" -ge 1
test ! -e "$probe_root"
mkdir -p "$probe_root"
if [ "$visible_gpus" -ge 3 ]; then
  probe_pids=()
  device_index=0
  for model_key in image camera oracle; do
    "$probe_python" doc/conditioning_investigation_2026-09-11/probe_transfer_checkpoints_gpu.py \
      --model-key "$model_key" --device-index "$device_index" \
      --fit-root "$fit_root" --output "$probe_root/$model_key.json" \
      > "$probe_root/$model_key.log" 2>&1 &
    probe_pids+=("$!")
    device_index=$((device_index + 1))
  done
  probe_failed=0
  for process_id in "${probe_pids[@]}"; do
    if ! wait "$process_id"; then
      probe_failed=1
    fi
  done
  test "$probe_failed" -eq 0
else
  for model_key in image camera oracle; do
    "$probe_python" doc/conditioning_investigation_2026-09-11/probe_transfer_checkpoints_gpu.py \
      --model-key "$model_key" --device-index 0 \
      --fit-root "$fit_root" --output "$probe_root/$model_key.json"
  done
fi
"$probe_python" doc/conditioning_investigation_2026-09-11/analyze_transfer_checkpoints.py \
  "$probe_root" --output "$probe_root/analysis.json"
)
```

Return `image.json`, `camera.json`, `oracle.json` and `analysis.json`. On failure, return the traceback/log instead of relaxing checks or retraining. Existing output paths are never overwritten. No additional training is queued.
