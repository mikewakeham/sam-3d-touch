# Four-arm training screen beyond the four fitted identities

**Completed:** all four reports have returned and passed validation. See `OBJECT_TRANSFER_RETURNED_FINDINGS.md`. Do not rerun or automatically extend this command; the next action is the frozen checkpoint probe in `TRANSFER_CHECKPOINTS_HANDOFF.md`.

## Question and budget

The four-object models fit aligned surfaces and benefited from visual dropout on new views of those same objects, yet every surface model lost to image+pointmap on every one of 32 new objects. More frozen tests of those checkpoints cannot establish whether the learned conditioner can benefit from broader training.

Run one matched **1024-update** screen with 16 training identities and 16 separate held identities. Four arms:

| Model key | Conditions | Training policy |
|---|---|---|
| image | RGB, masks, pointmap | Original, no surface |
| camera | Same visuals + camera-frame full surface | Original |
| oracle | Same visuals + object-frame full surface | Original |
| oracle_dropout | Same oracle inputs | Zero visual tokens on balanced half of updates |

Retain pretrained frozen raw VecSetX, the existing point normalization, no global position embedding, projector LR 1e-4, full shape cross-attention/norm2 LR 1e-5, bf16, batch size four and the native shape-only objective. No new encoder, bridge, pose head or broader finetuning. All models start from the same pretrained generator. Seed 37 gives a new matched initialization for the point adapter and new training draws; this is not a second replication of the old four-object experiment.

Camera dropout is not expanded first: the immediate question is whether *aligned* surface conditioning can show transferable utility, and whether the oracle dropout policy helps or hurts it with more identities. Camera versus oracle retains the coordinate contrast. Image+pointmap is the required utility baseline.

## Fixed data and exposure

`object_transfer_plan.json` selects the previously examined 16 dataset-train objects as training identities and the 16 dataset-validation objects as held identities. None were in the original four-object fits. This is an already inspected diagnostic development pool, not an untouched test set.

For every training object, reserve its two original diagnostic views and choose four other views deterministically from the remaining manifest records. Thus the final three assessments are:

- Fitted views: 16 training objects × four training views = 64 records.
- Held views: the same training objects × two unseen training views = 32 records.
- Held objects: 16 different objects × two views = 32 records.

The protocol validates object separation, sample uniqueness, actual manifest identities, dataset partition and view-independent target hashes. Sixteen batches of four distinct objects form each training cycle. Their order is independently shuffled with a local RNG for each of 64 cycles. Every training record is presented 64 times; every training object receives **256 presentations**. Held examples never enter an optimizer step.

For oracle dropout, each training batch has exactly 32 dropped and 32 normal presentations. The schedule is prespecified and does not consume the global flow RNG. Zero visual tokens preserve their token count, and a hook verifies that shape attention still receives the nonzero surface tokens. Other arms never drop inputs.

This budget has fewer presentations per identity than the old four-object experiment (1000). Dataset, views and seed also differ. Do not interpret old-versus-new loss changes as an isolated causal estimate of dataset size, or treat an underfitted 1024-step endpoint as an inherent limit.

## Measurement and provenance

Cache only frozen visual/VecSetX features. The point projector remains in the training graph. Record actual input/features/target hashes, initial generator/encoder/CA digests, trainable parameter counts and every executed training batch/drop decision. The analyzer requires identical generator initialization, visual inputs, schedule and CA scope across arms; all point encoders must have identical initial parameters, and both oracle arms must have identical features.

At steps 0, 256, 512 and 1024, assess **all visual inputs present**, with four fixed monitor draws. Final assessment uses eight separate common draws and correct/wrong surfaces. Per-object values are captured within the native scalar loss calculation and must reconstruct it. Aggregate separately by fitted views, held views and held objects, averaging within an object before comparing objects. Report paired mean/median changes and counts of improved objects against image+pointmap, oracle versus camera, and dropout versus original oracle.

Monitor bank base is 600000 and final bank base is 700000; draw seed is base + 37 + 100 × (batch index modulo 4) + draw. Both views of the same object share noise/time draws. Views/draws do not constitute independent objects. Raw training losses have different condition mixtures in the dropout arm and are not directly comparable; use all-input assessments.

Save trainable parameters and optimizer state at 256/512/1024. No existing trained checkpoint is used as a warm start. No decoding, rollout, Stage 2 or full-dataset training is included. GPU runtime is unmeasured; use four allocated GPUs concurrently if available, with one process per visible device.

## Branches after this bounded screen

- An aligned arm beats image+pointmap on held identities and the advantage is broad across objects: transferable conditioning is now supported in this diagnostic. Compare camera to that aligned arm before designing an explicit rotation-handling fix. Confirm promising behavior on fresh objects/another seed and appropriate Stage-1 generation checks before full training.
- Camera helps held objects while oracle does not: object-frame alignment is not automatically the best learned interface. Investigate that frame/readout interaction instead of forcing a pose-estimation solution.
- Oracle dropout only improves fitted/held views, but not held identities: reject it as a generalization fix at this scale. Strong wrong-surface sensitivity alone does not rescue that claim.
- Surface arms fit adequately but all lose on held identities: the transfer failure persists beyond four-object specialization. Interface/adaptation and data diversity remain hypotheses; alignment alone is not sufficient evidence for a pose-head solution.
- Fitted loss is still high and decreasing: optimization/exposure is unresolved. Inspect the saved curves and checkpoints before deciding on a bounded continuation. Do not automatically extend or start a full run.

## Paste into an allocated interactive terminal

Sync the investigation folder, including `object_transfer_plan.json`, `object_transfer_protocol.py`, the new runner/analyzer, and the archived original oracle reference. The command uses four concurrent processes when at least four GPUs are visible in the current allocation. Otherwise it runs sequentially on the first visible GPU. It does not override `CUDA_VISIBLE_DEVICES` or access devices outside the allocation. Parallel logs are saved beside the arm directories.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
transfer_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
transfer_root=outputs/conditioning_investigation/object_transfer/manual
visible_gpus=$("$transfer_python" -c 'import torch; print(torch.cuda.device_count())')
test "$visible_gpus" -ge 1
test ! -e "$transfer_root"
mkdir -p "$transfer_root"
if [ "$visible_gpus" -ge 4 ]; then
  transfer_pids=()
  device_index=0
  for model_key in image camera oracle oracle_dropout; do
    "$transfer_python" doc/conditioning_investigation_2026-09-11/fit_object_transfer_gpu.py \
      --model-key "$model_key" --device-index "$device_index" \
      --output-dir "$transfer_root/$model_key" > "$transfer_root/$model_key.log" 2>&1 &
    transfer_pids+=("$!")
    device_index=$((device_index + 1))
  done
  transfer_failed=0
  for process_id in "${transfer_pids[@]}"; do
    if ! wait "$process_id"; then
      transfer_failed=1
    fi
  done
  test "$transfer_failed" -eq 0
else
  for model_key in image camera oracle oracle_dropout; do
    "$transfer_python" doc/conditioning_investigation_2026-09-11/fit_object_transfer_gpu.py \
      --model-key "$model_key" --device-index 0 \
      --output-dir "$transfer_root/$model_key"
  done
fi
"$transfer_python" doc/conditioning_investigation_2026-09-11/analyze_object_transfer.py \
  "$transfer_root" --output "$transfer_root/analysis.json"
)
```

Return each arm's `results.json` and the combined `analysis.json`. Keep checkpoints on the cluster. On failure, return the traceback/log and any partial JSON; existing outputs are not overwritten. The source/interface/data checks must not be relaxed without investigating a failure.

Local validation passed: actual plan sample/object separation and view counts, balanced exposure/dropout schedule, global RNG isolation, Python/shell syntax, synthetic object-level paired effects, and rejection of held-object training and mismatched oracle features. GPU initialization, loss capture, concurrent-device execution and optimization remain untested locally.
