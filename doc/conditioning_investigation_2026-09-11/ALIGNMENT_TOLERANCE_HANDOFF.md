# Existing-checkpoint alignment tolerance — 12 September 2026

Status: implemented and checked locally with Python syntax, pure-math rotation checks and synthetic complete/incomplete reports. GPU execution is pending; no numerical outcome is claimed. Production training and VecSetX are unchanged.

## Why this is the next coordinate experiment

Starting evidence: pivotal findings F2–F5 establish that four objects can be reconstructed accurately, rotation causes a measured conditioning burden, and visual dropout nearly closes the aligned model's native-loss gap across views. F6–F8 show that neither alignment nor dropout has established useful transfer to new identities.

Two questions remain before prescribing a rotation estimator: does the *dropout checkpoint itself* reconstruct its target accurately across views, and how much residual orientation error can it tolerate? The first prevents using a low-loss but inaccurate model as the reference for the second. This is a Stage-1 check of the coordinate interface; it uses no Stage 2 or source-mesh CD.

The intervention rotates already oracle-aligned full-surface points by ±5°, ±15° and ±30° about each axis, before the existing normalization and VecSetX encoder. Visual inputs, target, weights and sampling noise remain fixed. Consequently the response includes the existing bounding-box/radius normalization's response to rotation: it represents an imperfect pose correction passed through the actual input path, not a new rotation-versus-normalization factorial.

The reference is decoded GT Stage-1 occupancy, whose agreement with itself is IoU 1. Original-oracle and dropout-oracle exact-alignment results distinguish the effect of dropout. Wrong-object surfaces with the dropout checkpoint check whether accurate outputs depend on the supplied object's surface in this fitted-identity setting. A wrong-surface penalty here cannot establish new-object utility.

## Scope and criteria fixed before results

- The same four fitted identities, four fitted views and three reserved views, two common sampling seeds, 25 steps, CFG 0. All visual inputs are present.
- Reference gate: dropout/exact alignment reserved-view mean IoU at least .95, with each object's view/seed-average at least .90.
- Tolerance gate: for every object and every sampled signed axis at an angle, its mean IoU loses no more than .02 versus exact alignment and remains at least .90. The reference gate must also pass.
- These are engineering criteria for a near-target fitting control, not universal quality standards or formal statistical equivalence. Report the full values even when a gate fails. Object averages can hide individual view failures; raw rows remain available. There are four independent identities, not 24 independent examples per treatment. Passing ±15° on coordinate axes does not establish tolerance to every 15° rotation or real estimator errors.
- Stored predicted/target occupancies and latents permit follow-up inspection without rerunning the generator. No registration is applied to erase a coordinate error.

The runner checks actual input and feature hashes, exact historical native-loss replay, checkpoint contents and parameter hashes, proper rotations and inverse recovery, the actual shape-attention context, and unchanged final weights. The analyzer additionally requires original-oracle sampled-result replay, complete coverage and common per-object sampling noise. A failed replay stops interpretation; do not silently relax tolerances or retrain to bypass it.

## Decisions after results

1. **Aligned dropout meets the target reference, wrong surfaces substantially degrade it:** we have a target-referenced same-identity endpoint for the coordinate interface. Rotation curves give a measured pose-accuracy requirement. A subsequent correction must approach this endpoint using observable inputs, then separately demonstrate benefit on new identities. This is not sufficient to authorize full training.
2. **Aligned dropout remains inaccurate despite its closed native-loss gap:** the historical dropout finding remains true but did not solve reconstruction. Inspect its saved support/latent errors against the existing accurate single-view fit; do not build a pose estimator on the claim that alignment already solves reconstruction.
3. **Accurate reference, even ±5° is damaging:** small residual pose errors are an actual deployment concern. Compare achievable pose accuracy with this measured requirement; a targeted correction or controlled training with residual rotations may be worth a short test. Failure of this checkpoint does not prove an architectural impossibility.
4. **Accurate reference, ±15° or ±30° remains close:** exact pose recovery is unnecessary within these sampled cases. Prioritize a coarse observable alignment and test its actual residuals through the conditioner. A sweep alone neither supplies that estimator nor resolves F6's new-identity failure.
5. **Wrong surfaces also reconstruct well:** accurate output has not established geometry-specific reconstruction; retain the adaptation/memorization alternative from F7 before assigning meaning to tolerance curves.

Output-coordinate requirements are separately awaiting the user's answer: camera/robot-frame reconstruction versus original asset axes. The probe does not change that contract. No camera-frame target training, latent-channel rotation or production pose estimator is implemented.

An overnight full-training candidate must first show useful correct-surface benefit on held objects against an image baseline and appropriate wrong/constant controls, with Stage-1 target-referenced checks. The failed tiny-transfer candidates are not silently promoted because this coordinate diagnostic passes. If no candidate earns that evidence today, recommend no full run rather than fill the overnight slot.

## Interactive execution

Sync the current repository/doc files to the cluster first. Use the existing sam3d environment and checkpoints; no new fit is requested. `run_alignment_tolerance_interactive.sh` contains the exact shell contents below. It uses up to four already-visible GPUs, in waves if fewer are visible. It does not request a scheduler allocation.

There are 21 conditions × seven four-object batches × two sampling seeds × 25 generator steps = 7,350 batched sampling steps across the four shards, plus replay and decoding. Four GPUs parallelize the shards. Runtime has not been measured; this is a bounded inference probe, not a claim of an instantaneous check.

The runner locates an exact matching dropout report under `outputs/conditioning_investigation/visual_dropout` and uses the recorded original multi-view fit directory. If either existing checkpoint was moved, pass `--dropout-fit-dir` and/or `--original-fit-dir` to the Python invocation. Do not retrain. If the output directory already exists, choose a new suffix; do not delete prior results.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
probe_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
probe_root=outputs/conditioning_investigation/alignment_tolerance/manual
test ! -e "$probe_root"
mkdir -p "$probe_root"
probe_gpu_count=$("$probe_python" -c 'import torch; print(min(4, torch.cuda.device_count()))')
test "$probe_gpu_count" -gt 0
for ((probe_base=0; probe_base<4; probe_base+=probe_gpu_count)); do
  probe_pids=()
  for ((probe_offset=0; probe_offset<probe_gpu_count && probe_base+probe_offset<4; probe_offset++)); do
    probe_shard=$((probe_base+probe_offset))
    printf 'Starting shard %s on visible GPU %s; log: %s/%s.log\n' "$probe_shard" "$probe_offset" "$probe_root" "$probe_shard"
    "$probe_python" doc/conditioning_investigation_2026-09-11/probe_alignment_tolerance_gpu.py \
      --shard "$probe_shard" --device-index "$probe_offset" \
      --output-dir "$probe_root/$probe_shard" \
      > "$probe_root/$probe_shard.log" 2>&1 &
    probe_pids+=("$!")
  done
  probe_failed=0
  for probe_pid in "${probe_pids[@]}"; do
    wait "$probe_pid" || probe_failed=1
  done
  if test "$probe_failed" -ne 0; then
    tail -n 30 "$probe_root"/*.log
    exit 1
  fi
done
"$probe_python" doc/conditioning_investigation_2026-09-11/analyze_alignment_tolerance.py \
  "$probe_root" --output "$probe_root/analysis.json"
)
```

Return `analysis.json` and `{0,1,2,3}/results.json`; NPZ files can stay on the cluster initially. If a check fails, return the traceback/log instead of continuing. GPU results are the next dependency.
