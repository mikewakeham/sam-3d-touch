# Complete the same-policy frame comparison using existing weights

Status: completed and analyzed in [CAMERA_DROPOUT_GEOMETRY_RETURNED_FINDINGS.md](CAMERA_DROPOUT_GEOMETRY_RETURNED_FINDINGS.md), now F12 in the pivotal ledger. Do not rerun. The protocol below is retained as the pre-result specification. Camera dropout fails the same-policy oracle shape reference even after pose adjustment; no training occurred in this control.

## Why this comparison is needed

The completed pose/shape analysis measured original camera, original oracle and dropout oracle. Comparing camera-original with oracle-dropout changes both frame and training policy. The already trained camera-dropout model has native reserved loss .05381 versus oracle-dropout .02273, but the investigation has repeatedly demonstrated that native loss is not a sufficient substitute for generated shape. This control supplies the missing sampled cell under the matched dropout policy.

Relevant findings: F3 (frame burden under original policy), F5/F9 (dropout oracle native/geometry success), F6 (new-object native-objective failure), F11 (pose/shape distinction and sampling gap). It does not repeat the old native comparison, invent a new encoder or use target R at inference.

## Fixed protocol and outputs

- Restore the completed `visual_dropout/.../camera/checkpoint_1000.pt`, found by exact equality to the archived returned report. If moved, use `--fit-dir`. Missing weights do not trigger retraining.
- Retain frozen VecSetX/no-position/full-CA architecture, the original camera point preprocessing, and all visual inputs at assessment. No weights change.
- Same four identities, four fitted and three reserved views, 25 steps, CFG 0, two sampling seeds, matched actual image/pointmap/target metadata. Point features replay the camera reports. Every modality's sampling-noise hash matches the completed oracle probe.
- Decode and hash the actual GT support against the oracle comparison. Replay all correct/wrong native-loss vectors from the saved camera-dropout endpoint. Check actual shape-attention context and unchanged parameters.
- Generate correct and cyclically swapped surface conditions: 28 batched rollouts ×25 steps =700 generator steps, plus 112 native-replay forwards and decoding. Requires one GPU; runtime has not been measured. Four-visible-GPU allocations are unnecessary for this script; it uses visible device 0.
- Save 112 individual metric rows and 28 full NPZ artifacts. Package reports and copied occupancy arrays automatically as `camera_dropout_geometry_bundle.zip`, with source/member hashes. Latents stay in the original NPZs on the cluster for later inspection if needed.

The compact `camera_dropout_sampling_reference.json` was derived from the validated oracle report and actual decoded-target support. Its originating report hash, all seven observation groups, eight per-object/draw noise dictionaries, and target-support content hash were checked locally. Original hashed probes and production sources remain unchanged.

## Interpretation fixed before results

Report fit and reserved views separately, per object and per individual draw. Keep native loss, fixed-frame IoU and one-/two-voxel shape proximity distinct.

1. Fixed-frame reference: retain the existing mean IoU ≥95% and each object-mean ≥90% fitting-control gate. This is not a universal new-object requirement.
2. Pose-independent reference: compare one-voxel precision and recall to oracle-dropout under identical analysis. A proper rigid transform or identity is a possible witness; retain the candidate with greater minimum precision/recall among the measured identity and registered transforms, applying the same rule to the oracle reference and wrong surfaces. Require each reserved object-average precision and recall ≥95% and no more than two percentage points below its reference. Report raw and RMS-selected registered results separately too; do not hide the previous registration objective's F-score tradeoff or claim a globally optimal transform.
3. Surface-dependence control: report paired correct/wrong differences and all object signs. A fitting-control success with little wrong-surface penalty does not establish dependence on supplied surface identity. As a strong-dependence screen, require a reserved mean F-score penalty ≥10 percentage points for each object under the same pose-analysis rule; this engineering threshold is not a generalization claim. Full numerical values remain primary, not only gate booleans.

Branches:

- **Camera dropout reaches the pose-independent reference with surface dependence:** precise input canonicalization is not shown necessary for this fitted-identity task. Do not automatically add pose estimation or synthetic jitter. Next assess generated surface utility on new identities with existing weights; the old native-loss failure remains real but did not sample shape.
- **Camera dropout remains worse after pose adjustment:** a frame-related sampled-shape penalty remains under the same improved policy. Select a shared observable target-frame intervention if original asset axes are unnecessary, or a local robustness intervention if the canonical interface is required. This result alone still does not choose encoder replacement/full finetuning.
- **Frame-free shape succeeds but fixed-frame IoU fails:** output-frame requirements decide whether the residual is a task requirement or merely a canonical-label burden. Preserve that distinction.
- **Mixed/rare failures or weak surface dependence:** inspect the reported cases and confidence limits; do not turn one aggregate improvement into a fix.

No full training is justified by this four-object comparison alone. It completes a targeted coordinate control; fresh-object confirmation and eventual sparse-location constraints remain separate requirements.

## Interactive command

Sync the two new files `probe_camera_dropout_geometry_gpu.py` and `camera_dropout_sampling_reference.json` into the investigation folder. The helper scripts used by the previous probes/bundle command must remain there. Then paste into an interactive GPU allocation:

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
probe_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
"$probe_python" doc/conditioning_investigation_2026-09-11/probe_camera_dropout_geometry_gpu.py \
  --output-dir outputs/conditioning_investigation/camera_dropout_geometry/manual
)
```

Return `outputs/conditioning_investigation/camera_dropout_geometry/manual/camera_dropout_geometry_bundle.zip`. No separate packaging command is needed. If a checkpoint/replay check fails, return the traceback; do not retrain, overwrite existing outputs or relax tolerances. For an already used output path, choose a new suffix.

Local checks cover Python syntax, resolution of imported local helpers, exact reference/observation/noise pairing and actual target-support digest. They do not claim local CUDA execution. CPU pose analysis will reuse the validated registration controls and report the identity/registration selection sensitivity explicitly.
