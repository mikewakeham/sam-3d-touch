**Superseded by the approved source integration:** Use [SOURCE_INTEGRATION.md](SOURCE_INTEGRATION.md) and the three normal four-GPU job scripts. The older standalone trainer/preflight/launcher below are not the current commands. Per-sample dropout replaces whole-batch dropout, image baselines are reused, and full cross-attention is now the CLI default.

# Oracle upper bound first

Status: prepared locally on 13 September; GPU preflight and full-data runs **not executed**. This is the active user-selected branch. Camera-target rotation augmentation, input canonicalization and inference pose recovery are paused. No new four-object fitting experiment is requested.

## Material Passport

- Task: establish a useful aligned full-surface Stage-1 upper bound before removing the oracle.
- Evidence: PIVOTAL_FINDINGS F3, F6–F9, F12 and F16; existing returned reports and local source.
- New work: training/evaluation implementation and explicit comparison plan, not new empirical findings.
- Access: local CPU only; user executes GPU commands on the cluster. No external data upload.

## Why this is the next test

F9 already established accurate reconstruction of four fitted objects at previously unseen views with oracle alignment plus visual dropout: 97.17% mean Stage-1 occupancy IoU against decoded target support, versus 2.49% with wrong surfaces. It is unnecessary to re-prove this with another tiny fit.

F6/F7 prevent calling the overall oracle route solved: four-/16-object training did not establish new-object utility, and constant surfaces could match apparent gains. Those small-data failures do not determine whether diverse training can learn a reusable mapping. The full-data experiment itself is needed to answer that question; success on an additional tiny transfer task is not a prerequisite.

Before spending that training budget, run a short implementation check. It loads the existing successful oracle/dropout checkpoint, checks historical inputs and losses on one fitted and one reserved view group, and compares the **new live training forward/backward** with the historical cached-feature/direct-generator formulation. It checks both visual-present and visual-zero states and constant-feature invariance. No Adam steps, target regeneration, new representation, or Stage-2 model. It also returns train/validation counts and limited timing measurements.

Historical BF16 loss replay uses rtol=.001, atol=1e-6 (consistent with the earlier cross-device replay correction). Live/direct losses use rtol=1e-5, atol=1e-7; gradients use rtol=1e-4, atol=1e-7. Input/feature/source hashes and parameter identity are exact. A failure is an implementation/provenance problem to inspect; do not loosen gates automatically. A pass confirms the port, not full-data conditioning success.

## Fixed full-data comparison

| Arm | Surface stream | Visual dropout | Purpose |
| --- | --- | --- | --- |
| oracle_dropout | Each example's observed full surface, transformed to its target object frame | 50% whole-batch token zeroing | Candidate aligned upper bound |
| constant_dropout | Frozen VecSetX features of one fixed training example for every input; projector remains trainable | Same schedule | Distinguish sample-specific geometry from a generic learned context |
| camera_dropout | Each example's observed full surface in existing camera preprocessing | Same schedule | Attribute a residual performance difference to frame treatment under matched training |
| image | Image + pointmap; no surface stream | None | Practical pretrained visual baseline with matched shape cross-attention adaptation |

The image arm retains its visuals on every update, so it is a practical baseline, not a dropout-matched causal control. The two other surface arms are the matched controls. This experiment does not isolate the causal effect of dropout itself; no-dropout oracle is not an additional required arm. Earlier oracle-only full runs are context, not automatically matched controls.

All arms start from pretrained SAM3D anew, with the same initial shape cross-attention. All surface arms start with the same adapter weights, frozen VecSetX, no position projection, original fixed asset-frame target latents, full shape cross-attention + its input norm only. Learning rates are 1e-4 for the surface adapter, 1e-5 for shape cross-attention; AdamW weight decay 0, gradient clipping 1, bf16. No broader shape-generator unfreezing, rotation augmentation, GT voxel conditioning, or sparse-touch run is added.

The constant features are encoded from one actual **training observation**, never a target latent or validation object. The raw features remain fixed, while the same projection/embedding parameters train. Wrong surfaces for inference controls are cyclic shifts between distinct objects within an evaluation batch. The constant arm is evaluated with its constant stream, not silently switched to real surfaces.

Default budget: **20 epochs**, matching the existing job scripts' dataset passes, one GPU and batch 4 per arm. This is not an exposure-matched replication of older 4-GPU/global-batch-16 runs: the new batch produces more optimizer updates per epoch. All new arms are matched to each other. Changing available GPUs changes how many arms run concurrently, never their batch or optimizer schedule. The launcher prioritizes oracle/constant on two GPUs. Train ETA is printed in individual logs; preflight timing is only an estimate, not measured overnight runtime.

Datasets use the existing full-surface manifest and object-disjoint train/val split; overlaps or duplicate sample IDs are rejected. Nothing is moved out of training. The evaluation selection is fixed from metadata before outcomes: up to 32 objects per split, first/last view per object, same IDs in all arms. Validation is a development set, not a pristine final test. All training records are used; this is not another 16-/32-object fit.

## Measurements and reference endpoints

At epochs **1, 10, 20** (and the final epoch for a shortened run):

- Native shape flow loss, two paired fresh noise/time draws, correct surfaces and three wrong-object choices. Per-object values retained; image/constant have their actual sole input condition.
- Stage-1 samples only: 25 integration steps, CFG 0, two paired noise draws, all visuals present, correct versus one wrong surface. Save decoded target and prediction supports. This decoder is a Stage-1 measurement instrument, not Stage-2 training/evaluation.
- Raw occupancy IoU and precision/recall/F-score within two original-grid voxels. The reference is each object's **decoded target latent**, not an arbitrary loss decrease. Object means average views/draws before aggregation.
- The analyzer verifies matched source/data/config, initial weights, step, selected observations/targets and actual sampling noise. Object-bootstrap intervals describe development-object variability at one training seed; they do not measure training-seed uncertainty.

Predeclared strict raw upper-bound screen: mean object IoU at least .95 and object-IoU 10th percentile at least .90. This is an operational near-reference goal, not a theorem about perceptual equality. Report per-object values and the number meeting .95 precision AND recall as well. If raw alignment misses this endpoint, **do not call shape reconstruction failed without inspecting proper-rigid geometry from saved supports**. Correct shape in another pose is acceptable to the user.

For useful generalization, the held-object correct surface should outperform image and constant controls, and correct should outperform wrong surfaces. Examine paired object differences/intervals, not only means. A reference-level reconstruction plus reliable geometry-specific improvements is the desired strong endpoint; an improvement far short of the reference is progress, not a solved upper bound. If the visual baseline itself reaches the reference, surface utility needs harder examples rather than demanding a meaningless improvement over a ceiling.

## What each result changes

1. **Oracle reaches reference on fitted AND held identities, beats the controls, camera remains worse:** establishes an aligned working upper bound and a consequential frame-treatment burden under this training recipe. Then investigate observable alignment/recovery or learning without privileged alignment. Camera/oracle includes normalization as well as rotation; F3 supports rotation's importance, but this comparison alone does not isolate rotation or prove it was the only historical cause.
2. **Oracle and camera both reach useful held-object fidelity:** establish a working recipe without claiming alignment was necessary. Broader data/training may have removed the small-set gap.
3. **Oracle fits accurately but does not transfer or constant matches it:** oracle is still not a transferable geometric solution. Stay on the aligned route; inspect whether increasing diversity/optimization, visual dependence, or adapter scope limits it. Do not move straight to pose recovery.
4. **Oracle still cannot fit the larger training set:** use intermediate native and sampled trajectories to distinguish underoptimization from failure to exploit geometry. Broader shape adaptation is a conditional next intervention, not automatically mixed into this first run. F16 proves a different shared-target recipe can overfit; it does not prove the oracle needs that scope.
5. **Early apparent improvement vanishes later:** retain fixed epoch endpoints; do not call whichever checkpoint happens to win a confirmed solution. A subsequent selection/replication protocol is needed.

A single full-data run is a strong practical checkpoint, not final proof across seeds. Replicate the winning setting and a fresh identity split before claiming a general solution. Do not require those replicas before inspecting the first full-data results.

## Run the short preflight now

Sync the new `doc/conditioning_investigation_2026-09-11/oracle_upper_bound/` directory. Historical investigation scripts/reports and the successful oracle checkpoint must remain available on the cluster; the locator checks for an exact matching saved run. It will not retrain or select an arbitrary checkpoint.

Paste in the existing interactive allocation:

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 LIDRA_SKIP_INIT=true
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python \
  doc/conditioning_investigation_2026-09-11/oracle_upper_bound/run_gpu.py \
  --preflight \
  --output-dir outputs/conditioning_investigation/oracle_upper_bound/preflight
)
```

Return `outputs/conditioning_investigation/oracle_upper_bound/preflight/preflight.json`. If the existing checkpoint was moved, add `--oracle-fit-dir /actual/path/to/the/oracle_dropout_run` (directory containing its original `results.json` and `checkpoint_1000.pt`). Do not rerun the old fit. The preflight does not need NPZ files returned.

## Full-data commands, prepared for after preflight review

These do **not** execute as part of the preflight. Review its counts/timings first. Four visible GPUs run all arms together; change only `--gpus 4` to `--gpus 2` or `--gpus 1` for waves. This is independent-arm concurrency, **not torchrun/DDP**. Logs appear in `full/ARM.log`.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 LIDRA_SKIP_INIT=true
upper_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
upper_scripts=doc/conditioning_investigation_2026-09-11/oracle_upper_bound
upper_root=outputs/conditioning_investigation/oracle_upper_bound
"$upper_python" "$upper_scripts/launch.py" \
  --preflight-report "$upper_root/preflight/preflight.json" \
  --output-dir "$upper_root/full" --gpus 4 --epochs 20
"$upper_python" "$upper_scripts/analyze.py" --root "$upper_root/full" --epoch 20
"$upper_python" "$upper_scripts/bundle.py" --root "$upper_root/full"
)
```

Return `full/oracle_upper_bound_bundle.zip` (reports plus Stage-1 supports; excludes large model/Adam tensors). Epoch-1 reports can be reviewed while later epochs run; do not restart arms to get them. `last.pt` is saved at every epoch boundary and the assessed epoch checkpoints are retained. Existing output directories are rejected to prevent mixing runs. A failed process is not automatically retried, and the launcher waits for its already-running paired arm before stopping further waves.

## Local verification and remaining limit

Seven CPU checks pass: scheduling/exposure/resume determinism, control settings, disjoint split/selection, 1/2/4-GPU launch commands, Stage-1 boundary, mocked dropout/constant behavior, and paired-summary/noise-mismatch rejection. Python source compilation passes. Historical `train.py`, `dataloader.py` and touch encoder were not edited. GPU loading, gradient parity, training and sampled assessment remain unexecuted locally; the preflight is required evidence before the longer run.
