# Short coordinate intervention: camera-oriented target labels

Status: implemented, local coordinate/source checks passed; GPU execution pending. The user paused overnight implementation and requested continuation of the short coordinate branch. No overnight trainer or production source was changed. This is an experimental target contract, not a commitment that the final product abandons asset-frame output.

**Returned preflight failure:** the first cluster attempt completed target preparation groups 0 and 1, then stopped within group 2 with `Expected proper rigid camera transform`. No flow-model training occurred. The exception omitted the sample/matrix/residual; the failing camera is not locally available. Acceptance tolerances are unchanged pending the CPU-only `audit_shared_orientation_cameras.py`, which records the exact 28 selected matrices, individual check outcomes, singular values, determinants, homogeneous rows and file hashes. Existing five local cameras pass with orthogonality residuals about 5e-8–3.5e-7. Those do not establish the failing matrix's validity. Do not infer a new coordinate bug or relax the gate from this traceback alone.

**Failure resolved from returned matrices:** all 28 matrices were supplied and replayed locally; only `146b6c1f…_015` failed the old orthogonality check (1.92064e-6). Its determinant is 1.000000158 and maximum singular-value deviation from one is 1.02605e-6. Relative to its nearest proper rotation, the maximum displacement of any point in the original unit cube is bounded by 8.88585e-7 object units, or 0.00005687 of a 64-grid voxel. This explains the validation failure as an overly strict new guard, not a newly established cause of the conditioning failures.

`shared_orientation_protocol.py` now validates positive determinant, homogeneous row, and maximum singular-value deviation ≤32 float32 epsilons (3.8147e-6). The bound permits less than 0.00022 voxel of geometric distortion on the original cube. It retains the recorded matrix exactly; no SVD projection, source camera change or input-feature change occurs. All 28 supplied matrices pass; inverse/normalization and displacement-bound checks pass, and reflection, scale, singular, nonfinite, shear and projective controls are rejected. Evidence: `shared_orientation_camera_audit/returned.json` and `validation.json`. No target-VAE or training acceptance gate changed.

Sync the updated protocol helper and rerun the original command with a fresh output directory, e.g. `outputs/conditioning_investigation/shared_orientation/manual_rigid_check_fix`. There is no trained checkpoint to resume from this failed attempt. The GPU target-quality/training result remains pending.

**Second preflight correction (13 September):** the corrected run passed camera validation, target generation, exact initialization hash and all seven recorded input/VecSetX feature hashes. It then failed the bitwise initial-loss replay: maximum absolute difference 3.51667404e-5, relative 9.47415049e-5 (0.009474%). No training occurred. The specific numerical source is not identified by the traceback. The new runner used one autocast scope and precomputed projected tokens across draws, whereas the historical comparator recomputed under a per-draw autocast scope; those scopes now match. This is a candidate source of numerical differences, not a confirmed explanation.

The loss guard now requires numerical agreement with rtol=1e-3 (0.1%), atol=1e-6; it records actual/expected vectors, errors and bitwise-equality status and saves them before checking. This permits the observed small BF16 discrepancy while retaining all exact source/parameter/input-hash checks. Target quality and scientific success gates are unchanged. A CPU check on the rounded displayed values passes and an injected 1% difference fails (`shared_orientation_replay_check.json`); no local GPU rerun is claimed. Sync the updated `fit_shared_orientation_gpu.py`, retain the already updated protocol helper, and use a new output suffix such as `manual_replay_fix`. Do not describe these tooling fixes as coordinate findings or successful training.

## Starting evidence and question

F1: inspected bookkeeping is consistent and the target is spatial/orientation-dependent. F3: input rotation adds a finite-budget learning burden. F9: exact oracle alignment plus dropout permits accurate reserved-view reconstruction on four fitted identities. F11: output pose correction is not a general rescue, and sub-5° precision is not universally required. F12: with dropout policy matched, camera inputs still fail badly; a bathtub view even produces the fitted shield. Stop repeating that comparison.

Question: **Does training the spatial target in the observed camera orientation substantially remove this finite-task failure, while keeping the existing camera VecSetX input unchanged?** This is an intervention on the supervised coordinate contract, not an encoder or attention redesign.

One new 1000-update arm uses the exact camera/dropout initialization, four identities, four fitted/three reserved views, learning rates, optimized parameters, feature tensors, visual tensors and balanced dropout schedule. The historical camera/dropout arm from F12 is the fixed-asset-target control. Historical oracle/dropout is the successful aligned reference. Both are reused; no baseline fit is repeated. The new target is per-view, so unlike the original task it varies across an object's views; a negative result may reflect this harder output distribution and is not an impossibility proof.

## Target construction and the clipping issue

For normalized object mesh vertices P and the already audited SAM camera rotation R:

```
rotated = P @ R.T
center = (rotated.min(axis=0) + rotated.max(axis=0)) / 2
extent = max(rotated.max(axis=0) - rotated.min(axis=0))
camera_target_mesh = (rotated - center) / extent
target_latent = frozen_stage1_encoder(voxelize(camera_target_mesh))
```

This retains camera axes and applies the same centered unit-bounding-box convention as existing target generation. Translation cancels in centered shape supervision; this is not metric camera placement. The current voxelizer clips vertices outside its cube, so merely rotating the original mesh without normalization would silently deform labels. The new path verifies bounds before voxelization; only the original 1e-6 boundary inset remains.

The target changes both orientation and its consequent bounding-box normalization; this is a target-convention intervention, not a pure rotation factorial. It does not assume latent equivariance: actual mesh geometry is rotated and re-encoded. Original stored latents are independently regenerated first. All new latents/meshes remain in this experiment's output folder; manifests, checkpoints and production data are not overwritten.

**Input boundary:** all camera features, images and pointmaps must hash-match history. Full meshes/extrinsics enter target construction only. The generator receives the same visual/touch interface; no transform, target support, or target-derived feature is added to conditioning. This retains VecSetX. Sparse touch still needs its global contact placement/scale carried appropriately; normalizing a tiny patch cannot define a full-object frame. This full-surface test does not claim that future interface is implemented.

## Automatic checks before any training

1. Historical source/config hashes; original target regeneration; exact initialization parameter hash; all seven original camera feature/input hashes; eight numerically matching initial native-loss replay values (rtol=1e-3, atol=1e-6, with exact vectors/errors retained).
2. Proper camera rotation, reversible affine transformation, unit-box target bounds. Local checks already cover 25 synthetic rotations, five saved surface/camera pairs and three malformed transforms.
3. Decode every new target through the existing Stage-1 decoder. Each target must have at least 95% IoU to its physically voxelized mesh.
4. Map decoded new targets back with the **known inverse label transform**, and compare with the original decoded target in original object units. Every target must have at least 95% bidirectional proximity within 1/64 original-object units. This prevents silently accepting label damage or apparent improvements from a relaxed spatial tolerance.

If a check fails, no optimizer update occurs. Return `results.partial.json` and the traceback; inspect the failure instead of weakening the gate or extending training. The script saves new target caches before this gate, so the failure is reviewable.

## Training and measurements

- Frozen VecSetX; projector and full shape cross-attention adaptation; same rates 1e-4/1e-5 and clipping 1.0. Exact prior 1000-step, group-balanced 50% whole-visual-zeroing schedule. Actual shape-attention context checked for visual-present/visual-zero cases. All visuals present for primary assessment.
- Fresh native losses on correct/wrong surfaces at steps 0/300/1000; checkpoint at 300 and 1000. Native loss across changed targets is descriptive, not directly a percentage of geometry learned.
- Final samples: seven groups × two seeds × correct/wrong surfaces × four objects =112 outputs, CFG0/25 steps. All per-modality noise hashes match the prior reference. Weights checked unchanged during sampling.
- Save predicted and decoded-target support, original decoded target support, physical new target support, full affine label transforms, reports and member/source hashes in `shared_orientation_bundle.zip`. Full latents remain in cluster NPZs.
- CPU follow-up reports native-target IoU, and shape proximity after undoing the known label transform in **original object units**, with proper rigid pose adjustment separately. Undoing a known label transformation is not a fitted scaling correction or a deployed oracle input. Do not compare raw one-voxel scores measured in differently scaled cubes as if they use identical physical tolerance.

## Decision rules fixed before results

Full numerical per-object/per-view results remain primary. The engineering gates are not population equivalence tests; all four identities were trained, views/draws are correlated, and there is one training seed.

- **Accurate shared-frame endpoint:** reserved mean native-target IoU ≥95% and each object mean ≥90%. In common original units, each object's precision and recall must be ≥95% and at most two percentage points below its own decoded-new-target positive control. Preserve comparison with the original oracle reference and camera baseline rather than hiding differences in VAE/discretization floors.
- **Surface dependence:** correct minus wrong mean one-voxel F-score ≥10 pp for each reserved object under the same common-unit scoring procedure. Report raw and measured identity/rigid witness results separately; use maximum minimum precision/recall among those candidates, as in F12. Wrong surfaces contradict visuals; passing is a fitted-surface dependence screen, not new-object utility.
- **Pass both:** changing the supervised frame can remove the measured finite-task failure without replacing VecSetX or adding conditioning information. Next test novel-object surface utility against a matched image baseline; another four-object fitting success is not a general conditioning solution.
- **Fit succeeds, reserved views fail:** the failure remains a view/orientation transfer problem even when labels share input orientation. Do not call longer training a fix by default; distinguish orientation interpolation and pretrained spatial adaptation before expanding.
- **Fit itself fails despite target gates:** this new target distribution may be poorly accommodated by the current adaptation scope at this budget. A matched scope intervention becomes relevant, but one finite run does not establish an architectural limitation.
- **Shape succeeds only after pose adjustment:** the required final output frame still matters. Preserve that distinction rather than declaring fixed-frame success.
- **Correct reconstruction with weak surface dependence:** could be visual fitting/identity retrieval; do not promote to full training.

The eventual asset-frame requirement remains open, but it need not block this diagnostic comparison. A positive shared-frame result offers a practical route if camera/robot output is acceptable; an asset-frame requirement would still need observable pose recovery.

## Interactive command

Sync only the new `fit_shared_orientation_gpu.py` and `shared_orientation_protocol.py` into the existing investigation directory; historical helpers/reports used by previous jobs must remain there. One visible GPU is sufficient. Runtime is unmeasured: 1000 backward updates, 700 sampling forwards, 336 fresh native-assessment forwards plus eight replay forwards, and target encoding/decoding. No full-dataset training.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
probe_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
"$probe_python" doc/conditioning_investigation_2026-09-11/fit_shared_orientation_gpu.py \
  --output-dir outputs/conditioning_investigation/shared_orientation/manual
)
```

Return `outputs/conditioning_investigation/shared_orientation/manual/shared_orientation_bundle.zip`. If it stops, return the traceback and `results.partial.json`. Use a new output suffix if this directory already exists. The target-VAE checkpoint defaults to the existing `checkpoints/hf/ss_encoder.ckpt`; no new download is requested.

Local validation: Python syntax, helper availability, actual archived reference/schedule assertions, affine controls, saved coordinate comparisons, and shell syntax. No local PyTorch/CUDA execution is claimed.
