# Matched orientation coverage after F15

Status: completed and verified; do not rerun. See [ORIENTATION_COVERAGE_RETURNED_FINDINGS.md](ORIENTATION_COVERAGE_RETURNED_FINDINGS.md). The control now fits nearly exactly; the short augmented arm fails to fit sampled added orientations and does not resolve transfer. The command below is historical. Stage1, frozen VecSetX, full-surface upper bound, no production changes.

## Question and matched treatment

Can additional orientation coverage improve the geometry path's reserved-view transfer, beyond equal extra exposure to the original orientations? F15 showed strong fitted point dependence but poor reserved reconstruction even without visual input. This test directly changes orientation coverage while holding object identities, point samples, target convention and adaptation scope fixed.

Both arms resume the completed broader shape-path `checkpoint_2000.pt`, including Adam state, for1000 additional updates. The three existing learning rates remain1e-4/1e-5/1e-5. No further generator parameters or VecSetX weights are unfrozen.

| Update type | Control | Augmented |
|---|---|---|
|500 visual-present updates|Original coherent points, visuals and targets|Same original inputs/targets|
|500 visual-dropped updates|Original points and targets|Jointly rotated observed points and target labels|

Group order, noise seeds and dropout schedule are identical. Each fitted view group receives125 updates of each type. For the treatment's dropped updates, each of23 nonidentity proper cube rotations occurs5 or6 times per group. The identity rotation is retained as the original reference, giving24 bank rotations in total. Model inputs never include a rotation-index token or GT frame transform.

Cube rotations are signed axis permutations with determinant+1. They preserve the voxel grid exactly and commute with the current bbox-center/max-radius point normalization in exact arithmetic. The test rotates the **same already prepared observed points**, preserving point selection and masks, then re-encodes them through frozen VecSetX. It rotates physical training occupancy labels and re-encodes them through the frozen Stage1 VAE. It does not rotate latent channels, infer normals from GT, replace the conditioner with voxels, or rotate geometry while leaving unmasked contradictory visuals.

This is a finite orientation-coverage test, not a guarantee of arbitrary-angle equivariance. Natural reserved views also differ in point samples and normalization. A negative result does not isolate all those factors or prove that more diverse objects cannot help. Eventually arbitrary rotations require reconsidering normalization and target voxelization; this first exact-grid test avoids introducing those changes simultaneously.

## Preparation, checks and outputs

The launcher first prepares one shared bank on one GPU, then starts both arms concurrently if two GPUs are visible, or sequentially if one is visible. It uses at most two GPUs from the existing allocation.

Preparation generates384 target labels:4 identities ×4 fitted views ×24 rotations. All28 original samples across seven batches are verified, but only the16 fitted observations enter the training bank. Original target encoding is replayed; identity features/target caches are checked. All new targets must pass the established pre-training95% VAE-IoU and95% common-unit two-voxel P/R gates. A failure stops before either arm trains and retains a partial report. Target-quality checks are not the trained-model success criteria.

Each arm verifies source/input/feature/cache hashes, the complete starting parameter digest, trainable groups and restored Adam contents. Its full visual-present and visual-zero native assessment must replay F15 within the established BF16 rtol0.001/atol0.000001. Checkpoint and assessment saves occur at total steps2300 and3000. Frozen parameters must remain unchanged.

At step3000, each arm returns:

- The same224 natural outputs as F15:7 view groups ×visual present/zero ×correct/wrong surfaces ×2 paired draws ×4 objects. Sampling remains25 steps, CFG0, same noise.
-32 augmented-fit diagnostic outputs:base group0 at rotation indices1,7,13,19, visual-zero/correct surfaces, two draws. These are fitted bank conditions, not held-out tests. They distinguish failure to learn the added conditions from failure to transfer beyond them; they do not cover the entire augmented bank.
- All384 decoded target-bank supports and physical rotated labels, so preflight geometry can be independently checked locally, not only read from scalar reports.

Native IoU uses the exact respective target. Common geometry uses the unchanged two-original-voxel tolerance, raw and established rigid witness separately. The common target reference remains100% P/R; the strict trained-model screens remain95% overall native IoU/90% each-object mean,98% each-object common P/R, individual95% P/R counts, and10pp each-object correct-minus-wrong surface advantage. Failure to pass does not erase useful partial treatment effects, but a mean improvement alone is not a resolved coordinate problem.

## Branches after the result

1. **Treatment improves reserved geometry and surface dependence beyond matched exposure, approaching/passing target references:** orientation coverage is an actionable training bottleneck. If improvements transfer to visual-present inputs too, this is a candidate for a larger controlled pilot. If only visual-zero improves, multimodal integration remains a separate obstruction.
2. **Treatment learns the added orientations but natural reserved views remain poor:** the finite bank is insufficient for broader transfer. Consider continuous orientation coverage, point-sampling/normalization differences and object diversity before declaring a structural limitation. Do not automatically add another1000 updates.
3. **Treatment cannot fit the added orientations:** broadening the target/feature distribution challenges the learned mapping even on fitted identities. Inspect the target checks and training/augmented-fit diagnostics before deciding between optimization, more diverse data or explicit input alignment. Do not blame VecSetX by default.
4. **Control improves equally:** extra exposure, rather than this orientation-coverage intervention, explains the improvement.

Four-object held-object superiority remains **unnecessary as a universal prerequisite for a larger diagnostic pilot**. A pilot may test object-diversity/sample-complexity directly; it must not be presented as an already established fix. The original full-dataset goal and eventual sparse structured touch remain unresolved. Rotating observed patches and their future normals/vector-valued attributes has a natural extension; GT labels remain training-only.

## Interactive command

Sync these three new runtime files under `doc/conditioning_investigation_2026-09-11/`:

- `orientation_coverage_protocol.py`
- `orientation_coverage_gpu.py`
- `run_orientation_coverage_pair.py`

The existing scope/visual-probe code, outputs and checkpoint are dependencies. Paste this into the interactive allocation:

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python \
  doc/conditioning_investigation_2026-09-11/run_orientation_coverage_pair.py \
  --fit-dir outputs/conditioning_investigation/shared_orientation_scope/manual/shape_path \
  --visual-probe-dir outputs/conditioning_investigation/shared_orientation_visuals/manual \
  --output-dir outputs/conditioning_investigation/orientation_coverage/manual
)
```

Return these two files:

```text
outputs/conditioning_investigation/orientation_coverage/manual/control/coverage_control_bundle.zip
outputs/conditioning_investigation/orientation_coverage/manual/augmented/coverage_augmented_bundle.zip
```

The prior broader1000-update phase took about13 minutes on the reported H100, including its intermediate assessments/checkpoints. This job has more assessments, a new bank preparation and final sampling, so13 minutes is not an end-to-end estimate. Two GPUs parallelize the arm fits, not bank preparation. No full-dataset overnight run is launched by this command.

## Local validation

`test_orientation_coverage_protocol.py` passes independent point/grid/frame agreement on asymmetric coordinates and boundary voxels, inverse/group closure for all24 rotations, rejection of reflections/scaling, normalization commutation, balanced scheduling, and384 actual training-support grid roundtrips. All four new Python files compile. Mocked launcher checks verify one-GPU sequencing, two-GPU concurrency after preparation, allocation-mask preservation, and stopping when preparation fails. Parent/probe/source hashes and report/native row schemas were checked against the actual returned bundles. The GPU-specific code was reviewed against the completed checkpoint writer, optimizer constructor, preprocessing, VAE layout and sampler. Torch/GPU execution and loading the cluster-only checkpoint remain untested locally; do not describe these CPU checks as a successful GPU run.
