# Existing broader checkpoint: does visual input obstruct orientation transfer?

Status: ready for GPU execution; no new training. F14 and its completed scope comparison remain the starting point. This is not another representation or target-convention change.

## Why this branch

The broader step2000 model meets the fitted common-unit geometry reference after measured rigid adjustment (all32 predictions pass95% precision/recall; all4 object means are within2pp of the decoded-target control). Reserved-view geometry and reliable surface dependence still fail. Natural reserved views change both geometry orientation and visual features, so the current results do not isolate which input path causes the transfer failure.

Compare the existing checkpoint with its whole visual context present versus zero. Zero includes RGB, silhouette and pointmap context, exactly the state used for50% of its training. Keep VecSetX features, target latents, model weights and noise fixed. Cross each visual state with correct and wrong-object surfaces; a visual-zero improvement alone would not establish useful surface conditioning without reconstruction fidelity and correct-surface advantage.

## Fixed protocol and interpretation

- Four fitted identities, four fitted views and three reserved views. No unseen-object claim.
- Four conditions: visual present/zero × correct/wrong surface. Wrong means the same deterministic batch roll used previously.
- Eight paired fresh-noise native-loss draws per condition/view group; two paired25-step CFG0 samples per condition/view group. Total224 individual sampled outputs. No optimizer steps or target encoding.
- The full visual-present native assessment replays the completed checkpoint with the established BF16 rtol0.001/atol0.000001. Parameters, source files, input features, target content and sampling-noise hashes are checked separately. Full parameter hashes must remain unchanged.
- Same-run visual-present sampling is the paired reference. Historical support equality/IoU is recorded without imposing bitwise equality on recomputed BF16 outputs. Both visual states use the same model mode, requires-grad flags and autocast path. Context hooks verify that only the intended visual/surface inputs change.
- Retain exact-target native IoU and common-original-unit precision/recall at two original voxels. The common-unit positive controls remain100% P/R. Report raw and the established rigid-witness comparison separately; no free scaling, reflection or nonrigid matching.
- The unchanged reconstruction screens are95% overall native IoU/90% each-object mean, and common-unit each-object mean P/R≥98%, with individual95% P/R pass counts. The unchanged surface-utility screen requires correct-minus-wrong F-score≥10pp for each object. Passing only a mean improvement is not resolution.

Branches, decided from object-level outcomes rather than a single pooled loss:

1. **Reserved geometry becomes accurate with visuals zero while present remains poor:** visual input is a causal contributor in this checkpoint. If correct surfaces also beat wrong ones reliably, pursue visual/geometry integration under the chosen frame convention. This does not by itself prove that coordinate conflict, rather than another visual interaction, is the internal mechanism.
2. **Reserved geometry remains poor with visuals zero but fitted geometry is accurate:** geometry-orientation transfer remains a bottleneck even without visual competition. Next select a matched jointly rotated point/target training intervention or explicit input alignment, based on these failures. Do not automatically launch both or keep extending the present fit.
3. **Both fitted and reserved geometry fail with visuals zero:** independent use of geometry is still inadequate. Do not attribute all failure specifically to unseen orientations. Inspect the correct/wrong contrast before selecting the intervention.
4. **Mixed or modest changes:** report partial sensitivity, not a resolved cause. Inspect which objects and raw versus pose-adjusted endpoints change; this factorial cannot prove impossibility or identify a unique internal mechanism.

Stage1 only. VecSetX remains unchanged, and observed surface points remain the conditioner. Mesh/voxel arrays are target/reference material only. Sparse structured touch remains the eventual goal. No production trainer changes or full-run promotion are included.

## Run in the existing interactive allocation

Sync the one new executable file `doc/conditioning_investigation_2026-09-11/probe_shared_orientation_visuals_gpu.py`. All other imports and references were used by the completed scope run. Keep that run's checkpoint, results,28 NPZs and bundle together in its existing directory. One visible GPU suffices; additional GPUs are unused.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python \
  doc/conditioning_investigation_2026-09-11/probe_shared_orientation_visuals_gpu.py \
  --fit-dir outputs/conditioning_investigation/shared_orientation_scope/manual/shape_path \
  --output-dir outputs/conditioning_investigation/shared_orientation_visuals/manual
)
```

Return `outputs/conditioning_investigation/shared_orientation_visuals/manual/shared_orientation_visuals_bundle.zip`. It contains all prediction/target supports, reports and physical labels needed for local analysis; no checkpoint upload is needed. If execution fails, return the traceback and `results.partial.json` if created. The output directory must be new; do not remove completed evidence to rerun.

## Local validation and limits

Syntax compiles. Against the actual returned broader bundle, source hashes, cached reference/target/physical member paths and hashes, all seven target-support shapes/dtypes, and native replay row ordering pass. The implementation was checked against the completed continuation's checkpoint-writing, feature-preparation, native-loss, sampling and bundle paths. GPU execution and loading the cluster-only checkpoint/latent NPZs cannot be tested on this laptop; the script performs those checks before collecting the paired sampling comparison.
