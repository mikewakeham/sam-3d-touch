# Presentation and remaining measurements

Status: no more full training runs are planned. Current runs are oracle surface/no-pointmap/no-dropout/shape cross-attention, shared-normalization image+pointmap, and shared-normalization image+pointmap+surface. Do not push or modify the remote without the user's explicit instruction.

The additional rotation measurement is fully specified in [ROTATION_EXPERIMENT_PLAN.md](ROTATION_EXPERIMENT_PLAN.md), including native-grid controls, the fixed-scale angular sweep, analytical flow-loss interpretation, figures, implementation and storage. Scripts are implemented and CPU checked; pretrained GPU runs remain pending. Commands are in [rotation_loss/README.md](../scripts/rotation_loss/README.md). The experiment tests target orientation sensitivity; matched checkpoint results test the benefit of oracle.

## Main slides

| Slide | Content | Evidence / claim boundary |
| --- | --- | --- |
| 1. Setup | Current SAM3D + VecSetX pipeline. Mark Stage 1 as the training scope. Full surface is an upper bound for eventual sparse touch. | Stage 2 is frozen and used for evaluation. |
| 2. Raw inputs | Three octopus RGB views, full surface, pointmap, and overlay. | Raw surface and pointmap share camera coordinates. |
| 3. Encoder preprocessing | Stock pointmap SSI versus VecSetX center/radius, then proposed shared transform. Use actual encoder-boundary arrays. | Different normalization conventions are a hypothesis, not a proved cause. Surface original shift/scale are omitted in these jobs. |
| 4. Target and oracle | Object mesh -> 64-cubed occupancy -> frozen SS encoder mean. One object-frame target is shared across views. Oracle inversely transforms the surface into that frame. | Target mean is spatial and orientation sensitive. A shared per-object target is not itself a dataset error. |
| 5. Orientation penalty | Controlled same-shape target rotations and latent/velocity-target discrepancy. Score a fixed generated endpoint directly against original and native-rotated target means. | No decoded F-score is evidence of latent loss. Endpoint MSE and native velocity MSE are different quantities; show each with its definition. |
| 6. Oracle full-run results | Camera/no-PM versus oracle/no-PM, no dropout, same scope. Include relevant image baseline and native flow-loss comparison. | This tests whether supplying orientation helps this learner, not whether every coordinate effect is absent. Generation quality can be assessed separately. |
| 7. Shared-normalization results | Stock/shared normalization crossed with absent/present surface tokens. | The shared image+PM control uses surface center/radius but no surface tokens. Separate normalization-only improvement from surface-token benefit. |
| 8. Conclusions | What was verified, what intervention helped, what remains inaccurate relative to GT-latent reference. | No claim that oracle solves conditioning or that representation is incompatible without supporting reconstruction. |

Appendix: tiny single-view and frame-factorial scopes/results; tiny dropout versus held identities and transfer limitation; camera-frame target pilot; exact run matrix and evaluation settings. Promote an appendix result only if it is needed to explain a main finding.

## Required checkpoint evaluation

- Use the full-training train and validation splits separately, not the tiny held-view bank as a substitute for validation identities. Select a common deterministic bank before scoring. Initial assessment: 32 train and 32 validation objects, two views and two paired draws each. Average over views/draws within object before population summaries. Expand with the same protocol only if differences remain uncertain.
- Required normalization matrix: existing stock image+PM (`xun3al7m`), stock surface+PM (`act988rs`), and both new shared runs. Verify the old manifests select the same evaluation IDs. Required orientation pair: existing camera surface/no-PM (`nouqb3mh`) and new oracle/no-PM, both no dropout. Existing oracle+PM/no-dropout (`fd2yiktq`) provides the PM-presence comparison. Image/no-PM (`fl7b2znc`) supplies its baseline.
- Evaluate best checkpoints for all paired comparisons. Evaluate last checkpoints when curves or best/last differences matter, especially any historical shape-full results included in the appendix.
- For the coordinate-training argument, report native flow-velocity MSE on paired noise/times and generated endpoint latent MSE against original/native-rotated target means. Keep train/validation identities separate. Decoded support F-scores are optional generation-quality evidence and do not establish a latent-loss penalty.
- Also run the same frozen Stage-2 sampler/decoder for predicted Stage-1 latents and GT Stage-1 latents, with identical images, settings and seeds. Report mesh CD in common object units, raw and rigid aligned. Report uniform-scale similarity alignment separately rather than silently folding scale into rigid alignment. Recompute the GT-latent reference on this same bank; do not assume the historical 0.0096 applies unchanged.
- Add correct/wrong surface-token tests on a small paired subset. For shared runs, hold the already normalized pointmap and its correct-surface center/radius fixed while swapping only surface tokens. Recomputing normalization from the wrong cloud would change two streams and confound the test.
- Save metrics for the entire bank and a few selected qualitative examples with prediction latent/support, meshes and alignment transforms. No frozen-model copies, bulk feature caches or automatic ZIPs.

## Short rotation diagnostic (no optimization)

Purpose: isolate the error produced by a correct shape in a different orientation. This is distinct from the previous conditioning-rotation experiment, where the model could change the generated shape as well as its orientation.

1. Encode deterministic SS posterior means for identical source geometry at declared rotations. Include native-size 90-degree voxel-grid permutations, which preserve voxel count and avoid clipping/resampling. Repeat zero degrees as the numerical reference.
2. For an angular curve, use 32 metadata-selected objects and rotations 0, 5, 15, 30, 60, 90 and 180 degrees about each axis. Use one fixed center and scale across the entire sweep, with fixed padding sufficient for all angles. Do not recompute AABB normalization at each angle, clip coordinates, or rotate latent channels. State that the padded zero-angle target differs from stock normalization and retain the native-grid controls separately.
3. Measure latent-mean MSE against zero-angle encoding and geometric error before/after the known inverse rotation. Decode every target variant so voxelization/VAE error is measured instead of attributed entirely to orientation.
4. For flow matching, u_theta = z_theta - (1-sigma_min)*epsilon. With the same epsilon, MSE(u_theta, u_0) = MSE(z_theta, z_0) exactly. Label this velocity-target discrepancy, not an observed generator loss.
5. If reporting actual generator training loss, use the native loss implementation with identical conditioning, addressed noise and timesteps across target variants. Each target changes its training noisy state. Report the result separately from the pure target discrepancy. The old target-orientation probe already tested coarse frozen-model target preferences; do not duplicate it solely to seek a better hidden yaw.
6. Plot object-level curves, not independent error bars from correlated axis/noise draws. Error need not grow monotonically: symmetry, periodic rotations, voxel discretization and the model's current orientation preference can change the curve.

## Optional decoded-geometry evidence (not the oracle training motivation)

Existing `coordinate_system_results/geometry/pose_shape_analysis/pose_shape_examples.png` and `rotation_shape_comparison.png` can be reused as exploratory evidence. The natural-camera drill example improves one-voxel F-score from 2.0% raw to 79.7% after rigid registration. That shows a substantial pose component and a residual geometric error; it does not establish a perfect shape.

For clearer presentation examples, use the checkpoint evaluation bank. Show target, raw prediction and rigid-aligned prediction under one display camera and scale. Print raw/aligned errors, recovered rotation and GT-latent reconstruction reference. If no prediction approaches the reference after alignment, do not claim that its shape was correct. Selected diagnostic examples must not stand in for aggregate performance.

The claims must remain separate: (1) target latents distinguish orientations of the same shape; (2) an actual endpoint is closer to a tested rotated target encoding; (3) oracle improves native training-objective performance in a matched comparison. Decoded F-scores establish none of these latent claims. Generation quality is a separate endpoint.
