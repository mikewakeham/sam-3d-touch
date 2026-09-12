# Coordinate diagnosis: reconciled state and next decision

12 September 2026. This records the coordinate diagnosis resynchronization and the original factorial plan. That experiment and the subsequent view/identity transfer screens have completed. The latest evidence and next decision are in `TRANSFER_CHECKPOINTS_RETURNED_FINDINGS.md`; the only current GPU dependency is `CONSTANT_SURFACE_HANDOFF.md`. The coordinate contracts and user constraints below remain applicable; its next-experiment sections are historical, not queued jobs.

## User objective and constraints

Make full-surface **point conditioning** work in Stage 1, or identify a specific demonstrated obstacle. Prioritize coordinate handling, retain pretrained VecSetX, and hold representation and fusion changes until coordinate-specific evidence calls for them. Full surfaces are the current upper-bound task. Sparse contacts, normals and per-point physical measurements are future compatibility constraints, not implementation work for this round. A complete-mesh/voxel bridge is not the proposed conditioner. No Stage 2, full-dataset oracle rerun, or speculative full training.

The newly drafted point encoder/adaptation comparison was not executed and is withdrawn. Its two untracked prototype files were removed. The bridge experiment remains diagnostic evidence only. No production model code or checkpoint was changed by those branches.

## Recovered experimental sequence

1. Source/data audit: camera points and pointmap share the SAM camera convention; inverse-transformed views agree, and source-to-render axes include the stock target conversion.
2. Predictions: some large errors correspond to object-dependent output rotations. This is evidence of orientation errors, not a global conversion fix.
3. Matched four-object single-view fitting: frozen VecSetX plus projector/full shape CA can fit both camera and oracle surfaces very well. A universal fitting impossibility was ruled out at this scale.
4. Same-object view transfer: holding oracle surface and target fixed while changing image/pointmap raises loss .01887 -> .12771. Surface-frame variation is not necessary for this particular view-transfer failure. RGB and pointmap were changed together, so their separate roles remain unresolved.
5. Matched multi-view fitting: camera versus oracle surfaces gives fit losses .03322 versus .02687 and reserved-view losses .08071 versus .06013. This establishes a frame-treatment benefit under matched learning. It does not separate rotation from view-dependent normalization. Both curves were still improving.
6. User-reported dataset-wide oracle run remained similar to the other variants. Its detailed results are not in the provided exports, but its existence is accepted. Repeating that run was withdrawn.
7. Frozen-prior target-orientation probe: six reencoded discrete target rotations produced no convincing transferable correction. The best prespecified held-view gain was .344% and lacked rollout corroboration. This was not a training experiment and had no VecSetX input; it cannot rule out all coordinate-learning problems.
8. Native VecSetX reconstructed substantial geometry in camera and oracle frames. That supports retaining it; it does not establish rotation-invariant features or SAM's ability to interpret them.
9. Direct readout and geometry-bridge probes investigated translation between representations. They established fitting and a useful alternate complete-shape conversion respectively. Neither decomposed the existing camera/oracle frame effect, and neither justified replacing the user's conditioner.

The drift was from “a global axis patch is unsupported” to prioritizing a new representation before resolving the frame effect in item 5. The representation probes were not evidence that the coordinate investigation was complete. The original larger oracle-fusion/adaptation plan was designed but never executed; it must not be counted as a negative result.

## Actual coordinate contract in current source

Use row-vector point notation. Let P be surface points in the saved normalized object frame. Target generation applies `T_normalized_from_source` to the source mesh, voxelizes in that frame, and encodes a spatial latent. It does not apply the stock axis conversion a second time. Stored targets are view-independent but orientation-dependent.

`sample_full_surface.py` uses the rigid transform formed from `diag(-1,-1,1,1) @ T_camera_from_object`. Thus stored camera points are C = P R^T + t. `dataloader.py` reads those points, and its oracle option supplies the inverse of that same transform. `train.prepare_batch` applies the inverse before point encoding for the oracle arm. Other arms apply pointmap SSI before encoding.

For the no-position full-surface path used in the controlled fits, positive isotropic SSI translation/scale cancels in `TouchEncoder.normalize_points_for_vecsetx`:

N(X) = (X - bbox_center(X)) / max_radius_about_that_center(X).

This N is not rotation-equivariant because the axis-aligned bounding-box center changes when the object rotates. Both the centering and radius can therefore differ between camera and object frames. Position-enabled production variants have an additional global shift/scale branch and must be treated separately; the cancellation argument alone does not cover them.

The no-position camera arm presents N(C) to VecSetX; the oracle arm presents N(P). Both supervise the same object-frame target, with the same camera-view image/pointmap input. Known inverse alignment is a privileged diagnostic; no inference-time object pose solution has been established.

## New local input-space factorization

`audit_frame_factorization.py` checks 64 samples / 32 objects using the existing local selection. Let c_o,r_o be the object-frame bbox center/radius, and c_c,r_c their camera-frame counterparts. It verifies

N(C) = (r_o/r_c) N(P) R^T + (c_o R^T + t - c_c)/r_c.

Maximum reconstruction error is 6.66e-16. After undoing rotation, normalization-only coordinate RMS discrepancy has median .02334 and maximum .13613 in normalized encoder units. Camera/object radius ratios range .90060 to 1.06748. Rotation-only RMS changes have median .57502; these input magnitudes cannot be used to infer relative neural or training effects.

A centroid/radius normalization commutes with the same transforms to within 1.46e-7 (camera matrices are stored at finite precision). That is an algebraic control, not a recommendation to change pretrained VecSetX preprocessing. Such a change requires a matched neural test.

This factorization isolates an already documented suspicion. It does not discover a corrupt extrinsic or establish that normalization explains the original dataset result.

## Next coordinate-specific experiment

Decompose the existing positive camera/oracle comparison before changing an encoder. Keep pretrained VecSetX, its feature variant, token projector, full shape CA scope, images/pointmaps, object targets, manifests, noise/time draws and optimization fixed. The only factor is which coordinates enter `encoder.encode`:

| Arm | Coordinates supplied to VecSetX | Orientation | Bbox/radius choice |
|---|---|---|---|
| O: original oracle | N(P) | Object | Object |
| R: rotation only | N(P) R^T | Camera | Object |
| N: normalization only | N(C) R^-T | Object | Camera |
| C: original camera | N(C) | Camera | Camera |

These are diagnostic interventions using known transforms. R and N are not asserted to be deployable preprocessing. Importantly, do not pass them through `prepare_points` and normalize them again; that would erase the intended factor. Keep the same 8192 points/mask and ordering as the original probe, with no new FPS/density confound.

First require O/C endpoint input/features and initial losses to reproduce the original multi-view experiment. If those checks pass, reuse the original O/C runs at the same 1000-update budget and run only the two missing mixed treatments. If they do not pass, resolve provenance/numerical differences before comparing; do not silently use mismatched historical baselines. The known four-object effect is the first mechanism test, not a claim about unseen-object generalization or convergence.

Measure native shape flow losses on fitted and reserved views, correct versus wrong-object surfaces, per-object effects, learning curves and independent final noise draws. Use the same monitoring bank as the previous experiment for reproduction; a fresh bank cannot retrospectively be claimed for historical results. A decisive subsequent confirmation should use a fresh common bank and another seed. Stage-1 rollout checks are corroboration, not the central coordinate diagnosis.

### Outcomes and specific next interventions

- **R deteriorates while N stays near O; C behaves like R:** orientation recovery is the dominant measured factor. Keep VecSetX fixed and test a targeted alignment/rotation-handling intervention against known-pose inversion. Establish what pose information is actually available, and whether the target convention requires an arbitrary per-object front, before proposing a deployable alignment method.
- **N deteriorates while R stays near O; C behaves like N:** view-dependent centering/radius is the main measured factor. Test an input-only rotation-consistent normalization with explicit normalization metadata, retaining VecSetX. A pretrained-distribution shift is a competing effect to measure, not ignore.
- **Both deteriorate or only C does:** both factors or their interaction matter. Compare the combined treatment with each factor separately; do not credit a combined fix to rotation alone.
- **All are similar, or the historical gap does not replicate:** do not force a coordinate diagnosis. Report this contrast as insufficient and return to the fixed-oracle view-dependence observation.

After isolating a factor, confirm the improvement on a modest multi-object subset before any full run. A correction must improve the actual Stage-1 conditioning task, not only make features numerically stable. Repeating the full oracle run is not part of this plan.

## Remaining oracle failure is a separate question

Even a fully isolated camera-frame penalty cannot explain all residual failure: the earlier oracle model has fixed surface/target frames yet remains view-sensitive. If the user asks for a coordinate explanation of that residual, the next distinction is pointmap-versus-RGB dependence with the oracle surface fixed. The previous probe changed both together. Separate them first using the existing checkpoint, label crossed views as inconsistent diagnostic inputs, and require a matched training intervention before blaming incompatible coordinate frames. Visual sensitivity alone is not proof of a pointmap-frame conflict.

A claim of inherent unidentifiability needs an explicit observation/target collision under the intended available inputs, or a matching formal symmetry argument. Current evidence supplies neither. Failed readouts and finite optimization budgets do not establish impossibility.

## Current status

Both mixed GPU treatments have returned and passed provenance and baseline replay checks. `FRAME_FACTORS_RETURNED_FINDINGS.md` records the orientation-dominant result; do not rerun the handoff. `POSE_INPUT_CONTRACT.md` follows that branch by tracing the available pose information and executing a 64-sample camera metadata check. The normal input has no explicit camera-to-target rotation, and the protected shape stream cannot read the layout latent. A usable rotation estimate and the residual failure under known-pose alignment are distinct remaining questions. No point-encoder, bridge, sparse-feature or broader-adaptation implementation is the current dependency.
