# Visual removal does not rescue shared-orientation transfer

The completed inference-only factorial uses the broader shape-path step2000 checkpoint. Removing the visual context worsens reserved-view reconstruction. It does not remove the conditioning failure. Independent point conditioning works well on most fitted examples, but transfers poorly to reserved camera observations/orientations. This supports an orientation-coverage/readout branch, not another attempt to fix the current failure solely by suppressing visuals.

## Verification

All144 payload member hashes/CRCs,56 cases and224 individual predictions validate. The included fit report is identical to the previously verified broader-arm report. Source hashes match; target frames, target quality, input/feature hashes, noise hashes, original supports and28 physical labels match the existing experiment. The full visual-present native assessment replays exactly, and all112 visual-present predicted supports are bitwise identical to their historical counterparts.

Native IoUs/counts and common-unit/raw proximity reproduce independently. Existing rigid registrations were reused only for the112 identical historical supports with identical frames and verified registration source; all112 visual-zero predictions were registered using that same method. GPU context hooks pass for all eight native/sampled input states. The GPU script reports unchanged complete parameter digest and no updates; raw checkpoint tensors remain on the cluster, so local verification does not independently read those weights.

The reported probe assessment/sampling time is289.05 seconds, excluding initial model setup. No training occurred. See `shared_orientation_visuals_analysis/{validation,summary,complete}.json` and `predictions.jsonl`.

## Correct-surface reconstruction

All common-unit values use the established two-original-voxel tolerance, with100% P/R decoded-target positive controls. Pose-witness scores select the raw or measured proper-rigid candidate by minimum precision/recall; they are not a certified global optimum. No fitted scale or nonrigid registration is used.

| Split | Visual context | Native target IoU | Raw F-score | Pose-witness F-score | Individual95% P/R passes, raw / witness |
|---|---|---:|---:|---:|---:|
| Fitted | Present |81.57%|98.78%|99.87%|28/32 /32/32|
| Fitted | Zero |69.16%|96.43%|98.81%|24/32 /27/32|
| Reserved view | Present |18.60%|59.96%|83.51%|0/24 /11/24|
| Reserved view | Zero |10.02%|36.87%|69.07%|0/24 /6/24|

Visual-zero fitted shape passes the strict common-unit per-object mean reference for3/4 objects, not all4. Object146b6c1 has mean P/R96.51%/96.74%, below the98% reference. Therefore do not describe the whole independent geometry path as fully resolved. Nevertheless the other three fitted objects pass that reference while all four fail it on reserved views: transfer failure is not merely an artifact of a completely nonfunctioning point-only fitted pathway.

Removing visuals decreases raw reserved F-score for every object. The pose-witness mean decreases for three; bathtub e846574 changes marginally83.47%→83.74%. There is no reconstruction rescue hidden by the pooled average. The visual-present model is better overall under both raw and pose-adjusted scoring, although that does not establish that every possible visual interaction is harmless.

Fresh native correct-surface loss also worsens with visual removal: fitted .04272→.05466; reserved .22330→.24177. These native values are not translated into geometry accuracy; the sampled supports supply the reconstruction evidence.

## Point dependence is present, but usefulness remains incomplete

| Split | Visual context | Correct-minus-wrong raw F-score | Correct-minus-wrong pose-witness F-score |
|---|---|---:|---:|
| Fitted | Present |+42.07pp|+23.16pp|
| Fitted | Zero |+82.03pp|+62.32pp|
| Reserved view | Present |+6.24pp|+5.08pp|
| Reserved view | Zero |+15.75pp|+23.77pp|

With visuals zero, all four fitted objects pass the predeclared10pp pose-witness dependence screen. Reserved object advantages are+41.45pp (146b6c1),+29.45pp (shield),+21.86pp (bathtub), and+2.32pp (drill). Thus3/4 reserved objects pass that screen, but none passes the reconstruction reference. This is partial surface-conditioned information, not an accurate conditioning solution.

The larger correct-minus-wrong gap with visuals zero should not be presented as better absolute performance: both correct and wrong predictions deteriorate, and wrong predictions deteriorate more. Crossed visual/wrong-surface conditions are contradictory inputs. Without visuals, wrong geometry is the only object-specific condition. Differences in the intervention gap do not by themselves quantify improved geometric reasoning. Because these are fitted identities, recognition from surface codes remains possible; the contrast does not prove general geometric readout on new objects.

## A failure that no rigid output correction can repair

For visual-zero reserved drill `fe200ce00a5d4c298eee89de0fc15f01_010`, draw1, the decoded prediction has only4 occupied voxels. Applying the existing conservative diameter/projection-interval argument certifies **target recall≤26.4412% under every rotation/translation**, at the same two-voxel tolerance. This is a shape/extent collapse, not merely output pose. The certificate does not allow fitted scaling and does not prove anything about asymptotic model capacity.

All28 faithful-target controls pass. Only one of112 correct-surface predictions is rejected by this conservative bound; non-rejection of the others is inconclusive. The earlier hull-only implementation encountered a planar four-point support; `check_visuals_rigid_bound.py` handles small supports using exact all-pairs diameter, with singleton/planar/rigid-invariance controls. Larger degenerate supports use a conservative bounding-box diameter upper bound, never a perturbed point cloud. No model output or geometric tolerance changed. See `shared_orientation_visuals_analysis/rigid_bounds.json`.

## What this narrows, and what it does not

1. A pure explanation that the visual stream prevents an otherwise successful geometry path from handling reserved camera orientations is not supported. The failure persists, and worsens overall, without visuals in a state used during training. This inference intervention does not rule out visual interference during the earlier learning process; both inference states share the same jointly trained weights.
2. Points are not simply ignored: the point-only fitted path reconstructs most examples closely and responds strongly to surface identity. Yet coordinate-dependent geometry codes do not transfer reliably across these reserved observations.
3. At least one reserved failure involves actual decoded shape/extent collapse, so post-hoc output alignment cannot be a universal fix.
4. The experiment does not isolate rotation from every observation difference: natural views also change point samples and normalization. F3 separately established a rotation treatment effect in the earlier contract. The new result is consistent with inadequate orientation coverage or difficult orientation-dependent readout; it does not choose between those, prove VecSetX inadequate, or prove that larger training cannot help.
5. The observations are repeatedly inspected development views of four fitted identities. They remain excluded from optimizer updates, but are not an untouched final test set. No new-object result is supplied.

## Selected next branch

Prioritize a matched orientation-coverage intervention while retaining VecSetX and the camera-oriented target contract. On visual-dropped updates, rotate the observed points and corresponding training target together, then re-encode both through their existing encoders. Compare against the same additional training exposure with the original orientation bank. Retain the original visual-present updates and reserved inputs; do not pair rotated geometry with contradictory unrotated visuals. No latent-channel rotation and no GT geometry as conditioner.

A finite bank of proper cube rotations is a useful implementation candidate: it preserves the voxel grid by exact axis permutations/flips and avoids introducing continuous target revoxelization as another confound. The finite bank is now implemented in [ORIENTATION_COVERAGE_HANDOFF.md](ORIENTATION_COVERAGE_HANDOFF.md), including point/target transform checks, identity replay, target-VAE gates and equal update/optimizer policy. CPU checks pass; GPU execution is pending. This is not a claim that24 rotations guarantee arbitrary-angle transfer.

If added orientation coverage improves reserved target-referenced geometry and surface dependence beyond matched exposure, that gives a concrete training intervention to take into a larger pilot. If improvements occur only on fitted/augmented orientations, or reconstruction remains poor, do not call equivariance learned or impossibility established; the remaining choices include broader object diversity and explicit input-frame handling. Do not automatically change the encoder or launch all branches.

The scale-up clarification in the pivotal ledger remains in force: held-object superiority after four-object training is not a mandatory prerequisite for a larger diagnostic pilot. Such a pilot can test a data-diversity hypothesis. This result does not make more training intrinsically pointless, nor certify the current configuration as fixed. No new full-training job is included with this analysis.
