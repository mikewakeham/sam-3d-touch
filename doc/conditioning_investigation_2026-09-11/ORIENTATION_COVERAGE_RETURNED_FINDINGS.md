# F16: exact fitted reconstruction is possible; short rotation coverage is not a fix

Both bundles are complete. All708 payload hashes/CRCs,512 model predictions and the shared384 rotated target controls validate. The two arms have matched initial parameter/Adam digests, source/input hashes, dropout/noise schedule, optimizer scope and1000 added updates. Both native initial assessments replay F15 exactly. All384 target physical grids equal the intended proper rotations; decoded-target metrics reproduce, with minimum VAE IoU99.9823%. Model/optimizer/feature tensors remain cluster-only; their hashes are source-audited reports, not independent local tensor loading.

Analysis: `orientation_coverage_analysis/{validation,summary,complete}.json`, `predictions.jsonl`, `learning_and_exposure.json`. Empty decoded predictions retain zero precision/recall/F-score and unavailable distances. A local analysis failure on comparing unavailable distances was corrected without changing model outputs or dropping observations; the eight completed rows were retained before resuming the remaining analysis. Existing registration code and tolerances were unchanged.

## Results at total step3000

All F-scores below use the same two-original-voxel tolerance. The rigid witness uses the established proper-rigid candidates; it is not a globally certified optimum or a fitted-scale correction.

| Inputs | Control IoU | Augmented IoU | Control raw / rigid-witness F-score | Augmented raw / rigid-witness F-score |
|---|---:|---:|---:|---:|
| Fitted, visual present |99.86%|94.17%|100 /100%|99.80 /100%|
| Fitted, visual zero |99.56%|13.68%|100 /100%|48.08 /68.49%|
| Reserved views, visual present |17.86%|18.32%|59.08 /81.84%|59.86 /84.40%|
| Reserved views, visual zero |10.50%|5.75%|36.89 /62.17%|24.51 /54.45%|
| Added-rotation diagnostic, visual zero |5.21%|4.74%|21.07 /42.61%|22.66 /49.54%|

**Pivotal positive:** the unaugmented control passes fitted native reconstruction, common-geometry and each-object surface-dependence screens with both visual states. All32 fitted predictions in each state pass95% bidirectional proximity; every object has100% common-unit mean P/R. Correct-minus-wrong rigid-witness F-score is+47.18pp with visuals and+63.29pp without. This establishes almost exact fitted reconstruction of shape AND orientation under the camera-oriented target convention. It resolves the earlier small-task overfitting deficit under this adaptation/exposure recipe, not the original full-dataset transfer problem. It does not establish broader finetuning is necessary; current-scope training at the same total duration was not run.

**Augmentation result:** a small visual-present reserved improvement (+0.46pp native IoU/+2.56pp rigid-witness F-score) does not meet reconstruction or surface-utility references. Visual-zero reserved geometry worsens. None of the32 sampled added-rotation predictions passes95% P/R after the measured rigid correction; all four object means fail the common-geometry reference. The sampled added training conditions themselves were not reconstructed accurately, so the result cannot test transfer from a successfully learned rotation mapping.

## Exposure and what failed to converge

The treatment replaces500 original-orientation visual-zero updates with23 nonidentity rotations per fitted group. Each new condition receives only5 or6 optimizer presentations, versus125 original-orientation zero-visual presentations in the control. Identity-orientation zero-visual examples are absent during this added treatment phase; they remain in visual-present updates and were seen earlier. The degradation of original visual-zero outputs therefore includes a retention/distribution-change issue, not a clean test that an otherwise unchanged point pathway spontaneously stopped working.

The sampled augmented diagnostic covers16 of368 new conditions, two draws each. Its four selected group0 rotations had6,6,6 and5 presentations. Do not claim every bank condition was individually sampled or that the optimization budget was sufficient for the expanded task.

Augmented dropped-update mean training loss is .18439 in the first100 updates and .16511 in the last100. This is not a matched-noise convergence test; noise and sampled rotations vary. Natural-input fresh losses improve on reserved views relative to the control despite poor sampled reconstruction:

| Arm/step | Fitted present | Fitted zero | Reserved present | Reserved zero |
|---|---:|---:|---:|---:|
| Both2000 |.04272|.05466|.22330|.24177|
| Control2300 |.02933|.03655|.24654|.26136|
| Control3000 |.01202|.01345|.30463|.32413|
| Augmented2300 |.04079|.08512|.17568|.18839|
| Augmented3000 |.02544|.08712|.18268|.18138|

The loss improvement is real under the fixed native probe, but does not establish good shape generation. Nor does this failed short fit prove that orientation augmentation cannot work with different exposure or training coverage.

## User objective clarification: shape accuracy does not require matching target pose

The user explicitly cares about correct shape, not reproducing a particular GT orientation. Exact native IoU/latent matching is a diagnostic of the existing frame-specific training objective; it must not become an additional product requirement. A faithful shape in another orientation can satisfy the requested output objective. The current training loss nevertheless compares oriented spatial latent entries, and the camera-oriented target branch asks the model to produce geometry in that frame. That is an implementation choice, not a mathematical necessity for shape-only reconstruction.

We already report geometry after proper rigid correction so pose mistakes alone are not treated as proof of wrong shape. The new visual-zero and augmented-condition failures persist under that check. A failed numerical registration alone is not an all-rigid impossibility certificate; the earlier F13/F15 specific geometric bounds remain the stronger certified examples.

The cube-rotation experiment tests a candidate training remedy for this chosen camera-frame output contract. It does not establish that the user's final system must support every cube rotation or retain this output contract. Alternatives include a consistent input/output canonical frame or an orientation-invariant training objective. Those alternatives require concrete design and validation; neither has been implemented as a fix here. No new training job follows automatically from this report. First explain the distinction between input-orientation robustness, output-pose requirements and the current spatial-latent loss, as the user has requested.
