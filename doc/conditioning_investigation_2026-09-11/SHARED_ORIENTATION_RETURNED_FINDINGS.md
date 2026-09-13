# Camera-oriented target intervention: geometry audit completed

13 September 2026. Source: `shared_orientation_returned_manual/results.json`, copied byte-for-byte from the returned attachment. Reproduce aggregate/provenance checks with `python doc/conditioning_investigation_2026-09-11/analyze_shared_orientation_report.py`. Outputs: `summary.json`, `validation.json` in that directory. The raw geometry bundle has now been received and independently verified. The original report-stage narrative below is retained, with the completed geometry follow-up and resulting branch decision immediately below.

## Completed bundle follow-up — 13 September

Bundle SHA-256: `8dbce7a49c70f3e602e5ba74bf33fda4cc231bd9ba11c20a15d566dd571189f1`. All 86 payload member sizes/hashes and CRCs pass; bundled and pasted reports match. All 28 target arrays reproduce the VAE roundtrip and common-unit metrics. Every predicted support count, native IoU, native occupied-center distance and common-unit metric reproduces across all 112 outputs. Original support content, source/reference hashes, initialization, input features, schedule and sample noise agree with history. See `shared_orientation_geometry_analysis/validation.json` and `complete.json`.

The existing proper-rigid registration helper is unchanged and matches the previously validated analytic controls. Known label inversion is applied before registration. No fitted scaling, reflection or nonrigid correction is permitted. The comparison selects between raw and the fixed RMS-registration candidate using greater minimum two-voxel precision/recall, identically for all arms. It does not claim globally optimal alignment.

| Correct surfaces | Fitted raw F-score | Fitted pose-adjusted witness F-score | Reserved raw F-score | Reserved pose-adjusted witness F-score |
|---|---:|---:|---:|---:|
| Original camera/dropout | 99.97% | 99.98% | 69.83% | 89.19% |
| Shared camera orientation | 78.60% | 91.57% | 54.48% | 77.53% |
| Oracle/dropout | 100.00% | 100.00% | 100.00% | 100.00% |

All values use two original-object voxels. Accurate decoded-new-target controls have 100% precision/recall at this tolerance. The stricter predeclared per-object target-reference screen (at most 2 pp below those controls) fails for every shared-arm object on both splits. Only 18/32 fitted and 4/24 reserved individual predictions have both precision and recall ≥95% after the measured pose witness. This corrects an overly coarse reading of low native IoU: many fitted outputs have recognizable/recoverable shape, but the intervention has not reliably fit the oriented shapes and poses.

Reserved shared-arm pose-adjusted F-score and correct-minus-wrong advantages:

| Object prefix | Correct F-score | Correct−wrong advantage |
|---|---:|---:|
| 146b6c1 | 74.71% | +19.79 pp |
| 479bbf9 | 91.16% | −2.80 pp |
| e846574 | 82.03% | +7.88 pp |
| fe200ce | 62.22% | −12.26 pp |

Mean advantage is only +3.15 pp. The predeclared ≥10 pp advantage for each object fails. Correct inputs are not uniformly ignored, but neither fidelity nor consistent surface utility is established.

### Evidence independent of registration convergence

For reserved `fe200ce00a5d4c298eee89de0fc15f01_000`, draw 0, the predicted center cloud has diameter 0.647327 in original object units. Any target centers within epsilon=2/64 of a rigidly transformed prediction must have pairwise separation ≤ diameter(prediction)+2epsilon. Consequently their projection onto any target axis lies in an interval of that width. Counting the largest target fraction inside any such interval gives a conservative upper bound on recall for EVERY rotation and translation.

For this prediction that upper bound is **82.7056%**, below the required 95%. Thus an unsearched rigid alignment cannot rescue this example. This is a rigorous finite-support geometric statement with 1e-10 numerical slack, not an ICP failure or inference from a low loss. No fitted scale is allowed; the bound establishes a non-rigid discrepancy relative to the target, not its semantic/topological cause. The same bound passes all 28 faithful target controls and is invariant under tested source rigid motions. Of all 56 correct-surface predictions it certifies failure for one; the other 55 are inconclusive under this conservative bound, not certified successes. See `rigid_bounds.json` and `check_shared_orientation_rigid_bound.py`.

A fitted drill example (view015 draw1) has measured F-score 37.76% before / 63.28% after rigid adjustment; the reserved example has 0.69% / 28.14%. [Diagnostic figure](shared_orientation_geometry_analysis/drill_failures.png) shows both selected examples with equal axes and scale. The figure was selected after results; aggregate conclusions use all outputs.

### Resulting branch decision

This is a mixed pose-and-geometry failure. Merely rotating the final output cannot be a universal fix. Changing target orientation is also insufficient under the existing adaptation scope and budget. This does not prove that arbitrary camera orientations are unrepresentable, that VecSetX must change, or that full finetuning is necessary. The VAE labels are sound; the generator's accommodation of a per-view spatial target distribution remains unresolved. Training losses still improve.

The next useful GPU comparison is a bounded continuation from the existing step1000 checkpoint with two matched arms: unchanged projector/full shape-CA scope versus additional generator adaptation, keeping VecSetX/visual encoders/target VAE frozen. Both arms should receive identical additional exposure, features, targets, dropout and noise; preserve old optimizer moments in both common parameter groups. The broader arm starts extra parameters from their existing pretrained values. This tests whether the current restricted adaptation is limiting the coordinate-contract change, rather than conflating it with more updates. Its precise broader parameter scope must be source-audited before implementation. Subsequent implementation is now ready: see [SHARED_ORIENTATION_SCOPE_HANDOFF.md](SHARED_ORIENTATION_SCOPE_HANDOFF.md). The CPU audit is complete; the matched GPU continuation is the next required evidence.

Branches: broader adaptation alone reaching target-referenced fidelity supports insufficient adaptation at the compared budget; both succeeding supports insufficient exposure under the original run; fit success with reserved failure localizes orientation transfer; neither succeeding leaves optimization/target-distribution/conditioning explanations open, not an impossibility claim. A negative at one learning rate is not a capacity proof. Do not scale this candidate to full-dataset training yet. The successful oracle route remains a separate viable coordinate contract; deriving a deployable input alignment remains open.

## Result against the planned endpoint

All 1000 optimizer updates and 112 sampled outputs completed. All 13 recorded source hashes and four reference hashes match local files. Initialization, all seven historical camera input/feature batches, the full training group/dropout schedule and all 112 sampling noise records match their historical references. Both visual-context interface checks pass. Parameters changed during training and remained unchanged during sampling. No nonfinite training losses/gradient norms or empty predicted supports were reported.

New target VAE fidelity remains 99.994%–100% IoU over 28 targets; each has 100% two-voxel precision/recall in original units. Original target regeneration errors are zero. Label encoding failure does not account for the poor sampled reconstruction reported here.

| Model, all with 50% visual dropout | Fitted-view native-target IoU | Reserved-view native-target IoU | Fitted common-unit F-score, 2 voxels | Reserved common-unit F-score, 2 voxels |
|---|---:|---:|---:|---:|
| Camera inputs, original fixed asset target | 88.32% | 56.17% | 99.97% | 69.83% |
| Camera inputs, camera-oriented target | 27.81% | 15.90% | 78.60% | 54.48% |
| Oracle inputs, original fixed asset target | 97.80% | 97.17% | 100.00% | 100.00% |

Native IoU compares each prediction to its own exact decoded target. Target grids differ across arms, so these are respective task-fidelity scores, not one shared voxel-label task. Common-unit F-score maps the new arm back using the known label inverse and uses identical two-original-voxel tolerance for all arms. These are raw scores without fitted pose adjustment. Historical values come from the previously independently verified geometry analyses. Target inverse mapping is a diagnostic conversion, not a deployed oracle correction.

The predeclared accurate endpoint requires reserved native IoU ≥95% overall and ≥90% per object; it fails decisively. This is not a case of accurate training-set reconstruction followed only by poor view transfer: fitted-view fidelity is also low. The new arm should not be promoted to a full-dataset run.

## Surface dependence and optimization

With wrong surfaces, new-arm native IoU is 17.64% fitted / 13.41% reserved. Correct minus wrong common-unit two-voxel F-score is +17.88 pp fitted but only +4.45 pp reserved. Reserved per-object advantages are +13.26 pp (146b6c1…), +6.33 pp (479bbf9…), +2.78 pp (e846574…), and −4.59 pp (fe200ce…). This indicates some condition sensitivity, not reliable full-surface recovery. The predeclared pose-adjusted dependence screen is still pending the bundle; raw scores alone must not be represented as its outcome.

Fresh native loss, correct surfaces:

| Update | Fitted views | Reserved views |
|---|---:|---:|
| 0 | .50203 | .51378 |
| 300 | .17069 | .19755 |
| 1000 | .13322 | .17711 |

Optimization is progressing, so 1000 updates do not establish an asymptotic ceiling. Loss comparisons with original fixed targets are not used as a percentage of geometry recovered. Continuing to optimize could help, but this report alone does not justify an open-ended extension or prove that full finetuning is needed.

## What this narrows, and what it does not

The proposed target-convention change is not a sufficient fix under frozen VecSetX plus projector/full shape-cross-attention adaptation at the matched budget. The audit supports unchanged observed inputs, and nearly lossless target encoding removes a major alternative explanation for this result.

Oracle input alignment and camera-oriented target supervision are not equivalent learning tasks. Oracle alignment removes the observed surface's changing orientation while retaining one fixed target per object. Shared camera orientation removes the explicit camera-to-asset orientation conversion but requires the spatial generator to learn view-dependent oriented/normalized targets. This introduces more target variation even on the same four identities. Rotation-dependent normalization also changes; this was a target-convention intervention, not a pure rotation factorial.

The result therefore does not show that coordinate alignment was irrelevant, that matching input/target axes is impossible, that VecSetX is inadequate, or that the pretrained prior cannot represent these orientations. The VAE's ability to encode/decode those labels is established; the flow generator's ability to learn their conditional distribution with limited adaptation remains separate. F3's controlled input-rotation burden and F9's oracle/dropout success remain intact.

## Historical report-stage branch (now resolved above)

First obtain the already generated `shared_orientation_bundle.zip`. Verify member hashes, arrays, target fidelity and reported metrics. Then apply the existing proper-rigid geometry analysis after the known target inverse, retaining native/raw scores and the same two-voxel tolerance for every arm. Reuse existing controls; do not train another baseline merely to obtain them.

1. If sampled shapes are accurate after rigid adjustment but native IoU is poor, localize the residual to orientation/placement of the generated output. Inspect actual transforms and per-view behavior before an adaptation-scope test. Post-hoc alignment still would not meet the fixed-frame endpoint.
2. If fitted shapes themselves remain inaccurate, the current target-convention intervention has a fitting limitation at this budget. A short matched adaptation-scope intervention becomes relevant: keep labels/inputs/encoder unchanged, compare current-scope continuation against additional generator adaptation under equal added exposure. A single longer fit would conflate capacity and optimization. This is a candidate branch, not an implemented or requested job.
3. If a branch achieves accurate fitted reconstruction but reserved views fail, investigate orientation transfer specifically. Do not equate fitting success with general conditioning; eventual held-out-object utility remains necessary before a full-run recommendation.

All measurements concern four training identities, four fitted and three reserved views, two sampling draws and one training seed. Views and draws are not independent population replicates. The sparse-touch interface, final deployment frame and full-dataset benefit remain unresolved. Overnight implementation stays parked.
