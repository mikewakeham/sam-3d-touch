# Matched generator-scope continuation: fitting improves, view transfer does not

13 September 2026. Both 1000-update continuations completed. Full analysis: `shared_orientation_scope_analysis/{validation,summary,complete}.json` and `predictions.jsonl`. The pending job in `SHARED_ORIENTATION_SCOPE_HANDOFF.md` is finished; do not rerun or extend it automatically.

## Matched intervention verified

Both format4 bundles pass CRC, manifest size/hash and safe-path checks: 87 payload members each. Bundled reports match the pasted reports. All 224 native IoUs/counts, occupied-center distances and common-unit distances/F-scores reproduce independently from returned arrays. Physical labels, decoded target supports and original references match the already verified parent bundle.

Both reports agree on the parent checkpoint hash, initial full parameter hash, restored optimizer content hash, common parameter names, all seven input/feature/target hashes, cached latent hashes, dropout schedule, sample noise and settings except scope. Initial replay vectors and first training loss match each other exactly. Replay against the historical parent differs by at most0.05399% relative, within the prespecified0.1% BF16 tolerance. Frozen parameters remained unchanged; the additional shape-path parameters had nonzero gradients and changed only in the broader arm. These are source-audited runtime reports; the full checkpoint/Adam tensors remain on the cluster rather than being independently reread on the laptop.

The current scope trains103,875,648 parameters; the broader scope adds456,045,576, for559,921,224 total. Both receive1000 identical additional examples-of-batches from the same step1000 model and preserve common Adam state. Reported update-phase times are713.6s and777.3s, respectively, including assessments/checkpoint work; these are not complete job runtimes. Same update count does not imply same compute, and the shared global clipping rule interacts with the larger gradient vector. The causal treatment is the specified adaptation scope at the tested learning rates/budget, not an abstract architecture-capacity comparison.

## Target-referenced results

All common-unit scores below use two original-object voxels. Each native IoU compares with that arm's exact decoded target. The pose witness chooses between raw and the existing proper-rigid registration candidate using greater minimum precision/recall; no fitted scaling/reflection/warping or global optimality claim.

| Camera-oriented targets | Fitted native IoU | Reserved native IoU | Fitted raw / pose-witness F-score | Reserved raw / pose-witness F-score |
|---|---:|---:|---:|---:|
| Parent, total1000 updates | 27.81% | 15.90% | 78.60% / 91.57% | 54.48% / 77.53% |
| Current scope, total2000 | 53.91% | 19.10% | 96.75% / 99.13% | 62.45% / 84.66% |
| Broader shape path, total2000 | 81.57% | 18.60% | 98.78% / 99.87% | 59.96% / 83.51% |

Historical oracle/dropout with original fixed targets reaches97.80%/97.17% native IoU and100%/100% common-unit F-score. It remains the accurate aligned reference on these fitted identities, not a new-object solution.

Broader adaptation improves fitted native IoU by27.66pp over the matched continuation, but reserved native IoU is0.50pp lower. Reserved pose-witness F-score is1.15pp lower. Do not call these small differences a robust population disadvantage: the supported result is no useful held-view improvement at the tested seed/budget despite a large fitting improvement.

After measured rigid alignment, **all32 broader-arm fitted predictions** meet95% bidirectional proximity. Each object's mean precision/recall is also within2pp of the100% decoded-target positive control. Thus the predeclared common-unit fitted-shape reference is met. Current scope meets that reference for3/4 object means and27/32 individual predictions. Raw broader-arm predictions pass95% P/R for28/32 fitted cases. Fixed-frame native IoU still does not meet the prespecified95% overall/90% each-object gate; accurate fitted geometry after alignment is not exact spatial-latent recovery.

For reserved views, neither arm has any raw prediction passing95% bidirectional proximity. After the measured pose witness, current scope passes7/24 and broader scope11/24 individually, but neither meets the per-object reference for any of the four objects. Report both the count and mean; they move differently because failure severity differs.

## Surface utility and loss trajectories

Broader-arm fitted correct-minus-wrong F-score is+42.07pp raw and+23.16pp after the pose witness, versus+30.83/+14.65pp for current scope. The fitted shield's pose-witness advantage is only6.38pp, so the strict each-object10pp screen still fails. Strong aggregate sensitivity does not certify a reusable geometric readout; it can still include fitted-identity recognition.

On reserved views, broader-arm advantages are+6.24pp raw and+5.08pp pose-witness. Per-object witness advantages:146b6c1 +12.45pp,479bbf9 −1.87pp,e846574 +8.37pp,fe200ce +1.37pp. Current scope has+6.40pp mean witness advantage. Neither provides reliable held-view surface utility by the prespecified screen.

Fresh native correct-surface loss (full four-group/three-group averages, not the one-group resume replay):

| Model/update | Fitted | Reserved |
|---|---:|---:|
| Parent1000 | .13322 | .17711 |
| Current1300 | .12607 | .17962 |
| Current2000 | .10478 | .18849 |
| Broader1300 | .10473 | .18193 |
| Broader2000 | .04272 | .22330 |

Broadening sharply improves optimization on fitted views while reserved native loss worsens. This is consistent with overfitting/view-specific adaptation. Native loss and sampled geometry remain distinct endpoints; the conclusion is supported by both, not by translating loss into geometry fidelity.

## What is newly resolved

1. The large fitting deficit at step1000 is not a demonstrated representational ceiling. More exposure helps, and broader adaptation helps fitted spatial reconstruction substantially beyond matched exposure. The existing VecSetX-to-generator path can represent accurate fitted geometry under the camera-oriented target convention.
2. Broader generator adaptation alone is not a solution for transfer across camera views/orientations. It does not remove the current full-surface-conditioning limitation or justify full-dataset promotion.
3. The next branch is the held-view transfer failure, not another automatic duration extension or a claim that SAM3D cannot generate rotated shapes. Precise native recovery is still incomplete, but increasing fit accuracy has not improved transfer.

These statements cover four fitted identities, four fitted/three reserved views and one training seed. No unseen-object claim follows. Natural reserved views change both visual features and geometric orientation; this experiment does not separate those two changes.

## Exploratory fitted-target resemblance, with a symmetry caveat

Comparing all24 reserved correct predictions against all16 fitted target supports, two broader-arm bathtub predictions (view006, both draws) have95.86%/97.48% IoU with the same object's fitted view007 target, versus about39.1% with their own target. This is observable resemblance, not proof of a retrieval mechanism. Crucially the two target orientations differ by about180°, while their own two-voxel F-score in the camera-normalized cube is94.12%: the shape is approximately symmetric. It would be misleading to present a180° pose error alone as a catastrophic shape failure. All comparisons are retained in `fitted_target_comparison_*.json`; they do not determine the primary branch.

## Next bounded question

Before another training intervention, the lowest-cost discriminating check is inference with the visual tokens zeroed on the already trained broader checkpoint, using the same fitted/reserved point features, targets and sampling noise. Compare correct/wrong surfaces and retain a same-run natural-visual reference. Whole-visual-zero inputs were present for50% of training, so this is a trained condition state rather than an entirely new token-removal state.

Question: **does the geometry path handle reserved camera orientations when the visual stream is absent, or does it still fail?** Prior visual-dropout findings concerned the original fixed-target contract; they do not answer this for a newly camera-oriented target distribution and broader generator adaptation. Training with dropout does not logically guarantee that the visual-present model and visual-zero model have the same transfer behavior.

- Accurate reserved reconstruction with visuals zero, but poor with visuals present: supports visual/geometry frame interaction as the immediate obstruction. Then reconcile their frame relationship or conditioning policy, without changing VecSetX.
- Reserved failure persists with visuals zero while fitted geometry remains accurate: localizes a geometry-orientation transfer failure even without visual competition. A matched jointly rotated point/target augmentation or explicit input-alignment route becomes relevant; select one rather than launching both automatically.
- Both fitted and reserved zero-visual predictions fail: inability to use points independently remains a bottleneck; do not claim a purely held-orientation issue from the natural-input results.

The inference-only follow-up is now implemented in [SHARED_ORIENTATION_VISUALS_HANDOFF.md](SHARED_ORIENTATION_VISUALS_HANDOFF.md), with local schema/provenance checks passed and GPU execution pending. No additional training or full-dataset run is justified by these results alone. The existing broader checkpoint suffices for the next check. Visual sensitivity would identify a causal input contribution, not uniquely prove an internal coordinate-conflict mechanism.
