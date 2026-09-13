# Camera plus visual dropout — returned Stage-1 geometry

## Material Passport and validation

Origin: user-returned GPU report and `camera_dropout_geometry_bundle.zip`; academic-research-suite validation workflow, performed inline. Status: returned execution ANALYZED; saved geometry and summaries reproduced on CPU. No independent local CUDA rerun. Scope: four fitted identities, four fitted/three reserved views, two sampling draws, unchanged existing 1000-step checkpoint, CFG 0 and 25 sampling steps. No Stage 2, training, or new identities.

Archive SHA256: `c351e2224ade1a98aed17513fdd8ceb833672d9af99901961c4149171251eb81`.

All 57 manifest members and ZIP CRCs passed. Bundled and separately pasted JSON agree. All reported source/reference hashes match local files. Checkpoint parameter digest matches the historical camera-dropout endpoint; the runner asserts parameters stayed unchanged. All 112 historical native-loss values replay **exactly** (maximum absolute difference zero). Seven input groups, per-object/per-draw sampling-noise hashes, actual target-support content, bool array shapes, occupied counts, raw IoUs and raw occupied-center distances were checked. All 112 predictions, including wrong surfaces, were retained.

The CPU analysis reuses the unchanged `pose_shape_geometry.py`, previous known-rigid/revoxelization/wrong-object controls and previous oracle results with their source/result hashes verified. No scaling, reflection, trimming or nonrigid alignment. The prespecified pose witness chooses the better minimum precision/recall among raw identity and the measured RMS-selected rigid transform. This is not a globally optimal registration certificate. Raw and rigid-only results remain separately available.

Sources: `camera_dropout_geometry_analysis/validation.json`, `complete.json`, `summary.json`, `predictions.jsonl`; executable `analyze_camera_dropout_geometry.py`.

## Main result: the missing same-policy comparison fails

Reserved views of the same four fitted identities:

| Input / training policy | Fixed-frame IoU | Raw one-voxel F-score | Rigid-only F-score | Prespecified raw-or-rigid witness F-score |
|---|---:|---:|---:|---:|
| Camera, original | 52.19% | 72.83% | 80.47% | 82.99% |
| Camera, visual dropout | **56.17%** | **67.13%** | **79.26%** | **79.26%** |
| Oracle, original | 66.95% | 84.47% | 85.32% | 88.78% |
| Oracle, visual dropout | **97.17%** | **99.997%** | **99.95%** | **99.997%** |
| Camera dropout, wrong surface | 61.52% | 78.84% | 87.82% | 88.97% |

F-score measures bidirectional proximity within 1/64, not exact occupancy. One-voxel and two-voxel measurements are both in the JSON. The large same-policy camera/oracle gap remains after pose adjustment; it is not explained away by the original versus dropout policy confound.

The camera model is substantially fitted on training views: 88.32% IoU and 99.63% raw one-voxel F-score, versus 97.80%/100% for oracle dropout. It does not pass the strict fixed-frame fitting gate. Its fitted drill recall is 97.95%, narrowly below the prespecified 98% reference-relative requirement; that small margin is not the basis for the main diagnosis. The much larger reserved-view failures are.

Native camera reserved loss improved from .07718 to .05381 under dropout, while its pose-witness F-score went **82.99% → 79.26%**. Therefore F5 must not be summarized as camera geometry or view dependence being resolved. Its positive sampled reconstruction claim applies to oracle alignment plus dropout (F9). Dropout is not a demonstrated general camera-frame repair.

## Object-level outcome and supplied-surface dependence

All values below are reserved-view means over three views and two draws per object. Reference test: both precision and recall at least 95%, and at most two percentage points below that object's oracle-dropout reference.

| Object | Correct-surface witness F-score | Wrong-surface witness F-score | Correct − wrong | Meets shape reference? |
|---|---:|---:|---:|---|
| `146b6c1f…` | 99.78% | 88.62% | +11.16 pp | Yes |
| Shield `479bbf90…` | 78.06% | 99.74% | −21.68 pp | No |
| Bathtub `e846574d…` | 77.25% | 99.90% | −22.64 pp | No |
| Drill `fe200ce0…` | 61.94% | 67.62% | −5.68 pp | No |

Only one of four objects passes the shape-reference gate; the fixed-frame and strong surface-dependence gates also fail. The mean wrong-surface advantage is 9.71 pp. At individual level, correct wins 14/24, wrong wins 9/24, with one tie: a few large failures drive the reversed mean. Do not say wrong surfaces improve every prediction. Swaps contradict the visuals and test conditional sensitivity, not natural-input benchmark accuracy.

## Concrete failure: bathtub input produces the fitted shield

An additional **exploratory post-result** CPU check compared all 24 reserved correct-surface predictions to all four fitted targets using the same raw/rigid rule (96 comparisons, 72 new registrations). This was not a preregistered identity-classification experiment; no cases were excluded.

For bathtub view `e846574df9334973955c80286af47ba7_000`, both draws match the **shield target** at raw F-scores **99.966% and 99.890%**. They match their actual bathtub target at only **31.74% and 31.80% even after the measured rigid alignment**. The near-perfect shield match requires no pose fitting. Both noise draws show the same wrong-object outcome; they are not independent training replicas.

For this input, the cyclic wrong-surface intervention supplies shield tokens to the bathtub visual input, and the output returns close to the bathtub. This is not a simple rule of copying whichever object's surface was supplied. Two bad shield predictions are somewhat closer to the fourth object (~53–55%) but do not reconstruct that object accurately; the remaining 20/24 predictions have their intended identity as the best of the four measured candidates. The mechanism is heterogeneous.

This is direct evidence of a wrong-shape failure in a natural camera-frame condition, beyond a generic failed-registration argument. It is consistent with an orientation-dependent learned interpretation or retrieval of fitted shapes. It **does not localize** the cause to VecSetX, its normalization, projector, cross-attention, or denoising dynamics; it does not prove a discrete classifier exists internally. Aligned oracle success may itself exploit stable per-object codes, so it does not establish general geometry transport.

Sources: `camera_dropout_geometry_analysis/cross_identity.json`, `check_camera_dropout_identities.py`, `bathtub_shield_failure.png`. Figure uses the diagnostic bathtub view, draw 0, selected after the results; the full numerical analysis retains every reserved prediction.

## What this narrows down, and the next coordinate branch

Starting evidence: F1 validates the inspected transforms; F2 rejects a completely broken frozen-VecSetX pathway; F3 isolates a finite-budget rotation burden under the original policy; F9 establishes an accurate oracle-dropout endpoint; F11 rejects a universal sub-5° requirement and a general output-alignment rescue. This result completes the missing same-policy cell. There is now sampled evidence of a large **input-frame treatment penalty despite visual dropout**, including wrong-object outputs. Camera versus oracle changes orientation and normalization together; F3 isolated rotation under the original policy, not quantitatively under dropout. Do not attribute all 20.74 pp to rotation alone.

The next useful step is a coordinate **intervention**, not another measurement of this four-object camera/dropout gap, another global-axis audit, a new encoder, or an unconditional full run.

Two meaningful branches depend on the output contract (asked asynchronously; no answer assumed):

1. **If a known camera/robot frame is acceptable:** the leading repair candidate is to train surface input and spatial target in a shared observable orientation. Re-encode physically transformed target geometry; never rotate latent channels as xyz. Keep VecSetX and the existing adaptation scope initially. Compare a short shared-frame training arm with the existing fixed-asset-frame task at matched budget and inputs, with its own decoded-target reference and image-only/wrong-surface controls. Training target construction may use complete GT, as current supervision already does; conditioning must not receive target occupancy. Verify VAE fidelity and clipping for continuously rotated meshes before attributing any outcome to learning. Define centering/scale separately and preserve transforms needed for sparse touch; per-patch bbox normalization is not a future global frame solution.
2. **If original asset axes are required:** frame recovery remains part of the task. A usable repair must estimate camera-to-asset alignment from available inputs, or change the training objective/architecture to handle that uncertainty. The known inverse is a positive control, not a deployable estimator. Small jitter addresses residual alignment error only and cannot solve arbitrary missing rotation. Do not begin a precision pose head merely because exact oracle alignment succeeded; F11 limits that inference.

A shared-frame success would remove the asset-axis conversion burden by changing the supervised contract, not prove asset-pose recovery. A failure with verified target encoding would test prior/adaptation compatibility with that new task, not prove that all coordinate strategies fail. Before any full run, the selected intervention must demonstrate useful **new-object** surface conditioning, since F6/F7 already show fitted-object success is insufficient. The previous 32-object development pool is not an untouched confirmation set.

No new GPU job or production change is introduced with this report. The output-frame answer selects the next implementation. No full training run is justified by this result.

## Interpretation audit

11/11 statistical fallacy categories checked. Simpson/ecological: object and individual results reported, including 14 individual correct wins despite reversed mean. Berkson/selection: four existing fitted identities, no population claim. Collider: no outcome-based adjustment of the primary sample. Base rates: this is support proximity, not clinical or population diagnostic accuracy. Regression to mean: matched saved checkpoints/noise with all cases, no selected-extreme before/after efficacy claim. Survivorship: all 112 expected predictions present. Look-elsewhere/forking paths: primary rule frozen in the handoff, additional identity matching explicitly exploratory; no p-values or formal equivalence claim. Causality: matched frame treatment supports the finite-task comparison, not a localized mechanism or universal limitation. Reverse causality: inference inputs intervene before outputs; no inverse causal inference is drawn. No views or draws are treated as independent objects; only one training seed is available.
