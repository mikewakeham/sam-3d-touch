# Pose versus shape in the saved predictions — 12 September 2026

## Evidence and validation

The supplied `alignment_geometry_bundle.zip` has SHA256 `ca514549e67974e85ecb8ae4379b9584d1dcbdbe19cbe7c0e1e314022b3e4f77`. All 705 manifest members and ZIP CRCs passed. All four alignment shards now reproduce the supplied aggregate exactly, including the previously missing 15° raw report. The natural-camera report exactly matches the historical multi-view reference.

Numerical analysis ran on CPU using `/private/tmp/sam3d-geometry-analysis/bin/python` (NumPy 2.5.3, SciPy 1.18.1). All 1,232 saved predictions were included: 1,176 from the 21-condition rotation study and 56 natural-camera predictions. Actual occupancy dimensions/dtypes, per-object target equality, occupied counts, raw IoUs and raw occupied-center distances reproduce the reports. There was no model inference or training.

Artifacts: `pose_shape_bundle/validation.json`, `pose_shape_analysis/controls.json`, `complete.json`, `predictions.jsonl`, `summary.json` and the two figures. Core/source hashes and the completed row-file hash are recorded in the result. Four fitted identities, seven views (four fit/three reserved), two sampling seeds; no new-object geometry is assessed.

## Registration and metric controls

- Proper rigid transforms only: rotation and translation, no reflections, scaling, trimming or deformation. Multiple proper-axis and PCA starts, coarse/fine balanced bidirectional ICP, full-cloud squared-distance selection; raw identity and known inverse are retained as analytic candidates.
- Twelve known continuous rigid-transform controls recovered 100% one-voxel precision/recall, with maximum p95 residual below 6.3e-16. Four revoxelized controls also achieved 100% within one voxel; worst p95 was .01104, below 1/64. Revoxelization used an unbounded lattice to avoid introducing artificial clipping.
- All twelve ordered wrong-object GT pairs failed: the highest registered one-voxel F-score was 30.94%. Four supplemental known transforms of aligned predictions also passed.
- The registration is **not globally certified**. Its squared-distance objective can lower RMS/tail error while worsening thresholded F-score. Report raw and registered F-scores together. Do not call the selected transform the mathematically best F-score or infer shape damage from an optimizer failure alone. As a secondary sanity check, keeping the better raw/registered F-score per sample does not make the natural-camera or original-oracle results approach the dropout reference.
- One initial revoxelization control correctly stopped because rotating a shape left the original cube; clipping was removed from the control before prediction analysis. One supplemental ad hoc control process exited with code 139; it supplied no evidence. The versioned supplemental script, run with single-thread numerical-library settings and fault reporting, completed successfully. The main 1,232-prediction process completed without failure.

**Metric distinction:** IoU requires exact occupancy agreement on the fixed voxel grid. F-score here combines bidirectional precision/recall within **one voxel spacing (1/64)**. It allows small spatial discrepancies; 100% does not mean exact geometry. The second reported tolerance is two voxel spacings. Comparing raw IoU with registered F-score would confound metric choice with registration, so comparisons below always show raw F-score first.

## Result 1: the sub-5° precision implication was too strong

Reserved views, averaged over all four identities, three views, two seeds and six signed axes at each nonzero angle:

| Input residual | Raw one-voxel F-score | After known inverse | After fitted rigid transform | Individual predictions failing ≥95% precision AND recall, before/after allowing either raw or fitted transform |
|---|---:|---:|---:|---:|
| 5° | 99.67% | 78.78% | 99.29% | 2/144 / 2/144 |
| 15° | 95.87% | 45.91% | 94.35% | 17/144 / 17/144 |
| 30° | 88.32% | 29.93% | 89.06% | 40/144 / 40/144 |

The individual count in this table uses the absolute 95% precision/recall test. The original plan's stricter object-average comparison against the aligned reference (at most two percentage points worse as well) is separately recorded in `summary.json`; these are different summaries, not replaced acceptance rules.

Most 5° cases were already shape-close before registration. Their lower exact-grid IoU should not have been translated into a general requirement for sub-5° pose estimation. There are exceptions: both failing 5° predictions are bathtub view `e846..._000`, draw 0, for −x and +y. Raw F-scores are 77.37% and 77.67%, with p95 distance four voxels. Thus “5° is always harmless” is also false. This is a sparse, view/noise-dependent failure in this selected four-object task, not a population failure rate.

Applying the known inverse lowers F-score in 144/144 cases at 5°, 144/144 at 15°, and 143/144 at 30°. This rejects the simple account that outputs generally rotate with the supplied small perturbation and can be corrected by its inverse. It does not exclude other output-pose errors or symmetry-equivalent outputs.

## Result 2: some large-rotation failures include genuine shape changes

The drill is the principal vulnerable identity. Averaged over the six signed axes, its raw F-score is approximately 99.93% at 5°, 85.87% at 15°, and 64% at 30%; the rigid-registration curve remains poor at larger angles. The other identities are substantially less sensitive. Do not hide this heterogeneity behind a single angular threshold.

The selected worst residual example, drill `fe200..._000`, +30° z, has registered F-score **16.38% at one voxel and 33.64% at two voxels**, with p95 error .12514 (about eight voxels). Projections show both orientation change and a visibly altered/thickened head/body. Its predicted support has 2,617 occupied centers versus 1,301 in GT. Its covariance eigenvalue ratios relative to GT are approximately **6.77, 1.11 and .936**; rigid transforms cannot change those eigenvalues, and uniform scaling would multiply them all by the same factor. For comparison, the known rotated/revoxelized drill control has ratios 1.043, 1.005 and .996. These checks corroborate a non-pose change in this example; they are not a global proof that every residual registration error is irreducible shape error.

The figures use explicitly post-result diagnostic examples, not an unbiased subset. `selected_rigid_invariants.json` and `revoxelization_invariants.json` record these supporting quantities and their sampling/discretization limits.

## Result 3: natural camera errors are partly pose, but alignment does not generally reach the accurate endpoint

These are the actual saved natural-view predictions, not artificial rotations of oracle inputs. Reserved means:

| Training/input | Original IoU | Raw one-voxel F-score | F-score after rigid fitting | Two-voxel F-score after fitting |
|---|---:|---:|---:|---:|
| Camera, original policy | 52.19% | 72.83% | 80.47% | 92.13% |
| Oracle, original policy | 66.95% | 84.47% | 85.32% | 95.35% |
| Oracle, visual dropout | 97.17% | 99.997% | 99.95% | 100% |

Rigid correction can be very useful for a particular prediction: one natural-camera drill goes from **2.03% to 79.72%** at one voxel and 96.53% at two voxels. This is substantial pose error accompanied by some residual geometry error, not a universally hopeless reconstruction.

Because fitted RMS registration can worsen F-score on already reasonable cases, an exploratory per-sample best-of-raw/registered check gives 82.99% camera, 88.78% original oracle and 99.997% dropout oracle. The substantive conclusion remains: the tested rigid correction does not generally recover the accurate dropout endpoint. At two voxels the original models are closer; “all geometry is wrong” would be too strong.

**Comparison limit:** camera/oracle under the original policy is a matched frame comparison; camera-original versus oracle-dropout changes both frame and training policy. This bundle does not contain generated geometry for the already trained **camera-plus-dropout** checkpoint. Its historical native loss .05381 versus oracle-dropout .02273 cannot be converted into a shape score. We must not repeat the loss-to-geometry inference just corrected.

## Consequence for the next branch

Branch A is now completed: the latest failures cannot generally be explained by output pose alone, but the original exact-IoU sweep exaggerated how broadly small-angle errors damage shape. This is neither a proof that precise canonicalization is required nor a proof that target/output registration solves conditioning.

Before selecting a new training repair, complete **one missing same-policy comparison using existing weights**: sample camera-plus-dropout on the same fit/reserved views and seeds, with historical replay and the same target-referenced/pose-adjusted analysis. This is a bounded inference-only control, not another training run. It is directly motivated by the metric mismatch established here, and prevents attributing a training-policy benefit to coordinates.

- If natural camera-plus-dropout reaches the aligned reference's shape criterion, a precise input-alignment repair has not been shown necessary for this fitted-identity task. Next test whether its surface utility survives new identities rather than augmenting synthetic perturbations automatically.
- If it remains worse after pose adjustment, a residual frame burden under the improved policy is established on sampled geometry. Then choose the intervention according to the required output frame: camera-oriented training targets are a direct alternative if original asset axes are unnecessary; residual-rotation augmentation tests local robustness when a canonical-frame interface must be retained. Small jitter is not a solution to arbitrary camera-to-asset rotation by itself.
- If the outcome is mixed, preserve object-level distinctions; do not claim one angular tolerance or a universal architecture limit.

The output-frame preference and new-object generated-shape utility remain unresolved. No full overnight training run follows from these four-object results alone. VecSetX and the eventual sparse-touch constraints are unchanged. No new GPU job is implemented in this results report.
