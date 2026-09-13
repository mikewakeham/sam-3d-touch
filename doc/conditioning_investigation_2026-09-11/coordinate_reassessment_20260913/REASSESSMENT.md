# Coordinate diagnosis: evidence review before another intervention

## Material Passport

- Date: 13 September 2026.
- Scope: Stage-1 oracle upper bound; experiment planning with a completed local numerical audit.
- Inputs: F0–F19 ledger, frame-factor and shared-target/scope findings, native VecSetX/readout results, actual production normalizer,64 available local surfaces, latest oracle/constant rollout reports and supports.
- Execution: CPU only; no new neural fitting, GPU sampling, production edit, external upload or simulated model result.
- Status: local arithmetic reproduced; proposed experiment outcomes below are conditional reasoning, not observed outcomes.

## Reassessment of the last proposals

Do not run the proposed strict shared-normalization/surface-only fit next. It changes target distribution and removes visual inputs; a two-arm control can match the visual policy but cannot distinguish a benefit from easier pretrained target scale versus removal of coordinate conversion. A negative fit still leaves target-distribution/optimization explanations. It is a potential intervention, not a clean certificate that coordinates are irrelevant. Metadata injection similarly tests a learned side-channel; earlier unsuccessful position encoding lowers its priority, without proving all metadata designs fail.

The more direct comparison already exists as completed training: full-data camera/dropout versus oracle/dropout. The recent oracle/constant probe measures sample-specific surface utility, not the cost of the frame treatment. The camera model has not received the same sampled-shape assessment. Finish that comparison before paying for a new target convention.

## Questions that must remain separate

| Question | Existing evidence | Remaining gap |
|---|---|---|
| Are actual input points associated and transformed consistently with the target? | Independent earlier mesh/frame audits and target regeneration; current source matches runtime; same-object oracle stability. | Current full-run observations' raw geometric evidence is not in the latest ZIP. Only3 latest identities overlap local surfaces; none of its exact views do. |
| Does full-surface normalization destroy required center/scale information? | Canonical full-object normalization is recoverable in the ideal complete-surface case; new finite-sample audit below measures small errors. | Does not prove easy extraction from feature tokens, sparse-patch recoverability, every cluster record, or correct mesh provenance. |
| Does frame treatment materially affect full-data shape learning? | Four-object controlled factorial establishes finite-budget rotation cost; full-data native losses are similar and inadequate as reconstruction endpoints. | Need paired sampled camera/oracle shapes, with raw and proper-rigid results, split and visual state preserved. |
| Does a useful VecSetX-to-SAM mapping exist? | Native VecSetX surface reconstruction;24-object direct readout reached96.57% fitted target IoU using processed features; existing four-object flow fits can be accurate. | Efficient reusable mapping at current dataset/interface scope remains unresolved. Absolute incompatibility is not supported. |
| What explains residual failure under verified alignment? | Full-data oracle improves correct-surface generation but not enough; later scope experiments demonstrate trainability in another contract. | Could involve restricted adaptation, optimization, transfer or spatial feature access. Do not relabel all of these as an unidentified axis mismatch. |

## New local normalization experiment

`audit_normalization.py` executes the actual `TouchEncoder.normalize_points_for_vecsetx` method extracted by AST, on CPU float32 oracle-transformed points, and independently recomputes the arithmetic in float64. It reads all64 locally available surfaces (32 identities;8192 points each), before FPS. It retains source/input hashes, per-sample values and point-ID-matched view comparisons in `normalization_audit.json`.

It compares:

1. Exact normalization inversion using the measured center/radius.
2. Recovery of the known centered maximum-extent-one dataset convention from normalized geometry alone: recenter normalized points and divide by their maximum bbox extent. No target mesh/support is used for this calculation.
3. How much the recent proposed `q/2` target convention would rescale the current object.

Results:

- CPU production normalization versus independent float64: maximum coordinate difference1.664e-7.
- Inverting normalization: maximum coordinate error3.833e-8 object units.
- Geometry-only canonicalization: median per-sample RMS displacement0.03320 target voxels; worst RMS0.71774; largest individual displacement0.84377 voxels. This is displacement relative to the recovered original point cloud, not mesh/VAE reconstruction error.
- Same-object matched points across saved camera views: maximum oracle-coordinate discrepancy3.577e-7; normalized discrepancy1.014e-6.
- Proposed strict `q/2` target scale relative to original: min0.58082, median0.95797, max1.00321. Some targets would be about42% smaller in linear scale. This is a potentially substantial change to the pretrained target distribution, not a negligible bookkeeping adjustment.
- Three identities overlap latest rollout supports (six historical input views, zero exact latest views). Their locally saved targets equal the returned target latents exactly. All their original and recanonicalized surface points lie within one voxel of the latest decoded target support. This is a narrow cross-check, not whole-bank certification or target recall.

Interpretation: large irrecoverable center/scale loss is not supported for these full surfaces. A model may still struggle to extract the deterministic relation from latent tokens. This test does NOT independently establish correct orientation: an incorrectly rotated normalized cloud could also pass canonicalization. It also does not validate CUDA/FPS or unavailable original meshes. These limits prevent a numerical audit from becoming an unwarranted neural or full-dataset claim.

## Counterfactual experiment review

These are possible outcomes, not predictions from a trained simulation.

| Candidate | If it improves | If it does not | Priority |
|---|---|---|---|
| More position metadata | Side-channel helps; extra adaptation must be controlled; not automatic explicit spatial alignment. | May fail to learn side-channel; does not rule out coordinate-aware mechanisms. | Low now; previous variant failed and measured normalization recoverability is strong. |
| Strict normalized targets + surface-only fit | Useful changed-task candidate; could benefit from target scale/distribution, not uniquely removed conversion. | New-target prior mismatch/optimization still open; negative cannot certify original coordinates. | Park. |
| Repeat four-object oracle fitting | Repeats known limited capability. | Introduces replay/debug questions. | Do not repeat. |
| Remove pointmap during new training | Pointmap involvement established, not specifically a coordinate conflict. | Does not rule out RGB/view dependence or learned representation issues. | Conditional after full-data frame-by-visual comparison. |
| More sampling steps | Addresses numerical generation accuracy, not input-coordinate correctness. | Leaves training/interface explanations. | Useful later; not the current coordinate discriminator. |
| Full-data camera checkpoint versus existing oracle/constant | Directly measures the already-trained frame intervention at the scale in question. | A null result limits the observed benefit of oracle, rather than just adding another failed architecture. | Highest next GPU information value, alongside exact input export. |

## Recommended next execution, still no new training

Complete the existing full-data frame comparison. Restore the camera/dropout last checkpoint, verify matched run/data/target/checkpoint scope, and use the exact prior32-object/two-view/noise bank. Assess correct and the same fixed wrong surface, visuals present/zero, with the same native/fixed-time and Stage-1 sampling conventions. Reuse oracle/constant results only if actual target/visual/noise/decoder hashes agree. One training seed is not a population causal guarantee. Report object-level effect distributions rather than treating views/draws as independent objects.

At the same time, export actual runtime pre-encoder points, normalization/FPS outputs and transforms for both frame paths on that bank, along with independently constructed target-frame reference geometry and target provenance. Use the exact production methods. Read-only geometry validation should happen before attributing model differences to frame handling. Raw source mesh data or sufficient independently sampled references are needed for a true geometric check; replaying the same inverse twice is insufficient. Coordinate evidence can be exported without launching another full fit. No cluster command is implemented by this note.

Decisions:

- **Input contract fails:** repair the identified issue and assess its affected scope before interpreting oracle/camera differences or retraining.
- **Oracle improves raw scores but not pose-adjusted shape:** frame treatment mainly improves measured output pose at this checkpoint/sampler; a deployable pose estimator would not explain the remaining intrinsic-shape failure. Registration is a witness, not globally optimal alignment.
- **Oracle improves pose-adjusted shape and useful surface dependence:** alignment has a practical shape benefit at full-data scale. The oracle residual is a separate remaining limitation; retain oracle as the intervention reference while studying that residual.
- **Camera matches/beats oracle with useful correct-surface dependence:** perfect supplied alignment is not a demonstrated net remedy at this scale. Do not promise inference pose recovery will solve the full-surface task. This does not show rotation is universally harmless or that better fitting cannot expose a gap.
- **Camera/oracle ordering changes with visual-present versus zero:** frame-treatment effect depends on visual availability. This motivates targeted fusion/visual controls, not a conclusion that pointmap specifically conflicts geometrically. Both models were trained with50% visual dropout; zero is an exercised input state, but inference comparison does not isolate training causation.
- **Both remain similarly poor after checked inputs, with weak condition utility:** further global-coordinate fixes lose priority; examine adaptation/optimization/spatial access. Do not infer impossible representation from finite fits.

A null/bad reconstruction does not prove all coordinate-aware architectures impossible. The attainable stopping point is scoped: the actual data transform is checked, the empirical cost of frame treatment is measured, and residual failure is documented even with known alignment. The goal is to localize the obstacle, not exhaust all imaginable normalization variants.

## Reproduction

From the repository root, the completed CPU analysis used:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /private/tmp/sam3d-training-tests/bin/python \
  doc/conditioning_investigation_2026-09-11/coordinate_reassessment_20260913/audit_normalization.py
```

This environment is local. Do not copy its path into cluster commands. Saved results are reproducible from the64 local files; no GPU execution occurred.
