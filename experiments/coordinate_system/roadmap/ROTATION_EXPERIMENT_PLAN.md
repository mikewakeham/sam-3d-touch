# Rotation penalty: experiment and presentation plan

Status: implemented in `scripts/rotation_loss/`; CPU checks and CLI entry points pass. Pretrained encoder and checkpoint GPU runs have not been executed. This is one additional diagnostic with no optimization. It does not introduce another full training run or change production source. Updated 2026-09-15.

See [runnable commands and exact implementation](../scripts/rotation_loss/README.md). Current scripts do not load a decoder; all headline scores are measured in latent space or with the native velocity objective. Any optional decoded reconstruction discussion below is outside this implementation.

## Question and claim boundary

**Question:** How much supervision error can arise from representing the same shape in a different orientation, are actual prediction latents closer to rotated target encodings, and does oracle improve native flow loss in a matched comparison?

These are separate measurements:

1. An encoder experiment quantifies the orientation sensitivity of the actual SS target means. It establishes the cost of a rotated answer even when its physical shape is unchanged.
2. Fixed prediction latents are scored directly against original and separately encoded rotated targets. No decoder is involved in those scores.
3. A matched camera/oracle checkpoint assessment measures the native flow-velocity loss actually used for training.

Physical-geometry controls establish that the constructed inputs really differ only by rotation. Decoded F-scores are excluded from the oracle-motivation argument. Optional decoding is only for supporting reconstruction checks or subsequent generation-quality evaluation.

None alone proves that oracle is necessary, that the original shared-object target convention is wrong, or that oracle solves conditioning. Oracle is a controlled way to remove the input-to-target orientation conversion. Whether that helps this learner is answered by matched camera/oracle training comparisons, particularly the no-pointmap/no-dropout pair already planned for assessment.

This experiment does not repeat the earlier frozen-model search for a preferred target yaw. That search did not identify a systematic correction. It also differs from the old experiment that rotated conditioning: here we rotate identical target geometry without allowing a generator to change its shape.

## Population and fixed settings

- Select 16 training and 16 validation object IDs from the existing full-surface manifest and production object splits, using seed 29. Selection is by ID, before any metric is inspected. Views are not independent samples in this encoder experiment; each object supplies one source mesh.
- Keep train/validation summaries separate, plus an explicitly descriptive pooled summary. There is no fitting in this experiment, so these labels indicate the associated full-training population, not a new train/test procedure.
- Use the same frozen SS encoder and decoder as target preparation and Stage-1 reconstruction. Encode deterministic posterior means in FP32, with sampling disabled and models in evaluation mode.
- Keep the stock 64³ occupancy grid, 8×16³ mean, and 4096×8 flattening. Do not rotate latent channels or assume a spatial permutation of the latent equals re-encoding rotated geometry.
- Use all three object-coordinate axes. Continuous angles: 0°, 5°, 15°, 30°, 60°, 90°, 180°. Symmetry and periodicity mean an increasing angle does not guarantee increasing loss.
- Include octopus `003d24749f2e4047b95f9d04a7c7957b` as a declared visual example. If it is outside the selected population, keep it separate from aggregate statistics. Use one second asymmetric example chosen before looking at loss, if needed for a clearer demonstration.

32 objects is an initial descriptive bank, not a power guarantee. Report object-level uncertainty. Extend the bank only if a population claim remains uncertain; proving that a target is orientation sensitive does not require universal failure on every symmetric object.

## Part A: exact rotations at the original target scale

For each object:

1. Load `objects/<id>/model.obj` and `generated_data/<id>/object_transform.npz`. Apply the existing saved source-to-normalized-object transform using the target builder's `load_normalized_mesh`.
2. Construct the stock unrotated occupancy once with the existing target voxelizer. Encode it to `z0`. Record agreement with the stored target mean; an unexpected discrepancy must be understood before using this object as evidence about its training label. This is a label reference for newly selected objects, not another oracle camera-inverse audit.
3. Create exact grid rotations: identity, 90° and 180° about X, Y and Z. Reuse `coordinate_audit/frame_contract.py::rotate_grid`. These are proper signed-axis permutations about the cube center and preserve occupancy exactly.
4. Encode each rotated occupancy separately. Keep identity only once; repeat its encoding once to measure numerical reproducibility. Decode selected examples only if optional reconstruction checks are requested.
5. Apply the exact inverse grid rotation to the rotated input occupancy. Its IoU with the original must equal 1 and occupied count must be unchanged. This is the clean known-same-shape control.

This part stays at native target size and does not need arbitrary-angle mesh clipping, interpolation, or a changed scale. It is the strongest evidence that orientation alone can change the target mean under the actual training convention.

## Part B: continuous angular curve

Rotating a mesh that touches the native cube boundary can move it outside the cube. The stock target voxelizer clips vertices; using it blindly would change the shape and invalidate the test.

For each object, construct one padded reference:

1. Use the same saved object frame and origin as Part A. Compute `r = max_vertex ||p||` and a single scale `s = 0.45/r`.
2. Set the reference vertices to `p_ref = s*p`. All vertices then lie within a radius-0.45 sphere, which fits inside the target cube under every proper rotation. All triangle interiors also fit.
3. Rotate with `p_theta = p_ref @ R_theta.T`. Do not recenter or recompute scale at any angle. The inverse is `p_theta @ R_theta`.
4. Voxelize with the same Open3D grid bounds and resolution, but without clipping. This requires a small experiment-local voxelizer because the production helper contains unconditional clipping. Check that the already padded vertices fit, then use them directly.
5. Encode each occupancy. Compare with the padded zero-angle mean and support, not with the larger stock target. Decoding is optional for selected examples.
6. At 90°/180°, also compare mesh-voxelized occupancy with the exact permutation of the padded zero-angle grid. Record the discrepancy as a discretization control. Do not silently replace the continuous curve with a grid-interpolated curve.

There are 19 distinct settings per object: one zero plus six nonzero angles on each of three axes. The padded zero differs from the training target; present this curve separately from the native-scale Part A table. Both parts are needed: Part A supplies the clean native control, Part B supplies the readable angle curve.

## Quantitative outputs

### Target and native flow-supervision discrepancy

Primary latent measure:

`D(theta) = mean((z_theta - z0)^2)`

Use every entry of the target mean, matching unweighted shape MSE reduction. Report zero-repeat discrepancy, raw MSE, per-object median/mean, and object-level 95% bootstrap intervals. Bootstrap object IDs and retain their whole axis/angle curves; do not treat axes or repeated computations as independent objects. Keep native and padded measurements separate.

The source of the flow calculation is `sam3d_objects/model/backbone/generator/flow_matching/model.py`, not a proposed new loss:

`a(t) = 1 - (1-sigma_min)*t`

`x_t = a(t)*epsilon + t*z0`

`u0 = z0 - (1-sigma_min)*epsilon`

With paired noise, two orientation-specific velocity labels satisfy:

`mean((u_theta-u0)^2) = D(theta)`

This is **velocity-target discrepancy**, not a measured generator loss. The two labels normally accompany different noisy inputs.

For a stricter counterfactual at the same noisy input, an endpoint estimate implied by velocity `v` is:

`z_hat = (1-sigma_min)*x_t + a(t)*v`

A velocity whose implied endpoint is exactly the rotated mean therefore incurs native shape MSE:

`L_rotated_endpoint(t) = D(theta)/a(t)^2`

This is an analytical loss for an otherwise exact but rotated endpoint. Label it that way; do not present it as a measured checkpoint result. Read `sigma_min` and any shape-loss weight from the installed pipeline configuration. Report `D` as the primary result, with this calculation as training-objective interpretation. At sigma_min=0, factors at t=0.2/0.5/0.8 are 1.5625/4/25; these factors are mathematics, not experimental results. The identity was checked numerically locally for sigma_min=0 and 0.001.

No separate generator target-yaw sweep is planned. Actual native checkpoint losses belong to the existing paired evaluation, where conditioning, settings and targets correspond to the trained setup.

### Geometry and reconstruction controls

Let `G0/G_theta` be voxelized inputs and `Q0/Q_theta` their decoded supports.

The input-geometry controls apply to the full bank. All Q-based reconstruction measures below are optional supporting checks on selected examples; they are not headline evidence for a latent-training claim.

| Measure | What it separates |
| --- | --- |
| Source points before/after the known inverse | Same physical geometry; inverse round trip should be numerical zero using identical sampled points. |
| Raw input support IoU and proximity F-score: G_theta versus G0 | Orientation can produce low target-frame agreement despite unchanged shape. |
| Input support after known inverse versus G0 | Exactly perfect for native grid permutations; arbitrary-angle voxel centers retain discretization error. |
| Q_theta versus G_theta | Reconstruction fidelity at the rotated orientation. |
| Raw Q_theta versus G0 | Pose plus reconstruction error. |
| Known-inverse Q_theta versus G0 | Residual reconstruction/discretization error after removing the declared rotation. |
| Q0 versus G0 | Frozen VAE reconstruction reference at zero angle. |

Reuse `shared/pose_shape_geometry.py::points/metrics` for occupied voxel-center distances and precision/recall/F1 within one and two voxels. Multiply distance by 64 for presentation in voxel units. Define the metrics explicitly: F1 within two voxels is not an F-beta=2 score.

For exact rotations, inverse-permute occupancy and calculate IoU directly. For arbitrary angles, inverse-transform occupied voxel centers and calculate distances/F-scores without snapping them into a new grid; snapping would introduce another discretization step. Any optional resnapped IoU must be labeled separately.

If decoded rotated targets are less faithful than the decoded zero target, report that change. A large latent discrepancy is still real, but the decoded examples then contain a VAE effect as well as pose. Do not imply perfect reconstructed shape from a known-same-mesh input.

### Cheap context: different-object discrepancy

Using zero-angle means already computed, pair distinct object IDs in a fixed seeded order within each split and calculate different-object MSE. Do this separately for native and padded means.

Show rotation discrepancy alongside this distribution as a magnitude reference. This costs no further encoder inference. It does not turn latent distance into a semantic shape-quality metric or prove that two kinds of mistakes are perceptually equivalent.

## Actual prediction evidence: reuse checkpoint assessment

The encoder experiment should not wait for training or generate another large rollout bank. It will accept selected results from the already planned common-bank checkpoint evaluation when those results exist.

### Primary prediction comparison in latent space (2026-09-15 clarification)

The historical raw/registered F-scores were calculated after decoding. They do not measure latent-space loss and must not be presented as such. For the requested score-difference demonstration, keep a generated Stage-1 endpoint latent `z_pred` fixed and compare it directly with:

- The original target mean: `MSE(z_pred, z0)`.
- Each separately encoded native-scale rotated target mean already constructed in Part A: `MSE(z_pred, z_theta)`.

No decoding or transformation of `z_pred` is involved in these scores. Print original-target MSE, best tested rotated-target MSE, and that candidate's rotation. Also report the full candidate scores. If the same generated latent is substantially closer to a rotated encoding of the same source shape, this provides latent-space evidence of an orientation component in that prediction's mismatch. Remaining MSE still includes shape and latent-representation error. A minimum over seven orientations is exploratory and does not certify a continuous pose optimum; no reduction does not exclude untested rotations. Report train/validation identities separately and aggregate over objects.

Use native-scale candidates for these actual predictions. Do not compare a native-scale generated latent with the smaller padded target family and attribute changes to rotation. The angular curve remains a controlled target-to-target comparison.

Endpoint latent MSE is distinct from the native teacher-forced velocity MSE used during training. Keep the latter in the matched camera/oracle checkpoint assessment, with paired targets, noise and timesteps. Do not rename endpoint MSE as observed training loss.

Decoded figures explain what the original/rotated targets and prediction look like. Their geometry metrics are supporting diagnostics; the headline scores for this demonstration are the direct latent MSEs above.

- Primary matched orientation comparison: camera full surface/no pointmap versus oracle full surface/no pointmap, both dropout 0 and shape cross-attention. Keep image/no-pointmap as a baseline. This avoids pointmap disagreement and dropout changing alongside oracle.
- Report Stage-1 supports in the target frame and after proper rigid registration, with identical seeds/views/sampler settings across models. Train identities and validation identities remain separate. Stage 2 is optional for this rotation slide and is not necessary to establish target orientation sensitivity.
- Use the saved target-latent decoded support as the target reference; source-voxel reconstruction is a separate reference. For synthetic rotations use the known inverse; for real predictions use the existing finite rigid-registration search, explicitly labeled as a diagnostic witness rather than a certified optimum.
- Show every aggregate raw/aligned result, then select an example by a stated rule: e.g. the largest rigid-registration improvement among the declared bank. Label that selection and show its residual shape error. Also show a median example to avoid presenting a rare pose-only success as typical.
- Prefer examples where registered shape approaches its reference. If none do, show the best available residual honestly and conclude that orientation explains only part of the error. A high registration gain does not mean the shape was correct.
- Do not decode, rotate and re-encode predictions to manufacture a purported training-loss improvement. That would introduce a VAE cycle and voxelization confound. Geometry alignment and actual checkpoint loss remain separate reported quantities.

Existing exploratory natural-camera drill: raw one-voxel F1 2.0%, registered 79.7%, on a tiny fitted-identity bank. This can motivate a pose component, with scope printed. It cannot replace the matched full-run population comparison or be labeled a perfect shape.

Existing frame-factorial result can also be reused as limited learning evidence: with object normalization held fixed, reserved-view native loss at 1000 updates was 0.080737 for camera-oriented surface versus 0.060126 for object-oriented surface. Its scope is four fitted identities, one seed, same-object held views. Source: retained `coordinate_system_results/fitting/frame_factorial/analysis.json`. This is evidence of a finite-budget orientation burden in that experiment, not evidence that oracle is required or that it solves full-dataset training. Do not rerun the old four-object setup merely to reproduce this motivation.

## Figures and slide placement

Keep the existing setup order: pipeline -> raw inputs -> encoder normalization -> target encoding. Place this experiment after the target-encoding slide, before oracle intervention results.

**Main rotation slide:**

- Three aligned panels under one fixed display camera, identical bounds/grid/axis convention: reference target support; the same support rotated 90°; the exact inverse-restored support. Include a compact colored overlay in the rotated panel if it improves readability.
- Show the native-scale latent MSE and raw/restored geometry metrics below the panels. A label such as "same voxelized shape; exact grid rotation" makes the construction explicit.
- Beside it, show the continuous angle-versus-latent-MSE plot, with an object-level interval and a zero-repeat floor. Distinguish the padded curve from native controls in the caption. If space is tight, move the curve to the next slide/appendix.
- One sentence: "The target loss distinguishes orientations; oracle supplies the conversion from camera-frame surface to object-frame target." Do not replace this with "oracle is required".

**Following results slide:** direct latent-score table: original-reference endpoint MSE versus tested rotated-reference endpoint MSE, plus the matched camera/oracle native velocity-loss table. Physical target/prediction figures may illustrate the tables, but decoded F-scores do not motivate oracle. Registered geometry belongs only in a separate generation-quality discussion if requested.

**Appendix:** all three axis curves; raw versus inverse-restored geometry curves; VAE reconstruction-versus-angle curves; native versus padded definitions; different-object MSE distribution; example-selection rule.

Reuse the current octopus plot style, scale ticks, grid and separate square triads. Scientific axes must denote actual coordinate axes; any display-only negative-Y convention must be explicitly distinguished from a physical coordinate transform. Use identical camera/bounds within each comparison. Keep native and padded examples separate rather than silently changing limits between them. No fabricated or AI-generated geometry illustrations are needed.

## Minimal implementation

Implemented code folder: `experiments/coordinate_system/scripts/rotation_loss/`.

1. `run_rotation_loss_gpu.py`: one CLI; choose IDs, load meshes and frozen encoder, run Parts A/B, calculate target/flow/input-geometry numbers, append per-object rows. No decoder. Use direct helper imports; no new framework, training class, source modification, W&B run or submission script.
2. `evaluate_stage1_latents_gpu.py`: accept arbitrary completed run directories and best checkpoints. Reuse existing restoration, conditioning and paired-loss/sampling helpers; use the rotation experiment's selected IDs. Collect native velocity loss and sampled endpoint latent MSE, including the native-scale rotated-reference comparison. Do not reuse the old launchers' hardcoded oracle/constant assertions or Stage-2 pipeline. The rollout targets remain original labels; rotated candidates are used only to score a fixed sampled endpoint afterward.
3. `plot_rotation_loss.py`: CPU-only aggregation, object bootstrap, tables, curves and selected scientific 3D examples. Consume compact reports from both runners. Reuse the current figure styling where practical rather than altering previously accepted input figures.
4. `rotation_utils.py`: shared selection and encoding functions. Small copies of the ignored target builder's mesh/voxelizer/encoder functions make this runnable after pulling experiment code alone. Reuse existing exact-grid and continuous-rotation helpers.
5. `README.md`: exact definitions, direct interactive commands, output descriptions and limits. `test_rotation_loss.py` covers physical rotation, fixed endpoint scoring, object averaging, encoder-runner plumbing and figure generation using explicitly fake features. No separate test/tool hierarchy is needed.

Reuse:

- Small copies of `generate_target_latents.py::load_normalized_mesh/load_encoder/voxelize_mesh` for stock zero-angle preparation. The ignored data-generation module is not required at runtime.
- `coordinate_audit/frame_contract.py::rotate_grid` for exact rotations; do not invoke its audit entry point.
- Existing active rotation-matrix helper for continuous mesh rotations.
- `shared/pose_shape_geometry.py::points/metrics/register` for geometry. Call registration only for supplied real checkpoint examples.
- Existing `make_bank`, `paired_loss`, `sample_from_noise`, `restore_run` and production `prepare_batch` are called directly for the Stage-1 checkpoint assessment. Do not invoke the old launchers' main functions.

The experiment-local unclipped voxelizer should be a short adaptation of the production Open3D routine, with fixed bounds and unchanged occupancy indexing. That is the only required change to geometry preparation.

CLI specification:

```text
--data-config configs/data_full_surface.yaml
--pipeline-config checkpoints/hf/pipeline.yaml
--encoder-checkpoint checkpoints/hf/ss_encoder.ckpt
--train-objects 16
--val-objects 16
--angles 0 5 15 30 60 90 180
--seed 29
--example-object 003d24749f2e4047b95f9d04a7c7957b
--output-dir experiments/coordinate_system/outputs/rotation_loss/<run>
```

The source root, manifest and splits come from the data config. Encoder checkpoint is explicit because inference pipeline configuration need not specify an encoder. `sigma_min` is read from the same referenced generator YAML node the pipeline instantiates; omission uses the existing FlowMatching default 0. Reading this setting does not require loading generator weights. The runners resolve their own repository import paths so direct execution works. Output directory is required: no implicit `preparation.json` or separate preparation command. Each command performs its own preparation and measurements; see README for runnable commands. Native shape-loss weighting is 1 in the reused Stage-1 training pipeline.

Required cluster files:

- `objects/<id>/model.obj` for selected objects; textures only for textured example figures.
- `generated_data/<id>/object_transform.npz` and `target_latent.npz`.
- Existing full-surface manifest and object split file.
- Pipeline YAML plus referenced SS generator YAML (to read sigma_min), and SS encoder checkpoint. No decoder checkpoint is required.
- Existing Python environment: torch/CUDA, Open3D, trimesh, NumPy, SciPy and plotting dependencies.

No VecSetX, DINO, full generator weights, image/pointmap batches or optimizer are needed for Parts A/B. The target geometry alone answers their question. The Stage-1 checkpoint runner additionally needs each run's `best.pt` and `config.yaml`, frozen SS generator/conditioning assets, and the image/pointmap/surface/camera files required by its saved conditioning mode.

## Outputs, cost and storage

Outputs belong under the experiment's ignored `outputs/rotation_loss/<run>/`, not root full-training `outputs/`, not tracked scripts or roadmap.

- `measurements.csv`: object, split, native/padded part, axis, angle, scale, latent MSE, analytical flow values, reconstruction and raw/inverse geometry metrics.
- `summary.json`: selected IDs, parameters, numerical controls, per-object summaries and intervals, metric definitions and runtime.
- `figures/`: PNGs plus vector curves when useful.
- `examples/`: compressed uint8/bool occupancies and a few surface samples for two selected objects at 0°, 30°, 90°. No full latent bank, model copies, meshes for every object or automatic ZIP.

Process an object at a time and discard its other arrays after writing metrics. For the proposed 32-object bank: 7 native settings + 19 padded settings = 832 encoder settings, plus identity repeats. An extra example outside the bank adds 26 settings. Decoder inference is optional and restricted to examples. The number of training updates is zero. One GPU suffices; no DDP. Batch encoder inference modestly and print elapsed time after each object. Estimate completion from the first object's measured runtime rather than promising an unmeasured H100 duration.

The runner can save its figures' selected arrays during measurement; plotting/selection can then happen locally from the small report. Retained laptop result copies continue to belong outside the repo in `coordinate_system_results/rotation_loss/<run>/` when appropriate.

## Decision branches

| Result | Conclusion / next action |
| --- | --- |
| Native rotated geometry restores exactly, latent discrepancy clearly exceeds zero-repeat error | Target is orientation sensitive under native preparation. Rotation can produce supervision error with unchanged shape. This is the rotation motivation, not proof of a dataset bug. |
| Curve has dips or axes differ | Check geometry symmetry and native controls. Do not insist on monotonicity or seek a hidden universal yaw correction. |
| Arbitrary-angle inverse-restored support/reconstruction changes substantially | Report discretization/VAE contribution. Use native exact controls for the clean rotation claim; do not use the padded curve as pure pose evidence. |
| Real predictions improve under registration but retain large residual error | Output pose explains a component; shape/conditioning remains deficient. Oracle cannot be claimed to solve the task. |
| Matched oracle no-PM run improves raw pose agreement but registered shape remains comparable | Oracle helps orientation conversion; it does not fix shape learning. Shared-normalization results answer their separate branch. |
| Matched oracle run improves registered shape too | Supplying the target-frame surface aids this learner's shape prediction as well as pose; state budget/split and distance to the reference. |
| Native discrepancy is near zero even for visually asymmetric, exact rotated inputs | Recheck the experiment's mean extraction and actual target correspondence before drawing a rotation-invariance conclusion. Do not infer it from symmetric objects alone. |

The finish line for this extra experiment is a valid supervision-sensitivity demonstration with quantified numerical/geometric controls and honest prediction examples. It is not an unlimited search for a coordinate convention that guarantees successful conditioning.

## Execution order and evaluation readiness (2026-09-15)

1. Run the implemented encoder rotation diagnostic. Main outputs are latent MSE, numerical floor, and physical-input round-trip checks; no decoder is required. This can run independently of the pending full training.
2. Run the implemented Stage-1 checkpoint launcher using existing helpers. Start with the same 16 train + 16 validation IDs, two views per identity and two paired sampled endpoints per view. Sampling uses the earlier conditional-only Stage-1 setting by default: 25 flow steps, CFG 0, pipeline time rescaling. Compare native flow losses using the same addressed noise/times across models: four native time/noise draws and two draws at each of t=0.05/0.5/0.95, following the existing bank helper. Report fixed-time results separately from native-time results. Dropout is disabled during assessment; models retain the effect of whatever dropout they had during training.
3. Orientation comparison after completion: stock camera surface/no-PM (`outputs/stage1_full_surface_no_pointmap_full_cross_attention`) versus oracle surface/no-PM (`outputs/stage1_full_surface_oracle_no_pointmap`), both shape cross-attention and dropout 0. Include stock image/no-PM (`outputs/stage1_image_no_pointmap_full_cross_attention`) as its baseline. Validate saved configuration rather than assuming names specify all settings. Score generated endpoints against original/native-rotated target means without transforming the predictions.
4. Keep the normalization comparison in its own table: stock image+PM and stock surface+PM versus `outputs/stage1_image_shared_normalization` and `outputs/stage1_full_surface_shared_normalization`. This separates the normalization-only treatment (which uses full-surface center/radius) from added surface tokens. Use the same original labels, views, times and seeds. A small correct/wrong-token check must hold the normalized pointmap and its correct-surface statistics fixed.
5. Afterward, use ordinary reconstruction evaluation if generation quality is requested. It is not the latent rotation proof.

Readiness checked against current local source:

- **Existing Stage-2/mesh evaluator:** `evaluate.py` already restores oracle, no-PM, no-visual, shape-full and shared-normalization checkpoint metadata. Shared image controls load surface points for normalization even without a touch encoder. It evaluates `best.pt` and includes the GT-Stage-1-latent-through-Stage-2 reference. Current CD protocol independently centers/scales meshes and applies similarity/PCA initialization followed by ICP. It does not yet export separate raw/rigid-only/similarity CD tables or native velocity loss/rotated-reference latent MSE.
- **Existing submission script:** `jobs/evaluate.sh` still contains historical run paths and H200 partition; it is not prepared for the new run list/H100 request. The evaluator can be called directly with new run directories; only the job arguments/partition need changing for ordinary mesh evaluation.
- **Stage-1 latent-only launcher:** implemented as `rotation_loss/evaluate_stage1_latents_gpu.py`. Reuses previous helpers without the old launchers' hardcoded run assertions. Old launchers should not be run unchanged for the new variants.
- **Rotation diagnostic and plots:** implemented. The plots contain encoded-input examples, angle curves, native velocity-loss bars and fixed-endpoint latent scores, with object-bootstrap summaries.
- **Validation limit:** five new CPU checks plus three existing pairing/sampling/restoration checks passed. The new end-to-end geometry/encoding/plot test uses real Open3D voxelization and explicitly fake features; it is not evidence about pretrained behavior. CLI help was checked directly for both GPU runners. OpenMP shared-memory restrictions required running CPU geometry tests outside the laptop sandbox. No GPU smoke test of the new full-run checkpoints is possible locally. Once a checkpoint is available, use one run and `--train-objects 4 --val-objects 0 --views 1 --draws 1` for the initial smoke run; the reused selector groups identities in fours.

All new diagnostic code stays in the experiment folder. No new production-source edits, automatic commit/push, W&B run or full training job is part of this plan.
