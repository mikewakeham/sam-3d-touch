# Key findings

The [roadmap](roadmap/COORDINATE_CHECKLIST.md) governs inference and next decisions. The [history](EXPERIMENT_HISTORY.md) records other experiments and limitations. No new scientific result was produced by reorganizing these files.

## Reference-convention clarification, 2026-09-14

The [cross-project convention survey](roadmap/CONVENTION_SURVEY.md) distinguishes asset-local axes, semantic up/front, and camera coordinates. **Our oracle supplies the target orientation through the surface; an unknown camera rotation is not still missing from that input.** It does not align all visual streams or eliminate the separate normalization/learned-interface questions. No new transform bug or successful upper bound was established. Keep the existing numerical audit and E15 checkpoint assessment; do not restart axis probes based on the word “canonical.”

## 1. The explicit oracle transform is consistent on the audited bank

Actual oracle points agree with independently reconstructed source points within about `1.6e-7` object units. All 32 exact-bank target latents regenerate exactly. Actual VecSetX normalization on 64 observations reproduces within `1.42e-7`. This supports the existing camera inverse, source-axis convention and normalization arithmetic on those observations. It does not certify every dataset record or pretrained semantic orientation preference.

Evidence: [frame report](../../../coordinate_system_results/full_training/frame_probe/camera/results.json), [source reference](../../../coordinate_system_results/full_training/frame_reference/local_reference_audit.json), [stock-axis check](../../../coordinate_system_results/coordinate_contract/stock_axes.json).

## 2. Normalization arithmetic and normalization learnability are separate

For a complete centered target surface with largest extent one, the radius-normalized cloud Q recovers the canonical geometry by `Q / max_extent(Q)`. For finite sampled points, extrema may be missing; the observed sample-based recovery displacement reaches 0.6101 target voxel. That is not a bound on neural prediction error or missing surface coverage.

For the no-position camera path, the preliminary positive isotropic SSI scale/translation cancels under VecSetX normalization: `N(aP+b)=N(P)`. Removing only that preliminary operation is not a substantive fix. Pointmap tokens, VecSetX normalization itself and sparse-patch placement are different questions.

Evidence: [normalization report](../../../coordinate_system_results/coordinate_contract/normalization_audit.json), [actual SSI cancellation](../../../coordinate_system_results/full_training/frame_probe/local_analysis/ssi_cancellation.json), [deduction and limits](roadmap/COORDINATE_CHECKLIST.md).

## 3. The existing surface pathway can fit a tiny task

Four-object, single-view fitting reached mean decoded-target occupancy IoU of 99.197% with camera surfaces and 99.188% with oracle surfaces, versus 90.070% for image+pointmap. Frozen VecSetX and projector/full shape-cross-attention training sufficed. This refutes a universally dead pathway. It does not show reusable geometric interpretation: a surface may identify a memorized target.

Evidence: [summary](../../../coordinate_system_results/fitting/single_view_summary.json), [raw oracle fit](../../../coordinate_system_results/fitting/single_view/oracle/results.json).

## 4. Rotation added a measured learning burden; dropout helped one aligned tiny task

In the four-object factorial, reserved-view native loss was .06013 with object orientation/object normalization, .08074 with camera orientation/object normalization, .06427 with object orientation/camera normalization and .08071 with both camera factors. This is a finite-budget frame effect, not a universal decomposition of error.

Oracle visual dropout subsequently reached 97.17% mean occupancy IoU on reserved views of those fitted identities, versus 66.95% for the original oracle policy. Camera+dropout remained at 56.17%. Crucially, the oracle surface sample is reused across views: this establishes fitted-identity visual robustness, not new-geometry generalization. Small-object transfer and constant controls prevented promoting it into a general solution.

Evidence: [factorial](../../../coordinate_system_results/fitting/frame_factorial/analysis.json), [dropout](../../../coordinate_system_results/fitting/visual_dropout/analysis.json), [camera geometry](../../../coordinate_system_results/geometry/camera_dropout/summary.json), [constant control](../../../coordinate_system_results/fitting/constant_control/analysis.json).

## 5. Full training shows real surface utility and an unmet upper bound

Same-budget full-data camera, oracle and constant-surface runs have similar pooled losses. Paired generation exposes differences:

| Split, visuals present | Constant F2v raw / rigid witness | Camera raw / rigid witness | Oracle raw / rigid witness |
|---|---:|---:|---:|
| 16 training objects | 63.72 / 79.14% | 64.14 / 81.17% | 75.87 / 84.52% |
| 16 validation objects | 49.69 / 68.10% | 51.02 / 67.83% | 64.51 / 77.23% |

Oracle wrong-surface validation witness F2v is 59.15%. Thus the real geometry contributes at this scope; the tiny constant-control result does not imply all gains are extra adaptation. Reconstruction remains inaccurate on training objects too.

F2v is occupied-voxel proximity F-score at two voxels, not semantic shape accuracy. These results use one training seed, a repeatedly consulted development bank, two views/two draws and 25 steps at CFG0. A matched image-only sampled baseline is absent. Rigid search is approximate and excludes scale; one all-rigid failure certificate does not exclude size correction.

Evidence: [paired losses](../../../coordinate_system_results/full_training/fixed_state_probe/summary.json), [combined geometry](../../../coordinate_system_results/full_training/frame_probe/local_analysis/summary.json), [oracle raw samples](../../../coordinate_system_results/full_training/oracle_constant_rollouts/oracle/results.json).

## 6. Other useful boundaries

- Native VecSetX decoding retained substantial but imperfect geometry in the reported 32-object probe. It does not guarantee access through the current projector. [Report](../../../coordinate_system_results/coordinate_contract/native_vecsetx/analysis.json).
- Shared-camera targets and broader shape adaptation could fit four identities, but did not establish transferable conditioning. The severely underexposed rotation augmentation was inconclusive; its outcome remains in history, not as an impossibility claim. [Scope](../../../coordinate_system_results/shared_frame/scope/summary.json), [coverage exposure](../../../coordinate_system_results/shared_frame/coverage/learning_and_exposure.json).
- No-PM training must be distinguished from the old camera/no-PM baseline, and inference-only visual removal from no-visual training. Updated W&B training results are recorded below; new checkpoint reconstruction assessments are still missing.
- Sparse touch needs retained location/scale information or an explicit transform. Full-surface normalization arguments do not apply to an isolated touch patch. [Future constraints](roadmap/FUTURE_TOUCH_REQUIREMENTS.md).

**Stopping boundary:** do not repeat passed transform checks without a changed code/data path or a concrete counterexample. The unresolved normalization, visual-interaction and pretrained-convention questions remain explicitly named in the roadmap; they are not all ruled out by the oracle audit.

## 7. W&B refresh on 2026-09-14: the modality-removal runs finished

All three new runs finished 20 epochs / 14,660 updates with frozen VecSetX and full shape cross-attention adaptation (not full-backbone finetuning).

| Run | W&B ID | Final validation loss |
|---|---|---:|
| Oracle, no pointmap, 50% visual dropout | `ssmsddtg` | 0.0894794014 |
| Constant oracle surface, no pointmap, 50% visual dropout | `pg4413ls` | 0.0890264520 |
| Oracle, permanently no visual conditioning | `8zpws0ez` | 0.1046997821 |

Original oracle+pointmap/dropout was 0.0893984329. Thus no-pointmap did not produce the hoped-for improvement in the logged pooled objective. The no-visual objective is worse, but its validation conditioning differs; this is not evidence that visual conflict is required or that geometry is unused. All three logged nonzero surface-projector and shape-cross-attention gradients. These establish gradient activity, not useful reconstruction.

The no-visual export records `no_visual=true`, `no_pointmap=false`, and `visual_dropout=0`. These are consistent: permanent disabling suppresses all visual modalities, including pointmap, independently of the stochastic dropout setting. Metadata records source commit `1c5208b`; the inspected implementation and existing CPU fixtures support that intended policy. Actual new checkpoint boundary tensors have not been inspected on GPU.

No paired native-time/noise probe or noise-only Stage-1 reconstruction assessment for these three checkpoints was imported. Do not assign the earlier oracle checkpoint's generation results to these runs. The next checkpoint-assessment round should reuse the audited bank and retain these distinctions; no retraining or repeated target encoding is needed simply to obtain the missing outcome.

Evidence: [W&B refresh analysis](../../../coordinate_system_results/full_training/wandb_refresh_20260914.json), with input hashes, recorded commits, configs, validation histories, and gradient summaries.

## 8. Shape-full W&B refresh: logged deterioration, separate from coordinate closure

All three newly exported shape-full runs finished 20 epochs / 14,660 updates. They are not the cross-attention checkpoints awaiting the E15 coordinate assessment.

| Run | ID | Best validation loss (epoch) | Final validation loss |
|---|---|---:|---:|
| Image, no PM, dropout 0 | `aikgqmzc` | 0.0889387 (3) | 0.1074300 |
| Oracle full surface, no PM, dropout 0.5 | `7zt1syig` | 0.0888329 (3) | 0.1026889 |
| Constant surface, no PM, dropout 0.5 | `xqoi250e` | 0.0886795 (2) | 0.1024156 |

The means of the last 50 training-log windows are 0.05382, 0.06274 and 0.07272 respectively; these training objectives differ in conditioning policy. Falling training loss with rising validation loss is consistent with overfitting, not a demonstrated causal mechanism or a generated-shape result.

**Dropout is not a sufficient explanation for all three deteriorating curves:** the image-only run used none and also deteriorated. This does not establish that dropout is harmless to the oracle run. A matched oracle/no-PM shape-full dropout-0 run is absent, so “would it have worked without dropout?” remains unanswered. The earlier tiny-task dropout benefit was fitted-identity view robustness, not evidence for an optimal full-data dropout rate.

At the user's direction, broader-scope/checkpoint-trajectory/dropout analysis is deferred. The active task remains the roadmap's existing no-PM/no-visual checkpoint assessment. No coordinate hypothesis was newly closed by this W&B refresh.

Evidence: [curve summary and input hashes](../../../coordinate_system_results/full_training/wandb_shape_full_refresh_20260914.json).

## 9. Camera-target rerun: shared pointmap normalization did not improve fixed-bank loss

The 16-object, 8-training/4-held-view, 1,000-update comparison completed. Final training-reference / held-view losses: object target + stock PM **0.06707 / 0.06273**; camera target + stock PM **0.12502 / 0.11681**; camera target + shared PM **0.12439 / 0.11809**. Recorded data, initialization, training order, code and settings are matched.

The camera/stock and camera/shared curves nearly overlap: shared normalization has no clear benefit under shape cross-attention training with 50% visual dropout. Camera-target loss is higher even on fitted views, but target distributions differ and this is not a reconstruction-quality comparison. Curves still improve over the second half of training, so no architectural limit or complete coordinate exclusion is established. Held-view loss below train-reference loss is not a controlled view-dependence finding because samples/noise banks differ.

Keep stock PM normalization as default pending final Stage-1 reconstruction assessment. No further training is justified by a shared-normalization advantage here. [Analysis and evidence locations](scripts/camera_frame_target_latent/RESULTS_20260914.md).

## 10. The full oracle run without visual dropout already exists

W&B run `fd2yiktq` (`stage1_full_surface_oracle`) completed 20 epochs / 14,660 updates with oracle full surfaces, pointmap present, visual dropout 0 and `shape_cross_attention` scope. Its best/final validation losses are **0.0888749 / 0.0892673**. The matched camera-frame, no-dropout run `act988rs` reports **0.0888587 / 0.0894101**; the oracle, 50%-dropout run `cps50h2l` reports **0.0889772 / 0.0893984**. These differences are very small and do not show a pooled-loss rescue from either oracle alignment or removing dropout.

This fills the oracle + pointmap + dropout-0 cell. At that export, oracle + no pointmap + dropout 0 was missing; it completed on September 15 (see finding 12). Checkpoint-level paired denoising and Stage-1 reconstruction remain required before interpreting geometric success or failure.

## 11. Target latent rotation sensitivity is now directly measured, 2026-09-15

The frozen SS target encoder probe completed on 16 training and 16 validation identities, plus one separate octopus example. Original-scale zero encodings exactly reproduce all stored means; repeated encodings have MSE 0. Exact 90/180-degree occupancy rotations retain occupied counts and inverse-grid IoU 1, eliminating clipping and mesh revoxelization as explanations for this control.

Averaged over objects and XYZ axes, original-scale target mean MSE is **0.19631 / 0.32430** (train/validation) at 90 degrees and **0.15714 / 0.26786** at 180 degrees. At a separate fixed padded scale, 5-degree rotations give **0.07731 / 0.12949**, with larger penalties through 60 degrees and lower means again at 90/180 degrees. Do not claim monotonic growth with rotation angle. The padded curve includes mesh-to-voxel discretization; the exact original-scale control independently establishes rotation sensitivity.

The presentation chart now shows validation only, with its legend below. The 16-object intervals reflect substantial object variation: padded X60 MSE ranges from 0.0701 to 0.6634, median 0.2867 and mean 0.2947. A fixed 64-validation-object rerun is recommended for more precise population means; it is not needed to establish sensitivity, and smaller intervals are not guaranteed.

This proves that the encoded target is orientation sensitive before decoding. It does **not** prove camera-frame conditioning is invalid, that oracle is required, or that orientation explains the actual generator residual. Actual checkpoint native velocity loss and fixed-prediction latent comparisons remain a separate experiment. Analytical velocity columns in the CSV are derived endpoint conversions, not observed checkpoint losses.

Evidence: local ignored [encoder report](outputs/rotation_loss/encoder/results.json), [measurements](outputs/rotation_loss/encoder/measurements.csv), and [plot summary](outputs/rotation_loss/encoder_figures/summary.json). These output files are not tracked in Git.

## 12. Oracle/no-PM/dropout-0 and shared-normalization full runs completed

All three September 15 runs finished 20 epochs / 14,660 updates with shape cross-attention training and visual dropout 0.

| Matched condition | W&B ID | Best validation loss | Final validation loss |
|---|---|---:|---:|
| Image + stock PM | xun3al7m | 0.0885780 | 0.0890978 |
| Image + shared-normalized PM | 460qbbvq | 0.0884645 | 0.0890844 |
| Image + stock PM + camera surface | act988rs | 0.0888587 | 0.0894101 |
| Image + shared-normalized PM + camera surface | p8i307r8 | 0.0889002 | 0.0895893 |
| Image, no PM + camera surface | nouqb3mh | 0.0890146 | 0.0896528 |
| Image, no PM + oracle surface | uvzaqbuf | 0.0893191 | 0.0902257 |
| Image, no PM baseline | fl7b2znc | 0.0892078 | 0.0898082 |
| Image + stock PM + oracle surface | fd2yiktq | 0.0888749 | 0.0892673 |

No clear pooled-validation-loss rescue is observed. Shared normalization with real surfaces is slightly worse by this metric, and the oracle/no-PM condition also does not improve it. These single-run, best-epoch differences do not establish statistical equivalence or generated shape quality. The oracle/no-PM run is nevertheless the needed controlled test without PM/surface conflict and without dropout.

Next: evaluate these eight best checkpoints on the same validation identities/views/seeds with existing evaluate.py (Stage-2 aligned mesh CD and decoded-GT reference). This evaluates generated shape quality; it does not itself establish Stage-1 latent orientation error. The already implemented Stage-1 latent checkpoint probe addresses that separately. Do not rerun training from these pooled losses alone. Broader shape-full/dropout studies remain deferred.

Evidence: exported run.json/history.csv in ../wandb-results and local ignored [refresh summary](outputs/rotation_loss/wandb_refresh_20260915.json). Evaluation commands are provided in chat; no new job script or production change is needed.

## 13. Generated-view diagnostic replaced: test the known camera inverse

The rotated-target, seven-orientation shortlist was discarded at the user's direction. Its selected examples do not establish that predictions retain their input camera orientation and are not the intended presentation evidence. That generated-example selection code has been removed.

The replacement in `scripts/generated_rotation_examples/` retains fixed metadata-selected validation views and all their actual generated predictions, copies each input image and raw surface/pointmap display clouds, and compares decoded prediction with the stored target before/after applying that view's known inverse camera rotation about the origin. No camera translation, rotation search, target re-encoding or score-based selection. Latent MSE is measured only against the stored training target. Defaults are eight validation objects, three views, one draw (24 predictions); no replacement GPU result is available yet. All three output plots share native XYZ axes. Raw input plots retain the documented display-only camera ZXY convention.
