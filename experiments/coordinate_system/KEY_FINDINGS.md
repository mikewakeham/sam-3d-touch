# Key findings

The [roadmap](roadmap/COORDINATE_CHECKLIST.md) governs inference and next decisions. The [history](EXPERIMENT_HISTORY.md) records other experiments and limitations. No new scientific result was produced by reorganizing these files.

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
