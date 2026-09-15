# What would justify stopping the coordinate investigation?

**Current status, 2026-09-14 (latest user authorization):** implement the [camera-frame target latent experiment](../scripts/camera_frame_target_latent/README.md). Three arms: object targets/stock PM, camera targets/stock PM, camera targets/shared surface-derived PM normalization. 16 objects with eight training and four held-out views each; no unseen objects; 1,000 optimizer updates, global batch 4, loss validation every 100 plus step 0. Matched shape cross-attention scope and 50% per-example training visual dropout. Reconstruction is deferred until after training. Standalone interactive experiment, one GPU per arm; experiment code only, local JSON logging, optional experiment-local three-job submission script; no W&B integration. The three runs completed; [returned results](../scripts/camera_frame_target_latent/RESULTS_20260914.md) show nearly identical camera/stock versus camera/shared losses, both higher than object/stock. Keep stock PM normalization; the previously planned final Stage-1 reconstruction comparison remains outstanding. No additional training experiment is queued. This supersedes the earlier pause ONLY for this authorized rerun; the removed surface-only assessment and other historical proposals remain inactive.

This directly tests camera-target convention and the remaining pointmap/surface scale-translation treatment separately. Target units stay in SAM3D's standard cube; no claim of exact shared target/VecSetX units or solved conditioning. Different target distributions require later reconstruction assessment before a performance conclusion.

## Reference-convention survey, 2026-09-14

The user requested a code/supplementary comparison before further experiments. [CONVENTION_SURVEY.md](CONVENTION_SURVEY.md) records that work, its precise source coverage, and unreleased-code limitations. It clarifies C2–C8 without changing the numerical audit status: oracle orientation is supplied, semantic canonicalization is a separate data convention, and the remaining mixed-stream/shared-normalization questions are not equivalent to an unknown inverse rotation. E15 below remains the next checkpoint assessment; no new full training or repeated transform bank is justified by this survey alone.

## Current priority after the refreshed overnight results

**User direction, 2026-09-14: resume the existing coordinate roadmap; defer further shape-full/dropout diagnosis.** The three overnight runs finished. Their W&B curves were inspected, but no new checkpoint reconstruction was supplied. This changes scheduling, not the earlier coordinate findings.

Historical proposed next experiment (subsequently paused, not currently queued) was **E15 / Gates A and C / C0, C5, C6, C10**: assess the completed cross-attention `ssmsddtg` oracle/no-PM, `pg4413ls` constant/no-PM, and `8zpws0ez` oracle/no-visual checkpoints on the same audited 16 training + 16 validation objects. Inspect the new restored modality boundaries once, measure paired correct/wrong-surface denoising and noise-only Stage-1 reconstruction. Reuse source/target evidence and sampler settings. Do not repeat inverse/axis checks or retrain these variants. Training reconstruction is the fitting question; validation is a separate transfer question.

If poor reconstruction persists without PM, PM conflict is not necessary for that residual. If it persists without all visual information, visual/surface disagreement is not necessary either. These are scoped exclusions, not proof that normalization units or pretrained coordinate preferences cannot affect learning. Accurate reconstruction would instead preserve a working recipe and redirect the corresponding branch. C4b and C7/C8 remain explicitly open; see the existing checklist for the optional, controlled normalization/convention treatment. Do not silently substitute more training-scope runs for that treatment.

Execution note: retained `full_checkpoints/checkpoint_probe_gpu.py` and `checkpoint_rollout_gpu.py` currently validate the original oracle/constant **with-PM** modes. They need a bounded extension for E15 and compact output handling before handing over commands; do not run them unchanged on the new checkpoints or remove their checks indiscriminately. Cluster reference locations after cleanup remain unconfirmed in STORAGE.md. No GPU assessment was performed in this refresh.

Deferred training TODOs (resume after the coordinate assessment):

- [ ] Assess shape-full best versus last checkpoints using paired Stage-1 denoising and generated reconstructions; training and validation separately.
- [ ] Compare real versus constant geometry at matched dropout; compare image-only as a separate complete recipe.
- [ ] Determine whether broader adaptation overfits or degrades useful pretrained behavior. Falling training loss/rising validation loss is a symptom, not that diagnosis.
- [ ] If warranted, compare oracle/no-PM shape-full with dropout 0 versus 0.5 at matched conditions. No such paired full-scope comparison exists. No-dropout image-only also deteriorated, so dropout is not necessary for the observed validation deterioration.

Results and exact curve summaries: [W&B shape-full refresh](../../../../coordinate_system_results/full_training/wandb_shape_full_refresh_20260914.json). Keep these raw results outside Git. Best validation checkpoints are at epoch 3 for image/real oracle and epoch 2 for constant; preserve both best and last for the deferred analysis.

## Authorized overnight runs, 2026-09-14

This supersedes the earlier continuation scheduling proposal below. The user authorized minimal source integration and three **fresh-from-pretrained** shape-full runs, with checkpoint assessment deferred until GPU access tomorrow. No cluster jobs have been submitted locally.

| Job in repository `jobs/` | Inputs | Visual dropout |
|---|---|---|
| `stage1_image_no_pointmap_shape_full.sh` | Image, no pointmap/surface | 0 |
| `stage1_full_surface_oracle_no_pointmap_dropout_shape_full.sh` | Image + oracle full surface, no pointmap | 0.5 |
| `stage1_full_surface_oracle_constant_no_pointmap_dropout_shape_full.sh` | Image + constant oracle surface, no pointmap | 0.5 |

All use `kempner_h100`, four GPUs, global batch 16, 20 epochs, generator LR 1e-5 and adapter LR 1e-4 where present. Outputs are `outputs/<job filename without .sh>/`; W&B uses the same run name and records `train_scope`. Only existing best/last checkpoints are saved, including newly trainable weights and optimizer state. VecSetX/DINO remain frozen; oracle preprocessing, features and labels are unchanged.

`--train-scope shape_cross_attention` is the default; `shape_full` additionally trains shape self-attention/MLPs/input-output mapping and shared time/modulation. Dedicated layout weights stay frozen; shared modulation is not exclusive to shape. Legacy `--cross-attention-scope full` and old checkpoints retain cross-attention-only meaning. Ordinary `--resume` requires the same scope; scope-switch warm-start tooling is not part of this minimal fresh-run integration.

Real versus constant is the matched geometry-information comparison. Real versus image compares complete recipes (dropout differs). Old cross-attention runs are scope references at matched budgets/configurations, not a new random-seed replication. Pooled W&B loss alone will not establish success. Tomorrow assess checkpoint reconstructions and paired correct/wrong surface effects on training objects, with validation transfer separately, using the saved audited geometry. Do not repeat the passed coordinate audit.

Local verification: reduced-width **actual SAM3D dense transformer**, all three modes, frozen-layout/encoder checks, finite gradients, parameter updates, checkpoint and evaluation restoration, resumed optimizer equivalence, and two-process CPU DDP. CUDA/pretrained-scale memory and convergence remain untested on this laptop. No new experimental outcome is claimed.

## Execution priority, 2026-09-14

The checklist IDs are a question map, **not an execution order**. This section updates the scheduling recommendation below: checkpoint assessment and a controlled training-scope pilot can overlap. Completing every coordinate hypothesis is not a prerequisite to holding coordinates fixed and testing adaptation.

1. Assess existing oracle/no-PM and constant/no-PM checkpoints on the audited bank; include the no-visual checkpoint for the remaining visual-conflict exclusion. Reuse verified geometry. This establishes reconstruction and surface utility, rather than inferring them from pooled W&B loss.
2. Verify a broader shape-path implementation: parameter selection, finite gradients/updates, restoration, and saving all newly trainable parameters. Production `cross_attention_scope=full` currently means shape cross-attention plus norm2, not the shape path. Retained shared-orientation scope code is a reference, not an already integrated full-data training option.
3. Proposed short comparison: initialize both arms from the **same completed oracle/no-PM checkpoint**. A continues current cross-attention training; B also trains shape self-attention, MLPs, input/output projections, and relevant time/modulation parameters. Keep oracle preprocessing, frozen VecSetX/DINO, existing adapter, no-PM, visual dropout 0.5, targets and loss unchanged. Use the full training loader, not another four-object fit. Pair batches, noise/times and dropout. Reset optimizer state consistently in both arms for this scope-switch diagnostic; do not silently preserve moments in only one arm. Start with existing generator/adapter learning rates. The proposed 500–1000-update budget is a screen, not a measured runtime or a proof that unsuccessful adaptation is impossible.
4. At step 0 and the pilot endpoints, measure fixed fresh-noise training-bank loss (including high noise), correct/wrong surface effects, and selected Stage-1 generations using the existing protocol. Keep validation results separate. Identical initial predictions verify the common starting function. A useful signal is improved fitted-object generation across objects/draws, supported by improved denoising and correct-surface utility; a lower online training curve alone is insufficient.
5. If promising, continue A and B overnight at matched budgets. A third, broader-scope constant/no-PM continuation from its corresponding completed constant checkpoint provides the capacity control. Its prior training history is deliberately constant-conditioned, so it is not the common-start A/B scope comparison. Do not simultaneously change point normalization, token representation or fusion architecture. A fresh-from-pretrained scope study is a separate option if continuation stalls; a negative continuation pilot does not exclude it.

Interpretation: B success demonstrates a working learner under the fixed oracle convention; it does not show coordinate conventions never affect learning. Success on training objects but failure on validation directs attention to transfer/data. Both arms failing does not reopen passed transform checks or establish architectural impossibility.

Paper review references: [Axolotl3D main paper and appendices A–E](https://arxiv.org/html/2607.20660v1), [author project](https://research.nvidia.com/labs/sil/projects/axolotl3d/). Exact encoder feature tap, encoder normalization boundary and trainable parameter list were not verified from an author implementation. Keep paper motivation separate from proof of our proposed scope treatment. Local code and the recorded no-PM run commits already contain LayerNorm + gated FFN surface projection; do not propose adding that as a missing integration.

Status at proposal time: no new GPU results. The later authorization and implemented jobs are recorded above. Diagnostic outputs follow [STORAGE.md](../STORAGE.md).

**Current answer: stop repeating the verified transform checks. Do not claim all coordinate-related learning effects are excluded.** The audit is complete; the empirical closure evidence is not. No new run is requested here.

**2026-09-14 update:** the new oracle/no-pointmap, constant/no-pointmap and oracle/no-visual runs have completed. Their refreshed W&B validation losses are 0.0894794, 0.0890265 and 0.1046998. The separately synced oracle + pointmap + dropout-0 run `fd2yiktq` also completed, with best/final validation loss 0.0888749 / 0.0892673, essentially matching the camera no-dropout and oracle-dropout references. Configs and histories are available; checkpoint-level reconstruction assessments are not. See [updated findings](../KEY_FINDINGS.md#7-wb-refresh-on-2026-09-14-the-modality-removal-runs-finished) and [the no-dropout oracle entry](../KEY_FINDINGS.md#10-the-full-oracle-run-without-visual-dropout-already-exists).

## The argument, without treating every open question as a prerequisite

```text
On a particular object, is reconstruction actually poor?
  No / only a similar pooled loss -> no demonstrated reconstruction failure to diagnose.
  Yes
   |
   +-- Do its actual points and target fail the independent geometric contract?
   |     Yes -> a concrete coordinate/data defect; repair and reassess that object.
   |     No  -> a wrong explicit transform cannot explain that object's residual.
   |
   +-- Does poor reconstruction persist in training without pointmap information?
   |     Yes -> pointmap disagreement is not necessary for that residual.
   |     No  -> removing pointmap helps; coordinate conflict is only one explanation.
   |
   +-- Does it persist when training has no image, mask or pointmap information?
         Yes -> sample-dependent visual disagreement is not necessary either.
         No  -> visual training policy matters; it is not yet a scale/rotation diagnosis.

What remains after those exclusions?
  A. Information/placement lost by finite-sample normalization.
  B. Difficulty interpreting the aligned features, including conversion between units.
  C. Pretrained preference for another consistent target convention.

These are not three unidentified sign/axis bugs. B and C interact with trainable scope.
They cannot be separated by another unsuccessful camera-inverse check.
```

The branches above compare **observed failures**, not hypothetical ones. Multiple causes can coexist. Failure after removing a stream excludes necessity, not the stream's contribution to the original run. A successful removal also removes useful information and changes learning; it does not isolate coordinates automatically.

## Which evidence is needed for which stopping claim?

| Claim you want to make | Required evidence | Current position |
|---|---|---|
| “The camera inverse/axis conversion is not wrong on these failing examples.” | Independently verified geometry for those exact examples, plus the actual encoder-boundary coordinates. | Available for the existing full-frame bank. Further whole-dataset checking is **not needed for this scoped statement**. |
| “The original full dataset contains no coordinate/data defects.” | Coverage of all relevant records and generation branches, not only the selected bank. | Not established. This stronger claim is **not required** to study an audited failing subset. |
| “Pointmap disagreement cannot be the whole explanation.” | Verified reconstruction failure in the oracle/no-PM training variant. | Completed no-PM training still has pooled validation loss 0.0894794; its checkpoint reconstruction assessment is missing. |
| “No visual-frame disagreement is required for the remaining failure.” | Verified failure when all visual information was absent throughout training. | Completed no-visual training has pooled validation loss 0.1046998 and the expected recorded config. Its checkpoint tensors/reconstruction have not been assessed; inference-only suppression is insufficient. |
| “Normalization/convention cannot affect learning.” | A universal claim over encoders, conventions and optimization—not something finite negative trials establish. | Do not use this as a finish line. Test a specified contribution or demonstrate a working recipe instead. |
| “This oracle convention supports a useful upper bound.” | Accurate target-referenced reconstruction and correct-surface utility at the stated scale; fresh objects if claiming transfer. | Tiny fitted-identity success exists; full-data accurate upper bound does not. |

## The clean decision after the missing exclusions

There are two legitimate routes. They answer different questions:

1. **Coordinate-treatment route:** test one explicitly specified normalization or target-convention change against a matched control. A negative rejects that treatment at its tested budget; it does not eliminate every convention. The [main report](COORDINATE_CHECKLIST.md) states the input-distribution, clipping and VAE controls this would require. It is not yet an executable design.
2. **Fixed-coordinate learning route:** hold the audited oracle inputs and labels identical while comparing current versus broader trainable scope. If broader scope works, that demonstrates a working learner under this convention. It does not prove that coordinates never made adaptation harder. If both fail, it gives no justification to repeat passed transform checks.

**Neither route requires claiming “coordinates are irrelevant.”** Moving to the second route is a controlled study of the remaining mapping, not a conclusion that every coordinate effect has been disproved. If the desired conclusion is which factor caused the original failure, a scope-by-convention comparison may eventually be needed; that is a different, larger claim.

The immediate missing evidence is assessment of the existing no-PM run, not another full training run or another easy four-object fit. No commands are issued until the experiment design is accepted. The supporting [question checklist](COORDINATE_CHECKLIST.md), [historical evidence ledger](../EXPERIMENT_HISTORY.md) and [review record](../../../../coordinate_system_provenance/AUDIT_REVIEW_LOG.md) preserve all remaining questions and the limits of each result.

## Proposed next round after the 2026-09-14 refresh

Assess the **existing** `fd2yiktq` oracle/PM/dropout-0, `ssmsddtg` oracle/no-PM/dropout-0.5, `pg4413ls` constant/no-PM/dropout-0.5, and `8zpws0ez` oracle/no-visual checkpoints on the already audited 16 training + 16 validation objects. This is a proposed GPU checkpoint assessment, not a completed test or a request for new training.

1. Verify checkpoint/config identity and the actual conditioner boundary once for these newly evaluated modes. No-PM must suppress both pointmap streams; no-visual must contain no sample-dependent visual information while retaining active surface tokens. The no-visual W&B dropout fraction is zero because stochastic dropout is separate from permanent disabling.
2. Reuse existing targets, views, noise/times and sampler settings. Do not repeat mesh target encoding or camera-inverse checks on the same unchanged records. Detect changed assets before claiming reuse.
3. Measure paired Stage-1 velocity losses with correct versus wrong surfaces, including the high-noise cases where prior full-run surface utility was clearest. Measure noise-only Stage-1 reconstruction against the saved decoded target supports using the existing 25-step CFG0 setting, two views/two draws. Keep native IoU, precision/recall and F2v separate from any pose/scale-adjusted diagnostics. No Stage 2 or CD is required.
4. Assess training objects first for fitting/generation, validation objects separately for transfer. Export compact per-object reports and small selected failure examples; keep bulk arrays/checkpoints in ignored cluster diagnostic outputs. Retained laptop results belong outside the repository.

Decision branches:

- Accurate fitted-object reconstruction: that recipe provides a fitted upper bound; any remaining unseen-object failure is transfer. Do not call it an oracle transfer solution without evidence.
- Persistent failure without PM: PM conflict is not necessary for those failures. Do not make PM alignment the next required fix for those examples.
- Persistent failure without any visual information, on audited geometry: sample-dependent visual/surface disagreement is not necessary either. Hold the oracle convention fixed and move to a matched training-scope or data/coverage investigation. Do not claim every normalization/prior preference is disproved.
- Expected modes/checkpoint checks fail: repair that concrete implementation/restoration defect before interpreting model quality.
- A materially better modality-removal recipe: preserve it as the working baseline. Further multimodal alignment is a separate effort to recover useful visual information, not a prerequisite to using that baseline.

One assessment round could therefore finish the practical coordinate exclusions if its outcomes are clear. It cannot promise a successful conditioner tonight, universal coordinate irrelevance, or a proof of architectural incompatibility. No GPU runtime estimate has been measured locally.
