# Full-surface conditioning investigation — 11 September 2026

**Current result — 16-object fitting still fails to transfer surface utility:** [OBJECT_TRANSFER_RETURNED_FINDINGS.md](OBJECT_TRANSFER_RETURNED_FINDINGS.md) validates all four completed runs and exact analysis reproduction. Held-object loss is .11930 image, .12640 camera, .13420 oracle and .13377 oracle dropout. Oracle improves all fitted identities but loses to image on all 16 held identities. Surface arms initially beat image at step 256, then deteriorate on held identities while fitting improves. Alignment and dropout are insufficient; no full training or automatic extension is justified.

**Next GPU action — no training:** [TRANSFER_CHECKPOINTS_HANDOFF.md](TRANSFER_CHECKPOINTS_HANDOFF.md) uses saved image/camera/oracle checkpoints at 256 and 1024, with exact historical replay and fresh held-object draws. Correct surfaces, three wrong-object surfaces and surface removal distinguish geometry-specific utility from changes to the shared attention mapping. VecSetX and coordinates remain unchanged. Local checks pass; GPU reports are pending.

**Completed training screen:** [OBJECT_TRANSFER_HANDOFF.md](OBJECT_TRANSFER_HANDOFF.md) records the 1024-update, 16-fitted/16-held-identity comparison. All four results returned; do not rerun it. Earlier plans below are historical unless explicitly referenced by the current handoff.

**Latest result — tiny-fit improvements fail on new identities:** [UNSEEN_OBJECTS_RETURNED_FINDINGS.md](UNSEEN_OBJECTS_RETURNED_FINDINGS.md) validates all five completed probes and exact anchor replay. Mean new-object loss is .10540 image+pointmap, .17043 camera, .17987 camera dropout, .21229 oracle and .27079 oracle dropout. Every point model loses to image+pointmap on all 32 objects. The oracle held-view improvement is specific to the fitted identities; it is not a transferable geometry fix. No full dropout run or pose estimator is justified yet. The next proposed diagnosis expands training-object diversity while retaining coordinate and image-baseline controls.

**Completed GPU check:** [UNSEEN_OBJECTS_HANDOFF.md](UNSEEN_OBJECTS_HANDOFF.md) records the five-checkpoint, 32-object probe. All reports have returned; do not rerun it.

**Earlier result — visual dropout repairs the fitted identities' oracle view gap:** [VISUAL_DROPOUT_RETURNED_FINDINGS.md](VISUAL_DROPOUT_RETURNED_FINDINGS.md) validates both completed runs and exact replay. With full inputs present, reserved-view loss improves camera .07718 → .05381 and oracle .05785 → .02273. Oracle fitted loss is .02205, so its same-identity view gap nearly disappears. The subsequent unseen-object test above shows this improvement does not transfer; do not promote dropout from this small-task result.

**Completed short GPU intervention:** [VISUAL_DROPOUT_HANDOFF.md](VISUAL_DROPOUT_HANDOFF.md) records the two 1000-update runs. Both reports have returned; do not rerun or automatically extend them.

**Latest result — aligned surfaces are used, visual view transfer remains weak:** [ORACLE_VISUAL_STREAMS_RETURNED_FINDINGS.md](ORACLE_VISUAL_STREAMS_RETURNED_FINDINGS.md) validates the complete factorial and exact endpoint replay. Reserved-view pointmap-only changes give .01917 loss versus anchor .01795, while coherent other visuals give .05591. Correct surfaces beat wrong-object surfaces in every recorded batch/noise comparison. RGB/silhouette dependence, rather than pointmap changes, dominates this oracle residual. The matched camera/oracle visual-dropout comparison above implements the resulting intervention; no full run is justified yet.

**Completed GPU check:** [ORACLE_VISUAL_STREAMS_HANDOFF.md](ORACLE_VISUAL_STREAMS_HANDOFF.md) records the oracle visual-stream factorial. Both reports have returned; do not rerun it.

**Latest result — rotation dominates the tested frame penalty:** [FRAME_FACTORS_RETURNED_FINDINGS.md](FRAME_FACTORS_RETURNED_FINDINGS.md) validates both mixed runs and exact baseline preflight. Reserved-view loss is .08074 for rotation-only, essentially the original camera .08071; normalization-only is .06427, closer to oracle .06013. This supports the orientation-handling branch while retaining VecSetX. It does not establish a global conversion bug or explain all oracle residuals. No full-dataset overnight run or automatic continuation of these fits is justified yet.

**Completed experiment — coordinate factors:** [FRAME_FACTORS_HANDOFF.md](FRAME_FACTORS_HANDOFF.md) records the two mixed-treatment runs and their historical baseline replay checks. Both reports have returned; do not rerun them.

**Current plan — coordinate diagnosis resynchronized (12 September):** [COORDINATE_DIAGNOSIS_RESYNC.md](COORDINATE_DIAGNOSIS_RESYNC.md) reconciles the chat, completed probes, and source. Retain VecSetX and focus on full surfaces. The completed contrast isolates orientation as the dominant measured factor. [POSE_INPUT_CONTRACT.md](POSE_INPUT_CONTRACT.md) traces the next branch: ordinary inputs omit camera-to-target pose, and shape cannot read the layout latent. Its accompanying 64-sample metadata audit rules out substituting camera translation for rotation. The point-encoder replacement comparison is withdrawn and its unexecuted prototypes removed. Older plans below are historical and are not queued jobs.

**Governing scope correction — bridge integration withdrawn:** [TOUCH_CONDITIONING_REQUIREMENTS.md](TOUCH_CONDITIONING_REQUIREMENTS.md) records the required learned representation of observed point/patch structure, including future normals and per-point physical measurements. Full surfaces must test that same path. The completed mesh/voxel bridge is diagnostic evidence only; its proposed Stage-1 integration is withdrawn and was not implemented. This supersedes older recommendations to build a bridge-conditioned model.

**Latest result — usable frozen spatial representation:** [SPATIAL_BRIDGE_RETURNED_FINDINGS.md](SPATIAL_BRIDGE_RETURNED_FINDINGS.md) validates the completed spatial bridge. Native surfaces passed through the target voxelizer and frozen Stage-1 VAE beat the mean-latent control on 29/32 objects; reserved latent MSE .15664 is below the mean control .21169 and trained processed readout .30898. This supports an actual Stage-1 spatial conditioning comparison, while oracle alignment and sparse-touch integration remain unresolved. The one-voxel sensitivity control also shows substantial latent error from small spatial perturbations. Do not rerun the completed probe.

**Completed design — frozen spatial bridge control:** [SPATIAL_BRIDGE_HANDOFF.md](SPATIAL_BRIDGE_HANDOFF.md) records the no-training probe, its controls and outcome branches. The GPU results have returned; see the findings above.

**Latest result — stop extending the readout:** [FEATURE_READOUT_CONTINUED_FINDINGS.md](FEATURE_READOUT_CONTINUED_FINDINGS.md) validates all three 8000-update runs and exact reproduction of their pre-resume endpoints. Fitted MSE falls to .02211 raw / .02405 slot-aware / .01514 processed, while reserved errors rise to .30201 / .29033 / .30898 (mean-target control .21169). The readout can learn substantial training-object mappings; more of this training is not justified as a transfer fix. The next branch is an explicit spatial interface comparison, preserving the sparse-touch endpoint. No further continuation is requested.

**Latest complete comparison — native processing improves fitting, transfer unresolved:** [FEATURE_READOUT_RETURNED_FINDINGS.md](FEATURE_READOUT_RETURNED_FINDINGS.md) validates all three matched reports. Processed fitted MSE .09019 beats raw .11625 on 23/24 objects, but reserved MSE .24418 remains worse than the mean-target control .21169. All arms are still improving on fitted objects. A bounded continuation to 8000 total updates follows the predeclared optimization branch; `run_feature_readout_continue.sh` runs on an already allocated interactive GPU node. No production fix is established.

**Completed diagnostic — frozen-feature spatial readout:** [FEATURE_READOUT_HANDOFF.md](FEATURE_READOUT_HANDOFF.md) records the original three-arm design using saved representation artifacts. Raw, slot-aware and native-processed features feed matched spatial-query heads; 24 objects fit and eight separate objects are reserved. All three initial GPU runs have returned; see the complete comparison above. This is not a sparse-touch completion model or a validated production fix.

**Latest returned representation evidence and sparse endpoint:** [REPRESENTATION_RETURNED_FINDINGS.md](REPRESENTATION_RETURNED_FINDINGS.md) validates the completed 32-object probe. Native VecSetX surface F-score averages 96.54% oracle / 96.96% camera at the declared tolerance; naive point-grid SAM encoding only changes mean occupancy IoU from 61.70% to 62.46%, so the proposed approximate-target spatial adapter is not justified on that input. The next branch concerns access/translation of usable point features. Sparse structured touch patches are the final objective: preserve location and observation validity; no-position bbox/radius normalization discards global shift/scale and cannot alone distinguish translated copies of a local patch. No new neural experiment is requested here.

**Ready to run — first no-training representation check:** [REPRESENTATION_PROBE_HANDOFF.md](REPRESENTATION_PROBE_HANDOFF.md) describes `run_representation_probe.sh`. It tests native VecSetX reconstruction in camera/oracle frames and frozen SAM VAE encoding of actual point-hit grids for 32 objects, without loading the diffusion model. Intermediate representations and validation hashes are saved. Run on the cluster and return the completed JSON plus Slurm log before choosing a new conditioning implementation.

**Latest plan review — test a constructive alternative first:** [PLAN_REVIEW_AND_CONSTRUCTIVE_PATH.md](PLAN_REVIEW_AND_CONSTRUCTIVE_PATH.md) supersedes the initial A–D ordering. First check the actual surface's native reconstruction and whether SAM3D's own VAE yields a usable spatial condition; if it does, test a direct spatial residual adapter against the current route. Keep broader Stage-1 finetuning as a targeted comparison. Decompose a successful combined intervention and then restore camera-frame input before a full run. A new five-object CPU audit confirms point-hit voxels agree with targets but cover only 47–97% of target occupancy, so naive point voxelization cannot be assumed equivalent to the target grid. No new neural experiment has executed.

**Current next-experiment plan — Stage 1 only:** [NEXT_STAGE1_EXPERIMENT_PLAN.md](NEXT_STAGE1_EXPERIMENT_PLAN.md) follows the user's scope correction. Establish an informative fitting scale, then cross visual content present/absent with restricted/broader Stage-1 shape adaptation while holding oracle surfaces and frozen VecSetX fixed. Outcome branches determine whether to test fusion changes, minimal additional trainable modules, representation positive controls, or camera-frame robustness. Stage 2 and mesh CD are outside this plan. This is a proposed design, not an executed experiment or another full-dataset oracle request.

**Latest returned result — no convincing target-orientation fix:** [TARGET_FRAME_RETURNED_FINDINGS.md](TARGET_FRAME_RETURNED_FINDINGS.md) validates the completed eight-object probe. Original targets regenerate exactly and all candidate VAE reconstructions have IoU ≥0.99959. A yaw chosen on one view improves independent-view/noise flow error by only 0.344%, while sampled IoU worsens from 10.96% to 9.16%; a global yaw correction also fails to transfer. A modest low-time loss effect remains, so this does not prove orientation irrelevant or architecture replacement necessary. No coordinate patch, dataset re-encoding or additional GPU run is recommended from this result. The next unresolved branch concerns learning from stable surface features despite visual-view dependence.

**Latest coordinate audit and GPU handoff:** [TARGET_FRAME_FINDINGS.md](TARGET_FRAME_FINDINGS.md) verifies that all 32 local saved source-to-render rotations exactly match stock SAM3D voxelization's rotation. Applying that rotation again would double it. A new no-training probe tests the distinct remaining hypothesis of object-specific target orientation versus the pretrained prior, with original-target regeneration, held-view/noise checks, and sampled geometry. Run `run_target_frame_probe.sh`; outputs use `outputs/conditioning_investigation/`. This supersedes the earlier “no additional GPU run” status below, while keeping the full-dataset oracle recommendation withdrawn.

**Current decision — no repeat full-dataset oracle run:** The user reports having already run oracle conditioning with results similar to the other variants. The current job scripts do not capture that history; their absence of the flag was insufficient evidence that the experiment had never been run. The recommendation to run `run_full_dataset_frame.sh` is withdrawn. The small multi-view benefit demonstrates some frame sensitivity, but is not a convincing explanation of the original failure given the oracle's privileged information. The strongest unresolved observation is the dependence on image/pointmap view despite essentially unchanged complete oracle surface features. No additional GPU run is currently requested. Future investigation outputs remain under `outputs/conditioning_investigation/`.

**Previous result — multi-view fitting:** [MULTIPLE_VIEW_FIT_FINDINGS.md](MULTIPLE_VIEW_FIT_FINDINGS.md) validates all three returned runs. Oracle surfaces lower fit loss by 19.1% and reserved-view loss by 25.5% relative to camera surfaces, with reserved-view IoU improving from 52.19% to 66.95%. Substantial errors and falling training losses remain; this is not a demonstrated solution to the original dataset-wide issue.

**Previous result — view transfer fails despite stable oracle surfaces:** [VIEW_TRANSFER_FINDINGS.md](VIEW_TRANSFER_FINDINGS.md) validates exact anchor reproduction and matched new-view probes. Oracle loss stays at 0.01887 when only same-object surface view changes, but rises to 0.12771 when image/pointmap changes with the surface fixed. The subsequent multi-view fitting experiment has now completed; see the latest result above.

**Previous result — successful tiny fitting:** [TINY_FIT_FINDINGS.md](TINY_FIT_FINDINGS.md) validates the three returned 1,000-update runs. Camera-frame surfaces achieve 99.20% raw target voxel IoU at CFG 0, versus 90.07% for image + pointmap; swapped surfaces triple native loss. Oracle coordinates provide no qualitative fitting rescue because both surface arms succeed. This demonstrates small-set fitting with frozen VecSetX and cross-attention adaptation, not dataset-wide generalization. The subsequent view-transfer probe has now completed; see the latest result above.

**Previous result — returned geometry bundle:** [GEOMETRY_FINDINGS.md](GEOMETRY_FINDINGS.md) documents verified inputs and sample-dependent output orientation errors. These do not by themselves explain aligned Stage-2 CD. The subsequent tiny-fit experiment has now completed; see the latest result above.

**Previous result — actual rollouts:** [ROLLOUT_FINDINGS.md](ROLLOUT_FINDINGS.md) shows guidance 7 improves sampled geometry relative to guidance 0. The large velocity-error increase did not predict reconstruction degradation. The requested CPU geometry bundle has now been received and analyzed; see the latest update above.

**Update after GPU job 46083371:** the pilot has completed. See [GPU_FINDINGS.md](GPU_FINDINGS.md) for validated results and the next runnable job. Strong CFG substantially worsens measured velocity error in both checkpoints; correct surfaces confer little consistent incremental benefit. Actual reconstruction rollouts are now the next dependency. The sections below preserve the initial source/local audit and its original hypotheses.

## Material Passport

- Scope: source audit, reanalysis of 21 exported runs, independent NumPy geometry checks, and a GPU diagnostic handoff.
- Repository inspected: `afb805859204e9056d8a007e54cf9a8f030e5789`, with pre-existing edits in `make_orbit.py`, `jobs/evaluate.sh`, `make_eval_orbit.py`, and a deleted weekly note. Those changes were not modified.
- Status: **local checks executed; root cause unresolved; GPU experiment required**. No SAM3D or VecSetX network was executed in this investigation.
- Evidence: `local_evidence.json`, generated by `audit_local.py`; current source; previous diagnostics under `outputs/diagnostics` (treated as prior evidence, not newly executed experiments).
- New files are confined to this folder. The folder was ignored during the initial audit; the current repository no longer ignores it. Check that new follow-up files are included when transferring or committing them.

## What the evidence establishes

**A simple stored-camera sign error is unlikely in the inspected subset.** Across 64 selected samples / 32 objects, the maximum difference between independently inverse-transformed views of each object's full surface is `2.136e-7` object units. Pointmap reprojection has worst per-sample 99th-percentile error `2.880e-5` pixels. Deliberately omitting the XY sign flip produces a median per-sample median error of `278.54` pixels. This is a sensitive check of that particular failure. It does not independently validate the source mesh, voxel axes, target VAE, or all cluster samples.

**The full surfaces contain hidden observations.** The median hidden-point fraction is `62.18%` in these files. They are not merely the visible pointmap. This does not establish that their hidden geometry is unpredictable from the images.

**Uniform SSI scale/shift cannot explain the non-position encoder's geometry loss by itself.** Define `N(P) = (P - bbox_center(P))/max_radius(P)`. For positive scalar `a`, `N(aP+b)=N(P)`. The current ObjectCentricSSI uses an isotropic scale; `TouchEncoder` subsequently applies N. The independent NumPy check gives maximum discrepancy `4.44e-16`. Thus SSI scale and translation cancel before VecSetX in the non-position, non-joint full-surface path. This result does not apply to non-affine remapping, anisotropic scale, clipping, the position side-channel, or a different cluster normalizer. Camera rotation does not cancel, and axis-aligned bbox centering does not commute with rotation. The oracle path removes both effects, not rotation alone.

**The available runs do not demonstrate an inability to fit training data.** Approximate interval-weighted training loss for non-position full surface drops from `0.09155` at epoch 9 to `0.08622` at epoch 20 while validation has already reached its minimum. These are stochastic flow losses, not decoded training reconstruction errors. No tiny-set overfit or paired training-set Chamfer results were found in the provided exports.

| Full cross-attention configuration | Best validation flow loss | Step |
|---|---:|---:|
| Image + pointmap | 0.08857805 | 8796 |
| Image + pointmap + full surface | 0.08885871 | 6597 |
| Image + pointmap + full surface, `learn` features | 0.08865379 | 8796 |
| Image, no pointmap | 0.08920783 | 8063 |
| Image + full surface, no pointmap | 0.08901460 | 8796 |

These differences are small, single-run, minimum-over-epochs comparisons. They have no paired sampling uncertainty or training-seed replication. In particular, the no-pointmap full-surface run has a slightly **better** best flow loss than its image counterpart; the supplied exports do not establish that every full-surface configuration performs worse. The `learn` run is marked failed, although its history includes step 14660 and 20 validation measurements; its output log ends earlier. Verify its checkpoint before using it. Neither oracle job appears in the 21 exported runs.

**There has not been a scratch-encoder comparison in this code path.** `TouchEncoder.__init__` always loads a pretrained VecSetX checkpoint, including when `--train-vecsetx` is set. That flag unfreezes selected encode parameters; it does not initialize from scratch. The exported full-surface full-CA arms freeze the encoder. Touch-patch finetuning does not answer whether a trainable full-surface encoder can fit.

## Two important measurement/interface confounds

### Guidance differs between training and mesh evaluation

`train.build_stage1_pipeline` sets `p_unconditional=0`, uses conditional-only training, and adapts shared shape cross-attention. `evaluate.build_pipeline` constructs the inference pipeline and places the generator in eval mode. The inference pipeline's defaults set Stage-1 CFG strength 7 on `[0,500]`; the available sibling pipeline YAML does not override that. Actual cluster YAML still needs verification.

The inspected CFG class computes `(1+s)*v_cond - s*v_uncond`. At s=7 this is `8*v_cond - 7*v_uncond`. With a perfect conditional prediction, an unconditional error delta contributes `-7*delta`; consequently conditional fitting alone does not guarantee faithful guided sampling. Updated K/V biases and full cross-attention weights can change the zero-condition branch despite receiving no unconditional training examples. This is a concrete risk, **not evidence that CFG caused the observed CD**. Both image and point models can be affected.

The current evaluator explicitly sets `no_shortcut=True`: a proposed d=0-versus-shortcut bug is **not supported** by this source. Its Stage-1 sampling uses d=0, consistent with training. Do not spend a run “fixing” that already-correct setting.

### The reported CD is downstream of more than Stage 1

`evaluate.evaluate_condition` decodes Stage-1 occupancy, prunes/downsamples support, samples Stage 2 using the image, generates a mesh, normalizes it, and fits similarity/rotation plus ICP against the target. Its CD is symmetric mean **Euclidean** nearest-neighbor distance, not squared CD. Even `stage1_aligned_chamfer` uses each model's Stage-2 mesh registration. Those aligned scores cannot diagnose target-frame correctness by themselves.

The user's 0.0096 and 0.036–0.040 values were not found as per-sample evaluation artifacts in the provided folders. They remain user-reported. If 0.0096 is the evaluator's GT-latent condition, it is the result of GT Stage-1 support followed by Stage 2 and registration, rather than a bare VAE reconstruction floor. It is a valuable reference, but compare identical objects/views/seeds and separate raw Stage-1 latent/occupancy errors from Stage-2 CD.

### VecSetX's native decoder accesses a richer representation

The default adapter feeds `encode()['x']` (1024 slots × 32 channels) through a shared per-slot MLP. Native VecSetX uses `bottleneck.post`, adds a distinct learned embedding to every slot, and runs 24 transformer blocks before spatial decoding. This is confirmed by the local implementation and [upstream VecSetX](https://raw.githubusercontent.com/1zb/VecSetX/master/vecset/models/autoencoder.py).

The current non-learn adapter and downstream cross-attention are invariant to permutation of those 1024 input slots: shared per-token transforms commute with permutation, and attention over paired K/V is invariant to their order. The shared modality vector and global position vector do not supply individual slot identity. Native `learn` adds slot-specific vectors before processing, so it has a different contract. This proves a difference in accessible information, **not that real geometry is lost**: identity may already be recoverable from token values, and arbitrary permutations may not occur among real encoder outputs. The near-baseline `learn` run also argues against treating missing slot embeddings as a demonstrated sole cause.

## First GPU job: measure whether this checkpoint uses shape information

Transfer this folder, then from the cluster repository root run:

```bash
sbatch doc/conditioning_investigation_2026-09-11/run_gpu.sh
```

The launcher now requests one H100 for at most one hour. Job 46083371 completed successfully and its returned JSON is archived here. It did not train or overwrite a checkpoint. The next requested job is `run_rollout_gpu.sh`; see the update above rather than rerunning this first probe.

The pilot compares the image+pointmap and full-surface best checkpoints on the same eight training objects, one view each, identical target latents and noise. It records four draws at t=0, .2, .5, .8 (0=noise; t=0 is a diagnostic endpoint rather than a normal continuous training draw). It checks its explicit loss against native `generator.loss` before accepting measurements. Controls retain point-token count:

- True full-surface tokens versus tokens from a different object, holding recipient image/pointmap/target/noise fixed. This intervention is deliberately inconsistent; degradation shows dependence, not automatically useful geometric reconstruction or generalization.
- Reverse slot order, which should leave predictions unchanged up to numerical error; this tests the stated interface symmetry, not encoder quality.
- Conditional predictions versus analytically combined CFG strengths 1 and 7 at the same states, using the inspected native formula. This identifies guidance-sensitive velocity error, not guided rollout/CD performance.

The probe logs runtime/config/source provenance and asserts pairing. It does not invent attention or flow implementations. Do not interpret a failed native-loss assertion as a model result; first resolve the objective/runtime difference. If permutation differences are material, rerun the affected case in fp32 before concluding the source-level invariant is violated.

## Decision sequence after the pilot

1. **If point swapping has negligible effect:** confirm against repeat-forward numerical variation and a broader paired sample. Measure gradients to projected tokens and optimize a single shared-per-example token code against many training noise draws, validating on fresh draws. The earlier `outputs/diagnostics/FINDINGS_AND_NEXT_EXPERIMENT.md` has the fitting protocol. This tests downstream capacity without replacing the encoder. Success localizes a representational/learning issue; failure alone does not prove impossibility.
2. **If points matter but CFG worsens velocity errors:** run matched Stage-1 sampling with CFG=0,1,7, identical initial noise, first on those training examples. Decode unregistered occupancy/latent metrics; only then compare paired Stage-2 CD. Change production guidance only if rollout evidence supports it.
3. **Establish real fitting capacity:** fix 1 object/view, then 8 objects/views; train current camera-frame and oracle-frame arms from matched initial weights with fresh flow noise, evaluating fresh noise and CFG=0 rollouts on the same training examples. Fix preprocessing and evaluate last checkpoints, not only validation-selected best. Use image-only fitting as a control. Verify gradients and actual parameter changes. Declare any short optimizer budget inconclusive if fitting is still improving; validate optimizer/numerics before claiming a capacity limit.
4. **If oracle fits and camera does not:** orientation/frame interpretation is implicated. Separate orientation from bbox-normalization order using the documented rotation-only bridge. An oracle success is not a deployable fix. Add learned alignment only after this result.
5. **If both fail but free tokens fit:** test shared projector fitting, slot-aware/learn readout, then full-surface encoder finetuning. Compare scratch only after proving the intended parameters can update and the downstream interface can fit. More data is not the first response to a demonstrated tiny-set fitting failure.
6. **If small-set fitting succeeds but validation does not:** investigate generalization/data diversity. Reevaluate all 95 validation objects with one paired score per object (or aggregate views within object), report paired differences and object-level uncertainty, then use repeated training seeds for small effects. A matched-visible/different-hidden-shape benchmark can test whether extra geometry resolves genuine ambiguity.

No finite failed optimization proves an inherent SAM3D limitation. Evidence for a hard limit would need a representational collision on relevant inputs or a controlled positive control that succeeds when a specific restriction is removed, alongside validated optimization and numerical behavior.

## Handoff boundary

The laptop lacks the pretrained SAM3D/VecSetX weights and GPU runtime. Further source-only discussion cannot choose among the remaining causal branches. **The next substantive evidence must come from the GPU pilot or existing oracle/paired evaluation outputs.** The overall investigation remains open; no root-cause fix has been claimed.

For local reproduction, run `audit_local.py` with a NumPy-enabled Python. This session used the bundled Python at `/Users/michaelwakeham/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`. The audit completed successfully, and both Python scripts passed compilation; the launcher passed `bash -n`.
