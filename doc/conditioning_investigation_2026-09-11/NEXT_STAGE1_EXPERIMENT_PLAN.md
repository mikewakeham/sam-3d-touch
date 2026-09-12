# Stage-1 diagnosis: adaptive experiment plan

## Material Passport

- Origin: local repository audit and returned conditioning experiments; experiment-agent planning workflow.
- Status: proposed design, not executed and not a validated fix.
- Scope: Stage-1 shape learning only. Stage 2 and mesh CD are outside this plan.
- Inputs: existing dataset/targets, frozen pretrained initialization, archived tiny-fit, multi-view and target-frame results in this directory.
- Outputs for future runs: `outputs/conditioning_investigation/stage1_diagnosis/` with immutable run directories.
- Current deliverable: design and decision rules. No new training launchers or production changes are made by this plan.

## Questions in priority order

1. Does restricting training to the existing projector and shape cross-attention prevent learning a robust surface-to-shape mapping?
2. Does the varying visual condition interfere with learning that mapping, even when surface and target frames agree?
3. If neither intervention rescues learning, is the restriction in the frozen surface representation/readout or farther downstream?
4. Once a configuration demonstrably learns from surfaces, how much does restoring camera-frame input degrade it, and what is the smallest effective remedy?

These are alternatives, not established diagnoses. Four-object fitting already succeeds, so reproducing that result is not the objective. Stable surface features plus poor view transfer establish visual sensitivity, not visual interference as the sole cause. The latest orientation experiment gives no reason for another global-axis patch or target-yaw search.

## Shared experiment contract

Start with 32 seeded training objects, four fixed views per object. Reserve four other views of those objects and 32 distinct objects for diagnostic transfer. Use object-level splits and exact sample manifests. Reuse the same objects, ordering, surface point samples, initial weights, targets and stochastic noise/time schedule across compared arms. The first scope is fitting the training set; transfer distinguishes memorization from learning a reusable geometric mapping. Do not tune hyperparameters on the reserved objects; retain the main dataset validation split for later confirmation.

Use oracle target-frame surfaces initially. This is an intentional diagnostic simplification: it removes the frame-recovery burden while examining the rest of the setup. Targets remain unchanged. Keep VecSetX frozen, position conditioning off, and the existing feature variant fixed in the first comparisons. Cache frozen features to save GPU time, verifying cached/uncached equality. Freezing does not by itself prove a representation is adequate.

Primary measurements:

- Native Stage-1 shape flow loss on the fitted records, using a fixed bank of fresh noise and times drawn from the actual training sampler. Also report loss by time bin; an equally weighted three-time average is not the training objective.
- Stage-1 conditional rollout latent error, plus decoded occupancy IoU as a secondary check. Start with CFG 0 to inspect learned conditional behavior. Use paired initial noise. Decoding corroborates learning; it is not the sole success criterion.
- Same measurements for reserved views and separate objects, reported separately and averaged per object. Surface-only oracle inputs for different views may be nearly identical, so reserved-view success there is an invariance check, not evidence of geometric generalization.
- Correct versus wrong-object surfaces with the same recipient target, visuals and noise. Report both prediction change and loss change; swapping can create contradictory conditions and is not a standalone proof of geometric understanding.
- Per-module gradient/update norms, numerical health, trainable parameter names/counts, learning curves and optimizer/RNG checkpoints.

Use fixed monitoring draws plus an independent final audit bank. Save all arms at the same milestones; do not compare each arm's best validation checkpoint. Initial screening uses one training seed; replicate decisive comparisons with a second seed before adopting a change. Statistical units are objects and training replicas, not repeated noise draws.

## Experiment 1: locate an informative fitting regime

Run A below first for 2,000 updates, effective batch four, balanced sampling of all 128 fitted records. That is 250 presentations per object, about 62–63 per exact view; it is substantially less exposure than the previous tiny fit. Checkpoints every 500 updates allow continuation without restart. Compare per-object exposure, not just update count.

If learning is still clearly improving, continue in matched blocks to 8,000 updates (1,000 presentations per object), before calling a persistently high error a failure. This is a compute cap, not a proof of convergence. Estimate runtime and peak memory from the first 100 updates before scheduling the remaining arms; no defensible wall-time estimate is available locally.

- If A fits well but transfer is poor, keep this set: it is informative about generalization, not a fitting-capacity failure.
- If A both fits and transfers well, move A to a predetermined larger subset, e.g. 128 training objects. Do not run all alternative arms at a scale where the baseline already succeeds.
- If A fails and still improves at the cap, label budget-limited. Compare alternatives as learning-efficiency tests, not architectural upper bounds.
- If A fails and stabilizes, check numerical behavior and a small training-only learning-rate bracket before interpreting the plateau. A single inappropriate optimizer setting is not evidence of a hard architectural limit.

Do not repeatedly grow object count until an arbitrary failure appears. At most one planned size increase precedes reassessment of the existing full-dataset checkpoint/training evidence.

## Experiment 2: separate visual fusion from adaptation capacity

At the informative scale, complete this 2×2 comparison, starting every arm from the same pretrained weights. A is reused from Experiment 1.

| Arm | Conditions | Trainable SAM3D components |
|---|---|---|
| A | Image + pointmap + oracle surface | Existing full shape cross-attention and norm2 |
| B | Oracle surface; visual content removed | Same as A |
| C | Image + pointmap + oracle surface | Broader Stage-1 shape computation |
| D | Oracle surface; visual content removed | Same broader scope as C |

All arms train the same surface projector; all keep VecSetX and image encoders frozen. Broader scope means shape self-attention, MLPs, relevant norms/modulation and shape input/output mappings as well as cross-attention. Include shared parameters that affect shape and document those explicitly; layout-exclusive modules and the VAE remain frozen. This is a Stage-1 adaptation comparison, not unfreezing the entire pipeline. The current `cross_attention_scope=full` selects cross-attention and norm2 only; it is not this broader intervention.

For B/D, zero visual condition content after the image/pointmap embedder and before concatenating surface tokens, preserving condition length. Do not set the global CFG-unconditional flag: the current wrapper zeros the concatenated surface tokens too. Verify the actual surface condition remains nonzero and trainable. Zero visual slots can still receive attention through biases, so this removes visual content, not all competition for attention slots. If B/D help, a subsequent removal-versus-zeroing control can separate that effect.

Use existing projector LR 1e-4 and CA LR 1e-5 as the first settings. Add a separate conservative LR group for newly trainable shape parameters (initial proposal 1e-6). A negative broader-adaptation result needs a brief training-only bracket, e.g. 3e-7/1e-6/3e-6, before it supports a restriction beyond scope. Give failing restricted arms an equivalent small CA-LR check. Compare equal exposure first and report GPU cost separately; these are not equal-compute models.

| Result pattern, at matched exposure | Supported interpretation | Next experiment and possible setup change |
|---|---|---|
| B improves substantially over A; D adds little over B | Varying visual content makes surface learning harder under this setup | Train A with per-example visual dropout; separate RGB/mask from pointmap dropout. If a modest dropout rate recovers the gain with visuals present at test time, adopt that training change. If only total removal works, test separate residual surface attention with a learned gate. |
| C improves substantially over A; B does not | Broader shape adaptation helps despite unchanged surface features | Start fresh with shape MLPs/norms added to CA, then add self-attention only if needed. Find the smallest scope retaining C's gain. Do not conclude all SAM3D weights or VecSetX must train. |
| Only D succeeds | Visual conditioning and frozen shape computation interact | Complete both comparisons D–B and D–C; neither factor alone suffices in this budget. Test visual dropout with broader adaptation, then reduce trainable scope. |
| B and C both help; D helps further | Both interventions contribute | Combine the effective interventions, then remove one at a time to identify a practical minimal setup. |
| All fit, but only some transfer to separate objects | The difference concerns geometric generalization rather than fitting capacity | Prioritize the transferring arm; compare against matched image+pointmap without surfaces, and apply surface swaps. Improvement must exceed an image-only prior before attributing it to geometry. |
| All fit; none transfers | Small-set fitting is not the barrier | Move to the representation/generalization branch below; a further tiny-fit success or larger finetuning scope alone cannot explain the failure. |
| All fail or remain similarly poor | These two interventions did not isolate the bottleneck | Run the free-code positive control below before changing encoders or invoking data quality. Treat still-falling runs as inconclusive. |

“Substantial” means a persistent paired difference across objects and training windows that changes the branch, corroborated by Stage-1 rollout behavior where appropriate. A tiny loss improvement with worse sampled latents does not earn a production change. Report magnitudes and repeat the decisive contrast; do not manufacture a significance claim from this screening set.

## Experiment 3, conditional: locate representation versus downstream restrictions

Only enter this branch when Experiment 2 leaves a relevant failure.

**Free-code fitting control:** replace frozen VecSetX feature arrays with a learnable 1024×32 code per training object, shared across its views; retain the same projector, modality embedding, token count and downstream path. Initialize codes from that object's normal oracle features. Use B's condition setup first; compare D's broader scope only if necessary. Fit across fresh flow noise and assess on independent noise. This removes the need to extract geometry from points without giving the model explicit target latents. It is a privileged memorization control, not a deployable encoder or a generalization score.

- Free codes succeed, fixed features fail: the downstream interface has fitting capacity with an adapted representation. Next hold downstream scope fixed and compare slot-aware/`learn` readout, pretrained VecSetX finetuning, and only then a scratch encoder if pretrained finetuning remains inadequate. Improvement localizes an accessibility/learnability problem; it does not prove the original codes lost information.
- Free codes fail under restricted scope but succeed under broader scope: downstream adaptation is a bottleneck even without point extraction. Focus on the minimal shape modules needed before changing the encoder.
- Free codes fail under both: inspect objective, optimization and trainable-state integrity; use direct supervised latent prediction from the same object codes as an implementation positive control. A finite failed fit still cannot prove SAM3D cannot represent the solution.

**If fitting succeeds but geometric transfer fails:** test a separately trained, spatial-query decoder from frozen VecSetX features to the target Stage-1 latent. Compare raw features, slot-aware features and trainable pretrained encoding under the same direct prediction head. Use exactly the same object split and a wrong-object control. This is a bypass diagnostic, not a proposed replacement model.

- Good direct-head transfer but poor SAM3D transfer suggests the geometry is accessible and the generative conditioning/training path is the issue.
- Trainable encoding improves both paths while fixed encoding does not: encoder adaptation becomes supported.
- Neither direct head nor SAM3D transfers: this alone does not establish information loss. Check native VecSetX reconstruction from these exact inputs and point preprocessing before blaming dataset size. Native reconstruction succeeds → representation may be difficult to translate; native reconstruction fails → isolate preprocessing, point density and orientation before retraining.

Avoid running every encoder variant at once. The preceding positive control determines which comparison is informative. Existing `--train-vecsetx` loads pretrained weights and unfreezes encode parameters; it is not a scratch condition.

## Experiment 4, conditional: return to camera coordinates with a functioning learner

After an oracle-frame configuration shows useful Stage-1 surface learning, compare it with camera-frame input under the same successful architecture, data and training exposure. Now a loss of performance has a working positive control; repeating old oracle training without that control adds little.

- Camera and oracle similar: coordinate recovery is not the important residual restriction at this scale. Keep current coordinates.
- Camera materially worse: distinguish the frozen encoder's rotation sensitivity from the need to recover target orientation. Apply known synthetic rotations to oracle points with visuals held fixed, and verify inverse rotation **before encoding** restores the original feature/prediction. Then train controlled rotation augmentation with canonical targets. This is a robustness intervention, not proof that arbitrary target orientation is identifiable from any point set.
- Rotation augmentation closes the gap: retain camera coordinates with that training intervention.
- It does not: test an explicit alignment module supervised by available camera transforms. Compare predicted alignment with oracle alignment; pose error and shape error together localize the remaining burden. Do not assume rotation-equivariant features alone solve canonical pose selection.
- Alignment remains ambiguous or unreliable: only then consider changing the output convention to a deterministic observation-defined frame and re-encoding targets consistently. This changes the learning task and must be tested as such. A few failed optimizers do not prove arbitrary-frame recovery impossible; an impossibility claim needs demonstrated ambiguous inputs or a formal identifiability argument with assumptions matching the dataset.

## Resources and stopping rules

First allocation: A's 2,000-update screen on one suitable cluster GPU, including throughput/memory profiling and resumable state. Then B/C/D at the informative scale and matched milestones. Broader-shape optimizer memory must be measured; use activation checkpointing/accumulation while preserving effective batch and stochastic schedule if needed. Do not silently narrow the scope after an OOM and call it a full-adaptation test. No full-dataset retrain is in the initial allocation.

Prioritize Experiment 2 over more coordinate sweeps, four-object memorization controls, scratch encoders or hyperparameter grids. Stop each branch once its result makes the next decision clear. An unresolved noise/precision/provenance discrepancy is an invalid experiment; a still-improving curve is an unfinished optimization experiment; a fitting success is not geometric generalization. No training/evaluation procedure in this plan invokes Stage 2.
