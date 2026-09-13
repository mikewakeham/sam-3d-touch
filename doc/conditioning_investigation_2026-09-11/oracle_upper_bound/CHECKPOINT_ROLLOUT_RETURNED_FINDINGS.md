# Full-data Stage-1 rollout findings and directed next decisions

13 September 2026. Continues F17/F18; no new training or production-source changes.

## Governing question

Does the full-data oracle/dropout checkpoint use correctly aligned surfaces to generate accurate Stage-1 geometry, including on its training objects? Establish this upper bound before attempting inference-time pose recovery. Full surfaces and frozen VecSetX remain the working setup; future sparse touch does not justify a GT-geometry conditioning shortcut.

The user requests an explicit explanation before every further experiment: the unresolved question, why prior evidence leaves it open, exactly what changes and what stays fixed, the reference/control, and what each possible result would cause us to do. A command is not an explanation. Do not launch another broad experimental campaign or full training run merely because this endpoint failed.

## Why the two post-training probes were needed

F17's similar pooled training/validation curves did not distinguish ignored surfaces from useful conditioning masked by losses at other noise levels. F18 therefore restored paired final oracle/constant checkpoints and compared correct/wrong surfaces with visuals present/zero, on training and disjoint validation identities, using identical actual noise and times. It established modest held-object utility near pure noise but no native-validation advantage. These losses evaluate velocity at states constructed using the target, not a complete generation trajectory.

The present rollout then started from pure Gaussian noise, without target latent values as model inputs, and integrated the model's own predictions. It tests whether the F18 benefit survives generation and whether generated supports approach supports decoded from the exact target latent. Only the Stage-1 decoder is involved. This is the Stage-1 equivalent of a decoded-target reference, not Stage-2 evaluation.

## Verification and measured scope

- ZIP SHA256: `644b04d4d8b09b7247fafa6b88ae49b7eec8190b31de28b01b1584124824f44a`.
- Archive CRCs, all224 NPZ payload hashes, all11 reported source hashes, summary reproduction and all three pasted JSON/archive equalities pass.
- Same final epoch20/step14660 oracle/constant checkpoints,16 train and16 validation objects, two views and two paired draws each.768 generated outputs across correct/wrong/constant and visual-present/zero cells.
-25 Euler steps, CFG0, no shortcut, existing time-rescaling convention. One sampler setting; numerical adequacy is not established by the checkpoint audit.
- Saved occupancies allow independent raw-metric recomputation and proper-rigid analysis. The validation objects are repeatedly used development observations, not untouched confirmation data. One training seed/checkpoint per arm;64 outputs per cell are not64 independent objects.

## Raw generated geometry

Values below are mean object F-scores in percent, with proximity tolerance two voxels in the64-cubed Stage-1 grid. Precision measures predicted support close to target support; recall measures target support close to prediction. F-score combines both. It is not a percentage of semantic shape recovered.

| Split / visual inputs | Correct oracle | Wrong surface, same oracle model | Trained constant control | Correct minus constant |
|---|---:|---:|---:|---:|
| Train / present |75.87|38.77|63.72|+12.15pp|
| Validation / present |64.51|42.19|49.69|+14.82pp|
| Train / zero |71.78|18.59|14.94|+56.84pp|
| Validation / zero |50.77|16.03|14.47|+36.30pp|

With visuals present, correct oracle beats constant on13/16 training and15/16 validation object means; it beats the fixed wrong surface on16/16 training and14/16 validation objects. Both visual-zero oracle/constant differences are positive on all16 objects.

**Established:** sample-specific surfaces improve generated target-referenced geometry over the matched constant pathway at this scope. The full-data run is not a complete conditioning failure. The earlier tiny constant-control result cannot be generalized to claim these new gains are merely extra adaptation.

**Still failed:** accurate upper-bound reconstruction. Raw mean oracle IoU is31.52% train /19.25% validation with visuals, far below the declared target-support endpoint. Visual removal increases relative surface dependence but reduces absolute oracle fidelity; this is not evidence that removing images fixes the current model.

No sampled image-only/image+pointmap checkpoint or camera checkpoint is in this comparison. Do not claim generated superiority over those baselines, or isolate a new oracle-versus-camera effect from these results.

## Distinguishing pose from remaining support error

The independent local analysis gives every prediction the same proper-rotation/translation search, preserving scale and excluding reflections. It compares the identity with the fixed registration candidate using minimum two-voxel precision/recall. Positive alignments are valid witnesses; poor registration is not a proof that no better rigid alignment exists.

Correct-oracle raw to pose-witness F-score: train/present75.87→84.52; validation/present64.51→77.23; train/zero71.78→79.01; validation/zero50.77→56.67. Only2/16 object means reach both95% precision and95% recall in each of the first three cells, and0/16 in validation/zero. Thus measured rigid correction improves fidelity but does not demonstrate the near-target upper bound.

An independent conservative all-rigid diameter/projection bound certifies one stronger failure: validation object `0f396dc864c74ed6a308dcb1610d78e5`, view000/draw0, visuals present, cannot exceed81.89% target recall within two voxels under **any rotation and translation**. All32 target self/known-rigid positive controls pass. This proves at least one error beyond rigid output pose, involving shape/extent. It does not exclude a scale correction, certify all other failures, or prove an architecture limitation.

The matched local analysis is complete: all768 raw support metrics independently reproduce, and four known-rigid registration positive controls pass. After giving all models the same pose search:

| Split / visuals | Correct oracle F-score | Constant F-score | Wrong-surface F-score | Oracle minus constant / positive objects |
|---|---:|---:|---:|---|
| Train / present |84.52|79.14|51.55|+5.38pp /11 of16|
| Validation / present |77.23|68.10|59.15|+9.13pp /14 of16|
| Train / zero |79.01|28.18|30.24|+50.83pp /16 of16|
| Validation / zero |56.67|24.95|26.41|+31.73pp /16 of16|

Thus the measured oracle/constant advantage shrinks after pose correction but persists, including on14/16 validation object means with visuals. The raw advantage must not be described as entirely intrinsic-shape improvement. The remaining positive witness comparison supports geometry benefit beyond what this common registration procedure removes, not a globally optimized pose-invariant comparison.

See `rollouts_returned_20260913_143515/rigid_bounds.json` and `rollouts_geometry_20260913_143515/summary.json`, with per-prediction transforms in `predictions.jsonl`.

## Recommended next decision tree — not a new GPU handoff

1. **Check the numerical sampling alternative before more training.** Reuse final oracle and constant checkpoints, paired starting noise, correct surfaces and visual-present inputs. A fixed subset spanning the same16 training/16 validation objects, one predetermined view/draw per object, suffices for a first25/50/100-step comparison. Keep CFG/time convention/decoder fixed. Reuse25-step outputs. If50/100 converge near the target reference, improve inference before changing training. If they converge far below it, additional steps are not the main remedy on those observations. If not converged, report that uncertainty rather than announcing a training limitation. This is a bounded check, not an open-ended sampler search. The previous four-object success at25 steps is positive evidence that the sampler can work, but not a guarantee for the new vector field.

2. **If fitting remains poor, test training flexibility against equal extra exposure.** A matched short continuation from the same full-data oracle checkpoint can compare the current projector/full shape-cross-attention scope with the existing broader shape-generator scope. Keep oracle points, VecSetX, targets, dropout policy, optimizer continuity for existing parameters, batches/noise and number of updates matched. Use a defined training-object subset to make fitting measurable quickly, retaining validation observations for transfer assessment. The exact subset/budget must be fixed in the handoff, not selected after seeing results. Current-scope recovery would support an optimization/exposure explanation; improvement only with broader scope would show additional generator flexibility helps beyond extra exposure. Neither alone proves full-dataset success or that broad finetuning is necessary. Earlier broader-scope evidence used camera-oriented targets and four objects, so it cannot settle this current aligned full-data checkpoint.

3. **If fitting becomes accurate but validation remains poor, focus on transfer.** Then the question changes to reusable aligned geometry across identities/data exposure. Do not attribute it to missing camera pose while oracle inputs are supplied. Promote a larger continuation only with a specific improvement and an intermediate checkpoint plan, not because a subset was memorized.

4. **If oracle fitting and transfer become useful and accurate, return to camera alignment.** Compare the existing camera model under the same assessment before selecting pose recovery or another target-frame strategy. Oracle is a controlled reference, not a deployable pose estimator.

If both sampling and matched training interventions fail, inspect the conditioning-to-generator mapping at the failing noise regimes before replacing the representation. Negative finite-budget trials cannot establish that VecSetX or SAM3D fundamentally cannot solve the task. No sparse-touch full run, GT-voxel bridge or renewed global-axis search is selected here.

## Remaining confounds to keep explicit

The full run means full dataset, not full backbone finetuning: most generator weights remain frozen. Tiny and full runs used the same nominal projector/CA learning rates, but exposure and task size differ substantially. Four-object1000-step fitting gave each view about250 presentations;20 full epochs give each view about20. This makes underoptimization plausible, not proven. Dropout changed from tiny whole-batch to SAM3D-style per-sample as already documented. These are reasons for matched controls, not grounds to automatically prescribe more epochs or reverse dropout integration.
