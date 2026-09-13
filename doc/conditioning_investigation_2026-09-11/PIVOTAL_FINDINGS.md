# Pivotal findings — living evidence and experiment decisions

Last reconciled: 13 September 2026. This is the short, governing evidence ledger requested by the user. Read it before proposing, implementing or interpreting the next experiment. Update it when evidence materially changes a claim; retain its scope and any counterevidence. Detailed findings and raw reports remain linked below. This is a selected subset, not a chronological experiment log.

**Objective:** make Stage-1 full-surface point conditioning work, while retaining VecSetX and a future path to sparse structured touch. Coordinate handling is the current focus. Better fitting, corrected input coordinates, robustness across views and useful conditioning on new objects are separate accomplishments.

**Current boundary:** continue from the completed coordinate controls. The full-dataset checkpoint inventory is optional corroboration, not a prerequisite or a restarted baseline campaign. No new GPU experiment is selected by this ledger. Do not restart completed fits or promote the failed separate-attention candidate.

## What counts as improvement, gap closure, or a solution?

| Claim | Comparison / reference required | What that reference does not establish |
|---|---|---|
| Input coordinates corrected | Transform camera points into the intended target frame; compare with independently known object-frame points and known inverse. Exact transformation has zero error up to numerical precision. | The network can decode or use those points, or that pose is available at deployment. |
| Rotation makes learning harder | Matched training treatments separating orientation from normalization, with the same target/visual inputs, initialization, scope and exposure. | A universal cause of poor performance or an asymptotic fitting ceiling. |
| Measured view gap nearly closed | Compare reserved-view and fitted-view errors under the same assessment protocol, before and after intervention. Also ensure the gap did not shrink simply by making fitted performance worse. | Accurate reconstruction: equally poor fit/held performance also has a zero gap. Formal equivalence needs a justified, prespecified margin and uncertainty. |
| Training examples reconstructed accurately | Compare generated Stage-1 support with support decoded from the exact target latent, in the same frame and with the same decoder. Perfect support agreement has IoU 1; target-latent equality has MSE 0. | Fidelity beyond the Stage-1 target representation, new-view/new-object transfer, or superiority caused specifically by geometric information. |
| Surface information is usefully conditioned on | Correct surface beats appropriate wrong-surface controls and an appropriate image baseline; a trained constant-surface control separates sample-specific geometry from extra adaptation where needed. Evaluate the claimed split. | Complete geometric fidelity. Surface identity can explain success on a handful of fitted objects. Token removal alone is an OOD intervention, not a measure of useful geometry. |
| A coordinate intervention solves the practical coordinate subproblem | It corrects alignment on the claimed observations without unavailable GT pose and meets a prespecified geometric tolerance; compare downstream behavior against the existing exact-alignment reference where appropriate. | Solving residual failures that already occur under perfect supplied alignment. |
| Full-surface conditioning works generally | Target-referenced Stage-1 performance plus useful surface dependence on new objects, with adequate replication for the size of the effect. | Sparse-touch readiness or a guarantee outside the tested distribution. |

Native flow loss is the actual training objective, not an arbitrary score. But it measures velocity prediction at target-derived noisy states; it is not an absolute reconstruction-quality scale. Its algebraic minimum is zero for exact velocity prediction. The fitted model's nonzero loss is not an established optimal floor. Oracle *input alignment* is also not an oracle *output prediction*. Do not translate .02 versus .06 directly into a percentage of recovered geometry, or compare direct-regression MSE, native flow MSE and sampled geometry as interchangeable errors.

Decoding GT latents supplies an achievable reference through the same downstream measurement path. The earlier tiny-fit experiment used this principle at Stage 1. It does not require Stage 2 or final mesh CD. Preserve raw fixed-frame results; report target-assisted rigid alignment separately when explicitly testing pose-independent shape. It cannot count as a deployed correction or erase a fixed-frame training error. A new Stage-1 reconstruction check is justified when needed to promote a candidate to a reconstruction claim, not automatically for every diagnostic.

**Current governing audit/branch selection:** [EVIDENCE_AUDIT_AND_BRANCH_PLAN.md](EVIDENCE_AUDIT_AND_BRANCH_PLAN.md). F11/F12 are completed. At the user's request, overnight implementation is parked and the short coordinate intervention continues: [SHARED_ORIENTATION_HANDOFF.md](SHARED_ORIENTATION_HANDOFF.md) implements one camera-oriented-target arm with unchanged camera inputs and a reused historical control. Target quality is checked before training; physical tolerance is compared in common original units. This is an experimental target contract, not a production output-frame decision. The 1000-update report has returned (F13): the shared-target intervention fails accurate reconstruction even on fitted views. Raw-array/pose verification is complete: the shared-target result has both pose and residual geometry errors; see the F13 follow-up. No new GPU fit, full run or encoder replacement is selected.

## F0 — Early saved predictions already demonstrated pose/shape disagreement

**Reference:** independent reanalysis of 20 predictions from five deliberately selected original-model cases. Target arrays and raw metrics reproduced. Several large raw Stage-1 occupied-center distances fell substantially under proper axis rotations: shield surface .13983→.01040, knife image .20161→.00512. Different objects and seeds required different rotations, and some residual shape errors remained.

**Established:** target-frame error can coexist with substantial recovered shape. A single global coordinate patch is not supported by these varying outputs.

**Limits:** selected older outputs, target-assisted best alignments, no population prevalence or deployment correction. This does not prove that the latest dropout perturbations are pose-only; it makes that a necessary distinction before interpreting them.

**Next-experiment consequence:** carry raw fixed-frame and separately pose-adjusted shape endpoints when diagnosing coordinate effects. This historical finding was omitted as a standalone ledger entry before the current audit; it is not a newly run experiment.

Source: [geometry bundle findings](GEOMETRY_FINDINGS.md).

## F1 — No simple global coordinate bookkeeping fix is supported

**Evidence/reference:** across 64 selected views of 32 objects, inverse-transformed full surfaces agreed to 2.14e-7 object units and pointmap reprojection errors were below 2.88e-5 px. A deliberately wrong camera sign produced large errors. Saved source-to-object axes already include the stock conversion; adding it again would double-rotate targets. The eight-object GPU target-regeneration check reproduced original targets exactly. Six tested alternative target orientations supplied no convincing transferable correction.

**Established:** the inspected data/source contract is internally consistent. The spatial target latent is view-independent but orientation-dependent.

**Not established:** correctness of every cluster record, a universal semantic front convention, or impossibility of a different learned coordinate strategy.

**Next-experiment consequence:** retain the audited transforms; do not repeat a global axis search or rotate latent channels as xyz.

Sources: [initial audit](README.md#what-the-evidence-establishes), [target convention](TARGET_FRAME_FINDINGS.md), [target probe](TARGET_FRAME_RETURNED_FINDINGS.md).

## F2 — The existing surface pathway can accurately fit four objects

**Reference:** Stage-1 predictions sampled from noise, decoded and compared directly with occupancy decoded from each GT target latent. Ideal agreement is 100% IoU.

After 1000 single-view updates, CFG-0 mean IoU was **99.197% camera surface**, **99.188% oracle surface**, and **90.070% image+pointmap**. The worst camera result across four objects and two sampling draws was 97.338%. Frozen VecSetX and projector/full shape-CA adaptation were sufficient. Camera wrong-surface flow loss rose .01995 → .06197.

**Established:** near-complete decoded-target-support recovery is possible in this fitting task. The pathway is not completely broken and full-backbone finetuning is not universally necessary to fit.

**Not established:** reusable geometry interpretation, success at dataset scale, or a new-object solution. The surface may identify a memorized target, and image alone also fits strongly.

**Next-experiment consequence:** another four-object fitting success alone has low diagnostic value; do not claim an inherent incompatibility based on larger-task failure.

Sources: [tiny-fit findings](TINY_FIT_FINDINGS.md), [archived aggregate](tiny_fit_summary.json), `tiny_fit_gpu.py` decodes GT support before assessing generated support.

## F3 — Rotation is a causal learning burden in the controlled multi-view task

Four objects, one training seed, 1000 updates; all factors except supplied surface coordinates held fixed:

| Orientation | Normalization basis | Reserved-view native loss |
|---|---|---:|
| Object | Object | .060126 |
| Camera | Object | .080737 |
| Object | Camera | .064269 |
| Camera | Camera | .080713 |

**Established:** rotation alone reproduces essentially the measured camera/oracle penalty. Removing rotation while retaining camera normalization helps. Oracle inversion removes the camera rotation from the points before encoding, and same-object oracle features become nearly identical across views.

**Reference limit:** oracle is the known-alignment treatment, not a perfect reconstruction endpoint. Its reserved-view Stage-1 IoU here was only 66.95% versus camera 52.19%; its fitted IoU was 96.93%. A better oracle score does not mean this task was solved. Rotation/normalization interact; do not assign additive percentages of blame.

**Audit clarification:** both the native objective and these sampled occupancy comparisons use the original fixed target frame. The experiment isolates a learning burden for that task; it does not quantify pose-independent shape damage. It also used the original visual policy, so it does not decompose the later dropout-policy gap.

**Next-experiment consequence:** a rotation-handling intervention has an established geometric reference and measured task effect. It need not rediscover that effect, but must distinguish accurate alignment from improved Stage-1 reconstruction.

Sources: [factorial findings](FRAME_FACTORS_RETURNED_FINDINGS.md), [analysis](frame_factors_returned_manual/analysis.json), [pose input contract](POSE_INPUT_CONTRACT.md).

## F4 — Alignment alone leaves visual dependence; pointmap was not its main measured cause

**Controlled comparisons:** with oracle surface and target fixed, changing visual view raised single-view-fitted loss .01887 → .12771. The later visual-stream factorial found anchor .01795, reserved pointmap-only .01917, RGB-only .06202, silhouette-only .04186, and coherent reserved visuals .05591. Crossed modalities can be inconsistent; use the coherent endpoint and factorial interactions rather than adding isolated penalties.

**Established:** the measured residual does not require rotating surface features. RGB/silhouette dependence dominated pointmap changes in this four-object oracle model.

**Not established:** all visual dependence is undesirable, a universal pointmap conclusion, or an internal memorization mechanism.

**Next-experiment consequence:** do not repeat the pointmap-versus-RGB diagnosis or call this residual a proven pointmap-frame conflict.

Sources: [view transfer](VIEW_TRANSFER_FINDINGS.md), [visual-stream factorial](ORACLE_VISUAL_STREAMS_RETURNED_FINDINGS.md).

## F5 — Visual dropout nearly closed a measured same-object view gap; reconstruction success was not established by that result

All visual inputs were present during the following fresh-bank assessments. Dropout was a training intervention; these are native Stage-1 losses:

| Oracle training | Fitted views | Reserved views of the same objects | Reserved minus fitted |
|---|---:|---:|---:|
| Original | .0238468262 | .0578488732 | .0340020470 |
| Visual dropout | .0220465500 | .0227320715 | .0006855215 |

The gap decreased about 98%, while fitted error also improved. Reserved loss ended 3.11% above fitted loss; every reserved-view group improved. Wrong-object surfaces raised reserved loss to .22535. Camera+dropout remained worse at .05381, so dropout did not erase the camera/oracle difference.

**Established wording:** “The aggregate native-loss gap between fitted and reserved views of these four identities nearly closed.” This is more specific than merely “did better.”

**Not established:** formal equivalence/invariance, accuracy of every individual object, accurate generated reconstruction for these dropout checkpoints, or generalization to new identities. No prespecified equivalence margin or target-decoded reconstruction endpoint was used to establish the dropout claim. Do not borrow the separate single-view tiny-fit experiment's 99.2% IoU for these weights/views. “Resolved view dependence” without these qualifications is too broad.

**Subsequent evidence:** F9 below now directly establishes high target-referenced Stage-1 reconstruction for these dropout weights on reserved views of the four fitted identities. The preceding paragraph records what the original native-loss experiment alone established; its reconstruction uncertainty is superseded by F9, while its new-identity and formal-invariance limits remain.

**Next-experiment consequence:** retain this checkpoint as a measured same-identity robustness reference, not a proven general geometric conditioner. A reconstruction claim needs an actual Stage-1 target-referenced check of the relevant checkpoint.

Sources: [dropout findings](VISUAL_DROPOUT_RETURNED_FINDINGS.md), [exact aggregate](visual_dropout_returned_manual/analysis.json). Gap arithmetic rechecked from that JSON on 12 September.

## F6 — The strongest tiny-task improvements did not transfer on the native objective

**References:** on 32 different objects, image loss was .10540, original oracle .21229 and dropout oracle .27079. Every surface variant lost to image on all 32. In the subsequent matched 16-object training screen, oracle held-object loss was .13420 versus image .11930, again losing on all 16; oracle improved fitted identities. Correct surfaces were not reliably preferred to wrong ones on held identities.

**Established:** alignment/dropout had not produced a transferable conditioning solution in these experiments. F2/F3/F5 remain true within their scopes; this is counterevidence to promoting them into a general fix.

**Not established:** that the original full-data models have the same cause, that more data/full finetuning is required, or that canonicalization is impossible. Training-object identification is consistent with the pattern, not directly observed internally.

**Audit clarification:** these particular 32-object and 16-object flow-model screens measured native loss, not sampled held-object shape. Their objective-level failure remains established; pose-independent generated-shape failure is not established by those numbers. The later 16-training/16-held pool is exactly the earlier 32-object probe pool (set equality checked during the audit). Held identities were excluded from fitting, but repeated investigation makes them a development set, not untouched confirmation data.

**Next-experiment consequence:** rotation recovery can be investigated as its own geometric subproblem. Do not promise that approaching oracle performance solves the failures that already occur under oracle alignment.

Sources: [32-object check](UNSEEN_OBJECTS_RETURNED_FINDINGS.md), [16-object training](OBJECT_TRANSFER_RETURNED_FINDINGS.md).

## F7 — Extra adaptation can mimic a conditioning gain without sample-specific surfaces

**Matched control:** training with one fixed surface for every example achieved fitted loss .05382 versus real oracle .05647 and image .06161. Constant held-object loss .12790 was still worse than image .11930. Early wrong-surface probes also preserved much of the apparent surface-model advantage.

**Established:** fitting/early improvement over image alone is insufficient evidence of useful object-specific surface information in this setup. Constant input and real input had the same trainable architecture; tokens are not automatically extra parameters.

**Not established:** all geometry is ignored, parameter count is the root cause, or all constant banks behave identically.

**Next-experiment consequence:** distinguish geometry-specific gains from an additional learned pathway. Neither token-presence sensitivity nor swap sensitivity on a few fitted objects is a complete success criterion.

Sources: [early/late controls](TRANSFER_CHECKPOINTS_RETURNED_FINDINGS.md), [constant training](CONSTANT_SURFACE_RETURNED_FINDINGS.md).

## F8 — Preserving the image model and separating surface attention did not fix transfer

**References:** separate oracle fitted loss .02897, but held-object loss .13854 versus frozen image .11930. Removing the branch exactly restored image outputs. Correct surfaces were preferred by all fitted identities, but only 7/16 held identities. Real surfaces beat constant on average, but both failed the image baseline.

**Established:** this candidate can fit through the new branch, yet fails the transfer criterion while preserving the original image function. Failure does not require changing image weights or a shared visual/surface softmax.

**Not established:** every fusion architecture fails or broader finetuning is necessary. This intervention combined several changes; it does not isolate all mechanisms of previous joint models.

**Audit clarification:** the rejection is against the native-loss and surface-utility criteria actually measured. Sampled held-object reconstruction up to pose was not assessed for these arms. Preserve that scope rather than promoting the rejection into a claim about every generated-shape metric.

**Next-experiment consequence:** reject this candidate at its tested scope; no automatic extension, full run or another architecture follows. Return to the identified coordinate subproblem with the residual failure explicit.

Source: [separate-attention findings](SEPARATE_SURFACE_RETURNED_FINDINGS.md).

## Guardrails retained from secondary experiments

- **Visual dropout is not a production default from this evidence.** F5 establishes improved same-identity view robustness; F6 shows worse new-identity performance in the tested four-object models. It encourages dependence on the surface stream but does not prove a transferable interpretation of geometry. Keep the distinction explicit when discussing “generalization.”
- **Sparse/hidden touch and dropout:** not experimentally tested. Whole-visual dropout removes complementary visible-shape information when sparse hidden-surface contacts do not specify the whole object. A generative model may still learn a conditional distribution from such inputs, but this is a different and more ambiguous task than the full-surface control. Neither benefit nor failure is established. Occasional partial visual dropout or a coverage-dependent policy are future hypotheses, not recommendations to enable now. The decisive eventual test uses both modalities present, held objects and structured hidden contacts, with correct/wrong-touch controls. Full surfaces remain the current priority; do not open that sparse-policy branch before the coordinate work calls for it.
- Native VecSetX reconstruction preserves substantial full-surface geometry. Small direct readouts eventually fit but fail transfer; this is not proof that VecSetX must be replaced. [Representation](REPRESENTATION_RETURNED_FINDINGS.md), [continued readout](FEATURE_READOUT_CONTINUED_FINDINGS.md).
- The complete-mesh/voxel bridge was diagnostic only and is withdrawn as the production conditioner. Sparse structured touch must remain feasible. [Requirements](TOUCH_CONDITIONING_REQUIREMENTS.md).
- Lower fixed-state flow/latent error did not always imply better sampled geometry: disabling CFG reduced some errors but worsened generated support in the earlier rollout comparison. Keep objective-specific and reconstruction claims separate. This does not queue another guidance or Stage-2 investigation. [Rollout correction](ROLLOUT_FINDINGS.md).

## Mandatory use before the next experiment

Record, briefly, before handing off a run:

1. **Starting evidence:** relevant F0–F12 IDs and the exact unresolved question; do not recreate completed evidence as a new discovery.
2. **Claim being tested:** coordinate correctness, fitting, view robustness, geometry-specific use, or transfer. Name any entanglement that requires changing another factor.
3. **Comparisons:** intervention, matched control, reference endpoint and claimed split. State whether the reference is perfect target agreement, exact alignment, a fitted-model score, or an image baseline.
4. **Success/failure/inconclusive criteria:** justify the tolerance or effect size before results; include uncertainty/replication appropriate to the scope. A smaller scalar alone is not “resolved.” Retrospective margins cannot convert prior results into formal equivalence.
5. **Decision branches:** what each outcome changes; distinguish a diagnosis from a deployable fix and a coordinate fix from an overall conditioning fix.

After the result, update the relevant finding, comparison and scope. Add a new finding only if it changes a material decision. Never let a later summary silently broaden a small-task claim. Do not launch new training merely to satisfy this documentation checklist.

All GPU evidence above consists of returned reports with the documented validation/replay checks. No independent local GPU rerun is claimed. Noise draws and multiple views of one identity are not independent objects or training replicas.

## F9 — Oracle plus dropout reaches the Stage-1 target on reserved views of fitted identities

**Reference:** decoded GT Stage-1 occupancy (IoU 1 is perfect agreement). Original oracle fit/reserved mean IoU is 96.93%/66.95%; dropout oracle is 97.80%/97.17%. The lowest dropout reserved object-mean is 91.96%, so it passes the prespecified ≥95% mean/≥90% each-object reference gate. Wrong surfaces give 2.49% reserved IoU. Raw shard 0 passed local validation and original sampled replay.

**Established:** F5's native-loss improvement has a sampled reconstruction counterpart; we now have an accurate aligned endpoint for this coordinate-interface test. Surface identity matters to these fitted-identity outputs.

**Limits:** four trained identities, three reserved views, two noise draws. Two individual reserved samples remain below 90%, minimum 88.58%. This does not distinguish geometric interpretation from memorized identity retrieval or supersede F6's new-object failure.

**Next-experiment consequence:** use this actual checkpoint's target-referenced endpoint when assessing imperfect or recovered alignment. No further loss-only verification of whether the dropout reference reconstructs these objects is needed.

Source: [alignment results](ALIGNMENT_TOLERANCE_RETURNED_FINDINGS.md).

## F10 — The accurate aligned endpoint loses fixed-frame agreement under some small residual rotations

**Intervention:** rotate oracle-aligned points before the existing VecSetX normalization/encoder; hold visuals, targets, weights and noise fixed. At 5°, reserved mean IoU spans 94.14–97.13% across six signed axes, but the worst object-mean is 85.51%; only +x and −y pass the prespecified per-object tolerance. At 30°, means span 64.74–85.63%, worst object-mean 13.56%, and no direction passes. These raw shards passed local validation. The supplied aggregate's 15° means span 79.00–93.25%, worst object 42.21%; its raw shard 2 is still missing, so those intermediate-angle values are not independently reproduced locally.

**Validation update:** the geometry bundle subsequently supplied shard 2. All four reports now reproduce the aggregate, and all saved raw occupancy metrics were independently reproduced on CPU. The earlier missing-data qualification is resolved.

**Established:** the complete current conditioning path can reconstruct near its target under exact alignment yet lose meaningful fixed-frame agreement under small residual pose errors. A coarse pose correction is not automatically adequate for that criterion.

**Limits:** no universal angular bound, arbitrary-axis coverage, learned pose estimate, localization of sensitivity within VecSetX/normalization/attention, new-object transfer, or inherent incompatibility is established. Normalization responds to the perturbation too.

**Interpretation correction after the user's output-frame question:** these IoU results measure agreement in the original target frame. They have not distinguished a correct shape rotated in the output from an actually distorted/wrong shape. “Coarse pose is insufficient” applies to the specified target-frame accuracy criterion, not yet to pose-independent reconstruction. F9's high aligned accuracy remains established.

**Next-experiment consequence:** before launching residual-rotation augmentation, inspect the already saved perturbed predictions for rigid-pose versus shape error. Compare their raw support, known inverse-rotation compensation, and a separately identified rigid-registration diagnostic against GT, accounting for voxel discretization and registration failure. This uses the existing artifacts and does not require a new training run. Registration to GT is a diagnostic, not a deployable inference correction. Small residual-rotation augmentation remains a candidate if orientation errors actually damage shape or if fixed-frame output is required; it is not yet implemented or launched. A successful local robustness repair would still need observable frame recovery and separate held-object utility before full training.

Source: [alignment results](ALIGNMENT_TOLERANCE_RETURNED_FINDINGS.md), [local validation](alignment_tolerance_returned_manual/local_validation.json).

## F11 — Pose adjustment does not generally rescue the errors, but the small-angle IoU result overstated broad shape fragility

**Evidence:** all 1,232 existing saved predictions analyzed on CPU, all raw IoU/count/distance values reproduced. Proper rigid registration passed known-transform, revoxelization and wrong-object controls. All four trained identities, fit/reserved views and two seeds retained; no new model inference. F-score uses one-voxel proximity (1/64), a different and less exact criterion than fixed-grid IoU.

**Residual rotations:** at 5°, raw F-score averages 99.67% before any alignment, with 2/144 individual reserved predictions below 95% precision/recall. Both failures are one bathtub view/noise draw under two directions; they are real exceptions. At 15°/30°, raw means are 95.87%/88.32%; corresponding individual failures are 17/144 and 40/144, and the tested rigid transform does not make those failures pass. Known inverse perturbation worsens F-score in 431/432 reserved cases. Outputs do not generally follow the imposed input rotation in a way that its inverse corrects.

**Natural inputs:** original-policy camera raw→registered F-score is 72.83→80.47%; original-policy oracle is 84.47→85.32%; oracle dropout is already 99.997% raw. Some individual predictions benefit greatly from pose correction (one camera drill 2.03→79.72%), but it does not generally reach the accurate endpoint. Registered two-voxel scores are 92.13%, 95.35% and 100%, respectively, so the severity depends on spatial tolerance.

**Corroborated shape failure:** a +30° drill prediction retains only 16.38% registered one-voxel /33.64% two-voxel F-score. Visible thickening and covariance eigenvalue changes, far beyond the revoxelization control, corroborate non-pose error. Registration is not globally certified and its RMS objective can worsen thresholded F-score; raw results and this limitation remain explicit. A secondary best-of-raw/registered check does not rescue the overall original-model comparison.

**Established consequence:** neither a universal sub-5° requirement nor a general post-hoc pose-only fix is supported. Small perturbations are usually shape-tolerable at voxel resolution, with rare failures; larger perturbations can change shape and pose. This is a fitted-identity result, not generalization or a deployed correction.

**Next-experiment consequence:** avoid automatically training a precision pose head or augmenting an already mostly robust four-object oracle model. First sample the already trained camera-plus-dropout model to complete the same-policy natural-input geometry comparison: its native loss alone cannot establish its shape quality. That inference-only control can decide whether a sampled frame burden remains under the improved policy. A shared camera-oriented target task, local augmentation or a new-object utility check follows the appropriate outcome; no full run is earned yet.

Source: [pose/shape findings](POSE_SHAPE_RETURNED_FINDINGS.md), [complete summary](pose_shape_analysis/summary.json), [comparison figure](pose_shape_analysis/rotation_shape_comparison.png).

**Execution update:** the control in [CAMERA_DROPOUT_GEOMETRY_HANDOFF.md](CAMERA_DROPOUT_GEOMETRY_HANDOFF.md) has returned and is analyzed below. Do not rerun it.

## F12 — Camera plus dropout retains a large frame-treatment penalty, including a wrong-object failure

**Matched reference:** same four fitted identities, fit/reserved views, two sampling draws, CFG 0, 25 steps, same visual-dropout training policy. All 112 returned predictions and 57 manifest members validated; reported code hashes match and all 112 historical native-loss replay values agree exactly. GT-support contents and actual sampling noise match the oracle comparison. All saved geometry metrics were reproduced on CPU with the unchanged proper-rigid analysis.

**Reserved shape:** camera-dropout IoU is **56.17%**, versus oracle-dropout **97.17%**. Raw one-voxel F-score is **67.13% vs 99.997%**; under the prespecified identity-or-rigid witness, **79.26% vs 99.997%**. Only **one of four** camera objects meets the precision/recall reference. Wrong surfaces reach **88.97%** witness F-score, beating correct surfaces by 9.71 pp on average and on three object means. Individual results are mixed: correct wins 14/24, wrong 9/24, one tie. Fitted camera views have 99.63% raw F-score; the major problem here is reserved-view behavior, not complete inability to fit.

**Concrete wrong-shape evidence:** an exploratory all-four-target comparison of every reserved correct prediction finds that bathtub view `e846574d…_000` produces the fitted **shield** for both draws: **99.966%/99.890% raw F-score to the shield**, versus **31.74%/31.80% after rigid fitting to the bathtub**. This is not merely the correct object in a different pose. It suggests view-dependent interpretation of fitted surface codes, but does not identify the responsible encoder/projector/attention mechanism or prove general shape transport in the successful oracle model.

**Correction to broad dropout language:** camera native reserved loss improved .07718→.05381, but its same-metric pose-witness F-score is **82.99%→79.26%**. Dropout has a strong sampled success with oracle alignment (F9); it has **not** resolved camera-frame view dependence or demonstrated camera geometry improvement. Do not enable it as a universal repair from these results.

**Established:** a large sampled frame-treatment penalty remains under matched dropout policy, including errors that are not only output pose. Camera/oracle changes orientation and normalization together; F3 isolates rotation under the original policy, but the full dropout gap is not quantitatively attributable to rotation alone. This is a finite-budget, fitted-identity result, not impossibility or new-object evidence.

**Next-experiment consequence:** stop measuring this missing cell; it is complete. Advance to a coordinate intervention. Shared observable target orientation is the leading branch if original asset axes are unnecessary; otherwise a deployment-available frame recovery strategy is required. Both retain VecSetX and sparse-touch constraints. No full run is earned without a short successful intervention and new-object surface utility. Await the output-frame preference before choosing the changed target contract.

Source: [returned findings](CAMERA_DROPOUT_GEOMETRY_RETURNED_FINDINGS.md), [numerical summary](camera_dropout_geometry_analysis/summary.json), [identity diagnostic](camera_dropout_geometry_analysis/cross_identity.json), [failure illustration](camera_dropout_geometry_analysis/bathtub_shield_failure.png).

**Subsequent execution decision, not new evidence:** the user paused overnight work and requested continued short coordinate experiments. [SHARED_ORIENTATION_HANDOFF.md](SHARED_ORIENTATION_HANDOFF.md) now implements the shared-orientation label intervention. Diagnostic comparison can proceed without committing the final product to that output frame. It reuses the existing camera/dropout control, gates new target VAE fidelity before updates, and retains known inverse transforms for comparison in common original units. Success would still require new-object utility before scaling; failure is not an architecture-impossibility result. GPU output is pending.

**Target preflight evidence, before any training:** all four original latents regenerate exactly. All 28 continuously camera-oriented target labels round-trip through the frozen VAE at 99.994–100% IoU in the returned report. The original one-voxel inter-grid gate nevertheless rejected 11 faithful labels. The geometric voxelization bound is 1.577–1.833 original voxels, and all labels agree 100% within two voxels. The upcoming shared-frame comparison is therefore calibrated at TWO original voxels for ALL arms/conditions, while retaining original one-voxel metrics and strict native-target IoU. This corrects an unattainable/miscalibrated reference requirement before optimization; it is not evidence that the trained conditioner works. See `shared_orientation_target_preflight/analysis.json` and the handoff for limits. No target-VAE encoding barrier is observed on these 28 labels; model adaptation remains untested.


## F13 — Camera-oriented target supervision does not by itself fix conditioning

**Reference:** completed 1000-update shared-orientation report; exact historical source/reference/init/input/feature/dropout/sampling-noise checks pass. Target encoding/decoding preserves all 28 new labels at 99.994%–100% IoU. Shared-arm geometry values are currently report-derived; the raw bundle is pending.

With correct surfaces, fitted/reserved native-target IoU is **27.81% / 15.90%**, versus **88.32% / 56.17%** for camera/dropout with original fixed targets and **97.80% / 97.17%** for oracle/dropout. Common-original-unit raw two-voxel F-score is **78.60% / 54.48%**, versus **99.97% / 69.83%** and **100% / 100%**, respectively. All arms use the same two-voxel physical tolerance here; each native IoU uses its own decoded target.

**Established at this budget:** unchanged camera inputs plus camera-oriented labels fail the prespecified reconstruction endpoint, including low fitted-view fidelity. A target-VAE encoding failure does not explain it. Matching axes is insufficient with the current frozen VecSetX and projector/shape-CA adaptation. Correct versus wrong surface advantage is only +4.45 pp raw two-voxel F-score on reserved views, with one object negative.

**Limits:** changing labels introduces view-dependent spatial targets and associated normalization, whereas oracle input alignment preserves a fixed target per object. Native loss continues improving (.17069→.13322 fitted from update300→1000); this is not an asymptotic ceiling or evidence that full finetuning is necessary. Pose-independent shape quality is pending the bundle. F3/F9 are not contradicted; no VecSetX replacement or architecture impossibility follows. Four fitted identities, one training seed.

**Next-experiment consequence:** obtain/verify the already generated geometry bundle and separate output pose error from fitted-shape failure before choosing another fit. If fitted shape fails, a matched adaptation-scope versus continuation comparison is relevant; if shape is accurate after pose correction, pursue the output-orientation branch. Do not promote this arm to a full run.

Source: [shared-orientation returned findings](SHARED_ORIENTATION_RETURNED_FINDINGS.md), `shared_orientation_returned_manual/summary.json`, `validation.json`.


### F13 follow-up — bundle verified; pose correction is not a universal rescue

All 86 bundle payload hashes/CRCs and all 112 prediction metrics reproduce independently; all 28 target roundtrips reproduce. At the common two-voxel tolerance, shared-arm fitted/reserved F-score improves **78.60%→91.57% / 54.48%→77.53%** with the predeclared identity/rigid witness. Matched old camera/dropout reaches **99.98% / 89.19%**, oracle/dropout **100% / 100%**. Only **18/32 fitted and 4/24 reserved** shared outputs pass 95% bidirectional proximity individually. No shared-arm object passes the stricter per-object target-reference endpoint. Reserved surface-dependence advantages after pose adjustment are +19.79, −2.80, +7.88 and −12.26 pp; the each-object ≥10 pp screen fails.

**Stronger specific result:** reserved drill `fe200ce…_000`, draw0, cannot exceed **82.71% target recall under ANY rotation/translation**, proved by a diameter/projection-interval coverage bound. All 28 faithful-target positive controls pass; other predictions may be inconclusive under this bound. This removes registration convergence as an explanation for at least this shape/extent failure. It is not a proof about all examples, fitted scaling, semantic topology or model capacity.

**Decision:** mixed pose-and-geometry failure, not a final-axis patch. Matching camera axes alone has not fixed conditioning. Next relevant GPU branch is matched additional training with current versus broader generator adaptation, keeping the encoder/labels/inputs fixed; compare equal additional exposure and preserve optimizer-state continuity. The matched continuation is now implemented: [SHARED_ORIENTATION_SCOPE_HANDOFF.md](SHARED_ORIENTATION_SCOPE_HANDOFF.md). CPU protocol checks passed; GPU execution pending. More training remains a live alternative because native loss is still improving. The oracle fixed-target route remains viable; this negative result does not refute F3/F9 or prove full finetuning necessary. Full-dataset promotion remains unjustified.

Evidence: [completed shared-orientation findings](SHARED_ORIENTATION_RETURNED_FINDINGS.md), `shared_orientation_geometry_analysis/{summary,complete,rigid_bounds}.json`, [selected diagnostic examples](shared_orientation_geometry_analysis/drill_failures.png).


**Current action after F13 (13 September):** one matched scope continuation is ready, using the completed step1000 checkpoint for both arms, 1000 added updates each, original Adam states preserved, identical new dropout/noise schedule, and unchanged camera observations/targets/VecSetX. The broader arm trains the remaining shape generator path; the control keeps existing scope. No production trainer edits or full-run promotion. Two visible GPUs execute concurrently, one sequentially. [Commands, exact scope and branch rules](SHARED_ORIENTATION_SCOPE_HANDOFF.md). GPU output is now the required next evidence; do not start another local analysis campaign while waiting.
