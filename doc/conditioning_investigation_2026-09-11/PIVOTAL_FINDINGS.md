# Pivotal findings — living evidence and experiment decisions

Last reconciled: 13 September 2026, after the completed full-data frame comparison and evidence reset.

**Read first:** [Evidence reset: meaningful progression and precise coordinate boundary](EVIDENCE_RESET_20260913.md). It supersedes historical next-action recommendations below and in older handoffs. Those recommendations are not a queue of approved or pending experiments. No new GPU experiment is selected by this audit.

**Objective:** accurate Stage-1 full-surface conditioning, retaining VecSetX and a future sparse-touch path. The user's immediate priority is a sound coordinate diagnosis before another representation or training branch.

**Subsequent comparison:** the user selected pointmap/surface frame interaction as the remaining coordinate question. [Pointmap isolation plan](POINTMAP_ISOLATION_PLAN.md) records the missing full-data oracle/no-pointmap treatment, its matched constant control, and conditional shared-frame pointmap follow-up. The existing no-pointmap surface run used camera coordinates. Current 50% visual dropout already exercises surface-only inputs but does not remove pointmap from every training update. Two template-matched four-GPU launchers are prepared under `jobs/`; no training has been launched. Constant is optional for the narrower pointmap-removal question, but supplies the matched geometry-utility control.

**Current result:** oracle removes camera-to-object rotation before encoding on the audited observations. Independent mesh correspondence, exact target regeneration and actual encoder-input checks support the transform and normalization arithmetic. VecSetX's later centering/isotropic scaling does not restore camera rotation. Its normalized units differ from target units, but ideal complete-surface geometry retains a deterministic canonical conversion; finite-sample error has been measured. Different conventions alone do not establish another coordinate bug.

**Model result:** the full-trained oracle uses real surfaces and improves sampled geometry over camera and matched constant conditioning. With visuals present, held-object F-score after common approximate rigid search is 77.23% oracle, 67.83% camera and 68.10% constant; wrong surfaces give 59.15% for oracle. Oracle training-object score is 84.52%, so an accurate upper bound remains unmet. These are Stage-1 two-voxel proximity scores, not Stage-2/CD or percentages of semantic shape recovered. Scope: one training seed, 16 train/16 held development identities, two views/two draws, one sampler setting. No sampled image-baseline comparison is available here.

**Coordinate stopping boundary:** ordinary oracle transform/normalization bookkeeping is verified at the audited scope. No remaining specific erroneous transform has been identified. This does not certify every dataset record or rule out all learning effects of coordinate conventions and visual fusion. Do not relabel an unknown representation-learning difficulty as a demonstrated coordinate mismatch, or use poor oracle generation to falsify an independently verified transform.

**Claim corrections:** the four-object dropout success is not a generalization solution; tiny constant-control results do not erase the later full-data geometry benefit; broader adaptation in camera-oriented-target experiments does not prove full finetuning necessary for the present oracle task. Decoded/query-feature conditioning is an interface change, not a coordinate-only proof. Details and supporting sources are in the evidence reset.

**Communication:** concise, concrete reports. Before any eventual new experiment, state its one unresolved question, changed variable, control/reference and branching interpretation. Do not issue a cluster command without that explanation. All new diagnostics remain in this investigation folder; the earlier production integration was separately authorized.

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

**Current governing evidence/branches:** F20 and [coordinate reassessment](coordinate_reassessment_20260913/REASSESSMENT.md), with F19's [full-data rollout findings](oracle_upper_bound/CHECKPOINT_ROLLOUT_RETURNED_FINDINGS.md). Earlier proposals, handoffs and execution-status paragraphs record historical decisions; their pending labels are superseded by subsequent findings/reviews. The recent metadata and strict-normalization suggestions above are preserved as reviewed alternatives, not active jobs.

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


### Interpretation clarification — original SAM3D also learns across frames

The phrase coordinate mismatch must not imply that different observation/target frames make training invalid. Original SAM3D also maps camera observations to an object shape and a corresponding camera-relative layout. Its paper explicitly couples shape and pose, observes that rotation must be anchored to the predicted shape (Appendix C.1), and supervises shape/rotation from pretraining onward (Table1). It does not establish rotation-invariant native shape loss or a universal semantic canonical orientation. A coherent alternative shape orientation and compensating pose can describe the same scene at inference; that does not make an arbitrarily rotated latent equal to a fixed training target.

The inspected protected attention lets layout read shape, not shape read layout. Hence original layout prediction can place generated shapes, but it is not an internal inverse-rotation step for our VecSetX features; omitting layout loss alone does not explain shape-learning failure. Image-only reconstruction already requires learning across observed viewpoints. Our added camera-frame geometric representation introduces a new mapping into a largely frozen generator; its finite-budget difficulty is measured by F3/F13, not a mathematical impossibility inferred from differing axes.

Appendix E.5 reports minimal average shape change with versus without pointmap (48% preference each), so original pointmap conditioning is not a demonstrated full-surface shape-conditioning solution. The local code allows pointmap information in shape conditioning; do not describe it as architecturally layout-only. Source: https://arxiv.org/html/2511.16624v1 (sections2.1–2.2, Table1, Appendix C.1, E.5); local `POSE_INPUT_CONTRACT.md` and the inspected MoT attention code. This clarification changes interpretation, not the pending scope experiment.


## F14 — Broader generator adaptation improves fitting, not reserved-view transfer

**Verified matched comparison:** both1000-update continuations returned, from the same shared-target step1000 model/Adam state with identical inputs, labels, dropout and sampling noise. All174 payload member hashes and all224 sampled prediction metrics reproduce; source hashes and initial replay/first loss agree across arms. Extra shape-path weights receive gradients and change; frozen weights stay fixed. Full checkpoint/Adam tensors remain on the cluster; their matched content digests are source-audited reports.

**Results:** at total2000, current versus broader scope gives fitted native IoU **53.91% vs81.57%**, but reserved IoU **19.10% vs18.60%**. Fitted common-unit two-voxel raw/pose-witness F-score is96.75/99.13% versus98.78/99.87%; reserved is62.45/84.66% versus59.96/83.51%. The broader arm passes the common-unit fitted-shape reference for **all4 object means and all32 individual predictions** after measured rigid alignment. Neither arm meets native fixed-frame or reserved geometry endpoints.

**Narrowed:** this is not a demonstrated inability of VecSetX/current generator family to represent the fitted geometry in the new target convention. More exposure helps fitting, and broader adaptation helps beyond matched exposure. The remaining demonstrated failure is transfer across natural held views/orientations; broader adaptation is insufficient to fix it. Native fitted loss improves .13322→.04272 while reserved worsens .17711→.22330. Do not keep extending by default or claim full finetuning is a general solution. One seed/four fitted identities; new-object utility still unproven.

**Condition dependence:** broader fitted raw correct-minus-wrong F-score +42.07pp; reserved only+6.24pp raw/+5.08pp after pose witness and inconsistent across objects. Fitted-identity cues remain an alternative to reusable geometric interpretation. A selected bathtub prediction resembles a fitted target, but the180° alternative is approximately symmetric; avoid inferring a general lookup/pose-collapse mechanism from that example.

**Next branch:** natural held views jointly change visuals and geometric orientation. An inference-only visual-zero comparison on the existing broader checkpoint can distinguish visual/geometry frame interaction from a geometry path that still cannot transfer without visuals. This is a different target contract from F5, and50% dropout training alone does not prove the visual stream is harmless. No new training script or full-run promotion follows from this result. See [scope returned findings](SHARED_ORIENTATION_SCOPE_RETURNED_FINDINGS.md) for reference endpoints and conditional branches.


**Execution status after F14 (not new evidence):** [SHARED_ORIENTATION_VISUALS_HANDOFF.md](SHARED_ORIENTATION_VISUALS_HANDOFF.md) now provides the inference-only visual-present/zero × correct/wrong-surface comparison on the existing broader step2000 checkpoint. One GPU,224 sampled outputs, no updates or new target encoding. Local source/archive/schema checks pass; GPU results are required next. Read the handoff for unchanged target-referenced endpoints and branches. Visual-zero improvement alone would show a visual-input contribution, not uniquely establish coordinate conflict or a solved geometry pathway. Overnight implementation remains parked.

### Scale-up decision clarification — tiny-set transfer is evidence, not a universal prerequisite

The previous conversational requirement to beat the image/control baseline on held-out objects before any larger training experiment was too strong. The present four-object fit (and earlier16-object fits) cannot establish how much object diversity a newly learned VecSetX-to-generator mapping needs. The visual pathway starts with substantial pretrained capability; correct full-surface information does not guarantee that a newly adapted interface can exploit it on unseen objects after such limited training. No measured sample-complexity curve establishes that held-object superiority should already occur at these scales.

Keep the negative evidence: earlier held-object losses and constant-surface controls show that demonstrated fitting gains do not establish transferable geometry use. They do not prove that more diverse training cannot help. Likewise reserved views of fitted objects are a more targeted coordinate-transfer test, but failure under finite view coverage is not an architecture-impossibility certificate.

Distinguish two decisions: (1) claiming conditioning is fixed requires held-object, target-referenced fidelity and useful correct-geometry dependence; (2) authorizing a larger diagnostic pilot can be justified by a functioning fitted pathway and a concrete unresolved data-diversity hypothesis, without already winning on held objects. Such a pilot needs a matched comparison, intermediate checkpoints and an explicit question about transfer, not an assumption of success. Tiny-set success thresholds remain unchanged and are not retrospectively relabeled as passed. A larger pilot is not automatically earned by failure either.

The pending inference-only probe remains the cheapest next discriminator and is unchanged. Its result will inform whether the next useful intervention concerns visual interaction, orientation coverage or a broader-data test. No new experiment has been launched, no claim that data scale is the cause has been established, and overnight implementation remains parked pending that decision. This clarification supersedes blanket earlier language requiring tiny-set held-object superiority before any scale-up pilot; it does not weaken the evidence required to claim a resolved conditioning problem.

## F15 — Visual removal does not rescue geometry-orientation transfer

**Verified:** completed inference-only broader-step2000 factorial, all144 payload hashes/CRCs and224 samples. Native visual-present replay is exact; all112 present supports equal F14 bitwise. Same weights, inputs, targets and paired noise. Local geometry reproduces; prior registrations reused only for identical supports/source/frames. Whole visual zero matches the training dropout state. No training, new objects or change to VecSetX.

**Reconstruction:** present→zero fitted native IoU81.57→69.16%, reserved18.60→10.02%. Common-unit raw F-score fitted98.78→96.43%, reserved59.96→36.87%; pose-witness fitted99.87→98.81%, reserved83.51→69.07%. Zero passes the strict fitted common-geometry reference for3/4 object means and27/32 individual predictions, but0/4 reserved object means (6/24 individual passes). Do not overstate the fitted independent pathway as completely solved.

**Surface dependence:** point-only correct-versus-wrong pose-witness advantage is+62.32pp fitted, +23.77pp reserved. All4 fitted objects and3/4 reserved objects pass the10pp dependence screen, but reserved absolute fidelity remains inadequate. Larger gaps after visual removal partly reflect greater deterioration of wrong-surface outputs, not improved correct reconstruction. Fitted-identity recognition is not excluded; new-object geometry readout remains unproven.

**Beyond pose:** zero-visual reserved drill view010/draw1 decodes to4 occupied voxels. A conservative all-rigid diameter/projection bound proves target recall≤26.44% for any rotation/translation at the fixed two-voxel tolerance. All28 faithful-target controls pass. Other predictions are not generally certified by this bound; no fitted-scale or architecture-impossibility conclusion follows.

**Narrowed:** visual suppression is not a rescue of this checkpoint. A substantial transfer failure exists even without visual input at inference, although most fitted shapes can be reconstructed using points alone. Effects of visual competition during training are not excluded by this shared-checkpoint inference test. Natural held observations change geometry orientation, normalization and point sampling; this factorial does not attribute every residual exclusively to rotation. Training with more diverse orientations/objects remains a plausible remedy, not a demonstrated one.

**Next branch:** matched jointly rotated point/target augmentation on visual-dropped updates, retaining the encoder, target contract and equal training exposure. Prefer this controlled orientation-coverage question before another generator-scope change. A cube-rotation bank is an implementation candidate, not yet a GPU handoff. Keep the broader-data pilot option open under the scale-up clarification; four-object held-object superiority is not a universal gate. See [returned findings](SHARED_ORIENTATION_VISUALS_RETURNED_FINDINGS.md) for endpoints, limitations and conditional decisions.


**Execution status after F15 (not new evidence):** [ORIENTATION_COVERAGE_HANDOFF.md](ORIENTATION_COVERAGE_HANDOFF.md) implements the selected matched coverage comparison. Both arms resume broader step2000 with Adam continuity for1000 updates; only the500 visual-dropped treatment updates receive joint point/label cube rotations. The identity control, all visuals-present updates and reserved inputs are unchanged. Three runtime files to sync; one GPU prepares the shared bank, then two GPUs run arms concurrently or one sequentially. CPU transform/protocol/source/schema checks pass; GPU target-bank validation and execution are required next. Do not replace this comparison with another generator-scope change while awaiting its output. Larger-data pilots remain possible under the scale-up clarification, but none is launched here.

## F16 — The control now overfits nearly exactly; augmentation has not learned its added conditions

**Verified:** both completed coverage arms,708 payload hashes/CRCs,512 sampled outputs,384 shared rotated-label controls. Exact initial native replay; matched parameter/Adam/input/source/schedule provenance. Every target rotation reproduces locally; minimum VAE IoU99.9823%. Empty predictions are retained as zero P/R/F-score failures, not discarded.

**Positive milestone:** unaugmented broader-scope control at step3000 reaches fitted IoU99.855% with visuals and99.557% without. Both fitted states reach100% common-unit P/R for all4 object means, pass all32 individual geometry screens, and pass each-object correct-surface dependence. The model can overfit shape AND orientation under this contract with frozen VecSetX. This is a sufficient recipe at this tiny scale, not proof broader scope is necessary or that full-dataset conditioning/generalization is fixed.

**Treatment:** reserved visual-present IoU17.86→18.32% and pose-witness F-score81.84→84.40%, still far below the target reference; reserved visual-zero IoU10.50→5.75% and witness62.17→54.45%. The sampled augmented training conditions themselves reach only4.74% IoU/49.54% pose-witness F-score, with0/32 geometry passes. This short augmentation recipe is not a demonstrated remedy. Native reserved losses improve while sampled geometry remains poor; keep those endpoints separate.

**Critical limitation:** each of368 new rotated conditions receives only5–6 presentations; the control gives each original zero-visual condition125 presentations. Original-orientation zero-visual examples are replaced during the treatment continuation, not retained alongside the rotations. Their fitted IoU falls99.56→13.68% versus control. The experiment does not establish sufficient optimization of the expanded task, rotation impossibility, or VecSetX failure. Do not automatically prescribe more iterations or promote this augmentation recipe to full training.

**User's output objective:** correct intrinsic shape matters; matching GT pose is not required. The current spatial-latent training loss is orientation-specific, but that is the chosen implementation contract. Exact native frame matching is diagnostic, not an extra product acceptance requirement. Rigid-adjusted geometry already distinguishes many pose-only errors from shape failures. Reground the next intervention around shape accuracy and this distinction before selecting another run; frame canonicalization or orientation-invariant supervision remain alternatives, not implemented solutions.

Details: [ORIENTATION_COVERAGE_RETURNED_FINDINGS.md](ORIENTATION_COVERAGE_RETURNED_FINDINGS.md), `orientation_coverage_analysis/{summary,complete,learning_and_exposure}.json`.


## Active decision after F16 — establish the oracle upper bound first

The user explicitly prioritizes a concrete useful oracle endpoint before further non-oracle alignment or pose recovery. This is a change in experimental priority, not new evidence. Preserve F9's accurate reserved-view result and F6/F7's unproven new-object geometry use. No additional four-object fitting result is needed before a larger diagnostic run.

[The oracle-upper-bound handoff](oracle_upper_bound/HANDOFF.md) provides a no-update implementation replay against the successful oracle/dropout checkpoint, then matched full-data oracle/dropout, constant/dropout, camera/dropout and practical image controls. Retain fixed object-frame targets, frozen VecSetX and original full shape-cross-attention scope; broader generator adaptation is conditional on aligned fitting failure. The full-data comparison, rather than another tiny transfer test, answers whether the aligned route scales. New-source GPU parity and all full training remain pending.

Success with oracle alone establishes an aligned working setup at its measured scope. A matched worse camera arm is needed to attribute a remaining cost to frame treatment; neither proves rotation was the only cause. If oracle fails new-object utility or constant matches it, remain on the aligned branch instead of immediately investigating inference pose recovery. Output orientation remains flexible; raw-pose failure must not be mislabeled intrinsic-shape failure.

## F17 — Full-data oracle/dropout has a small training advantage, no native validation advantage

All three authorized production runs finished20 epochs/14,660 steps. Exported final visual-present validation loss: oracle .08939843, constant .08887746, camera .08876911; historical image+pointmap .08909782. Oracle is worse than constant at19/20 checkpoints, but these correlated epoch measurements are not independent replicates or a significance test. The useful full-data oracle upper bound remains **unproven**.

Late mixed training loss does differ: update-weighted epochs19–20 oracle .09580256 versus constant .09870240 and camera .09850713 (oracle2.94%/2.75% lower). This may reflect some fitting utility, not established transfer or sampled fidelity. The new runs' logged dropout sequences match, average approximately50%; surface projector/embedding and shape CA gradients are finite/nonzero at all1,485 logged steps. A completely disconnected adapter or accidentally disabled dropout is not supported by these logs.

**Next consequence:** stay on aligned conditioning. Probe existing matched last checkpoints without updates, separating training/validation identities, visual-present/zero states, and correct/wrong geometry with actual paired noise/time. Establish both condition dependence and absolute Stage-1 fidelity before calling it a working upper bound. Do not move to non-oracle pose recovery, replace VecSetX, or extend training solely because curves plateau. No conclusion of intrinsic architectural or coordinate impossibility follows. Historical manifest equality remains unverified; oracle/constant is the strongest matched full-run control.

Source and reproducible numerical audit: [full-run returned findings](oracle_upper_bound/FULL_RUN_RETURNED_FINDINGS.md), `oracle_upper_bound/full_run_wandb_analysis.json`.

**Execution status after F17 (not new evidence):** the [completed-run checkpoint probe](oracle_upper_bound/CHECKPOINT_PROBE_HANDOFF.md) is ready for the user's GPU node. It restores oracle/constant production last checkpoints at14,660; checks actual input/noise/time pairing; crosses visual-present/zero with correct/three wrong surfaces on16 train and16 validation identities, two views each; and reports native plus three fixed-time losses. Four local CPU pairing/completeness tests pass and GPU-module imports/CLI succeed locally. Full SAM3D execution is untested and pending. No new training or production-source edits. This probe does not itself establish sampled geometry fidelity; use its result to choose that verification or the relevant aligned-fitting/transfer/fusion branch. GPU results are now required before further causal conclusions.

## F18 — Aligned surfaces have held-object utility at high noise; pooled loss hides its limited scope

The completed oracle/constant checkpoint probe verifies: archive CRCs, exact raw-report summary reproduction, matching pasted JSON, all nine local/runtime source hashes, matching observation/target/time/noise hashes across arms, epoch20 / step14,660, and unchanged restored adapted weights.16 training and16 disjoint validation objects, two views each; no training or decoding.

At t=.05, oracle versus constant reduces loss by4.44%/16.02% on train objects with visuals present/zero, and1.23%/7.06% on validation objects. Oracle beats constant for11/16 held objects with visuals and13/16 without. Correct surfaces beat the mean of three wrong-object surface controls for **all16 held identities in both visual states**. This establishes some useful sample-specific conditioning beyond fitted identities at this measured noise level; it does not establish dense shape readout or successful generation.

Native held-object oracle loss remains0.76% worse with visuals, and only0.42% better without (only6/16 object means improve). At fixed t=.5 and.95 it is worse than constant. Actual native-bank time disaggregation confirms that gains near noise are offset by costs elsewhere; t=.95 itself was not sampled natively and must not be used to explain the whole W&B plateau. Visual-zero increases the relative value of surfaces but **worsens absolute oracle loss**: it is not a demonstrated rescue of harmful visual interaction.

**Next consequence:** use the positive dependence branch to test free-running Stage-1 fidelity on these same checkpoints/identities with paired noise and correct/wrong/constant surfaces. Do not declare oracle solved, discard VecSetX, prescribe timestep reweighting or move to non-oracle pose recovery. The current evidence rules against complete surface neglect, while accurate generated geometry remains unresolved. No new camera-frame cost is identified by an oracle/constant-only comparison.

Source: [checkpoint probe returned findings](oracle_upper_bound/CHECKPOINT_PROBE_RETURNED_FINDINGS.md), `oracle_upper_bound/returned_20260913_133239/{verification,summary,time_disaggregation}.json`.

**Execution status after F18 (not new evidence):** [Stage-1 checkpoint rollouts](oracle_upper_bound/CHECKPOINT_ROLLOUT_HANDOFF.md) are ready. Exact prior checkpoints/observations/tokens replay, pure-noise25-step CFG0 generation, correct/one fixed wrong oracle surface versus saved constant bank, visual-present/zero, two draws. Save latents/supports for independent fidelity and pose analysis. Four CPU sampling/metric/completeness tests and module import/CLI checks pass; SAM3D GPU execution is pending. No training, source integration or Stage2. Do not repeat the loss-only probe or infer full-data oracle success before these fidelity results.


## F19 — Real surfaces improve full-data generation, but the accurate oracle upper bound remains unmet

**Verified scope:** final epoch20/step14660 oracle and constant checkpoints;16 train and16 validation objects, two views/two paired draws, visuals present/zero, correct/fixed-wrong/constant surfaces.768 noise-only Stage-1 outputs at25 Euler steps/CFG0. Archive integrity,224 payload hashes,11 local/runtime source hashes, summary reproduction and all three pasted/archive JSON equalities pass. Stage2 is not involved.

With visuals present, mean object two-voxel F-score is **75.87% oracle vs63.72% constant on training objects**, and **64.51% vs49.69% on validation objects**. Oracle beats constant on13/16 train and15/16 validation identities. Wrong-surface oracle falls to38.77%/42.19%. Without visuals, correct oracle is71.78%/50.77%, constant14.94%/14.47%. This establishes useful sample-specific conditioning in generated shapes at this measured scope. It supersedes the uncertainty in F18 about whether high-noise utility survives generation. It does not establish superiority over an unprobed image checkpoint or attribute a camera/oracle difference.

**Absolute limitation:** raw oracle IoU31.52% train/19.25% validation is far from the target-support reference. Proper-rigid search improves correct-oracle F-score to84.52% train/77.23% validation with visuals; only2/16 object means in either cell reach both95% precision and recall. A poor registration witness is not an all-rigid impossibility proof. Separately, a conservative diameter/projection bound certifies one validation prediction cannot exceed81.89% target recall under any rotation/translation at two-voxel tolerance. This proves at least one error beyond rigid pose (shape/extent; scale correction not excluded), not a universal architecture limitation. Visual-zero reduces absolute oracle accuracy despite increasing its advantage over constant.

**Matched pose audit completed:** all768 raw support metrics reproduce; four known-rigid registration controls pass. After the same pose search, correct oracle versus constant is84.52% vs79.14% on train and77.23% vs68.10% on validation with visuals (11/16 and14/16 object means positive). Raw advantage partly reflects pose, but a measured advantage remains after common rigid adjustment. This is a registration-witness comparison, not a globally optimized pose-invariant metric.

**Next consequence:** full training failed the accurate upper bound, not all geometry use. First cheaply distinguish coarse numerical sampling from an inadequately learned aligned generator using fixed25/50/100-step comparisons; if outputs remain inaccurate, compare equal extra exposure under current versus broader shape-generator adaptation. Do not jump to inference pose recovery, a new encoder, a second full training run, or a claim that geometry is ignored. Full dataset training still froze most generator weights. Tiny fitting received far more presentations per view; more optimization remains plausible but unproven.

Source: [full-data rollout findings and decision tree](oracle_upper_bound/CHECKPOINT_ROLLOUT_RETURNED_FINDINGS.md), `oracle_upper_bound/rollouts_returned_20260913_143515/`, `oracle_upper_bound/rollouts_geometry_20260913_143515/`.

## F20 — Local full-surface normalization is nearly recoverable; use the existing full-data frame control next

Completed CPU audit on all64 locally available full surfaces from32 identities, before FPS. Actual production normalization method replayed on CPU float32, independently checked against float64; maximum discrepancy1.664e-7 and normalization-inverse error3.833e-8 object units. Recentring the normalized cloud and setting its maximum bbox extent to1 recovers the known dataset convention without target geometry: median per-sample RMS displacement0.0332 voxels, maximum RMS0.7177, largest individual displacement0.8438 voxels. This is original-cloud displacement, not a mesh reconstruction endpoint or a statement about neural accessibility.

Same-object point-ID-matched views remain close (oracle xyz max3.577e-7; normalized xyz max1.014e-6). Only three latest-rollout identities overlap these local files and none of the exact latest views do. For those three identities, local and returned target latents match exactly; all points in six historical clouds, before/after canonicalization, are within one voxel of the returned target support. Raw meshes are unavailable, so this does not certify whole-bank mesh provenance, every current record, or CUDA/FPS behavior. An incorrect orientation could still satisfy the normalization algebra.

**Narrowed:** large irrecoverable center/scale loss is not supported for these full surfaces. Efficient extraction of the deterministic relationship from VecSetX features remains open. The proposed q/2 targets would change object scale by factors0.581–1.003 (median0.958): target-prior/scale effects would complicate a new training comparison. Do not run that treatment merely to declare coordinate closure.

**Recovered evidence to retain:** the completed24-object direct readout already achieved96.57% fitted target IoU using processed frozen VecSetX features, with poor8-object transfer. Thus substantial VecSetX-to-SAM translation has already been demonstrated beyond four-object flow memorization; neither universal incompatibility nor a reusable conditioner follows. See [continued readout](FEATURE_READOUT_CONTINUED_FINDINGS.md).

**Next decision:** assess the already-trained full-data camera checkpoint on the identical oracle/constant observation/noise bank, correct/wrong surface and visual-present/zero states, preserving raw and pose-adjusted shapes plus loss disaggregation. Export actual coordinates/normalization and independently checked target-frame references to close only missing exact-bank provenance. A camera/oracle difference measures the full frame treatment, not pure rotation independent of normalization; one seed and conditional visual interactions limit causal interpretation. No new training, target change, metadata branch or production edit is selected. This comparison is proposed, not run on the laptop.

Sources: [reassessment](coordinate_reassessment_20260913/REASSESSMENT.md), [CPU audit](coordinate_reassessment_20260913/normalization_audit.json), [reproduction script](coordinate_reassessment_20260913/audit_normalization.py).

**Execution status after F20 (not new evidence):** `coordinate_reassessment_20260913/frame_probe_gpu.py` now implements the camera checkpoint extension. First,32 normalized source meshes regenerate existing target latents and provide closest-surface checks for64 actual observations; geometry/reference checks precede generator inference. Then the restored epoch20 camera model supplies paired native/fixed-time losses and512 noise-only sampled outputs. Actual camera/oracle pre-encoder arrays are saved; oracle point hashes replay F18, and visuals/targets/noise/decoder match F18/F19. The wrapper bundles logs/partial outputs even on failure. Thirteen CPU tests and driver import/CLI plus syntax checks pass. CUDA/model/mesh execution has not run locally. [Exact commands, gates and decisions](coordinate_reassessment_20260913/FRAME_PROBE_HANDOFF.md). No new training, production changes or claim of resolved coordinates follows until returned evidence is verified.

## F21 — Exact-bank targets regenerate exactly; seeded mesh correspondence contradicts the failed nearest-query gate

The first full-frame probe completed32 source-mesh target encodings and64 observations before stopping at its reference gate. All32 regenerated training latents are bitwise equal. Independently seeded mesh samples, indexed by stored point IDs, match all64 float64-recovered oracle clouds: maximum component error1.21145e-7, Euclidean error1.22583e-7 object units. All96 payload hashes, pasted/archive JSON equality and numerical replay pass. This substantially strengthens F20's limited local scope and argues against raw oracle rotation/translation/scale errors on the exact bank, before VecSetX normalization.

Eleven observations failed an Open3D nearest-point gate, maximum distance6.43138e-4 against limit5e-5. Float64 arithmetic on the selected triangles does not repair every discrepancy; the exact query failure mechanism remains unknown. The independently seeded correspondence contradicts treating this distance as proof of misaligned input. Original sampled face IDs were not exported, so a final independent source-face witness check is still pending on the cluster.

**Current next consequence (supersedes the proposed source-face recheck):** accept the independently audited exact target/seeded mesh correspondence as the scoped raw reference. Reuse only the pinned170408 report and unchanged source/data/artifacts; preserve failed nearest-query diagnostics without relabeling them as passed. Do not repeat source-mesh membership or target encoding as a prerequisite. Proceed directly to the pending camera/oracle comparison and actual VecSetX-input replay. The nearest-query discrepancy mechanism remains unresolved but is not evidence of a raw alignment error. No new training, target convention, production edit, coordinate-resolution claim, or representation detour follows. No camera generator results were produced by this failed run.

Source: [reference return and correction](coordinate_reassessment_20260913/FRAME_REFERENCE_RETURNED_FINDINGS.md), `outputs/conditioning_investigation/full_frame_probe_20260913_170408/local_reference_audit.json`. New and returned ZIPs belong under `outputs/conditioning_investigation/`.

## F22 — Full-trained oracle improves held-object geometry beyond the pose benefit found by common rigid search

The completed camera extension contributes512 sampled Stage-1 outputs on the exact F18/F19 bank. All256 payload hashes, both pasted JSONs, raw-summary reproduction, prior/runtime sources, and actual64-view normalization arrays pass. Production oracle points differ from independent reference by at most1.59441e-7; normalization errors are at most1.67201e-7 camera/1.42144e-7 oracle. The cached mesh reference was reused; no raw-mesh audit or target encoding was repeated.

With visuals present, raw train camera/oracle F2v is64.14%/75.87%; validation51.02%/64.51%. After the **unchanged common proper-rigid search**, train81.17%/84.52%, validation67.83%/77.23%. Oracle-camera difference remains3.35pp train and9.41pp held objects, positive11/16 and13/16 identities. This selects the planned frame-benefit-beyond-measured-pose branch. It is a one-seed, repeatedly used development bank with approximate pose search, not a global shape-invariance or causal pure-rotation proof.

Held visual-present camera correct/wrong adjusted scores67.83%/65.27%, versus oracle77.23%/59.15%; the trained constant control scores68.10%. Thus useful transferred surface conditioning is substantially stronger under oracle treatment, while camera has essentially no mean benefit over constant in this cell. Neither result means camera never uses geometry. With visuals zeroed, adjusted camera/oracle scores46.23%/79.01% train and35.98%/56.67% held: the frame effect grows, but absolute oracle accuracy worsens without visuals, so visual removal is not a demonstrated rescue.

The accurate oracle upper bound remains unmet: only2/16 train and2/16 held object means reach both95% precision and recall even after adjustment. Ordinary raw-transform/numerical-preparation checks now have a scoped stopping point; remaining aligned mapping difficulty is not diagnosed as an axis bug. Keep oracle fixed and focus the next intervention on the residual learning problem, controlling training exposure and geometry-specific utility. Do not promise inference pose recovery alone, repeat mesh checks, or declare VecSetX intrinsically incompatible. No new training is launched by this finding.

Storage correction: the144MiB return includes49.5MiB repeated references and57.8MiB predicted latents; predicted occupancy payloads needed here are only1.4MiB. Future transfers should be analysis-specific, with references reused by hash and no default whole-folder ZIP. Read-only local inventory finds817MiB of byte-identical old occupancy duplicates; cluster checkpoint usage remains unmeasured.

Source: [completed frame comparison](coordinate_reassessment_20260913/FULL_FRAME_RETURNED_FINDINGS.md); `outputs/conditioning_investigation/full_frame_probe_20260913_172925/local_analysis/{verification,summary,storage_audit}.json`. The local registration analysis saves JSON/JSONL only.

## F23 — Pointmaps are SSI-normalized; the historical no-pointmap surface run did not use oracle coordinates

The archived pipeline YAML has SHA256 `53c3d226b21df85c0bb3d16e6e4fa63abde0d6167525765eb929d02bfa9d358c`, exactly matching F22's executed pipeline. It sets `normalize_pointmap: true`, `ObjectCentricSSI`, `use_scene_scale: true`. This differs from VecSetX bbox-center/max-radius normalization; the pointmap is not simply unnormalized metric input. Current oracle mode bypasses pointmap SSI for surface preparation and retains object-frame raw points before VecSetX normalization.

For the no-position camera surface path, an initial isotropic scale/translation is canceled by subsequent VecSetX centering/radius normalization. CPU replay on all64 saved F22 observations gives max difference5.511e-7 between normalized raw camera points and normalized SSI points (5.295e-7 against actual encoder inputs). Removing only that preliminary SSI step is therefore not a substantive surface-feature intervention in this setting. This does not cover removing the pointmap condition tokens, disabling VecSetX normalization, or the position-enabled path. Reproduction values: `outputs/conditioning_investigation/full_frame_probe_20260913_172925/local_analysis/ssi_cancellation.json`.

Historical full-surface no-pointmap run `nouqb3mh` finished14660 steps with `no_pointmap:true`, `oracle_point_frame:false`, `use_position:false`, frozen VecSetX/full shape cross-attention; final validation loss.08965278. It supports the user's observation that removing pointmap did not fix the aggregate loss, but it is not an oracle/no-pointmap ablation. It cannot exclude residual camera-to-object orientation burden, and pooled loss alone does not establish equal sampled fidelity (F18/F22). Earlier frame-factor normalization arms altered the camera-vs-object normalization basis, not normalization-on versus normalization-off.

The native coordinate-query decoder is established code, but using its intermediate features as a SAM conditioner would be a new unvalidated integration. Do not promote that proposal to the next fix on the strength of general precedent. If further normalization diagnosis is selected, prefer a small ablation of existing preprocessing with oracle fixed, pointmap availability controlled, and encoder reconstruction fidelity checked before interpreting an unsupported-input negative result. This is not authorization or implementation of a new conditioner or full training run.

### Implementation status — approved production integration (not a new experimental finding)

The user authorized one minimal source integration and requested full shape cross-attention as the new CLI default, with shorter job names. Production train.py now supports per-sample visual dropout and a fixed real-training-surface feature control; evaluate.py restores the bank consistently. Historical missing-scope checkpoint metadata still means KV. New jobs are `stage1_full_surface_oracle_dropout.sh`, `stage1_full_surface_oracle_constant_dropout.sh` and `stage1_full_surface_dropout.sh`, each four GPUs/global batch16/20 epochs with the existing resource template. Visual baselines xun3al7m and fl7b2znc are reused subject to manifest/example comparability.

The dropout granularity follows SAM3D modality dropout (per sample); the earlier successful tiny run used whole-batch dropout. No claim of exact training-policy replication follows. Ten real CPU PyTorch tests, including two-rank DDP and old-checkpoint compatibility, passed before handoff. The subsequent GPU/full-data runs and probes are recorded in F17–F19; an accurate full-data upper bound remains unmet. [Source integration handoff](oracle_upper_bound/SOURCE_INTEGRATION.md) supersedes the standalone trainer and old preflight. Further experimental edits return to the investigation folder.


## Historical planning preamble — superseded by the evidence reset

Preserved for provenance only. The proposed metadata, normalized-target and checkpoint experiments below must not be interpreted as active work. Completed results F21–F23 and the evidence reset govern the current state.


Last reconciled: 13 September 2026. This is the short, governing evidence ledger requested by the user. Read it before proposing, implementing or interpreting the next experiment. Update it when evidence materially changes a claim; retain its scope and any counterevidence. Detailed findings and raw reports remain linked below. This is a selected subset, not a chronological experiment log.

**Objective:** make Stage-1 full-surface point conditioning work, while retaining VecSetX and a future path to sparse structured touch. Coordinate handling is the current focus. Better fitting, corrected input coordinates, robustness across views and useful conditioning on new objects are separate accomplishments.

**Current boundary (after F20 reassessment):** complete the existing full-data camera/oracle comparison, with exact-bank coordinate evidence exported alongside it, before selecting another training intervention. Oracle/constant already establishes useful geometry but does not measure frame-treatment cost. A new local64-surface normalization audit finds strong recovery of the target normalization from explicit full-surface geometry; strict normalized targets could instead substantially change the pretrained target distribution. Metadata and strict shared-target fits are parked, not queued. Read [the reassessment and counterfactual decision table](coordinate_reassessment_20260913/REASSESSMENT.md). The [one-GPU checkpoint/coordinate handoff](coordinate_reassessment_20260913/FRAME_PROBE_HANDOFF.md) is now implemented and locally checked; cluster execution is required next. Do not repeat completed rotation factors or four-object fits.

**Communication requirement:** before every new experiment, explain the unresolved question, why existing evidence leaves it open, exactly what changes/stays fixed, the controls and target reference, and the next decision under each plausible outcome. Commands alone are insufficient. The user explicitly wants to understand and challenge the experimental logic.

**Coordinate-closure clarification (after F19 discussion):** the user's immediate priority is to distinguish a correct oracle coordinate mapping from unsuccessful learning, before moving to sampler/scope work. No new GPU handoff is selected. A failed oracle reconstruction does not falsify the supplied transform. F1/F3 already support removal of camera-to-object rotation on audited examples; full-run source/token replay establishes reproducibility, not independent geometric correctness of every record. First reuse those checks and identify only missing evidence connecting actual full-run pre-encoder points, normalization and independently generated target-frame geometry. A self-inverse roundtrip or repeated use of the same transform is insufficient independent validation. Passing that finite-data contract audit closes the axis/pose/normalization bookkeeping question at its audited scope, without requiring a successful generator. F19's generation failure then demonstrates a residual learned-mapping problem, not necessarily its cause.

The production oracle applies `object_from_camera` before `TouchEncoder.prepare_points`. The latter subtracts the observed bbox midpoint and divides by its maximum radius. Its units differ from SAM3D's target-cube units; `--no-touch-position` omits normalization metadata. This known difference must be accounted for explicitly when comparing geometry, not mistaken for an axis error or newly discovered bug. Full surfaces with a prescribed target normalization can retain a recoverable relationship despite this change; finite sampling and sparse patches require separate treatment. Same physical axes do not make VecSetX feature tokens equal to SAM3D spatial latent cells. Cross-representation spatial binding and visual/geometry frame interaction remain possible learned-interface issues; F19 cannot identify either mechanism. Zeroing visuals only at inference does not isolate their effect during training. No finite set of negative fits proves all coordinate-aware architectures impossible.

**Scale/translation follow-up:** oracle point encoding is `q=(p-c)/r`, while the target mesh is centered with maximum bbox extent1 and voxelized in the fixed[-.5,.5] cube. The pointmap remains in its preprocessed camera convention; oracle does not transform it into object axes. For an ideal complete surface matching the target mesh, `c=0` and `p=q/max_bbox_extent(q)` recovers the target convention without separate metadata. This algebra does not prove that VecSetX features expose that relation easily, and finite sampling, point selection, mesh association or outlier/clipping differences must be measured. Earlier camera/object normalization factorial tested view-dependent normalization, not whether explicitly supplying the remaining VecSetX-to-target normalization assists the full-data aligned model.

**Conditional coordinate experiment order:** (1) reuse existing raw geometry to compare actual full-run oracle points against independently generated target-frame geometry, then measure post-normalization reconstruction using the known inverse and using full-surface canonicalization alone. Inspect before and after point selection. Compare against sampling/voxelization reference error, not zero mesh-versus-voxel distance. If data needed for those pairings is absent, request only that extraction; no full fit. (2) If normalization is a plausible burden, use unchanged frozen VecSetX inputs/features and a zero-initialized side branch supplying the measured center/log-radius (or existing log-inverse-radius convention), with a matched same-architecture constant-metadata control. Match continuation exposure and initial function; assess training/held identities and correct/wrong surfaces. A gain alone does not establish faithful reconstruction, and a negative at one budget does not rule out every coordinate-aware interface. (3) Only if visual-coordinate interaction remains material, test pointmap inclusion during matched training while keeping RGB and verified surface treatment fixed. A pointmap-removal benefit identifies pointmap involvement, not specifically a frame conflict; a frame-specific control would still be required. Do not canonicalize the pointmap before its pretrained embedder without accounting for the changed input distribution. This is a plan, not implemented jobs.

**Subsequent review of that proposal:** the user reports prior unsuccessful position-encoding attempts and requests deterministic matching rather than more transformation metadata left for the model to interpret. The metadata arm above is withdrawn as the next recommendation; its negative/positive logic remains historical, not queued work. First complete only missing actual-path coordinate checks. A stricter candidate is an oracle surface-only matched fitting comparison: original targets versus targets constructed with the exact input VecSetX center/radius, with one fixed global grid-unit conversion. Keep oracle input features identical, remove all visual inputs during both fits, and compare equal initialization/exposure/scope. Regenerate labels from meshes, never interpolate spatial-latent channels; independently verify no clipping and accurate target VAE roundtrips in common physical units before training. This removes sample-dependent input-to-target rotation/translation/scale and visual-frame interaction by construction. It does not spatially index VecSetX tokens or establish a learned representation interface, and changing targets changes the pretrained task distribution. A failed finite fit is not architecture-impossibility proof. The older shared-camera-orientation experiment independently bbox-normalized labels and radius-normalized VecSetX inputs, so it was not this strict normalization comparison. No new script, training budget or GPU handoff is authorized by this design note alone; explain the exact comparison before implementation. Surface-only is diagnostic, not a proposed sparse-touch production recipe.
