# Pivotal findings — living evidence and experiment decisions

Last reconciled: 12 September 2026. This is the short, governing evidence ledger requested by the user. Read it before proposing, implementing or interpreting the next experiment. Update it when evidence materially changes a claim; retain its scope and any counterevidence. Detailed findings and raw reports remain linked below. This is a selected subset, not a chronological experiment log.

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

Decoding GT latents supplies an achievable reference through the same downstream measurement path. The earlier tiny-fit experiment used this principle at Stage 1. It does not require Stage 2 or final mesh CD. Do not use target-assisted alignment of a prediction to hide the coordinate error being tested. A new Stage-1 reconstruction check is justified when needed to promote a candidate to a reconstruction claim, not automatically for every diagnostic.

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

**Next-experiment consequence:** retain this checkpoint as a measured same-identity robustness reference, not a proven general geometric conditioner. A reconstruction claim needs an actual Stage-1 target-referenced check of the relevant checkpoint.

Sources: [dropout findings](VISUAL_DROPOUT_RETURNED_FINDINGS.md), [exact aggregate](visual_dropout_returned_manual/analysis.json). Gap arithmetic rechecked from that JSON on 12 September.

## F6 — The strongest tiny-task improvements did not transfer to new objects

**References:** on 32 different objects, image loss was .10540, original oracle .21229 and dropout oracle .27079. Every surface variant lost to image on all 32. In the subsequent matched 16-object training screen, oracle held-object loss was .13420 versus image .11930, again losing on all 16; oracle improved fitted identities. Correct surfaces were not reliably preferred to wrong ones on held identities.

**Established:** alignment/dropout had not produced a transferable conditioning solution in these experiments. F2/F3/F5 remain true within their scopes; this is counterevidence to promoting them into a general fix.

**Not established:** that the original full-data models have the same cause, that more data/full finetuning is required, or that canonicalization is impossible. Training-object identification is consistent with the pattern, not directly observed internally.

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

1. **Starting evidence:** relevant F1–F8 IDs and the exact unresolved question; do not recreate completed evidence as a new discovery.
2. **Claim being tested:** coordinate correctness, fitting, view robustness, geometry-specific use, or transfer. Name any entanglement that requires changing another factor.
3. **Comparisons:** intervention, matched control, reference endpoint and claimed split. State whether the reference is perfect target agreement, exact alignment, a fitted-model score, or an image baseline.
4. **Success/failure/inconclusive criteria:** justify the tolerance or effect size before results; include uncertainty/replication appropriate to the scope. A smaller scalar alone is not “resolved.” Retrospective margins cannot convert prior results into formal equivalence.
5. **Decision branches:** what each outcome changes; distinguish a diagnosis from a deployable fix and a coordinate fix from an overall conditioning fix.

After the result, update the relevant finding, comparison and scope. Add a new finding only if it changes a material decision. Never let a later summary silently broaden a small-task claim. Do not launch new training merely to satisfy this documentation checklist.

All GPU evidence above consists of returned reports with the documented validation/replay checks. No independent local GPU rerun is claimed. Noise draws and multiple views of one identity are not independent objects or training replicas.

## Active continuation — prepared, not a finding

[Alignment-tolerance handoff](ALIGNMENT_TOLERANCE_HANDOFF.md) continues F3/F5 using existing original/dropout oracle weights. It measures actual Stage-1 target agreement and residual rotation sensitivity, with predeclared references, scope and branches. GPU results are pending. It neither repeats training nor establishes new-object utility or a deployable pose correction. F6 remains an independent obstacle to promoting any full-training candidate. No overnight full run is currently justified by the returned evidence.
