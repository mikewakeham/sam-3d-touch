# Coordinate-system closure argument

**Start with [the concise decision argument](DECISION.md).** This report supplies the detailed question checklist and controls; the linked page distinguishes evidence needed for a scoped exclusion from evidence needed for a dataset-wide claim.

## Decision in plain terms

**Stop repeating the camera-inverse/sign/axis probes. Do not yet claim that all coordinate-related explanations are excluded.** The existing evidence supports correct oracle arithmetic on the audited records and useful surface conditioning in the full-trained model. It does not establish a successful full-surface upper bound, nor prove that normalization, visual fusion, or pretrained coordinate preferences are harmless.

The remaining work is finite once the claim is specified. This report distinguishes **a wrong coordinate transform**, **information removed by normalization**, **difficulty learning between valid conventions**, and **output pose/scale affecting the measurement**. These are different hypotheses with different stopping criteria. A failed optimizer run must not reopen all four.

No new training, inference, dataset mutation, source change, or experiment command is requested by this report. The latest no-pointmap failure is user-reported; its checkpoint metrics are not available in the local exports. The surface-only job exists but has no returned result here. These are not marked as passed or failed experiments.

## Material Passport

- Scope: Stage-1 full-surface oracle upper bound, retaining VecSetX; sparse touch and deployment pose recovery are separate future requirements.
- Verification status: **ANALYZED; existing artifacts locally rechecked**, not independently GPU-reproduced.
- Source snapshot: commit `1c5208bef5c2eed91113d115eee159bde1407835`; source hashes and raw report hashes in [evidence_recheck.json](../../../../coordinate_system_provenance/prior_evidence_recheck.json).
- Evidence inventory: [experiment ledger](../EXPERIMENT_HISTORY.md). Critical challenges and successive revisions: [review log](../../../../coordinate_system_provenance/AUDIT_REVIEW_LOG.md).
- Population: saved full-surface manifest has 928 objects / 14,701 views. Returned production split counts are 739 training objects / 11,713 views and 95 validation objects / 1,499 views. The full-frame diagnostic used only 16 objects from each split, two views and two draws. The laptop has 64 full-surface/camera observations and no source meshes; it cannot certify every cluster record.

## 1. The complete chain we are trying to diagnose

For one observation, the executed oracle path is:

```text
source mesh --saved object transform--> centered object mesh M
  |                                      |
  | sample 8192 surface points P          | voxelize fixed [-.5,.5]^3 grid
  |                                      | frozen SS encoder
  | camera transform A                   v
  v                                 fixed spatial target Y
stored camera points C = A(P)             |
  |                                      |
  | inverse A, loaded for this sample     |
  v                                      |
oracle points P_hat                      |
  | bbox-center c, maximum radius r       |
  v                                      |
Q = (P_hat - c) / r                       |
  | frozen VecSetX encode                |
  | trainable projection                 |
  v                                      |
surface context + visual context --> adapted Stage-1 generator
                                         |
                                   predicted spatial latent
                                         |
                                   frozen Stage-1 decoder
                                         |
                              target-referenced shape measurement
```

`A = diag(-1,-1,1,1) @ T_camera_from_object`. The loader uses the full matrix inverse; it does not assume the linear block is perfectly orthogonal. The oracle does not use pointmap SSI for surface preparation. VecSetX centers/scales afterward, so **oracle points before VecSetX normalization share target coordinates; VecSetX's actual input does not share the target's numerical units exactly**.

The target is `8×16×16×16`, flattened with spatial indices preserved to `4096×8`. It is view-independent, not rotation-invariant. The eight channels are not XYZ vectors. Rotating their channels is not a coordinate correction.

Likewise, radius-normalizing XYZ is different from standardizing latent feature values. The inspected target builder stores the SS encoder mean; the loader preserves it through a transpose/reshape, and native Stage-1 decoding reverses that reshape before the SS decoder. The audited path has no extra geometric transform hidden in this flattening. This does not certify every alternate checkpoint/configuration or establish that its feature distribution is optimal for the pretrained generator.

The selected VecSetX uses learned query slots. Its 1,024 context tokens are not declared XYZ grid cells. The local integration appends projected features to cross-attention context; it does not pair token index i with SAM voxel index i. Therefore different token counts/indexing are not an uncorrected geometric transform. Their learned interpretation is still a possible bottleneck.

Sources: [loader](../../../dataloader.py), [training preparation](../../../train.py), [surface encoder](../../../sam3d_objects/model/backbone/dit/embedder/touch.py), [VecSetX](../../../sam3d_objects/model/backbone/dit/embedder/vecsetx/autoencoder.py), [target builder](../../../data_generation/objaverse-dexonomy/generate_target_latents.py), [surface builder](../../../data_generation/objaverse-dexonomy/sample_full_surface.py). Paths here are relative to this report.

## 2. Question tree and actual closure status

Every row is a separate claim. “Scoped pass” means the tested records/code pass, not every dataset record or every coordinate convention.

| ID | Question and possible failure | Existing evidence | Status and what would close it |
|---|---|---|---|
| C0 | Are we diagnosing failed learning, failed generation, or only a similar loss curve? | Full-trained oracle has useful geometry but inaccurate Stage-1 rollouts on the 16+16 bank. Latest no-PM claim has no imported metrics. | **Established for the old oracle checkpoint, unverified for new runs.** Identify checkpoint, split, inputs and endpoint before assigning a failure. No new training needed to establish an existing checkpoint's result. |
| C1 | Are surface, camera, image and target associated with the same object/view and current assets? | Manifest checks, sample hashes and reference bank agree. Full dataset is not locally audited; object-disjoint IDs do not exclude duplicate meshes. | **Scoped pass / coverage gap.** For a full-dataset claim, audit all relevant records or explicitly restrict the claim. Independently check mesh/transform/target identities, not merely a self-consistent manifest. |
| C2 | Is camera inversion wrong: axes, sign, transpose, translation, handedness, scale/shear or double conversion? | Independent seeded source-sample replay and actual GPU-prepared oracle arrays agree at about 1.6e-7 object units. Stock/source axis conversion agrees. | **Scoped pass.** Do not rerun the same bank. Extend only to unaudited records, a changed source/config, or a concrete failing record. Full inverse versus rigid transpose matters if camera matrices are slightly non-rigid. |
| C3 | Is target indexing/frame/encoding wrong or different from the source geometry? | 32 exact-bank targets regenerate exactly; earlier source-axis and decoder checks agree. | **Scoped pass for label construction.** Regeneration through the same builder alone is not an independent proof of stock convention. Retain source-axis and physical grid/decoder checks as separate witnesses. This does not close pretrained semantic orientation preference (C8). |
| C4a | Does normalization itself have a numerical error, clipping, invalid-mask, FPS, anisotropic scaling or stochastic-frame bug? | Actual 64-observation encoder arrays reproduce; all 8192 points retained; maximum numeric discrepancy 1.42e-7. | **Scoped pass for this no-position/full-surface path.** Different point counts, sparse masks, remappers, clipping and source versions need their own contract, not borrowed clearance. |
| C4b | Does discarding c/r remove target-relevant position or scale? Can finite sampling omit extrema? | Ideal complete canonical surface is recoverable algebraically. Actual sample-based canonical recovery differs by up to 0.6101 target voxel on this bank. | **Partial.** No proof that unsampled extrema are recoverable from finite Q, or that the observed discrepancy is harmless to the network. Broader per-record bounds close geometric magnitude only; a model-level normalization claim needs a controlled intervention. |
| C5 | Does differing pointmap/surface frame cause harmful fusion during training? | Pointmap is SSI camera-frame; oracle surface is object-frame with its own normalization. Old no-PM full run was camera-frame. New oracle/no-PM results not imported. | **Open.** Verified poor reconstruction without PM excludes PM conflict as a necessary/sole cause of that residual. A matched training comparison tests contribution; a shared-frame PM intervention tests the usefulness of that complete preprocessing treatment. None alone proves scale/translation specifically. |
| C6 | Does image/mask perspective conflict with oracle geometry, or cause reliance that obstructs surface learning? | Four-object view/dropout studies; full oracle is worse with visuals removed only at inference. | **Open at full training scale.** Inference removal does not erase effects of mixed training. Persistent no-visual training structurally excludes sample-dependent visual conflicts in that run; its performance remains unmeasured here. Constant zero tokens can still affect optimization. |
| C7 | Does VecSetX encode geometry in a way the generator cannot readily interpret, including coordinate-dependent feature changes? | Native VecSetX decoding is good but imperfect on 32 reported cases; tiny fits succeed. No analytic feature-to-SAM spatial correspondence exists in this interface. | **Entangled, not closed and not an identified transform error.** XYZ alignment cannot certify a learned feature mapping. Native decoding does not prove the shallow projector has equivalent access. This remains a valid coordinate-conditioned learning hypothesis, not grounds to declare VecSetX broken. |
| C8 | Are correct object-frame labels unfavorable to the pretrained prior's preferred orientation/size conventions? | Eight-object frozen-prior orientation screen found no useful general correction among six candidates. Shared-camera-target/augmentation studies failed transfer under limited coverage. | **Open but weakly supported.** Those negatives do not exhaust orientations, target normalization, or trained adaptation. There is no specified pretrained per-object “correct front” available to invert. A convention treatment must change surface and targets consistently and control VAE fidelity/exposure. |
| C9 | Is the residual only output pose or size, rather than intrinsic shape? | Common rigid search improves full-run scores but leaves errors. One full-run prediction has an all-rigid recall upper bound of 81.89%. | **Pose-only excluded for that example; broader scope partial.** Search failure is not a global certificate. Existing search/bound excludes scale adjustment, so it does not rule out a similarity-scale explanation. Preserve raw, rigid and any later similarity result separately. |
| C10 | Do training, checkpoint restoration, sampling or layout introduce a different frame than the audited input path? | Saved-context hooks, restored metadata and paired rollouts support old evaluated checkpoints; protected shape attention does not read layout. New no-visual mode has CPU fixtures only. | **Scoped pass / new-mode GPU gap.** Verify the actual run flags and context once per changed implementation. Poor 25-step sampling is not automatically a training impossibility. Native-loss and rollout endpoints must remain separate. |
| C11 | Can oracle alignment be obtained without GT pose? Are symmetric shapes identifiable? | Not established; this is intentionally an oracle experiment. Full surfaces in supplied object axes remove the need to estimate camera rotation. | **Deferred, not a prerequisite to the oracle diagnosis.** Without oracle, arbitrary asset orientation may be unidentifiable for symmetric geometry; that is not evidence the aligned training labels are impossible. |
| C12 | Will this normalization preserve sparse touch locations, normals and physical measurements? | N(aP+b)=N(P) removes a patch cloud's global placement if no metadata is supplied. | **Known future constraint, not tested solution.** Full-surface closure must not be reused as sparse-touch clearance. Eventually preserve locations/transforms and transform normals correctly; no sparse experiment is selected now. |

This decomposition covers the coordinate entry/exit points in the inspected execution path. It is not a promise that an arbitrary future hypothesis about a neural network has been disproved.

## 3. The construction argument—and its exact limits

Let the complete normalized target surface P have bbox center 0 and largest extent 1. Its VecSetX input is Q=P/r. Then:

`max_extent(Q)=1/r`, so `P=Q/max_extent(Q)`.

Thus, for **complete geometry in the stipulated canonical convention**, normalization does not create an unknown camera rotation, global translation, or independent scale ambiguity. Both sides retain the same orientation. A radius-based unit and a max-extent-based unit describe the same shape by a deterministic conversion.

Three qualifications prevent turning this into “coordinates solved”:

1. A finite area-weighted sample need not contain mesh extrema. For sampled points, Q loses the original sample center and radius; canonical recovery uses the *sample's* extent. The measured 0.6101-voxel maximum is only the additional affine displacement of observed points, not unsampled-surface coverage, target-latent error, or generator error.
2. The generator receives `E(Q)`, not Q. Geometric recoverability is not a theorem that E preserves it perfectly or that the current trainable layers can compute the conversion. No decoder or new conditioner is implied by this algebra.
3. A correctly specified transformation can still make learning harder because a frozen encoder/prior was trained under another distribution. This is C7/C8. Calling it “representation” does not magically dispose of its coordinate dependence.

For the no-position camera pathway, an isotropic pre-transform satisfies `N(aP+b)=N(P)` for positive a. Consequently removing the preliminary pointmap SSI operation before the existing VecSetX normalization is algebraically ineffective, subject to masks/numerics/no nonlinear remapping. The actual saved-bank replay agrees to approximately 5.6e-7. This is a specific branch we **can** close without another training run. It does not justify disabling VecSetX normalization or claiming pointmap tokens have no effect.

Source: [normalization audit and formula](../EXPERIMENT_HISTORY.md), [actual bank replay](../../../../coordinate_system_results/full_training/frame_probe/local_analysis/ssi_cancellation.json).

## 4. What the important experiments actually connect

```text
Initial poor outputs
  -> fixed-frame scores can confuse pose and shape [E1]
  -> check input/target construction [E2, E14]
       arithmetic consistent on audited bank
       -> no supported global sign/axis repair

Can the pathway fit at all? [E3]
  -> yes, four objects, camera AND oracle
  -> refutes universally dead path; does not establish reusable geometry

Does camera frame make learning harder? [E4]
  -> yes in a four-object matched factorial
  -> oracle removes that input rotation, but reserved-view loss still poor
       -> vary visual streams / train dropout [E5]
       -> fitted-identity view robustness becomes strong
       -> test new identities and constant controls [E6–E8]
       -> tiny success does not transfer; extra pathway can mimic gains

Try common camera-oriented labels / scope / coverage [E10]
  -> fitted shapes can be learned, transfer not solved
  -> neither proves orientation impossible nor establishes a production fix

Return to full-data oracle/camera/constant [E11–E14]
  -> similar pooled losses
  -> matched-time losses show conditional signal
  -> rollouts show real surface utility and oracle advantage
  -> inaccurate absolute reconstruction remains, including training objects
       -> bookkeeping pass does not explain remaining learning error
       -> C4b, C5/C6, C7/C8 and measurement limitations remain distinct

New no-PM and no-visual variants [E15]
  -> integration exists; latest outcome not verified locally
  -> cannot yet supply the missing logical premises
```

The full-run visual-present comparison, reaggregated from returned per-prediction data:

| Object split | Constant F2v raw / rigid witness | Camera raw / rigid witness | Oracle raw / rigid witness |
|---|---:|---:|---:|
| 16 training objects | 63.72 / 79.14% | 64.14 / 81.17% | 75.87 / 84.52% |
| 16 validation objects | 49.69 / 68.10% | 51.02 / 67.83% | 64.51 / 77.23% |

F2v is occupied-voxel proximity F-score at two voxels, not percent of semantic shape reconstructed. Oracle wrong-surface validation witness score is 59.15%. The comparison shows useful geometry under oracle at this scope; it does not establish superiority over a matched sampled image baseline, or near-target reconstruction. The target self-score is 100%; a two-voxel score can nevertheless conceal thin/missing detail. Native occupancy IoU and precision/recall remain necessary companions.

## 5. Evidence standard: what must be rerun, and what need not

**Do not rerun an experiment merely because it was small.** Small deterministic checks can establish exact arithmetic for a record. A single successful fit refutes a universal inability to fit. Small tests become inadequate when promoted to population effect size, robust generalization, equivalence or absence of failure.

| Intended claim | Required upgrade | Unnecessary repetition |
|---|---|---|
| All relevant dataset records are geometrically consistent | Manifest-wide non-training audit, missingness and provenance, independent witnesses for distinct generation branches. Full target re-encoding only where provenance cannot establish identity. | Repeating the same 32 target encodings and 64 inverse checks. |
| Rotation dominates error generally | Fresh identities, replicated paired training seeds, rotation/normalization factorial, geometric endpoints. | More views/noise draws of the original four identities as if they were new objects. This broad attribution is not required to keep oracle fixed. |
| Dropout solves generalization | Matched training with/without dropout at claimed scale, fresh evaluation objects, uncertainty. | Rechecking the near-identical oracle surfaces on reserved views of four trained objects. |
| No pointmap contribution of practical size | Matched existing/new no-PM checkpoint assessment, training replication and a declared equivalence margin if asserting negligible effect. | Calling similar pooled losses equivalence. A demonstrated no-PM failure alone answers only necessity/sole cause. |
| Normalization error cannot materially explain residual | Quantify normalization/coverage independently; if making a learnability claim, compare a defined normalization treatment with a matched control and adequate exposure. | Repeating floating-point arithmetic or using a sub-voxel bound as a neural robustness bound. |
| A transferable oracle upper bound works | Fresh objects outside the repeatedly consulted development bank, paired correct/multiple-wrong surfaces and matched control, absolute target-referenced geometry and training replication. | Treating held views of a seen, fixed surface as held geometry; asserting failure at 16 training objects proves full-scale impossibility. |

Future inference must average draws/views within object before uncertainty calculations. Repeated noise does not replicate training. For an effect-size claim, use paired training seeds; for a null/“no meaningful effect” claim, use a prespecified practical margin and an interval tight enough to exclude that effect. Do not adopt a margin after seeing a negative result. Exact tolerances for geometry are numerical/error-budget questions; learning tolerances are practical-accuracy questions. They are not interchangeable.

No untouched confirmation score exists in the main comparison. Selection by seeded object IDs avoids direct selection of successful examples, but repeated development on the same validation pool still permits adaptation to it. Exact mesh duplicates across object IDs have not been excluded.

## 6. Finite checklist before any further experiment

These are **conditional requirements, not a queue to execute in full**. Each must name the claim it closes. No new job is authorized by this document.

### Gate A — define and verify the failed outcome

- [ ] Import the already running/completed no-PM run's config, step and checkpoint identity; do not infer its result from the old camera/no-PM run.
- [ ] Identify whether the failure is fitted-object native loss, noise-only Stage-1 reconstruction, or new-object transfer. Record raw target-reference results; keep pose/scale-adjusted diagnostics separate.
- [ ] Confirm the real checkpoint's forced-drop policy and active surface path. CPU fixtures establish implementation intent, not the cluster run's behavior.

**Branches:** accurate training reconstruction with poor validation is a transfer problem; inaccurate training reconstruction is a fitting/generation problem; similar pooled loss alone is inconclusive. This gate reuses runs and prevents testing a failure that was never measured.

### Gate B — finish the explicit coordinate contract at the claimed scope

- [ ] State whether the claim covers the audited 32 objects or the entire 739-object training split.
- [ ] For additional records, verify input/target association, full camera inverse, physical target grid and actual normalization/masks. Report all failures and distinct data-generation versions.
- [ ] Quantify discarded normalization/finite-sampling effects separately from arithmetic correctness. A witness from raw points and target units may close geometry bookkeeping without running SAM3D.

**Branches:** a demonstrable bad transform is repaired and the same failing examples reassessed; a consistent bank closes *bookkeeping on that bank*; significant missing-extrema/placement uncertainty keeps C4b open. A label builder agreeing with itself is not the only acceptance check.

### Gate C — close only the multimodal mechanism being claimed

- [ ] If no-PM fails with a verified active surface path, record: **PM disagreement is not necessary for this residual.** Do not automatically try to align a stream that is absent.
- [ ] If a fully no-visual model also fails on a task where the matched control exhibits the relevant failure, record: **sample-dependent visual disagreement is not necessary for that residual.** No separate constant run is needed for this narrow exclusion.
- [ ] If removing a stream improves accuracy, it implicates the complete training/input policy. To call it a coordinate effect specifically, compare a coherent shared-frame treatment while controlling modality availability.

**Do not select another four-object fit as the default gate.** Those models already fit. Choose an assessment/task where the known baseline actually has the failure being explained. An easy success does not answer the harder failure. If no-visual full training already exists, assess it; do not restart it to satisfy a new plan.

### Gate D — decide explicitly whether to test the remaining convention/learning interaction

If Gates A–C pass their scoped claims, the remaining coordinate-specific learning candidates are C4b and C7/C8. They are not an unidentified extra axis transform.

Possible finite treatments, **not selected or implemented here**:

| Treatment | Exact question | Required control and interpretive limit |
|---|---|---|
| Shared-frame pointmap preprocessing | Does a coherent common XYZ convention help retain PM information? | Keep pixels/validity fixed; transform XYZ, not image grid. Match both cropped/full streams. It changes pretrained PM distribution and rotation as well as normalization; failure cannot refute all fusion fixes. Only useful if PM involvement was demonstrated or retaining PM is itself a goal. |
| Consistent normalization of the supervision frame | Does eliminating the per-object radius-versus-extent conversion ease aligned learning? | Reuse existing mesh-transform → voxelize → frozen SS-encode machinery. Explicitly define physical grid units and transform full mesh/points consistently; preserve VecSetX's expected input normalization. Control VAE fidelity, target resolution, sample extrema and exposure. Changing only pre-VecSetX SSI cancels and is not this test. This is an unvalidated target-convention ablation, not an established fix. |
| Broader generator adaptation at the current fixed oracle convention | Can the aligned mapping be learned with more of the generator trainable? | Matched current-scope continuation, same initial checkpoint, examples, optimizer continuity for shared parameters and update budget. Success shows a workable recipe at that convention; it does not prove coordinates never affected optimization. Earlier broad-scope results used another target task. |

The second treatment can use a declared fixed grid map for radius-normalized geometry (for example unit-radius geometry mapped into the SS cube by a global factor of 1/2). It must not silently feed out-of-range coordinates or rotate latent channels. Its implementation basis is the already exercised target re-encoding path, **not evidence that this new normalization choice works**. Simply removing VecSetX normalization changes its pretrained input distribution and is a weaker negative test.

That example is not yet a runnable design: a radius computed from finite sampled points can be smaller than the full mesh radius. The design must explicitly handle that difference without clipping the target, and check the actual encoder-boundary coordinates. A complete-mesh normalization witness and an independently sampled cloud separate those issues. Until those details are fixed, this candidate cannot be called an exact common-frame control or handed off as a job.

These choices separate two defensible goals: quantify remaining coordinate-convention contributions, or find a learner that works while holding a verified oracle convention fixed. There is no sound requirement to exhaust every coordinate convention before comparing training scopes. The entanglement is stated, not hidden.

## 7. Exact stopping statements

**Already justified:** “The tested camera inversion, source-axis conversion and VecSetX normalization arithmetic are consistent. Repeating those same checks is not justified without changed inputs or a counterexample.”

**Justified after Gates A–C at their stated scope:** “Poor oracle reconstruction persists in geometrically audited examples without the suspected conflicting streams. An incorrect camera inverse or a necessary visual/surface frame conflict does not explain that residual. We will hold this oracle convention fixed and investigate learnability.” This does **not** close the size of possible contributions to the original multimodal run, finite-sample normalization effects, or pretrained convention preference.

**Not currently justified:** “Coordinates are irrelevant,” “the oracle has solved conditioning,” “full finetuning is necessary,” or “VecSetX/SAM3D are inherently incompatible.” No finite set of unsuccessful normalization/rotation trials proves those statements.

**Reopening rule:** a previously scoped-pass bookkeeping branch reopens only with a failing record, a changed transform/preprocessor/checkpoint path, or a new sample regime (e.g. sparse touch). A failed loss curve alone is not a counterexample. The explicitly open convention/learning rows remain open without contaminating the bookkeeping conclusion.

The logic of the important outcomes is:

| Premises | Valid conclusion | Invalid conclusion |
|---|---|---|
| Independent alignment passes; output is poor | Correct explicit alignment is insufficient for this model's success. | Alignment must still be wrong; or all coordinate conventions are harmless. |
| PM is absent throughout a verified run; output is poor | PM conflict is not necessary for this failure. | PM never harms the original model, or its removal had no offsetting costs. |
| No-PM and PM scores are similar | The observed endpoint is similar at this budget. | Equivalence, or a zero causal effect on every component of the error. |
| Broader adaptation works with the same audited oracle inputs | That coordinate convention supports a working learned solution at the demonstrated scope. | Broader adaptation is universally necessary, or coordinates never affected optimization. |
| Both adaptation scopes fail | Both tested recipes failed. | Representation impossibility, coordinate impossibility, or a complete search of either. |

For practical progress, one need not prove all rows false before comparing trainable scopes. Holding the same audited oracle inputs fixed makes that comparison interpretable. It cannot yield a pure decomposition of architecture versus coordinate-conditioned difficulty; such an attribution would require a planned scope-by-convention comparison. That distinction is the explicit limit, not an excuse to label the coordinate branch closed.

## 8. Why this does not require starting the investigation over

Keep the numerical witnesses and the actual positive/negative results. Withdraw the oversized interpretations. A full run was defensible as a scale diagnostic after tiny fitting successes; those successes did not establish that it would work. My immediately preceding suggestion to fall back to another few-object surface-only fit omitted that the relevant tiny-fitting question had already been answered. This report supersedes that suggestion and older automatically stated “next experiment” sections.

The immediate deliverable is this decision structure. The audit has not supplied missing cluster results or manufactured a global closure certificate. Whether to stop at engineering closure or quantify the remaining convention effects is now an explicit decision with named uncertainty, rather than another unbounded search for “coordinate mismatch.”
