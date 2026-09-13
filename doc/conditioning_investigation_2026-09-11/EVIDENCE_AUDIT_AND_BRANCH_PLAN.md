# Evidence audit and coordinate investigation branches — 12 September 2026

## Material Passport

- Origin: existing SAM3D-touch investigation; experiment-planning workflow of academic-research-suite.
- Mode: retrospective claim audit and prospective experiment selection.
- Verification status: source/report review and local arithmetic/identity checks; new experiment UNEXECUTED. Returned GPU evidence is not an independent local GPU rerun.
- Scope: Stage 1, full surfaces through VecSetX; eventual structured sparse touch remains a constraint. No Stage 2, encoder replacement, full training or production changes in this audit.

## The objective and the three endpoints we must keep separate

The objective is useful surface conditioning, not recovering an asset's arbitrary coordinate labels for their own sake. The model should use observed geometry to improve reconstruction, ultimately alongside incomplete visual evidence and structured touch. Whether its output must use original asset axes remains an unresolved product requirement; the recent user question is not authorization to assume either answer.

1. **Training-objective fidelity:** native flow loss against the current spatial target and target-derived noisy states. It answers whether the present supervised task is being learned.
2. **Generated Stage-1 shape fidelity:** decoded support relative to GT, both in the required output frame and, separately, allowing only rigid pose changes. A model can fail the first metric while getting much of the shape right. This is not a Stage-2 detour.
3. **Incremental surface utility:** correct surfaces must improve the relevant endpoint over suitable alternatives on identities excluded from fitting. Improvement over image alone, sensitivity to removal, or fitted-object swaps are individually insufficient.

The initial premise “full surfaces cannot even overfit” is contradicted on four objects, including camera-frame inputs. The original broad-data behavior is not thereby diagnosed. Our strongest isolated coordinate result is a finite-budget learning penalty under changing input orientation, not an established universal pose requirement or architecture limitation.

## Audit of completed experiment families

| Family / source | What narrowed the question | Remaining gap or correction |
|---|---|---|
| Data and source audit, [F1](PIVOTAL_FINDINGS.md), [target convention](TARGET_FRAME_FINDINGS.md) | Inspected camera inverse, reprojection and import axes are consistent; target is a spatial, orientation-dependent latent. Inverse error ≤2.14e-7 over 64 views. | Selected records do not certify every sample or establish observable semantic asset orientation. Missing explicit R does not prove R is impossible to infer from observations. |
| [Original fixed-state GPU probe](GPU_FINDINGS.md) | Surface swaps change predictions but improve target-velocity accuracy little in eight original training examples. Actual loss plumbing was checked. | Checkpoints were selected at different steps; these are descriptive original-model results, not a matched causal training comparison. Wrong-input contradictions and target-derived noisy states limit the interpretation. |
| [Guidance rollouts](ROLLOUT_FINDINGS.md) | Removing CFG worsened sampled geometry even though it improved local velocity error. | Earlier guidance-failure interpretation was withdrawn. This is direct evidence against treating lower native/latent error as automatically better generated geometry. No new CFG search follows. |
| [Early geometry bundle](GEOMETRY_FINDINGS.md) | Raw measurements were independently reproduced; several original predictions had substantial shape recovery after proper axis rotations. Shield surface CD .13983→.01040; knife image .20161→.00512. Required transforms vary with object/seed. | This crucial result was missing as a standalone pivotal finding. It already warned that wrong pose and wrong shape must be separated. Selected cases and GT-assisted best alignment do not prove a global fix or prevalence. These are old checkpoints, not the latest dropout model. |
| [Single-view tiny fit](TINY_FIT_FINDINGS.md) | Camera/oracle mean target-support IoU ~99.2%; camera minimum 97.34%. Frozen VecSetX plus current projector/full shape CA can fit. | Four fixed surface codes can identify memorized shapes. This rejects complete disconnection/universal inability to fit; it does not prove detailed geometry interpretation, adequate capacity at all scales, or full-data convergence. |
| [View transfer](VIEW_TRANSFER_FINDINGS.md), [multi-view fit](MULTIPLE_VIEW_FIT_FINDINGS.md) | Other visuals hurt even with oracle surface and target fixed. Multi-view oracle training improves over camera at a matched budget. | Same-object held views test robustness, not new-shape learning. One-to-four views reduces exact-record repetitions while holding object exposure fixed. Existing production training already used multiple views. |
| [Rotation × normalization](FRAME_FACTORS_RETURNED_FINDINGS.md) | Reserved native loss: oracle .06013, rotation-only .08074, normalization-only .06427, camera .08071. Rotation is a controlled contributor to the fixed-target task. | Four identities/one seed; curves still improve; normalization interacts with rotation. Geometry comparisons were also unregistered. It does not localize difficulty to VecSetX, demonstrate inability to predict shape up to pose, or decompose the later dropout gap. |
| [Pose-input audit](POSE_INPUT_CONTRACT.md) | Ordinary inputs omit explicit camera-to-target R; near-constant translation cannot replace it; protected shape self-attention cannot read layout. | RGB/surfaces may still reveal orientation. Predicted SAM layout is relative to its generated shape, not guaranteed the saved asset frame. No actual observation/target collision establishes impossibility. |
| [Frozen target-frame probe](TARGET_FRAME_RETURNED_FINDINGS.md) | Exact target regeneration; six alternative orientations are representable. Selected yaw gives only .344% held flow improvement, without sampled corroboration. | No surface-conditioned training under a new frame was performed. This does **not** reject camera-oriented targets, continuous pose treatment, or a carefully defined pose-independent training objective. |
| [Native VecSetX](REPRESENTATION_RETURNED_FINDINGS.md) | Native surface F-score ~.965 oracle/~.970 camera on 32 objects; substantial geometry survives both encodings. | Native reconstruction does not establish feature equivariance or easy access through SAM's learned readout. Some objects are weaker; resolution and thin shapes matter. No reason to replace VecSetX follows. |
| [Direct feature readouts](FEATURE_READOUT_CONTINUED_FINDINGS.md), [mesh bridge](SPATIAL_BRIDGE_RETURNED_FINDINGS.md) | Readouts fit their training identities, fail reserved ones; complete-mesh conversion is possible diagnostically. | A differently trained readout is not a causal explanation of the production surface interface. Complete-mesh bridge bypasses the intended input contract and remains withdrawn. No continuation or replacement follows. |
| [Oracle visual-stream factorial](ORACLE_VISUAL_STREAMS_RETURNED_FINDINGS.md) | RGB/mask changes dominate the measured residual compared with pointmap-only changes. | Crossed inputs can contradict one another. Does not establish pointmaps are useless, all visual dependence is harmful, or dropout will help hidden sparse contacts. |
| [Visual dropout](VISUAL_DROPOUT_RETURNED_FINDINGS.md) and [new sampled check](ALIGNMENT_TOLERANCE_RETURNED_FINDINGS.md) | Oracle reserved native gap nearly closes; now actual reserved target-support IoU is 97.17%, minimum object-mean 91.96%. Wrong surfaces yield 2.49%. | Accurate fitted-identity endpoint, not reusable geometry interpretation. Two individual samples <90%; full-surface tokens are nearly constant per identity. Native and occupancy results must retain their own references. |
| [32-object probe](UNSEEN_OBJECTS_RETURNED_FINDINGS.md), [16-object training](OBJECT_TRANSFER_RETURNED_FINDINGS.md) | Correct surface models lose to image on the fixed-frame native objective; oracle .13420 vs image .11930 in the latter, losing on 16/16 held objects. | These flow-model transfer screens did not sample held-object geometry. “No demonstrated transferable fix” is valid; “pose-independent shape transfer is disproven” is not. Four/16-identity training does not identify the original ~full-data cause. |
| [Early/late/removal](TRANSFER_CHECKPOINTS_RETURNED_FINDINGS.md), [constant input](CONSTANT_SURFACE_RETURNED_FINDINGS.md) | Wrong surfaces retain early gains; one constant surface reproduces fitting gains. Removal gives ~.44 native loss, but that is an out-of-training-distribution intervention. | Extra pathway adaptation can explain gains; tokens are not extra parameters by definition. One constant bank does not prove zero geometry use. Removal sensitivity is not geometry utility or proof of catastrophic forgetting. |
| [Separate attention](SEPARATE_SURFACE_RETURNED_FINDINGS.md) | Exact image fallback preserved, yet held native losses worsen. Separating softmax and protecting image weights did not rescue this tested candidate. | Not a one-variable test of all fusion mechanisms, nor proof every architecture fails. Candidate rejected on its actual native-loss criterion; pose-independent sampled utility was not assessed. |
| [Residual-rotation sweep](ALIGNMENT_TOLERANCE_RETURNED_FINDINGS.md) | Accurate aligned endpoint loses fixed-frame IoU under some small residual errors; 5° passes only two of six directions. | A zero-shot perturbation of an aligned-trained model, not a trained camera-frame model or measured estimator. Includes normalization response. Rigid output movement vs shape damage unseparated. No universal “must be under 5°” conclusion. Raw 15° shard missing; aggregate is present. |

The inspected `dataloader.py`/`train.py` still load fixed spatial targets and apply oracle inversion only to points before VecSetX; `TouchEncoder.prepare_points` uses bbox midpoint/radius normalization. The factorial normalizes before constructing its mixed conditions and avoids renormalizing them. The latest sweep rotates before ordinary normalization. These are deliberately different interventions; their effects cannot be equated.

## Cross-experiment gaps to keep visible

### G1. Pose and shape were repeatedly collapsed into one outcome

The early bundle demonstrated the distinction, and the CFG result demonstrated that native loss and generated geometry can rank models differently. Both must constrain F3/F6/F8/F10. Their observed native/fixed-frame differences remain real. What changes is the breadth of the inference from those measurements.

### G2. Even strong fitted surface dependence can be identity retrieval

F9 correct/wrong reconstructions establish dependence on supplied surface identity. They do not show the model transports novel surface structure into its output. Rotation augmentation on the same four objects could make recognition more tolerant without teaching transferable reconstruction. Any such experiment would repair a local robustness property only, unless followed by an identity-disjoint test.

### G3. The apparent chain of held-object confirmation is not independent replication

Direct set comparison during this audit: all 32 identities from the earlier unseen-object probe are exactly the later 16 training +16 held identities. They remain excluded from optimization where documented; there is no newly found training leakage. However, repeated results on them informed subsequent choices. They are a development pool, not an untouched final confirmation set. Multiple views/noise draws are repeated measurements, not more independent objects. Most decisive training comparisons have one training seed. New confirmation must use fresh identities and at least a second training seed for a claimed training-intervention gain.

### G4. Oracle alignment is useful but not an overall upper bound

It is a known-coordinate positive control for one interface/task. It removes rotation from the input, but does not guarantee a perfect output or best optimization/generalization. It also makes each fitted identity's surface code more stable, which can facilitate memorization. Better oracle view robustness and poorer new-object native loss can coexist without contradiction. A changed output-frame task can in principle outperform this particular oracle treatment; that possibility was not tested by six frozen target orientations.

### G5. Output orientation, center and scale need separate contracts

Allowing arbitrary rotation can remove an unnecessary original-asset-axis requirement. It does not automatically solve camera/robot placement, scale, sparse-patch location, or compatibility with a pretrained spatial prior. The no-position branch removes cloud shift and isotropic scale; using the full observed surface to normalize is not automatically valid for a tiny hidden patch. Preserve a shared observed frame or explicitly transmit normalization information in the eventual touch path. No full-shape GT may be used as an inference normalizer or hidden input.

### G6. Original broad-data underperformance remains only partly connected

The original checkpoints were inspected on a small selected training subset, and the user reports the full oracle run did not improve much. That report is evidence against simply repeating oracle training. It is not a matched quantitative decomposition of broad-data coordinate/shape/transfer errors. The optional original-checkpoint audit remains a later external-validity branch, not a prerequisite or a reason to restart the current sequence.

## What is actually narrowed down now?

| Question | Current answer |
|---|---|
| Is there a demonstrated simple sign/axis conversion bug? | No; inspected bookkeeping is consistent. Stop repeating global-axis searches. |
| Is the path completely broken or unable to fit with frozen VecSetX? | No; camera and oracle tiny fits contradict that. |
| Does changing camera orientation make the current fixed-target task harder? | Yes, in the controlled four-object finite-budget experiment. |
| Does exact alignment plus the tested dropout permit accurate output on unfamiliar views of fitted shapes? | Yes for Stage-1 support at the measured endpoint. Not an untouched-object result. |
| Do the latest rotation errors damage shape or primarily move it? | Unknown. The raw IoU sweep cannot answer this. |
| Do surface models improve new-object generated shape, allowing pose freedom? | Not established by the recent native-loss screens; requires actual samples. |
| Is original-asset canonicalization necessary for the user's eventual output? | Not decided. It is required by current labels, not inherently by 3D reconstruction. |
| Do we need new encoder/full finetuning or face an intrinsic impossibility? | No experiment establishes either. |

Working diagnosis: the implemented task couples surface interpretation with mapping camera-oriented observations into a fixed asset-oriented spatial latent. Removing that frame burden and reducing dependence on familiar visual views produces accurate fitted-identity outputs. It has not yet produced a demonstrated general conditioner. The missing distinctions are **output pose versus shape**, followed by **geometric transfer versus specialization**. A deployable frame choice/correction has not been tested.

## Candidate branches and why only one is selected first

| Branch | Experiment | What it distinguishes / when to activate | Cost and limitation |
|---|---|---|---|
| **A — selected now** | Reanalyze existing generated supports with controlled rigid-pose compensation; include natural camera/oracle multi-view predictions if their saved artifacts are available. | Pose-only disagreement vs non-rigid/scale error; whether artificial residual sensitivity also describes the natural camera/oracle result. | CPU analysis of existing NPZs. No training or new model inference. Does not prove deployment or new-object transfer. |
| B — target-frame intervention | Short matched training with current camera inputs, comparing original object-frame targets and explicitly camera-oriented targets; include an image baseline under each target contract. Re-encode transformed geometry correctly. | Whether fixed-asset-frame supervision, rather than information availability, prevents learning useful conditioning. Activate if A shows substantial pose disagreement or if observable-frame outputs are acceptable and the natural camera task still fails. | New target encoding and bounded GPU training; prior compatibility and normalization must be controlled. Six frozen target orientations did not test this. |
| C — local robustness repair | Resume the same aligned checkpoint with small residual input rotations versus equal additional training without them; preserve the aligned endpoint and test new mixed-axis perturbations. | Trainable local robustness vs an accuracy/robustness tradeoff. Activate if A shows real shape damage or fixed-frame precision is required. | Short GPU training. Does not recover an arbitrary camera pose, prove generalization, or explain all original failure. |
| D — observable frame recovery | Feed a measured/estimated transform from permitted inference inputs, compare to known inverse and ordinary camera input. Evaluate angular **and geometric** errors, accounting for symmetries. | Can a practical correction approach the proven reference? Activate only once output-frame requirements and shape/pose tolerance make this worthwhile. | A substantial method experiment; a pose head or native layout is not a guaranteed plug-in. Training-label R is not permitted at deployment. |
| E — surface utility on distinct shapes | Sample existing image/camera/oracle models on a fixed development cohort with correct and several wrong surfaces; use raw and pose-adjusted Stage-1 metrics. | Native-objective failure vs generated-shape failure, and geometry utility beyond fitted identities. Mandatory before a full-run claim, but not the first coordinate branch. | GPU inference if samples do not already exist. No new training. Reuse existing weights before launching another tiny fit. |
| F — conditioning/learning capacity | Once aligned, pose-adjusted **new-object** shape utility genuinely fails, compare one controlled learning change at a time (e.g. broader shape adaptation or representative data exposure), retaining real/constant input controls. | Capacity/optimization/specialization beyond coordinates. | Deferred. Current evidence does not choose full finetuning, scratch VecSetX, or a new fusion architecture. |

There is no automatic chain A→B→C→D→E→F. A's outcomes select a relevant branch; evidence from the selected branch can close it or return to alternatives. Stop completed negative candidates rather than automatically extending them.

## Selected A: a concrete design, before accessing new prediction arrays

**Question:** can the apparent rotation-induced error be explained by a rigid change of output pose, and does the answer differ between the artificial perturbations and natural camera-frame conditioning?

**Inputs and scope:** already saved occupancy NPZs from `outputs/conditioning_investigation/alignment_tolerance/manual`, paired with returned raw reports. Use all four identities, both fit/reserved splits, two seeds and all existing conditions; avoid selecting only visually convenient failures. Raw shard 2 is pending but does not block the available 0°/5°/30° evidence. The latest NPZs are not currently local. Original camera multi-view NPZs may provide the natural-input check without new sampling; if absent, mark that comparison unavailable rather than retrain or broaden the request silently. Old selected geometry-bundle results inform the hypothesis, not a pooled estimate with the new samples.

**Analysis sequence:**

1. Verify files against report IDs/conditions and reproduce raw occupied counts/IoU. Target arrays and paired shapes must agree. Preserve every failed/empty prediction in coverage and results.
2. Report raw output-frame agreement. Apply the **known inverse perturbation** to predicted voxel centers as an analytic test of outputs rotating with their point inputs. This is a tested hypothesis, not an assumption that output rotation equals input rotation. Keep the target fixed.
3. Separately estimate a proper rigid transform from output to target using multiple initial rotations and local refinement. No reflections, anisotropic scaling or non-rigid warps in the primary test. Record transform, objective, convergence and disagreement among starts. A good rigid match is a constructive explanation; a poor optimizer result alone is not proof that no such match exists.
4. Report bidirectional geometric precision/recall at one and two voxel spacings (1/64 and 2/64), distance distributions including tail errors, and direct visualization. Raw IoU remains unchanged as the original metric. Do not compare resampled/registered voxel IoU naively with original-grid IoU: rotation changes the voxel lattice. For the pose-independent comparison, apply the same method to the accurate aligned prediction too.
5. Calibrate on synthetic known rotations/translations of actual GT and aligned-prediction supports, and on wrong-object support pairs. This reveals discretization/registration error and whether permissive matching can make a wrong shape appear successful. Hold transformed-support discretization separate from network error. If necessary, diagnose isotropic scale separately and label it a different hypothesis; do not hide it inside “rigid pose.”
6. Include the existing wrong-surface predictions and, where available, the natural camera/oracle multi-view outputs. Fitting a transform independently to each recipient GT is a diagnostic, not inference performance. Consistent recovery of the known perturbation is stronger mechanism evidence than arbitrary best alignment alone. Symmetric objects require geometric, not unique-angle, interpretation.

**Decision criteria:** report paired per-object results before pooled means. A “pose accounts for the error within voxel resolution” classification requires a constructive rigid match recovering both precision and recall ≥95% at one voxel, no more than two percentage points below that object's aligned reference, with the known-transform positive controls succeeding. Also report tails/two-voxel sensitivity and any individual failures; these engineering thresholds are not perfect-shape equivalence. If wrong-object controls routinely pass, the criterion is not discriminating and cannot support the claim. Lock implementation/control conventions before reviewing new network arrays; do not tune them to make the actual predictions pass.

**Branches from A:**

- **Rigid compensation nearly recovers reference shape:** no evidence yet that this error requires repairing shape conditioning. If original asset axes are unnecessary, B's target-frame formulation or an observable output-placement strategy becomes more direct than demanding precise canonicalization. Still require E before claiming general conditioning.
- **Known inverse recovers it:** specifically supports an output following the input's rotation. A corresponding inference-frame strategy may be feasible, but GT-free output placement remains untested.
- **Best rigid fit recovers it but known inverse does not:** arbitrary output-pose drift is sufficient to explain the error; do not claim known input R gives a ready correction. Distinguish symmetry/ambiguous pose from irregular drift.
- **Substantial error remains with successful registration controls:** rotation also changes the generated shape/scale/support. C becomes a targeted local repair test; if it fails, inspect normalization and native VecSetX reconstruction under these exact perturbations before changing the encoder.
- **Artificial perturbations fail but natural camera outputs are already shape-accurate:** the sweep is a robustness property of that oracle-trained checkpoint, not necessarily the production bottleneck. Prioritize target/output-frame treatment and E rather than “fixing” every synthetic rotation.
- **Results vary strongly by object or registration is ambiguous:** retain separate cases and identify the missing control; do not collapse the outcome into “coordinates solved” or “architecture impossible.”

**Execution status:** plan selected; no new training/analysis driver launched. Existing outputs are the next material needed. CPU implementation and artifact packaging can follow this plan; no GPU allocation is required for A itself. No runtime claim until array sizes and registration implementation are measured. A failed data/replay check pauses interpretation; do not retrain to bypass it.

## What earns an overnight run

A may resolve an interpretation gap, but cannot alone justify broad training. A specific candidate from B/C/D must demonstrate the property it claims under a matched short comparison, and surface utility must be checked on separate objects with the intended output-frame metric. Use fresh confirmation identities after development on the reused pool, and another training seed before claiming a reproducible intervention gain. Retain native training metrics even if output pose is free: registering samples does not fix the existing supervised objective.

The full-surface criterion must not silently require near-perfect reconstruction on every new object before any broader run; the ≥95% fitting reference is not a universal generalization threshold. The decision is a material, consistent geometry-specific improvement over matched baselines at an affordable, representative training scope, with uncertainty and residual failure explicit. If broader data is the intervention being tested, name it as such rather than promise a validated fix.

Sparse-touch constraints persist: observed patch locations/normalization must be retained, missing space is unknown, and image information can be essential. A four-full-surface dropout success is not a reason to default to 50% visual dropout for hidden contacts. No full-mesh/GT-voxel conditioner is reintroduced.
