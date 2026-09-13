# Evidence reset: what the coordinate investigation established

13 September 2026. Stage 1, full surfaces. This report governs interpretation of the older experiment notes. Their historical next-action sections are not a current queue. No new experiment or production change accompanies this audit.

**Conclusion:** the oracle removes the camera-to-object transform correctly on the audited observations. It improves the fully trained model's use of real surface information. It does not produce an accurate reconstruction upper bound. The residual has not been traced to another specific coordinate error, nor to a particular representation or training limitation.

## Meaningful progression

1. **Established what alignment means.** The target is a spatial `8×16×16×16` latent in the normalized object's axes, flattened to `4096×8`. It is shared across views but is not rotation invariant. Source/target axes already contain the stock conversion; adding it again would double-rotate the object. Camera inversion, independent source-mesh correspondence and target regeneration support the existing transform. There is no supported global sign/axis patch. [Sources](TARGET_FRAME_FINDINGS.md), [target-orientation probe](TARGET_FRAME_RETURNED_FINDINGS.md).

2. **Showed that input orientation can make this task harder.** The four-object factorial separated orientation from the basis used for centering/scaling. Camera orientation reproduced most of that task's camera/oracle penalty; camera-derived normalization without the rotation did not. Oracle plus dropout subsequently reached 97.17% target occupancy IoU on reserved views of those four fitted objects. These establish a finite-task orientation effect and a working fitted-identity reference. They do not establish new-object reconstruction; the subsequent new-object loss tests did not transfer that success. [Factorial](FRAME_FACTORS_RETURNED_FINDINGS.md), [sampled reference](ALIGNMENT_TOLERANCE_RETURNED_FINDINGS.md), [counterevidence](UNSEEN_OBJECTS_RETURNED_FINDINGS.md).

3. **Learned which controls are necessary.** On the small transfer screen, a constant surface reproduced apparent fitting gains. Therefore extra-token gains and wrong-surface sensitivity alone were insufficient evidence of useful geometry. Native VecSetX reconstruction supported retaining the encoder; small direct readouts could fit but failed to transfer. These results did not identify a coordinate fault or prove representation incompatibility. [Constant control](CONSTANT_SURFACE_RETURNED_FINDINGS.md), [native representation](REPRESENTATION_RETURNED_FINDINGS.md), [readout](FEATURE_READOUT_CONTINUED_FINDINGS.md).

4. **Full training contradicted the claim that oracle conditioning had simply failed.** Camera, oracle and constant models completed the same 20-epoch training budget. Similar aggregate losses hid differences: paired noise-level probes found geometry utility near pure noise, and generation showed correct oracle surfaces beating both wrong surfaces and the trained constant control. This established useful surface information at full training scale, while reconstruction remained inaccurate. [Training](oracle_upper_bound/FULL_RUN_RETURNED_FINDINGS.md), [paired losses](oracle_upper_bound/CHECKPOINT_PROBE_RETURNED_FINDINGS.md), [generation](oracle_upper_bound/CHECKPOINT_ROLLOUT_RETURNED_FINDINGS.md).

5. **The final camera comparison measured the benefit of the oracle treatment.** Earlier oracle/constant results could not answer that question. The matched camera assessment did:

| Visuals present; Stage-1 F-score | Constant | Camera | Oracle |
|---|---:|---:|---:|
| Training objects, raw | 63.72% | 64.14% | 75.87% |
| Held objects, raw | 49.69% | 51.02% | 64.51% |
| Training objects, after common rigid search | 79.14% | 81.17% | 84.52% |
| Held objects, after common rigid search | 68.10% | 67.83% | 77.23% |

Oracle beats camera on 13/16 held objects after that search. Wrong surfaces reduce oracle's held score to 59.15%. Alignment therefore has measured utility beyond the output pose corrected by this procedure. The remaining error exists on training objects too; it is not exclusively a new-object generalization problem. [Complete comparison](coordinate_reassessment_20260913/FULL_FRAME_RETURNED_FINDINGS.md).

These are object-averaged occupied-voxel proximity F-scores at two voxels, not percentages of semantic shape recovered. The decoded target's self-score is 100%. Scope: one training seed per model, 16 train/16 held identities, two views/two draws, 25 steps and CFG0. The held bank is development evidence, not untouched confirmation. Rigid search is approximate, excludes scale, and is not a global optimum. Camera/oracle also have separately learned weights; their difference is the whole frame treatment, not a pure rotation-only estimate. Historical image baselines were not sampled in this comparison.

## Exactly where coordinate uncertainty stands

| Coordinate question | Supported conclusion |
|---|---|
| Is camera-to-object rotation actually removed? | Yes on the audited inputs, before encoding. Later centering/isotropic scaling cannot reintroduce a camera rotation. Actual oracle points match the independent reference within `1.6e-7` object units. |
| Are the target labels in the intended source-object frame? | All 32 targets in the final reference bank regenerate bitwise exactly from the source meshes. Earlier stock-axis checks agree. This is not an audit of every dataset record. |
| Are VecSetX normalization and its inverse implemented correctly? | Saved actual encoder arrays reproduce within `1.5e-7` for oracle; inversion error is below `4.7e-8` object units. All 8192 points are retained in this bank. |
| Does VecSetX normalization discard indispensable full-object scale/position? | Not in the ideal complete-surface case with this target convention; see the algebra below. Finite samples give an approximation: maximum observed point displacement after geometry-only canonical recovery was 0.61 target voxel in the final bank. This is not a learned-feature accuracy guarantee. |
| Are pointmap and surface coordinates numerically identical? | No. Pointmap uses camera-frame SSI normalization; oracle surface uses object-frame bbox/radius normalization. That known difference is not evidence that either transform is wrong. A harmful training interaction between these conventions has not been established or excluded at full scale. |

For the complete target surface `P`, bbox center is zero and maximum bbox extent is one. VecSetX supplies `Q=P/r`, so `max_extent(Q)=1/r` and **`P=Q/max_extent(Q)`**. Hence no unknown rotation, translation or scale is required to recover that canonical geometry from the ideal normalized full surface. Finite samples need not reach the mesh extrema, which explains the limited scope of the empirical recovery check. This argument does **not** apply to an isolated touch patch or prove that learned tokens expose the calculation conveniently.

Pointmap SSI is an isotropic positive scale and translation in this pipeline. Applying it to surface points before VecSetX normalization cancels algebraically: `N(aP+b)=N(P)`. Saved arrays confirm differences below `5.6e-7`. Removing only that preliminary SSI operation is therefore not a substantive fix for the current no-position surface path. Disabling VecSetX normalization itself would be a different, untested input-distribution intervention. [Runtime audit](coordinate_reassessment_20260913/FULL_FRAME_RETURNED_FINDINGS.md), [normalization audit](coordinate_reassessment_20260913/REASSESSMENT.md), [implementation](../../train.py), [normalizer](../../sam3d_objects/model/backbone/dit/embedder/touch.py).

## Corrections that must persist

- “Oracle failed, so its alignment remains wrong” is unsupported. Correct alignment, useful conditioning and accurate reconstruction are separate claims.
- “Dropout resolved generalization” was too broad. Its accurate reference concerned four fitted identities; no matched full-data no-dropout comparison establishes a general dropout benefit.
- “The gain is only additional capacity” does not survive the full-data constant/wrong-surface generation controls.
- “Full finetuning is necessary” is unproven. Broader adaptation helped fitting in a different, camera-oriented target task; it did not establish a remedy for the current full-data oracle model.
- Shared-camera targets, rotation augmentation and alternative readouts did not deliver a transferable fix or a proof that coordinates cannot be handled. Their negative results must not keep reopening the verified camera inverse.
- The historical no-pointmap full-surface run used camera coordinates (`oracle_point_frame=false`). It is relevant negative evidence, but does not isolate residual pointmap involvement under oracle alignment.
- A new decoded/query-feature conditioner would change the representation interface. It is not a coordinate-only certificate and is not selected here.

**Defensible stopping boundary:** ordinary oracle transformation and normalization bookkeeping are verified on the tested observations. There is no identified remaining erroneous transform to repair. We have not proved that every coordinate convention or multimodal interaction is harmless. Calling unknown feature-learning difficulty a remaining “coordinate mismatch” would claim a diagnosis we do not have. Keep the verified oracle convention fixed when eventually testing the remaining learning problem; do not treat a failed model fit as a reason to repeat the same transform audit.
