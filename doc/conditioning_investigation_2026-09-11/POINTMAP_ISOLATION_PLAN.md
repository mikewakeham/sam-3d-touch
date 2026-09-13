# Remaining pointmap coordinate question

13 September 2026. Comparison following the evidence reset. No training launched or production source changed. Two launchers are now prepared in `jobs/`: `stage1_full_surface_oracle_no_pointmap_dropout.sh` and `stage1_full_surface_oracle_constant_no_pointmap_dropout.sh`. They copy the existing four-H200 templates, changing only job/output names and adding `--no-pointmap`; both pass shell syntax checks.

## Existing evidence and exact missing cell

- `nouqb3mh`, `stage1_full_surface_no_pointmap_full_cross_attention`: completed 20 epochs/14660 steps, `no_pointmap=true`, **`oracle_point_frame=false`**, frozen raw VecSetX, no touch position, full shape cross-attention. It does not answer pointmap removal under oracle alignment.
- `cps50h2l`, `stage1_full_surface_oracle_dropout`: completed the same duration, oracle=true, no_pointmap=false, visual_dropout=0.5.
- `hwz5ifqt`: corresponding constant-surface oracle/dropout control, with pointmap present on visual-present updates.
- `fl7b2znc`: historical image-only/no-pointmap baseline. Its dropout/data-manifest policy differs from the new runs; use as context, not a fully matched control.
- No completed full-data oracle/no-pointmap run was found in the available exports or job scripts. Missing exports remain possible; this is not a claim about unavailable cluster files.

The current dropout recipe zeroes all visual tokens, including image/mask/pointmap, independently for half the examples. It already exercises surface-only training and inference. It is not a separately trained surface-only model: the other updates also determine the shared generator weights.

## Recommended first comparison

| Arm | Surface | Pointmap information | RGB/mask policy |
|---|---|---|---|
| A: existing reference | Real oracle | Present on visual-present examples | Existing 50% visual dropout |
| B: missing treatment | Real oracle | Always zero | Same 50% visual dropout |
| C: matched utility control | Same fixed oracle surface for every object | Always zero | Same 50% visual dropout |

B versus A tests whether excluding pointmap throughout learning improves the aligned model. B versus C tests benefit from actual surface information within the no-pointmap setup. The old constant run is not C because it retains pointmap information. Thus the rigorous complete comparison needs two new runs, with A reused.

The constant arm is not required to answer the narrower A/B question of whether excluding pointmap helps. It is the additional control for claiming sample-specific geometry benefit within the new setup. Prioritize B if only one run is practical. Retain dropout0.5 for comparability, not because its optimality is established: B trains on RGB/mask plus surface for half the examples and surface without visual information for the other half.

Use the same fresh pretrained initialization/seed, full-surface manifest, target labels, frozen VecSetX, projector/full shape cross-attention scope, learning rates, batch size, dropout decisions and 20-epoch budget as A. Do not resume A only for B: that would test removal after previous pointmap exposure. Save intermediate checkpoints if early interpretation is desired; an early failure does not establish the matched final outcome.

Existing `--no-pointmap` supports B/C without changing the representation. It zeros both cropped/full pointmap condition streams using the existing fuser convention, preserving token count. It does not mean RGB is absent. Pointmap preprocessing still executes, but the oracle surface path does not use its SSI transform; verify that changing pointmap content leaves the final condition invariant before interpreting B.

Assess final models with the same Stage-1 sampler and decoded-target reference, correct/wrong surfaces, train and held identities, raw and common rigid-adjusted geometry. Include native loss but do not promote a variant from pooled loss alone. Existing camera/oracle/constant geometry remains contextual; historical image-only generated geometry has not been measured under this exact probe.

## Branching interpretation

- B improves absolute accuracy over A and beats C: pointmap exclusion is useful. This identifies pointmap involvement, not scale/translation mismatch specifically.
- B improves over A but fails to beat C: changed visual adaptation can explain improvement; useful surface conditioning remains unestablished.
- B remains inaccurate: pointmap/surface conflict is not necessary for that residual because B never receives pointmap information. If B is nevertheless better than A, pointmap can still be a contributor to A.
- B fits accurately but fails on new identities: the remaining problem is transfer under the tested aligned/no-pointmap recipe.

No finite negative establishes that all coordinate conventions are harmless. This comparison does provide a precise stopping point for blaming a pointmap/surface mismatch as the sole cause of poor oracle reconstruction.

## Explicit shared-frame pointmap: conditional second comparison

Keep oracle surface/VecSetX input unchanged. For every valid pointmap XYZ, apply the same camera-to-object transform and the exact center/radius used for that sample's full surface:

`pointmap_shared = (R * pointmap_camera + t - surface_center) / surface_radius`

Apply the convention consistently to both cropped/full pointmap streams; preserve pixel correspondence and validity. Do not normalize pointmap and surface independently afterwards, which would destroy the intended common coordinates. Disabling only the preliminary SSI transform on surface points cancels under subsequent VecSetX normalization and is not this intervention.

This is explicit geometry preprocessing, not a decoder or new latent representation. However, it changes the pretrained pointmap encoder's coordinate distribution and changes rotation as well as scale/translation. Success would support the complete preprocessing treatment; failure would not cleanly refute coordinate alignment. To specifically attribute a benefit to shared center/scale, compare against a pointmap with the same oracle rotation but the original SSI centering/scale. That additional control is why the shared-frame run is not selected as the first full training comparison.

Pointmap pixel indexing and XYZ coordinate frame are separate. Transforming the XYZ stored at pixel `(u,v)` leaves its correspondence to that image pixel intact; do not rotate or resample the image grid. Projecting those transformed coordinates would require the corresponding camera transform. The present oracle path transforms only the full surface, leaving pointmap in its SSI camera convention. These can describe the same geometry consistently in different frames; they are not intrinsically contradictory. The model is not explicitly given the oracle frame conversion as a separate condition, so the different conventions may still complicate learned fusion. Pointmap removal tests that concern without changing the pointmap encoder's input frame.

If B is already inaccurate with no pointmap information, explicit pointmap alignment cannot explain or repair B's absent-stream conflict. Prioritize diagnosing that residual rather than automatically launching every normalization variant. If B improves convincingly, shared-frame pointmaps become a purposeful attempt to retain pointmap benefits without the observed interference.

## Considered stricter control: oracle surface alone throughout training

The user subsequently asked about removing image information as well. This directly tests whether poor aligned reconstruction persists without any visual-stream interaction during learning. Keep the same target convention, frozen VecSetX and adaptation scope. Compare against the existing oracle/dropout checkpoint under **the same zero-visual evaluation**, and against the no-pointmap run under that evaluation if available. The visual-present baseline remains a separate practical comparison; do not conflate changing evaluation inputs with changing training policy.

`--visual-dropout 1.0` alone zeroes all visual context during training, but `train.py::validate` calls `model(*prepared)` without a visual-drop mask. That flag alone still logs visual-present validation. The user authorized a permanent `--no-visual` integration instead; it is now implemented in `train.py`, `evaluate.py` and `diagnostics.py`. It uses the native fuser's forced modality dropping for every visual stream, preserving token count and leaving separately appended surface tokens active. The saved conditioning configuration restores this policy for Stage-1 inference. Missing `no_visual` in older checkpoints means visuals retain their original behavior.

`jobs/stage1_full_surface_oracle_no_visual.sh` uses the existing four-H200 template with new job/output names and `--no-visual` replacing `--visual-dropout 0.5`. Oracle alignment, full surfaces, frozen VecSetX and full shape cross-attention adaptation are retained. W&B receives the flag through the existing argument logging. No separate constant run is required for the narrower surface-only fitting question. No GPU run has been launched locally.

`oracle_upper_bound/test_no_visual_integration.py` executes native fuser and condition-preparation code on CPU with lightweight neural stand-ins. It checks all visual streams become zero without changing token counts, altered visual content has no effect, surface gradients remain active, validation receives zero context, saved flags control restored inference in both embedder locations, and incompatible resume modes are rejected. It does not execute pretrained SAM3D generation or CUDA.

Validation completed: three new CPU tests and all ten existing source-integration tests pass, including the two-rank Gloo regression. The distributed test initially failed because sandboxed loopback socket binding was prohibited; the same test passed with loopback access. Python compilation, job shell syntax, production CLI parsing of the job command and `git diff --check` pass. GPU execution remains unverified locally.

Accurate surface-only fitting would establish that this recipe can learn the aligned mapping without image/pointmap input; held-object reconstruction remains a separate test. Improvement over equally evaluated mixed-training models supports an effect of the visual training policy, not specifically a pointmap scale error. Continued surface-only failure cannot require a visual/surface frame conflict. It leaves aligned feature-to-target learning unresolved, without proving an architecture or representation impossibility.
