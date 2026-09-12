# Multi-view fitting: target-frame surfaces help, but do not close the gap

## Verified measurements

All three returned reports match the current driver and view-selector hashes, the source single-view result hashes, and logged training-source hashes. Initial weights match the original paired tiny fits. Initial anchor losses reproduce exactly. Each arm contains 1,000 updates, with 250 updates for each of four view groups; all 16 fitted records and 12 reserved records are disjoint and paired across arms. All targets remain the same four object latents. Sampling noise is shared across arms. All assessment and final sampled rows are complete and finite.

Results at update 1,000 (CFG 0 for decoded geometry):

| Condition | Fit loss | Reserved-view loss | Fit voxel IoU | Reserved-view voxel IoU |
|---|---:|---:|---:|---:|
| Image + pointmap | 0.06640 | 0.08574 | 65.22% | 40.62% |
| + Camera-frame full surface | 0.03322 | 0.08071 | 92.72% | 52.19% |
| + Oracle target-frame full surface | 0.02687 | 0.06013 | 96.93% | 66.95% |

The oracle treatment lowers native fitting loss by 19.1% and reserved-view loss by 25.5% relative to camera-frame surfaces. Reserved-view IoU improves by 14.76 percentage points. Oracle loss is lower in every fit group and every reserved group. Mean reserved-view IoU improves for every object:

| Object prefix | Image + pointmap | Camera surface | Oracle surface |
|---|---:|---:|---:|
| 479bbf90 | 53.43% | 77.20% | 78.78% |
| e846574d | 61.52% | 66.68% | 92.83% |
| fe200ce0 | 11.44% | 15.97% | 26.20% |
| 146b6c1f | 36.10% | 48.91% | 70.00% |

These means aggregate three reserved views and two sampling seeds per object, with equal object weighting. Four objects and one training seed do not support a dataset-wide significance claim.

Surface swaps give another useful distinction. On fitted views, loss increases from 0.03322 to 0.05858 for camera surfaces, and from 0.02687 to 0.09182 for oracle surfaces (all 32 paired draws worsen in both arms). On reserved views, camera loss increases only from 0.08071 to 0.08380; oracle loss increases from 0.06013 to 0.11430, with all 24 draws worsening. The oracle model has much more consistently useful object-specific surface dependence under this intervention. Wrong-object surfaces contradict the other modalities, so this is not an isolated surface reconstruction score.

## Interpretation

This is direct controlled evidence that the **surface frame treatment affects multi-view learning** in the inspected setup. It changes neither target latents nor image/pointmap inputs, pretrained initialization, trainable parameter selection, optimizer, nor exposure schedule. It supports the original frame hypothesis as a contributing factor.

It is not proof of a simple axis/sign bug. The treatment removes camera rotation before the encoder's axis-aligned bounding-box/radius normalization and FPS sampling. It therefore bundles orientation, normalization-order, and numerical sampling effects. The initial geometry audit did not find gross stored-camera transform corruption; the prior oracle feature probe demonstrated highly stable same-object features across views.

The result also does not make coordinates a complete explanation. The oracle model still has a large reserved-view gap, especially for the drill. Some individual reserved-view reconstructions remain poor: minimum IoU is 1.17% in the oracle arm. Even its fitted-view minimum is 84.81%, despite the much higher mean.

All models are still improving. Comparing updates 901–1,000 with 801–900, mean training loss drops by 21.6% for camera and 25.3% for oracle. Consequently this budget does not establish a capacity floor, a need for full-model finetuning, or an inherent incompatibility. The frozen VecSetX plus cross-attention/projector pathway already passes the earlier single-view fitting test and substantially improves multi-view fitting here.

The prior one-view transfer probe used the views that are now included in training. The newly reserved views are different. Do not claim a numerically matched before/after improvement in reserved-view generalization between those two experiments; the current camera-versus-oracle comparison is matched.

The original aligned Stage-2 CD remains untested by these pilots. Raw Stage-1 occupied-voxel distances are not interchangeable with that mesh score, even when the values happen to resemble it.

## Corrected interpretation and withdrawn full-run recommendation

The user subsequently clarified that they **already ran oracle conditioning and obtained results similar to the other variants**. This is user-reported evidence; the corresponding detailed metrics have not been provided here. Absence of `--oracle-point-frame` from the current job scripts did not establish that the experiment had never been run. The earlier inference about run history was unwarranted.

The recommendation to repeat a full-dataset oracle run is withdrawn. `run_full_dataset_frame.sh` is retained as a historical artifact and is not a requested next step. Do not relaunch or wait for that proposed job as the investigation's dependency.

The pilot's moderate benefit is evidence of some sensitivity to the frame treatment, but it does not justify treating frame correction as the main solution. Its information advantage is substantial: both full geometry and the correct object-frame transform are supplied. Remaining poor reconstructions and the user's prior full-dataset result weaken the case for allocating another full run to this treatment.

The more discriminating existing observation is the oracle view-transfer intervention: with surface and target fixed, changing the image/pointmap increases native loss from 0.0189 to about 0.128. Thus providing nearly identical full-shape features does not guarantee stable reconstruction across the accompanying views. The model can exploit surface-specific information to fit a few examples, but that does not establish a robust geometric mapping.

This shifts priority toward how image/pointmap and surface conditions interact during optimization, and how much the frozen downstream computation restricts use of the surface representation. These remain hypotheses. Cross-attention-only fitting succeeded on the earlier tiny set; that rules out a universal inability to fit, not a restriction at broader scale. No new GPU experiment is requested in this update, and no production change or full-model finetuning requirement is claimed.

## Output-directory change and artifacts

Per the user's correction, future investigation output paths and diagnostic reference roots in the investigation launchers now use `outputs/conditioning_investigation/`. Original production job scripts and historical returned JSON were not rewritten. The withdrawn full-dataset launcher also uses the same root.

Raw attachments are archived byte-for-byte in `multiple_view_returned_46083371/`. Run `analyze_multiple_view_fit.py` to regenerate `multiple_view_summary.json` and repeat the provenance, pairing, split, exposure, and finite-value checks. The new launcher passed shell syntax checks locally. No GPU training or new model evaluation was performed locally.
