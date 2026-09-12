# Rotation, rather than bbox normalization, drives the measured frame penalty

## Validation

Both mixed treatments completed 1000 updates. Driver and training-source hashes match. Their single-view and multi-view reference hashes match the archived camera/oracle reports. The returned baseline preflight exactly reproduces all initial camera/oracle correct/swap losses across the seven view groups, with identical baseline input/feature hashes and initial parameters. The returned combined analysis exactly matches recomputation by the unchanged analyzer. All training, assessment and sampled coverage checks pass.

Raw attachments and combined analysis are archived under `frame_factors_returned_manual/`. This validates the returned experiment and its recorded preflight; GPU computation and historical optimizer trajectories have not been independently rerun on the laptop.

## Matched result at 1000 updates

| Treatment | Orientation | Normalization basis | Fit flow loss | Reserved-view flow loss | Fit / reserved Stage-1 occupancy IoU |
|---|---|---|---:|---:|---:|
| Oracle | Object | Object | .026873 | .060126 | 96.93% / 66.95% |
| Rotation only | Camera | Object | .032779 | .080737 | 94.36% / 52.68% |
| Normalization only | Object | Camera | .028130 | .064269 | 96.88% / 63.62% |
| Camera | Camera | Camera | .033217 | .080713 | 92.72% / 52.19% |

Rotation alone reproduces essentially the full aggregate reserved-view camera/oracle penalty: .080737 versus the original camera .080713. Retaining camera-dependent normalization while removing rotation leaves the model substantially closer to oracle (.064269 versus .060126). The direction is corroborated by Stage-1 latent errors and occupancy, not just one flow-loss aggregate. Rotation-only has lower reserved-view occupancy IoU than oracle for all four objects; normalization-only beats camera for all four.

Conditional effects in reserved-view flow loss:

- Introduce rotation with object normalization: +.020611.
- Introduce rotation with camera normalization: +.016445.
- Introduce camera normalization in object axes: +.004143.
- Introduce camera normalization in camera axes: -.000024.
- Interaction: -.004167.

Do not describe these as additive percentages of blame: the interaction is nonzero. Normalization is not irrelevant, but fixing bbox/radius behavior alone is not supported as the primary remedy. Both mixed-treatment training curves are still falling: mean loss in the last 100 updates is about 21.6% lower than the preceding 100 for rotation-only and 24.4% lower for normalization-only. This remains a finite-budget, four-object, one-seed result, not a demonstrated asymptotic ceiling.

## Specific coordinate diagnosis now supported

In this controlled setup, presenting full-surface information in changing camera orientations makes learning and view transfer harder even when object-based normalization is retained. Removing rotation helps while preserving camera-derived normalization. Because all other training factors are fixed, orientation presented to the pretrained encoder/readout is a causal factor in the measured penalty.

The relevant mismatch is camera-oriented point features supervising an object-frame spatial target. The known camera-to-object rotation is not explicitly supplied to the normal conditioning path. The model must learn to express the observed geometry in the target frame. These results identify that burden; they do not isolate whether its difficulty lies in interpreting rotation-sensitive VecSetX features, combining their frame with visual conditions, or selecting the target's canonical orientation. Native VecSetX reconstruction remains evidence that geometry can survive encoding in either frame, so replacing it is not indicated.

This is not evidence of a wrong global axis/sign conversion, nor proof that the entire dataset problem is caused by coordinates. Oracle alignment still leaves substantial view dependence; the user's previous full-dataset oracle run also remained similar to the other variants.

## Overnight training decision

Do not start a full-dataset run from this result alone. No new deployable correction has been demonstrated. Repeating full oracle training would repeat an already reported experiment; a normalization-only production change is not supported as the main fix; scratch encoding or broader finetuning would change the investigation before following the isolated coordinate factor. The two mixed runs are complete and should not be extended automatically.

## Next branch, preserving the agreed plan

Follow the orientation-dominant branch of `COORDINATE_DIAGNOSIS_RESYNC.md`. Retain VecSetX and identify how the required camera-to-target rotation can be supplied or recovered under the actual input contract. Distinguish measured rotation sensitivity from an unidentifiable target convention. Known inverse alignment is the positive control already available; another arbitrary rotation search is not the next step.

A targeted coordinate-handling intervention needs a short matched test before a full run. It must improve camera-frame conditioning and maintain the point-input path; oracle poses used for diagnosis cannot be silently treated as available inference inputs. Rotation augmentation alone is not assumed to solve arbitrary canonical orientation recovery.

The remaining oracle failure is separate. The earlier RGB+pointmap view probe did not split their contributions. If that residual remains after a coordinate intervention, separate those conditions with the oracle surface fixed before claiming a conflicting pointmap frame. A synthetic gauge ambiguity is not proof of an actual dataset collision; an impossibility conclusion must match the intended observations and target convention.

No additional GPU job has been implemented or requested by this result analysis.
