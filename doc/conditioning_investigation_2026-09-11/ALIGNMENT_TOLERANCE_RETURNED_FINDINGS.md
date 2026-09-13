# Alignment-tolerance results — 12 September 2026

## Validation and scope

Archived the five supplied attachments in `alignment_tolerance_returned_manual/`: two identical aggregate analyses and raw shards 0, 1 and 3. Raw shard 2 (15°) was not attached. No rerun is needed; its existing report is sufficient to finish validation.

`validate_returned_subset.py` retains the original analyzer's validation checks, selects only supplied shards by their actual IDs, and compares their recomputed cells to the supplied aggregate. All supplied shards passed exact source/reference hashes, checkpoint parameter hashes, recorded native replay, input grouping, coordinate checks, complete unique row coverage, common sampling noise, and original sampled-result replay. Their file hashes match the supplied aggregate. Every supplied cell was reproduced exactly, and all aggregate gate arithmetic was rechecked. The 15° cells remain aggregate-only evidence until raw shard 2 arrives. See `local_validation.json`. Original hashed driver/protocol/analyzer files were not changed.

Four fitted identities, four fitted views and three reserved views, two sampling seeds, CFG 0, 25 steps. Weights were unchanged during this experiment. Reference: decoded GT Stage-1 occupancy, without registration; perfect agreement is IoU 1. This does not evaluate Stage 2, source-mesh CD, new identities, sparse contacts, or a learned pose estimator.

## F9 — The aligned dropout model has a target-referenced reconstruction endpoint

| Model/input | Fitted-view mean IoU | Reserved-view mean IoU | Lowest reserved object-mean IoU |
|---|---:|---:|---:|
| Original oracle, correct surface | 96.93% | 66.95% | 26.20% |
| Dropout oracle, correct surface | 97.80% | 97.17% | 91.96% |
| Dropout oracle, wrong surface | 2.55% | 2.49% | 1.40% |

The dropout checkpoint passes the prespecified aligned-reference criterion: reserved mean ≥95%, each object's view/seed mean ≥90%. This strengthens F5: its native-loss improvement now has an actual sampled Stage-1 reconstruction counterpart. The fit-minus-reserved IoU gap is 0.64 percentage points versus 29.98 for the original model. It is not an exact reconstruction claim: two of the 24 reserved object/view/seed samples fall below 90% IoU, with minimum 88.58%. Averages do not establish uniform success.

Wrong surfaces reduce reserved target agreement from 97.17% to 2.49%, so these weights' accurate fitted-identity reconstruction depends on which surface is supplied. This rules out an explanation based solely on the stream's presence for this particular result. It does **not** distinguish geometric reconstruction from identifying one of four memorized objects, nor show that the output follows the donor geometry. F6/F7 remain relevant: the prior new-object and constant-input results are not contradicted.

## F10 — High aligned accuracy coexists with direction- and object-dependent rotation fragility

Known residual rotations were applied to oracle-aligned points before the unchanged VecSetX normalization and encoder, with image, target, weights and sampling noise fixed. This measures the response of the complete conditioning path to imperfect alignment, including normalization's response to rotation. It does not separately assign sensitivity to encoder features, normalization or the learned projection/attention.

| Residual angle | Range of reserved mean IoU across six signed axes | Worst object-mean across those axes | Directions passing the prespecified criterion |
|---|---:|---:|---:|
| 0° reference | 97.17% | 91.96% | Reference passes |
| 5° | 94.14–97.13% | 85.51% | 2/6 (+x, −y) |
| 15° — aggregate only | 79.00–93.25% | 42.21% | 0/6 |
| 30° | 64.74–85.63% | 13.56% | 0/6 |

The tolerance criterion requires every object's mean to lose at most two IoU percentage points relative to exact alignment and remain at least 90%. No tested nonzero angle passes across all directions. At −5° about x, one object's IoU falls from 99.41% to 92.81%; at −5° about z, another falls from 91.96% to 85.51%. These are meaningful failures hidden by generally high aggregate means. The +5° z failure is marginal against the engineering thresholds, whereas the larger drops above are not.

Established: exact alignment produces an accurate fitted-identity endpoint, but even small residual rotation can hurt this checkpoint. A coarse orientation estimate cannot simply be assumed sufficient.

Not established: all 5° errors are unacceptable, a universal required error below 5°, a discontinuity at zero, all rotations of a given angle behave identically, pretrained VecSetX is inherently incompatible, or an architectural change is required. Real estimator errors, object symmetries and a camera-to-target correction have not been tested. Rotation sensitivity is not a new-object generalization result.

## Consequence for the next intervention

**Subsequent clarification from the user's output-frame question:** the reported degradation is fixed-target-frame occupancy disagreement, which can include rigid output rotation. It does not yet establish shape degradation after removing pose. Before launching augmentation, inspect existing saved NPZ predictions using known inverse-rotation compensation and a separate rigid-registration diagnostic. Compare against the accurate aligned prediction and GT support, with discretization controls; failure of a local registration optimizer alone is not proof of shape distortion. Target-based registration is an offline diagnostic and is unavailable for a genuinely unknown object at deployment. If shape survives and only pose changes, accurate canonical input alignment may be unnecessary for pose-independent reconstruction. If shape is damaged too, post-hoc alignment cannot repair it. This distinction qualifies the interpretation above without changing any measured IoU or gate result.

A potential coordinate repair candidate, after that distinction, is **small residual-rotation augmentation through the existing VecSetX path**, compared with an otherwise identical continuation without rotation augmentation. This would test whether the measured fragility is trainable, rather than demanding an arbitrarily precise estimator from the outset. It is a candidate, not implemented or launched by this results ingestion.

Keep the target unchanged when simulating an imperfect *correction*: perturb only the already aligned point input around its correct frame, as in this probe. This would not be arbitrary global rotation augmentation of camera inputs against an unchanged canonical target, and it would not rotate spatial latent channels. Full surfaces remain the current condition; future normals would rotate with their points.

Before handing off that experiment, specify a matched update budget, retain the exact-alignment endpoint, and test independent residual rotations (including mixed-axis errors) on reserved views. Success means retaining target accuracy while improving tolerance relative to equal additional training, not simply lower fitting loss. If successful, we would have repaired local alignment robustness for this fitted-identity setting. We would still need an observable approximate frame correction and the separate held-object utility criterion. If it fails, inspect the response of the existing encoder/normalizer and learned conditioning path before claiming a fundamental limit.

The user's output-frame preference (camera/robot frame acceptable versus original asset axes required) remains unanswered in the visible conversation. Do not silently change the target-frame contract. No overnight full-training candidate is earned by these four-object results alone; F6's transfer failure remains unresolved.
