# Returned target-frame probe: no convincing coordinate correction

The returned frozen-pretrained probe does not support changing dataset target orientations as the main fix. Original target generation reproduces exactly, alternative orientations remain accurately representable by the VAE, but choosing another orientation gives little transferable flow-loss benefit and worsens average sampled geometry under the prespecified selection rule.

## Validation

Raw attachment archived byte-for-byte in `target_frame_returned_46083371/results.json`; analysis and validation reports are alongside it. All five reported source hashes match the current source, including the driver, voxelizer and training path. Eight distinct objects have two views each, with complete unique coverage of 576 loss rows and 192 sampled comparisons. Input batches contain the expected sample IDs and noise hashes. Original regenerated targets equal stored targets exactly for all eight objects. Native and explicit loss differ by only 1.49e-8.

Candidate VAE reconstruction IoU is at least 0.999593 across all 48 object/orientation combinations; every identity reconstruction is 1.0. Mean squared latent magnitude is approximately 0.130 for all orientations. There is no evidence here that an orientation improvement comes from destroying the encoded shape. This is surface occupancy reconstruction, not the user's registered Stage-2 mesh CD reference.

The JSON reports H100 execution and CFG 7 for rollouts. Checkpoint hashes are recorded but cannot be independently checked without the cluster weights. NPZ artifacts and Slurm log were not supplied; the JSON/source checks support analysis without claiming independent reconstruction of its tensor measurements.

## Prespecified result

Each object's yaw was selected using view 0 / noise draw 0 and then assessed on view 1 / noise draw 1. Selection considered identity and three Z quarter-turn rotations, using mean loss over t=.2,.5,.8. The analysis code was written before results arrived and was not changed for this result.

| Held view and independent noise | Original orientation | Selected orientation |
|---|---:|---:|
| Mean native flow error over the three tested times | 0.822050 | 0.819223 |
| Mean sampled latent MSE | 1.005188 | 1.019536 |
| Mean sampled voxel IoU | 0.109626 | 0.091551 |
| Mean sampled unregistered occupied-center CD | 0.098269 | 0.114261 |

Mean flow error improves **0.344%**. Five objects improve, two worsen, and one selects identity. Sampled geometry does not corroborate a useful aggregate correction: IoU falls by 1.81 percentage points and CD increases. The shield and bathtub benefit geometrically from the selected yaw, while other objects worsen. Those exceptions explain some output-frame errors, not the original dataset-wide training failure.

Selecting one global yaw on the first four objects chooses Z270. On the remaining four objects it **worsens** held-view flow error by 0.265% (0.825286 to 0.827476). The ±90° X controls also worsen the overall held-view loss relative to identity. Together with the exact source-axis audit, there is no supported global coordinate patch.

## Important limits on the negative result

The three-time average is a diagnostic grid, not an estimate under the training lognormal time distribution. High-time errors are much larger and dominate that average. For the already selected rotations, held-view error changes are:

| Time | Original | Selected | Relative change |
|---|---:|---:|---:|
| .2 | .180588 | .175100 | −3.04% |
| .5 | .936751 | .934806 | −0.208% |
| .8 | 1.348809 | 1.347763 | −0.078% |

Thus “orientations have no effect” would be too strong. A modest low-time effect exists in this subset. The absence of sampled-geometry corroboration and global transfer still makes a dataset-wide re-encoding/retraining recommendation unjustified. This tested six discrete orientations, eight objects and two views. It does not rule out continuous alignment or every possible canonicalization scheme, and it does not test learning under changed supervision.

Rollouts generate all latent modalities normally, whereas the shape-only training probe retains the existing zero layout targets. The model is frozen and has no VecSetX/surface input in this experiment. It isolates target/prior compatibility within that setup, not the full causal effect of target orientation on a trained surface-conditioned model. No finite negative probe establishes architectural impossibility.

## Updated diagnosis and next decision

The coordinate evidence now has three levels:

1. Camera transforms and stock axis conversion agree in the inspected data.
2. Oracle surfaces remove cross-view surface variation but fail to resolve the overall problem; the user also reports a previous full-dataset oracle run with similar outcomes.
3. Re-encoding targets into candidate alternative orientations produces no convincing transferable correction in this frozen-prior probe.

Do not patch coordinate signs, rotate saved latents, repeat the full oracle dataset run, or expand the rotation search solely because the first search failed. No concrete coordinate fix has earned a training run from these results.

The strongest remaining causal observation is still the earlier fitted-model probe: changing only the visual view while keeping oracle surface tokens fixed increased loss from about .0189 to .1277. This proves sensitivity to visual conditions in that fitted model; it does not establish that the visual pathway is the sole cause, that the frozen surface encoder loses geometry, or that full backbone finetuning is necessary.

A more discriminating next experiment would isolate that visual dependence in training: keep target-frame full surfaces fixed and compare the existing fusion with an arm that removes visual content while retaining surface tokens and the same trainable parameter scope. The source allows modality dropping before surface concatenation. Train with the intervention; merely dropping images at inference is an out-of-distribution perturbation and cannot establish the attainable fitting limit.

Crucially, other views of the same object become effectively duplicate inputs in a surface-only oracle arm. High performance on those views would be automatic invariance, not evidence of geometric generalization. Such a control must establish fitting, require degradation under wrong-object surface swaps, and use separate objects if claiming geometric generalization. If it succeeds where ordinary fusion fails under matched optimization, this motivates fusion/modality-training changes before a full-backbone comparison. If it fails, a controlled richer-adaptation positive control is needed; failure alone cannot distinguish the encoder, readout and optimizer.

No new GPU job is requested in this findings update. The returned orientation job is complete; the remaining question is a conditioning/learning intervention, not another unrun orientation probe. The investigation remains unresolved.

### Follow-up design check: do not repeat the four-object fitting control

Reviewing the completed fitting results narrows the proposed surface-only experiment. Ordinary oracle fusion already achieved 99.19% single-view training IoU and 96.93% multi-view training IoU. Another four-object surface-only fit would therefore have little power to distinguish the explanations. Its apparent reserved-view success could follow simply from removing all varying input, as noted above. Do not interpret such a result as fixing the original failure or use it alone to justify a fusion rewrite.

A substantive comparison needs enough distinct objects that the existing model exhibits the reported failure, paired ordinary-fusion and surface-only training under oracle coordinates, and separate reporting of training-object reconstruction and unseen-object reconstruction. First establish the baseline failure at that chosen scale. If both arms fit but neither generalizes, this points away from the claim that fusion prevents overfitting. If surface-only improves only on other views of training objects, that establishes invariance rather than geometric generalization. If training is still improving, a short budget cannot establish a capacity limit. A richer-adaptation control is warranted only after locating a reproducible failure under these conditions.

This is a GPU dependency. The laptop cannot establish the necessary baseline, select an informative fitting scale from new model runs, or validate an architectural fix. No additional source-only coordinate search or automatic launch is justified while that dependency remains.
