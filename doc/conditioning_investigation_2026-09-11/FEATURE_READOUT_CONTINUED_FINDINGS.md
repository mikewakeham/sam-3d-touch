# Continued readout: substantial fitting, poor transfer; stop extending

All three arms completed 8000 total updates by resuming their 2000-update checkpoints. The paired checker passes. Each arm's resumed assessment exactly reproduces its previous final per-object latent MSE and wrong-object MSE. Source, ordered split, raw and processed feature hashes, targets, common initialization, parameter counts and mean-target controls match the initial reports. Driver hashes match the current file. Raw reports, validation and aggregate analysis are in `feature_readout_continued_returned_manual/`. These are verified returned reports; no GPU tensors or optimizer state were independently recomputed on the laptop.

| Input | Fitted MSE, 2000 → 8000 | Reserved MSE, 2000 → 8000 | Fitted wrong-object MSE at 8000 |
|---|---:|---:|---:|
| Raw | .116254 → .022109 | .245840 → .302009 | .290603 |
| Raw + slot identity | .121020 → .024049 | .234031 → .290328 | .286708 |
| Native processed | .090193 → .015138 | .244182 → .308976 | .303643 |

The mean-target control remains .151084 fitted / .211688 reserved. All three arms reduce fitted MSE by 80–83% during continuation, while reserved MSE worsens. At the endpoint, reserved errors are 37–46% above the mean-target control. Wrong-object inputs worsen all 24 fitted objects in every arm, but only 4/8 reserved objects in each arm. These swaps demonstrate training-object dependence, not reliable geometric transfer.

As a supporting Stage-1 decoder check, fitted target-occupancy IoU reaches 89.16% raw, 89.62% slot-aware and 96.57% processed. Reserved IoUs are 6.56%, 5.50%, and 4.68%. No Stage-2 result or mesh CD is involved. Direct regression MSE and prior flow-training losses are different quantities.

## Decision

Do not request another continuation of this readout. More updates might reduce fitted error further; this is not a claim of exact convergence. Processed improves only 2.5% over the final 500 updates, raw 9.4%, and slot-aware slightly worsens. The diagnostic question has nevertheless been answered sufficiently: a small readout can learn a substantial mapping from frozen VecSetX codes to Stage-1 targets on fitted objects, especially after native processing. The 2000-update endpoint was premature for assessing fitting.

The observed failure now concerns transfer beyond those fitted objects. Extending the same runs has repeatedly improved training fit while worsening reserved performance. There is no demonstrated reason to expect more of the same training to make this a reusable conditioner. This does not prove that dataset size alone causes the original SAM3D results, that all latent translation approaches fail, or that broader SAM adaptation is unnecessary. This head has 24 fitted objects and no visual or flow network; conclusions must stay at that scope.

## Coordinate and architectural implications

An absolute incompatibility between VecSetX information and SAM Stage-1 targets is not supported: substantial translation is learned. The diagnostic uses oracle-frame inputs, so its generalization failure does not require a camera-frame mismatch. Neither statement rules out camera sensitivity in the actual conditioning model. Native processing is a useful positive control for fitting, but its worse reserved endpoint does not establish a better production conditioner.

This follows the previously specified branch: fitted mapping becomes strong, reserved results remain poor, so stop extending direct latent regression. A next constructive experiment should compare an interface that preserves explicit observed xyz against this learned-code route on the same fixed targets and object split. First establish a geometry reconstruction control and strong fitted performance, then use new objects to assess transfer. Keep the old eight reserved objects diagnostic only; do not repeatedly tune to them and report benchmark generalization. Any additional data or spatial structure must be stated as part of the intervention rather than credited solely to coordinate alignment.

Sparse structured touches remain the endpoint. Full-surface success alone will not validate sparse shape completion. Preserve observed position and validity; treat unobserved structure as unknown and test clustered contacts with visual conditioning. Do not adopt a dense target-regression requirement as the sparse conditioning objective.

No additional GPU training job is requested by this result analysis.
