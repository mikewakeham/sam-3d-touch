# The four-object conditioning improvement does not transfer to new identities

## Validation

Five completed reports and the analysis are archived under `unseen_objects_returned_manual/`. The unchanged analyzer reproduces the returned analysis exactly, including report byte hashes. Source/reference/selection hashes and final checkpoint parameter digests match. Every original/dropout anchor vector replays exactly. The 32 evaluated identities are disjoint from all four fitted identities; each has two fixed views and eight paired draws. Per-object losses reconstruct every reported native batch scalar within the declared tolerance. Both train/validation selections contain 16 objects. `validation.json` records these checks.

These are validations of the returned reports and recorded replay; no independent GPU rerun occurred locally.

## Primary result

All visual inputs and the correct surface are present. Lower native shape flow loss is better; object means average both views and all eight draws.

| Model trained on the four original objects | Mean new-object loss | Objects beating image+pointmap |
|---|---:|---:|
| Image + pointmap | .105401 | — |
| Camera surface, original | .170432 | 0 / 32 |
| Camera surface, dropout | .179867 | 0 / 32 |
| Oracle surface, original | .212288 | 0 / 32 |
| Oracle surface, dropout | .270788 | 0 / 32 |

Dropout worsens camera mean loss by 5.54%, improving only 7/32 objects. It worsens oracle mean loss by 27.56%, improving only 2/32 objects. The direction is present in both dataset splits, not just one pooled outlier. Camera beats oracle on 26/32 objects under original training and 31/32 under dropout. Thus the fitted-identity camera/oracle ranking reverses on these identities.

Correct surfaces are preferred to wrong surfaces for 23/32 camera-original objects, 23/32 camera-dropout objects, 16/32 oracle-original objects and 18/32 oracle-dropout objects. Their mean wrong-minus-correct gaps remain positive, but that sensitivity does not translate into superiority to the no-surface baseline. In particular, the large oracle swap penalty on fitted identities did not establish usable geometry transfer.

## Revision to the working conclusion

The earlier dropout result remains valid: aligned surface conditioning learned a low-loss mapping on four identities and nearly removed their held-view loss gap. It is **not a validated general conditioning fix**. The identity-memorization alternative was not resolved by held views of those same objects; this experiment now demonstrates that the learned improvement fails to transfer to the 32 selected identities.

The result is consistent with learning a mapping specialized to the four shapes, and with greater specialization when alignment/dropout makes those shapes easier to identify. Oracle surfaces provide almost the same point code across views, so their effective point-shape diversity is four objects rather than sixteen independent shapes. This is an explanation consistent with the evidence, not a direct measurement of the network's internal lookup mechanism. Interference with the pretrained image-conditioned mapping is another possible contributor.

The old rotation-factor result remains a causal result for the original four-object task: changing input orientation affected fitting/view transfer while other variables were fixed. It does not establish that removing rotation improves new-object prediction. The new ranking reversal means that extrapolation is now contradicted for these checkpoints.

Nor does this prove that VecSetX lacks geometric information, that the conditioner cannot fit more objects, or that the full approximately 1000-sample setup has the same cause. Native VecSetX reconstruction and successful small fitting remain evidence against a universal information/fitting impossibility. All these checkpoints were intentionally trained on only four identities; they cannot establish the outcome after broader training.

## Decision

Do not promote 50% visual dropout to production or request full training with it on this evidence. Do not start a pose estimator on the premise that the aligned model already transfers. Correct pose is already supplied to the worst-performing new-object model; predicting that pose cannot by itself establish a transferable conditioning path.

Move the next training diagnosis beyond four identities, retaining VecSetX and the coordinate controls. A modest object-disjoint screen should compare image+pointmap, camera full surface, oracle full surface and oracle full surface with dropout, all from the same fresh pretrained initialization and matched optimizer scope. The camera arm preserves the coordinate question; the two oracle policies distinguish whether the recent fitting improvement helps or hurts transfer once object diversity increases; image+pointmap is the actual utility control. Camera-dropout need not be expanded first because it is not the decisive aligned upper-bound question.

Use a fixed training-object set and a separate reserved-object set; preserve within-training-object held views as a secondary diagnostic. Report training, held-view and held-object behavior separately, correct/wrong surface controls, per-object effects and native losses at early checkpoints. Keep the initial budget bounded. If fitted losses are still high and falling, a failed held-object endpoint is not a convergence or impossibility result. Training exposure per object changes when the set grows, so do not compare old/new losses as an isolated causal estimate of dataset size.

The existing preselected 16 train / 16 validation objects are a feasible diagnostic pool, but have already been examined. Any later confirmation must use fresh held objects and another seed rather than call this development set an untouched test set.

Only if aligned full-surface conditioning shows transferable utility should an explicit rotation-handling solution be tested against that improved aligned reference. If camera works while oracle does not, investigate the learned target-frame/readout relationship rather than asserting that object-frame alignment is automatically optimal. If neither point model helps despite adequate fitting and meaningful object diversity, adaptation/interface capacity remains a competing hypothesis; it is not settled by these four-object failures.

No new training driver or GPU job was implemented by this result analysis. No full-dataset run is recommended. Preserve the current checkpoints and all prior results; the generalization finding narrows their scope rather than invalidating their replayed measurements.
