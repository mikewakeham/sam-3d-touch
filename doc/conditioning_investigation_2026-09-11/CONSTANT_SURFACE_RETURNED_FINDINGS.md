# A fixed surface reproduces the main benefit and transfer failure

The complete constant-surface report and supplied analysis are archived in `constant_surface_returned_manual/`. Recomputed analysis matches exactly, including report hashes. All original sources and reference hashes match; exact real-surface initialization replay, the full schedule, fixed training-only bank selection, constant shape-attention context, scalar/per-object agreement and final swap invariance pass. The returned GPU measurements are validated, not independently rerun on the laptop.

## Main result: sample-specific geometry is not needed for the apparent fitting gain

The constant arm supplies the same actual training surface to every example, throughout training and assessment. Visual inputs, including pointmap, still vary. It has no sample-specific full-surface information; it is not a no-geometry image baseline.

| Arm | Fitted views | Held views, fitted identities | Held identities |
|---|---:|---:|---:|
| Image + pointmap | .061607 | .062941 | **.119303** |
| Real camera surface | .055440 | .058075 | .126404 |
| Real oracle surface | .056473 | .057185 | .134199 |
| Same fixed surface for every input | **.053823** | **.056386** | .127901 |

The constant arm improves fitted loss over image for all 16 identities. Real oracle loses to constant on 14/16 fitted identities and 10/16 held-view identities. Thus even the real-surface fitting improvement in this 16-object screen is not evidence that the model needs the current object's surface to achieve it. This does not invalidate the old four-object fit/swap results; it narrows what can be inferred from ordinary image-versus-surface comparisons.

On held identities, constant loses to image on 15/16 objects. Its mean is better than real oracle but slightly worse than camera. Do not claim oracle is uniformly worse than constant: oracle wins on 9/16 held objects and its median paired difference is -.001816, while its mean difference is +.006298. Outliers matter. This is not an equivalence proof between real and constant surfaces, nor evidence that every object ignores geometry.

## The same learning trajectory occurs without sample-specific surfaces

On the separate original monitor bank, constant held-object loss goes .456622 -> .124701 -> .120724 -> .124416 at steps 0/256/512/1024. At 256, constant has the best held-object mean among all four arms and beats image on 13/16 objects. Real camera and real oracle each beat constant on only 5/16. Later, constant continues improving fit while its held loss rises from 512 to 1024, just as in the real-surface runs.

This is stronger than the preceding wrong-surface inference probes: the sample-independent input was present during learning, so its success is not an artifact of removing or substituting a stream only at evaluation. An additional learned token pathway can reproduce the broad early-benefit/later-specialization pattern without observing the current object's full surface.

## What is now diagnosed, and what is not

The existing setup conflates two benefits: extra trainable conditioning capacity and use of the current surface's geometry. The former is sufficient to explain a substantial apparent benefit in this controlled task; the latter has not produced a convincing transferable improvement over the matched constant control and image baseline. Large removal penalties therefore cannot be interpreted as large geometric benefits.

This strengthens the adaptation/interface explanation and justifies a constructive intervention. It does not identify which internal operation is responsible: learned common token features, projector biases, shared attention updates and competition in joint attention remain coupled. A single fixed bank and seed do not characterize every sample-independent alternative.

Coordinate rotation remains a demonstrated burden in the earlier factorial. However, correct oracle alignment does not overcome this failure, and the fitting/transfer pattern appears without sample-specific surface rotation at all. A global frame patch or pose estimator alone is not justified as the solution. VecSetX's native reconstruction evidence still supports retaining it. No replacement encoder, mesh/voxel bridge, or sparse-touch implementation follows.

## Next: test a bounded corrective interface, with controls

Further probes of the same joint-attention checkpoints would have diminishing value. `SEPARATE_SURFACE_HANDOFF.md` implements a new experiment around the completed image+pointmap checkpoint:

- Freeze the image generator. Add an independent shape surface-attention residual at each block, initialized to zero output. Keep VecSetX and the current point projector. Visual and surface branches have independent attention normalization. With the surface branch absent, the original image prediction must be reproduced exactly after training.
- Compare real oracle surfaces against a constant surface under that same new interface. Actual geometry must earn an improvement beyond the additional branch's general adaptation capacity.
- Compare camera against oracle under that same interface to retain the coordinate question.
- Train an ordinary joint-attention oracle arm from the same image checkpoint. This controls for starting from an already adapted image model, so an improvement is not automatically attributed to the separate branch.

These are 1024 additional updates on the fixed diagnostic split, not full training. The proposal changes freezing, attention normalization and residual initialization together. It is a constructive package, not a one-factor proof of the mechanism. A successful pilot earns a fresh-object/seed confirmation and decomposition where needed; it does not complete the original goal by itself.

A preserved image fallback or improved fitted loss alone is not success. The real-surface arm must show coherent held-object improvement over both image and constant, alongside meaningful correct-versus-wrong surface utility. If aligned surfaces pass and camera does not, there is finally a useful aligned reference against which to test explicit rotation handling. Other outcome branches and the exact terminal command are in the handoff.
