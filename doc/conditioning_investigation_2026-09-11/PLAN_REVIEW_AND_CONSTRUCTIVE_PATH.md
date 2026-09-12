# Revised priority: test a usable surface signal, then a direct conditioning path

## Decision

The original A–D factorial is useful for distinguishing visual-content effects from trainable scope, but is not the highest-value first block. It holds the current VecSetX-to-SAM3D interface fixed, so it may spend four runs diagnosing how to optimize a difficult representation translation. The user permits changes to representation, encoder, fusion and objective, has access to up to four H100/H200 GPUs, and prioritizes iteration latency over compute cost. There is no requirement to preserve the current implementation.

The revised first block tests a concrete alternative: derive a spatial condition from the observed surface in SAM3D's native shape representation, then inject it into the corresponding shape-token locations. This is a candidate to test, not a proven fix. The old factorial becomes a conditional follow-up, not the initial four-run commitment. Stage 1 remains the sole objective.

## New bounded local evidence

`audit_surface_voxel_coverage.py` uses the five previously selected geometry-bundle objects, inverse camera transforms and actual recorded 8,192-point surfaces. No original mesh or target is used to construct the input grid. All point-hit voxels lie inside the saved decoded target occupancy. Their coverage of target occupied voxels is 47.21%, 58.88%, 87.86%, 97.12% and 61.31% respectively. All archived target occupancies for each object agree.

This does not mean the observations lack the information needed to reconstruct the object. Surface interpolation can infer unsampled areas; point-hit coverage is not an information bound. It does establish that naive point voxelization is not identical to the triangle-voxelized training target. One-voxel dilation increases recall but adds many false occupied cells and often worsens IoU; it is not a justified universal fix. The five-object diagnostic subset is not a prevalence estimate.

## Revised sequence

### 1. No-training representation test

Use the actual recorded surfaces for 32 prespecified objects. Keep target-frame inversion explicit and identical across alternatives. It is privileged alignment used to isolate representation, not a deployment solution. Compare:

- Current VecSetX input and native reconstruction, with its exact normalization inverted for geometric comparisons. Native decoding is a test of this encoder's input/representation contract, not proof SAM3D can read the same features through the current adapter.
- Binary point-hit occupancy encoded/decoded by the existing SAM3D Stage-1 VAE. Compare the resulting surface-derived latent with the stored target latent and inspect raw occupancy fidelity. Encode in the VAE's object-cube scale, not VecSetX's unit-radius scale.
- The stored mesh-derived target latent as a **scoring reference only**. It must not be an input to a claimed surface-only reconstruction.

Fix preprocessing in advance; do not select the best grid construction against each object's target. Report per-object errors, density and the point-hit coverage above, not only an average. A native decoder comparison across models is descriptive: different decoders and sampling procedures prevent a clean architecture ranking from that metric alone.

**Branch:**

- Surface-derived SAM latents are already good: a useful representation exists without learning VecSetX-to-SAM translation. Advance to spatial conditioning.
- SAM encoding is poor while native VecSetX reconstruction is good: do not force a VAE representation merely because it is convenient. Test direct supervised VecSetX-to-SAM latent prediction or a point encoder with explicit spatial features; the frozen VAE may not tolerate sparse point-hit grids.
- Both are poor: isolate input preprocessing/coverage before training SAM3D. A density ladder on the cluster from original meshes may test sampling loss, but higher-density resampling is an explicitly changed input, not a silent replacement of the user's recorded surfaces. Successful reconstruction at higher density supports a data/preprocessing intervention; failure at one density does not prove missing geometric information.

This test requires GPU execution but no optimization and can be performed alongside preparation of the short training comparison. It avoids committing to a new encoder before checking whether it supplies a useful signal.

### 2. Short test of a direct spatial conditioning path

If Step 1 produces an adequate surface-derived SAM latent `z_surface`, preserve its 16³ spatial correspondence and inject it into the existing shape hidden states with a trainable, zero-initialized residual projection. The noisy target state and time conditioning remain as before. Use the same oracle-aligned 32-object/four-view setup as the prior plan, and matched object exposures, fresh-noise assessments and optimizer checks.

The residual projection is trained by the existing Stage-1 shape flow objective. At initialization the new branch should reproduce the baseline because its output is zero; verify gradients reach it. No target latent is supplied during this surface-conditioning experiment. Do not claim a learned result from simply returning `z_surface`: its standalone accuracy is a separate no-training baseline.

First compare only:

1. Existing VecSetX + concatenated cross-attention conditioning.
2. Spatial residual conditioning derived from the actual surface, with the same backbone trainable scope.

This deliberately changes representation and access path together to test a plausible working solution rapidly. A win proves the **combined intervention** helps, not which component caused it. Save compute for the next ablation rather than making a causal claim from this comparison alone.

If the new route works, decompose it:

- Supply the same `z_surface` through a separate spatial cross-attention adapter (with explicit positions), versus direct aligned residual injection. State token count and parameter differences; do not call this an equal-capacity comparison unless they are controlled.
- Compare matched frozen versus trainable original representation/readout only as needed to ask whether the new representation or its access path is responsible.
- Remove or swap surface inputs under fixed image/noise to verify useful dependence. Surface-only versus visual-plus-surface training is then an informative test of visual interference in a route known to work.

If the spatial route fails **despite a good standalone surface latent**, use an exact-target-latent condition through the identical adapter as a deliberately privileged wiring/optimization positive control. This is target leakage by design and cannot count as task success. It tests whether that adapter can learn to exploit maximally informative spatial input. If even this fails, validate wiring, gradients and optimizer before training more of the backbone. If it succeeds, study sensitivity to errors in the surface-derived latent, rather than assuming the input is sufficiently accurate because its average score looks good.

The remaining requirement for production is camera-frame input. Oracle-frame success is a necessary localization result here, not completion of the task.

### 3. Use broader finetuning where the results call for it

The original broader-shape arm C can run concurrently on another GPU once the common baseline scale and memory profile are fixed. It answers whether the existing representation becomes useful with more adaptation. It should not delay the representation test or become a full dataset training run.

- Spatial route works under restricted training; original route works only with broader training: both are candidate fixes with different adaptation cost. Decompose and compare at matched input/exposure, then select the simpler reliable solution.
- Both improve: test whether combining them adds anything before paying for a large run.
- Neither improves: the exact-target spatial control and native reconstruction checks distinguish an invalid/weak signal from a pathway that cannot yet learn to exploit a strong signal. Return to the free-code/optimizer branch in the prior plan only if needed.

Do not demand that every exploratory variant fully converge. Use early checkpoints to discard invalid experiments and identify large learning differences; confirm the decisive contrast with sufficient exposure and another seed before a full run. An unresolved falling curve remains budget-limited, not an architectural impossibility result.

### 4. Restore the real frame and confirm before full training

Use the best demonstrated oracle-frame learner as the positive control. Test camera-frame surfaces under the same setup. Spatial injection requires an explicit mapping into the target grid; directly injecting camera-frame features as if aligned would introduce the very mismatch being investigated.

If camera input breaks the gain, evaluate supervised alignment, canonicalization or a representation/path that learns the transform, with known-pose alignment as the upper control. Any observed-frame target reformulation is a separate experiment with consistently re-encoded targets. No full training is justified until an end-to-end version using the intended available inputs shows a clear improvement on the moderate subset.

## Why this is a better order

It tests a potential solution early and uses that solution as a positive control. Each failure is interpretable in relation to the quality of its input signal; each success leads to a targeted ablation and then the real coordinate setting. It avoids spending the initial budget on four variants that all depend on the same possibly difficult representation interface. It also avoids pretending a representation-plus-fusion change alone identifies a unique cause.

## Status and resource use

Only the five-object input-coverage audit has executed. The representation test, spatial adapter and training comparisons are proposals requiring implementation and GPU validation. No new checkpoint or production behavior has changed. Use available GPUs for independent short branches once their dependencies are satisfied, not four copies of speculative full training. Future outputs stay under `outputs/conditioning_investigation/`.
