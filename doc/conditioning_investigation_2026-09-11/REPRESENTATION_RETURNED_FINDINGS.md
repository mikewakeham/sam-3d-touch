# Representation probe returned: retain the point representation branch

## Evidence and validation

The completed attachment is archived byte-for-byte in `representation_returned_46083371/results.json`. The unchanged prespecified analyzer produced `analysis.json`; `validation.json` records the attachment digest and checks. All eight reported source hashes match local source. The report has 32 distinct objects and both camera/oracle VecSetX rows for each. Original target regeneration and native VecSetX forward-composition checks have maximum error **0.0**. All native zero surfaces were extracted, with no reported query-boundary contacts.

The JSON is complete. Saved NPZ fields/meshes and the Slurm log were not supplied; those measurements have not been independently recomputed. Checkpoint hashes are recorded but cluster weights are unavailable locally. Results use one selected view (015) per object, not a rotation sweep or a sparse-touch experiment.

| Measurement | Mean | Median | Minimum |
|---|---:|---:|---:|
| Native VecSetX surface F-score, oracle frame | .96536 | .98695 | .80216 |
| Native VecSetX surface F-score, camera frame | .96956 | .99156 | .80549 |
| Raw point-grid IoU against decoded target | .61697 | .60153 | .26260 |
| SAM-decoded point-grid IoU against decoded target | .62455 | .61315 | .27847 |
| Target VAE decode IoU against original mesh voxelization | .999995 | 1.0 | .999879 |

The native F-score uses surface precision/recall within 1/64 object units. It is neither exact reconstruction nor interchangeable with occupied-voxel IoU. VecSetX oracle/camera means for precision are .98488/.98506 and recall .94918/.95667. Nine objects in each frame fall below .95 F-score; the shield is approximately .80 in both. Native decoding uses a 64-cell implicit-field grid, which can miss thin details. A poor individual case needs artifact inspection or a resolution check before attributing all error to encoding.

Camera-frame native fidelity is not systematically worse in this sample: it is higher for 19/32 objects, with individual camera-minus-oracle differences from −.05274 to +.06920. This does **not** prove rotation-equivariant features. It shows the native decoder can reconstruct substantial geometry from both tested orientations; SAM3D's ability to interpret those features remains a different question.

The SAM VAE improves raw point-grid IoU by only **.007585** on average. Mean surface-derived target latent MSE is **.48135**, versus **.17515** for the scoring-only leave-one-object-out average-target control. Surface-derived latents beat that control on only 7/32 objects and identify their own target as nearest among these 32 on 20/32. These controls do not make the generic latent a useful reconstruction, but reject the assumption that point-grid encoding consistently approximates the target latent already.

## Decision

Do not implement the proposed native-SAM spatial adapter on the premise that its point-grid input is a high-quality approximate target. That premise failed this test. This rejects the **naive point-hit grid plus frozen VAE** candidate, not every spatial representation or learned sparse encoder.

Retain VecSetX as a viable representation rather than moving to scratch encoding. Its native decoder successfully uses the bottleneck, learned slot embeddings and subsequent transformer computation to reconstruct much of the full geometry. The current SAM3D adapter exposes raw bottleneck tokens through a shared projector; the native decoder's success does not imply the adapter can learn the same mapping easily. Missing slot identity alone remains unproven as the cause, especially given the earlier `learn` variant's similar dataset results.

This is the previously specified branch “VecSetX reconstructs well, point-derived SAM latent is poor”: investigate access/translation of existing features and adaptation capacity. A direct spatial-query readout of VecSetX features to the Stage-1 latent is an appropriate diagnostic of accessibility; it should compare raw and native processed features under matched readout and optimizer settings. It is not a production sparse-touch reconstruction objective: full-shape direct regression from ambiguous sparse evidence can average possible shapes.

Before further implementation, distinguish two uses of that readout: (1) a diagnostic bypass of SAM3D's flow network to test whether frozen features can predict target latents on fitted and distinct objects; (2) a possible learned conditioner supplying features to the existing image-conditioned flow model. A bypass success earns the second experiment, not an immediate replacement of generative training. A bypass failure does not establish information loss; broader shape adaptation remains a competing controlled intervention.

## Sparse touches are the endpoint

The user clarified that full-surface learning is the first test; the final condition consists of sparse, structured touch patches. All proposed fixes must retain a credible path to those inputs.

Source inspection establishes an exact limitation of the **no-position** encoder branch. For positive scalar a and translation b, its normalization obeys N(aP+b)=N(P). The raw feature encoder and shared projector therefore cannot distinguish translated/scaled copies of the same observed patch (ignoring numerical differences). In `forward`, shifts/scales are supplied to the separate position projection only when `use_position=True`. The existing default position branch is consequently different from the no-position full-surface experiments; do not claim all touch runs omit location.

The loader correctly places local touch points in the camera frame using each contact's rotation and center, concatenates them, and carries a padding mask. Subsequent bbox/radius normalization removes the cloud's global shift/scale unless that information is conveyed separately. Multiple patches retain relative configuration after normalization, but their absolute placement is not recoverable from normalized geometry alone. The loader does not preserve explicit contact IDs as a model input; geometric point positions can still represent patch structure, so this omission is not automatically a bug.

This mathematical indistinguishability is a real architectural limitation of that particular conditioning branch. It is not evidence of collisions between actual dataset records, nor a sole explanation of the full-surface results. Full surfaces admit useful object-centric normalization; an isolated local patch does not define an object's center or overall extent.

Requirements for a sparse-compatible conditioner:

- Preserve observed locations in a shared image/pointmap frame, or pass the normalization transform explicitly. For a spatial branch, retain per-point/location features rather than assuming one normalized local patch spans the object.
- Treat missing measurements as unknown, not empty. Carry valid-point/observation masks; never use GT occupied cells to fill observations.
- Keep image/prior information available for unobserved structure. Full-surface reconstruction is the diagnostic upper-information case, not the expected sparse-input behavior.
- After a full-surface improvement, test separate changes in point count and coverage: global thinning, then the actual geodesic touch-patch sampling with matched point counts. Do not substitute thinning for touch structure.
- Test increasing contact counts, held-out contact patterns, empty-touch behavior and wrong-location interventions. A sparse model should respond to evidence at its actual location; unobserved regions need not be forced to match one deterministic completion.
- Use a learned alignment or supplied real-world pose only when it is available in the intended input setup. Oracle object transforms remain diagnostic controls.

No production code or weights changed. The raw-grid candidate is deprioritized; the Stage-1 access/translation investigation remains open. No new GPU job is requested by this report.
