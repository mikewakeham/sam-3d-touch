# Governing scope: learned conditioning of observed touch patches

The immediate task is full-surface conditioning as an upper bound. Sparse structured contacts, normals and other physical measurements are future compatibility constraints, not features to implement now. The latest priority is coordinate diagnosis while retaining pretrained VecSetX; see `COORDINATE_DIAGNOSIS_RESYNC.md`. Do not let these future requirements trigger a premature encoder replacement.

The proposed integration of reconstructed full surfaces -> voxel grids -> SAM shape latents is withdrawn. The completed bridge experiment remains diagnostic evidence only. No bridge-conditioned training job was implemented or launched. A method requiring completion and voxelization before producing its condition bypasses the requested representation problem, even if its shape scores improve.

## Required representation contract

- Input: observed point positions, point validity and contact/patch grouping. Allow optional normals and physical measurement channels with explicit availability masks. Missing measurements are not zero-valued measurements.
- Retain spatial relationships and patch location when encoding local geometry. If a patch is centered or scaled, retain the transform/anchor needed to locate it. Specify the frame contract for points, vectors such as normals, and scalar measurements separately.
- Encode these observations into learned point/patch condition tokens. They need not equal a complete-object SAM target latent. Compatibility means the Stage-1 network can use the tokens and their spatial information through the conditioning interface.
- Full-surface and sparse structured-contact inputs must use this same representation path. Complete surfaces may establish a fitting upper bound; sparse clustered patches must constrain design decisions from the start. Random thinning alone is insufficient.
- The image/prior can infer unobserved shape. Do not require the touch conditioner to turn sparse observations into a completed mesh, occupancy grid, or complete-object latent before SAM can consume them.
- Extra modalities must enter before their per-point association is discarded. Merely attaching a global modality vector after an xyz-only whole-object encoder does not satisfy the intended extension.

## What remains supported by evidence

Native VecSetX can reconstruct full-surface geometry, and frozen features can be mapped to Stage-1 targets on fitted objects. Our tiny direct readout failed to transfer. Those observations do not prove that VecSetX must be replaced, that the original dataset lacks enough examples, or that full SAM finetuning is necessary. The frozen geometry bridge shows a different complete-shape translation route is possible; it does not validate a touch representation.

The next implementation must evaluate point/patch encoding and SAM fusion under this contract. Audit the current local-normalization and position paths before selecting the smallest justified change. Test coordinate handling with location-sensitive controls, and keep adaptation capacity as a separate hypothesis. Do not inherit the withdrawn bridge-versus-token training plan from older notes.

No new GPU job is requested by this scope correction.
