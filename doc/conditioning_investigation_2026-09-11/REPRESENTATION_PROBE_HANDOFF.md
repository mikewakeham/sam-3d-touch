# First representation experiment: ready for cluster execution

## Run

From the cluster repository root, with the investigation folder transferred:

```bash
sbatch doc/conditioning_investigation_2026-09-11/run_representation_probe.sh
```

New required files: `probe_representations_gpu.py`, `representation_geometry.py`, `test_representation_geometry.py`, `analyze_representations.py`, and `run_representation_probe.sh`. Existing production code is imported, not modified.

The launcher requests one H100 and a one-hour scheduler limit. This is a cap, not measured runtime. It runs no training and loads no diffusion model, image model or Stage 2. It uses 32 seeded training objects, one recorded view each. VecSetX is evaluated in both camera and oracle frames; SAM point-grid encoding uses oracle alignment only. The full object-level selection and inputs are logged. This is an in-distribution representation test, not a generalization estimate.

Default prerequisites are the existing full-surface data config, original meshes for independent scoring, SAM `ss_encoder.ckpt` / `ss_decoder.ckpt`, and the same VecSetX checkpoint path used in `TouchEncoder`. All have explicit CLI overrides except the dataset root, which comes from the config. scikit-image is already pinned in the repository requirements. Missing files fail preflight. Output directories must be new; a previous partial result is never silently overwritten.

Outputs: `outputs/conditioning_investigation/representation_probe/JOB_ID/`. Return `results.json` and the Slurm log initially. Keep the NPZs on the cluster. A failed job may leave `results.partial.json`; return that and the exception rather than treating it as a complete result.

## What is measured

### SAM native shape representation

1. Load the recorded surface and inverse camera transform.
2. Make a fixed 64³ binary point-hit grid, with no filling, target-assisted snapping or dilation.
3. Encode and decode it using the frozen SAM sparse-structure VAE.
4. Only then load targets and original meshes for scoring. Re-encode the original mesh to check that the stored target is reproduced within the established fp32 tolerance.
5. Record latent MSE, target latent energy, raw point-grid coverage, decoded surface/target occupancy agreement, and target VAE reconstruction against original-mesh occupancy.

Scoring-only latent controls compare each surface-derived code with other objects' targets and compare a leave-one-object-out mean target with the true target. These help distinguish object-specific accuracy from a low error achievable by a generic code. They are never supplied as conditioning inputs or reported as deployable predictors.

This establishes whether actual observed points can provide a useful SAM-format condition without learning a cross-model latent translation. An accurate VAE reconstruction of the sparse input is not sufficient: compare to the complete target too. No target geometry is used to repair the point grid.

### Native VecSetX reconstruction

Reuse the production `TouchEncoder.prepare_points` and bbox/radius normalization. Load the same pretrained VecSetX architecture and weights, bypassing only the downstream surface projector. Run `encode -> learn -> decode`, preserving the production encoder mask. A repeated native forward with the same mask must reproduce that composition.

Probe both the original camera surface and its oracle-frame inverse transform. Camera input omits pointmap SSI: the previously verified positive-isotropic-affine cancellation means the subsequent bbox/radius normalization is mathematically equivalent for this no-position path. This probe does not measure RGB/pointmap interactions. Different floating-point normalization orders can still produce small differences from the complete training pipeline.

The native decoder yields an implicit field, not SAM surface occupancy. Query a fixed 65³ xyz grid in [-1.05,1.05], extract its zero surface, and undo the input-derived normalization. For the camera arm, then inverse-transform to the object frame. Use independently sampled original-mesh points to measure surface precision/recall at one target voxel width (1/64); also log distance percentiles and query-boundary contact. Original-mesh self-sampling is a numerical calibration, not model performance. This is native representation decoding only, not Stage-2 evaluation.

The outer computation and weights use fp32; the existing VecSetX attention implementation internally casts Q/K/V to bf16. This is recorded rather than misreported as entirely fp32. The native-forward check passes the same mask because masked and unmasked attention use different kernels.

## How the result chooses the next action

| Pattern | Next action |
|---|---|
| Surface-derived SAM latent and decoded geometry are consistently close to targets | Build the spatial residual-conditioning prototype, starting with oracle alignment as the controlled positive setting. |
| VecSetX reconstructs well but point-derived SAM latents are poor | Do not build the voxel-based adapter yet. Test access/translation of the existing features or a learned spatial point encoder. |
| Oracle VecSetX good, camera VecSetX substantially worse | Rotation sensitivity already appears inside the frozen representation. Isolate this before attributing the entire problem to SAM3D fusion. |
| Both poor | Check sampling/preprocessing and extraction resolution. Native decoding failure alone does not prove information loss. |
| Mixed by object | Inspect saved fields, point coverage and latent errors; use the object-level pattern to choose a targeted follow-up, not an average-only verdict. |

No automatic “good” cutoff is selected after seeing results. Report continuous errors and whether gains are substantial relative to raw point occupancy and the reference VAE. VecSetX surface F-score and SAM occupancy IoU are different measurements and are not interchangeable model rankings. High native VecSetX fidelity does not establish that the current per-token projector/cross-attention can access the same information used by its native decoder.

The 64-cell native field is a screening resolution. Thin surfaces can disappear during field extraction even if the learned implicit function contains them. Before diagnosing an encoder failure in an ambiguous case, inspect the saved field and run a prespecified higher-resolution check on that case; do not attribute all extraction errors to the encoder. Near-boundary geometry is also flagged.

## Artifacts and validation

- `*_sam.npz`: point grid, surface-derived latent, stored target latent, both decoded occupancies and original mesh occupancy.
- `*_vecset_{oracle,camera}.npz`: native field, encoded features, normalization and extracted object-frame vertices/faces when a zero surface exists.
- JSON: source and checkpoint hashes, full input records/hashes, all object metrics, normalization checks, native-forward checks and elapsed time per case.

Local validation: Python compilation, shell syntax, point-grid axis/boundary checks, precision/recall direction checks and deterministic mesh-sampling checks. The local environment lacks torch, checkpoint weights and scikit-image; the isosurface test is explicitly skipped locally and runs as a cluster preflight. GPU neural checks remain unexecuted until this job runs.

For a completed result:

```bash
python doc/conditioning_investigation_2026-09-11/analyze_representations.py \
  outputs/conditioning_investigation/representation_probe/JOB_ID/results.json \
  --output outputs/conditioning_investigation/representation_probe/JOB_ID/analysis.json
```
