# F18 continuation: does surface utility produce accurate Stage-1 shapes?

Status: implemented under the investigation folder only. Four CPU noise-only sampling/metric/completeness tests pass; imports/CLI, Python syntax and shell syntax pass. **Full SAM3D execution needs the user's GPU and has not run locally.**

## Why this is now the next branch

F18 established useful high-noise prediction from object-specific surfaces on held identities, but native loss mostly cancels that advantage. The planned positive-dependence branch asks whether the benefit survives free generation. Neither increasing training nor changing coordinate recovery is justified before answering this. This is a checkpoint diagnostic, not a new training recipe.

Use exactly the F18 oracle/constant last checkpoints, step14,660; the same16 train and16 validation objects, two views each; the same observation preprocessing and surface tokens. Reference the existing cluster probe folder `outputs/conditioning_investigation/full_checkpoint_probe_20260913_133239`. Source/checkpoint/config/data hashes are checked before sampling, and every prepared input/token tensor hash must replay the prior probe. Keep the old probe scripts unchanged because their hashes are part of this replay.

## What runs

- Generate from pure Gaussian noise, using the original Stage-1 sampler with25 integration steps, no-shortcut/d=0 and CFG strength0. These match the earlier successful diagnostic sampling convention. This is one sampler setting, not an assertion that other guidance/step counts cannot improve results.
- Oracle: correct surface versus the previously fixed cyclic wrong surface (shift1) × visuals present/zero. Constant: its saved bank × visuals present/zero. Two paired noise draws per observation. Target values are not passed to `sample_from_noise`; it receives only initial noise, visual tokens and surface tokens.
-768 sampled outputs in total:512 oracle and256 constant. Each arm uses one GPU; two or more visible GPUs run concurrently, one runs sequentially. No wall-clock estimate has been validated on the cluster.
- Load only the Stage-1 sparse-structure decoder to convert target/predicted latents into64³ occupancy. No Stage2, mesh generation, pruning, CD or retraining. The target latent supplies the fidelity reference, not the generated shape.
- Record raw occupancy IoU and bidirectional proximity within two grid voxels; save predicted final latents, target latents, supports, sample IDs, parameter/source/input/noise/file hashes. Empty predictions count as zero fidelity; empty decoded targets stop the probe as invalid references.

Raw-frame agreement is not the user's ultimate shape requirement. Saved supports allow local proper-rigid analysis using the established investigation code if raw fidelity is low. Do not classify a rotated correct shape as intrinsic-shape failure based on the first printed summary alone.

## Endpoints and decisions

The raw diagnostic endpoint is retained from the earlier full-data upper-bound analysis: mean object IoU≥.95, and the element at floor(.1*N) in sorted object IoUs≥.90 (the second-lowest object forN=16). Views/draws are averaged within objects first. This endpoint checks target-frame support fidelity; failing it is not proof of shape failure or a fundamental limitation. Also report per-object precision/recall at two voxels and correct-minus-wrong/constant differences. No new significance threshold or success label replaces the existing endpoint.

- Accurate training and held-object shapes plus useful surface dependence: establish the upper bound at that measured scope, then compare the camera checkpoint before attributing a remaining frame-treatment cost.
- Accurate training shapes, poor held-object shapes even after appropriate pose analysis: focus on transfer of aligned conditioning.
- Poor training shapes: focus on fitting/optimization/interface under oracle, keeping sampler adequacy as a live alternative. A loss benefit at one denoising time is insufficient evidence that the full pathway was fitted.
- Correct and constant outputs similarly accurate: generation works, but incremental surface utility is not established by the comparison.
- Strong wrong-surface penalty alone is insufficient: wrong inputs can be harmful. Absolute target-referenced fidelity and the constant comparison remain required.
- Raw failure but strong pose-adjusted fidelity: separate an orientation error from a shape error before selecting a training change.

## Sync and run

Sync new `checkpoint_rollout_gpu.py`, `checkpoint_rollout_core.py`, `analyze_checkpoint_rollouts.py`, and `run_checkpoint_rollouts.sh` into this directory. They reuse the unchanged existing `checkpoint_probe_gpu.py`, `checkpoint_probe_core.py` and `protocol.py` from F18. No production-source changes are needed. Keep the completed cluster probe folder and both original run directories.

In an interactive GPU allocation with `sam3d-objects` active, from the repository root:

```bash
bash doc/conditioning_investigation_2026-09-11/oracle_upper_bound/run_checkpoint_rollouts.sh
```

The complete shell contents are also given in chat for direct pasting. The wrapper creates a timestamped output folder, waits for both arms, verifies hashes/pairing/completeness, writes the raw summary, and creates a ZIP with JSON and NPZ evidence (no model checkpoints). Return that ZIP. Latents are included to preserve follow-up options, so this bundle is larger than the previous loss-only bundle. If a job fails, return its traceback and partial JSON rather than restarting training.

## Local checks and limits

CPU tests establish that the sampler helper supplies only tensor shapes and cloned starting noise, never target values; interventions preserve the noise bank; identity/empty/translated support metrics behave correctly; summary signs/endpoints are correct; and missing/duplicate/unpaired cells are rejected. Tests use a synthetic generator. `checkpoint_rollout_gpu.py --help` imports successfully in the temporary CPU environment. Production checkpoint/decoder loading and sampling remain untested until this cluster run. The analyzer checks NPZ file hashes and report pairing; independent recomputation from returned supports remains part of the local results audit.
