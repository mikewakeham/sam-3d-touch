# Rotation penalty in SS target latents

Two independent GPU commands and one CPU plotting command. No optimization,
Stage 2, decoder, W&B or submission jobs. `evaluate.py` remains the separate
Stage-2/CD evaluator. Run from the repository root in the existing environment.

## 1. Same shape, different target orientation

```bash
python experiments/coordinate_system/scripts/rotation_loss/run_rotation_loss_gpu.py \
  --example-object 003d24749f2e4047b95f9d04a7c7957b \
  --output-dir experiments/coordinate_system/outputs/rotation_loss/encoder
```

Defaults: 16 training and 16 validation identities, metadata-selected using the
existing `evaluation_groups` helper and seed 29. A separate declared example
outside that bank is labeled `example` and excluded from population summaries.

For each mesh, apply its saved object transform and prepare the actual native
64³ surface occupancy. Re-encode identity and exact 90°/180° rotations about each
axis. These are voxel permutations; inverse rotation restores the input exactly.
Check the zero encoding against the stored training target and repeat it to
measure the numerical floor. MSE is calculated on the SS means, not decodings.

For the angle curve, scale once to a radius-0.45 sphere, then rotate the same
geometry by 0°, 5°, 15°, 30°, 60°, 90°, 180° about each axis. This fixed padding
prevents cropping. Never re-normalize per rotation. Padded means are compared
with their padded zero and kept separate from native-target comparisons.

The geometry/voxelizer/encoder functions in `rotation_utils.py` are small copies
of `generate_target_latents.py`; the original file is ignored by Git. Exact grid
rotations and continuous rotation matrices reuse existing experiment helpers.

Outputs: `measurements.csv`, `results.json`, and compressed input occupancies for
the declared example. No latent bank is saved. The analytical velocity columns
describe an otherwise exact rotated endpoint at fixed noisy input:
`MSE(z_rotated,z0) / [1-(1-sigma_min)*t]^2`. They are not observed model losses.

## 2. Actual checkpoint loss and endpoint scores

After the oracle/no-pointmap run completes:

```bash
python experiments/coordinate_system/scripts/rotation_loss/evaluate_stage1_latents_gpu.py \
  --run-dirs \
    outputs/stage1_full_surface_no_pointmap_full_cross_attention \
    outputs/stage1_full_surface_oracle_no_pointmap \
    outputs/stage1_image_no_pointmap_full_cross_attention \
  --output-dir experiments/coordinate_system/outputs/rotation_loss/oracle_checkpoints
```

No encoder preparation output is required. This independently selects the same
identities from the data config, encodes seven native reference orientations in
memory, and restores each run's `best.pt` and saved conditioning mode.

Defaults: two views per object, two sampled endpoints per view, 25 flow steps,
CFG 0 (conditional-only, matching earlier Stage-1 probes), BF16 model inference.
No target values are used to initialize sampling. The same addressed initial
noise is used for each model. SS reference encoding is FP32.

Native velocity loss calls the existing generator objective through `paired_loss`.
Reuse the previous noise/time bank: four native draws plus two draws at each of
t=0.05/0.5/0.95. Native-time averages and fixed-time results remain separate.
Visual dropout is off during assessment; original training dropout is recorded.

Each generated endpoint stays fixed. Score it against the original target and
each separately encoded native rotated reference. A smaller rotated-reference
MSE suggests an orientation component in its latent mismatch, not perfect shape
or oracle necessity. A minimum over seven rotations is a diagnostic, not complete
continuous alignment. Endpoint latent MSE is not native velocity training loss.

The separate normalization comparison, once both shared runs finish:

```bash
python experiments/coordinate_system/scripts/rotation_loss/evaluate_stage1_latents_gpu.py \
  --run-dirs \
    outputs/stage1_image_full_cross_attention \
    outputs/stage1_full_surface_full_cross_attention \
    outputs/stage1_image_shared_normalization \
    outputs/stage1_full_surface_shared_normalization \
  --wrong-surface \
  --output-dir experiments/coordinate_system/outputs/rotation_loss/normalization_checkpoints
```

`--wrong-surface` adds only a native-loss token swap. The already processed
pointmap, correct-surface normalization statistics, images, target and noise stay
fixed. It does not generate extra wrong-surface rollouts. Keep this normalization
table separate from the oracle comparison.

Outputs: `velocity_losses.csv`, `endpoint_scores.csv`, `results.json`, and one
small prediction/reference latent example per model. No optimizer, full feature
cache, bulk prediction bank or ZIP is written. CSVs flush after every group.

For a quick cluster smoke test, use one run directory and add:
`--train-objects 4 --val-objects 0 --views 1 --draws 1`.
Object counts follow the reused selection helper: at least four in any included
split, grouped in fours. Counts that are not multiples of four are truncated.

## 3. Figures and object-level numbers

```bash
python experiments/coordinate_system/scripts/rotation_loss/plot_rotation_loss.py \
  --rotation-dir experiments/coordinate_system/outputs/rotation_loss/encoder \
  --checkpoint-dir experiments/coordinate_system/outputs/rotation_loss/oracle_checkpoints \
  --output-dir experiments/coordinate_system/outputs/rotation_loss/oracle_figures
```

Either input directory can be omitted. Plot normalization checkpoints into their
own figure directory. Nothing requires GPU for plotting.

`summary.json` contains native rotation controls, angle-curve numbers, numerical
floors, and per-checkpoint native/endpoint scores with object-bootstrap 95% CIs.
Views/draws are averaged within object first. Paired model differences use the
first listed model as baseline; negative candidate-minus-baseline means lower MSE.

Slide figures: same input → exact Z90 rotation → known inverse, with **encoded
mean MSE** printed; angle-versus-target-MSE curves; observed native velocity-loss
bars; original/best-tested-reference endpoint-score bars. Existing point-cloud
plot styling is reused, without changing accepted input figures. The figures
show actual encoder inputs, not decoded outputs. No decoded F-score motivates
oracle in this experiment.

All outputs stay in the ignored experiment outputs directory. Retained laptop
copies belong in `../coordinate_system_results/rotation_loss/`. Source scripts
are under this folder; no production source is changed.

### High-error validation examples

After the encoder probe, export the three distinct validation objects with the
largest native 90/180-degree latent penalties, then run the existing plotter:

```bash
python experiments/coordinate_system/scripts/rotation_loss/export_high_error_examples.py \
  --probe-dir experiments/coordinate_system/outputs/rotation_loss/encoder_val64 \
  --top 3
python experiments/coordinate_system/scripts/rotation_loss/plot_rotation_loss.py \
  --rotation-dir experiments/coordinate_system/outputs/rotation_loss/encoder_val64 \
  --output-dir experiments/coordinate_system/outputs/rotation_loss/encoder_val64_figures
```

These commands need the original dataset but no GPU/model weights. Small boolean
grids go in `examples/`; figures use the actual axis/angle and measured GPU MSE.
`high_error_examples.json` records the selection. These are explicitly selected
extremes, not representative objects. Native rotations preserve geometry exactly;
the angle chart instead uses the separate padded-scale protocol.

## Verification

```bash
python -m unittest experiments.coordinate_system.scripts.rotation_loss.test_rotation_loss
```

CPU checks cover physical grid rotations, fixed endpoint scoring, object-weighted
statistics, and end-to-end disk/geometry/plotting with an explicitly fake spatial
encoder. They do not test pretrained model behavior. A cluster smoke test is
required before interpreting a full checkpoint report.
