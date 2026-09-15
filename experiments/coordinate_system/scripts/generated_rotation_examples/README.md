# Actual generated predictions: shape and orientation

Use one existing camera-surface checkpoint. No optimization, Stage 2, W&B or jobs.
Run these commands sequentially from the repository root on one GPU.

## Find candidates (GPU inference, no decoding)

```bash
python experiments/coordinate_system/scripts/generated_rotation_examples/find_generated_examples_gpu.py \
  --run-dir outputs/stage1_full_surface_no_pointmap_full_cross_attention \
  --output-dir experiments/coordinate_system/outputs/generated_rotation_examples/find
```

Reuses `rotation_loss/evaluate_stage1_latents_gpu.py`: restore best.pt and its saved
conditioning, generate from pure noise at 25 steps / CFG 0, and score unchanged
predictions against the stored training target and six exact rotated target means
(X/Y/Z at 90/180 degrees). Only the ground-truth target references are re-encoded.
Target means never initialize generation. Native velocity probes are skipped here.

Defaults: 32 validation objects × 4 metadata-selected views × 2 draws = 256
predictions. Ranking is CPU by absolute original-minus-best-target latent MSE.
Save the top 12, at most two per object. These may be two seeds of one view;
view consistency is not guaranteed by selection. Identity-best cases have zero
reduction and can remain if too few nonidentity cases exist. No candidate is
declared a correct shape. Counts follow the existing grouped-by-four selector.

All scores are in endpoint_scores.csv. candidates/candidates.json records the
shortlist and total nonidentity-preferred count. Each candidate NPZ contains only
prediction, original target and best rotated target: 384 KiB uncompressed each,
4.5 MiB for 12. No bulk latent bank, checkpoint copies or archives.

## Decode only the shortlist and visualize (GPU decoding)

```bash
python experiments/coordinate_system/scripts/generated_rotation_examples/visualize_generated_examples.py \
  --find-dir experiments/coordinate_system/outputs/generated_rotation_examples/find \
  --output-dir experiments/coordinate_system/outputs/generated_rotation_examples/visualizations
```

Reuses the production checkpoint loader and existing `decode_support` function.
Only the frozen Stage-1 decoder loads. Decode the selected predictions, original
targets and rotated targets, using FP32. For target rotation R selected by latent
MSE, apply R inverse to the decoded prediction. Exact occupancy permutations:
no interpolation, clipping, translation fitting, scale changes or ICP.

Each comparison PNG/PDF shows decoded original target, actual generated prediction
over target, and inverse-rotated prediction over target. Both latent scores refer
to the **unchanged generated latent**; there is no “rotated predicted latent.”
Scores are not observed native training-time velocity losses.

geometry_results.json reports raw/aligned geometric distances and F-scores at
one/two voxels. The decoded rotated-target inverse versus decoded original-target
control reports decoder non-equivariance, so decoder artifacts are not silently
treated as generator orientation errors. No universal “good shape” threshold is
assumed: inspect residuals, reference control and selected-case scores together.

Decoded grids are small and retained separately. After copying find/candidates and
visualizations locally, regenerate figures on CPU using the same visualization
command with `--render-only` (no models or dataset needed).

This is a latent-selected shortlist over seven orientations, not an exhaustive
search for geometric matches. Report the full bank count and selection rule.
If alignment still leaves poor shapes, do not present it as an orientation-only
failure. New runs can use another output directory or checkpoint via CLI flags.
