# Actual generated latent: rotation-fit and re-encode diagnostic

This experiment asks whether a prediction produced in the unchanged validation
setup contains a close decoded shape at the wrong orientation, and whether
correcting that orientation reduces latent error after a controlled round trip.

It does not rotate an input or target, apply the camera oracle, search a fixed
list of target rotations, or change generation. It replaces both earlier
generated-example diagnostics in this folder.

## 1. Measure every fixed example on one GPU

```bash
python experiments/coordinate_system/scripts/generated_rotation_examples/measure_rotation_fit_gpu.py \
  --run-dir outputs/stage1_full_surface_no_pointmap_full_cross_attention \
  --output-dir experiments/coordinate_system/outputs/generated_rotation_examples/rotation_fit
```

Defaults: 32 validation objects × four metadata-selected views × two draws =
256 predictions. All are retained. Restore `best.pt` and its saved conditioning,
then generate from pure noise using the existing 25-step CFG-0 sampling path.
Copy each actual input image plus compact raw full-surface/pointmap clouds. The
pointmap is labelled as a reference when the checkpoint disables it.

For each unchanged generated latent and stored target latent:

1. Decode both with the frozen Stage-1 decoder.
2. Fit a continuous proper rotation from predicted occupied-voxel centers to
   target centers, about the fixed target-grid origin.
3. Do not fit translation, scale, reflection, camera pose or deformation.
4. Rasterize unaligned and aligned predicted centers into the same fixed 64³
   grid. Identity rasterization must exactly recover the decoded prediction;
   report any aligned points outside the cube.
5. Re-encode both grids with the same frozen SS target encoder.
6. Compare both re-encoded latents with the unchanged stored target.
7. Decode and re-encode the target itself to measure the round-trip floor.

`results.json` reports all geometric and latent measurements. Each small NPZ
retains the generated/stored latents, decoded grids, aligned grid/points and fitted
rotation. The original generated-to-target MSE remains separate from round-trip
comparisons. No Stage 2, training, W&B, job script or bulk feature bank.

## 2. Find examples on CPU

```bash
python experiments/coordinate_system/scripts/generated_rotation_examples/select_examples.py \
  --results experiments/coordinate_system/outputs/generated_rotation_examples/rotation_fit/results.json \
  --output experiments/coordinate_system/outputs/generated_rotation_examples/rotation_fit/selected.json
```

This summarizes the entire fixed bank, then selects up to six post-hoc examples
where raw F@2 is at most 0.6, rotation raises F@2 by at least 0.2, aligned
precision and recall are both at least 0.8, re-encoded latent MSE falls by at
least 10%, rotation is at least five degrees, and at most 1% of aligned points
leave the target cube. Ranking favors aligned precision/recall, then geometric
and latent gains, with at most two examples per object. `selected.json` includes
every row and the exact rule. Selection demonstrates existence, not prevalence.

## 3. Render the selected examples on CPU

```bash
python experiments/coordinate_system/scripts/generated_rotation_examples/visualize_rotation_fit.py \
  --measurement-dir experiments/coordinate_system/outputs/generated_rotation_examples/rotation_fit \
  --selection experiments/coordinate_system/outputs/generated_rotation_examples/rotation_fit/selected.json \
  --output-dir experiments/coordinate_system/outputs/generated_rotation_examples/rotation_fit_figures
```

Each figure shows the real validation image, decoded stored target, actual raw
prediction over target, and rotation-fitted prediction over target. Captions give
the original generated latent MSE, unaligned/aligned re-encoded MSE, target
round-trip floor, fitted angle and decoded F@2. Use `--all` to render all 24.

The aligned display uses the rasterized grid that was actually re-encoded. All
output geometry uses native XYZ with fixed bounds per comparison. Input clouds
use the documented display-only camera ZXY convention. Model arrays are unchanged.

## Interpretation

A useful example requires both close geometry after rotation and aligned
re-encoded MSE below the unaligned round-trip MSE. Compare the aligned MSE with
the target round-trip floor; it need not reach zero because decoding/re-encoding
is lossy. This supports an orientation component in that prediction's error. It
does not show that its raw latent can be rotated directly, that oracle conditioning
fixes training, or that orientation explains other predictions.

Old `find/`, `visualizations/`, `generation/`, and
`camera_inverse_visualizations/` outputs are obsolete and are not consumed.
