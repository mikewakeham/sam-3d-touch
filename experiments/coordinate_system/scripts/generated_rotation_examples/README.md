# Actual validation views: known camera inverse

This replaces the rotated-target shortlist experiment. Select validation views
from metadata before generation and retain every result. No synthetic rotated
targets, target re-encoding, rotation search or score-based selection.

Run from the repository root on one GPU:

```bash
python experiments/coordinate_system/scripts/generated_rotation_examples/generate_examples_gpu.py \
  --run-dir outputs/stage1_full_surface_no_pointmap_full_cross_attention \
  --output-dir experiments/coordinate_system/outputs/generated_rotation_examples/generation

python experiments/coordinate_system/scripts/generated_rotation_examples/visualize_generated_examples.py \
  --generation-dir experiments/coordinate_system/outputs/generated_rotation_examples/generation \
  --output-dir experiments/coordinate_system/outputs/generated_rotation_examples/camera_inverse_visualizations
```

Generation defaults: eight validation objects × three views × one noise draw =
24 predictions. CLI flags change these counts; object selection reuses the existing
grouped-by-four metadata helper. Restore best.pt with saved conditioning, sample
from pure noise at 25 steps / CFG 0 using existing sampler/prepare_batch helpers.
No target values initialize sampling. Measure latent MSE only against the stored
training target. Each NPZ contains actual predicted latent, stored target latent,
and saved SAM-camera-from-object matrix. Raw RGB images and small raw-input clouds
are saved per view. Generation does not load the target encoder or decoder.

Visualization loads only the frozen Stage-1 decoder and decodes prediction and
stored target. Apply the saved camera matrix's inverse 3x3 block to predicted
voxel-center coordinates about origin. Do not apply camera translation to the
already centered shape output. Do not fit scale, translation or any rotation.
The saved matrix orthogonality error is reported; the matrix is not fitted or
silently replaced. Target points stay unchanged. This tests specifically whether
prediction retained the camera orientation, not arbitrary orientation mismatch.

Per-view inputs folder: input_view.png, surface.png, pointmap.png,
surface_and_pointmap.png, and inputs.png/PDF. Raw pointmap is labelled reference
only when disabled by the checkpoint. Raw input plots use the existing display-only
camera ZXY → plotted XYZ convention; model arrays are unchanged. These are raw
inputs before preprocessing, not final encoder tokens.

Per-prediction comparison.png/PDF includes input image, decoded training target,
raw predicted shape overlay, and known-inverse-rotated prediction overlay. All
three output geometry plots share native XYZ axes, bounds and viewpoint. Geometry
metrics use all occupied points, while visualization may thin display points.
Latent MSE refers only to original, unmodified generated latent versus stored target.
There is no latent MSE claimed for the rotated geometry.

examples.json and geometry_results.json report every selected example, including
failures. No Stage 2, training, W&B, job submission, archives or checkpoint copies.
To redraw on CPU after copying the generation inputs/report and visualization
decoded folder, use the visualization command with --render-only.

Old outputs under find/ and visualizations/ belong to the discarded synthetic
rotated-reference experiment. The new commands do not consume them. Delete those
two old output directories on the cluster if they are no longer needed.
