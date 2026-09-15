# Conditioning visuals

`render_conditioning.py` renders the latest individual camera-frame inputs with XYZ grids, common camera bounds and 5,000 opaque points per cloud. It also renders the target-frame mesh/surface and a saved latent slice (not a decoded reconstruction).

Run from the repository root:

```bash
python experiments/coordinate_system/scripts/presentation/render_conditioning.py ../003d24749f2e4047b95f9d04a7c7957b experiments/coordinate_system/outputs/slide_visuals/003d24749f2e4047b95f9d04a7c7957b/individual_inputs
```

Only the latest assets are retained in that output directory. Source object data and historical experiment results are separate.

## Normalization stages

Run `normalization_coordinates.py OBJECT_DIR OUTPUT_DIR` with CPU Torch, NumPy and Pillow, then `render_normalization.py OBJECT_DIR OUTPUT_DIR` with Matplotlib and trimesh. Current output is `experiments/coordinate_system/outputs/slide_visuals/003d24749f2e4047b95f9d04a7c7957b/normalization`. This reproduces the original non-oracle normalization path, independently of model weights. All three stages use common numeric bounds and 1-unit grid spacing. The raw-only individual_inputs folder retains its closer view.

## Object-axis triads and target voxels

The 3D geometry plots have no triad. `render_triads.py` exports separate square boxed triads for views 004–006 and the target. In input markers the target basis is transformed by the saved object-to-SAM-camera rotation; normalization does not rotate it. The display marker labels object -Y as Y to point toward the octopus's face. Grid axes in the geometry plots retain their native +Y direction. Visual inspection of the textured mesh confirms +Z up and the face toward the -Y side.

`render_target_voxels.py OBJECT_DIR OUTPUT_DIR` reruns the actual data-generation voxelization function (Open3D, CPU) and renders the 64³ surface occupancy. The target output folder includes both the matching input-plot viewpoint and clearly named upright viewing references. These are encoder-input voxels, not decoded latent predictions.

## Axis labels versus viewing camera (2026-09-15)

The input plots pass raw XYZ coordinates directly to scatter, use X/Y/Z labels on the matching numeric dimensions, and originally set Matplotlib vertical_axis='y'. The input plots were regenerated with vertical_axis='z' on September 15 to match the target viewing camera. Upright target mesh/voxel plots pass unchanged object XYZ geometry and set vertical_axis='z'. This changes the viewing camera, not the arrays or labels. Camera Z appears along the screen's horizontal grid edge in the input plots; its physical meaning remains camera depth.

The boxed input triads show the target/object basis transformed by T_sam_from_object, not the camera grid basis. Object +Z maps to SAM XYZ [0, 0.987725, -0.156205] in view 005, so its blue Z arrow points close to camera +Y. Target triads show the untransformed object basis. The old green-arrow presentation convention additionally reverses object Y; *_plus_y markers instead show native +Y.

Consequently grid and triad letters should not be made to match by relabeling: they refer to different frames on the input figures. Use explicit slide captions: input grid = camera coordinates; input inset = object axes expressed in camera coordinates; target grid/inset = object coordinates. This is visualization/convention clarification, not evidence of a training defect or proof that oracle is needed.

## Common plotting viewpoint, 2026-09-15

The shared plot function now defaults to Z-up at elevation 12 and azimuth -65, matching the upright target and rotation figures. Raw inputs and both normalization stages were rerendered using existing coordinates, colors, point selection and bounds. Raw inputs remain in camera XYZ; target figures remain in object XYZ. A common plotting camera standardizes the axis layout without numerically aligning these frames. Existing standalone input triads were not regenerated and use the earlier Y-up plotting view; they must not be overlaid as matching-view markers on the new Z-up plots.
