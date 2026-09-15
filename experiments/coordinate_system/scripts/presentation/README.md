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
