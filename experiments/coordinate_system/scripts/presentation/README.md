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

## Latest display convention (2026-09-15)

Input figures retain the upright appearance used in the slides. They display camera (Z,X,Y) as plot (X,Y,Z), with the same Z-up plotting camera as target figures. This cyclic display-only conversion preserves the original screen projection to numerical precision (tested maximum difference 6.8e-17) while putting grid X along the horizontal edge, Y along the receding edge and Z upward. Colors, point selection, numeric values and model inputs are unchanged.

Grid labels denote this display frame, not native SAM camera axes: plot X = camera Z, plot Y = camera X, plot Z = camera Y. Use a slide footnote “Camera inputs shown in a Z-up display convention.” Target plots retain native object XYZ. This is not oracle alignment or evidence of a training defect.

The original input triad files 004.png/005.png/006.png (and their *_plus_y.png variants) again match the restored screen orientation. They show object axes, not grid axes. The intermediate *_z_up.png input triads do not match this latest converted display. Target uses target_plus_y_z_up.png.

## Shared normalization overlay

`render_shared_normalization.py OBJECT_DIR NORMALIZATION_DIR` writes shared/004–006/surface_and_pointmap.png. It reuses SurfacePointmapNormalizer's actual initializer/normalize methods and TouchEncoder's normalization method without loading model weights. Both clouds use the same raw-surface bounding-box midpoint and maximum Euclidean radius. It keeps the existing normalized-stage plot bounds and latest display convention. The overlay shows geometric inputs before crop/resize/token embedding, not the final model tokens.

## Inverse-transform figures

`render_oracle.py --object-dir OBJECT_DIR --rotation-dir ROTATION_PROBE_DIR --output-dir OUTPUT_DIR --view 005` makes two separate figures. `inverse_rotation_latent_mse.png` uses saved exact Z90 occupancy arrays and cluster-measured frozen Stage-1 encoder scores (octopus: 0 → 0.358552 → 0). Restored occupancy is exactly identical to the original; its score uses the measured repeat-encoding floor. This is a controlled rotation example, not an observed generator prediction or the actual camera rotation.

`camera_to_object_oracle.png` compares blue camera surface against green target-mesh surface samples before and after the inverse transform, then shows the target alone. The before panel subtracts only camera translation so both clouds share an origin; camera rotation is retained. All three panels use native XYZ, identical bounds and viewpoint, with no camera display permutation. This isolates orientation and differs from the upright display convention of earlier input figures. It precedes VecSetX normalization. No camera-view latent MSE has been measured by this rendering script. Figures and their numerical metadata live under ignored `outputs/slide_visuals/OBJECT_ID/oracle/`.
