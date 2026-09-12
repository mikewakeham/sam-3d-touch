# Frozen spatial bridge succeeds as a representation control

The complete report covers 32 objects and 96 rows. Driver, source representation report and checkpoint hashes match; sample order matches the earlier readout split. All reported target-regeneration errors and saved-field-to-vertex errors are zero. Aggregates recomputed from per-object rows agree with the report. Raw bytes, validation and analysis are archived under `spatial_bridge_returned_manual/`. The GPU computation and NPZ arrays were not independently rerun locally.

| Representation | First 24 latent MSE | Last 8 latent MSE |
|---|---:|---:|
| Training-mean target baseline | .151084 | .211688 |
| Point-hit grid through SAM VAE | .460220 | .544751 |
| Native surface triangles through SAM VAE | .104930 | .156643 |
| Learned processed-feature readout, 8000 updates | .015138 | .308976 |

The frozen native-surface bridge has no fitting on either group. It beats the mean control on 29/32 objects (23/24 first group; 6/8 reserved), and point hits on 31/32 objects. Its reserved error is 26.0% below the mean baseline and 49.3% below the trained processed readout. One reserved object remains particularly poor (latent MSE .648265), so the average is not a guarantee of per-object success.

Native surface input and decoded occupancy overlaps are equal in all reported aggregate values: 82.23% first group and 80.13% reserved. The frozen VAE does not repair the remaining support discrepancy. These support metrics describe the Stage-1 representation control; no Stage-2 conclusion is involved.

## Interpretation

We now have a target-relevant spatial representation that works on objects without fitting a new translation head. This is stronger evidence for a constructive conditioning route than the previous readout's training-object fit. It supports testing explicit spatial conditioning in the actual Stage-1 model.

It does not isolate why the original conditioner underperforms. The bridge changes both the interface and geometric processing: native decoding completes a triangle surface, and the pretrained SAM encoder supplies the mapping into the target representation. This is not a controlled proof that coordinates alone, slot identity alone, or the existing cross-attention alone caused the original issue. It is also still oracle-aligned; it does not solve deployment-time alignment.

No global rotation patch is indicated. The geometry is already in the target object frame, and no extra rotation or recentering was needed. The remaining native geometry outside the object cube reaches .00852 units at worst and is clipped by the original target voxelizer. Native field resolution and reconstruction errors remain possible sources of the residual mismatch.

## One-voxel sensitivity

The prespecified +x shift of the true target shell produces mean latent MSE .09516 across all 32 objects, compared with .11786 for the native bridge. This establishes that a small spatial perturbation can cause a nontrivial direct latent error; it does not define an acceptable error threshold or prove the bridge's errors are only translations.

The shift crops 5076 occupied voxels in total over 14 objects, so the aggregate combines translation and boundary loss. On the 18 objects with no cropped voxels, shifted-shell latent MSE still averages .08346. Latent MSE should therefore be read alongside spatial fidelity, rather than treating a moderately large value as proof that geometry was discarded. This control does not characterize flow loss, all displacement directions, or rotations.

## Next Stage-1 intervention

The previously proposed bridge-versus-token Stage-1 training comparison is withdrawn following the user's scope correction. It was not implemented or launched. The completed bridge is a diagnostic control, not the proposed conditioner. See [TOUCH_CONDITIONING_REQUIREMENTS.md](TOUCH_CONDITIONING_REQUIREMENTS.md): the actual representation must encode observed point/patch structure and allow normals and physical measurements without requiring complete-shape reconstruction first. Full-surface testing must exercise that same path.
