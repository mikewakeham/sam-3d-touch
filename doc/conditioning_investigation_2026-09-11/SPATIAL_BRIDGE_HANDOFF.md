# Next control: expose existing geometry before translating to SAM

The 8000-update readouts fit training objects but fail to transfer. Before training another spatial adapter, test whether a fixed spatial conversion can translate the already reconstructible geometry into Stage-1 targets without learning from those 24 objects.

`probe_spatial_bridge_gpu.py` uses the saved native **oracle** VecSetX meshes from the representation probe. It voxelizes their triangles with the exact target-generation voxelizer and passes that shell through the frozen SAM sparse-structure VAE. This differs from the previously unsuccessful point-hit grid: native VecSetX reconstructs surfaces between samples, and triangle voxelization represents that continuous surface. The comparison changes geometry completion as well as the interface; any improvement cannot be attributed uniquely to coordinates.

No further readout training, DiT, image encoder, VecSetX inference, or Stage 2 is needed. One allocated GPU is needed for the SAM VAE; local GPU execution is unavailable. The saved NPZ artifacts, both SAM VAE checkpoints, and original target files must be on the cluster. This is a representation control, not yet a production conditioner.

## Controls and evidence

- Build all candidate grids from saved native vertices before loading targets. Vertices are already in object coordinates. Do not apply another rotation, normalization, ICP, dilation, or target-assisted repair.
- Check source/checkpoint hashes, native feature hashes and saved geometry metadata. Reextract surfaces from saved fields and verify vertices/faces reproduce the saved native mesh. New artifact hashes record provenance; the original report did not hash field/mesh arrays, so this does not independently regenerate native neural outputs.
- Verify saved target arrays against original target-file hashes. Reencode the saved original mesh grids and require reproduction of target latents. Require the target decoder to reproduce saved occupancy exactly.
- Reencode point-hit grids and require reproduction of the earlier surface latents. This is a matched negative control, not another proposed point-hit conditioner.
- Encode a prespecified +x one-voxel translation of the target shell, cropping without wrapping. This target-derived sensitivity control never enters a candidate conditioner and never selects a correction. It measures one perturbation, not a rotation/translation optimization or complete characterization of latent sensitivity. Report cropped voxels because boundary loss is a confound.
- Report per-object direct Stage-1 latent MSE and the mean-target baseline. Occupancy overlap before/after encoding is supporting evidence to distinguish geometry error from encoding sensitivity. The old 24/8 split labels are preserved for comparison; this frozen bridge has no adaptation on either group.

The exact target voxelizer clips vertices to the object cube. Record the fraction and distance outside the cube for native surfaces so clipping is visible. The old native field uses 64 cells over [-1.05, 1.05] in normalized VecSetX coordinates; it is not necessarily as fine as the target voxel grid after inverse scaling. A failed bridge may therefore require a resolution check before ruling out geometric conversion. Do not silently upsample or tune extraction to targets.

## Outcome branches

1. **Native bridge substantially improves over point hits and the mean control on separate objects:** direct spatial conversion is a useful positive control for target-relevant conditioning. Next compare a spatial residual condition against the existing token interface in Stage 1 on a short matched fit. Keep an oracle target condition as an explicit privileged control if needed to isolate fusion. Restore camera-frame operation before a full training recommendation.
2. **Native grids match target support well but latent error remains poor:** inspect the one-voxel sensitivity control. If it is similarly large, small spatial discrepancies may be strongly amplified by the VAE target. Test local spatial features/conditioning rather than requiring point inputs to reproduce an exact full-object latent. This still does not prove the original flow objective is at fault.
3. **Native support is inaccurate despite high tolerance-based surface F-score:** the previous native reconstruction metric was insufficient to establish target-grid fidelity. Check field resolution and voxelization effects before considering this a representation ceiling. Avoid another token-head extension.
4. **Neither representation comparison nor sensitivity control isolates a useful route:** use an explicit observed-point spatial interface with its own geometric control; do not infer that full SAM finetuning or a new encoder is required from a failed frozen bridge alone.

## Sparse-touch endpoint

The native complete-surface reconstruction is only a full-surface positive control. It may hallucinate unobserved regions for sparse input. A final spatial conditioner must preserve observed positions/validity and distinguish evidence from inferred completion; uncertain complete geometry must not become hard constraints. Spatial query features, local point features, and a separate observation mask remain compatible with sparse structured touch patches. No sparse feasibility claim follows from this job.

## Paste into an allocated interactive terminal

Sync the new Python file into the investigation folder first. Paste the contents below; no batch submission is involved.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python \
  doc/conditioning_investigation_2026-09-11/probe_spatial_bridge_gpu.py \
  --representation-dir outputs/conditioning_investigation/representation_probe/46083371 \
  --output-dir outputs/conditioning_investigation/spatial_bridge/manual
)
```

Return `outputs/conditioning_investigation/spatial_bridge/manual/results.json`, or the traceback and partial JSON if it fails. Keep generated NPZs on the cluster. The output must not already exist; incomplete runs are not silently overwritten or resumed. Python compilation is verified locally; model, Open3D and marching-cubes execution are not validated on this laptop.
