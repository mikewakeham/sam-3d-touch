# Existing full-dataset checkpoints: surface utility without retraining

**Paused, 12 September:** the active coordinate investigation first inventories the recovered original oracle run and selects a comparable camera/oracle pair. Follow `CONTEXT_AND_COORDINATE_NEXT_STEP.md`. The scripts below are retained, but this image/camera-only command is not the current requested run.

The separate-attention pilot failed its held-object criterion despite exact image preservation and much better fitting. Before another architecture or training run, establish whether the same weak correct-surface utility occurs in the original dataset-wide models.

## Identified checkpoints

| Key | W&B run | Recorded output directory | Expected best step | Expected best loss |
|---|---|---|---:|---:|
| image | xun3al7m | outputs/stage1_image_full_cross_attention | 8796 | .0885780492374702 |
| surface | act988rs | outputs/stage1_full_surface_full_cross_attention | 6597 | .08885870724817051 |

The paths come from both checked-in job scripts and saved run configs. `full_dataset_probe_references/` archives those run JSONs and the minimum-validation-loss/step extracted from each exported history. The probe defaults to `best.pt` under the recorded directory. These historical paths are not rewritten to the investigation output prefix. If a checkpoint has moved, pass `--checkpoint` to the same command with its actual path; metadata must still match the identified run. Do not retrain a missing model.

Both configurations use full shared shape CA/norm2 adaptation, frozen VecSetX for the surface model, no touch position, RGB/masks/pointmap present, and ordinary camera-frame surfaces. Image's touch config is null. The probe checks checkpoint mode, conditioning config, touch config, attention scope, best loss and best step against the export; it also requires exact trainable parameter keys/shapes before loading. It does not assume an arbitrary similarly named checkpoint is equivalent.

## Measurements and scope

Reuse the established 16 dataset-train identities and 16 validation identities, two views each: batches 16 through 31 of the original object-transfer plan. Their old labels `held_view` and `held_object` applied to the tiny fits; this probe labels them `train` and `val` according to the current dataset partitions. Neither group is treated as unseen by full-dataset pretraining, and the selected pool has already been examined diagnostically.

For the full-surface checkpoint, compare correct surface tokens, each of the other three objects' tokens in the batch, and removal of surface tokens. For the image checkpoint, evaluate its normal visual context. All visual tokens stay present. No token dropout during training, optimization, Stage-1 rollout, decoder, Stage 2 or mesh CD is involved.

Every actual image, pointmap, visual condition, target and raw frozen VecSetX feature hash must match the corresponding earlier diagnostic batch. Record current eligible dataset object counts. Context hooks verify 7528 visual tokens alone or 8552 visual+surface tokens. Model parameters must remain unchanged. Native per-object losses must reconstruct the original scalar reduction.

Use eight fresh common draws, bank `900000 + 37 + 100*(batch % 4) + 400*(batch // 8) + draw`. Both views of a given object share draws; image/surface models and all condition interventions share them. Average within an identity before reporting means, medians and paired counts. Report train and validation separately. This is 768 batch forwards total on one GPU, with no backward passes; runtime is unmeasured.

The image and surface checkpoints were selected at different best steps, so their comparison is not a matched causal estimate of adding a surface. Correct-versus-wrong comparisons within the same surface checkpoint are paired interventions. Removal changes the training input distribution, and three distractors are not every possible wrong geometry.

These older checkpoints did not save historical input/source fingerprints. Matching metadata and the current established diagnostic inputs does not certify byte-for-byte historical training-data replay. Current pipeline/data YAMLs must match the earlier validated diagnostic reference. An unexpected result therefore needs that provenance limit retained; do not equate matching directory names with full training provenance.

Read outcome branches in `SEPARATE_SURFACE_RETURNED_FINDINGS.md`. No new full-data training or automatic extension of the failed separate-attention fits is requested.

## Interactive terminal command

Sync the two new scripts and `full_dataset_probe_references/` alongside the existing investigation folder and references. Paste:

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
probe_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
probe_root=outputs/conditioning_investigation/full_dataset_conditioning/manual
test ! -e "$probe_root"
mkdir -p "$probe_root"
for model_key in image surface; do
  "$probe_python" doc/conditioning_investigation_2026-09-11/probe_full_dataset_conditioning_gpu.py \
    --model-key "$model_key" --device-index 0 --output "$probe_root/$model_key.json"
done
"$probe_python" doc/conditioning_investigation_2026-09-11/analyze_full_dataset_conditioning.py \
  "$probe_root" --output "$probe_root/analysis.json"
)
```

Return `image.json`, `surface.json` and `analysis.json`, or a traceback if the checkpoint/input checks fail. Local Python/shell syntax and synthetic analyzer checks pass, including rejection of wrong best step, wrong coordinate config, missing distractor, scalar-reduction mismatch and changed targets. GPU checkpoint loading and measurements are pending. No GPU run can be performed locally.
