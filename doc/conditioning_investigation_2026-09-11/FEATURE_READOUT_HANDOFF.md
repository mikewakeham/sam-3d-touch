# Next diagnostic: can a small readout translate the existing features?

## Purpose and decision

Native VecSetX reconstruction is strong enough to retain it as a viable representation; point-grid SAM encoding failed the approximate-target premise. This experiment follows the specified access/translation branch. It trains a small readout from the saved **oracle-frame** VecSetX features to the stored Stage-1 latent, bypassing the SAM3D flow network. A successful mapping supplies a positive control for future conditioning changes; no direct regression result by itself solves sparse-touch conditioning.

Run from the cluster repository root after transferring this folder:

```bash
sbatch doc/conditioning_investigation_2026-09-11/run_feature_readout.sh
```

The default source is `outputs/conditioning_investigation/representation_probe/46083371`; pass a different representation directory as the launcher's first positional argument if needed. This uses the existing NPZ outputs, original target files for integrity checks, the same VecSetX checkpoint (processed arm), and `ss_decoder.ckpt`. It does not need the image encoder, source meshes, DiT or Stage 2.

The array requests three independent one-H100 tasks, with a one-hour limit each. They can run concurrently when capacity permits; the scheduler does not guarantee simultaneous start. This is a scheduler cap, not a measured runtime. No full-dataset training is requested.

## Arms and matched controls

| Arm | Input to readout | Question |
|---|---|---|
| raw | Saved 1024×32 bottleneck features | Can an unordered token readout learn this mapping? |
| raw_slot | Same features plus fixed sinusoidal slot IDs after projection | Does explicit slot identity help under the same trainable capacity? |
| processed | Frozen native `learn(features)`: bottleneck expansion, learned slot vectors, 24 transformer layers | Does exposing native processed information make translation easier? |

Every head uses the same two cross-attention blocks, width 256, eight heads, per-query MLPs and 4096 xyz Fourier queries. It predicts eight latent channels at each grid location. The raw and raw_slot models have identical trainable counts and initialization. Processed input width is 1024 instead of 32, so its input projection is larger; all shared head parameters initialize identically. A processed-arm win cannot be attributed uniquely to slot identity or a single native layer. The raw/slot contrast is narrower, but a negative slot result still does not rule out all slot-aware readouts.

The first 24 objects in the pre-existing seeded report order are fitted; the last eight are reserved. No reselection based on native reconstruction quality occurs. All three arms use identical targets, balanced batches of four and deterministic per-epoch shuffles. They run 2000 AdamW steps at 1e-4, zero weight decay and gradient clipping 1. At 2000 steps, each object receives approximately 333–334 presentations. These are fixed full-surface inputs; no camera views are falsely counted as additional generalization examples.

Assessment at initialization, every 500 updates and the endpoint reports each object's direct latent MSE and wrong-object feature MSE. A training-object mean latent is a scoring baseline for both splits. Final predictions are decoded to occupancy and compared with the already saved target decode. No ICP, pose fitting or Stage-2 scoring is performed. Checkpoints contain the model and optimizer for continuation if fitting is still improving.

Frozen feature hashes must match the representation report. Saved targets must equal the original target file and its reported file hash. The processed arm verifies the original VecSetX checkpoint hash. The head weights use fp32, with bf16 autocast for computation; VAE decoding uses fp32. Native VecSetX retains its existing internal attention precision.

## Interpretation branches

- **Raw fits and transfers:** a lightweight spatial readout can extract target-relevant information without changing VecSetX. Next test using its features or representation as a conditioner for the image-conditioned flow model; compare against broader adaptation of the current path. Direct fitting does not prove which part of the existing flow training prevented learning.
- **Slot-aware beats raw consistently:** explicit slot information is useful in this controlled readout. Confirm a second seed and test it in SAM3D. Existing dataset `learn` results remain relevant; this experiment cannot erase their negative evidence.
- **Only processed features work:** native processing exposes a more accessible representation. Test that representation with a more expressive spatial adapter rather than asserting raw codes lost geometry. The existing shared projector `learn` variant did not test this exact readout, but its similar results caution against claiming native processing alone fixes SAM3D.
- **All fit, reserved objects remain poor:** the head may memorize object codes. Do not call this reusable geometric translation; broaden object diversity or test a structured point readout before adopting the head.
- **All fail but improve through the endpoint:** continue matched runs or check optimization before drawing a capacity conclusion. This is a screen, not an impossibility test.
- **All plateau poorly despite a validated optimizer check:** a direct readout also struggles with translation. Compare broader shape adaptation or an explicit point/location encoder; do not discard a natively reconstructible representation based on one failed head.

Use paired object curves and per-object final results rather than only one aggregate. Reserved objects came from a previously inspected representation screen; they are exploratory diagnostic holdouts, not an untouched final benchmark. Do not tune repeatedly to these eight and call it generalization. A second seed and a larger untouched object subset are required before a production recommendation.

## Sparse-touch endpoint

This head is a **full-surface diagnostic**, not a final sparse-input reconstruction model. For sparse touches, complete target regression can average ambiguous unobserved structure. A final conditioner should preserve observed xyz and validity, supply evidence to the image/prior, and be trained on actual clustered touch patterns. The no-position normalization limitation remains: one cannot recover a patch's object location from centered/rescaled geometry alone. The current probe does not fix or test that limitation.

## Returning results and validation

Output: `outputs/conditioning_investigation/feature_readout/ARRAY_JOB_ID/{raw,raw_slot,processed}/`.

Return all three `results.json` files and Slurm logs. Keep final NPZ predictions and checkpoints on the cluster. If the job fails, return its partial report/traceback. Do not rank an incomplete arm against completed ones. Resume uses `--resume PATH/checkpoint_STEP.pt --steps HIGHER_TOTAL` and a new output directory; retain arm/seed/LR/source. Paired analysis requires matching resumed intervals across arms.

```bash
python doc/conditioning_investigation_2026-09-11/check_feature_readout.py \
  outputs/conditioning_investigation/feature_readout/ARRAY_JOB_ID \
  --output outputs/conditioning_investigation/feature_readout/ARRAY_JOB_ID/analysis.json
```

Python compilation and shell syntax are checked locally. Torch and GPU runtime are unavailable locally. The launcher runs `test_feature_readout.py` first, checking xyz flattening, shared initialization, raw-token permutation invariance, slot-ID sensitivity and gradient reach before spending on training. These model tests have not executed on this laptop; they are cluster preflight, not claimed passing GPU validation.
