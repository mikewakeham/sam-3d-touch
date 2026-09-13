# Source integration for the oracle upper bound

Implemented on 13 September with explicit user authorization for this one source-code integration. Subsequent experimental edits return to the investigation folder unless the user authorizes another source change. No cluster jobs have been submitted.

This supersedes the separate one-GPU-per-arm trainer and its historical-checkpoint preflight. Do not run `oracle_upper_bound/run_gpu.py` or `launch.py` for these full runs. Historical diagnostic scripts deliberately hash the old `train.py`; their source checks will now fail until a future diagnostic explicitly accounts for this authorized integration. Do not rewrite historical result hashes.

## What changed

- `train.py`: optional `--visual-dropout` (default 0), `--constant-touch` (default off), per-sample visual-token zeroing during training, shared surface-token generation, fixed training-reference feature initialization, checkpoint/resume metadata and constant-bank persistence, dropout logging. Full shape cross-attention is now the CLI default at the user's request. Explicit `--cross-attention-scope kv` remains available; missing scope metadata in older checkpoints still means KV, not full.
- `evaluate.py`: restore training settings and the constant bank; obtain surface tokens through the model's shared helper. Ordinary evaluation keeps visuals present. No Stage-2 algorithms or metrics changed.
- `diagnostics.py`: reject the constant control for activation diagnostics that assume live per-sample VecSetX encoding. This avoids labeling the supplied object's representation as the actual conditioning. Normal runs remain supported.
- Three new `jobs/` scripts copied from `stage1_full_surface_full_cross_attention.sh`. Same partition/account/CPU/memory/GPU/time/log template, batch 4 per GPU, 20 epochs, worker counts and learning-rate defaults. Names/output paths/variant flags differ, and the now-redundant scope flag is omitted.

No dataloader, model architecture, target generation, VecSetX implementation or dataset configuration changed. The constant reference is the lexicographically first sample ID in the training dataset. Its oracle-aligned observed surface is encoded once, broadcast consistently across ranks, and saved as a fixed raw feature bank. Reference initialization preserves the ordinary random states. The projection/embedding still train; the constant bank does not.

Visual dropout follows SAM3D's modality-dropout granularity: independent samples, all image/pointmap tokens for a selected sample zeroed together, surface tokens retained, no inverse-probability scaling. Each GPU uses its own deterministic step-addressed random stream. This differs from the earlier whole-batch tiny experiment; 50% is our experimental rate, not a claim about SAM3D's original training recipe. Existing unconditional CFG dropout remains disabled as before.

## Variants and comparisons

1. `stage1_full_surface_oracle_dropout`: oracle input alignment plus 50% visual dropout; candidate aligned upper bound.
2. `stage1_full_surface_oracle_constant_dropout`: same policy and trainable architecture with one fixed training surface's raw VecSetX features; tests sample-specific geometric utility.
3. `stage1_full_surface_dropout`: real camera-frame surfaces plus the same dropout; measures frame-treatment cost against oracle.

Reuse the completed visual references `xun3al7m` (image+pointmap, full CA) and `fl7b2znc` (image-only, full CA). Both exported runs finished 20 epochs/14,660 steps, global batch16, bf16, CA lr1e-5. Before interpreting them as matched examples, verify that the two manifest paths in data1.yaml/data_full_surface.yaml refer to the same visual examples/targets/split. The cluster dataset is not locally available for that final check. Adding optional surface-run dropout does not by itself require baseline retraining.

Oracle validation still uses privileged alignment. Full-data success has not been demonstrated by this integration. Assess training and held objects against decoded Stage-1 target support, with correct/wrong surfaces and the constant/image controls. If native pose agreement is low, distinguish output pose from intrinsic shape using saved supports. The production loop keeps its existing validation and best/last checkpoints; a new embedded evaluation framework was intentionally not added.

## Verification

Ten real PyTorch CPU tests passed using a temporary environment with Torch2.14.0: new CLI defaults; unchanged default forward and gradients; per-sample masks and RNG isolation; visual-only zeroing and trainable adapter gradients; constant-input invariance; deterministic reference selection/RNG preservation; checkpoint and optimizer restoration plus old defaults/scope metadata; evaluation restoration; visual-present validation; and two-process Gloo DDP for ordinary and constant paths (three updates each, synchronized trained weights).

Tests use small tensor models, not SAM3D/VecSetX checkpoints. CUDA/NCCL and full-model loading/training remain untested locally. Cluster uses its existing environment; do not upgrade it to match the temporary laptop test environment. Python parsing, shell syntax and template-resource equality also passed. See `source_integration_checks.json`.

Re-run the small tests in the existing cluster environment with:

```bash
python doc/conditioning_investigation_2026-09-11/oracle_upper_bound/test_source_integration.py -v
```

## Job scripts and interactive contents

Sync train.py, evaluate.py, diagnostics.py and the three job scripts together. Each script below requests four GPUs; these are separate jobs, not three one-GPU processes. Output is under outputs/conditioning_investigation. Jobs are prepared, not submitted.

### stage1_full_surface_oracle_dropout.sh

Submit: `sbatch jobs/stage1_full_surface_oracle_dropout.sh`

For an already allocated four-GPU interactive node, paste the script body:

```bash
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/torchrun --standalone --nproc_per_node=4 train.py \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --data-config configs/data_full_surface.yaml \
  --output-dir outputs/conditioning_investigation/stage1_full_surface_oracle_dropout \
  --batch-size 4 \
  --workers 8 \
  --val-workers 2 \
  --epochs 20 \
  --no-touch-position \
  --visual-dropout 0.5 \
  --oracle-point-frame
```

### stage1_full_surface_oracle_constant_dropout.sh

Submit: `sbatch jobs/stage1_full_surface_oracle_constant_dropout.sh`

For an already allocated four-GPU interactive node, paste the script body:

```bash
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/torchrun --standalone --nproc_per_node=4 train.py \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --data-config configs/data_full_surface.yaml \
  --output-dir outputs/conditioning_investigation/stage1_full_surface_oracle_constant_dropout \
  --batch-size 4 \
  --workers 8 \
  --val-workers 2 \
  --epochs 20 \
  --no-touch-position \
  --visual-dropout 0.5 \
  --oracle-point-frame \
  --constant-touch
```

### stage1_full_surface_dropout.sh

Submit: `sbatch jobs/stage1_full_surface_dropout.sh`

For an already allocated four-GPU interactive node, paste the script body:

```bash
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/torchrun --standalone --nproc_per_node=4 train.py \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --data-config configs/data_full_surface.yaml \
  --output-dir outputs/conditioning_investigation/stage1_full_surface_dropout \
  --batch-size 4 \
  --workers 8 \
  --val-workers 2 \
  --epochs 20 \
  --no-touch-position \
  --visual-dropout 0.5
```
