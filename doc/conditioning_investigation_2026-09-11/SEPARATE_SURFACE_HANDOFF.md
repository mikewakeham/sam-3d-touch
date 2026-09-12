# Constructive pilot: separate surface attention around a frozen image model

**First attempt / logging fix:** the first sequential arm passed CUDA module checks, exact initial image replay, zero-residual replay and its first backward pass. The legacy gradient logger then raised `AttributeError` because the wrapper has `.surface.to_kv` and `.visual.to_kv`, not a direct `.to_kv`. No optimizer update was reached. `fit_separate_surface_gpu.py` now uses an experiment-local logger that handles both separate and joint modules and records frozen visual gradients separately. Production `train.py`, the model interface and optimization policy are unchanged. Local logger regression checks cover both module layouts and absent gradients. The terminal and failure record are in `separate_surface_first_attempt/`. Sync the updated driver and retry using the new output directory below; preserve the failed attempt. GPU training remains pending.

## Why this intervention now

The constant-surface control reproduces fitting gains and early held-object gains without sample-specific full-surface information. The joint pathway depends on surface-token presence but has not demonstrated reliable geometric transfer, even with oracle coordinates. There is enough evidence to test an interface correction while retaining VecSetX. This is a candidate, not a proven root-cause fix.

Load the existing `object_transfer/manual/image/checkpoint_1024.pt` and verify its complete generator parameter hash against the returned image report. This anchor already learned the Stage-1 task; using it avoids asking a new surface branch to perform all image-domain adaptation from scratch. The warm-started joint control below is essential to interpret any gain.

## Four arms, 1024 additional updates each

| Key | Surface input | Shape attention policy |
|---|---|---|
| `separate_oracle` | Correct object-frame surface | Frozen visual attention + trainable independent surface residual |
| `separate_constant` | Same training surface for every input | Same separate residual |
| `separate_camera` | Correct camera-frame surface | Same separate residual |
| `joint_oracle` | Correct object-frame surface | Original concatenated visual/surface context, shared shape CA/norm2 trainable |

All four start from exactly the same completed image generator and matched original point-encoder/projector initialization. Retain seed 37, bf16, 16 fitted identities × four fitted views, two held views, 16 held identities × two views, batch four, fixed 1024-step schedule, projector LR 1e-4, attention LR 1e-5, gradient clipping 1, weight decay 0 and the native shape-only flow objective. No visual dropout. All four have the same prior image training and additional exposure. This is not a same-total-training-budget comparison to the old freshly initialized surface models.

The constant bank is the same prespecified training batch 0, row 0 feature code, repeated for all examples. It contains no sample-specific full-surface information. Its raw geometry and token statistics are those of an actual training object. Visual/pointmap inputs remain informative. The camera-versus-constant comparison alone changes both frame and content; oracle-versus-constant is the primary geometry-content contrast.

## Precise module change

For each shape cross-attention call with normalized hidden query h, visual tokens V and projected surface tokens S:

`output = frozen_visual_attention(h, V) + trainable_surface_attention(h, S)`.

Clone the anchor attention's structure and weights for the surface branch, then set its output projection weight and bias to zero. Train the clone's attention parameters plus the existing point projector/embedding. The original generator, including visual attention and norm2, remains frozen. The clone has its own queries, keys, values, output projection and existing query/key normalization; visual and surface softmax denominators are separate. There is no latent regression bridge, extra spatial target, or new point encoder.

Frozen visual operations remain differentiable with respect to hidden states so gradients can reach earlier surface residuals. With surfaces present, those hidden states may change; freezing does not mean the intermediate visual output is numerically constant. With surfaces absent at every layer, the exact complete image mapping is recovered. That fallback is a checkable invariant, not a guarantee of useful surface-conditioned outputs.

The separate arms add one trainable shape attention per block and leave norm2 frozen. Joint oracle instead trains existing shape attention and norm2. Attention parameter counts are recorded; the locations of trainable parameters, freezing and zero-output initialization differ intentionally. This package does not isolate softmax competition from freezing or initialization. Any success must first survive the constant and warm-start controls.

The implementation lives only in this investigation folder. `separate_surface_attention.py` wraps the shape attention modules in the diagnostic process; production backbone/training source files are not changed. Layout remains outside the training objective. Sparse observed points and future per-point features remain compatible with a learned point-token route, but no sparse experiment is being implemented here.

## Mandatory checks and measurements

Before optimization, require original dataset/features/target hashes, matched image checkpoint/configuration and parameter hashes, and exact all-example image loss replay with surfaces absent. All three separate arms must also replay those image losses exactly with their zero-initialized surface residual present.

A disposable actual CUDA module runs forward/backward checks before fitting: zero residual equals visual output, removal equals visual output, a nonzero residual equals the sum of the two independent attention calls, surface output weights receive gradients, frozen visual weights do not, hidden-state gradients are finite, and an invalid context length is rejected. Data-path hooks check that shape visual attention receives 7528 tokens and surface attention receives 1024; joint attention receives 8552. The first-layer hook is supplemented by full-model image replay and final fallback checks.

Record all updates, gradients, trainable counts, initial/final parameter hashes, source hashes and checkpoint hashes. Hashes of all frozen parameters must remain unchanged. Save trainable state and optimizer at 256/512/1024. At the end, every separate arm with its surface removed must exactly reproduce all original image final losses on all three splits. Constant surface swapping must change no final scalar or per-object loss.

Use original monitor bank 600000 (four common draws) at steps 0/256/512/1024. Final bank 700000 uses eight common draws for correct/wrong surfaces and surface removal. Average within identity before comparisons. These are the already inspected development objects and noise banks, not fresh population validation. One wrong-object assignment is a screening control, not an exhaustive geometry test.

## Predeclared decisions

- Real oracle beats frozen image and separate constant broadly on held identities, with correct-surface utility: this is a promising geometry-specific correction. Compare joint oracle to determine whether separation is necessary for that result. Confirm on fresh identities/seed before full training.
- Separate oracle passes but camera fails: retain the successful aligned interface and next test explicit rotation handling against it. This creates a meaningful oracle upper bound rather than predicting pose for an already failing conditioner.
- Both camera and oracle pass: the tested interface/training change handles enough of the coordinate relationship at this scale; do not add a pose estimator automatically.
- Joint oracle is equally good or better: starting from the image checkpoint may suffice; do not credit the separate branch with a benefit the warm-start control also achieves.
- Constant improves as much as real: generic branch adaptation persists. Neither fitting improvement nor a preserved image fallback establishes geometric conditioning. Do not scale that result to full training.
- All real arms fail on held identities despite improving fit: the candidate is unsuccessful at this scale. Preserved image fallback alone is not a solution; inspect geometry-specific learning and adaptation limits rather than running more of the same automatically.
- Fitted losses remain high and clearly improving: finite optimization/capacity is unresolved. Do not infer inherent impossibility; any continuation must be justified from curves and geometry-specific benefit.

No threshold based solely on a tiny mean improvement should trigger full training. Report effect sizes, medians, object counts and outliers against both image and constant. Even a positive pilot needs confirmation.

## Paste into the allocated interactive terminal

Sync the investigation folder, including the three new Python files and `object_transfer_returned_manual/`. Keep the existing image fit report and checkpoint at the default path. Four visible allocated GPUs run concurrently; fewer run sequentially on the first visible GPU. No device allocation is changed. Parallel output goes to one log per arm. Runtime of the new branch is unmeasured.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
branch_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
branch_root=outputs/conditioning_investigation/separate_surface/manual_logging_fix
visible_gpus=$("$branch_python" -c 'import torch; print(torch.cuda.device_count())')
test "$visible_gpus" -ge 1
test ! -e "$branch_root"
mkdir -p "$branch_root"
if [ "$visible_gpus" -ge 4 ]; then
  branch_pids=()
  device_index=0
  for model_key in separate_oracle separate_constant separate_camera joint_oracle; do
    "$branch_python" doc/conditioning_investigation_2026-09-11/fit_separate_surface_gpu.py \
      --model-key "$model_key" --device-index "$device_index" \
      --output-dir "$branch_root/$model_key" > "$branch_root/$model_key.log" 2>&1 &
    branch_pids+=("$!")
    device_index=$((device_index + 1))
  done
  branch_failed=0
  for process_id in "${branch_pids[@]}"; do
    if ! wait "$process_id"; then
      branch_failed=1
    fi
  done
  test "$branch_failed" -eq 0
else
  for model_key in separate_oracle separate_constant separate_camera joint_oracle; do
    "$branch_python" doc/conditioning_investigation_2026-09-11/fit_separate_surface_gpu.py \
      --model-key "$model_key" --device-index 0 --output-dir "$branch_root/$model_key"
  done
fi
"$branch_python" doc/conditioning_investigation_2026-09-11/analyze_separate_surface.py \
  "$branch_root" --output "$branch_root/analysis.json"
)
```

Return the four `results.json` files plus `analysis.json`, or a failing log/traceback. Do not relax replay/frozen checks or overwrite an existing directory. Local syntax and synthetic analyzer checks pass, including rejection of frozen-weight changes, altered fallback, wrong context allocation, constant-swap variation and mismatched separate-arm initialization. Actual torch/CUDA module checks, exact model replay, memory use and optimization are pending GPU execution. No full-dataset job is queued.
