# Training intervention: visual dropout under camera and oracle surface frames

## Decision being tested

The coordinate-factor experiment isolates a camera-rotation penalty. The visual-stream probe shows that even oracle-aligned surfaces leave a largely RGB/silhouette-driven reserved-view loss gap, despite correct surfaces being preferred to wrong ones. A pose correction alone is therefore not a demonstrated complete fix.

Train the existing conditioner to operate with and without visual input while preserving the same point representation. Compare the intervention in camera and oracle coordinates. This tests whether visual reliance contributes to the coordinate gap, or whether repairing visual robustness helps only after alignment. It does not replace the coordinate investigation with an encoder search.

## Matched design

| Surface coordinates | Original visual-present training | 50% visual-dropout training |
|---|---|---|
| Camera | Existing 1000-update checkpoint, evaluated again | New 1000-update run |
| Oracle | Existing 1000-update checkpoint, evaluated again | New 1000-update run |

Retain four objects, four fitted views, three reserved views, seed 29, bf16, frozen raw VecSetX features, original point normalization, no global position embedding, projector LR 1e-4 and full shape cross-attention/norm2 LR 1e-5. Original flow time/noise seeds and object exposure count remain fixed.

For exactly 125 of the 250 presentations of **each fitted view group**, replace its fused RGB/mask/pointmap tokens with zeros. Keep token count and touch tokens unchanged. The batch is dropped as a whole. A local Python RNG precomputes the schedule without changing the global Python or torch RNG used for flow draws; camera and oracle use the same schedule. Alternating global steps would confound dropout with the view-group cycle and is not used.

Zero visual tokens are not the same operation as removing them: their attention mass and linear biases remain. This is a concrete zero-conditioning policy, not a claim that all visual influence vanishes from the pretrained weights. CFG unconditional probability remains zero, so the touch stream must not be erased. A hook verifies the actual shape cross-attention context has zero visual tokens and unchanged, nonzero touch tokens before training proceeds.

## Replay and measurements

Each arm locates its exact historical completed multi-view report and checkpoint. It reproduces initial parameters, inputs/features and all seven initial loss vectors. It then temporarily loads the original fitted weights, validates their full digest, and replays all final correct/swapped vectors exactly. Without updating these baseline weights, it assesses both visual-present and visual-zero conditions using a new common eight-draw bank (base 400000).

The runner then restores the exact pretrained trainable tensors, verifies the full initial parameter digest and empty optimizer state, and starts training. This is not a continuation from the historical checkpoint.

Assess visual-present native loss and swapped-surface controls at steps 0/100/300/1000 using the historical bank. Save trainable parameters and optimizer checkpoints at 300/1000. At the final step, assess both visual-present and visual-zero conditions on the same new bank used for the historical fitted baseline. Primary outcome is **correct-surface loss with all visual inputs present on coherent reserved views**. Wrong-surface loss and fitted loss qualify the interpretation. Surface-only assessment is diagnostic, not the deployment success criterion.

No VAE, sampling rollout, Stage 2, sparse sampling, new encoder, broader backbone finetuning or full-dataset run is included. This screens native Stage-1 training behavior. A promising result needs another seed, a modest unseen-object confirmation and relevant Stage-1 generation checks before any full run. Failed finite-budget fitting is not proof of an inherent limitation.

The analyzer reports dropout effects separately in both frames, the camera-minus-oracle gap under both policies, their interaction, per-view-group results, and comparable all-input assessment curves. Do not compare raw mixed-objective training loss directly to historical visual-present-only training loss.

## Branches

- Both frames improve on coherent reserved views and the frame gap narrows: stronger surface reliance is a promising repair to test on more objects before adding pose machinery.
- Oracle improves, camera remains worse: establish the improved aligned bound, then test deployable rotation handling against it.
- Oracle improves, camera deteriorates: camera-frame conditioning may need visual orientation information; do not transfer the dropout policy unchanged.
- Only dropped-condition loss improves, or fitted loss improves while reserved loss worsens: reject the claimed robustness fix.
- Neither improves: reject this policy at this budget, inspect curves/controls, and avoid claiming that representation or architecture must be replaced.

Future sparse touch may require visual completion much more strongly than full surfaces. This dropout rate is an upper-bound diagnostic, not a prescribed sparse-touch training recipe.

## Paste into an allocated interactive terminal

Sync the investigation folder, including archived camera/oracle single- and multi-view references and the new three Python files. The runner searches the two known `multiple_view_fit` output roots. It needs the original `fitted_parameters.pt` files; JSONs alone are insufficient. If lookup is ambiguous, specify `--baseline-fit-dir /path/to/the/camera-or-oracle-directory` for that invocation. Do not retrain old baselines.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
dropout_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
for arm in camera oracle; do
  "$dropout_python" doc/conditioning_investigation_2026-09-11/fit_visual_dropout_gpu.py \
    --arm "$arm" \
    --output-dir "outputs/conditioning_investigation/visual_dropout/manual/$arm"
done
"$dropout_python" doc/conditioning_investigation_2026-09-11/analyze_visual_dropout.py \
  outputs/conditioning_investigation/visual_dropout/manual \
  --output outputs/conditioning_investigation/visual_dropout/manual/analysis.json
)
```

This runs the two arms sequentially on one GPU. No runtime estimate has been measured. Return `camera/results.json`, `oracle/results.json` and `analysis.json`. Preserve checkpoints on the cluster. Existing output directories are refused. On replay/interface failure, return the traceback and any `results.partial.json`; do not relax checks.

Local validation passed: Python/shell syntax; exactly 500 dropped updates and 125 within every group; schedule determinism without global RNG consumption; analyzer contrasts/interactions with known synthetic effects using real historical report schemas; rejection of an incorrect executed schedule and missing assessment coverage. Torch, checkpoint replay, interface hooks and actual training cannot be tested on this laptop.
