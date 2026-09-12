# Does a sample-independent token pathway explain the apparent benefit?

The returned checkpoint probe shows that early improvement over image survives three wrong-object surface assignments, while removing surface tokens collapses performance. A same-path training control removes sample-specific geometry without removing the input stream at inference.

Run **one** arm, `oracle_constant`, for 1024 updates. Reuse the completed real-oracle/image/camera reports from `object_transfer_returned_manual/`; do not rerun those baselines. Seed 37, bf16, same 16 training identities, four fitted views, two held views, and 16 held identities. Same batches, time/noise draws, trainable projector and full shape CA/norm2, learning rates and zero visual dropout. No full training.

The control fixes the VecSetX raw feature bank from **training batch 0, row 0**, as ordered in `object_transfer_plan.json`. It is one real oracle surface, not an invented average/random code. Every example, including held examples, receives that same bank. Its 1024 native feature slots and 32 channels go through the existing trainable projector; the projected tokens evolve during training but are identical across examples at a given parameter state. RGB/masks/pointmap still vary normally, so this removes only sample-specific full-surface information. The bank may represent useful generic geometry; it is not required to be numerically meaningless.

The driver is a scoped copy of `fit_object_transfer_gpu.py` so the original driver and its archived source hashes remain unchanged. It verifies identical initial generator/encoder/projector parameters, all actual original inputs/features/targets, and exact original step-zero losses using the real surfaces before any update. It then checks constant context at shape attention, trains exclusively on the constant bank and saves checkpoints at 256/512/1024. At final assessment, rolling the constant surface across batch examples must reproduce every scalar and per-object loss exactly. The bank's hash must remain unchanged. No per-sample surface feature enters training after the initial replay.

The analyzer checks the original schedule, complete updates, held-data exclusion from bank selection, equal actual inputs, optimizer scope, initial replay, all object-level/native-scalar reductions, and constant swap invariance. Final bank 700000 and monitor bank 600000 are matched to the completed real training reports. The frozen checkpoint probe used bank 800000; its absolute values are not the baseline for this new run.

Compare constant against image, and real oracle against constant, on fitted identities, held views and held identities separately. Camera is a contextual comparison; oracle is the closest matched treatment. Read the outcome branches in `TRANSFER_CHECKPOINTS_RETURNED_FINDINGS.md`. One constant bank and one seed cannot establish universal equivalence or failure of all sample-independent controls. This is not a proposed replacement conditioner.

Local Python syntax and synthetic analysis checks passed, including rejection of held-object bank selection, changed initial replay, nonconstant swap output, enabled dropout and altered token count. The real-surface assessment is statically restricted to the initial replay; the optimizer loop uses the constant default. GPU execution remains pending. This needs one allocated GPU and no new dataset/checkpoint downloads.

Sync the investigation folder, including the new scripts and `object_transfer_returned_manual/`. Paste into your interactive terminal:

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
control_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
control_root=outputs/conditioning_investigation/constant_surface/manual
test ! -e "$control_root"
"$control_python" doc/conditioning_investigation_2026-09-11/fit_constant_surface_gpu.py \
  --device-index 0 --output-dir "$control_root"
"$control_python" doc/conditioning_investigation_2026-09-11/analyze_constant_surface.py \
  "$control_root/results.json" --output "$control_root/analysis.json"
)
```

Return `results.json` and `analysis.json`, or the traceback/log if a check fails. Preserve saved checkpoints. Do not relax replay checks, reuse an existing output directory or automatically extend the training budget.
