# Check the existing improvement on objects outside the tiny fit

## Why no new training yet

The dropout experiment improved oracle fitted/reserved-view loss to .02205/.02273, but all those views belonged to four fitted objects. Surface-based identity lookup remains an alternative to transferable geometric use. Before asking for a larger training comparison, use the existing checkpoints to test that alternative directly.

`unseen_object_selection.json` packages the original diagnostic selection: 32 objects, two fixed views each, 16 from the dataset train split and 16 from validation. All 32 identities are disjoint from the four objects fitted by every compared checkpoint. The selection predates the dropout result and no object is selected based on a new checkpoint's performance. It has been used in previous diagnostic analyses, so this is not a pristine population benchmark. Pretraining exposure of SAM3D/VecSetX is unknown.

## Five existing checkpoints, zero training updates

- Image + pointmap (`image_original`): original matched four-object multi-view baseline; no touch tokens.
- Camera surface, original training.
- Camera surface, visual-dropout training.
- Oracle surface, original training.
- Oracle surface, visual-dropout training.

All receive the same normal visual input. Point-conditioned models are evaluated with the correct surface and with another selected object's surface, using a fixed within-batch permutation of four distinct identities. Oracle pose remains a privileged positive control, not an available inference measurement. No visual streams are crossed, no modality is dropped at evaluation, and no weights are updated.

The image+pointmap checkpoint supplies an actual separately trained no-surface comparison. A wrong-surface penalty alone does not establish improvement over it.

Each invocation validates source/configuration, the instantiated initial model, checkpoint settings and final parameter digest, then exactly reproduces an existing final anchor loss vector (plus its swapped-surface vector for point models). Dropout anchors use their returned fresh bank 400000; original anchors use the historical bank 100000. It then evaluates the disjoint selection using a new bank, base 500000. Each object receives the same noise/time draws across its two views and across all models/treatments.

Per-object native shape MSE is captured inside the original scalar loss calculation. The original loss function still returns the scalar used for replay. The runner requires the per-object mean to reproduce the native shape loss and total loss within tight floating-point tolerance. The analyzer independently reconstructs every batch scalar from the reported per-object values. This provides object-level comparisons without running batch size one or treating views/draws as independent objects.

The five invocations require 1224 batch-of-four native loss forwards including anchor checks, plus input embedding/model loading. No measured runtime estimate is available. No backward pass, decoder, sampling rollout or Stage 2 is involved. No full training run is warranted before these results.

## Decision criteria

Primary comparisons: correct-surface native loss, averaged first within each object over its two views and eight draws, then across objects. Report paired mean/median changes and counts of objects improved for dropout versus original, point-conditioned models versus image+pointmap, and camera versus oracle. Report train/validation selections separately as well as combined. Wrong-surface penalties are supporting evidence.

- Oracle dropout improves over original oracle and image+pointmap on a broad set of objects, while camera remains weaker: the aligned positive control extends beyond the fitted identities; prioritize explicit camera-to-target rotation handling.
- Both dropout models improve and the camera/oracle gap shrinks on unseen objects: stronger surface reliance itself transfers; confirm on a modest training subset/another seed before adding pose machinery.
- Dropout improves fitted identities but worsens unseen identities, or neither oracle variant improves over image+pointmap: do not interpret the four-object result as a transferable geometry fix. The next training test must expand object diversity with a reserved object set. This would not prove that VecSetX loses information or that the architecture cannot work.
- Only wrong-surface loss becomes large, without better correct-surface performance: do not count stronger sensitivity as successful conditioning.

This is a cheap test using existing artifacts before a new optimization run. Failure after training on four identities does not predict failure after training on more objects. Positive native losses alone do not establish generation quality; retain checkpoints for appropriate Stage-1 generation checks once the training candidate is supported.

## Paste into an allocated interactive terminal

Sync the investigation folder, including all original image/camera/oracle single- and multi-view JSONs, the returned dropout JSONs, `unseen_object_selection.json`, and the two new Python files. Original checkpoints are located by exact report match under the known `multiple_view_fit` roots. Dropout checkpoints default to `outputs/conditioning_investigation/visual_dropout/manual/{camera,oracle}/checkpoint_1000.pt`. If a checkpoint was moved, add `--checkpoint /path/to/file.pt` for that invocation; the parameter/settings checks still apply. Do not retrain to satisfy a path lookup.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
unseen_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
for model_key in image_original camera_original camera_dropout oracle_original oracle_dropout; do
  "$unseen_python" doc/conditioning_investigation_2026-09-11/probe_unseen_objects_gpu.py \
    --model-key "$model_key" \
    --output "outputs/conditioning_investigation/unseen_objects/manual/$model_key.json"
done
"$unseen_python" doc/conditioning_investigation_2026-09-11/analyze_unseen_objects.py \
  outputs/conditioning_investigation/unseen_objects/manual \
  --output outputs/conditioning_investigation/unseen_objects/manual/analysis.json
)
```

Return the five model JSONs and `analysis.json`; no NPZ upload is needed for this native-loss probe. If replay fails, return the traceback and any partial JSON. Existing outputs are refused. Do not lower the replay tolerance or rerun old training.

Local checks passed: source syntax, handoff shell syntax, actual selection disjointness, synthetic per-object paired comparisons and split counts, and rejection of fitted-identity leakage, missing observations and inconsistent scalar reconstruction. Actual checkpoint replay and native-loss capture must execute on GPU; the laptop has neither torch nor the required checkpoints.
