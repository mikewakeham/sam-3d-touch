# Separate the residual visual dependence after perfect surface alignment

## Why this now

The frame-factor experiment supports rotation handling as a real burden. However, its oracle endpoint is already the best result available from perfect pre-encoder alignment, and it still has substantial residual view dependence. The user's prior dataset-wide oracle run also did not resolve conditioning. A new pose estimator should not be scaled before understanding that limit.

The previous single-view probe changed RGB and pointmap together. This probe closes that specific gap on the existing **multi-view oracle checkpoint**, so a changed view is not wholly absent from the model's training experience. It preserves VecSetX, full surfaces, target axes and model parameters. This is the residual-oracle branch already recorded in the reconciled plan, not a new representation branch.

## Intervention and controls

For each of the three other fitted view groups and three reserved view groups, cross three independent streams between that group and anchor group 0:

- RGB: both cropped and full image tokens.
- Silhouette: both cropped and full mask tokens.
- Pointmap: both cropped and full pointmap tokens.

Use all eight combinations. Capture tokens after the original embedder/projector/positional processing and prove that concatenating the captured tokens exactly reconstructs the original fused condition. Each source view is preprocessed normally; do not cross raw image and pointmap pixels or recompute a crop on mismatched inputs. Token streams still describe different viewpoints in crossed conditions, which is deliberately inconsistent.

For every cell, use the same anchor's oracle surface tokens and target, then repeat with those surface tokens rolled between objects. This tests change in relative preference for correct geometry as well as native flow loss. Silhouette is its own factor rather than silently assigned to RGB or pointmap.

Before interpreting results, require historical source/config/init/final parameter hashes, all seven input/feature hashes, and all final natural correct/swapped losses to reproduce. The replay uses the historical eight draws; the main factorial uses eight fresh common draws (seed base 300000). Full-view endpoint cells must exactly reconstruct their original fused tokens. Final model parameter digest must remain unchanged.

There are 800 batch-of-four native loss forwards including replay and cached anchor controls. No optimization, VAE, rollout, or Stage 2 is run. No runtime estimate has been measured. One allocated GPU is sufficient for the same model/batch setup as the original fit.

## Interpretation and branches

- A large conditional pointmap effect, especially reduced correct-surface preference with RGB/silhouette held fixed, identifies pointmap as an intervention candidate. It does not establish a frame conflict: confirm with a short matched oracle training comparison with/without pointmap, retaining identical RGB/silhouette/surface inputs and optimization.
- If RGB/silhouette dominate instead, removing pointmap or changing its axes is poorly motivated. Test view robustness of aligned-surface use with the implicated visual stream, keeping coordinate and representation hypotheses distinct.
- If mixed-view interactions dominate while coherent endpoints behave well, treat the factorial as an inconsistency response. Do not remove a natural modality on this evidence.
- If correct surfaces remain preferred in every cell, yet coherent reserved-view loss stays high, this is a view-transfer/fitting problem rather than evidence that surfaces are ignored. A short training intervention must improve coherent inputs before a full run.

Do not assign additive percentages of blame. Shared objects, checkpoint and noise draws do not supply independent replication. This does not prove unseen-object generalization, pose unidentifiability, or a need to replace VecSetX. Coordinate correction remains the primary question; this probe measures the known residual under its positive control.

## Paste into an allocated interactive terminal

Sync the investigation folder first. The runner locates the completed multi-view oracle checkpoint by exact report content under either `outputs/conditioning_investigation/multiple_view_fit` or historical `outputs/multiple_view_fit`. If it finds zero or multiple matches, it fails with instructions to supply `--fit-dir /path/to/the/multiple_view_oracle_directory`; do not retrain anything. Archived reference JSONs alone are insufficient: the cluster must retain `fitted_parameters.pt`.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
probe_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
"$probe_python" doc/conditioning_investigation_2026-09-11/probe_oracle_visual_streams_gpu.py \
  --output outputs/conditioning_investigation/oracle_visual_streams/manual/probe.json
"$probe_python" doc/conditioning_investigation_2026-09-11/analyze_oracle_visual_streams.py \
  outputs/conditioning_investigation/oracle_visual_streams/manual/probe.json \
  --output outputs/conditioning_investigation/oracle_visual_streams/manual/analysis.json
)
```

Return `probe.json` and `analysis.json`. If replay fails, return the traceback and `probe.partial.json`; do not lower tolerances or rerun training. Existing output files are not overwritten.

Local verification: both Python files parse; the standard-library analyzer recovers known synthetic conditional effects including an RGB/pointmap interaction and change in correct-surface preference; missing coverage is rejected. Shell contents pass syntax validation. Actual checkpoint replay, token capture, and GPU execution remain untested locally because torch/checkpoints/GPU are unavailable.
