# Completed-run checkpoint probe

Status: prepared; four CPU pairing/report tests pass, GPU module import/CLI and Python/shell syntax checks pass. **No SAM3D GPU execution or results yet.** Production files are untouched. This continues F17's oracle upper-bound branch.

## Why this experiment

The completed oracle run has a small training advantage over constant but no visual-present validation advantage. The existing logs cannot separate geometry utility on fitted objects, generalization to other objects, and interaction with visual input. This probe makes those comparisons on saved weights without another update. It tests the trained Stage-1 objective directly; it does not load a decoder or Stage2.

Load oracle and constant `last.pt`, both epoch20 / step14,660. The constant bank comes from the checkpoint via production `get_touch_tokens`; it is not re-created. Architecture/scope/training flags are checked against the saved config, adapted tensors are checked immediately after restoration and after the probe, and observed shape-cross-attention context is checked against the requested tokens.

Select16 training and16 disjoint validation identities by a deterministic metadata hash, two extreme sorted view IDs per object. These are development diagnostics on the existing validation split, not a newly pristine test set. In each batch four distinct objects provide three cyclic wrong-object surfaces. Wrong tokens are swapped after encoding, retaining token count, projection, and visual context. For constant there is one bank, so no redundant wrong-object cells.

Each batch receives ten actual time/noise pairs: four draws from the configured native time sampler; two noise draws each reused at t=.05,.5,.95. For this flow path, x_t = [1−(1−sigma_min)t]x0 + t*x1; small t is near noise and large t exposes much of the target. Fixed-time panels are diagnostic, **not the native training distribution**. All full latent modalities receive paired noise, including dummy layout targets. The ordinary `generator.loss` is used; only its time, noise and zero shortcut-step draws are fixed. Unconditional dropout/self-consistency are disabled as in training.

Cross visual-present/visual-zero with correct/three-wrong surface sets for oracle, and with the fixed bank for constant. Defaults require1,280 batch forwards for oracle and320 for constant, batch4, with no backpropagation. Runtime is not yet measured on the cluster. Each arm uses one GPU. On two or more visible GPUs they run concurrently; otherwise sequentially. The wrapper waits for both children and refuses to summarize failed/incomplete runs.

Saved JSON contains every selected ID, input/target/token hash, actual time values and noise hashes, per-object losses, config/checkpoint/source hashes and restoration checks. The analyzer checks complete pairing across arms, then averages within each object before reporting mean losses and correct-versus-wrong/constant differences. It refuses to pool different actual visual/point/target/noise tensors; a hash failure is a pairing issue to inspect, not an experimental outcome. No p-values or proof of reconstructed shape follow from this loss-only probe.

## How results choose the next branch

- Benefit on training objects, absent on validation objects: target transfer of the aligned mapping. Do not switch to non-oracle rotation recovery.
- Benefit with visuals zeroed, absent with visuals present: examine multimodal interaction. Dropout alone has not established compatible fusion.
- Little correct-versus-wrong benefit even on training objects with visuals zeroed: audit actual aligned token construction, then compare successful tiny-fit optimization/exposure against this run before selecting an intervention. Nonzero gradients are insufficient.
- Useful correct-surface dependence and oracle advantage in both splits: verify actual Stage-1 sampled fidelity against the decoded target reference before declaring the upper bound working. A large wrong-surface penalty alone can reflect harmful distractors; constant and absolute fidelity comparisons remain necessary.

Fixed-time effects help distinguish condition utility at high noise from denoising when the target is already mostly exposed. Neither positive nor negative fixed-time velocity loss proves successful free generation. No training prescription is automatic from this probe.

## Sync and run

Sync these files together into the same investigation directory on the cluster:

- `checkpoint_probe_gpu.py`
- `checkpoint_probe_core.py`
- `analyze_checkpoint_probe.py`
- `protocol.py` (existing selection helper)
- `run_checkpoint_probe.sh`

The cluster should retain the production source already used for the three runs. No new source integration is needed. Default inputs are `outputs/conditioning_investigation/stage1_full_surface_oracle_dropout/{last.pt,config.yaml}` and `stage1_full_surface_oracle_constant_dropout/{last.pt,config.yaml}`.

In the active `sam3d-objects` environment and an existing GPU allocation, from the repository root:

```bash
bash doc/conditioning_investigation_2026-09-11/oracle_upper_bound/run_checkpoint_probe.sh
```

The wrapper's complete contents are provided in chat for direct interactive pasting. Outputs use a new timestamped folder below `outputs/conditioning_investigation`. Return the ZIP printed at the end; it contains the raw JSON reports and paired summary, with no checkpoints. If anything fails, return the traceback and any `results.partial.json` already written. Do not rerun training.

## Local verification scope

`test_checkpoint_probe.py` ran four tests under the existing temporary CPU Torch environment. They check actual bank preservation and intervention pairing through the original scalar-loss callback, restored monkeypatch state, no self-object distractors, constant swap invariance, correct summary signs, and rejection of missing/duplicate/unpaired cells. The model in the pairing test is synthetic. It does not certify SAM3D loading, CUDA numerics, cluster dataset availability, or full GPU execution.
