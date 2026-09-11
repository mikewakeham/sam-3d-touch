# Touch-conditioning context pack

This directory is a reusable technical handoff for adding touch conditioning to
SAM 3D Objects Stage 1. It records what is actually present in the fork and the
local data snapshot, what can be concluded from the released code, and what still
depends on unavailable checkpoint configuration or a future design decision.

A 2026-08-30 follow-up audited the official sparse-structure VAE artifacts, released
generator YAML, target voxelization paths, and renderer/camera transforms. The target
latent and coordinate-frame contract below supersedes earlier “target unresolved”
statements preserved in the initial audit history.

The pack was produced on **2026-08-30** from fork commit
`f91db411c50efee93d8db7aeb323885650f6f722` on branch `touch-training`.
The fork had no tracked modifications at the start of the audit. The pre-existing,
untracked `SAM3D_DATA_CONVENTIONS.md` is user-owned and was read but not changed.

## How to interpret this pack

The integration direction is intentionally not frozen here. Statements use these
labels:

- **Confirmed**: observed in the audited code, files, arrays, or an identified
  official source.
- **Inference**: a conclusion drawn from confirmed behavior, with the reasoning
  stated.
- **Provisional**: a plausible implementation direction, not a requirement.
- **Open**: cannot be resolved from the current local artifacts, or requires an
  experiment/product decision.

Future agents should revalidate commit-dependent claims after code or data changes.
In particular, they should not treat the proposed encoder, unfreezing policy,
target representation, or milestone ordering as immutable objectives.

## Documents

1. [Repository and evidence map](01_repository_and_evidence.md) — repository state,
   important files, available/missing artifacts, and source-of-truth hierarchy.
2. [Data pipeline and touch format](02_data_pipeline.md) — rendering, frames,
   pointmaps, touch sampling, manifest behavior, real sample statistics, and loader
   implications.
3. [Stage 1 model internals](03_stage1_model.md) — inference control flow,
   conditioning tokens, MoT attention, VAE targets, pose conventions, and flow loss.
4. [Training integration design space](04_training_integration.md) — missing
   infrastructure, insertion seams, parameter-freezing choices, target generation,
   validation, and a provisional implementation sequence.
5. [Open questions and execution checklist](05_open_questions_and_checklist.md) —
   exact items to settle before implementation and the checks that should gate each
   phase.

The broader [SAM3D data conventions](../../SAM3D_DATA_CONVENTIONS.md) document is a
useful companion for generic SAM 3D input/output conventions. It is not a native
upstream file and is currently untracked.

## Executive summary

### What already exists

- The fork contains the complete released **inference** architecture for Stage 1:
  image/mask/pointmap preprocessing, condition embedders, the multimodal MoT flow
  model, sparse-structure VAE encoder/decoder, pose converters, checkpoint loading,
  and the generic rectified-flow loss implementation.
- The external data tree contains deterministic object rendering, pointmap
  generation, and dense geodesic touch-neighborhood generation. The real local
  example has 16 views and 32 hidden-surface contacts per view in touch format v4.
- The condition fuser already supports a list of token-producing modalities and,
  in its normal mode, concatenates their tokens. This is the cleanest released seam
  for adding a touch stream.
- Each Stage 1 MoT block cross-attends every latent stream to the same fused condition
  context. Exact condition-sensitive parameter paths are documented in
  `03_stage1_model.md`.
- The sparse-structure encoder can create an 8-channel, `16^3` latent target from a
  `64^3` occupancy tensor. Deterministic cached targets should use the posterior
  **mean**, not a sampled `z`.
- The target is one object-shared `mean [8,16,16,16]` in renderer-normalized object
  coordinates. The loader flattens it to `[4096,8]` in `(x,y,z)` order.
- Target generation needs only the original OBJ, the saved
  `T_normalized_from_source`, and the frozen official SS encoder. Existing render,
  pointmap, camera, and touch arrays do not need regeneration.

### What does not exist

- `make_data.py::make_stage1_target` is a placeholder and every inspected manifest
  row has `target_path: null`.
- There is no released task-specific `Dataset`, collator, training entry point,
  optimizer setup, distributed trainer, checkpoint/resume layer, touch encoder,
  touch-aware inference API, or hidden-geometry evaluation suite.
- Checkpoint payloads are still ignored by the fork. The SS VAE files and released
  generator YAML were audited outside it, resolving the target and core generator
  dimensions. Full instantiated condition-token counts, pipeline-selected pose/
  pointmap behavior, and parameter paths still require runtime probes.
- The local data snapshot has only one complete generated object and no source OBJ.
  Its full manifest refers to the cluster dataset, so local end-to-end target
  generation is not possible from this snapshot alone.

### Highest-risk technical points

1. **Target frame must not be transformed twice.** Apply the renderer's saved
   `T_normalized_from_source` exactly once, then voxelize directly in fixed normalized
   object bounds. Calling the stock voxelizer afterward would repeat axis conversion/
   normalization and can misalign shape, pointmap, and touch.
2. **Shape-only supervision is not automatically supported by the released wrapper.**
   `FlowMatching.loss` can recurse over dictionaries, but the MoT backbone asserts
   that every configured input latent is present. A custom loss/input strategy or
   valid targets for all configured latent streams may be required.
3. **Condition length has no attention mask.** Variable contact counts cannot be
   padded harmlessly: even zero tokens receive key/value bias and softmax mass.
   Fixed contact-token counts are the safest first implementation.
4. **Unfreezing cross-attention is a design choice, not a theorem.** A new touch
   encoder can learn to emit features compatible with frozen condition projections.
   Unfreezing shared `to_kv` changes how both image and touch tokens are interpreted.
5. **The touch orientation is synthetic.** Its third axis is the center face normal,
   but the tangent roll is selected from object-coordinate axes. Supplying the full
   rotation can expose canonical object orientation/pose beyond local geometry.
6. **Current neighborhoods are not hidden-only.** Centers are hidden, while the
   default `neighborhood_mode="all"` includes hidden, visible, and unknown master
   points inside the geodesic ball. This matters for leakage claims and ablations.

## Minimum context a future agent should load

Before changing code, read all five numbered files in this directory, then inspect:

- the actual downloaded checkpoint YAMLs;
- `model.named_parameters()` and condition-token shapes from one real forward hook;
- `make_data.py`, `sample_touch.py`, and one current `touches.npz`;
- any changes since commit `f91db41`;
- the latest decision record in `05_open_questions_and_checklist.md`.

Use the audited target dimensions directly, but continue resolving condition widths,
token counts, pose behavior, and exact trainable parameter paths from the instantiated
modules/config. Fail with a clear diagnostic when the runtime checkpoint differs from
the recorded artifact hashes or expected target contract.
