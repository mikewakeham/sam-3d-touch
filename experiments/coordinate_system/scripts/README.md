# Significant experiment code

Storage and future organization follow [STORAGE.md](../STORAGE.md). Keep code grouped by experiment; keep generated artifacts out of Git and retained laptop results outside the repository.

Scripts are grouped by the experiment they support. Shared numerical routines and rollout helpers are in `shared/`; result utilities are in `tools/`. Each group keeps its associated analysis and tests. Imports between groups use explicit package paths, and scripts can still be invoked directly from the repository root.

| Folder | Why it is retained |
|---|---|
| `initial_probe/` | Initial condition sensitivity and saved-geometry evidence (E0–E1) |
| `coordinate_audit/` | Camera inverse, axes, normalization, and pose-observability checks (E2/E4) |
| `prior_target_frame/` | Tested whether changing the target orientation fixes the pretrained-prior discrepancy (E2) |
| `tiny_fit/` | Single-view fitting, multiple views, and view-change controls (E3–E5) |
| `frame_factorial/` | Separates rotation from normalization treatment (E4) |
| `visual_dropout/` | Visual-stream dependence and matched dropout training (E5) |
| `identity_transfer/` | New identities, constant-surface controls, and checkpoint trajectories (E6–E7) |
| `native_representation/` | Native VecSetX/target reconstruction measurements; excludes the retired readout and bridge experiments (E9) |
| `alignment_robustness/` | Rotation perturbations and separation of pose error from shape error (E10a) |
| `camera_dropout/` | Matched camera-versus-oracle dropout comparison (E10b) |
| `shared_orientation/` | Common-frame targets; `scope/` holds the finetuning-scope comparison, and `visuals/` the visual-zero probe (E10c) |
| `rotation_coverage/` | The successful unaugmented continuation control and its paired, underexposed augmentation treatment (E10c) |
| `full_checkpoints/` | Completed full-run fixed-state probes, rollouts, geometry bounds, and W&B analysis (E11–E13) |
| `full_frame_comparison/` | Matched full-trained camera/oracle comparison and independent coordinate reference (E14) |
| `integration_tests/` | Current production dropout/constant/no-visual training plumbing (E15) |
| `shared/` | Geometry, sampling, dropout, packaging, and source-location helpers imported by several groups |
| `tools/` | Read/verify compressed results and inventory training checkpoints |

## Removed

- Direct feature-readout and explicit voxel-bridge experiments.
- Separate-surface-attention training and analysis.
- The superseded full-dataset loss probe and standalone oracle training implementation; production training remains in `train.py` and `jobs/`.
- One-off repair/resume/subset checks and obsolete initial bundle packaging.

These outcomes remain in [EXPERIMENT_HISTORY.md](../EXPERIMENT_HISTORY.md). Original code and the removal record are outside the repository in `../coordinate_system_provenance/` (relative to the repository root). Their removal does not erase negative findings. The rotation-coverage driver is retained because the unaugmented arm provided a significant fitted upper-bound result; its augmentation arm remains inconclusive.

## Invocation and storage

Run from the repository root. Diagnostic `--output-dir` / `--output` paths belong under `experiments/coordinate_system/outputs/<experiment>/<run>/`. Full training and its checkpoints stay in the repository's top-level `outputs/`.

For example, the relocated tiny-fit driver is `experiments/coordinate_system/scripts/tiny_fit/tiny_fit_gpu.py`. The retained-result reader is `experiments/coordinate_system/scripts/tools/evidence.py`.

Historical sequential experiments still enforce their recorded reference-bank, source-hash, initialization, and checkpoint checks. Do not disable those checks to run against different sources or checkpoints. Old paths inside returned JSON identify historical artifacts; use the result reader/index for their retained locations. Grouping code does not recreate cluster-only weights or establish that an old protocol applies to a new run.

The file moves change imports and source-file lookup, not experiment settings, objectives, or acceptance checks. GPU experiments have not been rerun as part of this organization.

## Current formal rerun

[`camera_frame_target_latent/`](camera_frame_target_latent/README.md) contains per-view target preparation, transformation figures, three-arm training with shared pointmap normalization, and CPU checks. User authorized implementation on September 14; GPU execution pending. Loss validation only during training; reconstruction afterward. Run directly in an interactive GPU allocation; optional experiment-local three-job submission script; no W&B integration. Results go in the ignored `experiments/coordinate_system/outputs/camera_frame_target_latent/`.
