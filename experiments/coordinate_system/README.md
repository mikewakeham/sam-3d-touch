# Coordinate-system investigation

Stage-1 full-surface conditioning in SAM3D, retaining VecSetX and a future sparse-touch path. This is the authoritative home of the former `doc/conditioning_investigation_2026-09-11` collection.

**Before adding code or importing results, read [the current storage and organization convention](STORAGE.md#current-convention--updated-2026-09-14).** It records exact local paths, future output locations, and which cluster moves are still unconfirmed. Saved results do not belong in Git.

**Current position:** oracle transformation and normalization arithmetic pass on the audited observations. The fully trained oracle uses real surface information but does not reach an accurate reconstruction upper bound. Coordinate-dependent learning difficulty is not comprehensively excluded. The refreshed no-pointmap/no-visual runs completed training without a pooled-loss breakthrough; their checkpoint reconstruction assessments are still missing. The user subsequently authorized a larger camera-frame target / shared-pointmap-normalization rerun; implementation is ready, GPU results are pending.

**Current work, 2026-09-14:** [formal camera-frame rerun](scripts/camera_frame_formal/README.md): 16 objects, eight training and four held-out views each; no unseen objects, three matched arms, 1,000 updates at batch 4, loss validation every 100; reconstruction deferred until after training. Standalone interactive experiment, local JSON logs, one GPU per arm; optional experiment-local submission script; local logging without W&B. Preparation/training code and CPU checks are complete; no local GPU run. Production code is unchanged.

**Earlier status, 2026-09-14:** the user stopped the surface-only assessment and requested its removal. The roadmap below retains earlier proposals as reference, not active instructions. The three H100 shape-full runs finished and their W&B curves are recorded; further training-scope/dropout analysis is deferred on the [roadmap TODO list](roadmap/DECISION.md#current-priority-after-the-refreshed-overnight-results). No new checkpoint reconstruction was supplied by the W&B sync.

| Read / use | Contents |
|---|---|
| [Key findings](KEY_FINDINGS.md) | Established results and their limitations |
| [Reference conventions](roadmap/CONVENTION_SURVEY.md) | Paper/code comparison: asset axes, semantic up/front, cameras, and normalization |
| [Experiment history](EXPERIMENT_HISTORY.md) | What each experiment tested and found |
| [Decision](roadmap/DECISION.md) and [checklist](roadmap/COORDINATE_CHECKLIST.md) | Remaining questions and stopping rules |
| [Scripts](scripts/README.md) | Restored experiments, their helpers, analyses, and tests |
| External `../coordinate_system_results/` (from repo root) | Saved reports, figures, and arrays; not stored in Git |
| `outputs/` | New diagnostic run outputs, caches, and diagnostic checkpoints (ignored by Git) |
| [Storage](STORAGE.md) | Full-run versus diagnostic storage and cluster retention |

Full training outputs stay in the repository's top-level `outputs/`. Experiment outputs go in `experiments/coordinate_system/outputs/<experiment>/<run>/`.

Cleanup provenance was moved outside this repository to `../coordinate_system_provenance/` (relative to the repository root). It is not required by the result-reading tool and will not be pulled to the cluster with this repository.

## External results layout

- `coordinate_contract/`: independent frame/normalization checks and native VecSetX/target-reference reports.
- `fitting/`: tiny fitting, frame factorial, visual dropout, new-identity and constant controls.
- `geometry/`: saved pose/shape comparisons, oracle rotation robustness and camera/dropout failures.
- `shared_frame/`: informative shared-camera-target fitting/transfer results; underexposed augmentation raw payloads were removed.
- `full_training/`: full-run losses, oracle/camera/constant rollouts and exact-bank geometry audit.

Saved results were moved out of the repository to `../coordinate_system_results/`. Readable JSON reports and images remain normal files there. Repeated raw arrays are stored once in each family's `arrays.zip`; the external `index.json` resolves cross-family duplicates too. **Returned JSON is immutable:** paths/hashes inside it describe the historical run, not executable paths in this reorganized repository. Use the evidence tool to resolve those original file names. Do not run old commands copied from historical reports.

## Local checks

From the repository root, using an environment with NumPy:

```bash
python experiments/coordinate_system/scripts/tools/evidence.py --verify --metrics
```

Set `SAM3D_COORDINATE_RESULTS` or pass `--results-dir` if the external results are stored elsewhere. They are not downloaded by a Git pull.

This verifies original retained bytes and recalculates the 1,280 full-run occupancy IoUs/counts. It does not rerun models or registration. For production-integration checks, use a CPU PyTorch environment:

```bash
python experiments/coordinate_system/scripts/integration_tests/test_source_integration.py -v
python experiments/coordinate_system/scripts/integration_tests/test_no_visual_integration.py -v
python experiments/coordinate_system/scripts/integration_tests/test_shape_full_integration.py -v
```

The distributed tests require local loopback socket access (on macOS set `GLOO_SOCKET_IFNAME=lo0`). The shape-full suite exercises the actual dense transformer at reduced width without pretrained weights; it is not a CUDA memory test. Restored experiment sources are ordinary Python files under `scripts/`; see its index for experiment families and reproduction requirements.
