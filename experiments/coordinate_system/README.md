# Coordinate-system investigation

Stage-1 full-surface conditioning in SAM3D, retaining VecSetX and a future sparse-touch path. This is the authoritative home of the former `doc/conditioning_investigation_2026-09-11` collection.

**Current position:** oracle transformation and normalization arithmetic pass on the audited observations. The fully trained oracle uses real surface information but does not reach an accurate reconstruction upper bound. Coordinate-dependent learning difficulty is not comprehensively excluded. Latest oracle/no-pointmap poor performance remains user-reported without imported checkpoint assessment. No next experiment is automatically queued.

| Read / use | Contents |
|---|---|
| [Key findings](KEY_FINDINGS.md) | Established results and their limitations |
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
```

The distributed test requires local loopback socket access. Production training and `jobs/` are unchanged. Restored experiment sources are ordinary Python files under `scripts/`; see its index for experiment families and reproduction requirements.
