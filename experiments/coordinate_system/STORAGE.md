# Storage and cluster retention

## Current convention — updated 2026-09-14

This is the authoritative storage and organization reference for future work. Read it before creating an experiment, importing results, or reorganizing files.

Repository root on the laptop: `/Users/michaelwakeham/project/visuotactile-objects/sam-3d-touch`.

| Material | Location | Git policy |
|---|---|---|
| Significant experiment code | `experiments/coordinate_system/scripts/<experiment>/` relative to the repo root | Track code; group drivers, analyses, and tests by experiment. No loose top-level scripts. |
| Helpers shared across experiments | `experiments/coordinate_system/scripts/shared/` | Track; preserve dependencies when retaining an experiment. |
| Result-reading utilities / production integration tests | `scripts/tools/` / `scripts/integration_tests/` inside the investigation | Track these small Python files. |
| Findings, history, and plans | `KEY_FINDINGS.md`, `EXPERIMENT_HISTORY.md`, and `roadmap/` inside the investigation | Track Markdown. Keep the findings document at the root, without a one-document `findings/` folder. |
| Future diagnostic outputs on laptop or cluster | `experiments/coordinate_system/outputs/<experiment>/<run>/` relative to that checkout | Ignored; never commit reports, arrays, caches, checkpoints, or transfer ZIPs. |
| Full training outputs | Top-level `outputs/<run>/`; existing `outputs/conditioning_investigation/stage1_*` paths may remain | Ignored; reserve top-level `outputs/` for full training going forward. |
| Returned diagnostic evidence retained on laptop | `/Users/michaelwakeham/project/visuotactile-objects/coordinate_system_results/` | Outside the repo. Put future retained result imports here, not back under the tracked experiment folder. |
| Cleanup records, old source snapshots, and recovery information | `/Users/michaelwakeham/project/visuotactile-objects/coordinate_system_provenance/` | Outside the repo; do not pull onto the cluster as part of the code. |
| W&B exports on laptop | `/Users/michaelwakeham/project/visuotactile-objects/wandb-results/` | Outside `sam-3d-touch`; refresh with the sibling `export_wandb.py`. |

Keep significant experiments, including informative negative controls, with the helpers needed to use them. Remove retired/uninformative or superseded code only after checking dependencies; preserve its scientific outcome in the single experiment history. Do not replace useful runnable code with an archive. Full-training job scripts belong in `jobs/`.

### What actually moved

- **Completed locally:** `sam-3d-touch/experiments/coordinate_system/results/` → sibling `coordinate_system_results/` (about 291 MB). Reports, figures, array ZIPs, small reference inputs, and `index.json` are there.
- **Completed locally:** `sam-3d-touch/experiments/coordinate_system/provenance/` → sibling `coordinate_system_provenance/`.
- **Completed locally:** the retained scripts from `doc/conditioning_investigation_2026-09-11/` now live under `experiments/coordinate_system/scripts/`, grouped by experiment. The old documentation tree is no longer the active location.
- **Cluster moves are not confirmed.** No cluster files were moved by the assistant. Existing diagnostic folders may still be under `outputs/conditioning_investigation/`; inspect before assuming the new paths exist. Move diagnostic folders individually, while leaving full `stage1_*` training folders in top-level `outputs/`.

The result reader defaults to the sibling `coordinate_system_results/`. For another location, use `SAM3D_COORDINATE_RESULTS` or its `--results-dir` argument. A Git pull transfers code/docs, not these saved results. Historical paths embedded in returned JSON remain unchanged; use `index.json` and the reader to locate the retained bytes.

## Results are outside the repository

Saved results now live at `../coordinate_system_results/` relative to the repository root (291 MB). Scripts and documentation stay in Git. After GitHub rejected the large ZIP, the three unpublished cleanup commits were consolidated into `0662fdc` on 2026-09-14, preserving the final tracked tree and excluding results from the outgoing commits. Published remote history was not rewritten. The old local tip remains recoverable through `refs/backup/pre-large-file-cleanup-20260914-8418628`; the repair record is in the external provenance directory. Ignoring files alone does not remove files that were already committed.

## What changed locally

The old documentation collection held 2,648 files / 1.915 GB. Two referenced local full-frame output directories held another 0.208 GB. Significant evidence from both is now consolidated here. Repeated payloads resolve to one copy by SHA256; large arrays are losslessly compressed. The final measured totals and deletion counts are in [cleanup verification](../../../coordinate_system_provenance/CLEANUP_VERIFICATION.json).

The cleanup removed obsolete handoffs, caches, superseded partial reports, and selected retired raw payloads. Significant Python experiments and their dependencies are retained under `scripts/`, grouped by experiment. Retired readout/bridge/separate-attention code, superseded standalone training launchers, and one-off repair scripts have been removed. The removal list and rollback copy remain outside the repository. The underexposed augmentation treatment retains its summary/exposure result but not its large raw bundle. Every experiment family and its result remains in [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md). Significant negative results are retained; “negative” does not mean “uninformative.”

The [migration manifest](../../../coordinate_system_provenance/migration_manifest.json) records every original file's hash, disposition and retained location. A `duplicate_of` entry resolves to the exact same original bytes at another retained location. Array archives preserve bytes; returned report hashes and numbers have not been rewritten. Current Markdown links were updated separately. The external source snapshot preserves original bytes; the readable experiment code is restored under `scripts/`. Historical checks still refer to the original experiments and are not a generic assessment of arbitrary new checkpoints.

Earlier tracked files remain recoverable from the recorded Git commits and local recovery reference; Git object storage was not pruned. The only history change was consolidation of the three unpublished cleanup commits described above. Live documents are retained in the investigation, with original audit evidence and source snapshots outside the repository.

## What to retain on the cluster

**Do not delete all of `outputs/conditioning_investigation/`. The local collection contains no training `.pt` checkpoints or optimizer states.** Most heavyweight model state remains cluster-only. I have not inspected current cluster disk usage or unsynced files.

| Cluster material | Recommendation | Why |
|---|---|---|
| Full-trained oracle/dropout, camera/dropout and constant/dropout final checkpoints | Keep final checkpoint, saved config and run metadata. Keep a distinct best checkpoint if it represents a different useful evaluation point. | These are the matched full-run references; quantitative reports cannot recreate weights. |
| New oracle/no-PM, constant/no-PM and surface-only runs | Keep final/latest checkpoint, config and relevant logs; retain optimizer state while continuation remains an option. | Their scientific assessment is still incomplete and they are not mirrored locally. |
| Successful four-object oracle/dropout and its matched camera/dropout checkpoint | Keep one canonical evaluated checkpoint per arm and associated input/config metadata. | These provide the measured accurate fitted-identity reference and frame comparison. |
| Shared-frame unaugmented step-3,000 control | Keep its canonical checkpoint/config if reproducible re-evaluation is desired. | Useful existence proof that shape and orientation can be fitted; local geometry is preserved, model weights are not. |
| Dataset, source meshes, camera/object transforms, full surfaces, target latents and split manifests | Keep in the dataset location. | They are source data, not expendable experiment output. This cleanup does not certify every cluster record. |
| Native VecSetX reconstructed surfaces/fields from the representation probe | Keep if available and future encoder-fidelity investigation remains possible. | Only the quantitative report was supplied locally; those geometry assets were never mirrored. |
| Cached feature banks / new labels used by an experiment you still intend to resume | Keep until no longer needed or verify they can be regenerated from retained sources. | They are not automatically mirrored just because the result JSON is local. |

## What is usually safe to remove on the cluster

These recommendations concern completed investigations. **No cluster files were deleted.** Check for unique unsynced files before removing a whole directory.

- **Transfer ZIPs** after the corresponding files are preserved locally and verified. They are packaging, not checkpoints.
- **Exported predictions, repeated GT supports, coordinate arrays and report copies** whose exact bytes resolve as `retained` in the migration manifest. The oracle/constant full rollouts and both full-frame probe exports have retained local payloads. There is no reason to keep repeated uncompressed copies merely for local analysis.
- **Intermediate checkpoints from retired feature-readout, voxel-bridge and separate-attention variants**, if you accept retiring those models. Their outcomes are logged, but their weights are not backed up here. Deleting them means rerunning training if those exact models are needed again.
- **Intermediate rotation-augmentation checkpoints/expanded-label caches** from the underexposed, unselected treatment, on the same basis. Its diagnostic outcome is preserved; its raw experiment is intentionally retired.
- **Redundant earlier checkpoints of continuing full runs**, once a chosen final/evaluated state and any needed optimizer state are secure. Keep checkpoints needed for a specific learning-trajectory comparison; do not retain every step by default.

Known exported run roots relevant to the local copies:

```text
outputs/conditioning_investigation/full_checkpoint_probe_20260913_133239/
outputs/conditioning_investigation/full_checkpoint_rollouts_20260913_143515/
outputs/conditioning_investigation/full_frame_probe_20260913_170408/
outputs/conditioning_investigation/full_frame_probe_20260913_172925/
```

This list does not assert that every current cluster file under those roots was exported. Returned reports and their declared payloads were inspected; any additional feature caches, model files or later results may still be cluster-only. Keep those until separately classified. In particular, a folder name matching this list is not sufficient evidence for deleting everything in it.

## Retrieve an array without restoring the old directory tree

From the repository root:

```bash
python experiments/coordinate_system/scripts/tools/evidence.py --list train_g0_present_s0_d0.npz
```

The tool reads the external `coordinate_system_results/index.json`; it does not need the external provenance folder. Use an exact original path printed by the command:

```bash
python experiments/coordinate_system/scripts/tools/evidence.py \
  --extract doc/conditioning_investigation_2026-09-11/oracle_upper_bound/rollouts_returned_20260913_143515/oracle/train_g0_present_s0_d0.npz \
  --output /tmp/oracle_train_g0_present_s0_d0.npz
```

Extraction refuses to overwrite an existing file. The historical `doc/...` string is a result-index identifier, not a live filesystem dependency. Do not unpack the whole archive unless a specific analysis needs it.

## Output convention going forward

- Full training: repository-root `outputs/<run>/` (existing full-training paths need not change).
- Diagnostics: `experiments/coordinate_system/outputs/<experiment>/<run>/`.
- Curated returned evidence: `../coordinate_system_results/` (outside the repository).

On the cluster, move diagnostic folders individually into the experiment output directory and pass their new locations to scripts. Keep full training folders in top-level `outputs/`, even if their current parent is `outputs/conditioning_investigation/`. Do not delete cluster-only weights or caches simply because reports were copied locally. No cluster files were moved by this reorganization.
