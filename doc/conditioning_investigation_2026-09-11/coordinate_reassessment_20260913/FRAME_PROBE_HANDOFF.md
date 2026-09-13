# Complete the existing full-data frame comparison

Status: first cluster run completed all32 target encodings and64 reference observations, then stopped at the nearest-query gate before generator inference. See `FRAME_REFERENCE_RETURNED_FINDINGS.md` for the verified return and correction. Continuation now reuses the independently audited reference arrays and performs the unchanged camera comparison. It does not repeat mesh queries, seeded sampling, or target encoding. No new fitting, target convention or production edit.

## Questions and exact operations

1. **Fresh-run input contract (not repeated on resume):** for the exact previous16 train/16 validation identities and two selected views each, load the normalized source mesh using target-generation code. Regenerate the saved target latent with the frozen Stage-1 encoder (or reuse the verified completed encodings). Independently repeat seeded mesh sampling, retain each sampled face, and require recovered points to match both their sampled reference and a float64 convex-combination witness on that actual mesh face. Save the triangles and witnesses. Nearest-surface queries remain recorded diagnostics; their first-run disagreement is no longer the membership gate. A changed sampling order deliberately stops this exact-seed protocol rather than silently weakening it.
2. **Actual runtime path:** load the completed camera/dropout last checkpoint (epoch20/step14660). Use the production camera preparation and `get_touch_tokens`, capturing actual VecSetX inputs. Also compute oracle preparation for auditing, and require its points hash to match the prior oracle probe. Save camera points before/after SSI/VecSetX normalization and oracle points/normalization. This fixed bank has8192 valid points and the configured encoder accepts8192, so the no-resampling normalization check explicitly rejects a changed point count rather than pretending to verify FPS.
3. **Frame treatment:** camera checkpoint losses use the exact previous native/fixed-time banks, visuals present/zero, correct/three cyclic wrong surfaces. Free-running sampling uses exact prior noise,25 steps,CFG0, correct/one fixed wrong surface, visuals present/zero. Compare against saved oracle/constant results, not fresh unpaired references.

The reference mesh and target are used only for supervision-path audit/scoring, never as camera conditioning. No Stage2, final-mesh generation or CD. Target encoder use is verification of the existing Stage1 labels, not training.

## Compute and outputs

One visible GPU suffices. A fresh execution first regenerates32 targets; the current continuation accepts only the pinned, independently audited170408 report and verifies unchanged source/data/encoder/artifact hashes. It copies the saved reference arrays without invoking mesh libraries or the target encoder. It then loads the existing camera model and decoder, producing512 generated shapes (128 batches of4 ×25 sampling forwards) plus1280 batched loss forwards. Mesh checks run on CPU. Wall-clock time has not been measured on the cluster.

The shell wrapper saves a log, partial report on in-driver failures, current source copies, and a ZIP even when the probe returns nonzero. The bundle contains camera predicted latents/supports, decoded target supports, actual coordinate arrays, normalized-mesh references and source/input hashes. It copies four old JSON reports for pairing but does not duplicate their large NPZ payloads or model/optimizer checkpoints. Existing oracle/constant NPZ bundles already available locally supply pose-adjusted references.

After return, independently verify arrays/hashes and compute camera proper-rigid geometry under the exact procedure used for oracle/constant. The printed cluster summary is raw-frame only; do not declare intrinsic-shape superiority before that local check.

## Gates and their interpretation

- Match data config/manifests/splits, global batch16/world size4, training exposure, learning rates, no-position/full shape-CA/frozen-VecSetX settings, and observation/target/noise hashes. Different surface points and tokens are intended; different visuals/targets/noise are not.
- Target regeneration uses `rtol=1e-4, atol=1e-5` in latent units. Record actual max error. Numerical disagreement is a failed verification, not automatically corrupt target geometry.
- Oracle distance to a witness on its independently sampled source triangle and maximum coordinate difference from the seeded point must each be below5e-5 normalized object units (0.0032 voxel). The original distance limit is unchanged. Actual GPU oracle preparation must agree with float64 inversion within5e-6. GPU normalization is checked independently within5e-6. These are arithmetic/geometry consistency gates, not generated-shape success thresholds.
- Known mesh vertices pass the existing closest-point positive control; translating them four object units fails by over two units. The new source-triangle check also rejects a four-unit translation of every observed point. Convex-combination witnesses establish mesh membership without relying on global nearest-face selection.
- If a geometry/reference gate fails, generation does not proceed. Return the resulting bundle to identify whether provenance, runtime numerics or data itself caused the failure. Do not relax gates or rerun training blindly.

## Outcome branches

- Bad input/target contract: identify and fix that issue before interpreting frame-treatment effects.
- Oracle advantage disappears after rigid adjustment: mostly a measured output-pose benefit; do not credit it as equivalent shape improvement.
- Oracle improves adjusted shape: full-data frame-treatment benefit established at the observed scope; aligned residual remains a separate problem.
- Camera matches/beats oracle with useful correct-surface dependence: exact supplied alignment is not a demonstrated net full-data remedy. The tiny factorial is not a universal effect estimate.
- Ordering changes with visual-present/zero: frame effect interacts with visual availability; motivates a targeted fusion/visual test, not immediate blame of pointmap coordinates.

Wrong-surface sensitivity alone is insufficient; retain the trained constant comparison and absolute target-decoded reference. One seed and a repeatedly used development sample cannot prove population equivalence or architecture impossibility. Camera/oracle treatment includes normalization/encoder effects, not pure rotation in isolation.

## Sync and interactive execution

Sync these four new files into this folder on the cluster:

- `frame_probe_gpu.py`
- `frame_probe_core.py`
- `analyze_frame_probe.py`
- `run_frame_probe.sh`

Existing `oracle_upper_bound` helpers must remain unchanged: their hashes are checked against the completed probes. Keep both previous output folders, the original camera/oracle run folders, and `outputs/conditioning_investigation/full_frame_probe_20260913_170408/camera`. The updated wrapper passes that last folder to `--resume-geometry-from`. Use the active `sam3d-objects` Python environment. Full shell contents are in `run_frame_probe.sh` and are pasted in chat; no `sbatch` or new allocation script is required.

## Local validation

The current resume integration also passes three regressions: actual returned artifact reuse without CUDA/Open3D, rejection of a changed current target, and rejection of an unaudited report. External cluster file hashes are mocked in the local integration test because those files are unavailable; all96 real returned artifacts are verified and copied. The original13 CPU regressions passed before the first handoff. The repair passes eight local frame tests: five pairing/normalization tests plus three source-triangle tests covering known distances, collapsed/thin triangles, float32 camera storage, and wrong frame/seed rejection. Saved payload hashes and target/inverse/seeded-point metrics reproduce for all32 objects/64 observations. Driver import/CLI and Python/shell syntax pass. Fresh source-face checks remain unexecuted but are no longer a prerequisite for this audited continuation. CUDA encoder-input inspection and camera inference remain pending. New code bundles and returned archives are kept under `outputs/conditioning_investigation/`; the historical `frame_probe_code.zip` is superseded by `frame_probe_resume_20260913.zip` there. This supersedes the earlier repair bundle, whose resume path still repeated mesh checks.

## Resume evidence policy (current)

The170408 report SHA256 is pinned to `c62912b9045eef005f66c282832e1098e15180519a6c8aae40de9b055f4266fa`. Its independently verified exact target equality and seeded source-mesh correspondence are accepted as the raw input reference. The failed nearest-query metrics remain unchanged in the new report, with `original_nearest_query_gate_passed: false` and an explicit acceptance basis. `coordinate_reference_passed` means the raw reference is accepted on that independent evidence; it does not mean the failed nearest queries passed. Arbitrary failed reports cannot use this path.

This is a restart correction, not new scientific evidence. No raw-oracle alignment experiment is repeated. Actual production encoder-input capture remains part of the pending model comparison.
