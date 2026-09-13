# Complete the existing full-data frame comparison

Status: implemented under the investigation folder only. GPU execution pending; no new fitting, target convention or production edit. See `REASSESSMENT.md` and F20 for the reason this takes priority over metadata/shared-target training.

## Questions and exact operations

1. **Input contract:** for the exact previous16 train/16 validation identities and two selected views each, load the normalized source mesh using target-generation code. Regenerate the saved target latent with the frozen Stage-1 encoder. Measure oracle-transformed points against the mesh with CPU Open3D closest-surface queries; save the closest triangles/points for independent local checks. This is independent of merely applying a transform and its own inverse. Seeded surface resampling is saved as supporting evidence, but seed-replay differences alone do not reject the contract because sampling/library ordering can differ.
2. **Actual runtime path:** load the completed camera/dropout last checkpoint (epoch20/step14660). Use the production camera preparation and `get_touch_tokens`, capturing actual VecSetX inputs. Also compute oracle preparation for auditing, and require its points hash to match the prior oracle probe. Save camera points before/after SSI/VecSetX normalization and oracle points/normalization. This fixed bank has8192 valid points and the configured encoder accepts8192, so the no-resampling normalization check explicitly rejects a changed point count rather than pretending to verify FPS.
3. **Frame treatment:** camera checkpoint losses use the exact previous native/fixed-time banks, visuals present/zero, correct/three cyclic wrong surfaces. Free-running sampling uses exact prior noise,25 steps,CFG0, correct/one fixed wrong surface, visuals present/zero. Compare against saved oracle/constant results, not fresh unpaired references.

The reference mesh and target are used only for supervision-path audit/scoring, never as camera conditioning. No Stage2, final-mesh generation or CD. Target encoder use is verification of the existing Stage1 labels, not training.

## Compute and outputs

One visible GPU suffices. The same device first regenerates32 targets, then loads the existing camera model and decoder. It produces512 generated shapes (128 batches of4 ×25 sampling forwards) plus1280 batched loss forwards. Mesh checks run on CPU. Wall-clock time has not been measured on the cluster; no speculative runtime promise.

The shell wrapper saves a log, partial report on in-driver failures, current source copies, and a ZIP even when the probe returns nonzero. The bundle contains camera predicted latents/supports, decoded target supports, actual coordinate arrays, normalized-mesh references and source/input hashes. It copies four old JSON reports for pairing but does not duplicate their large NPZ payloads or model/optimizer checkpoints. Existing oracle/constant NPZ bundles already available locally supply pose-adjusted references.

After return, independently verify arrays/hashes and compute camera proper-rigid geometry under the exact procedure used for oracle/constant. The printed cluster summary is raw-frame only; do not declare intrinsic-shape superiority before that local check.

## Gates and their interpretation

- Match data config/manifests/splits, global batch16/world size4, training exposure, learning rates, no-position/full shape-CA/frozen-VecSetX settings, and observation/target/noise hashes. Different surface points and tokens are intended; different visuals/targets/noise are not.
- Target regeneration uses `rtol=1e-4, atol=1e-5` in latent units. Record actual max error. Numerical disagreement is a failed verification, not automatically corrupt target geometry.
- Oracle point-to-mesh distance must be below5e-5 normalized object units (0.0032 voxel). Actual GPU oracle preparation must agree with float64 inversion within5e-6. GPU normalization is checked independently within5e-6. These are arithmetic/geometry consistency gates, not generated-shape success thresholds.
- Known mesh vertices pass a closest-point positive control; translating them four object units must fail by over two units. Mesh sampling is not used as the sole nearest-distance reference. API verified against [Open3D distance-query documentation](https://www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html).
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

Existing `oracle_upper_bound` helpers must remain unchanged: their hashes are checked against the completed probes. Keep both previous output folders and the original camera/oracle run folders. Use the active `sam3d-objects` Python environment. Full shell contents are in `run_frame_probe.sh` and are pasted in chat; no `sbatch` or new allocation script is required.

## Local validation

Thirteen CPU tests pass: five new tests using real archived report schemas and explicitly synthetic camera values, plus eight existing noise/loss/sampling regressions. New checks allow intentional frame differences, reject changed visual/target/oracle-reference tensors, reject missing/duplicate/changed-noise/wrong-identity/nonfinite samples, enforce loss pairing/completeness, and reject incorrect normalization scale/translation/axes and changed point count. The GPU driver imports and `--help` executes locally; Python and shell syntax pass. GPU mesh/encoder/pipeline execution remains untested locally. Tests do not establish experimental outcomes. See `frame_probe_local_checks.json`. `frame_probe_code.zip` contains the new runtime files and this handoff, with repository-relative paths; unzip it from the cluster repository root after transferring it there.
