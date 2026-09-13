# Exact-bank reference return and gate correction

**Current execution decision:** the source-triangle recheck described below was
implemented as a proposed repair, then removed from the resume prerequisites after
reviewing its redundancy. The raw target/seeded-mesh evidence is already sufficient
for this scoped reference. The current resume path accepts only that exact audited
report, verifies unchanged source/data/artifact hashes, and proceeds to the camera
model without mesh queries, resampling, or target encoding. It preserves the failed
nearest-query diagnostics and explicitly records the alternate acceptance basis.
This changes execution policy, not the observations below. Original source-face
witnesses remain unmeasured and the query discrepancy mechanism remains unknown.


Returned bundle: `outputs/conditioning_investigation/full_frame_probe_20260913_170408.zip`.
Moved from the project root at the user's request; extracted next to the ZIP.
SHA256: `753a8cc2b2e43170f5492064a0db7e5ef5e1eeb4de16eaffe134eee4822633b7`.
The pasted partial JSON equals the archived report. All96 reported NPZ hashes pass.
Reproduction: `audit_returned_frame_reference.py`; output saved inside the returned
folder as `local_reference_audit.json`.

## What executed and what failed

All32 source-mesh target encodings and64 view reference checks executed. Every
regenerated target latent is **bitwise equal** to its saved training target
(maximum error0), stronger than the permitted numerical tolerance. No camera
generator inference executed: input/loss/sampling result lists remain empty.

Eleven views failed the nearest-point distance gate of5e-5 object units.
Maximum reported distance was0.000643137883 (0.0412 voxel). However, independently
replaying the mesh's original seeded sampling and indexing by the stored point IDs
matches **every view**: largest component error1.2114519965e-7, largest Euclidean
error1.2258291214e-7. Float64 camera inversion and all these values reproduce from
the saved arrays. This is correspondence to separately sampled mesh coordinates,
not merely a transform followed by its own inverse.

Recomputing float64 witnesses on the nearest-query-selected triangles fixes some
discrepancies but leaves others. In particular, the largest case remains about
0.000643136. Thus correcting arithmetic on the selected face alone is insufficient.
The exact internal reason for the query disagreement has not been established;
do not assert a specific Open3D bug, degenerate-face mechanism, or input corruption.
Original full meshes are not available locally. Their actual sampled faces were
not exported by the first version, so the final source-face check remains pending.

## Correction and preserved controls

The source mesh is resampled using the same seed/count, now retaining sampled face
IDs. For each recovered point, compute a float64 point on its sampled source face
using convex combinations of that triangle's vertices. Require both distance to
that witness and seeded correspondence error below the **unchanged5e-5 limit**.
A witness on any actual source face proves proximity to the mesh; a global nearest
search is unnecessary. Clamped barycentric and edge candidates may overestimate
distance but cannot invent an off-mesh witness. A translated negative control must
fail. Save sampled triangles and witness coordinates for independent verification.

Nearest-query results remain in each report/NPZ as diagnostics. No points, target
latents, normalization, model inputs, weights, or learning settings are changed.

`--resume-geometry-from` verifies the previous report's reference/checkpoint/config/
dataset/pipeline/encoder/source/version hashes, payload hashes, complete identity
sets, current mesh/transform/target/surface/camera hashes, and exact seeded mesh
resampling. It reuses only completed target encodings/occupancy; checks the current
targets against the saved and regenerated arrays; and repeats mesh/point checks.
It writes a new output directory and keeps the failed run intact. Changed inputs
are rejected rather than silently reusing stale results.

## What this narrows and what it does not

The exact-bank evidence strongly supports correct raw oracle-to-target-mesh
alignment, including translation/scale, before VecSetX normalization. The error
message is not evidence that the training target has the wrong frame. It records
a failed measurement gate whose nearest-query output conflicts with seeded mesh
correspondence. Require the repaired source-face witnesses before marking the
independent membership gate complete.

This is not proof of a working conditioner or full coordinate closure. Actual
CUDA/VecSetX prepared-point replay is still pending, as is the matched full-data
camera-versus-oracle comparison. Subsequent raw and common rigid-adjusted shape
comparisons distinguish a measured output-pose benefit from a benefit beyond pose.
Neither result alone explains the remaining aligned-generation error or proves
encoder/generator incompatibility. Keep F18/F19's positive geometry utility and
unmet accurate upper bound in view. No new fitting or representation branch is
selected by this gate failure.

## Validation and handoff

Eight local frame tests pass, including wrong scale/axes/translation/correspondence,
off-plane points, collapsed/thin triangles, and float32 camera-storage perturbation.
Driver imports and CLI work locally. Repaired real source-face execution and
generator sampling need the cluster; they have not been tested on this laptop.
Run the updated `run_frame_probe.sh` contents in the existing interactive allocation.
Its reuse input is the170408 camera folder. All new ZIPs/output folders are under
`outputs/conditioning_investigation/`.
