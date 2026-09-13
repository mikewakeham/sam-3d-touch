# Full-data frame comparison: completed results

Returned bundle: `outputs/conditioning_investigation/full_frame_probe_20260913_172925.zip`.
SHA256: `5d83953a6b853f841124cc67239a237636890bf3202f93c4e07a2260452c59c1`.
The bundle was moved from the project root into the investigation outputs folder.
Both pasted JSONs equal their archived versions. All256 reported payload hashes,
runtime source copies, prior source hashes, and the raw summary reproduce. No new
training occurred. The resume path did not repeat mesh queries or target encoding.

Local reproduction: `analyze_camera_frame_return.py`. Numerical outputs are under
`outputs/conditioning_investigation/full_frame_probe_20260913_172925/local_analysis/`.
Only JSON/JSONL analysis products were added, not copied geometry or checkpoints.

## Question and design

Measure the already-trained oracle frame treatment at full-data training scale.
Compare camera/oracle/constant final epoch20, step14660 checkpoints with the same
16 train and16 validation objects, two views, two generation noises, visuals
present/zero, and correct/wrong-object surfaces. The new camera arm contributes
512 Stage-1 outputs plus paired native/fixed-time denoising losses. Oracle/constant
outputs and rigid-registration results are reused from F18/F19.

Camera versus oracle includes their separately learned weights and frame-dependent
normalization/encoder responses. It is not a pure rotation intervention holding
learned weights fixed. One training seed and a repeatedly used development bank
limit population inference; views/draws are averaged within objects.

F-score uses occupied Stage-1 voxel centers, precision and recall at two voxels
(2/64 object units), and their harmonic mean. The decoded target compared with
itself is100%; this defines the reference, not a guarantee that every held-out
object is recoverable from finite surface samples. No Stage2 or CD is involved.

## Actual encoder-input check

All64 actual production camera/oracle normalization arrays independently reproduce.
Maximum production oracle vs independent raw reference discrepancy:1.59441e-7.
Maximum camera/oracle normalization errors:1.67201e-7 /1.42144e-7.
Inverse errors:2.28072e-7 /4.65822e-8 object units. Actual oracle point hashes and
visual/target/noise hashes match the previous oracle experiment. All8192 points
are valid and no FPS resampling is used in this bank.

This completes the previously missing numerical runtime check at the tested scope.
It does not demonstrate neural access to coordinate information or equate the
VecSetX normalized coordinate convention with the target grid. Oracle geometry-only
recanonicalization has maximum point displacement0.61008 voxel in this bank.
The generic helper also emits a camera recanonicalization-vs-raw displacement;
that includes camera translation/frame changes and is NOT a normalization-error
metric or evidence of a huge camera bug.

## Raw and common proper-rigid results

All512 camera raw support scores reproduce from arrays. The unchanged F19 registration
code is used: multistart proper rotation/translation, no scaling/reflection/warps.
Retain identity if it gives a better minimum precision/recall than the registration
candidate. Oracle/constant use their existing results from the identical procedure.
This is a common approximate pose search, not globally optimized pose-invariant
scoring or an impossibility proof when a candidate remains poor.

Visuals present; mean object F-score (%):

| Split / scoring | Constant | Camera | Oracle |
|---|---:|---:|---:|
| Train, raw |63.72|64.14|75.87|
| Train, rigid-adjusted |79.14|81.17|84.52|
| Validation, raw |49.69|51.02|64.51|
| Validation, rigid-adjusted |68.10|67.83|77.23|

Oracle-camera raw difference is11.73pp train,13.49pp validation; positive on14/16
and15/16 identities. Under common rigid search the difference is3.35pp train,
9.41pp validation; positive on11/16 and13/16 identities. This supports a frame-treatment
benefit beyond the output-pose benefit found by this search, especially on held
objects. Do not subtract these differences and call the remainder an exact causal
fraction mediated by pose: registration and the score are nonlinear/approximate.

Visuals zeroed, rigid-adjusted correct-surface F-score:
train camera46.23%, oracle79.01%, constant28.18%; validation35.98%,56.67%,24.95%.
Oracle-camera differences32.78pp/20.70pp, positive on16/16 and13/16 objects.
Removing visuals exposes a larger dependence on frame treatment, but oracle's
absolute fidelity worsens without visuals. This does not show that removing
visual conditioning fixes the oracle residual or that pointmaps specifically cause it.

## Useful surface information versus added capacity

Visual-present, rigid-adjusted correct/wrong surfaces:

| Split | Camera correct | Camera wrong | Oracle correct | Oracle wrong |
|---|---:|---:|---:|---:|
| Train |81.17|63.36|84.52|51.55|
| Validation |67.83|65.27|77.23|59.15|

Oracle correct-minus-wrong validation difference18.08pp, positive14/16 objects;
camera difference2.55pp, positive10/16. Camera correct also has essentially no mean
advantage over the trained constant control on visual-present held objects
(-0.27pp; positive9/16). This is not evidence of literally zero camera geometry
use: training effects and visual-zero results show dependence. It shows much
stronger transferred utility in the aligned treatment at this measured endpoint.
Wrong-surface harm alone is not sufficient proof of net benefit; the constant
control remains necessary. This experiment does not sample the historical
image-only or image+pointmap checkpoint, so do not claim superiority to them.

## Why loss curves looked similar

Visual-present native validation loss is0.105266 camera and0.105603 oracle, so
oracle is slightly worse on this aggregate despite its sampled fidelity advantage.
At t=.05 the ordering reverses:0.180833 camera versus0.176377 oracle. At t=.5/.95
oracle is worse. This supports the earlier finding that pooled denoising loss can
hide condition utility that is visible in generation. It does not establish a
particular timestep reweighting prescription.

## What is narrowed, and where to stop repeating checks

1. Wrong raw oracle transform and arithmetic errors in the actual audited encoder
   preparation are strongly disfavored. Ordinary rotation/translation/scale
   bookkeeping has a defensible stopping point on this bank.
2. Frame treatment has a practical full-training benefit; this was not established
   by the prior oracle/constant-only comparison. The selected branch is the
   reassessment's oracle-improves-adjusted-shape branch, not the null-frame branch.
3. The accurate aligned upper bound remains unmet. Even on train objects, adjusted
   oracle F-score is84.52%, and only2/16 object means have both precision and recall
   at least95% (also2/16 validation). The95% criterion is an operational fidelity
   marker, not a universal learnability bound. Existing F19 extent evidence already
   gives one certified limitation beyond arbitrary rigid pose; do not rerun it merely
   to restate that some residual errors remain.
4. The residual is not yet localized among restricted shape-generator adaptation,
   optimization/exposure, feature spatial access, or transfer. Known geometric
   recoverability from points does not certify easy recovery from feature tokens.
   Earlier oracle direct readout fitting argues against absolute representation
   incompatibility, not for a ready reusable mapping.

Retain oracle as the reference treatment. Do not repeat transform audits, prescribe
inference pose recovery as the missing solution, or replace VecSetX based on these
results. The next diagnostic should address learning the already-aligned mapping
with the coordinate/target convention fixed. A matched continuation comparing
current versus broader shape adaptation is a candidate informed by prior scope
experiments; it must control extra exposure and test correct-surface utility,
not merely reward added capacity. No new training script is issued by this analysis.

## Storage audit and retention decision

This archive is143.70MiB; expanded files147.25MiB. Its NPZ categories include:
49.51MiB already-returned reference geometry,27.26MiB coordinate arrays,
7.70MiB target arrays,59.33MiB predictions. Predicted latent tensors contribute
57.84MiB, whereas predicted occupancy payloads needed by current pose analysis
are only1.41MiB. Packaging the entire folder was overcollection for this analysis.
Raw latents were retained for possible later decoder/latent diagnostics; that is
not justification to transfer them by default.

The older local investigation directory is about1.8GiB. Uncompressed target and
prediction occupancy files total1592.20MiB; byte-identical duplicates account for
693.09MiB targets and124.02MiB predictions. Existing ZIP plus extracted copies add
further duplication. This is a local inventory, not measured cluster storage.
The current probe saves no checkpoints; production training retains `last.pt` and
`best.pt`, including optimizer state. Their cluster sizes have not been measured.

Future handoffs should send JSON/logs first, reuse prior reference arrays by hash,
and add only the compact new occupancy arrays needed for the stated analysis.
Dense target arrays should be stored once per distinct target, with compressed
boolean/sparse output instead of uncompressed per-draw copies. Raw latent exports
must have an explicit analysis purpose and stay optional. Keep one reference copy;
do not automatically duplicate it into every archive. No files or checkpoints were
deleted by this audit, and no production training source was changed.
