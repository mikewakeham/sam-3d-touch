# Target coordinate convention: source audit and next discriminating test

## New local result

The suspected missing global axis conversion is **not supported**. All 32 locally available `object_transform.npz` files have a linear part equal to a positive scalar times

```
B = [[1, 0,  0],
     [0, 0, -1],
     [0, 1,  0]]
```

Maximum elementwise rotation difference: **0.0**. This is exactly the column-vector rotation used in stock `inference_utils.voxelize_mesh`: row vectors are multiplied by `B.T`. The custom target generator first applies the saved source-to-render transform, then voxelizes without a further rotation. The stock mesh exporter applies row-vector multiplication by `B`, which is the inverse conversion. Thus adding stock voxelization's rotation to the already transformed training mesh would apply it twice.

Several stock comments label both opposite operations “z-up to y-up.” The actual matrix maps +Y to +Z and +Z to −Y; comments alone were misleading. `frame_contract.py` extracts the stock matrix from its AST, verifies the export inverse, and checks the 32 saved transforms. `frame_contract_results.json` records source/input hashes. This establishes the shared **axis conversion**, not semantic front alignment, identical meshing/normalization in every path, or pretrained statistical compatibility.

No production coordinate change is justified by this result. Earlier camera inversion and reprojection checks also passed. Together these make a simple global sign/axis patch a weak explanation of the conditioning failure.

## Remaining coordinate hypothesis

Shared axes do not establish that an object's stored front direction matches the pretrained model's preferred orientation. A spatial target latent is orientation dependent. Previously returned predictions sometimes recovered much of a shape but required different Z rotations across objects and even noise seeds. Those were finetuned predictions and target-assisted alignments, so they did not establish a pretrained target convention mismatch.

The oracle experiments changed **surface coordinates**, while preserving the same target latents and frozen initial shape prior. They therefore did not test whether a different target orientation is substantially more compatible with that prior. Conversely, successful tiny fitting already rules out an absolute inability of the existing cross-attention adapter to represent these four examples. Neither a fundamental architecture failure nor a coordinate solution has been established.

## Prepared GPU probe — no training

From the cluster repository root:

```bash
sbatch doc/conditioning_investigation_2026-09-11/run_target_frame_probe.sh
```

Transfer `probe_target_frames_gpu.py`, `frame_contract.py`, and the launcher alongside the existing investigation scripts. This requests one H100 with a one-hour scheduler limit, not a measured runtime. It requires the original object meshes and `checkpoints/hf/ss_encoder.ckpt`, which are unavailable on this laptop. Outputs go to `outputs/conditioning_investigation/target_frame_probe/JOB_ID/`. Return `results.json` and the Slurm log first; keep the NPZs on the cluster.

The probe uses eight seeded training objects, two views each, the frozen pretrained image+pointmap model, and no VecSetX or optimizer. It reconstructs the original mesh occupancy and asserts that fp32 encoding reproduces the existing target latent. It then exactly rotates the **occupancy**, re-encodes it, and tests identity, three quarter-turn Z rotations, and ±90° X axis controls. Rotating a spatial latent directly is not valid without proving VAE equivariance. The integer occupancy rotation implementation has been checked against physical voxel coordinates and inversion for all 24 proper signed-axis rotations on CPU.

Primary measurements are per-object native shape flow errors at t=.2,.5,.8 with two noise draws, matching the training path and holding observations/model/noise fixed across target candidates. The first explicit loss is checked against `generator.loss`. The same zero layout targets used by the existing training setup are retained, so this tests compatibility within that setup, not the complete original pretraining objective.

Secondary checks use two 25-step native pretrained rollouts per input at configured CFG (default 7), compared against every candidate target. Candidate rotations do not change model outputs. VAE reconstruction and latent magnitude are logged for each orientation to detect apparent loss gains caused by degraded or lower-amplitude targets. No Stage 2 or final registered mesh CD is measured.

`analyze_target_frames.py` fixes the analysis before results arrive:

```bash
python doc/conditioning_investigation_2026-09-11/analyze_target_frames.py \
  outputs/conditioning_investigation/target_frame_probe/JOB_ID/results.json \
  --output outputs/conditioning_investigation/target_frame_probe/JOB_ID/analysis.json
```

For each object, choose a Z rotation using view 0 / noise draw 0 only; assess it on view 1 / independent draw 1. Also choose one global Z correction on half the objects and check it on the remaining half. Report paired object results, not stochastic draws as independent experimental units. These target-assisted choices are diagnostics, not inference algorithms or deployable evaluation improvements.

## What would constitute a concrete fix

- **A consistent global correction that transfers:** re-encode targets in that frame and transform object-frame surfaces and layout metadata consistently. Validate a short paired fitting intervention before changing the dataset. The source audit predicts no missing X conversion; the X controls check that prediction against the model.
- **Different but stable per-object corrections, with much better losses and sampled shapes:** the next candidate is consistent object canonicalization or orientation-aware shape supervision. A per-object corrected target is `E(voxelize(R_object mesh))`; oracle points must receive the same R, and camera extrinsics become `T_camera_from_object @ inverse(R)`. Camera observations stay as recorded. Existing latent channels cannot simply be rotated. This changes the supervision convention; it does not require concluding that full backbone finetuning is necessary. Choosing rotations from GT during diagnosis is not a deployment pose estimator.
- **Small, unstable or non-transferring improvements:** there is no convincing coordinate relabeling fix from this hypothesis. Do not launch a dataset-wide retrain or call this proof of impossibility. The investigation then needs to isolate the conditioning/optimization restriction responsible for the already observed visual-view dependence despite stable oracle features.

A successful compatibility probe would still require a small causal training intervention to show that correcting targets improves surface learning. A negative probe only rejects these candidate orientation corrections. Neither outcome alone proves that an architectural replacement is necessary.

## Validation and boundary

Executed locally: matrix/source audit, all 24 occupancy indexing checks, Python compilation, shell syntax, and synthetic checks that the analyzer chooses rotations without using the held view/draw. GPU code has not executed here. The next model-level evidence needs the cluster; no repeat full-dataset oracle run is requested.
