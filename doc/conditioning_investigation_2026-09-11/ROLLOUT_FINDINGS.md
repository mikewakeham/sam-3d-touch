# Rollout results: guidance removal is not the fix

## Material Passport

- Evidence: completed user-returned Stage-1 rollouts, eight training objects, two paired noise seeds, CFG strengths 0/1/7, 25 native solver steps, float16 sampling.
- Raw results: `gpu_rollout_train_46083371.json`; analysis: `analyze_rollout.py`; summary: `gpu_rollout_summary.json`; attachment hashes: `rollout_input_provenance.json`.
- Status: sampled Stage-1 results analyzed; geometry artifacts not yet received; root cause unresolved.
- Both pasted files describe the same experiment. The second adds the final paired-input digest; it is not an independent replication. The output directory reuses the earlier job number; the number alone does not independently establish Slurm provenance. No Slurm log was supplied.

## Checks and measured results

Both checkpoints have all 48 expected rows (8 objects × 2 seeds × 3 strengths), no duplicate measurements, no recorded numerical failures, and finite metrics. Target occupied counts agree across conditions. Native CFG formula checks report exactly zero difference. The completed output records input/noise/target-decode pairing, and all six source hashes match the current local files. NPZ metrics have not yet been independently recomputed.

| Checkpoint | CFG | Latent MSE ↓ | Voxel IoU ↑ | Unaligned voxel-center CD ↓ |
|---|---:|---:|---:|---:|
| Image + pointmap | 0 | 0.21150 | 0.12509 | 0.07586 |
| Image + pointmap | 1 | 0.22054 | 0.14059 | 0.07632 |
| Image + pointmap | 7 | 0.23457 | 0.19609 | 0.05300 |
| Image + pointmap + full surface | 0 | 0.21791 | 0.10209 | 0.09120 |
| Image + pointmap + full surface | 1 | 0.21813 | 0.13771 | 0.06459 |
| Image + pointmap + full surface | 7 | 0.23076 | 0.16679 | 0.06025 |

CFG 7 versus 0 improves mean voxel-center CD by 30.1% for the image model and 33.9% for the surface model. Within each model it improves seed-averaged CD on 7/8 objects and IoU on 6/8. It increases mean latent MSE. Thus the direction of latent/velocity error does not predict the direction of decoded geometric quality in this pilot.

**Correction to the first interpretation:** calling the large guided fixed-state velocity error a “substantial guidance problem” was too strong. The measured error was real, and the native wrapper reproduces it, but disabling guidance worsens sampled geometry here. The hypothesis that simply disabling CFG fixes the observed reconstruction gap is contradicted on this pilot. This does not establish that 7 is globally optimal or that every individual sample benefits.

A likely interpretation is that local squared-error fidelity and decoded support quality reward different outputs; guidance also changes the trajectory and support density. This is an interpretation, not a demonstrated mechanism. We have not measured latent-distribution geometry, decoder sensitivities, or pretrained-versus-finetuned unconditional behavior. Do not claim that the unconditional branch is broken based on the first probe.

This CD compares occupied voxel centers against the decoded target in its original frame. It is not the Stage-2 aligned mesh CD reported as approximately 0.0096 / 0.036–0.040. Keep these metrics separate.

## Object-level evidence changes the next step

The full-surface model remains worse on mean CFG-7 CD (0.06025 versus 0.05300) and IoU (0.16679 versus 0.19609), but the difference is heterogeneous. Median object-mean CD is very similar: 0.02186 surface versus 0.02239 image. These are eight training objects, two repeated seeds, and different validation-selected checkpoint steps; no population or controlled training effect is established.

Examples at CFG 7:

- `fe200ce00a5d4c298eee89de0fc15f01_011`: surface CD is 0.02569 for seed 29 but 0.29160 for seed 30; image CD is approximately 0.023 for both. One severe surface failure can dominate an average.
- `c5867189a04446d7b20b79fcc24c6afa_004`: image CD is 0.20161 for seed 29 but 0.02239 for seed 30; surface CD is approximately 0.024 for both. Severe seed sensitivity is not unique to the surface model.
- `479bbf9062de4b3196f08e52d1c12821_000`: both models consistently have CD around 0.13 and IoU around 0.05.
- `e846574df9334973955c80286af47ba7_007`: both models consistently have CD around 0.10 and IoU around 0.08–0.10.
- `146b6c1f10bd445da977424c64dbb051_002`: a low-CD control, about 0.009–0.011 with both models.

These observations justify inspecting actual support geometry before choosing another training intervention. High unaligned CD can result from frame/pose errors or incorrect geometry; scalar metrics alone cannot distinguish them. Seed-sensitive errors could reflect alternate orientations, shape failures, translations, or support outliers. None is proven yet.

## Next action: send a selected geometry bundle, no GPU required

Transfer `bundle_geometry.py` into the investigation folder on the cluster. Run from the cluster repository using Python (standard library only):

```bash
python doc/conditioning_investigation_2026-09-11/bundle_geometry.py \
  --rollout-dir outputs/conditioning_rollout/train-46083371 \
  --output outputs/conditioning_geometry_bundle.zip
```

The source directory above is the exact directory recorded in the supplied JSON. If the files were moved, point `--rollout-dir` to the directory containing the actual `results.json` and model subdirectories. The default probe path is the archived completed probe beside this script. A `--data-root` override is available if the data were moved.

Send `conditioning_geometry_bundle.zip`. It gathers:

- 20 CFG-7 NPZ files: five selected cases × two models × two seeds.
- Their image, pointmap, camera transform, full surface, object transform, and target latent.
- Probe/results metadata and a manifest with byte hashes.

It reads existing files and creates one new ZIP; it does not load networks, train, or launch GPU work. The script was syntax-checked and tested locally using synthetic input files: all 52 source files were archived with matching byte hashes, ZIP integrity passed, and a missing required input failed before writing an archive. Actual cluster paths still require execution there.

The case selection is deliberately based on observed outcomes for diagnosis; it must not be presented as an unbiased evaluation subset.

## What will be checked once the bundle arrives

1. Verify archive hashes and independently recompute raw IoU/CD from support arrays.
2. Compare saved raw target latent with each NPZ target; invert full-surface camera transforms and compare the target decoded support to the observation geometry. This checks whether a gross target-frame mismatch is already present before considering predictions; point samples and decoded occupancy will not match exactly.
3. Inspect orthogonal projections/3D geometry, centroids, bounds, occupied counts, and directional distance distributions, including outliers.
4. Test known saved camera rotations and axis transformations as diagnostic alternatives. Report original metrics alongside transformed metrics. A fitted rotation improving CD does not prove a coordinate bug; specific consistent transforms across cases provide stronger evidence than independent best-fit transforms.
5. Compare both seeds for the failure cases to determine whether the same target shape appears in different frames, or whether geometry itself changes.

If a consistent transform explains errors, follow it through generation/target/conditioning code before changing production preprocessing. If geometry is simply wrong or frame differences are not systematic, proceed to the tiny-set fitting comparison: matched camera-frame and oracle object-frame conditioning, fresh-noise validation, shared initialization, and an image baseline. A one-object memorization success alone cannot establish useful point conditioning because image-only can also memorize one target; follow it with multiple objects and point-identity controls.

No new GPU training is requested at this step. The investigation is waiting for existing geometry artifacts, not more source-only speculation. No root-cause fix or inherent architectural limitation has been established.
