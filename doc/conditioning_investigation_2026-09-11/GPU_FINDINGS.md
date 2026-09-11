# First GPU evidence and next test

## Material Passport

- Evidence: user-returned job `46083371`, NVIDIA H100 80 GB, torch 2.5.1+cu121, bf16, eight training objects × one view × four noise draws × four timesteps.
- Status: completed fixed-state GPU probe; no new training or sampled reconstruction results yet.
- Raw evidence: `gpu_probe_train_46083371.json`; reproducible analysis: `analyze_probe.py`; summary: `gpu_probe_summary.json`; attachment provenance: `gpu_input_provenance.json`.
- The two pasted files are the same experiment. All shared content is identical; only the completed file adds `paired_input_sha256`. They are not two seeds or replications.

## Validation

All 384 image-model rows and 640 full-surface-model rows are present, unique, finite, and nonnegative, with the same eight sample IDs and full time/draw/variant coverage. Donors differ from recipients. The completed file records the paired-input check. All six recorded source hashes match the current local files. Native versus explicit loss agrees to `1.49e-8` for the image model and exactly for the full-surface model on the first checked batch/state. This validates that state, not a whole sampled trajectory.

The returned pipeline YAML confirms `ObjectCentricSSI(use_scene_scale=true)` and no Stage-1 CFG override, so the inspected inference defaults apply: strength 7, interval [0,500], time scale 1000. The source inspection's isotropic SSI cancellation argument now has the runtime YAML behind it.

## Guidance is the strongest measured failure candidate

Full-surface checkpoint, mean velocity MSE:

| Time (0=noise, 1=data) | Conditional / CFG 0 | CFG 1 | CFG 7 | CFG 7 / conditional |
|---|---:|---:|---:|---:|
| 0.0 | 0.125264 | 0.138325 | 0.603320 | 4.82× |
| 0.2 | 0.096495 | 0.201425 | 5.129515 | 53.16× |
| 0.5 | 0.076661 | 0.970847 | 43.483178 | 567.21× |
| 0.8 | 0.107173 | 0.107173 | 0.107173 | 1.00× |

For all eight objects, CFG 7 increases mean error at each of the three active guidance times. At t=.8 guidance is inactive, and the outputs are exactly unchanged, as expected. The image model has the same large degradation: at t=.5, MSE rises from 0.075932 to 43.064349. This is a shared inference issue candidate, not a surface-specific effect.

Training adapts shared cross-attention under conditional-only flow loss; evaluation combines the conditional and zero-condition outputs as `8*c - 7*u`. The probe shows that this combination is badly matched to the known velocity targets at the measured states. **It does not establish why:** the zero-condition branch could have changed through finetuning, could already be unsuitable for this target distribution, or there could be other differences. The experiment did not compare pretrained versus finetuned unconditional predictions. Do not describe catastrophic forgetting of that branch as proven.

**It also does not prove a CD fix.** Guidance need not minimize supervised velocity MSE, and ordinary sampling visits different states from noised-ground-truth interpolation. The next test must be actual native guided/unguided rollouts. The first probe combined outputs algebraically in bf16 forward / float32 arithmetic; the next script additionally verifies the native CFG wrapper and uses the existing evaluation's configured sampling precision (float16 in this YAML).

## Surface identity produces a response, with little consistent accuracy benefit

| Time | Correct-surface MSE | Other-object-surface MSE | Relative change from wrong surface | Objects worse with wrong surface |
|---|---:|---:|---:|---:|
| 0.0 | 0.125264 | 0.125493 | +0.182% | 6/8 |
| 0.2 | 0.096495 | 0.096364 | −0.135% | 6/8 |
| 0.5 | 0.076661 | 0.077003 | +0.446% | 7/8 |
| 0.8 | 0.107173 | 0.107764 | +0.552% | 5/8 |

The t=.2 mean is negative despite 6/8 positive object differences because one object has a larger improvement with the wrong surface. This is why both magnitudes and individual-object behavior matter.

Swapping surfaces yields prediction-difference MSE 23–58 times larger than reversing slot order. Thus the model is not numerically indifferent to the supplied surface tokens. But a response is not necessarily useful shape conditioning: mean correct-versus-wrong error differences are below 0.6% and mixed across examples. This intervention keeps recipient image, pointmap, target and noise fixed, so it also introduces contradictory modalities. It is not a clean measure of information recoverable from the surface alone.

Slot reversal is not exactly invariant numerically: its prediction-difference MSE ranges from `6.39e-6` to `1.30e-4`. Source-level permutation invariance still holds in exact arithmetic. BF16 reduction/order effects are a plausible explanation, but repeat-forward/fp32 controls were not measured. Do not call this a clean runtime permutation pass, or use tiny differences as evidence of a new positional mechanism.

Even without CFG, the surface checkpoint does not clearly beat the image+pointmap checkpoint: relative conditional MSE differences are −0.10%, +1.52%, +0.96%, +2.29% at the four times. The models come from different validation-selected training steps (6597 and 8796). This is a descriptive checkpoint comparison, not a matched-initialization causal training ablation.

## What changes in the investigation

There are now two separate observations to explain:

1. Strong guidance greatly worsens fixed-state prediction error for **both** trained models. It could account for a common reconstruction-performance gap and prevents treating guided CD as a straightforward measure of training fit.
2. The full-surface branch has weak incremental correctness under the measured conditional objective. Turning guidance off cannot by itself explain or fix that already-visible limitation.

Coordinates remain unresolved. Neither run in this probe used oracle object-frame points, and no encoder was retrained. The data-scale and scratch-encoder hypotheses also remain untested. Full surfaces containing hidden geometry and having correct stored transforms do not guarantee that this frozen encoder/readout can learn the required target-frame relationship.

No population confidence intervals or significance claims are made: eight training objects are a pilot; noise draws are repeated measurements, and cyclic donor assignments couple the perturbations. Treat the observation sizes as descriptive.

## Run next: actual Stage-1 reconstruction, identical noise

Transfer the updated investigation folder to the cluster and run from the repository root:

```bash
sbatch doc/conditioning_investigation_2026-09-11/run_rollout_gpu.sh
```

This launches `rollout_gpu.py` with the same two checkpoints and eight sample IDs, two fixed seeds, 25 native solver steps, and CFG strengths 0, 1, 7. It requests one H100 with a two-hour time limit; this is an allocation limit, not a runtime estimate. There is no optimizer or checkpoint write.

The script reuses the native Stage-1 generator, inference override and decoder. It sets `no_shortcut=True`, preserves the evaluation time schedule/CFG interval, and defaults to the pipeline's sampling precision. Native input/noise/target-decode hashes must agree across checkpoints, and native CFG must match its stated formula on the first checked state. Nonfinite generated latents/decoder results are recorded as failures rather than silently turned into occupancy or excluded from denominators.

Outputs in `outputs/conditioning_rollout/train-JOBID/`:

- `results.json`: per-object, per-seed latent MSE, voxel IoU, unaligned occupied-voxel-center Chamfer, occupied counts, and any numerical failures.
- `.npz` artifacts: predicted/target latent tensors and occupancy grids for inspection.
- `partial.json`: completed measurements if interrupted. An interrupted run is not a completed comparison.

Return `results.json` and the Slurm log first; retain the NPZ files for inspecting examples. The script has been compiled and its launcher syntax-checked locally; GPU execution remains unverified. Existing training/evaluation source files were not changed.

The voxel-center CD is **not** directly comparable to the user's 0.0096 Stage-2 mesh CD. Its purpose is to isolate Stage 1 in the target's fixed coordinate system, without normalization, rotation fitting, pruning, or Stage 2.

If CFG 0 improves paired sampled Stage-1 geometry, test whether the same improvement survives the existing Stage-2 evaluation before declaring a final fix. If CFG 0 remains poor, prioritize the tiny-set camera-versus-oracle fitting experiment / free-token capacity control. If guidance removal fixes much of the common gap but surfaces still add little, pursue the encoder/frame issue separately rather than declaring the entire goal complete.

## Local reproduction

```bash
python doc/conditioning_investigation_2026-09-11/analyze_probe.py \
  doc/conditioning_investigation_2026-09-11/gpu_probe_train_46083371.json \
  --output doc/conditioning_investigation_2026-09-11/gpu_probe_summary.json
```

This analysis uses only the standard library. The archived completed JSON preserves the pasted bytes; input hashes document both attachments.
