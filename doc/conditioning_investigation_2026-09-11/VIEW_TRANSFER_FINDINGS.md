# Returned view-transfer probe: fixed-view fitting did not generalize

## Validated result

The experiment-driver hash and source-fitting-result hashes match. All three arms reproduce their eight anchor losses **exactly**, beyond the probe's required tolerance. The four batches have paired sample IDs, RGB hashes, pointmap hashes, and identical target hashes across arms. All 240 scalar loss measurements are present and finite.

The extra views are unseen by the tiny fitting experiment, but belong to the larger dataset's training manifest. Every target is still one of the same four fitted object latents. This is a controlled test of view transfer, not new-object generalization and not a measurement of the original full-dataset checkpoints.

Mean native losses (the three additional-view groups are equally weighted):

| Condition | Image + pointmap | Camera surface | Oracle surface |
|---|---:|---:|---:|
| Original fitted observation | 0.03847 | 0.01995 | 0.01887 |
| Other-view image + pointmap + corresponding surface | 0.10101 | 0.13283 | 0.12773 |
| Original image + pointmap, other-view same-object surface | — | 0.04620 | 0.01887 |
| Other-view image + pointmap, original same-object surface | — | 0.12597 | 0.12771 |
| Other-view inputs, surface from wrong object | — | 0.13436 | 0.15464 |

Fresh-noise loss rises 2.63× for the baseline, 6.66× for camera surfaces, and 6.77× for oracle surfaces when the natural view changes. Both surface models have worse loss than the baseline in each of the three new-view groups. These are results for one training seed and four objects; the 24 paired draws are not 24 independent objects or training replicas.

## What the oracle intervention establishes

Oracle surface feature MSE between views is only 3.08e-7–4.43e-7, compared with 0.620–0.767 for camera-frame features. Camera-frame features are expected to change under rotation; their change alone is not evidence of encoder corruption.

For the oracle model, exchanging the surface for another view of the **same** object leaves loss essentially unchanged: 0.0188706 → 0.0188659, holding image/pointmap fixed. Conversely, keeping the original surface and changing image/pointmap raises loss to 0.127705, almost exactly the 0.127726 loss when both are changed naturally.

Thus changing surface frame/features is not necessary for this oracle model's loss increase. The image/pointmap condition changes while the target, surface, and noise/time draws are fixed. This directly localizes a strong sensitivity to that condition bundle in these fitted weights. RGB and pointmap were changed together; it does not distinguish their individual effects or preprocessing interactions.

A useful interpretation is that the one-view model learned an association involving the particular image/pointmap and surface, rather than a view-independent mapping from complete surface information to the target. That is a behavioral interpretation, not proof of a specific internal lookup mechanism. It could still represent geometry while relying excessively on other conditions.

The surface is not simply ignored on new views. In the oracle arm, wrong-object surfaces increase average loss from 0.12773 to 0.15464 (21.1%), with 21/24 paired draws worsening. In the camera arm, the mean increase is only 1.2%, with 15/24 draws worsening; group 2 actually improves with wrong-object points. These contradictory-modality interventions measure conditional dependence, not independent surface reconstruction accuracy.

## Implications for the original concern

The previous tiny fit showed near-complete recovery of target support with camera-frame surfaces, frozen VecSetX, and the existing trainable projector/cross-attention. The new experiment shows that successful fit did **not** imply robust surface use across views. Providing the full shape does not force the trained network to use it independently of the image/pointmap.

This narrows the investigation beyond a simple global coordinate conversion. An oracle transform stabilizes the full-surface features but does not prevent the view-transfer loss increase. It does not rule out frame difficulties at dataset scale or an advantage from oracle coordinates when training on many views.

Crucially, the original full-dataset model already trained on multiple views. A one-view model failing on additional views does not prove why that original model underperformed. The next experiment must test **fitting multiple views**, rather than infer a training-capacity limitation from this transfer test. No full-model finetuning requirement, production fix, or explanation of the original aligned Stage-2 CD has yet been established.

## Next experiment: directly fit four views per object

`fit_multiple_views_gpu.py` keeps the four objects, fresh pretrained initialization, frozen VecSetX, trainable projector/full shape cross-attention, optimizer rates, batch size four, noise seeds, and total 1,000 updates. It changes the training observations from one to four views per object: the original anchor plus the three views just probed. Three further deterministic views per object are reserved from optimization.

- Fit records: 16. Reserved-view records: 12. All refer to the same four object targets.
- Every update includes one view of each object. A balanced four-batch cycle gives 1,000 updates per object and 250 appearances per exact training record. This holds object exposure and total updates fixed relative to the single-view experiment; it does not hold exact-view exposure fixed.
- Eight fixed fresh noise/time draws are evaluated on every fit and reserved batch at updates 0, 100, 300, and 1,000. Surfaces are also swapped across objects. Reserved-view losses are never used for training or checkpoint selection.
- The initial anchor loss must reproduce the previous experiment. Initial parameter hashes, anchor inputs/features, targets, configs and logged source hashes are checked. Final parameters and component gradients are recorded.
- Final CFG-0 rollouts, two matched noise draws for every fit/reserved group, provide a secondary raw Stage-1 geometry check. There is no Stage 2 or geometric alignment.

Run after transferring the new scripts alongside their existing investigation imports:

```bash
sbatch doc/conditioning_investigation_2026-09-11/run_multiple_view_fit.sh
```

The launcher starts three fresh arms sequentially, using the prior JSON reports as provenance references rather than resuming their fitted weights. It defaults to reference directory `outputs/frame_tiny_fit/46083371`; another root can be passed as the first argument. One H100 is requested for at most four hours; this is an allocation limit, not a measured runtime. Return `outputs/multiple_view_fit/JOB_ID/{image,camera,oracle}/results.json` and the Slurm log. Retain NPZ files on the cluster initially.

If all fitted views achieve low loss but reserved views do not, multi-view fitting is possible but view generalization remains weak. If oracle substantially improves multi-view fit over camera, the frame burden remains relevant. If both surface arms fail to fit the training views despite the single-view positive control, inspect optimization and competing view constraints before changing encoder/backbone scope. If fit and transfer both become strong, object diversity/exposure at the original scale becomes the next controlled variable.

The new script passed compilation and shell syntax checks locally; selection/count/disjointness checks were run without GPU dependencies. It has not been executed on a GPU. No production training files were changed.

## Reproduction and artifacts

`view_probe_returned_46083371/` preserves both attached JSON files exactly. `image.json` is transcribed from the inline user message; its whitespace is reconstructed. See that directory's provenance note. `analyze_view_probe.py` verifies the pairing, full row coverage, finite values, exact anchor reproduction and hashes, then writes `view_probe_summary.json`.
