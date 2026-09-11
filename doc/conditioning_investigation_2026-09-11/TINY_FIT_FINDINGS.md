# Returned tiny-fit experiment: the surface pathway can fit

## Result

The fixed camera-frame surface arm successfully fits the four training objects using frozen pretrained VecSetX and trainable surface projection/shape cross-attention. Oracle coordinates and full-model finetuning are not necessary to pass this particular fitting sanity check.

At update 1,000:

| Condition | Native loss, fresh noise | Swapped-surface loss | Mean raw voxel IoU, CFG 0 | Mean raw voxel IoU, CFG 7 |
|---|---:|---:|---:|---:|
| Image + pointmap | 0.038466 | — | 90.070% | 96.155% |
| + Camera-frame full surface | 0.019949 | 0.061968 | 99.197% | 99.476% |
| + Oracle target-frame full surface | 0.018871 | 0.075295 | 99.188% | 99.645% |

Camera-frame conditioning lowers final fresh-noise loss by 48.1% relative to the image + pointmap baseline. It has lower loss on all eight paired assessment draws. Swapping its surface tokens across objects raises mean loss by 3.11×, with all eight draws worsening. Oracle swaps raise loss by 3.99×. These are repeated noise draws on four fixed examples, not eight independent objects or training replicas.

The oracle's final native loss is 5.4% lower than the camera arm's. That is a modest advantage, not a qualitative rescue from failure. At CFG 0 their mean IoUs differ by less than 0.01 percentage point, with camera slightly higher. One training seed does not justify a reliable general ranking between these arms.

The camera result is not an average hiding failures: across four objects × two noise seeds, minimum final CFG-0 IoU is 97.338%; minimum CFG-7 IoU is 98.474%. The baseline's worst CFG-0 IoU is 63.892%. Mean per-object CFG-0 IoUs:

| Object prefix | Image + pointmap | Camera surface | Oracle surface |
|---|---:|---:|---:|
| 479bbf90 | 96.22% | 99.49% | 99.57% |
| e846574d | 99.24% | 99.94% | 99.97% |
| fe200ce0 | 71.67% | 97.83% | 97.64% |
| 146b6c1f | 93.14% | 99.53% | 99.57% |

See `tiny_fit_curves.png`, `tiny_fit_summary.json`, and the three untouched returned JSON files under `tiny_fit_returned_46083371/`. Reproduce the standard-library analysis with `python doc/conditioning_investigation_2026-09-11/analyze_tiny_fit.py` from the repository root.

## Validity checks

- Cross-arm pairing passed for image, pointmap, target, initial trainable cross-attention weights, configs, logged source hashes, assessment steps, and sampling noise. Camera/oracle full-model initialization hashes match. All logged source hashes match the current local files.
- Each arm contains all 1,000 updates and all four assessment checkpoints; all 192 sampled metric rows are present and finite. Target occupied counts remain consistent across checkpoints, seeds, and guidance within each arm.
- Cross-attention weights changed. Surface projector gradients are nonzero; logged VecSetX encoder gradients are zero. Trainable parameter counts are 100,810,752 cross-attention/norm parameters plus 3,064,896 surface parameters for the surface arms.
- The original fitting script checks cached frozen features against the normal encoder forward pass. The selected normalized VecSetX bottleneck is deterministic in the inspected code. The diagnostic still differs from the production loop through fixed cached observations, batch size, and repeated presentation of only four examples.
- No new network execution occurred locally. The results are returned GPU measurements; NPZ predictions from this particular job have not been independently recomputed on the laptop. The three logged source hashes do not include the experiment driver itself.

## What changes in the diagnosis

The statement that full surface conditioning cannot overfit is contradicted by this experiment for these four cases. The model can recover almost the entire decoded target support, even with camera-frame surfaces and the existing frozen encoder. Strong guidance is also not required for this result.

Successful fitting plus swap sensitivity demonstrates that the fitted model uses information specific to the supplied surface tokens. It does **not** establish that the tokens are being interpreted as transferable geometry. Four fixed surface encodings could serve as object identifiers for memorization. Image-only fitting also improves substantially. More trainable surface parameters may contribute to faster fitting; there is no parameter-matched arbitrary-code control here.

This result rejects a universal hard incompatibility or universal need for full-model finetuning. It does not reject a capacity/optimization restriction at dataset scale, or a difficulty learning consistent geometry across camera views. A model can memorize four frame conversions while failing to learn a reusable conversion.

The original production runs trained for 20 epochs, and the non-position full-surface best checkpoint was selected at epoch 9 (step 6,597). The tiny test presents each exact object/view 1,000 times. An original epoch visits an exact record approximately once (distributed sampler padding can add duplicates). Different views also provide related object exposures; therefore 1,000 versus 9/20 is an exact-record exposure comparison, not a proof that more epochs solve the original task. These runs also differ greatly in the number of shapes whose gradients must be satisfied together.

The original aligned Stage-2 CD remains unexplained. Near-perfect Stage-1 support is strong fitting evidence, but it is not a measured Stage-2 mesh score and cannot be compared numerically with the user's 0.0096 reference. Avoid replacing the original evaluation or claiming its floor has been reached.

## Next bounded probe: transfer across views, using existing fitted weights

`probe_tiny_fit_views.py` does no training or decoding. It restores each returned `fitted_parameters.pt`, verifies initial/final parameter hashes and exact trainable archive keys, and requires the anchor's eight native losses to reproduce within 1% relative tolerance before accepting any new-view result. It also checks source/config and anchor input/feature hashes.

For each object it selects three distinct additional views deterministically from the existing training manifest. These views were not used in the tiny fit, even though the larger dataset calls them training records. Every batch retains the same four object targets in the same order. Native loss is assessed with the original eight noise/time seeds for:

1. Natural additional-view image, pointmap, and same-view surface.
2. Anchor image/pointmap with a different-view surface from the same object.
3. Different-view image/pointmap with the anchor surface.
4. Natural additional-view inputs with surface tokens swapped across objects.

The last three are for surface arms only. Crossed camera-frame conditions may contradict each other, so they measure sensitivity rather than natural reconstruction quality. Oracle-frame same-object feature differences are also recorded; large differences require checking normalization/sampling before interpreting them as a learned view effect.

Run after transferring `probe_tiny_fit_views.py` and `run_tiny_fit_view_probe.sh` (alongside their existing investigation imports):

```bash
sbatch doc/conditioning_investigation_2026-09-11/run_tiny_fit_view_probe.sh
```

The launcher defaults to the returned `outputs/frame_tiny_fit/46083371` directory; pass another fit-root path as its first argument if the cluster location differs. It requests one H100 for at most one hour, a limit rather than a measured runtime. Return `outputs/tiny_fit_view_probe/JOB_ID/{image,camera,oracle}.json` and the Slurm log.

If anchor losses reproduce but natural other-view loss increases substantially, the demonstrated capability is narrower than same-object view transfer. Comparing the controlled interventions can locate image/pointmap versus surface sensitivity. A failure on unseen views does not establish inability to fit multiple views: the next training experiment would hold object count fixed and add views. If transfer is already strong, increasing object diversity or inspecting original last-versus-best training checkpoints becomes more informative.

The new probe passed Python compilation and shell syntax checks locally; it still requires GPU execution. No full-model finetuning or production changes have been added.
