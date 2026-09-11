# Geometry bundle and training-fit handoff

## Findings from the returned bundle

The ZIP's integrity check and all 52 manifest file hashes passed. It contains five deliberately selected diagnostic cases, with two checkpoints and two noise seeds at CFG 7: 20 saved predictions. This is a selected diagnostic subset, not an estimate of dataset-wide prevalence.

`analyze_geometry.py` independently reproduced all 20 saved latent MSE, raw occupied-voxel CD, and IoU measurements. Saved target latents exactly match the dataset target arrays. Target occupancies agree across runs/seeds. Full surfaces transformed back from SAM camera coordinates align with decoded targets in all five examples; mean Euclidean surface-to-occupied-center CD is 0.00625–0.00908. These distances reflect different point samplings and voxelization as well as geometric discrepancies; they are not a new VAE reconstruction floor or the user's Stage-2 CD.

Several high raw reconstruction errors are primarily orientation errors:

| Example / checkpoint / noise seed | Raw Stage-1 CD | CD after best proper axis rotation |
|---|---:|---:|
| Shield `479bbf90`, image, 29 | 0.13232 | 0.00860 |
| Shield `479bbf90`, surface, 29 | 0.13983 | 0.01040 |
| Bathtub `e846574d`, surface, 30 | 0.11236 | 0.01418 |
| Drill `fe200ce0`, surface, 30 | 0.29160 | 0.02712 |
| Knife `c5867189`, image, 29 | 0.20161 | 0.00512 |
| Knife `c5867189`, surface, 29 | 0.02464 | 0.00310 |

The first five listed corrections are +90 degrees about Z; the last is 180 degrees about Z. The drill's other three predictions require no correction. The low-error control `146b6c1f` requires no proper-axis correction. The knife's image seed 30 also prefers 180 degrees, unlike image seed 29. Thus a single global axis conversion would not correct these outputs. Residual geometric errors remain, especially for the drill.

The script searches all 48 signed axis permutations and reports the best of the 24 proper rotations separately. A few reflections score slightly better on near-symmetric shapes; this is not evidence of a handedness bug. These are target-assisted diagnostic alignments, not corrected model outputs or deployable performance. Orthogonal projections and full numerical results are in `geometry_analysis/`.

**This does not explain the original aligned Stage-2 score by itself.** The production evaluator already estimates rotation/alignment. Its mesh CD and the user's 0.0096 reference cannot be compared numerically with these raw Stage-1 occupied-center distances. The useful discovery for training is that some latent errors encode incorrect target orientation despite substantial recovery of shape.

## Why orientation remains relevant to training

The dataset target is camera-view independent, but not rotation invariant. `dataloader.load_target` loads an 8×16×16×16 spatial grid and flattens its spatial locations into 4096 tokens. The target occupancy stays in the normalized source object's coordinate frame, and all camera views share that target. An object pointing along X occupies different grid locations from the same object pointing along Y. Native flow loss compares the corresponding latent components; there is no rotation-invariant matching in that loss.

Full surfaces enter in camera coordinates. Center/radius normalization removes global translation and isotropic scale, but does not undo camera rotation. Mapping those observations to the target therefore includes recovering the object's saved orientation. This is a learning burden, not proof of corrupt data. Arbitrary or ambiguous object orientations may also be difficult to infer from observations; the present subset does not establish an irreducible ambiguity.

The supplied stock generator YAML has shape loss weight zero and nonzero layout losses. The custom training builder explicitly replaces these weights with shape-only loss. This is consistent with investigating a pointmap pathway adapted for layout, but one configuration does not prove the full pretraining history. The source feeds fused conditions into the shape-capable backbone; there is no basis here for asserting that pointmaps cannot influence shape. Winning with image + pointmap also does not independently establish a useful shape contribution from the pointmap.

## Next GPU experiment: training fit, with geometry as a secondary check

From the cluster repository root, after transferring the investigation scripts:

```bash
sbatch doc/conditioning_investigation_2026-09-11/run_tiny_fit.sh
```

This requests one H100 for at most four hours; that is a resource limit, not a measured runtime. Three sequential arms start from fresh pretrained SAM3D:

1. `image`: image + pointmap, no surface.
2. `camera`: image + pointmap + camera-frame full surface.
3. `oracle`: same observations, with full surface transformed into the target's object frame using the known camera transform.

Each fits the same four fixed training objects/views for 1,000 updates. VecSetX stays frozen; the touch projector and full shape cross-attention train with the existing optimizer settings (1e-4 and 1e-5, respectively). Position conditioning is disabled. Frozen preprocessing/encoder features are cached; the trainable projector is recomputed inside every loss graph. Its cached path is checked against the normal encoder forward pass. This tests the currently used frozen-encoder setup, not scratch training or full-model capacity.

**Primary outcome:** native shape flow loss on eight fixed fresh stochastic draws at steps 0, 100, 300, and 1,000, for the same objects being fitted. Training uses different, explicitly matched stochastic seeds. These fresh draws test fitting across noise/time rather than memorization of one noisy input; they are not unseen-object validation.

At each checkpoint, cyclically swap the surface tokens across the four examples and repeat those same loss draws. Worsening with swaps supports useful dependence under this intervention, although the modalities are deliberately contradictory and the result does not establish geometric generalization. Fitting alone could succeed by memorizing the images.

**Secondary outcome:** paired 25-step native samples at CFG 0 and 7, two fixed noise draws, with saved latents and decoded raw support. This verifies whether loss improvements translate into recovering the target and distinguishes orientation from other shape error. It does not run Stage 2.

The script logs gradients by component, clipping norms, parameter changes, source/config provenance, and input/target hashes. The launcher runs `check_tiny_fit.py` to verify matching initial cross-attention weights, camera/oracle full initialization, observations, targets, and sampling noise across arms. Decoder/nonfinite/pairing failures are diagnostic failures, not negative fitting results.

Return the three `results.json` files plus the Slurm log. They will be under `outputs/frame_tiny_fit/JOB_ID/{image,camera,oracle}/`. Keep the NPZ files available; no new full geometry bundle is required initially. The saved `fitted_parameters.pt` is a diagnostic parameter archive, not a production resume checkpoint.

## Interpretation and stopping rules

- Oracle fits substantially better than camera: target-frame information or normalization order materially helps this setup. Because the oracle also changes axis-aligned bounding-box centering and point sampling, follow with a rotation-only control before assigning the effect solely to rotation. It is not a deployable fix by itself.
- Both surfaces fit, with losses degrading under swaps: the current interface can use surface information on this tiny set. Dataset-scale/generalization and multi-view canonicalization become the next questions.
- All arms fit equally and swaps do little: training-set memorization is possible, but useful surface use remains unproven.
- All arms fit poorly: inspect gradients, numerics, and loss trajectories; reduce to one fixed object (`--objects 1`) and add an optimized free-token positive control before blaming VecSetX or architecture.
- Both surface arms fail but a validated free-token control succeeds: investigate the frozen encoder/readout interface and then finetuning or scratch initialization.

A finite 1,000-update run cannot establish impossibility. A still-improving curve is inconclusive about the attainable fitting floor. One view per object does not test conflicting supervision across views. No production change or root-cause fix is claimed.

Local validation: Python compilation and shell syntax checks only for GPU code; SAM3D training cannot execute on this laptop. Bundle geometry analysis executed successfully on CPU with NumPy/SciPy. GPU execution remains necessary for the causal fitting test.
