# Sixteen-object training: fitting improves, surface utility does not transfer

All four 1024-update runs returned. Original attachment bytes are archived in `object_transfer_returned_manual/`. The supplied analysis reproduces exactly from the four reports. Source, plan, reference and report hashes match; common initialization, actual inputs, held-identity separation, every executed batch/dropout decision, per-object/native-scalar agreement and dropout context checks pass. See `validation.json` there. This is Stage-1 native flow loss only.

## Independent final bank

Each cell averages within identity first: 16 fitted identities and 16 different held identities. Lower is better. The image control includes pointmap.

| Model | Fitted views | Held views of fitted objects | Held objects | Held objects beating image |
|---|---:|---:|---:|---:|
| Image + pointmap | .061607 | .062941 | **.119303** | — |
| Camera surface | **.055440** | .058075 | .126404 | 2/16 |
| Oracle surface | .056473 | **.057185** | .134199 | 0/16 |
| Oracle + visual dropout | .059668 | .060871 | .133765 | 1/16 |

Oracle improves fitted and held-view loss over image on all 16 training identities. It nevertheless loses on every held identity. Thus the conditioning is not globally inactive, and neither perfect input alignment nor the tested dropout policy supplies a transferable fix at this scale.

On held objects, mean wrong-surface minus correct-surface loss is +.000153 camera, -.006745 oracle and -.003624 oracle dropout. Only 6/16, 8/16 and 9/16 respectively prefer the correct surface. This is a single deterministic wrong-object pairing per object; it is stronger evidence of failure than simply comparing separately trained models, but does not characterize every possible distractor or prove the model has no geometric information.

Do not summarize the frame comparison as uniformly camera-better. Oracle's held-object mean is worse than camera by .007796, but its median paired difference is -.001506 and it wins on 10/16 objects. Large failures on a minority drive the mean ranking. Alignment is demonstrably insufficient here; there is no stable universal camera/oracle ordering established by these small runs.

## Learning trajectory changes the next decision

The fixed monitor bank is separate from the final bank above. Do not mix their absolute values.

| Step | Image held-object loss | Camera | Oracle | Oracle dropout |
|---|---:|---:|---:|---:|
| 0 | .456311 | .456655 | .456625 | .456625 |
| 256 | .134816 | .126671 | .127910 | .129088 |
| 512 | .120392 | .121665 | .125321 | .125877 |
| 1024 | .116415 | .124371 | .128607 | .129146 |

At step 256, camera/oracle/dropout beat image on 12/16, 13/16 and 14/16 held identities. At step 512 the counts are 7/16, 6/16 and 4/16. At step 1024 they are 2/16, 1/16 and 2/16 on this monitor bank.

From 512 to 1024, fitted losses fall for every arm. Held-object loss improves for image by .003976, but worsens for camera by .002706 (11/16 objects), oracle by .003286 (12/16) and dropout by .003270 (10/16). This is an observed training/generalization divergence, not just an insufficient endpoint comparison. However, step-256 surface models still have worse absolute held losses than the later image model; the early same-step advantage is not yet an overall best-model improvement.

The early monitor contained no wrong-surface intervention. Its advantage could reflect useful geometry, different optimization speed, or another effect of the added tokens/shared attention adaptation. Calling it early transferable geometry would go beyond the evidence. Fresh common draws and condition interventions at the saved checkpoints can separate these explanations without more training.

## What this says about coordinates

Earlier four-object factorial training isolated camera/object rotation as the dominant measured same-identity frame penalty. That result remains valid. The normal camera input does not explicitly supply target-frame rotation; targets are spatial object-frame latents. Native VecSetX normalization is not rotation-equivariant, and the existing oracle path supplies the known inverse before encoding.

Here that oracle intervention is already applied, yet held-identity conditioning fails. A missing inverse transform alone therefore cannot explain all the failure. This does not rule out difficult translation from VecSetX's coordinate-dependent features into SAM's spatial target, image/target conventions interacting with pretraining, or a more effective learned coordinate interface. None is proven by these losses. There is still no demonstrated global sign/axis patch or proof of inherent impossibility.

The strongest current diagnosis is **learned surface utility that is specific to the fitted objects, accompanied by deterioration on held identities as adaptation continues**. Internal identity lookup remains a hypothesis, not a measured mechanism. With 100,810,752 shared shape CA/norm2 parameters and 3,064,896 point-adapter parameters trained on 16 identities, specialization is plausible; parameter count alone is not a causal diagnosis. Both fitted and held-view losses are still falling, so convergence and asymptotic fitting capacity are unresolved too. More optimization cannot be assumed to fix the observed transfer decline.

## Next action and branches

Do not extend these fits or launch full training now. `TRANSFER_CHECKPOINTS_HANDOFF.md` prepares a frozen step-256/1024 comparison for image, camera and oracle on the same held identities. It uses correct surfaces, all three other identities in each batch, and complete removal of surface tokens, with fresh common draws and exact historical replay.

- If the early correct surface beats distractors and removal broadly, and this disappears late, early geometric utility is supported. A short learning-policy intervention that preserves this utility is justified; it must also beat the best image checkpoint, not merely the image model at the same update count.
- If early correct and wrong surfaces behave similarly, the early model-vs-model gain does not establish geometric conditioning. Avoid declaring early stopping a solution.
- If removing surface tokens largely restores the image baseline late, the harmful incoming surface contribution is localized. A separately controlled surface branch is a candidate intervention while retaining VecSetX and the camera/oracle contrast.
- If removal remains worse than image, the forward-path residual is in the differently trained shared shape CA/norm2 parameters: the point encoder/projector is then inactive and the remaining generator parameters are fixed. This motivates testing protection of the visual mapping, not simply more point alignment. Removal is outside the point models' training distribution; this diagnostic does not establish that freezing shared CA will by itself train a useful surface branch.
- If frame-specific utility differs despite the controls, retain that difference explicitly in the next targeted intervention. A pose head is not automatically warranted by an oracle that still fails transfer.

The next probe does not modify representations, targets, optimizer settings or production inference. Sparse structured touch remains the future endpoint; no voxel bridge is introduced. The experiment is a development diagnostic on already inspected objects and one seed, not a population benchmark.
