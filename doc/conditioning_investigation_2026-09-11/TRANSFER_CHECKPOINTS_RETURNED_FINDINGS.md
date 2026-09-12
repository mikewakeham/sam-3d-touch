# Surface-token presence matters far more than average surface identity benefit

The three completed reports and supplied analysis are archived byte-for-byte in `transfer_checkpoints_returned_manual/`. The unchanged analyzer reproduces the analysis exactly. Source/fit-report hashes, every held-object historical replay at both checkpoints, final parameter hashes, frozen-parameter preservation and actual shape-attention context checks pass. See `validation.json` there. This validates the returned measurements and recorded GPU checks; it is not an independent GPU rerun on the laptop.

## Fresh-bank results

These losses use bank 800000 and 16 held identities, two views each, eight shared draws. Wrong mean averages the three other identities in each four-object batch. Lower loss is better.

| Step | Model | Correct surface | Mean of three wrong surfaces | Surface removed |
|---|---|---:|---:|---:|
| 256 | Image + pointmap | — | — | .132361 |
| 256 | Camera | .124598 | .125027 | .442418 |
| 256 | Oracle | .125767 | .126585 | .443141 |
| 1024 | Image + pointmap | — | — | .114380 |
| 1024 | Camera | .122080 | .122447 | .443887 |
| 1024 | Oracle | .129655 | .127642 | .444972 |

## What the early advantage actually supports

At 256, correct camera/oracle surfaces beat image on 15/16 and 14/16 objects. Replacing them with the mean of three wrong-surface evaluations preserves those same counts, with mean advantages of .007334 and .005776 over image. Each of the three individual distractor assignments also beats the image model in the aggregate, improving 14 or 15 objects depending on the assignment.

The average wrong-minus-correct gap is only .000429 camera and .000817 oracle. Therefore the early model-vs-model advantage is largely retained without the correct object geometry. It cannot be credited mainly to accurate surface identity on this evidence. A generic contribution of the additional learned token pathway is a concrete competing explanation, rather than simply a hypothetical confound.

Do not describe identity dependence as absent. At 256, correct beats the distractor mean for 11/16 camera and 12/16 oracle objects. Oracle per-object differences range from -.04076 to +.03093, much larger than its small mean; only 3/16 oracle and 5/16 camera objects beat all three individual distractors. This is weak/unreliable aggregate utility with heterogeneous individual responses, not a formal equivalence result or proof of zero geometry use. The three distractors per object are limited controls.

## What happened later

At 1024, camera and oracle lose to image on 14/16 and 15/16 objects. Camera's mean identity benefit stays small (.000366); oracle's mean is negative (-.002012), although its median is positive (.000949) and 11/16 objects prefer the correct surface to the distractor mean. One oracle object has a large negative benefit (-.130707). The negative mean must not be presented as every object preferring incorrect geometry.

Relative to step 256, camera's absolute held loss improves .002518 on this bank, while image improves .017981. Oracle's mean worsens .003887 even though 11/16 individual objects improve. Thus the early *relative* advantage disappears robustly, but the 256-to-1024 change is not uniform absolute deterioration across models or identities. The earlier 512-to-1024 monitor result is a separate interval/bank and remains valid.

## Removing tokens does not recover the image mapping

Removal raises loss to approximately .44 at both checkpoints, while any tested complete surface usually keeps the aggregate near .12-.13. Context hooks verify that removal leaves all 7528 visual tokens and removes only the 1024 surface tokens; it does not zero visual inputs or turn on CFG. The source has no alternate target or backbone path selected by this removal: touch tokens are simply absent from the concatenated attention context.

The strong dependence on surface-token presence is already established by step 256. It is not solely something that appears during late overfitting. The picture is joint adaptation to the added token pathway, not an image model that can recover by discarding a slightly harmful extra input.

With touch absent, the point encoder/projector is inactive; the forward-path parameter differences from the matched image model are in the trained shared shape cross-attention/norm2 modules. That locates the residual for this intervention. However, removal changes token count and leaves the surface models' training distribution. It does **not** establish that their coherent visual mapping is catastrophically forgotten, that attention competition is the specific mechanism, or that freezing shared attention will itself make geometry conditioning work. It would also be incorrect to conclude that the .32 removal penalty measures useful object geometry: wrong objects avoid most of it too.

## Coordinate implications

The same presence-versus-identity pattern occurs with camera and already aligned oracle surfaces. A missing inverse rotation is therefore insufficient to explain this behavior. The earlier rotation-factor result remains a valid coordinate burden on fitted identities, but an oracle alignment/pose estimator cannot by itself guarantee the missing transferable identity benefit.

This supports an interface/adaptation hypothesis while retaining VecSetX; it does not prove the representation is deficient. Sample-independent information in projected tokens, learned biases, feature statistics, and changes in attention normalization are candidates. None is isolated by deleting an entire token stream.

## One targeted training control, then an intervention decision

`CONSTANT_SURFACE_HANDOFF.md` specifies one 1024-update run using exactly one existing oracle training surface for every sample. Keep the original VecSetX, projector, token count, visuals, targets, seed, batches and shared-attention training scope. The bank is the first object/view in the prespecified training plan; no held-object feature contributes. The real oracle baseline is already complete and is reused after exact initial replay. This is a sample-independent conditioning control, not a production representation or a sparse-touch proposal.

Unlike removing tokens at inference, the constant condition is present from the start of training. Unlike substituting several wrong objects at inference, it contains no input-dependent surface variation. It therefore tests whether the added adaptation pathway can explain the apparent benefit without sample-specific surface information.

- Constant matches or exceeds real oracle on held identities: the real surface has not earned useful incremental geometry over this same-path control. Prioritize a targeted interface/training change, with this constant control retained to reject generic adaptation gains. A separately controlled surface contribution that protects the visual mapping is a candidate, not a validated fix.
- Real oracle beats constant on training identities but not held identities: surface content supports fitted-object specialization without measured transfer. Do not claim a missing global coordinate conversion or launch more of the same training.
- Real oracle beats constant broadly on held identities, but both lose to image: surface content helps within the added pathway, while its overall adaptation offsets that benefit. This gives stronger motivation for protecting the visual mapping in the next constructive intervention.
- Constant is much worse throughout: the generic-token account has not explained the behavior with this bank. Real surface variation may matter through common geometry/statistics even when specific distractor identities give similar losses. One fixed bank is not every possible geometry-free control; do not use its failure to declare fine-grained geometric decoding solved.

No equivalence claim should follow simply from a nonsignificant/small mean difference. Inspect paired magnitudes, medians, individual objects and curves. A small pilot cannot establish an inherent architecture limit or a generalization guarantee. No full-dataset training, broader finetuning, new point encoder, bridge or pose estimator is requested now.
