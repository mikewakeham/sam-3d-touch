# Separate attention fits strongly but fails the held-object criterion

**Follow-up status:** results below remain valid. The subsequent context/source review recovered the original oracle job paths from Git. `CONTEXT_AND_COORDINATE_NEXT_STEP.md` supersedes the next-action section below; obtain the CPU checkpoint inventory before any GPU comparison.

All four logging-fix runs returned and passed validation. Original attachment bytes are archived in `separate_surface_returned_manual_logging_fix/`. The supplied analysis reproduces exactly. Source and reference hashes match; CUDA module/gradient tests, actual context allocation, exact initial image replay, zero-residual image equality, full training coverage and native-scalar/per-object agreement pass. Every separate arm preserves all frozen parameters and exactly recovers the original image outputs with surface tokens removed at the end. Constant swaps are exactly invariant. These checks validate the returned reports, not an independent local GPU rerun.

## Outcome: do not promote or extend this candidate

| Arm | Fitted views | Held views of fitted identities | Held identities | Held identities beating image |
|---|---:|---:|---:|---:|
| Frozen image anchor | .061607 | .062941 | **.119303** | — |
| Separate oracle | .028968 | .036458 | .138540 | 0/16 |
| Separate constant | **.021440** | **.028826** | .149936 | 1/16 |
| Separate camera | .028290 | .035993 | .140719 | 1/16 |
| Joint oracle, same image start | .045687 | .051873 | .122571 | 4/16 |

Separate oracle halves fitted loss relative to the image anchor, improves fitted and held-view losses on every training identity, and beats joint oracle fitting on 15/16 identities. It nevertheless loses to image AND joint oracle on every held identity. The intervention failed its prespecified geometric-transfer criterion. Its preserved image fallback is a successful implementation invariant, not a conditioning solution.

Separate attention trains 100,761,600 attention parameters plus the 3,064,896 point-adapter parameters; joint trains 100,810,752 CA/norm2 parameters plus the same adapter. This is not evidence that a larger number of unfrozen weights is required simply to fit. The trained module locations, freezing and initialization differ; the pilot does not isolate a single attention mechanism.

## Geometry-specific behavior differs between fitted and held identities

Separate oracle's fitted wrong-minus-correct loss is +.025928, with all 16 objects preferring their correct surface. On held views of these identities it remains +.024169 with 16/16 preferring correct. Separate camera also has substantial fitted surface dependence (+.017282, 16/16).

On held identities, those preferences fail: oracle wrong-minus-correct is -.004342 with only 7/16 preferring correct; camera is -.003751 with 6/16 preferring correct. Thus the implementation can learn object-specific dependence through the new branch, but that dependence does not become reliably useful for these different objects. Internal identity lookup is consistent with this pattern, not directly measured.

Real oracle beats the separate constant on 12/16 held objects, lowering mean loss by .011396; camera beats it on 14/16, lowering mean by .009217. This is an improvement over the constant branch, but both real branches still lose to image and fail correct-versus-wrong surface utility. Real feature variation could constrain overfitting compared with the constant branch; that is a hypothesis, not proof of successful geometric decoding. Do not summarize this as either no effect of geometry whatsoever or a transferable solution.

## Coordinates and the image mapping

The oracle/camera held difference is small compared with their common failure relative to image: .138540 oracle versus .140719 camera. Oracle is better in aggregate and on 9/16 objects; it offers no qualitative rescue. Camera slightly outperforms oracle on fitted views. The original four-object rotation-factor result remains valid, but removing camera-to-object rotation does not solve this new interface's transfer problem.

The separate arms recover image loss EXACTLY when removed. Consequently, this failure does not require destructive parameter changes to the original visual generator or competition between visual and surface tokens in one softmax. An additional learned residual can itself be harmful on new identities even when those properties are controlled. This rules out those two mechanisms as a sufficient explanation/remedy for the observed failure in this pilot, not as contributors in all previous joint models.

Warm-started joint oracle has removed-surface held loss .120058, close to image .119303, versus .122571 with correct surface. The catastrophic removal loss in earlier freshly initialized point models is therefore not universal. This comparison includes prior image training and does not isolate initialization from total training exposure.

## Curves do not justify an automatic extension

At step 256, separate oracle's monitor held loss is .115581 versus the frozen image monitor .116415 (about 0.7% lower); camera's is .115724. These are small development-bank gains without early swap controls, not an established solution. By 512 both are worse than image, and by 1024 their held losses are .134665/.137445. Fitting drops strongly over the same interval. Constant shows an even stronger fitting/held divergence. Joint changes less and ends with the smallest held penalty.

No arm has demonstrated an asymptotic fitting ceiling. Nevertheless, extending these same runs in pursuit of transfer is not supported by their trajectories. Do not start full training or additional architectural variants on this 16-object split automatically.

## Next decision: reconnect the diagnosis to existing dataset-wide runs

Recent controlled training uses only 16 identities. The local split file lists 819 nominal training identities (the usable full-surface manifest may contain fewer), and the original dataset-wide runs already exist. It would be an overreach to claim that the pilot's specialization fully diagnoses those original results, or to declare that more unfrozen weights, a new encoder or more data is the proven fix.

Before requesting another broader-data training screen, probe the existing dataset-wide image/full-surface checkpoints. The checked-in job scripts and W&B exports identify `outputs/stage1_image_full_cross_attention/best.pt` (run `xun3al7m`, best step 8796) and `outputs/stage1_full_surface_full_cross_attention/best.pt` (run `act988rs`, best step 6597). These are original historical paths; new probe outputs stay under `outputs/conditioning_investigation/`.

`FULL_DATASET_CONDITIONING_HANDOFF.md` prepares a no-training test with correct, three wrong and removed surfaces, measured separately on current dataset-train and validation identities. This tests whether weak surface-identity utility is also present in the original broader-data setup. The cross-model comparison is descriptive because their best checkpoints have different training steps. The within-surface-model interventions are paired.

- If broader-data validation shows strong correct-surface utility, the tiny-fit failure is not a sufficient diagnosis of the original setup. Investigate how that existing utility relates to training progress and the coordinate-conditioned target task; do not keep redesigning based only on 16 identities.
- If utility appears on training identities but not validation identities, specialization extends to the actual broader-data model. This motivates a controlled learning/regularization or data-diversity intervention with constant controls, rather than another view-only test.
- If correct/wrong surfaces behave similarly on both splits while presence matters, the generic adaptation-pathway finding extends beyond the tiny fits. That would justify a more substantive geometric-interface/adaptation experiment at a representative scale.
- If checkpoint metadata or current diagnostic input checks fail, resolve that provenance/input discrepancy before interpreting model behavior. No retraining is requested to replace missing historical checkpoints.

The original full-dataset oracle run reported by the user is accepted, but its checkpoint location and matching export are not known. This handoff targets two independently identified existing runs; it does not repeat oracle training or claim the oracle run never occurred.
