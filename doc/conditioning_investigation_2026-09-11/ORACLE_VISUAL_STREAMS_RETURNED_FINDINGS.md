# Aligned surfaces are used; remaining view dependence is primarily visual

## Validation

The two returned attachments are archived under `oracle_visual_streams_returned_manual/`. Source hashes, single/multi-view reference hashes, final parameter digest and all seven input/feature provenance entries match the local reference artifacts. All 14 natural correct/swapped endpoint vectors reproduce exactly (maximum absolute difference zero). All 96 factorial rows are complete and the supplied analysis is exactly equal to local recomputation. The probe records no parameter changes. These validate the report and replay; the laptop did not independently execute the GPU operations.

## Fresh-bank results

All rows below hold the same anchor oracle surface and target fixed. Each changed stream comes from another normally preprocessed view of the same object. RGB includes cropped/full RGB tokens, silhouette includes cropped/full mask tokens, and pointmap includes cropped/full pointmap tokens.

| Changed streams | Other fitted views: native loss | Reserved views: native loss |
|---|---:|---:|
| None (anchor) | .017954 | .017954 |
| Pointmap only | .018995 | .019166 |
| Silhouette only | .035972 | .041855 |
| RGB only | .028556 | .062019 |
| RGB and silhouette | .020206 | .058158 |
| All three | .018986 | .055911 |

The three other fitted-view groups have coherent full-visual losses .01855, .02089 and .01752. Reserved groups have .03957, .08225 and .04591. Thus even with fixed aligned full-surface features, changing to unseen complete visual views produces the remaining gap. Those coherent endpoints avoid the mixed-view inconsistency explanation, though the oracle setup is still privileged and this is not an unseen-object experiment.

Pointmap-only replacement changes reserved loss by +.001212. Replacing pointmap with the matching reserved-view pointmap when RGB/silhouette already come from that view changes the mean by -.002247. Conditional pointmap effects throughout the reserved factorial are between -.002725 and +.001212. They are small compared with the +.037957 coherent anchor-to-reserved change. This does not imply that pointmap never matters or cannot help layout; it argues against pointmap replacement being the principal cause of this measured residual.

RGB and silhouette have a substantial interaction. On fitted views, crossing only one is harmful, but changing both together largely restores low loss. This cautions against interpreting every bad mixed condition as a coordinate conflict. On reserved views, changing RGB raises loss under each silhouette/pointmap setting, but that is still a conditional dependence result, not an additive fraction of causal blame.

## The surface is not being ignored

With coherent reserved visuals, correct-surface loss is .055911 and wrong-object-surface loss is .110605, a positive difference of .054694. At the anchor that difference is .065597; with coherent other fitted visuals it is .067829. Surface preference is reduced on reserved visuals but remains substantial.

Every one of the 384 recorded batch/noise paired comparisons has lower loss for the correct surface. These are not 384 independent observations: batches share four objects, noise draws are paired, and all-anchor comparisons are reused across view groups. The reports do not contain per-object loss decomposition. Do not translate this into a claim about every individual object or about superiority to an absent-surface/image-only model.

Correct-versus-wrong surface sensitivity proves a response to object-specific information at this scale. It does not prove dense geometric fidelity: four-object identity recognition could also make swapping harmful. The prior single-view fitting experiment separately established high fitting ability under the existing interface.

## Updated diagnosis

Two experimentally distinct obstacles remain:

1. **Rotation handling:** the completed camera/oracle factorization shows that camera rotation, rather than centering/radius normalization alone, causes most of the measured camera-frame penalty. That result is unchanged.
2. **Residual dependence on visual viewpoint after alignment:** this probe localizes the remaining oracle sensitivity mostly to RGB/silhouette rather than pointmap. The model uses aligned surface information but its shape prediction is not sufficiently invariant to visual view changes for these objects.

It would be inaccurate to claim a global axis bug, total rejection of VecSetX features, pointmap conflict as the main oracle problem, or an inherent architectural impossibility. It would also be premature to claim that learned pose alignment alone will solve full-surface conditioning: perfect alignment already leaves obstacle 2.

## Next bounded intervention, before full training

The next useful test should change training behavior, rather than repeat another frozen dependence probe. Test **visual modality dropout** while retaining the same full-surface VecSetX path, projector, shape cross-attention scope and target frame. This is a candidate to make surface information usable when a familiar visual view is absent; it is not yet a recommended production change.

Use the existing camera/oracle runs as the no-dropout cells, with exact initial/input/loss replay, and add two matched 1000-update runs: camera + visual dropout and oracle + visual dropout. On a prespecified balanced half of training presentations, replace the entire fused visual condition with the existing zero-condition convention while leaving touch tokens present. Keep evaluation visuals present. Do not drop only pointmap: the new result does not motivate that.

The dropout schedule must be balanced independently within all four fitted view groups and must not consume the RNG used for flow noise/time draws. Otherwise an alternating-step implementation would confound dropout with view group. This also creates a surface-only training task, which can be harder in camera axes; measure that effect rather than assume dropout is benign.

Branch on the interaction:

- Oracle improves on coherent reserved views, camera remains substantially worse: visual robustness is repairable under aligned coordinates, and rotation handling remains a separate limitation. Pursue a deployable alignment method against the improved oracle bound.
- Both improve and the camera/oracle gap narrows: reliance on visual views contributes to the measured coordinate burden; confirm the unchanged point conditioner on more objects before adding pose machinery.
- Oracle improves but camera worsens: visual inputs may help camera-to-target orientation recovery. Do not deploy the dropout policy unchanged; alignment and visual reliance must be treated jointly.
- Neither improves: reject this intervention at the tested budget. It is not proof of insufficient capacity or inherent pose ambiguity. Review fitted curves and correct/swap effects before proposing a broader adaptation test.

Require coherent-input improvement on fresh common draws and maintained correct-surface preference, rather than improvement only on dropped/crossed conditions. A second seed and a modest unseen-object confirmation remain necessary before a full run. Sparse touch is not being implemented: a dropout policy useful for full surfaces may need different treatment when sparse observations require visual completion.

No new training driver has been implemented by this result analysis. No full-dataset overnight run is justified by these findings alone.
