# Visual dropout nearly closes the oracle native-loss view gap on four fitted objects

**Claim boundary:** the reference is fitted-view native flow loss under the same assessment protocol, not a ground-truth reconstruction floor. The measured gap nearly closes without worsening fitted loss. No prespecified equivalence margin or decoded-target reconstruction endpoint established complete resolution for these dropout checkpoints. Subsequent unseen-object results reject a general conditioning fix. See F5–F6 in `PIVOTAL_FINDINGS.md` before promoting or extending this finding.

## Validation

Returned camera/oracle reports and analysis are archived under `visual_dropout_returned_manual/`. The unchanged analyzer validates the complete protocol and locally recomputed output exactly matches the returned analysis, including report byte hashes. All current source and historical reference hashes match. Both reports contain exact initial and final historical replay, restored initialization, the same balanced schedule, 1000 executed updates, all assessments and fresh-bank evaluations. Shape-attention context checks passed: 7528 visual tokens plus 1024 nonzero surface tokens; dropping visual input preserved the surface stream and token count. The validation record is `validation.json`.

This is validation of the returned GPU reports and their recorded replay, not an independent GPU rerun on the laptop.

## Primary result: coherent inputs, visual conditioning present

These results use the fresh common time/noise bank, with original fitted checkpoints reevaluated on that same bank. Native shape flow loss, lower is better:

| Frame / policy | Fitted-view loss | Reserved-view loss | Reserved wrong-surface loss |
|---|---:|---:|---:|
| Camera, original | .029912 | .077178 | .081215 |
| Camera, visual dropout | .036930 | .053811 | .064309 |
| Oracle, original | .023847 | .057849 | .116087 |
| Oracle, visual dropout | .022047 | .022732 | .225346 |

Camera reserved loss improves 30.28%, although its fitted loss worsens 23.46%. Oracle reserved loss improves 60.70%, with fitted loss also improving 7.55%. Oracle fitted-to-reserved gap falls from .034002 to .000686; reserved loss is only 3.11% above fitted loss. This is a successful training intervention for the measured oracle view-transfer problem at this scale.

All three reserved-view groups improve for both arms. Camera reductions are 16.94%, 40.81% and 22.70%; oracle reductions are 43.35%, 73.78% and 50.74%. Each group averages four objects; the report does not resolve individual-object losses. No claim of independent replication across groups/draws is made.

The camera-minus-oracle reserved gap increases from .019329 to .031078 (interaction +.011750). Thus the outcome is **both improve, oracle improves more, frame gap widens**, rather than the predeclared possibility that both improve and the gap shrinks. Dropout alone does not resolve the frame burden. The prior rotation/normalization experiment implicated rotation under the original policy; do not assume it quantitatively decomposes the new dropout-policy gap without testing that interaction.

## Surface use and visual-zero control

With full visuals, wrong-minus-correct surface loss increases from .004037 to .010498 for camera reserved views and from .058238 to .202614 for oracle reserved views. This supports stronger dependence on the supplied surface, particularly after alignment. Large wrong-condition errors alone do not demonstrate detailed geometric use or superiority to an absent-surface model.

Dropping all visual tokens at evaluation gives .055474 camera reserved loss and .022256 oracle reserved loss. The aligned model performs similarly with full visuals (.022732) and zero visuals (.022256). The primary improvement is therefore not restricted to a dropped-input evaluation. Oracle visual-zero features/targets are almost identical across views, so their matched fit/reserved losses are an expected control, not independent evidence of view generalization.

Because only four objects were trained, recognizing object identity from surface features and retrieving a memorized shape remains a competing account. That still satisfies a useful fitting/robustness diagnostic, but it does not establish a transferable geometric conditioner. Do not call this a solution to the full dataset problem yet.

## Learning has not reached a demonstrated ceiling

The historical monitoring bank corroborates the endpoint direction. Dropout camera fit/reserved loss at step 1000 is .040096/.057245; oracle is .025240/.025984. Both training curves still decrease late in the run. Even separating dropped and visual-present updates, the last-100 mean is below the preceding-100 mean. These stochastic windows have different view/drop schedules and do not quantify convergence, but an asymptotic inability claim is unsupported.

Raw training losses mix two condition distributions and should not be compared directly with the original always-visual training loss. Use the matched all-input assessments above.

## What this establishes and changes

The existing frozen VecSetX plus trainable projector/full shape cross-attention can learn to rely on aligned full-surface observations and largely remove the measured same-object visual-view gap. No new encoder, spatial bridge, larger backbone adaptation or target-frame change was necessary for that particular repair.

Visual-present-only training was therefore a remediable contributor to the original oracle residual in this task. The experiment does not isolate an internal attention mechanism, prove that 50% is optimal, or establish that the camera-frame residual is impossible to learn. Camera coordinates still incur a substantial penalty after this intervention.

Retain visual dropout as a **validated small-task candidate**, not a silently enabled production default. Keep pretrained VecSetX and preserve the future sparse-touch path. Full surfaces may support frequent visual dropout more readily than sparse patches; no sparse-policy claim follows.

## Next decision

The improved oracle is now the appropriate positive control for explicit rotation handling. Inference is still assumed to provide camera-frame observations rather than target-frame pose metadata. Supplying ground-truth rotation again would only reproduce an existing positive control; it would not establish a deployable fix.

Before full training, confirmation must address both the four-object memorization alternative and the frame gap. Use a modest object-disjoint experiment, retaining camera/oracle controls and the dropout policy, with a matched no-dropout comparison where claiming a transferable dropout benefit. Include fresh common draws and another training seed. A low oracle loss on fitted identities alone is insufficient evidence to scale a pose-estimation solution or declare representation transfer solved.

If aligned conditioning transfers but camera conditioning does not, prioritize a targeted, supervised camera-to-target alignment test while keeping VecSetX. If aligned conditioning fails on unseen objects, resolve that actual transfer failure before assuming pose alone is sufficient. Any conclusion of unidentifiable canonical orientation needs evidence under the actual input/target convention, not simply a failed finite training run.

No full-dataset overnight run, additional automatic continuation, or new GPU job is requested by this result analysis. Existing checkpoints should be preserved for subsequent Stage-1 generation checks if the candidate is confirmed.
