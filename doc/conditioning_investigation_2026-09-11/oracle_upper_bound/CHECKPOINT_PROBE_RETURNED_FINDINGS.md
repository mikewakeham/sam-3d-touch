# Oracle/constant checkpoint probe — returned 13 September 2026

## Verification and scope

Archive `full_checkpoint_probe_20260913_133239.zip` passes CRC validation. Both completed JSON reports and the supplied summary match the pasted attachments exactly. Re-running `analyze_checkpoint_probe.analyze` on the raw reports reproduces the supplied summary exactly. All nine reported runtime-source hashes match current local files. Oracle/constant dataset, pipeline, source, actual observation/target hashes, and time/noise-bank hashes agree. Checkpoints restore at epoch20 / step14,660, and adapted tensors are reported unchanged by the source-audited before/after checks. These are audited GPU reports, not an independent GPU rerun locally.

Both GPU runs used Torch2.5.1+cu121 on H10080GB. Current manifest counts are739 training objects /11,713 views and95 validation objects /1,499 views. The selected panel is16 identities from each disjoint split, two views per identity. Native loss uses four time/noise draws; each fixed time uses two draws. Oracle has1,280 batch-forward rows and constant320, four objects per batch. No Stage2, decoding, or training occurred.

Evidence is preserved in `returned_20260913_133239/`: raw reports, supplied/recomputed summary, `verification.json`, and `time_disaggregation.json`. The last file reports simple percentage transformations and a post-hoc decomposition of native paired loss differences by actual sampled time.

## Pivotal result: useful held-object conditioning exists at high noise

At t=.05, where x_t=.95*x0+.05*x1 for this sigma_min=0 flow:

| Split / visual state | Oracle loss | Constant loss | Oracle reduction | Objects oracle beats constant | Objects correct beats mean wrong |
|---|---:|---:|---:|---:|---:|
| Train / present | .092609 | .096914 | 4.44% | 14/16 | 16/16 |
| Train / zero | .103248 | .122948 | 16.02% | 14/16 | 16/16 |
| Validation / present | .176377 | .178571 | 1.23% | 11/16 | 16/16 |
| Validation / zero | .201542 | .216855 | 7.06% | 13/16 | 16/16 |

The same oracle checkpoint's mean wrong-object loss at this time is .200686 with visuals present and .238286 with visuals absent on validation objects, increases of13.78% and18.23% over correct conditioning. Each wrong-object mean averages three different identities in the same batch. All16 held identities have the correct-versus-mean-wrong advantage in both visual states. The frozen constant bank is correctly identical across batch examples; the oracle token path is actually object dependent.

This is evidence against **complete neglect of the sample-specific surface features** and against explaining all gains solely by a generic extra adapter/token bank. Utility extends beyond training identities at this tested noise level. It is not yet evidence of dense geometric readout, point-level fidelity, or successful free generation: category/shape-family cues can also help. The positive loss effect does not establish a working oracle upper bound.

## Aggregate native performance is still weak on held objects

| Split / visual state | Oracle native loss | Constant native loss | Oracle reduction |
|---|---:|---:|---:|
| Train / present | .063039 | .065425 | 3.65% |
| Train / zero | .069655 | .073868 | 5.70% |
| Validation / present | .105603 | .104806 | −0.76% |
| Validation / zero | .117502 | .118003 | 0.42% |

The validation-zero native mean advantage is small and occurs for only6/16 object means; do not present it as consistent generalization. Probe loss is not a replay of the full-dataset .089 validation number: it uses a selected32-view subset per split and a new explicit noise/time bank. Within-probe comparisons are paired.

At t=.5, oracle validation loss is1.90% worse than constant with visuals and2.30% worse without. At t=.95 it is6.54% /6.36% worse, with all16 objects worse in both states. These fixed-time panels are not averaged into the native metric. Native draws contain no t=.95 sample: the maximum is .87151, with just3/256 sampled times above .8 across both splits. Consequently, **do not claim the t=.95 result specifically explains the W&B plateau**.

Disaggregating the actual native draws supports a narrower cancellation account. On validation/zero, t<.1 contributes −.001502 to oracle-minus-constant loss, while [.1,.3), [.3,.6), and [.6,1) contribute +.000186, +.000541, and +.000274. Their sum is −.000501, exactly the small total oracle advantage. With visuals present, the corresponding sum is +.000797 (oracle worse). These are post-hoc descriptive bins of the sampled bank, not independent repeated experiments or causal attribution of a free-running trajectory.

Velocity MSE is also not a time-invariant measure of clean-shape error. Under sigma_min=0, x1_hat=x_t+(1−t)*v_hat, so its one-step clean-latent squared error equals (1−t)^2 times velocity squared error for a fixed pair. This algebra does not alter the training objective or prove rollout accuracy; it cautions against interpreting a high t=.95 velocity loss as the same amount of shape error as a high t=.05 loss.

## What has and has not narrowed

- The aligned surface path is connected, object dependent, and can improve high-noise prediction for unseen objects. A completely unusable representation/interface is not consistent with these results.
- Useful conditioning is strongest relative to the constant control when visuals are absent. **Visual removal is not an absolute rescue:** correct oracle validation loss at t=.05 worsens from .176377 to .201542, and native loss worsens from .105603 to .117502. Visual input still helps. Relative gains cannot by themselves establish harmful visual competition or justify disabling visuals.
- The training-object advantage exceeds the held-object advantage, and the effect varies with noise level. This supports partial utility with limited transfer across the tested conditions, not an established universal coordinate or architecture limitation.
- There is no camera comparison in this probe. It cannot estimate a causal benefit of oracle alignment over camera-frame inputs. Earlier frame experiments retain their own narrower scope.
- The working full-data oracle upper bound remains unproven because absolute sampled shape fidelity has not been measured for these checkpoints. Correct-versus-wrong dependence alone is insufficient; wrong inputs can simply be harmful.

## Next branch

The predeclared positive-condition-dependence branch now applies. The next useful test is **free-running Stage-1 sampling from noise on these same checkpoints and selected identities**, with correct surfaces, a fixed wrong-object control, and constant conditioning. Preserve visual-present/zero comparison, paired starting noise, and unchanged generation settings. Save predicted Stage-1 supports and decoded target supports, with raw frame fidelity and subsequent pose-adjusted comparison where needed. This requires only the Stage-1 decoder, not Stage2 or mesh/CD evaluation.

This distinguishes a real generative benefit obscured by pooled velocity loss from a conditioning signal that helps some teacher-forced predictions but fails to produce faithful shapes. If training-object reconstruction fails, focus on fitting/optimization/interface under oracle; if training works but validation fails, focus on transfer; if validation reconstructs faithfully with useful surface dependence, establish that measured upper bound before returning to non-oracle alignment. Do not change the time weighting, encoder, scope or visual dropout solely from these panels.

No new GPU job is included in this results report. The next sampling implementation should reuse the source/data/checkpoint hashes and the exact selections already verified here. The no-update loss probe is complete; these results should not trigger another native-loss-only repetition.

## Statistical interpretation limits

All comparisons are descriptive at the object level, aggregating repeated views/draws before reporting object counts. This remains one trained checkpoint per arm and a16-object diagnostic selection per split, not a population equivalence test. Eleven methodological fallacies checked: subgroup aggregation matters and is explicitly separated by split/visual/time; individual inference stays limited to measured objects; no outcome-based selection or collider adjustment was introduced; base-rate classification is inapplicable; fixed-step checkpoints avoid selecting extremes; all planned cells/runs are retained; the fixed-time panels were planned while native time bins are labeled post-hoc; prior failed branches remain documented; no unique coordinate mechanism, training-time competition or reverse-causal mechanism is asserted. Dense geometry interpretation and sampled fidelity remain open rather than inferred from the loss effects.
