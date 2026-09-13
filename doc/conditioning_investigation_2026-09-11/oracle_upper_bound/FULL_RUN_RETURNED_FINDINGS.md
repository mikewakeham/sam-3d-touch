# Full-data oracle/dropout results — 13 September 2026

Verification status: **ANALYZED**. Material: local W&B exports for the five named runs; full histories and configs, with input hashes in `full_run_wandb_analysis.json`. No GPU checkpoint replay or reconstruction has been performed for these new runs. This report concerns Stage 1 only.

## What completed

All three new runs and both visual references finished 20 epochs / 14,660 optimizer steps. The new runs share global batch16, bf16, full shape cross-attention, frozen VecSetX, no position projection, CA lr1e-5, adapter lr1e-4, and per-sample visual dropout0.5. Their logged dropout fraction sequences match exactly; the update-weighted overall fraction is0.500410. Equality of those aggregate counts is not a full tensor-level training replay.

| Run | Final visual-present validation loss | Best validation loss | Best epoch |
|---|---:|---:|---:|
| Oracle + dropout (`cps50h2l`) | 0.08939843 | 0.08897717 | 15 |
| Constant oracle + dropout (`hwz5ifqt`) | 0.08887746 | 0.08852100 | 16 |
| Camera + dropout (`mpd9bbwh`) | 0.08876911 | 0.08873022 | 17 |
| Image + pointmap (`xun3al7m`) | 0.08909782 | 0.08857805 | 12 |
| Image only (`fl7b2znc`) | 0.08980808 | 0.08920783 | 11 |

At equal final step, oracle is0.586% worse than constant and0.709% worse than camera on this metric. Oracle beats constant at only1/20 validation checkpoints and camera at2/20. These checkpoints reuse validation data and seeds: they are **not20 independent experiments**. Neither a significance claim nor equivalence claim follows. Best checkpoints have unequal training exposure and are secondary descriptions, not the primary matched comparison.

Validation calls the model without a visual-drop mask. Thus the absence of an oracle validation advantage cannot be explained by averaging visual-present and visual-zero validation states. Training does mix those states, so its absolute value must not be compared directly with historical visual-only training losses.

## A small fitting difference remains

Update-weighted training losses over epochs19–20:

| Run | Mixed training loss |
|---|---:|
| Oracle | 0.09580256 |
| Constant | 0.09870240 |
| Camera | 0.09850713 |

Oracle is2.94% lower than constant and2.75% lower than camera. These are estimates formed by weighting each logging interval by its optimizer-update count. Individual log intervals use sample weighting internally; exact dataset sample weights are not exported, so do not label these exact epoch means. Early epoch1 constant also fits faster than oracle (.12475 vs .12770); rapid initial decline was not specific evidence of useful object geometry.

All1,485 logged gradient observations in each new run are finite and nonzero for shape cross-attention, the surface output projection, and touch embedding. Frozen VecSetX and disabled position projection have zero gradients as intended. This rules out an entirely disconnected adapter at logged steps; it does not establish successful geometric interpretation or correct cluster input coordinates.

## What this narrows

The full-data run has **not established the useful oracle upper bound**. Merely providing alignment plus visual dropout with this adaptation scope and budget has not produced a visual-present native validation advantage over the matched constant control. We should remain on the aligned branch rather than moving to inference-time rotation recovery.

The runs are not literally identical: a small training advantage is compatible with some aligned input utility. Aggregate logs cannot tell whether that advantage is restricted to fitted identities, visual-zero examples, or particular noise levels. They also cannot show whether sampled intrinsic shape is correct. Do not turn a .09 loss plateau into a claim that geometry is ignored, coordinates are irrelevant, or the architecture cannot solve the task. The earlier F9 oracle and F16 shared-frame fitted successes remain valid within their small-set scope.

## Next diagnostic and branch decisions

Use existing **last.pt** checkpoints at step14,660, first oracle and constant, with a short no-update Stage-1 probe. Select16 training and16 validation identities deterministically before examining outcomes; two views each, four distinct objects per batch. Keep the same examples, actual latent-noise tensors and times across compared cells. Restore the constant checkpoint's saved bank through `get_touch_tokens`; do not encode the supplied objects for that control.

1. Separate visual-present and visual-zero states, retaining the trained token conventions. Record correct-surface oracle versus its own wrong-object surfaces, plus the constant checkpoint. Save per-object losses and losses at fixed noise levels in addition to ordinary native draws. This isolates useful condition dependence from pooled loss. Zero-visual constant is deliberately information-poor; its failure alone is not evidence of strong oracle reconstruction.
2. If oracle has useful correct-versus-wrong dependence, assess actual Stage-1 sampled outputs against the decoded target endpoint, on the same identities. Include pose-adjusted comparison when needed; no Stage2 or CD campaign. Dependence and absolute fidelity are separate requirements.
3. Oracle useful on training objects but poor on validation objects: the immediate issue is transfer of the aligned mapping; alignment alone is insufficient at this scale/recipe. Broader adaptation or exposure becomes a targeted branch, not an automatic rerun.
4. Oracle useful with visuals absent but weak with visuals present: focus on multimodal interaction. Do not claim dropout has universally solved it.
5. Oracle weak even on training objects with visuals absent: first audit restored tokens/coordinates and compare the previously successful tiny recipe with full-run optimization/exposure. Broader adaptation is conditional here. Do not investigate non-oracle pose recovery first.
6. Strong oracle reconstruction and dependence in both splits: establish that measured upper bound, then compare camera using the same probe before attributing a frame cost.

No training extension, learning-rate change, encoder replacement or production source edit is justified solely by these logs. The next required causal evidence needs the saved GPU checkpoints. The numerical gap is small enough that proper paired input/noise control matters.

## Evidence limits and audit

The historical image+pointmap configuration uses `samples.jsonl`, whereas new runs use `samples_full_surface.jsonl`. The laptop has the former manifest and split file but not the latter; identical visual examples/targets cannot yet be certified. All three new runs share the full-surface configuration, making oracle/constant the strongest currently available comparison. W&B records source commit IDs, not proof of an unmodified working tree during cluster execution.

Statistical fallacy scan11/11: subgroup reversal and ecological inference remain unresolved without per-object/visual-state outputs; selection/collider bias is limited by reporting all three authorized arms and both planned references, with no outcome filtering; base-rate diagnostic classification is inapplicable; regression-to-mean/best-selection addressed by fixed final-step comparison; all planned runs completed (no run attrition hidden); look-elsewhere and forking-path risks addressed by descriptive reporting of both best and final values and preserving old failed branches; no unique causal mechanism or reverse-causality claim is made. No p-values, independent-repeat confidence intervals, population equivalence or universal failure claims are warranted from these aggregate single-run exports.

Reproduce from repository root:

```bash
python3 doc/conditioning_investigation_2026-09-11/oracle_upper_bound/analyze_wandb_full_runs.py \
  --exports ../wandb-results \
  --output doc/conditioning_investigation_2026-09-11/oracle_upper_bound/full_run_wandb_analysis.json
```
