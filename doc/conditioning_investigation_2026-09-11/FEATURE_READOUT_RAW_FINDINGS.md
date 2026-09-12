# Raw readout returned; paired comparison incomplete

The supplied JSON contains only `arm=raw`, 2000 updates, seed 29, output `feature_readout/manual/raw`. The driver and source representation-report hashes match exactly. Sample ordering, 24/8 object split, all five assessment milestones, unique coverage and finite nonnegative errors pass checks. Raw bytes and a validation summary are archived in `feature_readout_returned_manual/raw/`. No slot-aware or processed report has been received; do not infer those jobs ran or finished.

| Update | Fitted-object latent MSE | Reserved-object latent MSE |
|---|---:|---:|
| 500 | .15427 | .21275 |
| 1000 | .15200 | .22724 |
| 1500 | .13581 | .23062 |
| 2000 | .11625 | .24584 |
| Training-mean target control | .15108 | .21169 |

This 1.60M-parameter raw-feature readout learns some object-specific mapping: fitted MSE falls below the mean-target control, and wrong-object inputs increase final fitted error to .19365 (all 24 objects worsen). Reserved error rises across the recorded post-initialization checkpoints and ends worse than the mean-target control. Only three of eight reserved objects worsen under swaps; aggregate wrong-object error is .24992 versus .24584 for correct features. There is no demonstrated useful transfer from this arm.

Final decoded target-occupancy IoU is 23.83% on fitted objects and 2.94% on reserved objects. Eight fitted predictions and one reserved prediction decode to empty occupancy. A low absolute latent error alone is consequently insufficient to claim a useful reconstructed shape. These are direct latent regression results, not native flow losses; their magnitudes cannot be compared directly with previous flow-loss numbers.

Fitted loss is still falling: the 1500-to-2000 assessment change is roughly −14.4%. Do not call the remaining error a capacity floor or proof that raw codes lose geometry. The pattern is consistent with increasing training-object specificity without demonstrated geometric generalization at this dataset size and optimization budget. This does not establish that additional training would fix transfer.

The next action is the originally paired `raw_slot` and `processed` comparison, not a new architecture or another raw run. If those are already complete, return their reports and logs. If only a manual invocation was run, the launcher defaults to raw when no Slurm array task ID is set; `manual/raw` is consistent with that, but does not prove how it was invoked. Submit remaining arms with:

```bash
sbatch --array=1-2 doc/conditioning_investigation_2026-09-11/run_feature_readout.sh
```

Keep the source representation directory and all settings matched. That command writes the remaining arms under a new array job ID; the eventual analysis should assemble immutable report copies under one analysis root and verify their hashes/splits rather than change any original output.

Interpretation remains conditional: a slot-aware or processed win localizes representation access within this readout; similar failure leaves the small readout, optimization and dataset scale unresolved. Native VecSetX fidelity still supports geometry being accessible to its own decoder. Nothing here establishes sparse-touch feasibility or a full-surface conditioning fix. No new GPU job was launched from the laptop.
