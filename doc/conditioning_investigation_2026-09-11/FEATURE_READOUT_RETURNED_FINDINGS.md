# Three-arm feature readout: native processing helps fitting, transfer unresolved

All three reports are complete at 2,000 updates. The paired checker passes: identical ordered 24/8 object split, targets, raw input features, source representation report, shared initial head weights, driver, seed, learning rate, and start step. The current driver and archived representation report match the returned hashes. Reports and aggregate analysis are in `feature_readout_returned_manual/`. This validates reported results and provenance; GPU tensors and checkpoints have not been independently recomputed locally.

## Stage-1 latent evidence

| Input | Fitted-object MSE | Reserved-object MSE | Wrong-object fitted MSE |
|---|---:|---:|---:|
| Training-mean target control | .151084 | .211688 | — |
| Raw | .116254 | .245840 | .193649 |
| Raw + slot identity | .121020 | .234031 | .185902 |
| Native processed | .090193 | .244182 | .220348 |

Native processing reduces final fitted MSE by 22.42% relative to raw, winning on 23/24 fitted objects. All 24 fitted objects worsen under wrong-object conditioning. This supports improved access to object-specific target information within this head. It does not isolate slot identity: processed features include the pretrained native transformer and a larger input projection (1,856,776 trainable parameters versus 1,600,840).

There is no convincing reserved-object rescue. Processed beats raw on only 4/8 reserved objects; its mean error remains above the training-mean control. Its reserved error rises from .211548 at update 500 to .244182 at update 2000 while fitted error falls. Explicit sinusoidal slot identity gives slightly worse fitted error than raw and lower reserved error on all eight objects, but remains worse than the mean control. This does not rule out other slot-aware designs.

Fitted errors drop another 14.4% raw, 13.0% slot-aware, and 21.9% processed between updates 1500 and 2000. These endpoints are not plateaus. Failed convergence and information loss cannot be inferred. Direct latent MSE is not the previous flow-training loss, so their numerical magnitudes should not be compared.

## What this changes

This experiment has already removed camera-frame mismatch and the image-conditioned SAM flow network. The raw-to-spatial-target readout still struggles to fit quickly, and exposing native processing improves fitting without fixing transfer. A coordinate correction alone is therefore insufficient to explain this diagnostic's remaining error. This is evidence about this readout, not proof of the cause of the original training results.

Together with native VecSetX reconstruction, the evidence supports keeping geometry availability separate from efficient translation into SAM's representation. Neither adding slot IDs nor exposing `learn` alone is a demonstrated production fix. There is no basis here for declaring broader SAM finetuning necessary or sufficient.

## Next bounded decision

Continue the existing three checkpoints from update 2000 to 8000 with the same optimizer, LR, split, features and targets. This follows the predeclared still-improving branch. It tests whether the disappointing fitted endpoint was premature. It is not intended to repair the already worsening held-out trend. No further extension is automatic.

Interactive-node command from the repository root:

```bash
bash doc/conditioning_investigation_2026-09-11/run_feature_readout_continue.sh
```

The launcher reads `outputs/conditioning_investigation/feature_readout/manual` by default and writes a separate `feature_readout_continued/manual` directory. Arguments can override the source fit root, output root, and representation directory, in that order. Return the three new `results.json` files; preserve checkpoints on the cluster.

Decision branches after this continuation:

- If fitted error falls substantially while reserved error stays poor, there is a learned training-object mapping but no reusable translation. Stop extending this regression probe. A next constructive comparison should preserve explicit observed point locations and test a spatial conditioning interface; it must include a simple geometry reconstruction control so a failed SAM interface can be separated from an unusable point representation.
- If processed fitting becomes substantially stronger than both raw arms, carry processed features forward as the positive control for that comparison. This does not by itself recommend replacing the production conditioner.
- If all remain far from targets and flatten, stop extending. First validate the readout's optimization/capacity with a known learnable spatial mapping before treating its failure as evidence against VecSetX or deciding on broader SAM adaptation.
- If meaningful reserved improvement emerges, reproduce with another seed and a fresh object subset before integrating the direct translation into Stage 1. These eight objects are an exploratory diagnostic set.

Sparse structured touches remain the endpoint. This full-surface regression cannot establish sparse completion feasibility. Any constructive conditioner must retain observed xyz and validity, avoid treating unobserved regions as empty, and be tested with clustered contact patches rather than only random point thinning. Oracle alignment remains a diagnostic privilege to remove before claiming a usable solution.
