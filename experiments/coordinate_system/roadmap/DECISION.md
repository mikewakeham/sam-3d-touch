# What would justify stopping the coordinate investigation?

**Current answer: stop repeating the verified transform checks. Do not claim all coordinate-related learning effects are excluded.** The audit is complete; the empirical closure evidence is not. No new run is requested here.

## The argument, without treating every open question as a prerequisite

```text
On a particular object, is reconstruction actually poor?
  No / only a similar pooled loss -> no demonstrated reconstruction failure to diagnose.
  Yes
   |
   +-- Do its actual points and target fail the independent geometric contract?
   |     Yes -> a concrete coordinate/data defect; repair and reassess that object.
   |     No  -> a wrong explicit transform cannot explain that object's residual.
   |
   +-- Does poor reconstruction persist in training without pointmap information?
   |     Yes -> pointmap disagreement is not necessary for that residual.
   |     No  -> removing pointmap helps; coordinate conflict is only one explanation.
   |
   +-- Does it persist when training has no image, mask or pointmap information?
         Yes -> sample-dependent visual disagreement is not necessary either.
         No  -> visual training policy matters; it is not yet a scale/rotation diagnosis.

What remains after those exclusions?
  A. Information/placement lost by finite-sample normalization.
  B. Difficulty interpreting the aligned features, including conversion between units.
  C. Pretrained preference for another consistent target convention.

These are not three unidentified sign/axis bugs. B and C interact with trainable scope.
They cannot be separated by another unsuccessful camera-inverse check.
```

The branches above compare **observed failures**, not hypothetical ones. Multiple causes can coexist. Failure after removing a stream excludes necessity, not the stream's contribution to the original run. A successful removal also removes useful information and changes learning; it does not isolate coordinates automatically.

## Which evidence is needed for which stopping claim?

| Claim you want to make | Required evidence | Current position |
|---|---|---|
| “The camera inverse/axis conversion is not wrong on these failing examples.” | Independently verified geometry for those exact examples, plus the actual encoder-boundary coordinates. | Available for the existing full-frame bank. Further whole-dataset checking is **not needed for this scoped statement**. |
| “The original full dataset contains no coordinate/data defects.” | Coverage of all relevant records and generation branches, not only the selected bank. | Not established. This stronger claim is **not required** to study an audited failing subset. |
| “Pointmap disagreement cannot be the whole explanation.” | Verified reconstruction failure in the oracle/no-PM training variant. | Latest poor performance is user-reported; its checkpoint assessment has not been imported. |
| “No visual-frame disagreement is required for the remaining failure.” | Verified failure when all visual information was absent throughout training. | No returned surface-only training result here. Inference-only suppression is insufficient. |
| “Normalization/convention cannot affect learning.” | A universal claim over encoders, conventions and optimization—not something finite negative trials establish. | Do not use this as a finish line. Test a specified contribution or demonstrate a working recipe instead. |
| “This oracle convention supports a useful upper bound.” | Accurate target-referenced reconstruction and correct-surface utility at the stated scale; fresh objects if claiming transfer. | Tiny fitted-identity success exists; full-data accurate upper bound does not. |

## The clean decision after the missing exclusions

There are two legitimate routes. They answer different questions:

1. **Coordinate-treatment route:** test one explicitly specified normalization or target-convention change against a matched control. A negative rejects that treatment at its tested budget; it does not eliminate every convention. The [main report](COORDINATE_CHECKLIST.md) states the input-distribution, clipping and VAE controls this would require. It is not yet an executable design.
2. **Fixed-coordinate learning route:** hold the audited oracle inputs and labels identical while comparing current versus broader trainable scope. If broader scope works, that demonstrates a working learner under this convention. It does not prove that coordinates never made adaptation harder. If both fail, it gives no justification to repeat passed transform checks.

**Neither route requires claiming “coordinates are irrelevant.”** Moving to the second route is a controlled study of the remaining mapping, not a conclusion that every coordinate effect has been disproved. If the desired conclusion is which factor caused the original failure, a scope-by-convention comparison may eventually be needed; that is a different, larger claim.

The immediate missing evidence is assessment of the existing no-PM run, not another full training run or another easy four-object fit. No commands are issued until the experiment design is accepted. The supporting [question checklist](COORDINATE_CHECKLIST.md), [historical evidence ledger](../EXPERIMENT_HISTORY.md) and [review record](../../../../coordinate_system_provenance/AUDIT_REVIEW_LOG.md) preserve all remaining questions and the limits of each result.
