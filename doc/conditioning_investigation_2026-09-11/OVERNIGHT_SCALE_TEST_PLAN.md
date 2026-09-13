# Proposed overnight scale test after F12

Status: parked at the user's request; not implemented or launched. The user asked to continue the short coordinate branch first. Keep this overnight option available for later, without treating it as the active experiment. No overnight trainer code was changed. This is a proposed scale/transfer diagnostic of a supported small-task intervention, not promotion of a proven general fix. Full surfaces remain primary.

Starting evidence: F9 shows accurate oracle+dropout reconstruction on reserved views of four fitted identities. F12 shows that camera+dropout does not achieve the same result. F6 shows poor new-object native losses after four-/16-object training, but cannot decide whether full-data aligned-dropout training learns reusable geometry. The user reports previous full-data oracle runs were similar to other variants; do not assume oracle-only full training is new.

## Priority arms

1. Full surface, oracle coordinates, 50% whole-visual dropout.
2. Matched full surface, oracle coordinates, zero visual dropout. Reuse an existing oracle run if its actual config, split, initialization, optimizer scope, global batch and step budget match; otherwise a concurrent control makes the overnight comparison interpretable.

Retain frozen VecSetX, no-position full-surface interface, full shape cross-attention adaptation, image+pointmap present for primary assessment, and fixed asset-frame targets. Keep effective batch, training exposure, data/sampling and checkpoint steps matched. Fifty percent follows the small-task tested treatment; it is not an optimized or generally recommended rate.

Existing image+pointmap checkpoints provide a practical reference when their settings match. If no matched reference exists, add an image+pointmap control before treating a gain over oracle-only as useful surface conditioning. Do not compare unrelated best checkpoints selected by different objectives/steps as a clean causal result.

Assess training objects and identity-disjoint validation objects, both with correct and wrong surfaces. Preserve intermediate and final checkpoints; do not use lowest native validation loss alone to declare success. Primary scientific decision is useful target-referenced Stage-1 geometry on held identities, with raw frame and pose-adjusted proximity reported separately. Native loss remains the training-objective endpoint. Use fixed sampling settings/noise and actual decoded-target support, not Stage 2. Predefine the quantitative evaluation/selection rule before launching the prepared implementation.

Branches: improved fit only means scale did not yet yield transfer; improvement over oracle-only without beating image or benefiting from correct surfaces does not establish geometric utility; replicated held-object shape and surface-utility gains justify broader confirmation and then sparse-touch adaptation. A poor result with exact alignment shows alignment plus this dropout policy is insufficient at the tested scale/budget; it does not make the coordinate burden imaginary or establish architectural impossibility.

## Why touch is not the primary second arm

`configs/data1.yaml` specifies 32 contact neighborhoods, 256 points/contact and 0.10 geodesic radius, so it changes spatial coverage/structure, not simply point count. Current `dataloader.py` rejects oracle coordinates for touch input. `train.py` requires no-position conditioning for oracle; that normalizes the sampled cloud and omits its center/scale. This is tolerable as the existing full-surface control but is not automatically a faithful globally located touch representation. Whole-visual dropout also asks incomplete contacts to predict missing visible geometry; 50% is untested for that task.

An oracle+dropout touch run is therefore a separate exploratory configuration requiring an explicit location/scale contract and its own no-dropout comparison. It should not replace the full-surface control or be interpreted as a clean test of sparse touch's potential. No unsupported command is handed off.

## Short coordinate work in parallel

Retain F12's selected intervention branch: shared observable target orientation if camera/robot-frame output is acceptable, or observable frame recovery if original asset axes are required. That output-contract preference remains unanswered. The overnight scale test addresses whether the known aligned positive control scales; the short intervention addresses how to remove the unavailable oracle transformation. These are separate, complementary questions.

Implementation prerequisite: visual dropout currently exists in diagnostic fit scripts only, not production `train.py`. Port its visual-token-zeroing semantics with explicit training-only activation, reproducible masks and unchanged random-noise scheduling for the paired control, DDP behavior, checkpoint metadata/resume validation, and all-visual assessment. Preserve hashed historical probes rather than editing their recorded sources silently. Provide actual interactive shell contents when implemented; no sbatch-only handoff.
