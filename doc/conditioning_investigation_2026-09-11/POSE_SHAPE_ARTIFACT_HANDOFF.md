# Saved occupancy transfer for the selected pose/shape branch

**Completed:** the bundle arrived and all 1,232 predictions were analyzed on CPU. See [POSE_SHAPE_RETURNED_FINDINGS.md](POSE_SHAPE_RETURNED_FINDINGS.md). The command below is historical; do not rerun it.

Continues branch A of [EVIDENCE_AUDIT_AND_BRANCH_PLAN.md](EVIDENCE_AUDIT_AND_BRANCH_PLAN.md). No new inference or training. The latest prediction arrays are not local, so no pose-adjusted outcome is claimed.

`bundle_alignment_geometry.py` uses only the Python standard library. It runs the unchanged four-shard report analyzer, requires every saved alignment NPZ, and collects all 294 batch artifacts (21 conditions × seven view groups × two seeds), without selecting favorable cases. It copies their `predicted_occupancy.npy` and `target_occupancy.npy` members byte-for-byte, omitting latent arrays to reduce transfer size. The archive includes all four raw reports, validated aggregate analysis inside its manifest, original NPZ hashes and copied member hashes. This also supplies the previously missing raw shard 2 report.

The helper additionally looks for the 56 historical natural-camera multi-view predictions beside the recorded oracle fit directory. It includes this comparison only if the exact historical report and all its NPZs are present. Missing camera artifacts are explicitly recorded as unavailable; the entire optional camera comparison is omitted rather than partially selected. If the camera report differs from its reference, the helper stops. A moved directory can be supplied with `--camera-fit-dir /actual/path`.

Preflight verifies required files and NPY members and ZIP CRCs. Numerical array dimensions, target agreement and raw metric reproduction belong to the subsequent CPU analysis. Existing archives are never overwritten. No source artifacts or model weights are modified.

Local verification: synthetic 294-artifact transport checks passed for coverage, copied-member hashes, omission of latent members, existing-output protection, missing-required-file rejection and path containment. The report validator was stubbed only in that synthetic fixture; no fake report or array was added to actual evidence. Real reports/arrays will be checked by the unmodified analyzer on the cluster. CLI help and source syntax passed.

After syncing the new helper, paste into a cluster terminal where the saved outputs are accessible. A GPU allocation is not required:

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
python doc/conditioning_investigation_2026-09-11/bundle_alignment_geometry.py \
  --alignment-root outputs/conditioning_investigation/alignment_tolerance/manual \
  --output outputs/conditioning_investigation/alignment_geometry_bundle.zip
)
```

Return `alignment_geometry_bundle.zip`. If required files or source/replay checks fail, return the traceback instead of rerunning training/sampling. The printed archive size is measured; no size/runtime estimate is claimed in advance.

After receipt, validate archive hashes and reproduce the raw metrics first. Then follow branch A's fixed-frame, known-inverse and separately estimated rigid-alignment comparisons with positive/negative registration controls. Preserve individual failures, distinguish scale from rigid pose, and keep target-assisted results separate from deployment claims. Full training remains unearned.
