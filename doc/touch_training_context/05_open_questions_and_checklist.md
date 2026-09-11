# Open questions and execution checklist

This file is the living edge of the context pack. It separates confirmed blockers
from design choices and provides gates for future work. Update it when a decision is
made; preserve rejected alternatives and evidence so later agents do not rediscover
the same ambiguity.

## Current status summary

| Area | Status | Blocking fact |
|---|---|---|
| Base fork/source audit | complete for commit `f91db41` | re-audit after source changes |
| Real touch/data audit | complete for local snapshot | only one object is materialized locally |
| Released Stage 1 source path | mapped; generator YAML audited | runtime condition probes remain |
| SS VAE YAML/weights | audited and hashed outside fork | cluster run needs checkpoint path |
| Stage 1 target | contract resolved; generator absent | alignment fixture before full cache |
| Training dataset/collator | absent | must be built |
| Touch encoder | absent | architecture remains provisional |
| Condition insertion | source seams identified | final context boundary needs runtime probe |
| Multimodal loss targets | unresolved | wrapper requires every configured latent key |
| Optimizer/trainer/checkpoint | absent | must be built |
| Hidden-geometry evaluation | absent | protocol/regions/metrics must be implemented |

## Decision 1: exact checkpoint contract

### Resolved in the target-latent audit

- SS encoder input/output is `[B,1,64,64,64] -> [B,8,16,16,16]`.
- Shape flow input is `[B,4096,8]`.
- Generator width/depth/heads are `1024/24/16`, with q/k RMS normalization.
- Exact latent keys are `shape`, `6drotation_normalized`, `translation`, `scale`, and
  `translation_scale`.
- Released image backbone is `dinov2_vitl14_reg` at input size 518.
- Released pointmap embedder uses input size 256 and patch size 8.
- VAE YAML/checkpoint hashes are recorded in `01_repository_and_evidence.md`.

### Open questions

- What class is `module.condition_embedder.backbone`?
- What class is `module.generator.backbone.condition_embedder`?
- Does raw modality fusion happen externally, internally, or in both?
- What is `ss_condition_input_mapping` in `pipeline.yaml`?
- What is final `[Lcond,Dcond]`?
- Is fuser fusion token-concatenation or feature-compression mode?
- What pose convention and pointmap normalizer are selected by `pipeline.yaml`?
- What are the runtime attention backend, dtype, and exact parameter paths?

### Required evidence

- authenticated official checkpoint download;
- resolved YAML snapshots and SHA-256 hashes;
- instantiated module tree and `named_parameters()` dump;
- forward-hook shape trace on one real preprocessed sample;
- untouched fixed-seed Stage 1 output fixture.

### Gate

No implementation should encode guessed `Dcond`, token count, latent names, pose
convention, or condition path.

## Decision 2: canonical shape target frame

### Decision record

- Target frame is renderer-normalized object frame `O`.
- Apply `T_normalized_from_source` to the original OBJ exactly once.
- The saved axis mapping is `(x,y,z) -> (x,-z,y)` before scale/translation.
- Do not call the stock voxelizer's rotation/normalization after applying the saved
  transform.
- Use fixed-bound Open3D surface occupancy at `64^3`, bounds `[-0.5,0.5]^3`.
- Occupancy axes and latent spatial axes are `(x,y,z)`; `z` varies fastest after
  flattening to `[4096,8]`.
- Cache raw posterior `mean [8,16,16,16]` as float32, one file per object.
- Do not apply camera transforms or Stage 2 latent normalization to shape targets.
- Meta's private target builder is unavailable; the surface policy is selected from
  the agreeing SAM main helper and TRELLIS dataset tool. The conflicting later fill
  demo is not used.

### Required validation before full generation

Create an asymmetric fixture with labeled/extreme features on +X, +Y, and +Z. Run
the selected transform through render, pointmap, touch, voxelization, SS encode, and
SS decode. Project decoded occupied coordinates to the pointmap frame with
`diag(-1,-1,1,1) @ T_camera_from_object` and compare with image/depth/touch. Record:

- voxel IoU against intended occupancy;
- decoded occupancy IoU;
- axis/centroid/bounds alignment;
- rendered projection overlay;
- behavior on an open mesh and disconnected mesh.

### Gate

The documented matrix chain and surface policy must pass automated and visual checks
before target-cache generation scales beyond fixtures.

## Decision 3: pose/layout target strategy

### Open questions

- Is first training shape-only or jointly shape/layout?
- If shape-only, will zero-weight placeholder layout targets preserve shape behavior
  under the actual protected-modality config?
- If joint, what scene scale/shift comes from the selected pointmap preprocessor?
- How exactly does `T_sam_from_object` map into `InstancePose` fields?
- Which outputs are log-scaled or normalized in the selected convention?
- What are the actual latent dictionary keys and dimensions?

### Required tests

- `InstancePose -> PoseTarget -> InstancePose` round trip on random and real transforms;
- model latent target -> pose decoder round trip;
- camera projection of transformed asymmetric target mesh;
- shape output equivalence under two different placeholder layout target values when
  only shape is supervised;
- gradient check proving zero-weight layout branches do not update selected parameters.

### Gate

Every configured latent key is supplied intentionally. No raw matrix field is treated
as a flow target without conversion tests.

## Decision 4: touch semantics and ablations

### Open questions

- Use all saved neighborhood points or hidden-only points?
- Use local XYZ, camera XYZ, local normals, geodesic distance, or visibility labels?
- Use center position only, center normal, 6D rotation, or full rotation?
- Is synthetic tangent roll acceptable as a feature?
- How many contacts and points/contact fit the compute budget?
- One token/contact, several tokens/contact, or an additional global token?
- Should contact order receive an index embedding?
- Is the synthetic ground-truth center-selection process representative of eventual
  sensor placement?

### Minimum ablation matrix

Keep this small enough to execute but sufficient to identify leakage:

| Content | Placement | Neighborhood filter | Purpose |
|---|---|---|---|
| none | none | none | exact image-only baseline |
| none | center only | n/a | oracle location control |
| local XYZ | center | hidden-only | local geometry without explicit orientation |
| local XYZ | center + normal | hidden-only | meaningful approach-axis orientation |
| local XYZ | center + full R | hidden-only | tests tangent-roll/canonical-pose value |
| local XYZ | center + selected orientation | all | quantifies visible/unknown inclusion |
| shuffled local XYZ | true placement | same | content causality control |
| true local XYZ | shuffled placement | same | placement causality control |

Nested contact-count and point-count curves can follow after one representation works.

### Gate

Any hidden-geometry claim must outperform center-only and shuffled controls on
object-held-out data.

## Decision 5: condition insertion and no-touch semantics

### Open questions

- Wrap the final base conditioner or extend `EmbedderFuser`?
- Can touch kwargs survive the pipeline's optional external pre-embedding path?
- Is exact image-only behavior required when the adapted model is loaded?
- How will per-example no-touch and CFG be represented without an attention mask?
- Is batch-level touch dropout sufficient initially?
- Does touch need a learned type vector or position group?

### Preferred first probe

With hooks, identify the tensor passed as `context` to the first shape cross-attention.
Construct a wrapper that returns the exact original tensor when touch is absent and
appends a known test token when touch is present. Confirm:

- no-touch fixed-seed output matches the unwrapped model;
- context length increases by exactly the planned token count;
- hooks see the appended values in every block;
- gradients return to the test token/touch projection in a direct loss call.

### Gate

Base checkpoint loading is strict except for an explicit allowlist of new adapter
state, and the no-touch behavior is characterized rather than assumed.

## Decision 6: parameter policy

### Open questions

- Touch encoder only, shape `to_kv`, or full shape cross-attention?
- If shared cross-attention is trained, what retention regularization or image-only
  batches are needed?
- Should layout cross-attention remain frozen?
- Is a zero-gated touch-specific residual branch warranted?
- What learning rate(s), precision, clipping, and scheduler fit the selected set?

### Required ladder

Run the same tiny-set protocol for:

1. touch encoder/appender only;
2. plus shape `to_kv`;
3. plus full shape cross-attention;
4. touch-specific gated adapter only if shared attention causes unacceptable drift.

For each, record trainable parameter count, peak memory, steps/sec, training loss,
true-versus-shuffled touch effect, hidden metric, and image-only retention.

### Gate

The chosen level should be the smallest one showing held-out touch sensitivity beyond
controls, not simply the level with lowest training loss.

## Decision 7: evaluation regions and success criteria

### Open questions

- How is a target voxel labeled visible/hidden from image/depth?
- What radius/dilation defines “inside supplied touch patches” in voxel space?
- Which occupancy threshold(s) and surface thresholds are reported?
- What minimum improvement and maximum image-only regression justify scaling?
- Is Stage 1 occupancy sufficient, or must final Stage 2 mesh quality be measured?

### Recommended region partition

For each target occupied voxel/surface point:

1. project into the input camera;
2. classify visible/hidden/unknown with the same tolerance policy as touch sampling;
3. compute distance to supplied touch patches in the agreed target frame;
4. report metrics for:
   - all geometry;
   - image-visible;
   - image-hidden;
   - hidden and near touch;
   - hidden and away from touch.

This separates direct local completion from broader shape-prior improvement.

### Gate

Success criteria must be written before the first larger run and evaluated with paired
noise seeds on held-out objects.

## Pre-implementation checklist

- [ ] Record current fork commit and dirty state.
- [x] Download and hash SS VAE YAML/weights and inspect released generator YAML.
- [ ] Save resolved config and module/parameter inventories.
- [ ] Capture untouched fixed-seed inference fixture.
- [ ] Resolve final condition boundary and all real tensor shapes.
- [x] Specify the target frame, tensor order, and first-cache occupancy policy.
- [ ] Validate the target contract with an asymmetric encode/decode/projection fixture.
- [ ] Resolve pose convention or validate shape-only placeholder strategy.
- [ ] Build immutable validated manifest from existing files.
- [ ] Decide all-vs-hidden neighborhood and fixed C/N contract.
- [ ] Write data/frame invariants before writing the trainer.

## One-batch training checklist

- [ ] Input paths exist and versions match the accepted index.
- [ ] Image/mask/pointmap preprocessing matches inference.
- [ ] Touch center/offset normalization matches pointmap normalization.
- [ ] Shape target is posterior mean in exact flatten order.
- [ ] Every configured latent key is present.
- [ ] Final condition dtype/width matches cross-attention.
- [ ] Context length is fixed within the batch.
- [ ] Total/per-modality losses are finite.
- [ ] Intended parameters have finite, nonzero gradients.
- [ ] Frozen parameters have no gradients.
- [ ] One optimizer step changes only allowlisted parameter tensors.
- [ ] Adapter checkpoint round-trips to identical forward output.

## Tiny-overfit checklist

- [ ] Fix object/view IDs, flow noise, and preprocessing.
- [ ] Save decoded occupancy every small number of steps.
- [ ] Confirm loss reduction is not only a zero-weight/layout artifact.
- [ ] Compare true, no, shuffled-content, and shuffled-placement touch.
- [ ] Inspect hidden and touch-near regions separately.
- [ ] Check for axis flips, mirrored shape, wrong depth, or scale drift.
- [ ] Confirm no-touch regression remains understood.
- [ ] Resume from checkpoint and reproduce the next step/loss.

## Small-run checklist

- [ ] Object-disjoint train/validation index.
- [ ] Complete target-cache validation and encoder hash match.
- [ ] Data-loader failure policy and corrupt-sample log.
- [ ] Paired validation seeds and fixed sample panel.
- [ ] Center-only and shuffled controls.
- [ ] Hidden-near-touch and hidden-away metrics.
- [ ] Per-object distributions, not only aggregate means.
- [ ] Peak memory, throughput, and adapter size recorded.
- [ ] Exact resolved config and code commit saved with results.

## Decision-record template

Append decisions below this section or in dated companion files. Do not erase the
alternatives that were considered.

```markdown
## YYYY-MM-DD — <decision title>

Status: proposed | accepted | superseded

Question:

Evidence:

Decision:

Rejected alternatives and why:

Affected files/configs:

Tests/metrics that validate it:

Rollback/migration notes:
```

## Work deliberately not performed during this audit

- No source/data/config/checkpoint file was edited.
- No model or dataset generation job was run.
- No tests were run; current touch tests are visibly stale relative to format v4.
- No gated checkpoint was downloaded or authenticated.
- No target representation or touch encoder was selected as final.
- No training code, optimizer, or cluster job was created.

The only intended changes are the Markdown files in this context directory.

## External references inspected

- [Official SAM 3D Objects repository](https://github.com/facebookresearch/sam-3d-objects)
- [Official setup/checkpoint instructions](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
- [Official Hugging Face model repository](https://huggingface.co/facebook/sam-3d-objects)
- [SAM 3D paper, arXiv 2511.16624](https://arxiv.org/abs/2511.16624)
- [SAM 3D paper HTML, v2](https://arxiv.org/html/2511.16624v2)

The checkpoint repository was gated during the initial audit. The later target-latent
audit obtained the official SS encoder/decoder YAML and weights plus the released
generator YAML; their verified hashes are recorded in
`01_repository_and_evidence.md`.

## Maintenance rule for this context pack

When implementation begins:

1. replace **Open** entries with measured values and evidence paths;
2. add a dated decision record for every material choice;
3. update commit/data/config hashes;
4. keep historical caveats if old data/checkpoints remain in circulation;
5. distinguish runtime observations from design intent;
6. re-run the documented invariants after any frame, preprocessor, target, or touch
   format change.
