# Training integration design space

Everything in this document is **Provisional** unless explicitly marked Confirmed.
It is a map of viable implementation choices and their consequences, not a fixed
project specification.

## Objective envelope

A minimal first experiment can be stated without committing to one architecture:

> Add a bounded number of tokens derived from local surface neighborhoods and their
> camera-relative placement to the Stage 1 condition context; optimize a small set of
> new and/or condition-facing parameters; measure whether the decoded sparse structure
> improves in image-hidden geometry while retaining image-only behavior.

The current files are synthetic hidden-mesh conditioning, not a physical tactile
simulation. Claims should use language such as “touch-like local surface geometry” or
“touch-conditioned hidden geometry” unless a later data source models sensor contact,
noise, force, approach, and visibility more realistically.

## Confirmed constraints that shape the design

1. Stage 1 ultimately needs condition tokens `[B,L,Dcond]` shared by all MoT blocks.
2. Current data has 32 contacts but variable points/contact, roughly 4k–6k at full
   density in the local sample.
3. Full self-attention over every dense point is impractical: one view has about
   170k flattened points. Points must be subsampled, pooled pointwise, or processed
   with a sparse/local method.
4. Stage 1 condition attention exposes no padding mask. Fixed output token length is
   the safest first batching contract.
5. Adding touch directly to `EmbedderFuser` can alter its shared width, module indices,
   projection shapes, and learned position table, which conflicts with strict base
   checkpoint loading.
6. A zero touch token is not equivalent to no token because K/V projections may have
   bias and softmax still includes its position.
7. The inference API is enclosed in `no_grad`; training needs a direct differentiable
   path.
8. Stage 1 target generation and the full training stack are absent, although the
   target tensor/frame contract is now resolved.
9. The VAE config/weights and released generator YAML have been audited; full runtime
   condition-path and parameter probes are still required before model integration.

## Recommended separation of concerns

Keep these components independently testable:

```text
validated manifest/index
       |
       +--> image/mask/pointmap loader + released preprocessor
       +--> object-level shape target loader
       +--> per-view touch loader/collator
                         |
                         v
                  TouchEncoder
                         |
                         v
             TouchConditionAppender
       base condition tokens + touch tokens
                         |
                         v
              released Stage 1 flow loss
                         |
                         v
       adapter checkpoint + validation sampler
```

Target creation should be a separate offline command. Do not voxelize and run the
frozen SS encoder in every training iteration.

## Touch input contract

A practical in-memory batch can use:

```text
points_local:        [B,C,N,3]
point_mask:          [B,C,N] bool
centers_camera:      [B,C,3]
R_camera_from_local: [B,C,3,3]
contact_mask:        [B,C] bool          # preferably all true initially
```

Optional fields:

```text
normals_local:       [B,C,N,3]
point_visibility:    [B,C,N]
geodesic_distance:   [B,C,N,1]
patch_radius:        [B,C,1]
```

Use one consistent coordinate space:

- local offsets must be scaled consistently with pointmap scene normalization;
- centers must be transformed by the same pointmap normalizer;
- full rotations/normals must be transformed correctly if the normalizer includes
  more than isotropic scale and translation.

Keep raw IDs, priorities, and barycentric coordinates as audit/debug metadata; they
need not be model features in the first experiment.

## Point selection before the encoder

The local sample averages 5,315 points/contact. A small transformer over that many
points per contact is quadratic and not “minimal.” Candidate first contracts:

### Fixed priority top-K

For each contact, select the `N` points with smallest `keep_priority` and retain the
center explicitly. Suggested experiment range is 64–256 points/contact, selected by
memory profiling rather than assumption.

Advantages:

- fixed shape;
- deterministic and nested as N increases;
- uses the same area-uniform master pool;
- easy per-contact self-attention mask.

Caveats:

- it is not an exact density fraction;
- very small contacts require padding/repetition policy;
- random area sampling may undersample distinctive edges despite deterministic order.

### Priority threshold

Keep `keep_priority < fraction`. This is ideal for nested density ablations but leaves
variable counts. Use masked pointwise pooling or pad only inside the touch encoder,
where a real point mask exists.

### Geometric downsampling

Farthest-point or voxel downsampling can preserve spatial coverage better, but adds
compute and another geometry policy. It should be compared only after the deterministic
priority baseline works.

Whichever policy is chosen, force-include the zero-offset center point and unit-test
the exact selection order.

## Provisional minimal touch encoder

One conservative architecture is one output token per contact:

```text
per point:
  local XYZ (+ selected optional features)
  -> small MLP / linear position embedding
  -> 1–2 self-attention or residual point blocks within each contact
  -> masked mean/max/CLS pooling

per contact placement:
  normalized camera center
  + chosen orientation representation
  -> small MLP

fuse local-patch summary + placement embedding
  -> contact token [B,C,Dtouch]
```

With `C=32`, this adds only 32 condition tokens. The model can preserve contact-level
structure through centers and token content without an attention mask.

### Orientation choices

Test progressively rather than assuming the full rotation is beneficial:

1. no explicit orientation;
2. center normal only (third rotation column);
3. a continuous 6D rotation representation;
4. full 3x3 rotation flattened;
5. per-point local normals in addition to center orientation.

The full rotation contains a synthetic tangent roll tied to object axes. It may help
canonicalization while also leaking pose. Center-normal and camera-space-XYZ controls
are needed to interpret gains.

### Contact-to-contact interaction

An optional small transformer over the 32 contact tokens can model spatial relations.
It is not necessary for the first smoke test because the Stage 1 transformer already
attends all condition tokens jointly. If added, avoid relying solely on contact index;
contact order is a sampler construction, not an intrinsic physical sequence.

### Required encoder API

The encoder should:

- expose `embed_dim` if inserted into `EmbedderFuser`;
- accept a single structured touch batch object or explicit tensors through a clearly
  documented wrapper;
- implement true point masking before softmax/pooling;
- return fixed `[B,C,Dtouch]` in the first version;
- have deterministic eval behavior;
- validate finite values and orthonormal rotations at the boundary.

## Where to append touch tokens

The real checkpoint probe must identify where final context `[B,L,Dcond]` is formed.
Two main approaches are viable.

### Option A: output wrapper around the base conditioner

Load the untouched base checkpoint/config first, then wrap the conditioner that emits
the final cross-attention context:

```python
base_tokens = frozen_base_conditioner(original_inputs)
touch_tokens = touch_projection(touch_encoder(touch_batch))
context = torch.cat([base_tokens, touch_tokens], dim=1)
```

Advantages:

- preserves base conditioner parameter names/shapes;
- avoids changing the original fuser's `module_list`, `projection_nets`, or `idx_emb`;
- permits explicit omission of touch tokens for an exact image-only path;
- keeps adapter checkpoint state compact and obvious.

Risks:

- the pipeline currently discards original kwargs after an optional external
  conditioner produces tokens, so the wrapper must be placed where touch data is
  still available or the condition plumbing must preserve it;
- `Dcond` must be measured from the base output;
- modality dropout/CFG needs explicit handling outside the original fuser.

This is the safest **provisional first choice** after the exact condition path is
probed.

### Option B: extend `EmbedderFuser`

Append a touch encoder entry and a position group to the fuser.

Advantages:

- uses released multimodal projection, grouping, and dropout machinery;
- touch is configured like other modalities.

Risks and required safeguards:

- if `Dtouch` exceeds the previous maximum encoder width, `embed_dims` grows and every
  existing projector changes shape;
- appending a new learned position group grows `idx_emb`, causing strict checkpoint
  shape mismatch;
- inserting rather than appending changes module/projection indices;
- both `embedder_list` and registered `module_list` references must remain consistent;
- fuser `freeze=True` freezes the new touch encoder as well;
- compression mode may be incompatible with a new token length.

If this route is used, keep `Dtouch <= existing Dshared`, append rather than reorder,
and either:

- instantiate/load the original fuser first and extend it with an explicit state
  migration; or
- use a loader that verifies the only missing/new state belongs to touch modules and
  intentionally copies old position rows.

Never hide broad missing/unexpected checkpoint keys behind uninspected
`strict=False`.

## Image-only and no-touch behavior

There are three distinct requirements:

1. **Exact original inference:** no touch tokens are appended at all. This can and
   should match the untouched base model bit-for-bit or within the expected numerical
   tolerance under fixed seed/hardware.
2. **Learned no-touch condition:** fixed touch-token slots contain learned null values.
   This is useful for mixed per-example batches but is not initially equal to the
   original context length.
3. **Dropped touch during CFG/training:** zeroed or null tokens remain in attention and
   must be learned as unconditional behavior.

A simple first trainer can alternate whole batches with touch appended and whole
batches using the exact original condition length. This avoids per-example variable
length inside one batch and supports a retention loss/diagnostic if cross-attention is
unfrozen. More sophisticated mixed-batch masking requires extending attention or
accepting learned null-token semantics.

## Which parameters to train

Treat this as an empirical ladder. Always start by printing and asserting the exact
trainable set.

### Level 0: touch encoder/appender only

Train:

- touch point/contact encoder;
- touch-to-`Dcond` projection;
- touch type/position embedding or gate, if used.

Freeze:

- image/mask/pointmap encoders;
- base fuser;
- all Stage 1 flow transformer parameters;
- SS encoder/decoder;
- pose decoder and Stage 2.

Rationale: the new encoder may learn tokens compatible with frozen `to_kv`. This is
the cleanest capacity/retention baseline.

### Level 1: add shape `to_kv`

Additionally train, in every block:

```text
cross_attn.<shape_transformer_key>.to_kv.*
```

and possibly cross K normalization only if the actual config and rationale support
it.

Rationale: directly adapts context interpretation. Cost: image and touch share this
projection, so image behavior can drift.

### Level 2: full shape cross-attention

Additionally train shape:

- `norm2` affine;
- `to_q`;
- `to_kv`;
- q/k RMS norms if present;
- `to_out`.

This is what “unfreeze shape cross-attention” should mean when used precisely. It is a
larger adapter and can alter latent querying/output mixing, not only touch uptake.

### Level 3: modality-specific residual adapter

Add a separate touch cross-attention or low-rank residual branch with a gate initialized
to zero. This can preserve the pretrained image path exactly at initialization and
avoid changing shared image K/V.

It is architecturally cleaner for retention but modifies every block and is less
minimal. Consider it if Levels 0–2 cannot gain touch sensitivity without image drift.

### Layout parameters

If the experiment supervises pose/layout or wants touch to refine layout, unfreeze the
actual merged layout transformer key's cross-attention separately. Shape and layout
have distinct cross-attention modules. Do not assume a Python key such as `pose` until
printing `latent_names`.

For shape-only experiments, leave layout cross-attention frozen initially.

### Freeze implementation checklist

1. Call `requires_grad_(False)` on the entire loaded model.
2. Enable named new modules by object reference.
3. Enable the selected cross-attention submodules explicitly.
4. Set frozen pretrained modules to eval; set trainable modules to train.
5. Build the optimizer from `p for p in model.parameters() if p.requires_grad`.
6. Print names/shapes/counts and compare with an allowlist.
7. After one backward pass, assert:
   - every intended parameter has a finite gradient;
   - every frozen parameter has no gradient;
   - optimizer parameter IDs equal intended parameter IDs exactly.

## Loading and saving adapters safely

The official generator checkpoint is large. Prefer an adapter checkpoint containing:

- touch encoder/appender state;
- any selected cross-attention state;
- optimizer/scheduler/scaler state for resume;
- resolved experiment config;
- base fork commit;
- base YAML hashes and checkpoint SHA-256;
- target-cache version and SS encoder hash;
- touch format/version and validated dataset-index hash;
- current epoch/step and RNG states.

On load, instantiate the exact base model first, verify its identity, then apply the
adapter with strict checking against an explicit adapter-key set. Refuse to run when
the base hash/config differs unless a deliberate migration is requested.

## Stage 1 target generation

### Confirmed frame decision

Targets use the renderer-normalized object frame `O`. Load the original OBJ and apply
`object_transform.npz["T_normalized_from_source"]` exactly once. That saved transform
already contains the actual axis conversion `(x,y,z) -> (x,-z,y)`, AABB centering,
and largest-extent normalization. Do not apply the stock SAM voxelizer's rotation or
normalization afterward, and do not transform the target into any camera frame.

For a view-only validation overlay:

```text
T_sam_from_object = diag(-1,-1,1,1) @ T_camera_from_object
p_sam = T_sam_from_object @ p_object
```

The target itself remains object-shared and view-independent.

### Selected occupancy policy

Use fixed-bound Open3D surface voxelization at resolution 64, voxel size `1/64`, and
bounds `[-0.5,0.5]^3`. Write Open3D grid indices directly to occupancy axes
`(x,y,z)`. Do not use the later demo notebook's Trimesh-local `sparse_indices` as
global coordinates.

The private SAM training-data builder is unavailable, so surface-versus-solid cannot
be proven from hidden training code. Fixed-bound surface occupancy is selected because
SAM's main helper and TRELLIS's released dataset builder agree on it, it preserves an
explicit global frame, and it behaves consistently on the open/nonmanifold meshes in
this dataset. Do not mix filled targets for watertight objects with surface targets for
other objects in the first experiment.

### Deterministic latent

Use:

```text
occupancy [1,1,64,64,64] float
-> frozen SS encoder in eval/no-grad
-> posterior mean [1,8,16,16,16]
-> permute/flatten [4096,8]
```

Avoid sampled posterior `z` for cached supervision.

### Target artifact

Save one file per represented object:

```text
generated_data/<object_id>/target_latent.npz
  mean [8,16,16,16] float32
```

Keep the spatial tensor in the file and flatten only in the loader:

```python
shape_target = mean.permute(1, 2, 3, 0).reshape(4096, 8)
```

A single run-level JSON should record target format version, occupancy policy,
resolution, encoder/config SHA-256, and source commit. This avoids duplicating the same
metadata in every object file while preserving reproducibility. Use `allow_pickle=False`,
write files atomically, skip already valid targets, and make every manifest row for an
object reference the same relative path.

### Self-contained offline generator boundary

`sam-3d-touch-data/objaverse-dexonomy/generate_target_latents.py` should be independent
of rendering and touch sampling. Its responsibilities are:

1. read unique accepted object IDs from `generated_data/samples.jsonl`;
2. resolve each source `objects/<id>/model.obj` and saved
   `generated_data/<id>/object_transform.npz`;
3. load the official SS encoder once on the selected GPU;
4. apply each saved source-to-normalized transform once and validate center/extent;
5. build fixed-bound surface occupancy without touch-mesh cleanup/subdivision;
6. encode deterministic posterior means, optionally in small batches;
7. atomically save `target_latent.npz` in each generated object folder;
8. resume by validating/skipping existing targets and log failures separately;
9. rewrite manifest `target_path` values to the object-shared relative path;
10. print object-level progress, elapsed time, and ETA.

It must not rerender, resample touch, alter source meshes, or create one target per
view. A single-object mode is useful for the initial alignment test.

## Handling multimodal latent targets

The wrapper requires every configured latent key even if only shape loss is desired.
Three practical routes should be tested in order of simplicity.

### Route 1: correct targets for all modalities

Convert `T_sam_from_object` through the checkpoint's selected pose convention and
pointmap scene normalization. Use actual generator keys and configured loss weights.

This is the most faithful route but requires frame/convention round-trip validation.

### Route 2: all keys present, zero loss on layout

Supply shape target plus deterministic placeholder targets for every layout key, and
set layout loss weights to zero. Layout latents still run through the model, satisfying
`project_input`, while only shape MSE contributes.

Because the released self-attention protects shape from layout, this may be a viable
minimal shape-only route. It remains **Provisional** until the actual config confirms
the protected key and a gradient/forward equivalence test shows placeholders do not
affect shape output.

### Route 3: custom shape-only wrapper/loss

Bypass or specialize multimodal projection to run only shape. This can reduce compute
but creates a larger divergence from the pretrained architecture and checkpoint. Use
only with clear tests against the original shape branch.

Do not silently use raw R/t/s numbers as flow targets. Pose convention errors can
produce apparently finite losses while training the wrong geometry/layout relation.

## Dataset and collator

The trainer needs a new validated dataset layer. Its responsibilities should include:

- consume a frozen validated index, not append-oriented `samples.jsonl` directly;
- filter to existing files and compatible target/touch versions;
- enforce object-level split and optionally complete-view requirements;
- load RGBA, pointmap, camera, touch, and object-shared target;
- run the same Stage 1 preprocessor/config as inference;
- select touch contacts/points deterministically;
- transform touch placement with the same pointmap normalization;
- return all configured latent targets or explicit placeholders;
- collate fixed contact tokens and masked within-contact points;
- preserve sample/object/view IDs for diagnostics.

Avoid random 2D/3D augmentation initially. Crops, flips, rotations, or scales must be
co-applied consistently to image, mask, pointmap, camera pose, touch placement,
occupancy frame, and layout target. A partially applied augmentation is a silent
cross-modal corruption.

Cache object-level shape targets and, optionally, frozen image condition tokens. Do
not cache final condition tokens if the base image preprocessing/config may change or
if any base conditioner parameters are trainable.

## Training entry point responsibilities

No released trainer exists. A minimal reliable entry point needs:

- resolved config and command-line overrides;
- deterministic seed setup for Python, NumPy, torch, workers, and distributed ranks;
- base checkpoint/config verification;
- model construction and explicit freeze policy;
- dataset/sampler/collator;
- optimizer and optional scheduler;
- autocast/gradient scaling appropriate to the actual GPU/dtype;
- gradient accumulation and clipping;
- periodic adapter checkpoint and exact resume;
- structured logging of total and per-modality losses;
- validation with fixed sample IDs and fixed flow noise seeds;
- distributed-safe metric reduction;
- graceful handling of NaN/Inf and corrupted samples.

The paper reports AdamW with zero weight decay. That is a reasonable initial optimizer
reference, but learning rate/betas/schedule for a small adapter should be tuned by
one-batch and small-set experiments rather than copied from full 1.2B-parameter
training.

## Validation and metrics

### Paired inference protocol

For every validation sample, use the same base checkpoint, image, mask, pointmap,
initial flow noise seed, solver, and step count for:

- untouched base image-only;
- adapted model image-only with no touch tokens;
- adapted model with true touch;
- adapted model with shuffled touch from another object/view;
- adapted model with center/orientation only;
- optionally adapted model with touch points but shuffled placement.

Paired seeds reduce flow-sampling variance and make touch sensitivity interpretable.

### Geometry metrics

At Stage 1 occupancy level:

- voxel IoU, precision, recall, and F1 at decoder threshold 0;
- visible-region and hidden-region versions using camera/depth projection;
- metrics inside a dilation of supplied touch patches and outside those patches;
- connected-component and occupied-voxel-count diagnostics.

At surface level after extracting a mesh/point set:

- symmetric Chamfer distance;
- F-score at one or more normalized thresholds;
- normal consistency if target normals are reliable;
- hidden-surface variants.

If layout is trained, report rotation geodesic error, translation error in the decoded
metric frame, and relative scale error after the configured pose conversion.

### Retention and causality checks

- The adapted no-touch path should match the untouched base when touch modules are
  omitted and no shared base parameter has changed.
- True touch should outperform shuffled/wrong touch if the model uses contact content.
- Center-only versus patch geometry separates oracle location from local shape value.
- Full rotation versus normal-only exposes dependence on synthetic tangent roll.
- All-point versus hidden-only patches exposes visible/unknown-point leakage.
- Contact-count and point-density curves should be nested and evaluated under the
  same object/view/noise set.

## Provisional implementation sequence and gates

### Phase 0 — recover exact model contract

Deliverables:

- gated configs/weights downloaded and hashed;
- resolved config snapshot;
- final-condition and latent shape probes;
- parameter inventory;
- untouched inference fixture.

Gate: no guessed dimensions, latent keys, or pose convention remain in the planned
code path.

### Phase 1 — make targets trustworthy

Deliverables:

- asymmetric frame fixture;
- selected occupancy policy with frozen encoder/decoder round trip;
- object-level deterministic target schema/generator;
- target alignment visualization and tests.

Gate: target, render, pointmap, touch, and pose transforms agree.

### Phase 2 — validated dataset/collator

Deliverables:

- immutable filtered index;
- data loader with fixed touch contract;
- normalization round-trip tests;
- batch shape/dtype/range report.

Gate: repeated loads are deterministic and every batch invariant passes.

### Phase 3 — touch encoder in isolation

Deliverables:

- minimal point/contact encoder;
- point-mask and permutation tests;
- memory/throughput profile at chosen C/N;
- overfit of a toy supervised contact task if useful.

Gate: output is finite `[B,C,Dtouch]`, gradients reach every intended parameter, and
padding does not affect valid outputs.

### Phase 4 — condition integration

Deliverables:

- token appender/fuser extension chosen from real config;
- exact image-only regression path;
- fixed touch context length;
- condition forward hooks and checkpoint-load tests.

Gate: base checkpoint loads with only explicitly allowed new state, no-touch output
matches baseline, and touch tokens reach every intended cross-attention block.

### Phase 5 — one-batch flow training

Deliverables:

- complete latent target tree or validated shape-only placeholder strategy;
- explicit freeze/optimizer allowlist;
- finite forward/backward in intended precision;
- gradient audit;
- adapter save/load/resume test.

Gate: only intended parameters change after an optimizer step.

### Phase 6 — tiny-set overfit

Use one object/few views, fixed noise, frequent decoded occupancy inspection.

Gate: loss falls, true touch affects output in the expected direction, wrong-touch
controls degrade, and there is no coordinate-frame symptom.

### Phase 7 — small split experiment

Train on a small object-disjoint subset with paired validation and ablations.

Gate: hidden-geometry gain survives held-out objects and exceeds center-only/shuffled
controls without unacceptable image-only regression.

### Phase 8 — scale and robustness

Only now add distributed training, larger target generation, hyperparameter sweeps,
more contact/density settings, corrupted-mesh policy, and physical-touch data if
available.

## Suggested code boundaries

Names are provisional; responsibilities matter more than paths.

```text
sam3d_objects/model/backbone/dit/embedder/touch.py
  TouchEncoder
  TouchConditionAppender

sam3d_objects/data/dataset/touch/
  manifest validation
  target/touch loading
  collator and normalization adapter

tools/touch_targets.py
  offline occupancy/latent cache generation

tools/train_touch_stage1.py
  trainer entry point

tests/touch/
  data/frame/encoder/fuser/freeze/loss/checkpoint regression tests
```

Do not force this layout if the project acquires an established training framework.

## Common failure modes to guard against

- double-rotating or double-normalizing the target mesh;
- feeding OpenCV camera touch coordinates while pointmaps use SAM axes;
- normalizing centers but not local offsets;
- using sampled VAE `z` as a changing target;
- treating format-v2 keys/tests as current;
- assuming all neighborhood points are hidden;
- full attention over thousands of points/contact;
- zero-padding Stage 1 condition tokens without a mask/null-token design;
- increasing fuser shared width and breaking every pretrained projection;
- loading a modified model with broad `strict=False` and missing pretrained weights;
- freezing the whole fuser after attaching the touch encoder;
- unfreezing shared `to_kv` while claiming only touch behavior can change;
- calling inference sampling under `no_grad` during training;
- providing only shape in a latent dictionary that requires layout keys;
- computing validation with different flow noise for baseline and touch;
- splitting by views rather than objects;
- scaling training before an asymmetric coordinate fixture passes.
