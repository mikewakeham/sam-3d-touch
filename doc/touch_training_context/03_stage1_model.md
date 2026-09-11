# Stage 1 model internals

This document traces the released Stage 1 source from input preprocessing through
conditioning, multimodal flow prediction, sparse-structure decode, and pose decode.
The sparse-structure VAE and released generator YAML were inspected in the target-
latent follow-up audit. Items that still require the full instantiated pipeline or a
runtime probe remain marked **Open**.

## Stage 1 at a glance

The paper and released source agree on the following conceptual model:

```text
image + mask (+ optional pointmap)
          |
          v
condition encoders/fuser -> condition tokens [B, Lcond, Dcond]
          |
          v
multimodal flow transformer (MoT)
  shape latent:       [B, 4096, 8]
  rotation latent:    [B, 1, 6]      (paper-level representation)
  translation latent: [B, 1, 3]
  scale latent:       [B, 1, 3]
  translation scale:  [B, 1, 1]
          |
          +--> shape [B,4096,8] -> reshape [B,8,16,16,16]
          |                         -> SS decoder -> [B,1,64,64,64] logits
          |                         -> active voxels where logit > 0
          |
          +--> layout latent fields -> configured pose decoder -> R, t, s
```

The inspected released generator YAML uses the exact keys `shape`,
`6drotation_normalized`, `translation`, `scale`, and `translation_scale`.

## Model initialization and checkpoint boundaries

`InferencePipeline` loads Stage 1 components separately:

- generator config: `ss_generator.yaml["module"]["generator"]["backbone"]`;
- generator checkpoint state prefix: `_base_models.generator.`;
- optional external condition embedder config:
  `ss_generator.yaml["module"]["condition_embedder"]["backbone"]`;
- external condition state prefix: `_base_models.condition_embedder.`;
- SS encoder and decoder from their standalone config/checkpoint paths;
- pose decoder name from `module.pose_target_convention` unless explicitly overridden.

For ordinary checkpoint files, `load_model_from_checkpoint` is called with strict
loading, `freeze=True`, and `eval=True`. Safetensors loading is `strict=False` and
sets eval mode but does not visibly apply the same helper freeze. A training entry
point must explicitly set `requires_grad` rather than relying on how inference loaded
the checkpoint.

The pipeline then overrides several runtime generator settings, including inference
steps, CFG strengths/interval, time rescaling, unconditional handling, and
`ss_generator.reverse_fn.backbone.condition_embedder.normalize_images = True`.
This last assignment confirms that the released runtime expects an internal condition
embedder object with that attribute; its exact class remains config-dependent.

## Input preprocessing

### Image and mask

The base pipeline accepts an RGBA image or merges an RGB image and mask. Its
preprocessors construct both object-crop and full-image streams. Conceptually, the
released conditioning system can encode:

- cropped RGB;
- cropped mask;
- full RGB;
- full mask;
- optional pointmap streams in the pointmap pipeline.

The exact kwarg names, enabled streams, input resolution, and grouping are selected by
the missing YAML. Do not hard-code the inferred four-stream token count.

The base path binarizes alpha using `alpha > 0`. The pointmap-specific path has a
slightly different mask flow that can preserve fractional alpha. Match the actual
preprocessor selected by `tdfy.val_preprocessor`.

### External pointmaps

`InferencePipelinePointMap` can estimate a pointmap or accept one. An externally
provided pointmap is expected to be a torch HWC tensor already in SAM camera axes
(left, up, forward). It bypasses the internal R3/OpenCV-to-PyTorch3D rotation used for
estimated pointmaps.

If image and pointmap spatial sizes differ, the pointmap is resized with nearest
neighbor. The pointmap preprocessor returns crop/full pointmaps together with a scene
normalization scale and shift. Layout postprocessing uses those values to return pose
to the metric camera frame.

Touch centers/camera points must use this exact same scene normalization when they are
fused with pointmap-conditioned inference. A raw-touch/raw-pointmap frame match is not
enough after preprocessing.

## The two-stage condition call path

There are potentially two condition-processing objects:

1. the optional pipeline-level model loaded from
   `module.condition_embedder.backbone` and stored as
   `self.condition_embedders["ss_condition_embedder"]`;
2. `ss_generator.reverse_fn.backbone.condition_embedder` inside
   `SparseStructureFlowTdfyWrapper`, defaulting to identity only if its config omits
   one.

The pipeline behavior is exact:

1. `get_condition_input` maps keys listed in `ss_condition_input_mapping` to
   positional arguments. The constructor default is `['image']`. All other
   preprocessor fields become keyword arguments.
2. If the optional pipeline-level condition embedder exists, it is called once. Its
   returned token tensor replaces all original args/kwargs: the generator receives
   one positional tensor and no condition kwargs.
3. If it does not exist, raw mapped args/kwargs are passed to the generator.
4. The MoT wrapper always calls its own `condition_embedder` on whatever it receives,
   then passes the resulting tensor as shared context to every transformer block.

Therefore the precise place where raw modalities become fused tokens cannot be named
from source alone. It could be in the optional external object, in the wrapper's
internal object, or split across both. The checkpoint YAML must show the instantiated
classes.

**Integration implication:** add touch to the condition fuser instance that actually
receives raw modality fields. Do not blindly edit both condition embedders or assume
the pipeline-level model is the fuser.

## `EmbedderFuser`

`sam3d_objects/model/backbone/dit/embedder/embedder_fuser.py` is the clearest released
multimodal insertion seam.

### Configuration contract

`embedder_list` contains pairs:

```text
(condition_encoder, [(kwarg_name, position_group), ...])
```

For each named kwarg, the fuser calls `condition_encoder(input_cond)`. If one touch
input requires points, masks, centers, rotations, and metadata, those values need to
be wrapped in one object accepted by the touch encoder, or the encoder/fuser interface
needs a deliberate extension.

The fuser computes:

```text
Dshared = max(encoder.embed_dim for encoder in embedder_list)
```

Each encoder has one shared projection network used for all kwargs assigned to that
encoder. With the default positive projection multiplier, the projection is:

```text
LayerNorm(Dencoder)
-> Llama FeedForward(w1/w3 gated SiLU, then w2)
-> Dshared
```

Relevant parameter paths within an `EmbedderFuser` are structurally:

```text
module_list.<encoder_index>.*
projection_nets.<encoder_index>.0.*           # LayerNorm when enabled
projection_nets.<encoder_index>.1.w1.weight
projection_nets.<encoder_index>.1.w2.weight
projection_nets.<encoder_index>.1.w3.weight
idx_emb                                       # when learned positions are used
```

### Position groups

A named position group maps to one learned or random vector of width `Dshared`. It is
broadcast across every token in that stream. This is a modality/group identity vector,
not per-contact or per-point positional encoding. A touch encoder must create its own
contact/point structure before or within token production if that structure matters.

### Fusion modes

Normal mode (`compression_projection_multiplier == 0`) concatenates along token
length:

```text
[B,L1,D] + [B,L2,D] + ... -> [B,L1+L2+...,D]
```

This is the natural touch-token seam.

Compression mode concatenates along features and projects back to `Dshared`. It
requires equal token lengths across streams and is a poor fit for appending arbitrary
contact tokens unless the whole configuration is redesigned.

The actual released mode is **Open** until YAML inspection.

### Modality dropout

The fuser can sample configured modality-drop patterns per batch element. Dropped
streams are multiplied by zero after encoder projection and position-vector addition;
their tokens are not removed.

This is not equivalent to absence at the attention layer:

- token length remains larger;
- cross-attention `to_kv` may have bias;
- zero tokens still receive softmax probability mass.

For an added touch stream, exact preservation of the original image-only model is not
achieved merely by appending zero touch tokens. Options are discussed in
`04_training_integration.md`.

If `freeze=True` is set on the fuser, it freezes every encoder, projector, and learned
position vector. A touch implementation must avoid globally freezing newly added
modules or re-enable only the intended new parameters explicitly.

## Released image/mask embedders

### DINO

`Dino` resizes inputs bilinearly to its configured image size, applies ImageNet mean
and standard deviation, and returns a CLS token plus patch tokens. Its exact backbone,
patch size, input size, register-token behavior, output layer, and resulting token
count come from config.

For illustration only, an input of 518 with patch size 14 would produce
`1 + 37^2 = 1370` CLS+patch tokens per stream under the straightforward path. This is
not confirmed for the released checkpoint.

`DinoForMasks.forward(image, mask)` ignores the image argument and encodes the mask
after repeating it to three channels. That signature can explain why some fuser
configurations may bind image and mask streams in non-obvious ways; inspect the YAML.

### Pointmap embedder

`PointPatchEmbed`:

1. resizes pointmaps with nearest neighbor;
2. marks finite XYZ locations;
3. remaps XYZ (default remapper includes perspective-like XY and `log1p(Z)` features);
4. projects XYZ and substitutes a learned invalid-point token;
5. divides the image into windows;
6. prepends a CLS token to each window;
7. applies one timm transformer block;
8. returns one CLS token per window with a 2D window position embedding.

Defaults in source are input size 256, patch/window size 8, and embedding width 768,
which would yield 1024 tokens. These are constructor defaults, not confirmed released
settings.

Coordinate resize/remapping is wrapped in `torch.no_grad`, but the point projection,
invalid token, block, and positions are trainable modules unless frozen externally.

## MoT latent representation

`SparseStructureFlowTdfyWrapper` owns a `latent_mapping: ModuleDict` of `Latent`
objects. Each latent object has:

- `input_layer: Linear(in_channels, model_channels)`;
- a fixed or learned positional embedding `[L, model_channels]`;
- `out_layer: Linear(model_channels, in_channels)`.

Shape positional embeddings are generated from a 3D absolute grid; layout positions
can be random fixed or learned depending on config.

`latent_share_transformer` merges configured input latent token sequences under one
transformer modality name. The paper says rotation, translation, and scale share the
layout transformer while shape has its own stream. The source implements arbitrary
configurable groups; inspect the actual merged key before selecting parameters.

At inference, latent allocation is data-driven:

```python
{
    name: (batch, latent.pos_emb.shape[0], latent.input_layer.in_features)
    for name, latent in backbone.latent_mapping.items()
}
```

This is preferable to hard-coded shape/layout sizes in new code.

## One MoT block

Each `MOTModulatedTransformerCrossBlock` performs, for every transformer modality:

1. adaptive-normalized multimodal self-attention;
2. residual addition;
3. affine LayerNorm (`norm2`) followed by cross-attention to the shared condition;
4. residual addition;
5. adaptive-normalized MLP;
6. residual addition.

Cross-attention occurs in every block for every current transformer latent name.
There is no late single injection point.

### Multimodal self-attention directionality

`MOTMultiHeadSelfAttention` defaults to `protect_modality_list=['shape']`:

- protected shape queries attend only shape keys/values;
- non-protected modalities attend one another plus detached shape keys/values;
- gradients from layout queries do not flow back through the appended shape K/V;
- shape does not receive layout information through self-attention.

Every modality still cross-attends the external condition independently. Thus touch
can affect shape directly through shape cross-attention and affect layout through its
own cross-attention; it does not need layout-to-shape self-attention.

### Cross-attention math and shapes

For one latent stream:

```text
queries x: [B, Llatent, Dmodel]
context c: [B, Lcond, Dcond]

q  = to_q(x)          -> [B, Llatent, H, Dh]
kv = to_kv(c)         -> [B, Lcond, 2, H, Dh]
h  = attention(q,k,v) -> [B, Llatent, H, Dh]
out = to_out(h)       -> [B, Llatent, Dmodel]
```

Cross-attention is always full attention. No attention-mask argument is exposed.
The attention backend is selected by lower-level code/environment; exact runtime use
of PyTorch SDPA/Flash depends on hardware and configuration.

### Exact condition-sensitive parameter paths

Within the MoT backbone, the structural paths are:

```text
blocks.<block_index>.norm2.<transformer_latent>.weight
blocks.<block_index>.norm2.<transformer_latent>.bias

blocks.<block_index>.cross_attn.<transformer_latent>.to_q.weight
blocks.<block_index>.cross_attn.<transformer_latent>.to_q.bias
blocks.<block_index>.cross_attn.<transformer_latent>.to_kv.weight
blocks.<block_index>.cross_attn.<transformer_latent>.to_kv.bias
blocks.<block_index>.cross_attn.<transformer_latent>.to_out.weight
blocks.<block_index>.cross_attn.<transformer_latent>.to_out.bias
```

When cross-attention q/k RMS normalization is enabled, add:

```text
blocks.<block_index>.cross_attn.<transformer_latent>.q_rms_norm.gamma
blocks.<block_index>.cross_attn.<transformer_latent>.k_rms_norm.gamma
```

Only `to_kv` is applied directly to condition tokens. `to_q` and `norm2` operate on
latent state; `to_out` remixes the attended result. This distinction should guide
the meaning of “unfreeze cross-attention.”

The exact root prefix when accessed through the full pipeline is likely under
`models.ss_generator.reverse_fn.backbone`, but a trainer should select modules by
object identity/type and verify printed names rather than depend solely on a brittle
string prefix.

### Shared projection across modalities

Image and touch tokens concatenated in the fuser pass through the same latent-specific
`to_kv`. Unfreezing shape `to_kv` changes how the shape stream interprets **all**
condition modalities, not only touch. There is no released modality-specific K/V
adapter at this seam.

A frozen `to_kv` does not make touch impossible: a trainable touch encoder/projector
can learn features that map through the existing K/V basis. Whether that has enough
capacity is empirical.

## Sparse-structure VAE

### Confirmed source behavior

`SparseStructureEncoder` accepts dense occupancy as float, applies a convolutional
encoder, and outputs twice the latent channel count split into posterior mean and
log-variance:

```text
mean, logvar = h.chunk(2, dim=1)
z = mean + exp(0.5 * logvar) * noise    # when sample_posterior=True
z = mean                               # otherwise
```

The wrapper defaults to `sample_posterior=True, return_raw=True`, returning a dict
with `z`, `mean`, and `logvar`.

The official encoder YAML and checkpoint contract are:

```text
input occupancy: [B,1,64,64,64]
latent mean:     [B,8,16,16,16]
flow shape:      [B,4096,8]
channels:        [32,128,512]
resblocks:       2 per level + 2 middle
```

Flattening a deterministic target must invert inference reshape exactly:

```python
shape_target = mean.permute(0, 2, 3, 4, 1).reshape(batch, 4096, 8)
```

`ShapePositionEmbedder` builds a 16-cube with `meshgrid(..., indexing="ij")` and
flattens it in the same order. The token order is therefore `(x,y,z)`, with `z`
varying fastest. Inference exactly inverts it with:

```python
shape_tokens.permute(0, 2, 1).reshape(batch, 8, 16, 16, 16)
```

No Stage 1 shape-latent mean/std standardization constants were found in source.
Stage 2 structured latents do have separate mean/std constants; do not accidentally
apply those to Stage 1.

For a cache, posterior `mean` is preferable to wrapper `z`: sampled `z` introduces new
noise each target-generation pass and makes target files non-deterministic.

### Decoder threshold

Inference reshapes generated shape tokens to `[B,8,16,16,16]`, calls the SS decoder,
and considers voxels active where decoder output is greater than zero. The output is
therefore treated as logits; `> 0` corresponds to sigmoid probability greater than
0.5.

### Occupancy construction and target frame

The target mesh must be voxelized in the renderer's normalized object frame. For
source point `p_S`:

```text
p_O = T_normalized_from_source @ p_S
```

`T_normalized_from_source` already contains source-axis conversion, centering, and
uniform scale. Apply it once to the original OBJ. Do not pass the result through the
stock `inference_utils.voxelize_mesh`, because that function would apply another axis
rotation and may renormalize.

The selected first-cache policy is the fixed-bound surface portion of that helper:

- clips vertices into the cube;
- creates an Open3D triangle-mesh voxel grid with hard-coded voxel size `1/64`;
- marks surface voxels in a `[1,64,64,64]` tensor;
- uses fixed bounds `[-0.5,0.5]^3`.

An Open3D grid index `(ix,iy,iz)` is written directly to occupancy axes `(x,y,z)`.
Its normalized-object center is `(index + 0.5)/64 - 0.5`.

Meta's private training target builder is not released. SAM's main helper and the
released TRELLIS dataset tool agree on fixed-bound surface voxelization. A later SAM
encoder notebook attempts interior filling but then treats Trimesh-local sparse
indices as global 64-cube indices, which can shift the target. Use the fixed-bound
surface policy consistently for the first cache and record it in cache metadata.

The shape latent remains in normalized object frame `O`; it is not transformed into
the camera. Every view of one object uses the same target. For an alignment overlay,
map voxel centers into the pointmap frame with:

```text
T_sam_from_object = diag(-1,-1,1,1) @ T_camera_from_object
p_sam = T_sam_from_object @ p_O
```

## Pose/layout targets

The data tree provides normalized-object-to-camera transforms, but it does not provide
ready model-space R/t/s targets.

Raw normalized object to SAM camera is:

```text
T_sam_from_object = diag(-1,-1,1,1) @ T_camera_from_object
```

The occupancy target uses this same normalized object frame. No additional voxelizer
rotation is permitted.

`PoseTargetConverter` supports multiple representations:

- `ScaleShiftInvariant`;
- `ScaleShiftInvariantWTranslationScale`;
- `DisparitySpace`;
- `ApparentSize`.

The pipeline picks the configured convention from `ss_generator.yaml`. These
conventions combine object pose with scene scale/shift in different ways, and some
decode scale-like outputs through exponentiation. Do not equate raw camera translation
and renderer scale directly with flow latent values.

The inference code also contains six-dimensional rotation normalization constants:

```text
mean = [-0.06366085,  0.00843822,  0.00017085,
         0.00071266, -0.00309167,  0.51660938]
std  = [ 0.66569720,  0.67870123,  0.30345010,
         0.43945044,  0.39817974,  0.61762869]
```

When a normalized 6D representation is actually configured, a rotation matrix's
first two columns form the ordinary 6D input before applying these statistics. Verify
this against the selected decoder and a round-trip test; do not infer it merely from
the paper's dimensionality.

## Flow-matching training loss

The inspected released generator YAML uses:

- `sigma_min = 0` (rectified flow);
- time scale `1000` before the reverse network;
- logit-normal training-time sampler with mean `-1`, standard deviation `1`;
- mean MSE;
- Euler inference solver;
- configurable scalar/tree `loss_weights`.

For target tree `x1`:

```text
t  ~ configured sampler
x0 ~ N(0,I)
xt = [1 - (1 - sigma_min)t] x0 + t x1
v* = x1 - (1 - sigma_min) x0
v  = reverse_fn(xt, 1000 t, condition)
loss = sum_modality weight * MSE(v, v*)
```

At `sigma_min=0`, this reduces to the paper's rectified path and velocity.

The paper reports modality weights shape `1`, rotation `0.1`, translation `1`, and
scale `0.1`, plus AdamW with no weight decay. The inspected released shortcut YAML,
however, currently sets `shape: 0`, rotation `0.1`, translation `1`, scale `0.1`, and
translation-scale `0`. A shape/touch training experiment must deliberately set a
nonzero shape loss, normally `1.0`; copying the released inference YAML unchanged
would provide no shape supervision. The paper's full-model optimization schedule is
not a mandated adapter recipe.

### Shape-only target trap

`FlowMatching.loss` itself can recurse over a dictionary containing only supplied
keys. However, `SparseStructureFlowTdfyWrapper.project_input` iterates every key in
`input_latent_mappings` and asserts that it exists in the latent dictionary.

Therefore this naïve call can fail:

```python
ss_generator.loss({"shape": shape_target}, condition_tokens)
```

if the configured generator also expects rotation/translation/scale latents. The
paper describes partial-modality training, but the orchestration/masking mechanism is
not present in the released inference source.

Viable directions include:

- generate correct targets for all configured Stage 1 modalities and apply selected
  loss weights;
- supply all latents but implement a custom loss mask/weighting strategy;
- create a shape-only wrapper/config initialized from the multimodal checkpoint, with
  careful state transfer;
- recover the original training wrapper from another official artifact if available.

This must be solved before the first real optimization run.

## Inference APIs are not training APIs

`sample_sparse_structure` wraps condition embedding, generator sampling, and decode in
both `torch.no_grad()` and CUDA autocast. It generates random starting noise and runs
the ODE sampler. It is unsuitable for training the touch encoder.

A training step should instead:

1. preprocess inputs with gradient behavior made explicit;
2. compute condition tokens without a surrounding `no_grad` for trainable touch
   modules (frozen image encoders may be no-grad separately);
3. call `ss_generator.loss(...)` or a well-tested custom equivalent;
4. backpropagate only through explicitly selected modules.

The eventual touch-aware inference path can reuse sampling logic after extending its
input dictionary and condition mapping, but it should have an image-only regression
mode that invokes the original path unchanged.

## Runtime shape probes required after checkpoint access

Capture and persist these values from one batch:

| Probe | Required observation |
|---|---|
| preprocessor output | every key, shape, dtype, value range |
| optional external condition input/output | object type and token shape |
| wrapper internal condition input/output | object type and token shape |
| fuser streams | kwarg name, encoder class, raw and projected shape, position group |
| final context | `[B,Lcond,Dcond]` and dtype |
| each `latent_mapping` entry | token count and channel count |
| merged `latent_names` | actual shape/layout transformer keys |
| one shape cross-attention | q shape, kv/context shape, output shape |
| SS encoder | occupancy, mean, logvar shapes |
| SS decoder | input and output shape/range |
| pose converter | target/decode keys and round-trip errors |

These probes should become automated smoke tests or saved audit artifacts, not remain
one-off terminal output.
