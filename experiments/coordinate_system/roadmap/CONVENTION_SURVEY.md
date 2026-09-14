# Coordinate conventions across the reference methods

2026-09-14. This extends C2–C8 of [the existing checklist](COORDINATE_CHECKLIST.md), rather than restarting its numerical audits. Source inspection and mathematical interpretation; no new GPU experiment or training change.

## Result

**There is no universal requirement that image-to-3D models recover an asset's uniquely correct, hidden orientation.** There are two different tasks:

1. Generate a plausible object, optionally with a separately predicted transform that places it in the image/scene.
2. Reconstruct geometry in a coordinate system explicitly supplied by measured 3D inputs.

The second is the appropriate full-surface upper-bound task. Our oracle supplies the surface in the target's orientation. It does not require recovering that orientation from RGB. However, it does not make all conditioning streams share numerical coordinates: the pointmap remains in normalized camera space, and VecSetX applies another surface normalization.

The sweep found concrete differences in how these boundaries are handled. It did **not** identify a new sign/axis error in our executed oracle transform, or prove why the full-data upper bound remains inaccurate.

## 1. What the papers and released code establish

### SAM 3D: shape and placement are separate predictions

The paper defines a conditional distribution over shape and layout, rather than a deterministic inverse of the photograph. Its MoT permits layout to depend on shape; the supplementary text explicitly links meaningful rotation to the generated shape. Human layout annotation places candidate meshes in the scene. Pointmap conditioning has little reported effect on shape preference. The release does not expose the complete pretraining asset-preparation pipeline or establish a semantic-front annotation for every asset. [Paper §§2.1–2.2, supplementary A.4, C.1, E.5](https://arxiv.org/html/2511.16624v1).

The released code provides more precise, narrower witnesses:

- `normalize_mesh_verts` centers the AABB and makes its longest extent one. `voxelize_mesh` applies an explicit matrix taking `(x,y,z)` to `(x,-z,y)`, then voxelizes the normalized geometry. Follow this matrix and the input format, not the adjacent verbal up-axis comment alone. This helper is not proof of the entire original training pipeline. [Source](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/pipeline/inference_utils.py#L721).
- `ObjectCentricSSI` subtracts the masked object's median XYZ; with `use_scene_scale=True`, its scalar comes from the scene's centered coordinates. It retains camera-axis directions. **The pointmap is scale-normalized, but not by the same rule as the full surface or target.** [Normalizer](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py#L519). The locally installed `checkpoints/hf/pipeline.yaml` selects this mode.

Mathematically, ignoring scale for this example, a scene object is `R S + t`. Rotating the represented shape to `U S` and replacing its placement by `R U^-1` leaves the scene geometry unchanged. This does not imply arbitrary training-label rotations are harmless. It explains why correct scene reconstruction does not require recovering an unknowable asset-authoring convention. In our protected shape path, predicted layout does not feed back to rotate the surface context.

### TRELLIS: stored asset orientation, multiple conditioning views

The released pipeline gives a direct training witness:

1. The Blender renderer imports an asset, centers/scales it, renders cameras around it, and exports the mesh. `normalize_scene` does not identify semantic front or fit a ground plane. Camera orbit height is the Blender scene's Z coordinate. [Renderer](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/dataset_toolkits/blender_script/render.py#L366).
2. Voxelization consumes that exported mesh in a fixed `[-.5,.5]^3` grid. [Voxelizer](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/dataset_toolkits/voxelize.py#L14).
3. The image-conditioned dataset selects a random rendered view but retains the object's saved target latent. It uses the selected image, not its extrinsic matrix, as the condition. There is no per-view target rotation in these loaders. [Image loader](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/datasets/components.py#L92), [latent loader](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/datasets/sparse_structure_latent.py#L180).

The supplementary rendering description distinguishes VAE views from the augmented-FoV images used as generation conditions. Cameras used to construct a representation are not thereby supplied to the image-conditioned generator. [Supplement B.4](https://arxiv.org/html/2412.01506v2).

**Implication:** our fixed object target with multiple image views is not, by itself, an unusual labeling mistake. A stochastic model can learn a distribution of compatible object orientations; exact asset-axis recovery from an ambiguous image is not guaranteed.

### Axolotl3D: explicitly prepared geometry and posed images

The paper normalizes meshes before sampling surface conditions and renders, uses Hunyuan3D 2.1 targets, and adds camera Plücker embeddings. Supplement C.3 explicitly names `+Y up, +Z forward`: ground-plane alignment plus horizontal PCA/SVD, or Orient Anything with possible manual correction. C.4 manually orients captured points and reverses that transform on outputs; that application omits camera conditioning. It does not document enough to reconstruct every training asset's semantic-front assignment, exact VecSetX checkpoint, or any extra per-partial-cloud normalization. [Methods and supplementary C.3–C.4](https://arxiv.org/html/2607.20660v1).

No author model/training implementation was verified through the project page and repository search. Paper-level preprocessing must not be described as code-verified. [Project](https://research.nvidia.com/labs/sil/projects/axolotl3d/).

The [Kaolin preview documentation](https://kaolin.readthedocs.io/en/web_framework_prerelease/modules/kaolin.visualize.dash.html) also mentions an Axolotl `mesh_edit` app. Both public Kaolin source trees checked (`master` and `web_framework_prerelease`) lacked that path or an Axolotl-named implementation. This lead therefore did not expose the missing model preprocessing; it is not evidence that the paper's preprocessing was absent.

### ShapeR: a shared measured frame, not an unknown CAD orientation

ShapeR generates from posed capture images and SLAM points. Its paper normalizes object points before generation and restores metric scale afterward. The point encoder is a sparse 3D ResNet; Dora/VecSet is its **target** representation, not our frozen VecSetX conditioning encoder. [§§3.1–3.3 and supplement C](https://arxiv.org/html/2601.11514v1).

The release makes the coordinate contract unusually explicit:

- The data notebook defines `points_model` in object coordinates, `bounds` as box half-extents, and `T_model_world` as world-to-model. The name is not camera-to-world. [Data format](https://github.com/facebookresearch/ShapeR/blob/d4402f55dc698104c69aaa7b5dc5e2c8e03bacd6/explore_data.ipynb).
- The loader computes `a = 0.9 / max(bounds)` and applies it to both points and optional reference mesh vertices. Output restoration divides by the same `a`, then optionally applies the inverse saved frame transform. No independent point-only recentering occurs there. [Loader](https://github.com/facebookresearch/ShapeR/blob/d4402f55dc698104c69aaa7b5dc5e2c8e03bacd6/dataset/shaper_dataset.py#L51).
- Camera-to-model translations receive that same scale. Cropping/rectification updates camera parameters, and rays are constructed from the resulting cameras. [Image processing](https://github.com/facebookresearch/ShapeR/blob/d4402f55dc698104c69aaa7b5dc5e2c8e03bacd6/dataset/image_processor.py#L185), [ray construction](https://github.com/facebookresearch/ShapeR/blob/d4402f55dc698104c69aaa7b5dc5e2c8e03bacd6/preprocessing/ray_utils.py#L11).
- The **experimental** Depth Anything workaround explicitly establishes Z-up using a ground-plane estimate, centers points, and applies the same transforms to cameras. It does not estimate semantic front. This is evidence of the expected inference convention, not proof of original training augmentation. [Workaround](https://github.com/facebookresearch/ShapeR/blob/d4402f55dc698104c69aaa7b5dc5e2c8e03bacd6/experimental/workaround_dataproc.py#L490).

The original object/scene training-data construction is not fully released in this tree; inference preprocessing cannot certify its undisclosed training details.

### Orientation Matters: inconsistent semantic orientation is a documented problem

This work identifies inconsistent canonical orientations in existing generators. It constructs Objaverse-OA with VLM assistance and manual correction, including ambiguous fronts and non-upright objects. It fine-tunes TRELLIS's sparse-structure generator to obtain consistent semantic orientation. This is direct precedent for treating semantic alignment as a data/prior issue, distinct from a wrong inverse camera matrix. It is **not** evidence that our dataset needs that treatment or that oracle-aligned surface conditioning requires it. [Paper §§3–4 and appendix](https://arxiv.org/html/2506.08640v1).

The released repository supplies the dataset/checkpoint references. Its included TRELLIS renderer/loader alone does not reproduce the complete human/VLM annotation procedure. [Release](https://github.com/YichongLu/Orientation_Matters/tree/9f13691127610640ea31519226ed2aeae3fe80ce).

### VecSetX and related conditioning references

**VecSetX:** the official inference convention is AABB centering followed by maximum-radius normalization; our `TouchEncoder` follows it. The released training entry selects an Objaverse loader. That loader applies the **same** random axis permutations, signs, and rotations to surface points and SDF query coordinates. The sign/permutation combinations include reflections; this is not merely yaw augmentation or a proof of strict SO(3) equivariance. [Normalization instructions](https://github.com/1zb/VecSetX/blob/8efecd0b5d7135f47915e69d62ef4e9bae4fb787/README.md), [training entry](https://github.com/1zb/VecSetX/blob/8efecd0b5d7135f47915e69d62ef4e9bae4fb787/vecset/main_ae.py#L117), [paired augmentation](https://github.com/1zb/VecSetX/blob/8efecd0b5d7135f47915e69d62ef4e9bae4fb787/vecset/utils/objaverse.py#L132).

**Caveat:** the exact provenance of our downloaded checkpoint-125 is not established by this current source tree. In particular, Axolotl's description of its encoder's training corpus must not be copied onto our checkpoint. Nevertheless, “VecSetX only accepts one canonical upright orientation” is not supported by its released training recipe.

**3DShape2VecSet:** its available surface/occupancy loader transforms surface points and query positions with the same sampled scaling and normalization, while adding jitter only to observed surface points. That is a useful precedent for preserving the input/target coordinate relation. It does not prove arbitrary sparse-touch conditioning works. [Transform](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/util/datasets.py#L5).

**Hunyuan3D-Omni:** the released point-control example centers/scales the supplied point geometry to maximum extent 1.96, then passes XYZ directly as control. Its voxel example separately applies a -90° X rotation to its example asset. That branch-specific conversion is not a universal rotation to copy into our point path. The original training preprocessing is not fully exposed by these inference examples. [Point and voxel entry points](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni/blob/4d47c0cc2bd0c4281963a7314ab330a5af36bfa8/inference.py#L255). The paper constructs control signals in the shape's normalized space, including incomplete/noisy point conditions. [§3.2](https://arxiv.org/html/2509.21245v1).

## 2. Exact comparison with our implementation

| Boundary | What our code does | Conclusion |
|---|---|---|
| Asset → target | OBJ importer-axis conversion; AABB center/longest-side normalization; saved transform reused for target mesh | Establishes numeric axes. Does not discover semantic up/front. |
| Camera surface → oracle surface | Full inverse of the recorded object-to-stored-camera transform | Removes camera pose on audited records. No missing per-view rotation needs guessing from RGB afterward. |
| Oracle surface → VecSetX | Recenter sampled surface and normalize maximum radius | Preserves orientation, changes units and possibly center relative to the mesh-based target. Follows VecSetX, not a shared-coordinate contract across all streams. |
| Image/pointmap → visual context | Cropped/full image/mask tokens; camera-frame ObjectCentricSSI pointmap | Oracle does not transform this branch. Unlike ShapeR, there are no explicit camera rays in the surface/target frame. |
| Latent → shape grid | Spatially indexed SS latent | A shape latent is not orientation-free merely because it was encoded in 3D. |

Local witnesses: [renderer](../../../data_generation/objaverse-dexonomy/render_blender.py#L95), [target construction](../../../data_generation/objaverse-dexonomy/generate_target_latents.py#L73), [oracle matrix loader](../../../dataloader.py#L166), [batch preparation](../../../train.py#L330), [VecSetX normalization](../../../sam3d_objects/model/backbone/dit/embedder/touch.py#L169). Current main-repository revision: `650faef85f59d4d7e17e46ddc17e411a975b12ff`. Source inspection retains the previous independent numerical audit as a separate witness.

Changing an XYZ pointmap's reference frame does **not** inherently break its pixel grid. For each pixel, transform its stored XYZ value; the pixel still denotes the same observed surface location. If projecting geometry again, transform the camera consistently too. For `p'=a R p+b`, a camera origin changes to `o'=a R o+b` and its ray direction to `d'=R d`. This is a mathematical contract, not a claim that an existing pretrained pointmap embedder accepts the changed value distribution without adaptation.

## 3. What this changes in the diagnosis

1. **Do not equate “canonical” with a semantically standardized front.** An object-local coordinate system can simply be the asset's stored axes. Our importer does not annotate semantic front; neither does the inspected ordinary TRELLIS normalizer.
2. **Do not claim the oracle still requires an unknowable rotation.** The oriented full surface supplies the target orientation. Learning to use its encoded features remains necessary, but the input rotation is not missing information.
3. **Keep two remaining questions separate:** cross-modal shared-frame fusion, and adaptation to the target/surface normalization and pretrained distributions. Published methods supply concrete shared-frame examples; they do not establish that our mixed-frame fusion caused the observed failure.
4. **Do not start another full run from this survey alone.** The existing no-pointmap and permanent-no-visual checkpoints already remove the respective visual conflicts. E15 should assess those checkpoints before proposing an additional fusion treatment. Poor reconstruction without visual information would exclude visual/surface disagreement as a necessary explanation for that residual; it would not certify the remaining feature/normalization interface.
5. **No reason to repeat the same sign/axis bank.** Reopen it only for a concrete discrepant record or changed preprocessing. A semantic-front dataset intervention is a different C8 experiment, with changed targets and controls, not another inverse-matrix test.

This survey supplies a clearer explanation of how the reference methods train. It does not establish a successful oracle upper bound or close every coordinate-dependent learning hypothesis.

## 4. Coverage and reproducibility

Inspected coordinate-relevant main/supplementary material for SAM3D, Axolotl3D, ShapeR, TRELLIS, Orientation Matters, and Hunyuan3D-Omni; corresponding available source paths above, plus original 3DShape2VecSet and VecSetX. Scope is training/conditioning conventions, not a claim to have executed these training systems or inspected every unrelated UI/renderer dependency. SAM3D and ShapeR original full training-data pipelines, Axolotl model implementation, and exact VecSetX checkpoint provenance remain release/evidence limitations.

Public paper pages and selected source snapshots are outside the repository under `../../../../coordinate_system_provenance/convention_survey_20260914/` (about 5 MB before final metadata). No datasets, model weights, or diagnostic result ZIPs were downloaded. Repository trees record exact remote SHAs; existing local SAM3D/TRELLIS source revisions are linked above. Future diagnostic outputs retain the convention in [STORAGE.md](../STORAGE.md).
