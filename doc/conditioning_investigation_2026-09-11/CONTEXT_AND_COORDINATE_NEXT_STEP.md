# Reconciled context and the active coordinate investigation

12 September 2026. This records the paper/code/history review. **Subsequent user steering:** continue from the existing coordinate experiments; the inventory proposal below is optional, not a prerequisite. [PIVOTAL_FINDINGS.md](PIVOTAL_FINDINGS.md) is now the governing evidence ledger and must be consulted/updated before new experiment design. The review remains evidence; the former next-step proposal below is historical, not a queued job.

## Material Passport

- Objective: make Stage-1 full-surface point conditioning useful, retaining VecSetX and a future path to sparse structured touch.
- Scope this round: coordinate handling only; investigate an entangled factor only when needed to interpret a coordinate control.
- Reviewed checkout: `3478ea9b3cbbab6f65d198587195549bc0f54395`; local upstream reference: `f91db411c50efee93d8db7aeb323885650f6f722`. This is comparison with the locally available upstream revision, not a claim of fetching latest upstream.
- Evidence: conversation constraints, completed findings and archived analyses, current training/data/model sources, historical jobs, paper architecture/training/pointmap sections.
- Status: source/context review complete; historical oracle jobs recovered. Actual cluster oracle metadata is needed only if that optional historical comparison is pursued. No model executed locally this round.

## SAM3D paper versus the actual local implementation

The paper describes a geometry model that generates spatial shape and layout, using separate transformer streams. Its shape latent has 4096 locations with eight channels. The architecture supports training with shape-only labels, and layout rotation is defined relative to the generated shape. Its pointmap ablation reports little shape-quality change, with each version preferred 48% of the time. This supports neither an orientation-free shape latent nor an architectural prohibition on pointmaps influencing shape. [SAM 3D, Sections 2.2, C.1–C.2 and E.5](https://arxiv.org/html/2511.16624v1); the published supplement reports the pointmap result in [F.5](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Chen_SAM_3D_3Dfy_CVPR_2026_supplemental.pdf).

The following details are verified in the local source, not inferred solely from the paper:

| Component | Actual contract and implication |
|---|---|
| Shape latent | `tdfy_dit/models/mm_latent.py:ShapePositionEmbedder` supplies fixed 3D grid positions. `dataloader.py:load_target` flattens saved `[8,16,16,16]` means to `[4096,8]` without rotating them. The target is view-independent, not orientation-independent. |
| Shape/layout self-attention | `tdfy_dit/modules/attention/modules.py:MOTMultiHeadSelfAttention` protects shape: shape reads shape only; other modalities read detached shape. `modules/transformer/modulated.py` constructs that default and separately applies condition cross-attention to each stream. |
| Pointmap | `dit/embedder/pointmap.py` encodes xyz in image windows, with image-window positions and invalid-point handling. The fuser makes these condition tokens available to shape cross-attention. Camera-frame pointmaps therefore are not an explicit shape-frame converter, but are not blocked from shape either. |
| Stock pose output | Its relationship to shape does not create a path from predicted layout rotation back into shape. Training layout alone cannot add that missing dependency. The source does not establish that shape cannot infer orientation directly from observations. |
| Stage-1-only adaptation | `train.py` loads the generator and condition encoder, freezes the pretrained model, enables shape CA/norm2 as selected, and gives only shape a nonzero loss. Zero layout targets do not contaminate shape through protected self-attention in this path. Shape-only training is not itself evidence of misuse. |

The upstream-to-current diff inside `sam3d_objects/` is small: the new touch encoder, its vendored VecSetX files, touch-token concatenation in `mot_sparse_structure_flow.py`, and logging changes in `inference_pipeline.py`. The protected self-attention, spatial shape mapping, pointmap encoder and original fuser are unchanged against the recorded upstream revision. The custom training/data-generation code and investigation drivers sit outside that upstream backbone diff.

## Added surface integration and coordinate contract

Use row-vector points. The source mesh is transformed by the saved `T_normalized_from_source` into object points P. Both target voxelization and full-surface sampling use `generate_target_latents.load_normalized_mesh`. The target is `E(voxelize(P))`, encoded once per object. `render_blender.normalize_scene` supplies import-axis conversion plus centering/scaling; it does not implement semantic front canonicalization. Whether asset fronts are already consistent is not established.

`sample_full_surface.py` samples 8192 points once per object and transforms the same points into every view with `diag(-1,-1,1,1) @ T_camera_from_object`: C = P R^T + t. The ordinary dataloader does not expose that R to the conditioner. The oracle dataloader supplies the inverse transform and `train.prepare_batch` applies it before point encoding.

The no-position path used in the principal comparisons is:

1. Camera points C, or oracle-aligned points P.
2. Bbox-center / maximum-radius normalization N. Positive isotropic SSI shift/scale cancels under N; rotation does not. Bbox centering itself is not rotation-equivariant.
3. Pretrained VecSetX `encode()['x']`, 1024 slots × 32 channels.
4. Shared per-token projection to 1024 channels plus a shared touch modality vector.
5. Append these 1024 tokens to the 7528 visual/pointmap tokens; shape queries attend to the combined context.

There is no explicit per-token xyz correspondence to the shape grid and no explicit camera-to-target rotation input. This is a learned geometric interpretation task, not a pointwise copying operation. That structural observation does not establish that VecSetX is deficient or that this interface cannot learn it.

Optional integration variants are separate factors: `use_learn` exposes native processed VecSetX features; `use_position` adds global center/scale; `joint_pointmap` combines visible pointmap points with touch points; `train_vecsetx` unfreezes encode parameters after loading pretrained weights, rather than training from scratch. None should silently change in the coordinate comparison. The separate-attention candidate exists only in the investigation driver and has not replaced production fusion.

For future sparse touch, the current no-position normalization discards absolute cloud location/scale, and the loader merges patches to xyz/mask. Normals and per-point physical channels are not yet represented. These are recorded future requirements, not reasons to replace the full-surface encoder in this round. The mesh/voxel bridge remains withdrawn as a production conditioner.

## Completed evidence: what each step actually narrowed

| Completed step | Finding | Limit / decision |
|---|---|---|
| Camera inverse, reprojection, hidden-surface audit | 64 views / 32 objects agree under inverse transforms to 2.14e-7; reprojection errors below 2.88e-5 px; surfaces include substantial hidden geometry. | Against gross camera-sign/extrinsic corruption in the inspected subset. Does not certify every cluster record. |
| Target-axis and regeneration checks | Saved import conversion already includes stock axis rotation; original targets regenerated exactly in the eight-object GPU check. | Do not double-apply an axis correction. Semantic front compatibility remains distinct. |
| Original checkpoint fixed-state and rollout probes | Surface swaps had weak benefits in the inspected original training examples; removing CFG worsened sampled geometry despite reducing local velocity error. | Surface utility was already inspected on eight training objects; a new probe must add oracle and validation information, not merely repeat it. Guidance-removal recommendation withdrawn. |
| Four-object single-view fitting | Camera and oracle fit strongly with frozen VecSetX and shape CA/projector adaptation. | Rejects universal inability to fit; four identities can be memorized. |
| Fixed-oracle surface, changed visual view | Loss .01887 → .12771 while surface and target remain fixed. | Surface-frame variation is unnecessary for this particular view gap. |
| Four-object multi-view fit | Reserved-view loss camera .08071, oracle .06013. | Frame treatment helps at this scale; does not solve transfer or establish a ceiling. |
| Rotation/normalization factorial | Rotation-only .08074; normalization-only .06427; oracle .06013. | Rotation reproduces most of this controlled penalty. Strongest coordinate-specific causal result; one seed/four objects/finite budget. |
| Pose-input audit | No explicit rotation in ordinary inputs; layout cannot feed shape; near-constant camera translation does not reveal varying rotation. | Known inverse is available diagnostically. No deployable pose estimate or actual observation/target ambiguity proven. |
| Frozen-prior target orientation probe | Six discrete orientations give no convincing transferable correction; selected held flow gain .344%, no sampled-shape corroboration. | Against tested global/relabeling fixes; not proof that all target conventions or continuous alignment are equivalent. |
| VecSetX native reconstruction and readouts | Native geometry is retained; learned readouts fit 24 identities but fail on eight others. | Retain VecSetX. This detour did not isolate the camera-frame penalty. Stop readout extensions. |
| Frozen native-mesh bridge | A complete-shape conversion accesses useful geometry on unseen identities. | Diagnostic evidence only; changes too much and bypasses intended sparse conditioning. Production integration withdrawn. |
| Oracle visual-stream factorial | Pointmap-only view changes are small (.01795 → .01917); RGB/mask changes dominate the remaining oracle gap. | No support for pointmap-frame conflict as the principal cause of that measured residual. Do not repeat this test. |
| Visual-dropout matched training | Oracle reserved-view loss .05785 → .02273 with full inputs present. | Repairs the fitted identities' view gap without encoder replacement. |
| 32 unseen identities | All five four-object-trained surface variants lose to image on all 32 objects. | Dropout/alignment was not a general conditioning solution. |
| 16-object matched training + early/late swaps | Real camera/oracle improve fit; oracle held loss .13420 versus image .11930. Wrong surfaces preserve much of early benefit. | Correct geometry not reliably useful on these new objects. Not a diagnosis of full-data training yet. |
| Constant-surface training | A single fixed training surface reproduces fitting gains and later transfer failure. | Extra learned pathway is a sufficient contributor to this small-task pattern; not proof that all geometry is ignored. |
| Separate surface attention, image warm start | Stronger fitting, but held oracle .13854 vs image .11930; removed branch exactly restores image. | Preserving image weights and separating softmax is insufficient at this scale. Candidate rejected; do not extend. |

These are validated returned GPU reports, not independent local GPU reruns. Detailed provenance and scope remain in their respective findings documents. Native flow MSE, direct target-regression MSE and sampled geometry are different measurements; do not compare their absolute values as one common error floor. Reserved views, reserved identities, and dataset-validation identities are different splits.

## New finding: the original oracle jobs are recoverable

Commit `8828d933ca4b11fbfb34a6f78631c716e72190be` added the oracle jobs. Commit `47ce9d4f60e24346d606876f016574b0e8e3e7a4` removed them. Exact copies and hashes are in `recovered_oracle_jobs/`; they are archived text, not queued jobs.

The recorded outputs are:

- Camera: `outputs/stage1_full_surface_full_cross_attention`
- Oracle: `outputs/stage1_full_surface_oracle_full_cross_attention`
- Camera without pointmap: `outputs/stage1_full_surface_no_pointmap_full_cross_attention`
- Oracle without pointmap: `outputs/stage1_full_surface_oracle_no_pointmap_full_cross_attention`

Camera/oracle job definitions specify the same full-surface config, pretrained pipeline, four GPUs, global batch 16, 20 epochs, no position embedding and full shape cross-attention scope. Oracle adds `--oracle-point-frame`. Source at that commit applies the inverse during training and validation and saves the flag in `conditioning_config`.

This establishes intended job comparability and historical paths, not completion or actual checkpoint equivalence. Local exports cover 21 runs and omit oracle; local output configs identify camera and image, but oracle configs/checkpoints are not available here. The user's report that oracle training already ran is accepted. Checking only current jobs earlier missed useful Git history.

### Local checkpoint metadata follow-up

Direct filesystem inspection also found the camera and image best/last checkpoint files locally. Their saved metadata was read without importing torch or loading tensor storage, using a restricted reader of the ZIP `data.pkl` record. Only OrderedDict, a FloatStorage marker and a tensor-metadata stub were allowed. Results and metadata hashes are archived in `local_checkpoint_metadata.json`; these hashes are not weight-content hashes. Cross-check with the ordinary PyTorch inventory when returned.

- Camera best: epoch 9 / step 6597; image best: epoch 12 / step 8796, matching the W&B references.
- Both last checkpoints: epoch 20 / step 14660. Thus equal-step final camera/image references are available; the oracle last step remains unknown.
- Camera records frozen VecSetX, no position and full CA; optimizer rates are 1e-4 for touch adaptation and 1e-5 for shape CA. Image records full CA at 1e-5.
- These older checkpoints omit `conditioning_config`. Missing metadata is not an explicitly saved false flag; interpret it with the historical source/config, rather than claiming the checkpoint alone proves the frame.
- `best_loss` inside `last.pt` is the best-so-far loss, not the loss measured at the last checkpoint. No new performance measurement follows from this inventory.

Oracle config/checkpoints and the requested cluster inventory are still absent locally. No actual inference process is known to be running; this is an external-input dependency, not a live-job wait.

## Historical full-dataset proposal — optional, not the active prerequisite

**Question:** Under comparable full-data training, does exact pre-encoder alignment enable useful object-specific surface conditioning that ordinary camera coordinates do not?

**Immediate step:** run `inventory_coordinate_checkpoints.py` on the cluster CPU. It reads saved configs and best/last checkpoint metadata, including recorded frame flags, trainable parameter schema/count, optimizer groups, epoch and step. It performs no model construction, CUDA operation, training, or decoder work. Missing paths/fields stay explicit. It does not replace unknown metadata with guessed defaults or certify historical base weights/data from matching filenames.

Review actual configs before enabling the comparison: data/splits, normalization and feature mode, trainable scope, seed, global batch, optimizer settings, exposure, resume history and available same-step checkpoints. Prefer matching completed last checkpoints for the coordinate comparison when their metadata supports that choice; best checkpoints selected at different steps are descriptive evidence, not a one-factor causal estimate. The historical code saves only trainable weights, so base checkpoint provenance remains a separate requirement.

After metadata supports a pair, adapt the no-training Stage-1 utility probe for camera AND oracle at their own trained frames, on identical current train/validation observations and shared native noise/time draws. Check real-versus-three-wrong surface conditions; retain removal as an OOD presence control, and image as a separate utility baseline. Report per-object effects and uncertainty, not only a pooled mean. Do not test an untrained frame switch at inference and call its failure evidence against alignment. Do not call the already examined object pool an untouched test set.

Decision branches, fixed before those results:

- Oracle shows a material, consistent correct-surface benefit on validation and camera does not: pursue the isolated rotation-handling branch with VecSetX retained. Confirm improvement over image as well as wrong surfaces before calling aligned conditioning a useful reference. Pose estimation is then a candidate with a meaningful endpoint, not an assumed fix.
- Both show useful benefit: coordinates do not explain an absence of all conditioning utility; quantify the remaining frame penalty before proposing a correction.
- Neither shows useful benefit, despite fitting: exact alignment alone is insufficient for these broader-data models. Close the simple pre-encoder-alignment remedy at that scope; discuss the aligned-conditioning failure as a separate, necessary next aspect. Do not declare all coordinate architectures impossible.
- Results are small, heterogeneous, noisy, or configs/exposures differ materially: remain inconclusive. Identify the specific missing control before proposing one short matched experiment. No automatic architecture or full-training run.

The no-pointmap pair is inventoried because its historical jobs already exist. It is not automatically added to the GPU matrix: use it only if the primary pair or configuration differences make pointmap entanglement relevant. The previous four-object visual-stream test already weakened that explanation in its own setting.

The earlier image/camera-only `FULL_DATASET_CONDITIONING_HANDOFF.md` is paused pending this coordinate-focused pair selection. No new full-data training, pose head, target re-encoding, encoder replacement, separate-attention extension or Stage-2 work is queued.

## Optional historical inventory command — not currently requested

Sync the new inventory script, then paste this on the cluster. A GPU allocation is not required; use a node where the existing environment and files are available.

```bash
(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
inventory_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
inventory_root=outputs/conditioning_investigation/coordinate_checkpoint_inventory/manual
"$inventory_python" doc/conditioning_investigation_2026-09-11/inventory_coordinate_checkpoints.py \
  --output "$inventory_root/inventory.json"
)
```

Return `inventory.json`. If directories moved, the script accepts `--run-dir oracle=/actual/path` (and the other keys printed in the script). Missing checkpoints are reported rather than triggering retraining. Local syntax and synthetic metadata/config-difference checks pass; real PyTorch checkpoint loading is pending on the cluster.
