# Weekly progress: SAM3D point conditioning

**Period:** September 1–7, 2026, America/New_York, with a September 8 update below.
**Prepared for:** weekly research meeting.  
**Scope:** factual record of implementation, experiments, recorded outcomes, and pending work. No recommendations are included.

## Summary

The September 8 addendum supersedes the earlier pending-work statuses where indicated. Earlier numerical results remain unchanged.

- Replaced the custom, randomly initialized contact-cluster encoder with a pretrained VecSetX encoder operating on a global point cloud.
- Reworked the dataloader and Stage-1 trainer around that representation, retaining native image/mask/pointmap conditioning and distributed training.
- Added masked point preparation, a SAM3D-style projection and modality embedding, optional normalization metadata embedding, component-level gradient logging, and diagnostics.
- Ran comparisons involving frozen/trainable VecSetX, touch-only point clouds, joint visible-pointmap plus touch clouds, full-surface clouds, and K/V-only versus full shape cross-attention training. “Touch-only point cloud” describes the extra encoder input; these runs still condition on images and native pointmaps.
- Generated a full-surface data variant without rerendering images or changing object splits. Added standalone VecSetX reconstruction and latent-space experiment scripts.
- The three locally archived full-cross-attention runs reached similar validation losses. The image/mask/pointmap baseline had the lowest best validation loss among those three.
- Added a frozen intermediate-decoder-feature option, `--vecsetx-learn`; its training was reported running at report preparation. Added no-pointmap and privileged object-frame options, verified locally at the plumbing level; no training results for these options were available.

## 1. Starting state

The starting point was the handcrafted encoder retained in Git immediately before VecSetX integration. Contacts were kept as separate patches/clusters rather than treated as one unstructured global input.

The previous encoder used absolute and relative coordinate MLPs, continuous local positional encoding, point-transformer blocks within contacts, and contact-transformer blocks. Its defaults were width 128, four point blocks, two contact blocks, and eight heads. It supported a center-token version and query-pooling variants with multiple tokens per contact. The associated setup selected 32 contacts and up to 256 points per contact; earlier jobs compared one, eight, and 32 tokens per contact.

The September 2 repository-review chat documents that starting implementation. The report uses it as September 1 baseline context, not as a claim that it was newly implemented during this week. The historical encoder is now replaced; its authoritative historical version is `git show 80d5632^:sam3d_objects/model/backbone/dit/embedder/touch.py`.

## 2. Implementation chronology

Dates below are commit dates, not necessarily the exact time work began or cluster jobs ran.

| Date | Recorded change | Git evidence |
|---|---|---|
| Sep 2 | Imported VecSetX autoencoder, bottleneck, and attention utilities; replaced custom touch encoder with a wrapper | `80d5632`, `915d542`, `b8e6914` |
| Sep 2 | Replaced torch-cluster sampling dependency with PyTorch3D; added masks and point-count handling | `33fda22`, `54c3508`, `7e659a4` |
| Sep 2 | Added native Llama3 FFN projection and learned touch modality embedding; configured pretrained checkpoint loading | `4179fc8`, `52f8873` |
| Sep 2–3 | Retired earlier job scripts; rewrote dataloader/trainer; added image and touch jobs and explicit defaults | `6d0f42c`, `f6ec78d`, `aaafbfe` |
| Sep 3 | Added trainable-VecSetX job; diagnostics and a single-GPU diagnostics job | `463d4ce`, `aadf4fa`, `e880899`, `8adf4be` |
| Sep 3 | Added global normalization-metadata embedding and component gradient logging | `8265709` |
| Sep 4 | Added joint object-masked pointmap plus touch input and optional position embedding; introduced logarithmic scale | `3eff360`, `e7c5224`, `2cf7b2e`, `f54090a` |
| Sep 4 | Brought data-generation scripts into this repository; added full-surface sampling and loading | `acd4838`, `8a5b435`, `f6f4866` |
| Sep 4 | Added full-surface job and full shape cross-attention training scope | `f694511`, `907e4fd` |
| Sep 4 | Added positive-radius safeguard and zero-initialized position output; position/full-CA and image/full-CA jobs | `43af99a`, `a05f1ee`, `35a468d` |
| Sep 6–7 | Added standalone reconstruction/latent experiments, viewer, orbit rendering, and visualization refinements | `de3ed51` through `3fc8be1` |
| Sep 7 | Added frozen VecSetX `learn()` feature variant and full-surface job | `daba25d` |
| Sep 7 | Added no-pointmap and oracle-frame options, checkpoint/diagnostics integration, CPU tests | Present as uncommitted changes at report preparation |

## 3. Current encoder and training setup

### Point representation and fusion

The selected pretrained model is `learnable_vec1024x32_dim1024_depth24_nb`, using the SDF/NB checkpoint `checkpoint-125.pth` from Zbalpha/VecSetX.

- Contact points are assembled into one global point cloud per sample.
- VecSetX receives 8,192 points after existing preparation. FPS handles excess points; shorter valid clouds use padding with attention masks rather than repeated observations.
- Normalization subtracts the valid-point bounding-box midpoint and divides by the maximum Euclidean radius. This is isotropic scaling.
- The usual feature path returns 1,024 tokens of width 32 from `encode()`.
- LayerNorm and the existing SAM3D Llama3 gated FFN project these to width 1,024. A learned touch-group embedding is added, and tokens are concatenated to native conditioning.
- The optional position branch embeds the normalization shift and scale; the current scale representation is logarithmic. Its final projection is zero-initialized. Historical runs differ in these details, so current defaults should not be applied retroactively to all earlier runs.

Sources: [touch.py](../sam3d_objects/model/backbone/dit/embedder/touch.py), [VecSetX autoencoder](../sam3d_objects/model/backbone/dit/embedder/vecsetx/autoencoder.py), [training code](../train.py).

### Trainable components

The trainer optimizes the Stage-1 shape flow-matching objective. Other output streams receive zero loss weight; this is not joint shape/layout fine-tuning. The shape target is the frozen occupancy encoder's posterior mean, stored as `[8,16,16,16]` and loaded as `[4096,8]` tokens.

- **K/V scope:** train shape cross-attention K/V projections plus the enabled touch adapter parameters.
- **Full cross-attention scope:** train shape query, K/V, output projections and cross-attention input normalization, plus the enabled touch adapter parameters.
- **Trainable VecSetX:** unfreeze the encoder-side parameter prefixes used by `encode()`; this did not mean training its 24 decoder-side transformer blocks.
- Shape self-attention/MLPs and the rest of the pretrained backbone remain frozen under these scopes.

The three archived full-CA runs use four GPUs, batch size four per GPU (16 global), bf16, 20 epochs, seed 29, eight train workers and two validation workers per GPU. Adapter learning rate is 1e-4; cross-attention learning rate is 1e-5; gradient clipping threshold is 1.0. Their final step is 14,660. Submitted job templates request four H200s, 64 CPU cores, 256 GB RAM, and 12 hours. Job requests do not independently certify which hardware each completed run received.

## 4. Training comparisons and results

### Earlier comparisons: recorded in conversation

The chat contains completed-training reports for image/mask/pointmap versus added touch conditioning, frozen versus trainable VecSetX, joint pointmap-plus-touch encoder input, and position-embedding variants. The user reported closely similar loss trajectories across many of these variants. Position-enabled runs showed gradient spikes in shared charts, and were reported worse at some checkpoints. Full cross-attention initially reached lower validation loss faster than the earlier K/V-only runs.

Complete local numerical histories for all those earlier variants were not available. Consequently, this report does not assign exact final scores or effect sizes to them. A job file alone is not evidence that a run completed.

### Archived matched full-cross-attention runs

These values come from the transferred W&B records and the existing checkpoint audit. Lower validation loss is better. The metric is the training-style shape flow-matching validation objective, not a decoded geometric accuracy metric.

| Run | Extra geometric conditioning | Best validation loss | Best epoch / step | Final validation loss |
|---|---|---:|---|---:|
| `stage1_image_full_cross_attention` | None; image/mask/native pointmap remain | 0.088578049 | 12 / 8,796 | 0.089097818 |
| `stage1_full_surface_full_cross_attention` | Frozen VecSetX full surface, no position | 0.088858707 | 9 / 6,597 | 0.089410126 |
| `stage1_full_surface_position_full_cross_attention` | Frozen VecSetX full surface, log-scale position | 0.088965676 | 16 / 11,728 | 0.089191866 |

Relative to the image/mask/pointmap baseline, best loss is higher by approximately 0.000281 without position and 0.000388 with position. The position run has a lower final loss than the no-position full-surface run, but a higher best loss. These are single-run comparisons, not multi-seed significance estimates.

Saved trainable-parameter counts are 100,810,752; 103,875,648; and 106,781,760 respectively. The archived runs save no trainable VecSetX tensors, consistent with their frozen encoder configuration. The audit records parameter changes in full-CA Q/KV/output/input-normalization components, and in enabled touch adapter components.

Sources: [run audit](../outputs/diagnostics/run_audit.json); corresponding `outputs/<run>/config.yaml`, `best.pt`, `last.pt`, and W&B `files/wandb-summary.json`. Final W&B summaries were read directly and agree with the audit's final validation values.

## 5. Full-surface data variant

Data generation was moved under `data_generation/objaverse-dexonomy`. The new script samples 8,192 points directly from the mesh surface with area-weighted sampling. It reuses the normalized mesh loading and saved camera transforms, attaches per-view visibility labels, and retains the existing rendered samples and object split assignments.

The first version attempted to regenerate the historical touch pool, which encountered pool-index mismatches and touch-specific mesh-fragmentation checks. It was revised to direct full-surface sampling, without requiring the historical geodesic pool to reproduce exactly. The new full-surface sample IDs identify the new sample, not the historical touch pool.

The point source is selectable through `configs/data_full_surface.yaml`. This changes extra point conditioning rather than rerendering images or constructing new target latents. [Sampling script](../data_generation/objaverse-dexonomy/sample_full_surface.py), [data configuration](../configs/data_full_surface.yaml).

## 6. Standalone VecSetX experiments

Added `experiments/vecsetx/reconstruct.py`, `latent_space.py`, a runner, viewer, and orbit visualization. Reconstruction uses the existing encoder/preparation and matching pretrained autoencoder decoding functions, comparing full-surface, touch, and joint inputs. Visualization additions include textured references, input/reconstruction comparisons, and error coloring; representative objects were selected at reconstruction-error percentiles in the separate experiment chat.

A recovered results attachment reports **95 validation objects, 378 views**, at up to four views/object, reconstruction resolution 128, 32,768 metric sample points, seed 29, and bf16. Each of the three input sources produced 378/378 successful meshes.

| Encoder input | Reported Chamfer L2 | Supplied points → reconstruction mean distance | Reconstruction → reference mean distance |
|---|---:|---:|---:|
| Full surface | 0.000386035 | 0.008509439 | 0.012749239 |
| Touch | 0.010075749 | 0.022576174 | 0.013186537 |
| Joint | 0.004955314 | 0.019362321 | 0.017297651 |

These are historical exported metrics, matching the initial reconstruction script's schema (`de3ed51`), not the later revised metric names in the current script. Each input's normalization shift/scale was applied to its full-surface reference. Thus distances are in input-normalized VecSetX units, not meters or percentages, and different input sources can have different scales. Distances and squared Chamfer values are different quantities. Partial-input reconstructions are evaluated against the full reference as well as supplied points; the metrics should not be read as equivalent reconstruction tasks or common physical-unit errors. Full-surface masked/unmasked outputs were also compared; the exported bottleneck relative-L2 difference was 0.001650021, not exact numerical equality.

The user reported qualitatively reasonable reconstruction across full, touch, and joint inputs. This is evidence about the matching pretrained autoencoder reconstruction. It does not directly measure how much the SAM3D conditioning projection preserves or how effectively SAM3D uses those features.

Source: [reconstruction-results attachment](/Users/michaelwakeham/.codex/attachments/1f289849-4c7b-4635-950f-9269feaaf6e0/pasted-text.txt), recovered through the separate `Integrate VecSetX touch encoder (2)` experiment conversation. The presence of a latent-space script is recorded separately; this report does not claim a completed latent-space result that was not recovered.

## 7. Diagnostics and coordinate checks

Diagnostics now inspect parameter changes, component gradients, conditioning activations/attention, representation statistics, normalization metadata, and raw/canonical/VecSetX coordinate checks. W&B logging was expanded to include component gradient norms and optimization diagnostics. Diagnostic backward passes do not reconstruct historical training gradients; they evaluate gradients on the selected checkpoint and examples.

The locally transferred diagnostic selection contains 64 samples from 32 objects, two views each. The latest integration tests reported a maximum cross-view object-frame point discrepancy of 3.58e-7, using float32 inverse transforms. Five CPU tests passed, covering the new flags' transform/padding, native dropout, checkpoint, and guard behavior. Existing source-based checks also verify target tensor reshaping and normalization arithmetic.

These checks do not establish alignment of freely generated occupancy with source geometry. An independently verified decoded-target/source-mesh comparison remains distinct from coordinate round trips. The discussion also distinguished shared-noisy-target, per-location denoising comparisons from coordinate validation; that analysis was proposed, not completed in the evidence reviewed here.

Sources: [diagnostics.py](../diagnostics.py), [CPU tests](../tests/test_conditioning_variants.py), [sample selection](../outputs/diagnostics/sample_selection.json), [coordinate audit](../outputs/diagnostics/coordinate_research/preliminary.json).

## 8. Status at report preparation

| Item | Status |
|---|---|
| VecSetX bottleneck conditioning; frozen/trainable encoder options | Implemented; training comparisons reported |
| Touch, joint, and full-surface input sources | Implemented; comparisons reported |
| Full shape cross-attention scope | Implemented; three full runs archived locally |
| Position metadata embedding, log scale, zero output initialization | Implemented; historical variants differ; archived position/full-CA run available |
| Standalone pretrained-AE reconstruction experiments | Implemented; 378-view exported results recovered |
| `--vecsetx-learn` | Committed; user reports run in progress; no resulting scores reviewed |
| `--no-pointmap` | Implemented in current uncommitted worktree; CPU verified; no training results |
| `--oracle-point-frame` | Implemented in current uncommitted worktree; CPU verified; no training results |
| Full shape-backbone fine-tuning, alternative replacement encoders, learned pose-alignment module | Discussed; no implementation/run established by reviewed evidence |
| New ambiguity-controlled or alternative mesh dataset | Discussed; not established as created |

`learn()` uses the pretrained intermediate decoder features: encode `[1024,32]` → bottleneck expansion and 24 transformer blocks → `[1024,1024]` → touch projection. It does not invoke the final SDF-query output stage. Its parameters remain frozen; the adapter width is selected automatically.

No-pointmap uses the existing fuser's forced modality dropping for cropped/full pointmap tokens while retaining preprocessing. Oracle mode transforms only the full-surface conditioning into the saved object frame before VecSetX normalization. Both options are checkpointed and restored by current diagnostics. Oracle mode is restricted to full surface with no joint-pointmap concatenation and no position embedding. It is privileged-input experimentation, not an inference-ready learned alignment method.

## Evidence coverage

Reviewed Git history from September 1 through `daba25d`, the pre-VecSetX encoder, current worktree, current jobs/configurations, local W&B summaries, saved run-audit metadata, the reconstruction attachment, and relevant accessible conversations. Main implementation history was also present in the current conversation.

Additional conversations inspected: `Review sam-3d-touch repos`, `Assess PointWorld PTV3 usage`, the separate reconstruction-focused `Integrate VecSetX touch encoder (2)`, and `Point Encoder Alternatives`. Literature alternatives were discussed, but prior assistant interpretations were not treated as measured results. Conversation retrieval was selective, not an exhaustive export of every chat or every historical turn. Unrelated personal conversations were not reviewed.

Only three downstream run folders are locally archived. Cluster-only artifacts, live scheduler status, and the running learn experiment were not independently accessed. Some historical diagnostic outputs are existing audits rather than freshly recomputed measurements. The report labels user-reported outcomes and proposals separately from saved numerical results. No slides were created and no production code was changed for this report.

## September 8 update

Source: recent completed turns in `Integrate VecSetX touch encoder` (`01a06381-2bd5-7003-9d37-ffaba52c9865`), checked against Git through `e401a8b` and the current scripts. No new downstream metrics folder is available locally. Implementation and configuration are verified below; completed cluster results are not inferred from commands or job files.

### Conditioning experiments and jobs

- The oracle/no-pointmap integration is now committed (`8828d93`).
- A subsequent correction applies forced dropping to the actual external condition embedder when present, falling back to the backbone condition embedder (`76bf2e4`). This corrects the initial integration's assumption about where the fuser lives.
- Added a matched image/mask-only baseline with `--no-touch --no-pointmap --cross-attention-scope full`.
- Added/configured full-surface conditioning without native pointmap. The chat records a disk-quota-related failure, followed by a resume job using `last.pt` and the existing W&B run. Current completion status and final loss have not been independently recovered.
- The recent chat describes the learn-feature trajectory as matching the ordinary full-cross-attention trajectory. This is a conversation-recorded observation, not a newly recovered final metric. No local learn-run results are available.

The resulting four ordinary-generation conditions are image/mask with or without pointmap, each with or without full-surface conditioning. These are distinct from the separate oracle-orientation comparisons. Removing pointmap retains native zeroed modality slots and existing preprocessing.

### Current ordinary-generation evaluation

`evaluate.py` replaces the older `eval_quantitative.py`; the obsolete `eval_qualitative.py` was removed. It restores each run's saved configuration and adapted weights through shared restoration code, preserving its encoder, point source, position, pointmap, and oracle settings. The selection configuration chooses examples without overriding each model's saved training data configuration.

The evaluator supports:

- Official pretrained SAM3D, the supplied trained runs, and `decoded_gt`.
- `decoded_gt` takes the saved target Stage-1 latent through occupancy decoding and Stage 2. It is distinct from the actual source mesh and remains subject to decoding/Stage-2 error.
- One view per validation object selected by the largest hidden full-surface fraction. The chat records 95 available validation objects for this selection.
- Deterministic paired per-sample seeds, 25 Stage-1 sampling steps, and 25 Stage-2 sampling steps.
- Stage-2 mesh normalization and source-mesh ICP alignment, followed by geometric metrics and saved artifacts.
- Resume behavior that skips successful condition/sample rows and retries missing or failed results.

The current `jobs/evaluate.sh` requests one H200, 16 CPUs, and 64 GB RAM and invokes Python directly. It lists the image+pointmap and image+pointmap+full-surface runs. Official and decoded-GT conditions are automatic. The chat provides a command adding the two no-pointmap runs, but the checked-in job has not yet added those run directories.

Saved outputs include `metrics.csv`, `summary.yaml`, selected samples/views, source target meshes, and per-condition Stage-1 latents/occupancy, Stage-2 latents, meshes, and alignment transforms. No local `outputs/evaluation` results were present during this update, so this report gives no new F-score, Chamfer, IoU, or qualitative outcome.

Sources: [evaluate.py](../evaluate.py), [evaluation job](../jobs/evaluate.sh), commits `35cb16b`, `1c6a55c`, `3d8bc4d`.

### Stage-1 voxel visualization and alignment

The current evaluator and orbit renderer retain actual Stage-1 decoded occupancy. These voxels are different from the surface-voxel IoU computed from Stage-2 meshes.

The evaluator applies the associated Stage-2 mesh normalization and ICP alignment to Stage-1 coordinates and records `stage1_aligned_chamfer`, `stage1_aligned_voxel_iou_64`, and ICP fitness/RMSE. Prediction and decoded-target support each use their corresponding alignment.

This provides evaluation-only, post-aligned shape comparisons. It removes pose differences and depends on the quality of the Stage-2-derived alignment. It does not establish native prediction/target frame agreement or measure layout accuracy.

`make_eval_orbit.py` now renders mesh and Stage-1 voxel orbits, including the actual source mesh and decoded-GT reference. Recent changes make GIF the default and MP4 optional, fit shared camera framing, enable sRGB material handling, and clean up renderers. Lighting remained unchanged at the user's request. Source: [orbit renderer](../make_eval_orbit.py), `e401a8b`.

### Paired noisy-target experiment: now implemented

Added [experiments/noisy_target/evaluate.py](../experiments/noisy_target/evaluate.py) and [visualize.py](../experiments/noisy_target/visualize.py) in `35cb16b`. This supersedes the earlier report's proposed-only status.

The script reuses the current loader, checkpoint restoration, batch preparation, native noise generation, interpolation, target velocity, and conditional model forward. Each model retains its normal conditioning and receives matching targets/noise. Input hashes check pairing across runs. It evaluates the training-style conditional flow path without inference CFG.

Current defaults:

- 32 objects, up to two randomly selected views per object.
- Two noise draws per batch and fixed times 0.2, 0.5, 0.8, where 0 denotes noise and 1 denotes data.
- Batch size four, bf16, one GPU.
- Per-sample error arrays with axes `[time, noise_draw, latent_x, latent_y, latent_z]`; the spatial grid is 16³ and velocity MSE averages the eight latent channels.

The visualizer subtracts comparison-model error from baseline error, averages paired draws, and plots central slices. Positive/blue values mean lower target-velocity error for the comparison model. They do not mean closer to a learned data distribution. A latent-grid location is not an individual 64³ occupancy voxel.

The current implementation does not perform partial denoising or decode clean-latent estimates. It independently selects samples and does not yet consume ordinary evaluation's `selected_samples.yaml`. Noise is deterministic for a fixed batch arrangement, but changing batch size changes its assignment. No resulting noisy-target measurements or heatmaps were recovered in this update.

### Slide-ready factual outline

1. **Point conditioning changes:** custom clustered encoder replaced by pretrained VecSetX; touch, joint, and full-surface variants; K/V versus full cross-attention.
2. **Training results:** show matched validation curves and the three archived best-loss values. State that full-surface conditioning has not improved this validation objective over the matched image/mask/pointmap baseline.
3. **Encoder reconstruction check:** show input and reconstructed geometry for the same object under full, touch, and joint input; report the 95-object/378-view scope. Distinguish autoencoder reconstruction from downstream conditioning utility.
4. **Evaluation and current experiments:** ordinary generation with official/decoded-GT/source-mesh references; matched no-pointmap baselines; paired noisy-target error maps now implemented. Mark results pending where none are available.

For a single update slide, use the title **Point conditioning: current results and diagnostic experiments** with the archived validation-loss comparison as the main figure and a short implementation/status list. Keep code-cleanup details, job settings, and speculative causes in backup material.
