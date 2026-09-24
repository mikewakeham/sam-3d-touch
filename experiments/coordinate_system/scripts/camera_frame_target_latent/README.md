# Camera-frame experiment — quick same-object / held-view test

Status: three cluster runs completed. [Results and interpretation](RESULTS_20260914.md): shared pointmap normalization gives no clear loss benefit; final reconstruction evaluation remains outstanding. No production edits or W&B. The user subsequently requested `submit.sh`, which submits three separate one-H100 jobs. The user replaced the earlier 128-object/5,000-step proposal with this quick test on September 14.

## Question and splits

Does camera-oriented target supervision improve learning across views of the SAME objects, and does aligning the pointmap with surface normalization help?

| Use | Objects | Views per object | Observations |
| --- | --- | --- | --- |
| Train | 16 | 8 | 128 |
| Validation | SAME 16 | 4 different held-out views | 64 |
| Fixed training-loss reference | SAME 16 | 2 of the 8 training views | 32 |
| Unseen-object validation | None | — | 0 |

Objects are hash-selected from the original training split; views are hash-selected without looking at scores. Training and held-out sample IDs cannot overlap. Any additional views are unused. Objects must have at least 12 views; failures are not silently replaced. Validation includes every selected held-out view. Preparation encodes only these 192 observations, not the whole dataset.

This tests fitting and transfer to other views of familiar shapes. It does not test new-object generalization. Camera-oriented surfaces AND targets change across views; there is no fixed oracle input code or constant-surface arm. Nevertheless, familiar-object memorization is still possible; success is scoped to held-out views.

## Three matched arms

| Arm | Target | Pointmap | Full surface |
| --- | --- | --- | --- |
| `object_stock` | Existing object-frame latent | Stock SAM3D normalization | Raw camera XYZ -> native VecSetX normalization |
| `camera_stock` | Re-encoded camera-frame latent | Stock SAM3D normalization | Same surface path |
| `camera_shared` | Same camera-frame latent | Surface center/radius, BOTH pointmap branches | Same surface path |

`camera_stock` versus `object_stock` changes the target convention. `camera_shared` versus `camera_stock` changes only pointmap normalization. All arms retain image, pointmap and full surface. Surface position tokens are disabled; frozen VecSetX, projector architecture and image processing are shared. No oracle transform is applied to any conditioning input.

## Coordinate operations

1. Load the source mesh in its saved centered object frame.
2. Apply recorded object-to-camera transform, including the SAM axis conversion.
3. For camera targets, center the full mesh's camera-frame AABB and divide by its longest extent. Voxelize in SAM3D's standard `[-.5,.5]^3` cube and encode one latent per selected view.
4. For VecSetX, use the observed surface's AABB center and maximum radius. All three arms feed the same RAW camera surface into the existing TouchEncoder, avoiding redundant stock-SSI normalization followed by radius normalization.
5. In `camera_shared`, replace BOTH cropped and full-image pointmap normalizers with that same observed-surface center/radius before existing crop/resize operations. Do not independently normalize the visible subset. RGB and mask transformations stay unchanged.
6. Target longest-extent units still differ from VecSetX radius units. The saved affine between them is for auditing/plotting, not an extra generator input. This is not an exact shared-unit target experiment.

Preparation checks seeded source-surface replay, target bounds and affine round trips; reports original/new frozen-VAE reconstruction fidelity without an arbitrary IoU gate. Eight figures (four train, four held-view examples chosen before scoring) show source mesh, camera geometry, camera target, stock versus shared PM normalization, the remaining target/VecSetX unit difference, frozen-VAE target reconstruction, and image. Before/after conditioning panels use identical axes. Plot subsampling does not change training inputs. These figures verify spatial bookkeeping, not learned feature compatibility.

## Training budget and validation

- **1,000 optimizer updates**, microbatch/global batch **4**, no accumulation needed. 4,000 observations, about **31.25 passes** over 128 training views.
- Existing shape cross-attention plus touch projector training; frozen VecSetX/DINO. Generator LR `1e-5`, projector LR `1e-4`, AdamW weight decay 0, gradient clip 1, BF16.
- **50% per-example visual dropout during training**, no visual dropout in validation. Seed 29; same data order, initialization and addressed dropout/flow RNG across arms.
- Loss evaluation at **step 0, every 100 updates, and final**: `loss/train_fixed` on the 32 fixed training observations; `loss/held_view_fixed` on all 64 held-out observations. There is NO unseen-object loss in this quick run.
- Fixed batches and native flow noise/timesteps, one draw per observation, reset identically at every evaluation. Separate RNG contexts keep preprocessing/evaluation from advancing training randomness. Losses are sample-weighted means, not bootstrap estimates.
- Local training-loss logging every 10 steps (plus first/final). Measured seconds per update and remaining TRAINING time are printed; that ETA excludes future validation/checkpoint overhead.
- **No generation or reconstruction metrics during training.** Use saved checkpoints afterward. Similar loss across different target conventions does not prove similar reconstruction.

This is a quick one-seed learning-curve comparison. If a curve is still improving at 1,000, insufficient training remains possible; do not call the architecture or coordinate convention impossible. Broader training scope, dropout ablation, repeated seeds and unseen objects are separate follow-ups.

## Runtime estimate and its basis

The saved W&B export for the existing camera full-surface run `mpd9bbwh` reports median training throughput **18.82 observations/sec across four GPUs**, batch 4 per GPU. Oracle `cps50h2l` reports 18.90. Dividing by four gives a rough 4.7 observations/sec per GPU; this extrapolation is not a benchmark of the new script.

- Superseded plan: 5,000 updates × batch 16 = 80,000 observations -> roughly **4.7 hours of training per arm**, before validation/loading.
- Quick plan: 1,000 × batch 4 = 4,000 observations -> roughly **14 minutes of training**, or budget **20–30 minutes per arm including loading/validation/checkpoints**.
- Three concurrently allocated GPUs: roughly the same elapsed training time for all three arms. One GPU sequentially: roughly 60–90 minutes total.

Target preparation is separate and has not been timed. It now processes 192 selected views rather than 2,560. Hardware, filesystem throughput and startup can change these estimates; use the printed measured update times after launch. Older 4-object timing results used a different cached experiment path and are not the primary estimate.

## Existing prepared data

The source folder was renamed from `camera_frame_formal` to `camera_frame_target_latent`. Existing preparation outputs under the OLD output folder remain valid: pass their actual directory to `submit.sh`. Do not re-encode targets just because the source folder was renamed. The submission script requires a nonempty preparation directory, resolves it to an absolute path and checks for `preparation.json` before requesting GPUs.

## Interactive commands

First prepare the SMALL selection. Do not pass the old 128-object preparation to this quick test.

```bash
(
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
PREP=experiments/coordinate_system/outputs/camera_frame_target_latent/quick_preparation_$(date +%Y%m%d_%H%M%S)
python experiments/coordinate_system/scripts/camera_frame_target_latent/prepare_camera_frame_gpu.py \
  --data-config configs/data_full_surface.yaml --pipeline-config checkpoints/hf/pipeline.yaml \
  --train-objects 16 --train-views 8 --held-views 4 --val-objects 0 --seed 29 \
  --output-dir "$PREP"
printf 'Preparation directory: %s\n' "$PREP"
)
```

To submit three separate one-H100 training jobs after preparation, from the repository root:

```bash
PREP=experiments/coordinate_system/outputs/camera_frame_target_latent/quick_preparation_YYYYMMDD_HHMMSS
bash experiments/coordinate_system/scripts/camera_frame_target_latent/submit.sh "$PREP"
```

Set `PREP` to the completed preparation directory if using a new shell. `submit.sh` only submits training; preparation must already be complete. It requests the existing Kempner H100 partition/account, one GPU, 16 CPU cores, 64 GiB RAM and two hours per job. Each job runs the same Python command with its arm. Logs and results go in the printed `quick_fit_<timestamp>` experiment directory. No W&B. Slurm decides when each job starts.

The submission script can run from a login node. Preparation needs a GPU and is run once in an interactive allocation using the command above.

Then set `PREP` to the directory printed above. For THREE already allocated GPUs, run:

```bash
(
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
PREP=experiments/coordinate_system/outputs/camera_frame_target_latent/quick_preparation_YYYYMMDD_HHMMSS
RUN=experiments/coordinate_system/outputs/camera_frame_target_latent/quick_fit_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [ "${#devices[@]}" -lt 3 ]; then echo "This command needs three allocated GPUs"; exit 1; fi
pids=()
gpu=0
for arm in object_stock camera_stock camera_shared; do
  CUDA_VISIBLE_DEVICES="${devices[$gpu]}" python experiments/coordinate_system/scripts/camera_frame_target_latent/fit_camera_frame_gpu.py \
    --preparation-dir "$PREP" --arm "$arm" --output-dir "$RUN/$arm" \
    --steps 1000 --batch-size 4 --global-batch-size 4 --validate-every 100 \
    --workers 4 --train-scope shape_cross_attention --visual-dropout 0.5 \
    > "$RUN/${arm}.log" 2>&1 &
  pids+=("$!")
  gpu=$((gpu+1))
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
)
```

The command preserves assigned GPU IDs/UUIDs. With ONE GPU, run the same Python invocation sequentially for each arm, without `CUDA_VISIBLE_DEVICES` overrides or background `&`. This interactive alternative does not use Slurm; both launch methods use local JSON logging only.

## Files and checkpoints

Code lives here; outputs stay in ignored `experiments/coordinate_system/outputs/camera_frame_target_latent/`. New target means require about **24 MiB uncompressed for 192 views**, shared across arms; no bulk feature caches or automatic ZIPs.

Per arm: `config.json` (IDs/settings/source and initialization hashes), `metrics.jsonl`, `results.json` (loss curves/completion), and ONE rolling `latest.pt` trainable-weights checkpoint every 100 updates. At the end, `final.pt` replaces it. No optimizer history or frozen-model copies. Atomic replacement temporarily needs room for both files. Checkpoint size depends on trainable scope, not dataset size; keep weights/targets on cluster and return small JSON/selected figures.

Checkpoints use `camera_frame_target_latent_v1`. They need this experiment's conditioning wrapper, not vanilla `evaluation/evaluate.py`. `fit_camera_frame_gpu.restore_for_evaluation()` restores weights/components; use `training_runtime.prepare_batch()` with the saved arm and appropriate target manifest. Reconstruction scoring is deferred until after training. Resume optimization is not implemented; interrupted weights remain evaluable. Existing directories are never overwritten.

## Checks

CPU tests verify coordinate arithmetic, both actual pointmap preprocessing branches, agreement with actual VecSetX normalization, unchanged RGB/stock processing, batch accumulation, deterministic epochs and RNG restoration. Split tests verify exactly 16×8 training and 16×4 held-out views, identical object identities, no overlapping views, and zero unseen objects. Python syntax/CLI checks pass. GPU execution is untested locally.
