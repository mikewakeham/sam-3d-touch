"""Paired native velocity loss and fixed-endpoint latent scoring. No decoder."""
import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("LIDRA_SKIP_INIT", "true")
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from omegaconf import OmegaConf
from dataloader import TouchDataset, collate_touch_batch, load_data_config
from evaluation.evaluate import read_run, restore_run, stable_seed
from train import amp, build_stage1_pipeline, prepare_batch, checkpoint_train_scope
from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_probe_gpu import make_bank
from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_probe_core import paired_loss, tensor_sha
from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_rollout_core import sample_from_noise
from experiments.coordinate_system.scripts.rotation_loss.rotation_utils import (
    select_groups, load_encoder, native_means, native_rotations, endpoint_scores,
)


def run(args):
    data = load_data_config(args.data_config)
    groups = select_groups(data, args.train_objects, args.val_objects, args.views, args.seed)
    selected = {r["object_id"]: r for _, records in groups for r in records}
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    encoder = load_encoder(args.encoder_checkpoint, device)
    candidates = {}
    for index, (oid, record) in enumerate(selected.items()):
        candidates[oid] = native_means(encoder, data["dataset"]["root"], record, device)
        print(f"Encoded scoring references {index+1}/{len(selected)}", flush=True)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    cfg = OmegaConf.load(args.pipeline_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rotations = [name for name, _, _, _ in native_rotations()]
    loss_fields = ["model", "split", "object_id", "sample_id", "surface",
                   "time_kind", "draw", "time", "bank_seed", "noise_sha256", "velocity_mse"]
    endpoint_fields = ["model", "split", "object_id", "sample_id", "draw", "seed",
                       "identity_mse", "best_rotation", "best_mse"]+[f"mse_{name}" for name in rotations]
    report = {"complete": False, "settings": {k: [str(p) for p in v] if k == "run_dirs"
              else str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "selection": [{"split": split, "sample_ids": [r["sample_id"] for r in records],
                             "object_ids": [r["object_id"] for r in records]} for split, records in groups],
              "models": [], "definitions": {
                  "velocity_mse": "Observed native shape flow loss with paired time/noise; no visual dropout or CFG",
                  "endpoint_mse": "Direct MSE of a fixed sampled latent against separately encoded native-scale targets",
                  "best_rotation": "Minimum over seven tested orientations; not a continuous alignment optimum",
                  "wrong_surface": "Cyclic token swap after preprocessing; correct pointmap and surface normalization statistics remain fixed"}}
    started = time.monotonic()
    with (args.output_dir/"velocity_losses.csv").open("w", newline="") as lf, \
         (args.output_dir/"endpoint_scores.csv").open("w", newline="") as ef:
        losses = csv.DictWriter(lf, fieldnames=loss_fields)
        endpoints = csv.DictWriter(ef, fieldnames=endpoint_fields)
        losses.writeheader()
        endpoints.writeheader()
        for run_dir in args.run_dirs:
            torch.manual_seed(args.seed)
            checkpoint, saved, run_data = read_run(run_dir/args.checkpoint)
            conditioning = checkpoint.get("conditioning_config", {})
            pipeline = build_stage1_pipeline(args.pipeline_config, device)
            model = restore_run(pipeline, checkpoint, device)
            model.requires_grad_(False).eval()
            gen = pipeline.ss_generator
            gen.fm_eps_max = 0.
            gen.self_consistency_prob = 0.
            gen.reverse_fn.p_unconditional = 0.
            pipeline.override_ss_generator_cfg_config(
                gen, cfg_strength=args.cfg_strength, inference_steps=args.inference_steps,
                rescale_t=float(cfg.get("ss_rescale_t", 3)),
                cfg_interval=list(cfg.get("ss_cfg_interval", [0, 500])),
                cfg_strength_pm=0.)
            gen.no_shortcut = True
            datasets, lookups = {}, {}
            use_touch = model.touch_encoder is not None
            for split in {split for split, _ in groups}:
                config = copy.deepcopy(run_data)
                config["dataset"]["split"] = split
                datasets[split] = TouchDataset(config,
                    include_touch=use_touch or conditioning.get("shared_pointmap_normalization", False),
                    oracle_point_frame=conditioning.get("oracle_point_frame", False))
                lookups[split] = {r["sample_id"]: i for i, r in enumerate(datasets[split].records)}
            report["models"].append({"name": run_dir.name, "checkpoint": str(run_dir/args.checkpoint),
                "step": checkpoint["step"], "train_scope": checkpoint_train_scope(checkpoint),
                "mode": checkpoint["mode"],
                "conditioning": conditioning, "training": checkpoint.get("training_config", {}),
                "sigma_min": gen.sigma_min, "time_sampler": repr(gen.training_time_sampler_fn)})
            del checkpoint, saved
            with torch.no_grad():
                for gi, (split, records) in enumerate(groups):
                    dataset = datasets[split]
                    batch = collate_touch_batch([dataset[lookups[split][r["sample_id"]]] for r in records])
                    targets, ca, kw, xyz, mask = prepare_batch(
                        pipeline, batch, device, args.precision, use_touch,
                        joint_pointmap=report["models"][-1]["mode"] == "image_touch_joint",
                        oracle_point_frame=conditioning.get("oracle_point_frame", False),
                        shared_pointmap_normalization=conditioning.get("shared_pointmap_normalization", False))
                    if len(ca) != 1 or kw:
                        raise ValueError("Expected the existing precomputed visual-token context")
                    expected = np.stack([candidates[r["object_id"]]["identity"] for r in records])
                    np.testing.assert_array_equal(targets["shape"].float().cpu().numpy(), expected)
                    with amp(device, args.precision):
                        tokens = model.get_touch_tokens(xyz, mask)
                    seed = stable_seed(args.seed, "|".join(batch["sample_id"]))
                    bank = make_bank(gen, targets, seed)
                    # Flip only CFG's branch flag; backbone/encoders remain eval.
                    gen.reverse_fn.training = True
                    token_conditions = [("correct", tokens)]
                    if args.wrong_surface and tokens is not None:
                        token_conditions.append(("wrong", tokens.roll(1, 0)))
                    for surface, conditioned in token_conditions:
                        for kind, draw, times, noise in bank:
                            with amp(device, args.precision):
                                values = paired_loss(gen, targets, ca[0], conditioned, times, noise)
                            noise_digest = tensor_sha(noise["shape"])
                            for i, record in enumerate(records):
                                losses.writerow(dict(model=run_dir.name, split=split,
                                    object_id=record["object_id"], sample_id=record["sample_id"],
                                    surface=surface, time_kind=kind, draw=draw, time=float(times[i]),
                                    bank_seed=seed, noise_sha256=noise_digest, velocity_mse=values[i]))
                    gen.reverse_fn.training = False
                    for draw in range(args.draws):
                        sample_seed = seed+500000+draw
                        torch.manual_seed(sample_seed)
                        # Shapes only: ground-truth latent values never initialize sampling.
                        noise = gen._generate_noise({k: tuple(v.shape) for k, v in targets.items()}, device)
                        with amp(device, args.precision):
                            prediction = sample_from_noise(gen, noise, ca[0], tokens)
                        for i, record in enumerate(records):
                            scores = endpoint_scores(prediction[i].float().cpu().numpy(), candidates[record["object_id"]])
                            best = min(scores, key=scores.get)
                            endpoints.writerow(dict(model=run_dir.name, split=split,
                                object_id=record["object_id"], sample_id=record["sample_id"], draw=draw,
                                seed=sample_seed, identity_mse=scores["identity"], best_rotation=best,
                                best_mse=scores[best], **{f"mse_{name}": value for name, value in scores.items()}))
                            if gi == 0 and i == 0 and draw == 0:
                                example_dir = args.output_dir/"examples"
                                example_dir.mkdir(exist_ok=True)
                                np.savez_compressed(example_dir/f"{run_dir.name}_{record['sample_id']}.npz",
                                    prediction=prediction[i].float().cpu().numpy(),
                                    target=candidates[record["object_id"]]["identity"],
                                    best_target=candidates[record["object_id"]][best], best_rotation=best)
                    lf.flush()
                    ef.flush()
                    print(f"{run_dir.name}: group {gi+1}/{len(groups)}; {time.monotonic()-started:.1f}s elapsed", flush=True)
            del model, gen, pipeline
            gc.collect()
            torch.cuda.empty_cache()
    report["complete"] = True
    report["elapsed_seconds"] = time.monotonic()-started
    (args.output_dir/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Saved:", args.output_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--data-config", type=Path, default=REPO/"configs/data_full_surface.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPO/"checkpoints/hf/pipeline.yaml")
    parser.add_argument("--encoder-checkpoint", type=Path, default=REPO/"checkpoints/hf/ss_encoder.ckpt")
    parser.add_argument("--train-objects", type=int, default=16)
    parser.add_argument("--val-objects", type=int, default=16)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--inference-steps", type=int, default=25)
    parser.add_argument("--cfg-strength", type=float, default=0.)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wrong-surface", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
