"""Compare trained models on identical partially noised target latents."""

import argparse
import gc
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from dataloader import build_dataloader
from evaluate import read_run, restore_run
from train import amp, build_stage1_pipeline, prepare_batch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, help="Override saved run data configuration")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/noisy_target/outputs"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--objects", type=int, default=32, help="0 uses every object")
    parser.add_argument("--views-per-object", type=int, default=2, help="0 uses every view")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--times", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--noise-draws", type=int, default=2)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def select_samples(records, objects, views, seed):
    grouped = {}
    for record in records:
        grouped.setdefault(record["object_id"], []).append(record)
    rng = random.Random(seed)
    object_ids = sorted(grouped)
    rng.shuffle(object_ids)
    selected = []
    for object_id in object_ids[:objects or None]:
        candidates = sorted(grouped[object_id], key=lambda record: record["sample_id"])
        rng.shuffle(candidates)
        selected.extend(record["sample_id"] for record in candidates[:views or None])
    return selected


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def predict(generator, targets, noise, time, condition_args, condition_kwargs, touch_tokens):
    """Native flow target and conditional forward, matching the training d=0 path."""
    if generator.self_consistency_prob != 0 or generator.fm_eps_max != 0:
        raise ValueError("This experiment requires the training flow-matching path with d=0")
    t = targets["shape"].new_full((len(targets["shape"]),), time)
    xt = generator._generate_xt(noise, targets, t)
    target = generator._generate_target(noise, targets)
    kwargs = dict(condition_kwargs)
    if touch_tokens is not None:
        kwargs["touch_tokens"] = touch_tokens
    prediction = generator.reverse_fn(
        xt, t * generator.time_scale, *condition_args, d=torch.zeros_like(t), **kwargs
    )
    error = (prediction["shape"].float() - target["shape"].float()).square().mean(dim=-1)
    return error.reshape(-1, 16, 16, 16)


def main():
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use python on one GPU, not distributed torchrun")
    if args.batch_size < 1 or args.noise_draws < 1 or args.objects < 0 or args.views_per_object < 0:
        raise ValueError("Invalid sample/batch counts")
    if not all(0 <= t < 1 for t in args.times) or len(set(args.times)) != len(args.times):
        raise ValueError("Times must be distinct and in [0,1): 0 is noise, 1 is data")
    names = [path.parent.name for path in args.checkpoints]
    if len(set(names)) != len(names):
        raise ValueError("Checkpoints must have distinct run-directory names")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "metrics.json").exists():
        raise FileExistsError("Choose a new output directory for a new comparison")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    selected = None
    paired_inputs = {}
    results = {}

    for name, path in zip(names, args.checkpoints):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        checkpoint, config, data = read_run(path, args.data_config, args.split)
        conditioning = checkpoint.get(
            "conditioning_config", {"no_pointmap": False, "oracle_point_frame": False}
        )
        loader = build_dataloader(
            data, args.batch_size, args.workers, shuffle=False,
            include_touch=checkpoint["touch_config"] is not None,
            oracle_point_frame=conditioning["oracle_point_frame"],
        )
        if selected is None:
            selected = select_samples(loader.dataset.records, args.objects, args.views_per_object, args.seed)
            if not selected:
                raise ValueError("No matching samples")
        records = {record["sample_id"]: record for record in loader.dataset.records}
        loader.dataset.records = [records[sample_id] for sample_id in selected]
        pipeline = build_stage1_pipeline(args.pipeline_config, device)
        model = restore_run(pipeline, checkpoint, device)
        # Keep the native CFG wrapper in the same conditional-only mode as training.
        # Do not call model.eval(): that would enable inference guidance.
        generator = pipeline.ss_generator
        run_dir = args.output_dir / name
        run_dir.mkdir(exist_ok=True)
        rows = []
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                targets, cond_args, cond_kwargs, points, mask = prepare_batch(
                    pipeline, batch, device, args.precision, model.touch_encoder is not None,
                    checkpoint["mode"] == "image_touch_joint", conditioning["oracle_point_frame"],
                )
                with amp(device, args.precision):
                    tokens = model.touch_encoder(points, mask) if model.touch_encoder is not None else None
                maps = {time: [] for time in args.times}
                for draw in range(args.noise_draws):
                    torch.manual_seed(args.seed + batch_index * args.noise_draws + draw)
                    noise = generator._generate_x0(targets)
                    key = (batch_index, draw)
                    identity = (
                        tuple(batch["sample_id"]), generator.sigma_min, generator.time_scale,
                        tensor_digest(batch["image"]), tensor_digest(batch["pointmap"]),
                        tuple((field, tensor_digest(targets[field]), tensor_digest(noise[field]))
                              for field in sorted(targets)),
                    )
                    if key in paired_inputs and paired_inputs[key] != identity:
                        raise ValueError("Models did not receive identical targets/noise and flow settings")
                    paired_inputs[key] = identity
                    for time in args.times:
                        with amp(device, args.precision):
                            error = predict(generator, targets, noise, time, cond_args, cond_kwargs, tokens)
                        maps[time].append(error.cpu().numpy())
                for index, sample_id in enumerate(batch["sample_id"]):
                    # [time, draw, x, y, z], retaining individual draws rather than hiding variation.
                    errors = np.stack([np.stack(maps[time])[:, index] for time in args.times])
                    filename = sample_id.replace("/", "_") + ".npz"
                    np.savez_compressed(run_dir / filename, errors=errors, times=args.times)
                    rows.append({
                        "sample_id": sample_id, "object_id": records[sample_id]["object_id"],
                        "error_by_time": errors.mean(axis=(1, 2, 3, 4)).tolist(), "file": filename,
                    })
                print(f"{name}: batch {batch_index + 1}/{len(loader)}", flush=True)
        results[name] = {
            "checkpoint": str(path), "step": checkpoint["step"],
            "conditioning_config": conditioning, "touch_config": checkpoint["touch_config"],
            "cross_attention_scope": checkpoint.get("cross_attention_scope", "kv"),
            "data": data, "samples": rows,
            "mean_error_by_time": np.mean([row["error_by_time"] for row in rows], axis=0).tolist(),
        }
        del model, generator, pipeline, tokens, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.output_dir / "metrics.json", "w") as file:
        json.dump({
            "settings": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items() if key != "checkpoints"},
            "time_convention": "0=noise, 1=data; native flow velocity error, no CFG",
            "map_axes": "time, noise_draw, latent_x, latent_y, latent_z",
            "runs": results,
        }, file, indent=2)


if __name__ == "__main__":
    main()
