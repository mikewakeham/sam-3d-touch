import argparse
import copy
import csv
import os
import random
import re
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from dataloader import build_dataloader, load_data_config
from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
from train import build_stage1_pipeline, normalize_touch, preprocess_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--pipeline-config", type=Path, default=Path("checkpoints/hf/pipeline.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def latent_cube(latent):
    return latent.permute(0, 2, 1).contiguous().view(latent.shape[0], 8, 16, 16, 16)


def decode_voxels(decoder, latent):
    voxels = decoder(latent_cube(latent)) > 0
    return voxels[:, 0] if voxels.ndim == 5 else voxels


def voxel_metrics(prediction, target):
    dimensions = tuple(range(1, prediction.ndim))
    intersection = (prediction & target).sum(dim=dimensions).float()
    predicted = prediction.sum(dim=dimensions).float()
    expected = target.sum(dim=dimensions).float()
    union = predicted + expected - intersection
    return {
        "iou": intersection / union.clamp_min(1),
        "dice": 2 * intersection / (predicted + expected).clamp_min(1),
        "precision": intersection / predicted.clamp_min(1),
        "recall": intersection / expected.clamp_min(1),
        "predicted_voxels": predicted,
        "target_voxels": expected,
        "volume_error": (predicted - expected).abs() / expected.clamp_min(1),
    }


def sample_shape(pipeline, condition_args, touch_tokens, batch_size, inference_steps, device):
    generator = pipeline.ss_generator
    previous_steps = generator.inference_steps
    generator.inference_steps = inference_steps
    latent_shapes = {
        name: (batch_size, mapping.pos_emb.shape[0], mapping.input_layer.in_features)
        for name, mapping in pipeline.backbone.latent_mapping.items()
    }
    condition_kwargs = {"touch_tokens": touch_tokens} if touch_tokens is not None else {}
    result = generator(latent_shapes, device, *condition_args, **condition_kwargs)
    generator.inference_steps = previous_steps
    return result["shape"]


def restore_official_kv(cross_attention_kv, official_kv):
    for module, state in zip(cross_attention_kv, official_kv):
        module.load_state_dict(state)


def load_run(run_dir, pipeline, cross_attention_kv, official_kv, device):
    restore_official_kv(cross_attention_kv, official_kv)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    for module, state in zip(cross_attention_kv, checkpoint["shape_cross_attention_kv"]):
        module.load_state_dict(state)

    encoder = None
    if checkpoint["touch_encoder"] is not None:
        encoder = TouchEncoder(pipeline.backbone.cond_channels).to(device)
        encoder.load_state_dict(checkpoint["touch_encoder"])
        encoder.eval()
    return encoder


def make_touch_cache(loader, pipeline, device):
    cache = {}
    with torch.no_grad():
        for batch in loader:
            inputs = preprocess_batch(pipeline, batch["image"], batch["pointmap"])
            xyz = normalize_touch(batch["touch_xyz"].to(device), inputs, pipeline.ss_preprocessor).cpu()
            for index, sample_id in enumerate(batch["sample_id"]):
                cache[sample_id] = (xyz[index], batch["touch_mask"][index].clone())
    return cache


def shuffled_ids(sample_ids, object_ids, seed):
    if len(set(object_ids.values())) < 2:
        return None
    donors = sample_ids.copy()
    rng = random.Random(seed)
    for _ in range(100):
        rng.shuffle(donors)
        if all(object_ids[sample] != object_ids[donor]
               for sample, donor in zip(sample_ids, donors)):
            return dict(zip(sample_ids, donors))
    return {sample: next(donor for donor in donors if object_ids[sample] != object_ids[donor])
            for sample in sample_ids}


def evaluate_condition(name, pipeline, decoder, encoder, loader, touch_cache, donor_ids,
                       records, args, mode="correct"):
    rows = []
    condition_dir = args.output_dir / "voxels" / safe_name(name)
    condition_dir.mkdir(parents=True, exist_ok=True)
    dtype = pipeline.shape_model_dtype
    completed = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            inputs = preprocess_batch(pipeline, batch["image"], batch["pointmap"])
            touch_xyz = None
            if encoder is not None and mode != "omitted":
                ids = batch["sample_id"]
                if mode == "shuffled":
                    ids = [donor_ids[sample_id] for sample_id in ids]
                touch_xyz = torch.stack([touch_cache[sample_id][0] for sample_id in ids]).to(args.device)
                touch_mask = torch.stack([touch_cache[sample_id][1] for sample_id in ids]).to(args.device)

            torch.manual_seed(args.seed + batch_index)
            with torch.autocast(
                device_type=torch.device(args.device).type, dtype=dtype,
                enabled=not args.no_amp and torch.device(args.device).type == "cuda"
            ):
                condition_args, _ = pipeline.get_condition_input(
                    pipeline.ss_condition_embedder, inputs, pipeline.ss_condition_input_mapping
                )
                touch_tokens = encoder(touch_xyz, touch_mask) if touch_xyz is not None else None
                prediction = sample_shape(
                    pipeline, condition_args, touch_tokens, len(batch["sample_id"]),
                    args.inference_steps, args.device
                )
                predicted_voxels = decode_voxels(decoder, prediction)
                target = batch["target_shape"].to(args.device)
                target_voxels = decode_voxels(decoder, target)

            metrics = voxel_metrics(predicted_voxels, target_voxels)
            latent_mse = (prediction.float() - target.float()).square().mean(dim=(1, 2))

            for index, sample_id in enumerate(batch["sample_id"]):
                path = condition_dir / f"{safe_name(sample_id)}.npz"
                centers = touch_cache[sample_id][0][:, 0].numpy()
                np.savez_compressed(
                    path,
                    prediction=predicted_voxels[index].cpu().numpy(),
                    target=target_voxels[index].cpu().numpy(),
                    touch_centers=centers,
                )
                record = records[sample_id]
                row = {
                    "condition": name,
                    "sample_id": sample_id,
                    "object_id": record["object_id"],
                    "image_path": str(loader.dataset.path(record["image_path"])),
                    "voxel_path": str(path.relative_to(args.output_dir)),
                    "latent_mse": float(latent_mse[index]),
                }
                for metric, values in metrics.items():
                    row[metric] = float(values[index])
                rows.append(row)

            completed += len(batch["sample_id"])
            print(f"{name}: {completed}/{len(loader.dataset)}", flush=True)
    return rows


def summarize(rows, primary_conditions, diagnostic_conditions, no_touch, best_touch):
    metrics = ["iou", "dice", "precision", "recall", "volume_error", "latent_mse"]
    summary = {
        "primary_conditions": primary_conditions,
        "diagnostic_conditions": diagnostic_conditions,
        "no_touch": no_touch,
        "best_touch": best_touch,
        "conditions": {},
        "comparisons": {},
    }
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], {})[row["sample_id"]] = row

    for condition, samples in by_condition.items():
        summary["conditions"][condition] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in samples.values()])),
                "median": float(np.median([row[metric] for row in samples.values()])),
            }
            for metric in metrics
        }

    baselines = ["official"] + ([no_touch] if no_touch else [])
    for condition in primary_conditions + diagnostic_conditions:
        for baseline in baselines:
            if condition == baseline:
                continue
            common = sorted(set(by_condition[condition]) & set(by_condition[baseline]))
            deltas = [by_condition[condition][sample]["iou"] - by_condition[baseline][sample]["iou"]
                      for sample in common]
            summary["comparisons"][f"{condition}_vs_{baseline}"] = {
                "mean_iou_delta": float(np.mean(deltas)),
                "median_iou_delta": float(np.median(deltas)),
                "improved_fraction": float(np.mean(np.asarray(deltas) > 0)),
            }
    return summary


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_config = load_data_config(args.data_config)
    data_config["dataset"]["split"] = args.split
    data_config["touch"]["contacts"]["shuffle_after_selection"] = False
    loader = build_dataloader(data_config, args.batch_size, args.workers, shuffle=False)
    if args.max_samples and args.max_samples < len(loader.dataset.records):
        loader.dataset.records = random.Random(args.seed).sample(
            loader.dataset.records, args.max_samples
        )

    records = {record["sample_id"]: record for record in loader.dataset.records}
    object_ids = {sample_id: record["object_id"] for sample_id, record in records.items()}
    sample_ids = list(records)

    pipeline = build_stage1_pipeline(args.pipeline_config, args.device)
    config = OmegaConf.load(args.pipeline_config)
    decoder = pipeline.init_ss_decoder(config.ss_decoder_config_path, config.ss_decoder_ckpt_path)
    dtype = config.get("shape_model_dtype") or config.get("dtype", "float16")
    pipeline.shape_model_dtype = pipeline._get_dtype(dtype)
    pipeline.override_ss_generator_cfg_config(
        pipeline.ss_generator,
        cfg_strength=config.get("ss_cfg_strength", 7),
        inference_steps=args.inference_steps,
        rescale_t=config.get("ss_rescale_t", 3),
        cfg_interval=config.get("ss_cfg_interval", [0, 500]),
        cfg_strength_pm=config.get("ss_cfg_strength_pm", 0.0),
    )
    pipeline.ss_generator.no_shortcut = True
    pipeline.ss_generator.eval()
    decoder.eval()

    cross_attention_kv = [block.cross_attn["shape"].to_kv for block in pipeline.backbone.blocks]
    official_kv = [
        {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for module in cross_attention_kv
    ]

    touch_cache = make_touch_cache(loader, pipeline, args.device)
    donor_ids = shuffled_ids(sample_ids, object_ids, args.seed + 1)
    rows = []
    primary_conditions = ["official"]
    diagnostic_conditions = []

    restore_official_kv(cross_attention_kv, official_kv)
    rows += evaluate_condition(
        "official", pipeline, decoder, None, loader, touch_cache, donor_ids,
        records, args
    )

    touch_runs = []
    no_touch = None
    for run_dir in args.run_dirs:
        name = safe_name(run_dir.name)
        encoder = load_run(run_dir, pipeline, cross_attention_kv, official_kv, args.device)
        rows += evaluate_condition(
            name, pipeline, decoder, encoder, loader, touch_cache, donor_ids,
            records, args
        )
        primary_conditions.append(name)
        if encoder is None:
            no_touch = name
        else:
            touch_runs.append((name, run_dir))

    means = {
        name: np.mean([row["iou"] for row in rows if row["condition"] == name])
        for name, _ in touch_runs
    }
    best_touch = max(means, key=means.get) if means else None
    if best_touch is not None:
        best_run = next(run_dir for name, run_dir in touch_runs if name == best_touch)
        encoder = load_run(best_run, pipeline, cross_attention_kv, official_kv, args.device)
        modes = ["omitted"] if donor_ids is None else ["shuffled", "omitted"]
        for mode in modes:
            name = f"{best_touch}_{mode}"
            rows += evaluate_condition(
                name, pipeline, decoder, encoder, loader, touch_cache,
                donor_ids, records, args, mode=mode
            )
            diagnostic_conditions.append(name)

    fieldnames = list(rows[0])
    with open(args.output_dir / "metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(
        rows, primary_conditions, diagnostic_conditions, no_touch, best_touch
    )
    with open(args.output_dir / "summary.yaml", "w") as file:
        yaml.safe_dump(summary, file, sort_keys=False)
    with open(args.output_dir / "config.yaml", "w") as file:
        yaml.safe_dump({
            "run_dirs": [str(path) for path in args.run_dirs],
            "pipeline_config": str(args.pipeline_config),
            "data_config": str(args.data_config),
            "split": args.split,
            "samples": len(loader.dataset),
            "batch_size": args.batch_size,
            "inference_steps": args.inference_steps,
            "seed": args.seed,
        }, file, sort_keys=False)
    print(f"saved evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
