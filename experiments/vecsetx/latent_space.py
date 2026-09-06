import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from dataloader import TouchDataset, collate_touch_batch, load_data_config
from train import (
    amp,
    build_stage1_pipeline,
    combine_pointmap_and_touch,
    normalize_touch_to_pointmap_frame,
    preprocess_batch,
)
from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--touch-config", type=Path, default=Path("configs/data1.yaml"))
    parser.add_argument(
        "--full-surface-config",
        type=Path,
        default=Path("configs/data_full_surface.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/vecsetx/outputs/latent_space"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--objects", type=int, default=0, help="0 uses every object")
    parser.add_argument(
        "--views-per-object", type=int, default=0, help="0 uses every view"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shuffle-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    return parser.parse_args()


def make_dataset(config_path, split):
    config = copy.deepcopy(load_data_config(config_path))
    config["dataset"]["split"] = split
    return TouchDataset(config, include_touch=True)


def select_pairs(touch_dataset, surface_dataset, object_count, views_per_object, seed):
    surface_indices = {
        record["sample_id"]: index
        for index, record in enumerate(surface_dataset.records)
    }
    by_object = {}
    for touch_index, record in enumerate(touch_dataset.records):
        if record["sample_id"] in surface_indices:
            by_object.setdefault(record["object_id"], []).append(touch_index)

    object_ids = list(by_object)
    if views_per_object:
        object_ids = [
            object_id
            for object_id in object_ids
            if len(by_object[object_id]) >= views_per_object
        ]
    object_ids.sort()
    rng = random.Random(seed)
    rng.shuffle(object_ids)
    if object_count and object_count > len(object_ids):
        raise ValueError(
            f"Requested {object_count} objects with {views_per_object} views, "
            f"but only {len(object_ids)} are available"
        )
    if object_count:
        object_ids = object_ids[:object_count]

    pairs = []
    for object_id in object_ids:
        selected = list(by_object[object_id])
        if views_per_object:
            selected = rng.sample(selected, views_per_object)
        selected.sort(key=lambda index: touch_dataset.records[index]["view_id"])
        for touch_index in selected:
            record = touch_dataset.records[touch_index]
            pairs.append((touch_index, surface_indices[record["sample_id"]], record))
    return pairs


def code_distance(first, second):
    first = F.normalize(first.float().flatten(1), dim=1)
    second = F.normalize(second.float().flatten(1), dim=1)
    return 1 - (first * second).sum(dim=1)


def pairwise_distances(codes):
    codes = F.normalize(codes.float().flatten(1), dim=1)
    return (1 - codes @ codes.T).clamp_min(0)


def difference(first, second):
    delta = (first.float() - second.float()).flatten()
    return {
        "mean_absolute_difference": float(delta.abs().mean()),
        "max_absolute_difference": float(delta.abs().max()),
        "relative_l2_difference": float(
            delta.norm() / first.float().flatten().norm().clamp_min(1e-12)
        ),
    }


def view_metrics(codes, object_ids):
    distances = pairwise_distances(codes).cpu().numpy()
    object_ids = np.asarray(object_ids)
    same_object = object_ids[:, None] == object_ids[None, :]
    diagonal = np.eye(len(object_ids), dtype=bool)
    same = distances[same_object & ~diagonal]
    different = distances[~same_object]

    nearest = distances.copy()
    np.fill_diagonal(nearest, np.inf)
    nearest_indices = nearest.argmin(axis=1)
    return {
        "same_object_distance_mean": float(same.mean()),
        "same_object_distance_median": float(np.median(same)),
        "different_object_distance_mean": float(different.mean()),
        "different_object_distance_median": float(np.median(different)),
        "different_to_same_median_ratio": float(
            np.median(different) / max(np.median(same), 1e-12)
        ),
        "nearest_neighbor_object_accuracy": float(
            np.mean(object_ids[nearest_indices] == object_ids)
        ),
    }


def source_alignment(codes, object_ids):
    full = codes["full_surface"]
    full_distances = pairwise_distances(full)
    object_ids = np.asarray(object_ids)
    different_mask = torch.from_numpy(
        object_ids[:, None] != object_ids[None, :]
    )
    different_baseline = full_distances[different_mask].median().item()

    results = {}
    for source in ("touch", "joint"):
        distances = code_distance(full, codes[source])
        cross_distances = 1 - (
            F.normalize(codes[source].float().flatten(1), dim=1)
            @ F.normalize(full.float().flatten(1), dim=1).T
        )
        nearest = cross_distances.argmin(dim=1).cpu().numpy()
        results[source] = {
            "full_to_source_distance_mean": float(distances.mean()),
            "full_to_source_distance_median": float(distances.median()),
            "full_to_source_over_different_object_median": float(
                distances.median() / max(different_baseline, 1e-12)
            ),
            "nearest_full_exact_sample_accuracy": float(
                np.mean(nearest == np.arange(len(object_ids)))
            ),
            "nearest_full_same_object_accuracy": float(
                np.mean(object_ids[nearest] == object_ids)
            ),
        }
    results["different_object_full_surface_distance_median"] = different_baseline
    return results


def target_alignment(codes, targets, object_ids, shuffle_trials, seed):
    first_view = []
    seen = set()
    for index, object_id in enumerate(object_ids):
        if object_id not in seen:
            first_view.append(index)
            seen.add(object_id)

    targets = targets[first_view]
    target_distances = pairwise_distances(targets).cpu().numpy()
    triangle = np.triu_indices(len(first_view), 1)
    rng = np.random.default_rng(seed)
    results = {}
    for source, source_codes in codes.items():
        source_codes = source_codes[first_view]
        source_distances = pairwise_distances(source_codes).cpu().numpy()
        correlation = spearmanr(
            source_distances[triangle], target_distances[triangle]
        ).statistic

        shuffled = []
        for _ in range(shuffle_trials):
            permutation = rng.permutation(len(first_view))
            shuffled_distances = source_distances[permutation][:, permutation]
            shuffled.append(
                spearmanr(
                    shuffled_distances[triangle], target_distances[triangle]
                ).statistic
            )
        results[source] = {
            "spearman": float(correlation),
            "shuffled_spearman_mean": float(np.mean(shuffled)),
            "shuffled_spearman_std": float(np.std(shuffled)),
            "shuffled_spearman_p95": float(np.quantile(shuffled, 0.95)),
        }
    return results


def main():
    args = parse_args()
    if (
        args.objects < 0
        or args.views_per_object < 0
        or args.views_per_object == 1
        or args.batch_size < 1
    ):
        raise ValueError(
            "Use zero or at least two views per object and a positive batch size"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    touch_dataset = make_dataset(args.touch_config, args.split)
    surface_dataset = make_dataset(args.full_surface_config, args.split)
    pairs = select_pairs(
        touch_dataset,
        surface_dataset,
        args.objects,
        args.views_per_object,
        args.seed,
    )
    if len({record["object_id"] for _, _, record in pairs}) < 2:
        raise ValueError("Latent-space comparisons require at least two objects")

    pipeline = build_stage1_pipeline(args.pipeline_config, device)
    touch_encoder = TouchEncoder(
        encoder_name="vecsetx",
        output_dim=pipeline.backbone.cond_channels,
        trainable=False,
        use_position=False,
    ).to(device).eval()
    model = touch_encoder.encoder

    codes = {"full_surface": [], "touch": [], "joint": []}
    targets = []
    object_ids = []
    sample_ids = []
    checks = None
    with torch.inference_mode():
        for begin in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[begin : begin + args.batch_size]
            touch_batch = collate_touch_batch(
                [touch_dataset[touch_index] for touch_index, _, _ in batch_pairs]
            )
            surface_batch = collate_touch_batch(
                [surface_dataset[surface_index] for _, surface_index, _ in batch_pairs]
            )
            inputs = preprocess_batch(
                pipeline, touch_batch["image"], touch_batch["pointmap"]
            )

            touch_mask = touch_batch["touch_mask"].to(device)
            surface_mask = surface_batch["touch_mask"].to(device)
            touch = normalize_touch_to_pointmap_frame(
                touch_batch["touch_xyz"].to(device),
                touch_mask,
                inputs,
                pipeline.ss_preprocessor,
            )
            surface = normalize_touch_to_pointmap_frame(
                surface_batch["touch_xyz"].to(device),
                surface_mask,
                inputs,
                pipeline.ss_preprocessor,
            )
            joint, joint_mask = combine_pointmap_and_touch(inputs, touch, touch_mask)

            sources = {
                "full_surface": (surface, surface_mask),
                "touch": (touch, touch_mask),
                "joint": (joint, joint_mask),
            }
            prepared_sources = {}
            for source, (points, mask) in sources.items():
                prepared, prepared_mask, _, _ = touch_encoder.prepare_points(points, mask)
                prepared_sources[source] = (prepared, prepared_mask)
                with amp(device, args.precision):
                    code = model.encode(prepared, prepared_mask)["x"]
                codes[source].append(code.float().cpu())

            if checks is None:
                prepared, prepared_mask = prepared_sources["full_surface"]
                if not prepared_mask[0].all():
                    raise ValueError("Full-surface checks require 8192 valid points")
                with amp(device, args.precision):
                    masked = model.encode(prepared[:1], prepared_mask[:1])["x"]
                    unmasked = model.encode(prepared[:1], None)["x"]
                    permutation = torch.randperm(
                        prepared.shape[1], device=device
                    )
                    permuted = model.encode(prepared[:1, permutation], None)["x"]
                checks = {
                    "all_true_mask_vs_none": difference(masked, unmasked),
                    "point_permutation": difference(unmasked, permuted),
                }

            targets.append(touch_batch["target_shape"].float())
            object_ids.extend(record["object_id"] for _, _, record in batch_pairs)
            sample_ids.extend(record["sample_id"] for _, _, record in batch_pairs)
            complete = min(begin + args.batch_size, len(pairs))
            print(f"[{complete}/{len(pairs)}] encoded", flush=True)

    codes = {source: torch.cat(values) for source, values in codes.items()}
    targets = torch.cat(targets)
    report = {
        "settings": {
            "split": args.split,
            "objects": len(set(object_ids)),
            "objects_limit": args.objects,
            "views_per_object_limit": args.views_per_object,
            "samples": len(pairs),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "precision": args.precision,
            "checkpoint": "TouchEncoder ENCODERS['vecsetx']",
            "representation": "VecSetX encoder.encode()['x']",
            "distance": "cosine distance after flattening each representation",
        },
        "implementation_checks": checks,
        "view_consistency": {
            source: view_metrics(source_codes, object_ids)
            for source, source_codes in codes.items()
        },
        "source_alignment": source_alignment(codes, object_ids),
        "target_latent_alignment": target_alignment(
            codes,
            targets,
            object_ids,
            args.shuffle_trials,
            args.seed,
        ),
        "samples": [
            {"sample_id": sample_id, "object_id": object_id}
            for sample_id, object_id in zip(sample_ids, object_ids)
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w") as file:
        json.dump(report, file, indent=2)
    printed = {key: value for key, value in report.items() if key != "samples"}
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
