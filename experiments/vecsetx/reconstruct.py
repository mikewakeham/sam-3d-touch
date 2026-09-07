import argparse
import copy
import json
import math
import random
from pathlib import Path

import mcubes
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

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
        default=Path("experiments/vecsetx/outputs/reconstruction"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-id")
    parser.add_argument("--objects", type=int, default=0, help="0 uses every object")
    parser.add_argument(
        "--views-per-object", type=int, default=0, help="0 uses every view"
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--metric-points", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    return parser.parse_args()


def make_dataset(config_path, split):
    config = copy.deepcopy(load_data_config(config_path))
    config["dataset"]["split"] = split
    return TouchDataset(config, include_touch=True)


def select_pairs(
    touch_dataset, surface_dataset, object_count, views_per_object, seed
):
    surface_indices = {
        record["sample_id"]: index
        for index, record in enumerate(surface_dataset.records)
    }
    by_object = {}
    for touch_index, record in enumerate(touch_dataset.records):
        if record["sample_id"] in surface_indices:
            by_object.setdefault(record["object_id"], []).append(touch_index)

    object_ids = sorted(by_object)
    random.Random(seed).shuffle(object_ids)
    if object_count and object_count > len(object_ids):
        raise ValueError(
            f"Requested {object_count} objects, but only {len(object_ids)} are paired"
        )
    if object_count:
        object_ids = object_ids[:object_count]

    pairs = []
    for object_id in object_ids:
        indices = sorted(
            by_object[object_id],
            key=lambda index: touch_dataset.records[index]["view_id"],
        )
        if views_per_object:
            indices = indices[:views_per_object]
        for touch_index in indices:
            record = touch_dataset.records[touch_index]
            pairs.append(
                (touch_index, surface_indices[record["sample_id"]], record)
            )
    return pairs


def make_grid(resolution, device):
    # Copied from VecSetX/vecset/infer.py.
    x = np.linspace(-1, 1, resolution + 1)
    y = np.linspace(-1, 1, resolution + 1)
    z = np.linspace(-1, 1, resolution + 1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = np.stack([xv, yv, zv]).astype(np.float32)
    return torch.from_numpy(grid).view(3, -1).transpose(0, 1)[None].to(device)


def encode_and_decode(model, points, point_mask, grid, block_size):
    bottleneck = model.encode(points, point_mask)
    x = model.learn(bottleneck["x"])

    # Copied from VecSetX VecSetAutoEncoder.forward so the mask can be passed to encode.
    if grid.shape[1] > block_size:
        outputs = []
        for block_index in range(math.ceil(grid.shape[1] / block_size)):
            output = model.decode(
                x,
                grid[:, block_index * block_size : (block_index + 1) * block_size],
            ).squeeze(-1)
            outputs.append(output)
        output = torch.cat(outputs, dim=1)
    else:
        output = model.decode(x, grid).squeeze(-1)
    return bottleneck["x"], output


def make_mesh(output, resolution):
    # Copied from VecSetX/vecset/infer.py.
    volume = output.view(resolution + 1, resolution + 1, resolution + 1)
    volume = volume.permute(1, 0, 2).float().cpu().numpy()
    if volume.min() > 0 or volume.max() < 0:
        return None, float(volume.min()), float(volume.max())
    vertices, faces = mcubes.marching_cubes(volume, 0)
    vertices *= 2.0 / resolution
    vertices -= 1.0
    return trimesh.Trimesh(vertices, faces), float(volume.min()), float(volume.max())


def point_errors(mesh, points, point_count, seed):
    reconstructed, _ = trimesh.sample.sample_surface(mesh, point_count, seed=seed)
    if isinstance(points, torch.Tensor):
        points = points.detach().float().cpu().numpy()
    return cKDTree(reconstructed).query(points)[0]


def reconstruction_metrics(mesh, points, point_count, seed, symmetric=False):
    if isinstance(points, torch.Tensor):
        points = points.detach().float().cpu().numpy()
    input_to_reconstruction = point_errors(mesh, points, point_count, seed)
    metrics = {
        "input_to_reconstruction_mean": float(input_to_reconstruction.mean()),
        "input_to_reconstruction_p95": float(
            np.quantile(input_to_reconstruction, 0.95)
        ),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
    }
    if symmetric:
        reconstructed, _ = trimesh.sample.sample_surface(
            mesh, point_count, seed=seed
        )
        input_tree = cKDTree(points)
        reconstruction_to_input = input_tree.query(reconstructed)[0]
        metrics.update({
            "reconstruction_to_input_mean": float(reconstruction_to_input.mean()),
            "reconstruction_to_input_p95": float(
                np.quantile(reconstruction_to_input, 0.95)
            ),
            "chamfer_l2": float(
                np.square(input_to_reconstruction).mean()
                + np.square(reconstruction_to_input).mean()
            ),
        })
    return metrics


def relative_error(first, second):
    difference = (first.float() - second.float()).flatten()
    return float(difference.norm() / first.float().flatten().norm().clamp_min(1e-12))


def summarize(results):
    summary = {}
    for source in ("full_surface", "full_surface_unmasked", "touch", "joint"):
        rows = [result[source] for result in results if source in result]
        keys = sorted({key for row in rows for key in row.get("metrics", {})})
        summary[source] = {
            key: float(
                np.mean(
                    [row["metrics"][key] for row in rows if key in row["metrics"]]
                )
            )
            for key in keys
        }
        summary[source]["successful_meshes"] = sum(row["status"] == "ok" for row in rows)
        summary[source]["attempted_meshes"] = len(rows)
    return summary


def main():
    args = parse_args()
    if (
        args.objects < 0
        or args.views_per_object < 0
        or args.resolution < 1
        or args.metric_points < 1
    ):
        raise ValueError("Counts must be non-negative and metric settings must be positive")

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
    if args.sample_id:
        pairs = [pair for pair in pairs if pair[2]["sample_id"] == args.sample_id]
        if not pairs:
            raise ValueError(f"Unknown sample {args.sample_id!r}")

    pipeline = build_stage1_pipeline(args.pipeline_config, device)
    touch_encoder = TouchEncoder(
        encoder_name="vecsetx",
        output_dim=pipeline.backbone.cond_channels,
        trainable=False,
        use_position=False,
    ).to(device).eval()
    model = touch_encoder.encoder
    grid = make_grid(args.resolution, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with torch.inference_mode():
        for number, (touch_index, surface_index, record) in enumerate(pairs, 1):
            touch_batch = collate_touch_batch([touch_dataset[touch_index]])
            surface_batch = collate_touch_batch([surface_dataset[surface_index]])
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

            sample_result = {
                "sample_id": record["sample_id"],
                "object_id": record["object_id"],
                "view_id": record["view_id"],
            }
            full_masked_code = None
            full_masked_metrics = None
            for source_name, (source, source_mask) in sources.items():
                prepared, prepared_mask, _, _ = touch_encoder.prepare_points(
                    source, source_mask
                )
                input_points = prepared[0, prepared_mask[0]]
                points_path = (
                    args.output_dir / f"{record['sample_id']}_{source_name}_points.npy"
                )
                np.save(points_path, input_points.float().cpu().numpy())

                with amp(device, args.precision):
                    code, output = encode_and_decode(
                        model, prepared, prepared_mask, grid, args.block_size
                    )
                mesh, sdf_min, sdf_max = make_mesh(output[0], args.resolution)
                row = {
                    "status": "ok" if mesh is not None else "no_zero_crossing",
                    "valid_input_points": int(prepared_mask.sum()),
                    "points": str(points_path),
                    "sdf_min": sdf_min,
                    "sdf_max": sdf_max,
                }
                if mesh is not None:
                    mesh_path = args.output_dir / f"{record['sample_id']}_{source_name}.obj"
                    mesh.export(mesh_path)
                    row["mesh"] = str(mesh_path)
                    row["metrics"] = reconstruction_metrics(
                        mesh,
                        input_points,
                        args.metric_points,
                        args.seed,
                        symmetric=source_name == "full_surface",
                    )
                sample_result[source_name] = row

                if source_name == "full_surface":
                    if not prepared_mask.all():
                        raise ValueError("Full-surface mask parity requires 8192 valid points")
                    full_masked_code = code
                    full_masked_metrics = row.get("metrics")
                    with amp(device, args.precision):
                        unmasked_code, unmasked_output = encode_and_decode(
                            model, prepared, None, grid, args.block_size
                        )
                    unmasked_mesh, sdf_min, sdf_max = make_mesh(
                        unmasked_output[0], args.resolution
                    )
                    unmasked_row = {
                        "status": "ok" if unmasked_mesh is not None else "no_zero_crossing",
                        "valid_input_points": int(prepared_mask.sum()),
                        "points": str(points_path),
                        "sdf_min": sdf_min,
                        "sdf_max": sdf_max,
                    }
                    if unmasked_mesh is not None:
                        mesh_path = (
                            args.output_dir
                            / f"{record['sample_id']}_full_surface_unmasked.obj"
                        )
                        unmasked_mesh.export(mesh_path)
                        unmasked_row["mesh"] = str(mesh_path)
                        unmasked_row["metrics"] = reconstruction_metrics(
                            unmasked_mesh,
                            input_points,
                            args.metric_points,
                            args.seed,
                            symmetric=True,
                        )
                    sample_result["full_surface_unmasked"] = unmasked_row
                    sample_result["mask_parity"] = {
                        "bottleneck_mean_absolute_difference": float(
                            (full_masked_code.float() - unmasked_code.float()).abs().mean()
                        ),
                        "bottleneck_max_absolute_difference": float(
                            (full_masked_code.float() - unmasked_code.float()).abs().max()
                        ),
                        "bottleneck_relative_l2_difference": relative_error(
                            full_masked_code, unmasked_code
                        ),
                    }
                    if full_masked_metrics and unmasked_row.get("metrics"):
                        sample_result["mask_parity"]["chamfer_l2_difference"] = abs(
                            full_masked_metrics["chamfer_l2"]
                            - unmasked_row["metrics"]["chamfer_l2"]
                        )

            results.append(sample_result)
            print(f"[{number}/{len(pairs)}] {record['sample_id']}", flush=True)

    report = {
        "settings": {
            "split": args.split,
            "objects": len({record["object_id"] for _, _, record in pairs}),
            "views": len(pairs),
            "objects_limit": args.objects,
            "views_per_object_limit": args.views_per_object,
            "resolution": args.resolution,
            "metric_points": args.metric_points,
            "seed": args.seed,
            "precision": args.precision,
            "checkpoint": "TouchEncoder ENCODERS['vecsetx']",
        },
        "summary": summarize(results),
        "mask_parity": {
            key: float(
                np.mean(
                    [
                        result["mask_parity"][key]
                        for result in results
                        if key in result["mask_parity"]
                    ]
                )
            )
            for key in sorted(
                {key for result in results for key in result["mask_parity"]}
            )
        },
        "samples": results,
    }
    with open(args.output_dir / "metrics.json", "w") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
