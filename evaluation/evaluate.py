import argparse
import gc
import json
import csv
import hashlib
import os
import random
import re
import shutil
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch
import yaml

from evaluation.geometry import load_dataset_mesh
from evaluation.metrics import (
    normalize_mesh, sample_surface, align_mesh, mesh_metrics, voxelize_points,
    symmetric_distance,
)


def read_run(checkpoint_path, data_config=None, split="val"):
    from dataloader import load_data_config

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with open(checkpoint_path.parent / "config.yaml") as file:
        config = yaml.safe_load(file)
    data = load_data_config(data_config) if data_config else config["data"]
    data["dataset"]["split"] = split
    return checkpoint, config, data


def restore_run(pipeline, checkpoint, device):
    """Call on a freshly loaded pipeline; checkpoints contain adapted weights only."""
    from train import (
        TouchTrainingModel, build_optimizer, disable_pointmap_conditioning,
        disable_visual_conditioning, load_trainable_state_dict, checkpoint_train_scope,
    )

    conditioning = checkpoint.get(
        "conditioning_config", {"no_pointmap": False, "oracle_point_frame": False}
    )
    if conditioning["no_pointmap"] or conditioning.get("no_visual", False):
        fuser = pipeline.ss_condition_embedder
        if fuser is None:
            fuser = pipeline.backbone.condition_embedder
        if conditioning.get("no_visual", False):
            disable_visual_conditioning(fuser)
        else:
            disable_pointmap_conditioning(fuser)

    touch_encoder = None
    if checkpoint["touch_config"] is not None:
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder

        config = dict(checkpoint["touch_config"])
        # Early checkpoints predate the explicit use_position field.
        config.setdefault("use_position", any(
            name.startswith("touch_encoder.position_projection.")
            for name in checkpoint["model"]
        ))
        touch_encoder = TouchEncoder(**config).to(device)

    pipeline.ss_generator.requires_grad_(False)
    model = TouchTrainingModel(pipeline.ss_generator, touch_encoder, **conditioning,
                               **checkpoint.get("training_config", {}))
    build_optimizer(touch_encoder, pipeline.backbone, argparse.Namespace(
        learning_rate=0.0,
        cross_attention_learning_rate=1.0,
        cross_attention_scope=checkpoint.get("cross_attention_scope", "kv"),
        train_scope=checkpoint_train_scope(checkpoint),
    ))
    load_trainable_state_dict(model, checkpoint["model"])
    model.load_constant_touch(checkpoint.get("constant_touch"))
    if touch_encoder is not None:
        touch_encoder.eval()
    return model


def decode_voxels(decoder, latent):
    if not decoder.reshape_input_to_cube:
        latent = decoder.flat_to_cube(latent)
    voxels = decoder(latent) > 0
    return voxels[:, 0]


def parse_args():
    parser = argparse.ArgumentParser()
    runs = parser.add_mutually_exclusive_group(required=True)
    runs.add_argument("--run-dirs", type=Path, nargs="+", help="Evaluate best.pt in each directory")
    runs.add_argument("--checkpoints", type=Path, nargs="+", help="Explicit checkpoints with sibling config.yaml")
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--skip-decoded-gt", action="store_true")
    parser.add_argument("--pipeline-config", type=Path, default=Path("checkpoints/hf/pipeline.yaml"))
    parser.add_argument("--data-config", type=Path, help="Override saved run data configuration")
    parser.add_argument("--selection-data-config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--selection", choices=["random", "hidden"], default="random")
    parser.add_argument("--max-samples", type=int, default=50, help="Objects, one view each; 0 selects all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--metric-workers", type=int, default=0)
    parser.add_argument("--inference-steps", type=int, default=25)
    parser.add_argument("--stage2-inference-steps", type=int, default=25)
    parser.add_argument("--surface-points", type=int, default=1_000_000)
    parser.add_argument("--icp-points", type=int, default=20_000)
    parser.add_argument("--emd-points", type=int, default=2048, help="0 disables optional Sinkhorn distance")
    parser.add_argument("--save-points", type=int, default=8192)
    parser.add_argument("--selection-seed", type=int, default=29)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def stable_seed(seed, value):
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1_000_001)


def selected_touch_indices(data, contact, radius, points_per_contact):
    start, end = data["offsets"][contact:contact + 2]
    point_ids = data["point_ids"][start:end]
    center = np.flatnonzero(point_ids == data["center_point_ids"][contact])[0]
    eligible = np.flatnonzero(data["geodesic_distance"][start:end] <= radius)
    others = eligible[eligible != center]
    priorities = data["keep_priority"][start + others]
    others = others[np.argsort(priorities)[:points_per_contact - 1]]
    return start + np.concatenate(([center], others))


def hidden_fraction(record, dataset, data_config):
    touch = data_config["touch"]
    if touch.get('source') == 'touch_patches':
        from data_generation.general.sample_touch_patches import select_patch_indices
        with np.load(dataset.resolve_path(record['touch_path']), allow_pickle=False) as data:
            indices = select_patch_indices(data, touch['contacts']['count'],
                                           touch['point_sampling']['points_per_contact'])
            return float(np.mean(data['point_visibility'][indices] == 0))
    if touch.get("source") == "full_surface":
        with np.load(dataset.resolve_path(record["full_surface_path"]), allow_pickle=False) as data:
            return float(np.mean(data["point_visibility"] == 0))
    count = touch["contacts"]["count"]
    radius = touch["neighborhood"]["max_geodesic_distance"]
    points_per_contact = touch["point_sampling"]["points_per_contact"]
    with np.load(dataset.resolve_path(record["touch_path"]), allow_pickle=False) as data:
        count = min(count, len(data["offsets"]) - 1)
        indices = [
            selected_touch_indices(data, contact, radius, points_per_contact)
            for contact in range(count)
        ]
        visibility = data["point_visibility"][np.concatenate(indices)]
    return float(np.mean(visibility == 0))


def select_records(dataset, data_config, max_samples, seed, selection):
    from tqdm.auto import tqdm

    records_by_object = {}
    for record in dataset.records:
        records_by_object.setdefault(record["object_id"], []).append(record)

    rng = random.Random(seed)
    object_ids = sorted(records_by_object)
    rng.shuffle(object_ids)
    if max_samples:
        object_ids = object_ids[:max_samples]
    object_ids.sort()

    selected = []
    details = []
    iterator = (
        tqdm(object_ids, desc="choosing most-hidden views", unit="object")
        if selection == "hidden" else object_ids
    )
    for object_id in iterator:
        views = sorted(records_by_object[object_id], key=lambda record: record["sample_id"])
        if selection == "random":
            record = random.Random(stable_seed(seed, object_id)).choice(views)
            score = None
        else:
            scores = [(hidden_fraction(view, dataset, data_config), view) for view in views]
            score, record = max(scores, key=lambda item: item[0])
        selected.append(record)
        item = {
            "sample_id": record["sample_id"],
            "object_id": record["object_id"],
            "view_id": record["view_id"],
        }
        if score is not None:
            item["hidden_fraction"] = score
        details.append(item)
    return selected, details


def build_pipeline(config_path, device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_path = config_path.resolve()
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    config.workspace_dir = str(config_path.parent)
    config.device = device
    config.compile_model = False
    config.decode_formats = ["mesh"]
    config.depth_model = None
    config.slat_decoder_gs_config_path = None
    config.slat_decoder_gs_ckpt_path = None
    config.slat_decoder_gs_4_config_path = None
    config.slat_decoder_gs_4_ckpt_path = None
    pipeline = instantiate(config)
    pipeline.ss_generator = pipeline.models["ss_generator"]
    pipeline.ss_condition_embedder = pipeline.condition_embedders["ss_condition_embedder"]
    pipeline.backbone = pipeline.ss_generator.reverse_fn.backbone
    pipeline.models.eval()
    for embedder in pipeline.condition_embedders.values():
        if embedder is not None:
            embedder.eval()
    return pipeline, config


def preprocess_stage2(pipeline, images):
    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline

    items = [
        InferencePipeline.preprocess_image(pipeline, image.numpy(), pipeline.slat_preprocessor)
        for image in images
    ]
    return {key: torch.cat([item[key] for item in items]) for key in items[0]}


def sample_shape(pipeline, condition_args, condition_kwargs, touch_tokens, inference_steps, device):
    generator = pipeline.ss_generator
    previous_steps = generator.inference_steps
    generator.inference_steps = inference_steps
    latent_shapes = {
        name: (1, mapping.pos_emb.shape[0], mapping.input_layer.in_features)
        for name, mapping in pipeline.backbone.latent_mapping.items()
    }
    condition_kwargs = dict(condition_kwargs)
    if touch_tokens is not None:
        condition_kwargs["touch_tokens"] = touch_tokens
    try:
        result = generator(latent_shapes, device, *condition_args, **condition_kwargs)
    finally:
        generator.inference_steps = previous_steps
    return result["shape"]


def trimesh_from_result(result):
    import trimesh

    return trimesh.Trimesh(
        vertices=result.vertices.detach().float().cpu().numpy(),
        faces=result.faces.detach().cpu().numpy(),
        process=False,
    )


def load_target_mesh(record, dataset, output_dir, surface_points, icp_points, save_points, seed):
    import trimesh

    object_id = record["object_id"]
    mesh, dataset_transform = load_dataset_mesh(record, dataset)
    mesh, normalization = normalize_mesh(mesh)

    target_dir = output_dir / "targets" / safe_name(object_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = target_dir / "mesh_normalized.ply"
    mesh.export(mesh_path)
    icp_points_array, _ = sample_surface(mesh, icp_points, seed + 1)
    points, normals = sample_surface(mesh, surface_points, seed + 2)
    saved = min(save_points, len(points))
    points_path = target_dir / "points.npz"
    np.savez_compressed(
        points_path,
        points=points[:saved],
        normals=normals[:saved],
        dataset_transform=dataset_transform,
        evaluation_normalization=normalization,
        icp_seed=np.int64(seed + 1),
        surface_seed=np.int64(seed + 2),
    )
    return {
        "mesh": mesh,
        "icp_points": icp_points_array,
        "points": points,
        "normals": normals,
        "mesh_path": mesh_path,
        "points_path": points_path,
    }


def save_generated_artifacts(output_dir, condition, sample_id, prediction, predicted_voxels,
                             target_voxels, coords_original, coords, downsample_factor, slat,
                             raw_mesh, normalized_mesh, aligned_mesh, normalization, alignment,
                             icp_error, icp_fitness, icp_inlier_rmse, points, normals,
                             touch_centers, seed, save_points, joint_input=None):
    artifact_dir = output_dir / "artifacts" / safe_name(condition) / safe_name(sample_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = artifact_dir / "stage1.npz"
    slat_path = artifact_dir / "slat.npz"
    raw_path = artifact_dir / "mesh_raw.ply"
    normalized_path = artifact_dir / "mesh_normalized.ply"
    aligned_path = artifact_dir / "mesh_aligned.ply"
    alignment_path = artifact_dir / "alignment.npz"

    factor = downsample_factor.item() if torch.is_tensor(downsample_factor) else downsample_factor
    np.savez_compressed(
        stage1_path,
        latent=prediction.detach().float().cpu().numpy(),
        prediction=predicted_voxels.cpu().numpy(),
        target=target_voxels.cpu().numpy(),
        coords_original=coords_original.cpu().numpy(),
        coords=coords.cpu().numpy(),
        downsample_factor=np.asarray(factor),
        touch_centers=touch_centers,
        **(joint_input or {}),
    )
    np.savez_compressed(
        slat_path,
        coords=slat.coords.detach().cpu().numpy(),
        feats=slat.feats.detach().float().cpu().numpy(),
    )
    raw_mesh.export(raw_path)
    normalized_mesh.export(normalized_path)
    aligned_mesh.export(aligned_path)
    np.savez_compressed(
        alignment_path,
        prediction_normalization=normalization,
        icp_transform=alignment,
        icp_error=np.float64(icp_error),
        icp_fitness=np.float64(icp_fitness),
        icp_inlier_rmse=np.float64(icp_inlier_rmse),
        points=points[:save_points],
        normals=normals[:save_points],
        stage1_seed=np.int64(seed),
        stage2_seed=np.int64(seed + 1_000_000),
        icp_seed=np.int64(seed + 1),
        surface_seed=np.int64(seed + 2),
    )
    return stage1_path, slat_path, raw_path, normalized_path, aligned_path, alignment_path


def aligned_stage1_points(row, output_dir):
    with np.load(output_dir / row["stage1_path"], allow_pickle=False) as data:
        factor = int(data["downsample_factor"])
        if factor == 1:
            grid = data["prediction"]
            points = np.argwhere(grid).astype(np.float64) / np.asarray(grid.shape) - 0.5
        else:
            points = data["coords"][:, 1:].astype(np.float64) / 64 - 0.5
    with np.load(output_dir / row["alignment_path"], allow_pickle=False) as data:
        transform = data["icp_transform"] @ data["prediction_normalization"]
    import trimesh

    return trimesh.transform_points(points, transform), factor


def add_stage1_metrics(rows, output_dir, workers):
    by_sample = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], {})[row["condition"]] = row

    for conditions in by_sample.values():
        ground_truth = conditions.get("decoded_gt")
        if ground_truth is None or ground_truth["error"]:
            for row in conditions.values():
                row["stage1_aligned_chamfer"] = np.nan
                row["stage1_aligned_voxel_iou_64"] = np.nan
                row["stage1_downsample_factor"] = np.nan
            continue
        target_points, _ = aligned_stage1_points(ground_truth, output_dir)
        target_voxels = voxelize_points(target_points)
        for row in conditions.values():
            if row["error"]:
                row["stage1_aligned_chamfer"] = np.nan
                row["stage1_aligned_voxel_iou_64"] = np.nan
                row["stage1_downsample_factor"] = np.nan
                continue
            points, factor = aligned_stage1_points(row, output_dir)
            prediction_voxels = voxelize_points(points)
            intersection = np.logical_and(prediction_voxels, target_voxels).sum()
            union = np.logical_or(prediction_voxels, target_voxels).sum()
            row["stage1_aligned_chamfer"] = symmetric_distance(points, target_points, workers)
            row["stage1_aligned_voxel_iou_64"] = float(intersection / max(union, 1))
            row["stage1_downsample_factor"] = factor


def relative(path, root):
    return str(path.relative_to(root))


def load_metrics(path):
    if not path.exists():
        return []
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def append_metric(path, row):
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_metrics(path, rows):
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_condition(name, pipeline, encoder, loader, records, target_cache,
                       completed, metrics_path, args, joint_pointmap=False,
                       oracle_point_frame=False, use_gt_latent=False, touch_token_fn=None,
                       shared_pointmap_normalization=False, fixed_joint_patches=False):
    from train import prepare_batch
    from sam3d_objects.pipeline.inference_utils import downsample_sparse_structure, prune_sparse_structure

    rows = []
    dtype = pipeline.shape_model_dtype
    remaining = sum(
        (name, record["sample_id"]) not in completed for record in loader.dataset.records
    )
    print(
        f"evaluating {name}: {remaining} new, "
        f"{len(loader.dataset) - remaining} already complete",
        flush=True,
    )

    with torch.inference_mode():
        progress = 0
        for batch in loader:
            sample_id = batch["sample_id"][0]
            key = (name, sample_id)
            if key in completed:
                continue
            record = records[sample_id]
            seed = stable_seed(args.seed, sample_id)
            try:
                touch_xyz = touch_mask = None
                if not use_gt_latent:
                    _, condition_args, condition_kwargs, touch_xyz, touch_mask, inputs = prepare_batch(
                        pipeline, batch, torch.device(args.device),
                        "fp32" if args.no_amp else "bf16", encoder is not None,
                        joint_pointmap, oracle_point_frame,
                        shared_pointmap_normalization,
                        use_normals=bool(encoder is not None and getattr(encoder, "requires_normals", False)),
                        return_inputs=True,
                        fixed_joint_patches=fixed_joint_patches,
                    )
                stage2_inputs = preprocess_stage2(pipeline, batch["image"])
                torch.manual_seed(seed)
                with torch.autocast(
                    device_type=torch.device(args.device).type,
                    dtype=dtype,
                    enabled=not args.no_amp and torch.device(args.device).type == "cuda",
                ):
                    encode_touch = touch_token_fn if touch_token_fn is not None else encoder
                    touch_tokens = encode_touch(touch_xyz, touch_mask) if touch_xyz is not None else None
                    prediction = batch["target_shape"].to(args.device) if use_gt_latent else sample_shape(
                        pipeline, condition_args, condition_kwargs, touch_tokens,
                        args.inference_steps, args.device,
                    )
                    predicted_voxels = decode_voxels(pipeline.models["ss_decoder"], prediction)
                    target_voxels = decode_voxels(
                        pipeline.models["ss_decoder"], batch["target_shape"].to(args.device)
                    )

                touch_centers = (
                    touch_xyz[0, touch_mask[0], :3].cpu().numpy()
                    if touch_xyz is not None else np.empty((0, 3), dtype=np.float32)
                )
                joint_input = None
                if joint_pointmap and touch_xyz is not None:
                    from evaluation.input_visualizations import joint_camera_points
                    camera_points = joint_camera_points(touch_xyz[0, touch_mask[0], :3], inputs,
                                                        pipeline.ss_preprocessor)
                    source_indices = np.arange(int(batch['touch_mask'][0].sum()))
                    if fixed_joint_patches and 'touch_patch_count' in batch:
                        from data_generation.general.sample_touch_patches import joint_touch_indices as touch_indices
                        source_indices = touch_indices(int(batch['touch_patch_count'][0]))
                    touch_count = len(source_indices)
                    if not np.allclose(camera_points[-touch_count:],
                                       batch['touch_xyz'][0, batch['touch_mask'][0]].numpy()[source_indices],
                                       atol=1e-5, rtol=1e-5):
                        raise ValueError('Joint visualization does not round-trip to camera-space touch points')
                    joint_input = dict(encoder_input_camera=camera_points, touch_count=np.int64(touch_count),
                                       touch_source_indices=source_indices)

                coords_original = torch.argwhere(predicted_voxels).int()
                coords = coords_original
                if pipeline.downsample_ss_dist > 0:
                    coords = prune_sparse_structure(coords, pipeline.downsample_ss_dist)
                coords, downsample_factor = downsample_sparse_structure(coords)

                torch.manual_seed(seed + 1_000_000)
                slat = pipeline.sample_slat(
                    stage2_inputs, coords,
                    inference_steps=args.stage2_inference_steps,
                    use_distillation=False,
                )
                result = pipeline.decode_slat(slat, ["mesh"])["mesh"][0]
                if not result.success:
                    raise RuntimeError("Stage-2 mesh decoder returned an empty mesh")

                raw_mesh = trimesh_from_result(result)
                normalized_mesh, normalization = normalize_mesh(raw_mesh)
                prediction_icp_points, _ = sample_surface(normalized_mesh, args.icp_points, seed + 1)
                target = target_cache[record["object_id"]]
                aligned_mesh, alignment, icp_error, icp_fitness, icp_inlier_rmse = align_mesh(
                    normalized_mesh, target["mesh"], prediction_icp_points,
                    target["icp_points"], args.metric_workers,
                )
                points, normals = sample_surface(aligned_mesh, args.surface_points, seed + 2)
                metrics = mesh_metrics(
                    points, normals, target["points"], target["normals"],
                    args.emd_points, args.device, args.metric_workers,
                )

                paths = save_generated_artifacts(
                    args.output_dir, name, sample_id, prediction, predicted_voxels[0],
                    target_voxels[0], coords_original, coords, downsample_factor, slat,
                    raw_mesh, normalized_mesh, aligned_mesh, normalization, alignment,
                    icp_error, icp_fitness, icp_inlier_rmse, points, normals,
                    touch_centers, seed, args.save_points, joint_input=joint_input,
                )
                stage1_path, slat_path, raw_path, normalized_path, aligned_path, alignment_path = paths
                row = {
                    "condition": name,
                    "sample_id": sample_id,
                    "object_id": record["object_id"],
                    "view_id": record["view_id"],
                    "image_path": str(loader.dataset.resolve_path(record["image_path"])),
                    "stage1_path": relative(stage1_path, args.output_dir),
                    "slat_path": relative(slat_path, args.output_dir),
                    "mesh_raw_path": relative(raw_path, args.output_dir),
                    "mesh_normalized_path": relative(normalized_path, args.output_dir),
                    "mesh_aligned_path": relative(aligned_path, args.output_dir),
                    "alignment_path": relative(alignment_path, args.output_dir),
                    "target_mesh_path": relative(target["mesh_path"], args.output_dir),
                    "target_points_path": relative(target["points_path"], args.output_dir),
                    **metrics,
                    "icp_error": icp_error,
                    "icp_fitness": icp_fitness,
                    "icp_inlier_rmse": icp_inlier_rmse,
                    "error": "",
                }
            except Exception as error:
                row = {
                    "condition": name,
                    "sample_id": sample_id,
                    "object_id": record["object_id"],
                    "view_id": record["view_id"],
                    "image_path": str(loader.dataset.resolve_path(record["image_path"])),
                    "stage1_path": "",
                    "slat_path": "",
                    "mesh_raw_path": "",
                    "mesh_normalized_path": "",
                    "mesh_aligned_path": "",
                    "alignment_path": "",
                    "target_mesh_path": relative(target_cache[record["object_id"]]["mesh_path"], args.output_dir),
                    "target_points_path": relative(target_cache[record["object_id"]]["points_path"], args.output_dir),
                    "fscore_0.01": np.nan,
                    "f_precision_0.01": np.nan,
                    "f_recall_0.01": np.nan,
                    "voxel_iou_64": np.nan,
                    "chamfer": np.nan,
                    "normal_consistency": np.nan,
                    "emd": np.nan,
                    "icp_error": np.nan,
                    "icp_fitness": np.nan,
                    "icp_inlier_rmse": np.nan,
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"{name}: {sample_id} failed: {row['error']}", flush=True)
            rows.append(row)
            append_metric(metrics_path, row)
            if not row["error"]:
                completed.add(key)
            progress += 1
            print(f"{name}: {progress}/{remaining} new samples", flush=True)
    return rows


def summarize(rows, primary_conditions, diagnostic_conditions, no_touch, best_touch):
    metrics = [
        "fscore_0.01", "f_precision_0.01", "f_recall_0.01", "voxel_iou_64",
        "chamfer", "normal_consistency", "emd", "icp_error", "icp_fitness",
        "icp_inlier_rmse", "stage1_aligned_chamfer", "stage1_aligned_voxel_iou_64",
    ]
    higher_is_better = {
        "fscore_0.01", "f_precision_0.01", "f_recall_0.01", "voxel_iou_64",
        "normal_consistency", "icp_fitness", "stage1_aligned_voxel_iou_64",
    }
    summary = {
        "primary_metric": "fscore_0.01",
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
            "completed": sum(not row["error"] for row in samples.values()),
            "failed": sum(bool(row["error"]) for row in samples.values()),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in samples.values() if not row["error"]], dtype=float)
            values = values[np.isfinite(values)]
            summary["conditions"][condition][metric] = {
                "mean": float(values.mean()) if len(values) else float("nan"),
                "median": float(np.median(values)) if len(values) else float("nan"),
            }

    baselines = (["official"] if "official" in by_condition else []) + ([no_touch] if no_touch else [])
    for condition in primary_conditions + diagnostic_conditions:
        for baseline in baselines:
            if condition == baseline:
                continue
            common = sorted(set(by_condition[condition]) & set(by_condition[baseline]))
            comparison = {}
            for metric in metrics:
                pairs = [
                    (float(by_condition[condition][sample][metric]), float(by_condition[baseline][sample][metric]))
                    for sample in common
                ]
                pairs = [(value, base) for value, base in pairs if np.isfinite(value) and np.isfinite(base)]
                if not pairs:
                    continue
                sign = 1 if metric in higher_is_better else -1
                improvements = np.asarray([sign * (value - base) for value, base in pairs])
                comparison[metric] = {
                    "mean_improvement": float(improvements.mean()),
                    "median_improvement": float(np.median(improvements)),
                    "improved_fraction": float(np.mean(improvements > 0)),
                }
            summary["comparisons"][f"{condition}_vs_{baseline}"] = comparison
    return summary


def artifacts_complete(row, output_dir):
    if row.get("error") != "":
        return False
    keys = ("stage1_path", "slat_path", "mesh_raw_path", "mesh_normalized_path",
            "mesh_aligned_path", "alignment_path", "target_mesh_path", "target_points_path")
    return all(row.get(key) and (output_dir / row[key]).is_file()
               and (output_dir / row[key]).stat().st_size > 0 for key in keys)


def validate_resume(args, checkpoints, selection_data, selected, rank):
    def identity(path):
        path = Path(path).resolve()
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    settings = {key: getattr(args, key) for key in (
        "split", "selection", "max_samples", "selection_seed", "seed", "inference_steps",
        "stage2_inference_steps", "surface_points", "icp_points", "emd_points", "save_points", "no_amp",
    )}
    settings["format_version"] = 1
    settings["checkpoints"] = [identity(path) for path in checkpoints]
    settings["run_configs"] = [identity(path.parent / "config.yaml") for path in checkpoints]
    settings["data_override"] = identity(args.data_config) if args.data_config else None
    settings["pipeline"] = identity(args.pipeline_config)
    settings["selection_data"] = selection_data
    dataset = selection_data["dataset"]
    settings["dataset_files"] = [identity(Path(dataset["root"]) / dataset[key])
                                 for key in ("manifest", "split_file") if dataset.get(key)]
    settings["samples"] = [record["sample_id"] for record in selected]
    path = args.output_dir / "evaluation_settings.json"
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError("Evaluation inputs/settings changed. Use a new --output-dir to avoid mixing results.")
    elif any(args.output_dir.glob("metrics*.csv")):
        raise ValueError("Existing results have no resume settings. Use a new --output-dir; old meshes can still be re-scored/viewed.")
    if rank == 0:
        temporary = path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(settings, indent=2) + "\n")
        temporary.replace(path)


def main():
    args = parse_args()
    from omegaconf import OmegaConf
    from dataloader import build_dataloader, load_data_config
    from train import checkpoint_train_scope, configure_encoder_data

    checkpoints = args.checkpoints or [directory / "best.pt" for directory in args.run_dirs]
    names = [safe_name(path.parent.name) for path in checkpoints]
    if len(set(names)) != len(names) or set(names) & {"official", "decoded_gt"}:
        raise ValueError("Checkpoint parent directory names must be unique and not official/decoded_gt")
    if min(args.surface_points, args.icp_points, args.save_points) < 1 or args.emd_points < 0 or args.max_samples < 0:
        raise ValueError("Invalid sample counts")
    if args.batch_size != 1:
        raise ValueError("Full Stage-2 mesh evaluation currently requires --batch-size 1")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = 0
    if distributed:
        if not torch.cuda.is_available():
            raise ValueError("Distributed evaluation requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        # Independent samples can take very different times before the final merge.
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=12))
        rank = torch.distributed.get_rank()
    if args.metric_workers <= 0:
        cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        args.metric_workers = max(1, cpu_count // world_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _, _, data_config = read_run(checkpoints[0], args.data_config, args.split)
    selection_data = (
        load_data_config(args.selection_data_config)
        if args.selection_data_config else data_config
    )
    selection_data["dataset"]["split"] = args.split
    loader = build_dataloader(selection_data, 1, args.workers, shuffle=False, include_touch=False)
    selection_path = args.output_dir / "selected_samples.yaml"
    available_objects = len({record["object_id"] for record in loader.dataset.records})
    selected_objects = (
        min(args.max_samples, available_objects) if args.max_samples else available_objects
    )
    view_selection = "one random view" if args.selection == "random" else "the most-hidden view"
    print(
        f"uniformly selecting {selected_objects} of {available_objects} objects and "
        f"{view_selection} per object with seed {args.selection_seed}", flush=True,
    )
    selected, selection_details = select_records(
        loader.dataset, selection_data, args.max_samples, args.selection_seed, args.selection
    )
    if not selected:
        raise ValueError("No samples selected")
    validate_resume(args, checkpoints, selection_data, selected, rank)
    if distributed:
        torch.distributed.barrier()
    if rank == 0:
        views_dir = args.output_dir / "selected_views"
        views_dir.mkdir(parents=True, exist_ok=True)
        for record, details in zip(selected, selection_details):
            source = loader.dataset.resolve_path(record["image_path"])
            destination = views_dir / f"{safe_name(record['sample_id'])}{source.suffix}"
            shutil.copy2(source, destination)
            details["image_path"] = str(destination.relative_to(args.output_dir))
        with open(selection_path, "w") as file:
            yaml.safe_dump(selection_details, file, sort_keys=False)

    all_sample_ids = [record["sample_id"] for record in selected]
    local_sample_ids = all_sample_ids[rank::world_size]
    selected = selected[rank::world_size]
    loader.dataset.records = selected
    if rank == 0:
        print(
            f"selected {len(all_sample_ids)} views across {world_size} GPU(s)",
            flush=True,
        )
        for index, item in enumerate(selection_details, 1):
            hidden = f" hidden {item['hidden_fraction']:.4f}" if "hidden_fraction" in item else ""
            print(
                f"selected {index}/{len(selection_details)}: {item['sample_id']} "
                f"object {item['object_id']} view {item['view_id']}{hidden}",
                flush=True,
            )
    print(f"rank {rank}: {len(local_sample_ids)} samples", flush=True)

    metrics_path = args.output_dir / "metrics.csv"
    rank_metrics_path = (
        args.output_dir / f"metrics_rank{rank}.csv" if distributed else metrics_path
    )
    rows_by_key = {}
    metric_sources = [metrics_path, *sorted(args.output_dir.glob("metrics_rank*.csv"))]
    for source in metric_sources:
        for row in load_metrics(source):
            rows_by_key[(row["condition"], row["sample_id"])] = row
    completed = {key for key, row in rows_by_key.items() if artifacts_complete(row, args.output_dir)}
    if rows_by_key:
        print(f"resume: found {len(completed)} completed evaluations", flush=True)
    if distributed:
        # Finish reading old rank files before any rank starts writing new results.
        torch.distributed.barrier()

    target_cache = {}
    for index, record in enumerate(selected):
        print(f"preparing target meshes: {index + 1}/{len(selected)}", flush=True)
        target_cache[record["object_id"]] = load_target_mesh(
            record, loader.dataset, args.output_dir,
            args.surface_points, args.icp_points, args.save_points,
            stable_seed(args.seed, f"target:{record['object_id']}"),
        )
    print("target meshes ready", flush=True)

    primary_conditions = []
    diagnostic_conditions = []
    touch_runs = []
    no_touch = None
    run_settings = {}
    include_base = not args.skip_official or not args.skip_decoded_gt
    for checkpoint_path in ([None] if include_base else []) + checkpoints:
        run_dir = checkpoint_path.parent if checkpoint_path is not None else None
        name = "official" if run_dir is None else safe_name(run_dir.name)
        if name in primary_conditions:
            raise ValueError(f"Duplicate condition name: {name}")
        checkpoint = None
        run_data = data_config
        conditioning = {"no_pointmap": False, "oracle_point_frame": False}
        if run_dir is not None:
            checkpoint, _, run_data = read_run(
                checkpoint_path, args.data_config, args.split
            )
            conditioning = checkpoint.get("conditioning_config", conditioning)
            run_settings[name] = {
                "checkpoint": str(checkpoint_path),
                "step": checkpoint["step"],
                "conditioning_config": conditioning,
                "touch_config": checkpoint["touch_config"],
                "mode": checkpoint["mode"],
                "cross_attention_scope": checkpoint.get("cross_attention_scope", "kv"),
                "train_scope": checkpoint_train_scope(checkpoint),
                "data": run_data,
            }
        use_touch = checkpoint is not None and checkpoint["touch_config"] is not None
        if use_touch:
            run_data = configure_encoder_data(run_data, checkpoint["touch_config"]["encoder_name"])
        shared_pointmap_normalization = conditioning.get(
            "shared_pointmap_normalization", False
        )
        loader = build_dataloader(
            run_data, 1, args.workers, shuffle=False,
            include_touch=use_touch or shared_pointmap_normalization,
            oracle_point_frame=conditioning["oracle_point_frame"],
            joint_pointmap=conditioning.get('joint_pointmap_points') == 1024,
        )
        run_records = {record["sample_id"]: record for record in loader.dataset.records}
        loader.dataset.records = [run_records[sample_id] for sample_id in local_sample_ids]
        print(f"loading fresh pipeline for {name}", flush=True)
        pipeline, pipeline_config = build_pipeline(args.pipeline_config, args.device)
        pipeline.ss_generator.no_shortcut = True
        model = restore_run(pipeline, checkpoint, args.device) if checkpoint is not None else None
        encoder = model.touch_encoder if model is not None else None
        if run_dir is not None or not args.skip_official:
            new_rows = evaluate_condition(
                name, pipeline, encoder, loader, run_records, target_cache, completed,
                rank_metrics_path, args,
                joint_pointmap=checkpoint is not None and checkpoint["mode"] == "image_touch_joint",
                oracle_point_frame=conditioning["oracle_point_frame"],
                touch_token_fn=model.get_touch_tokens if model is not None else None,
                shared_pointmap_normalization=shared_pointmap_normalization,
                fixed_joint_patches=conditioning.get('joint_pointmap_points') == 1024,
            )
            for row in new_rows:
                rows_by_key[(row["condition"], row["sample_id"])] = row
            primary_conditions.append(name)
        if run_dir is None and not args.skip_decoded_gt:
            gt_rows = evaluate_condition(
                "decoded_gt", pipeline, None, loader, run_records, target_cache,
                completed, rank_metrics_path, args, use_gt_latent=True,
            )
            for row in gt_rows:
                rows_by_key[(row["condition"], row["sample_id"])] = row
            primary_conditions.append("decoded_gt")
        if encoder is None and run_dir is not None:
            no_touch = name
        elif encoder is not None:
            touch_runs.append((name, run_dir))
        del encoder, model, pipeline, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    active_conditions = set(primary_conditions + diagnostic_conditions)
    local_rows = [
        row for row in rows_by_key.values()
        if row["condition"] in active_conditions and row["sample_id"] in local_sample_ids
    ]
    add_stage1_metrics(local_rows, args.output_dir, args.metric_workers)
    write_metrics(rank_metrics_path, local_rows)

    if distributed:
        torch.distributed.barrier()
        if rank != 0:
            torch.distributed.destroy_process_group()
            return
        rows_by_key = {}
        for source in [metrics_path, *sorted(args.output_dir.glob("metrics_rank*.csv"))]:
            for row in load_metrics(source):
                rows_by_key[(row["condition"], row["sample_id"])] = row

    rows = [
        row for row in rows_by_key.values()
        if row["condition"] in active_conditions and row["sample_id"] in all_sample_ids
    ]
    write_metrics(metrics_path, rows)

    means = {}
    for name, _ in touch_runs:
        values = [float(row["fscore_0.01"]) for row in rows
                  if row["condition"] == name and not row["error"] and np.isfinite(float(row["fscore_0.01"]))]
        if values:
            means[name] = float(np.mean(values))
    best_touch = max(means, key=means.get) if means else None

    summary_rows = [
        row for row in rows
        if row["condition"] in active_conditions and row["sample_id"] in all_sample_ids
    ]
    summary = summarize(
        summary_rows, primary_conditions, diagnostic_conditions, no_touch, best_touch
    )
    with open(args.output_dir / "summary.yaml", "w") as file:
        yaml.safe_dump(summary, file, sort_keys=False)
    with open(args.output_dir / "config.yaml", "w") as file:
        yaml.safe_dump({
            "arguments": {
                key: [str(path) for path in value] if key in ("run_dirs", "checkpoints") and value else str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "selected_objects": len(all_sample_ids),
            "selection": {
                "one_view_per_object": True,
                "mode": args.selection,
                "method": "uniform objects without replacement, then " + (
                    "one uniform view per object" if args.selection == "random"
                    else "the view with the largest hidden fraction per object"
                ),
                "seed": args.selection_seed,
            },
            "mesh_protocol": {
                "normalization": "each mesh independently centered and longest bounding-box extent scaled to 2",
                "alignment": "PCA 24-orientation similarity initialization; best four refined by point-to-point ICP",
                "icp_threshold": 0.2,
                "icp_iterations": 50,
                "stage1_alignment": "each Stage-1 support uses its own Stage-2 mesh registration into the source-mesh frame",
                "stage1_downsampling": "full decoded occupancy when factor is 1; exact Stage-2 support coordinates otherwise",
                "fscore_threshold": 0.01,
                "voxel_resolution": 64,
                "voxel_iou_64": "sampled surface occupancy, not filled volume",
                "chamfer": "symmetric mean Euclidean nearest-neighbor distance",
                "emd": "entropic Sinkhorn approximation of Wasserstein-2; epsilon 0.01; 100 iterations",
                "normal_consistency": "symmetric mean absolute nearest-neighbor normal dot product",
            },
            "pipeline_config": OmegaConf.to_container(pipeline_config, resolve=True),
            "data_config": data_config,
            "selection_data_config": selection_data,
            "runs": run_settings,
        }, file, sort_keys=False)
    print(f"saved evaluation to {args.output_dir}")
    if distributed:
        for path in args.output_dir.glob("metrics_rank*.csv"):
            path.unlink()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
