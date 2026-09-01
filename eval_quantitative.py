import argparse
import csv
import hashlib
import itertools
import os
import random
import re
import shutil
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import open3d as o3d
import torch
import trimesh
import yaml
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from dataloader import build_dataloader, load_data_config
from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
from sam3d_objects.pipeline.inference_utils import (
    downsample_sparse_structure,
    prune_sparse_structure,
)
from train import normalize_touch, preprocess_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--pipeline-config", type=Path, default=Path("checkpoints/hf/pipeline.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--selection", choices=["random", "hidden"], default="random")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--inference-steps", type=int, default=25)
    parser.add_argument("--stage2-inference-steps", type=int, default=25)
    parser.add_argument("--surface-points", type=int, default=1_000_000)
    parser.add_argument("--icp-points", type=int, default=20_000)
    parser.add_argument("--emd-points", type=int, default=2048)
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
    count = touch["contacts"]["count"]
    radius = touch["neighborhood"]["max_geodesic_distance"]
    points_per_contact = touch["point_sampling"]["points_per_contact"]
    with np.load(dataset.path(record["touch_path"]), allow_pickle=False) as data:
        count = min(count, len(data["offsets"]) - 1)
        indices = [
            selected_touch_indices(data, contact, radius, points_per_contact)
            for contact in range(count)
        ]
        visibility = data["point_visibility"][np.concatenate(indices)]
    return float(np.mean(visibility == 0))


def select_records(dataset, data_config, max_samples, seed, selection):
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
    items = [
        InferencePipeline.preprocess_image(pipeline, image.numpy(), pipeline.slat_preprocessor)
        for image in images
    ]
    return {key: torch.cat([item[key] for item in items]) for key in items[0]}


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
        config = checkpoint.get("touch_encoder_config")
        if config is None:
            config = {
                "output_dim": pipeline.backbone.cond_channels,
                "tokens_per_contact": 1,
                "architecture_version": "center_v1",
            }
        else:
            config = dict(config)
            config["output_dim"] = pipeline.backbone.cond_channels
        encoder = TouchEncoder(**config).to(device)
        encoder.load_state_dict(checkpoint["touch_encoder"])
        encoder.eval()
    return encoder


def make_touch_cache(loader, pipeline, device):
    cache = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            inputs = preprocess_batch(pipeline, batch["image"], batch["pointmap"])
            xyz = normalize_touch(batch["touch_xyz"].to(device), inputs, pipeline.ss_preprocessor).cpu()
            for index, sample_id in enumerate(batch["sample_id"]):
                cache[sample_id] = (xyz[index], batch["touch_mask"][index].clone())
            print(f"preparing normalized touch inputs: {batch_index}/{len(loader)}", flush=True)
    return cache


def shuffled_ids(sample_ids, object_ids, seed):
    if len(set(object_ids.values())) < 2:
        return None
    donors = sample_ids.copy()
    rng = random.Random(seed)
    for _ in range(100):
        rng.shuffle(donors)
        if all(object_ids[sample] != object_ids[donor] for sample, donor in zip(sample_ids, donors)):
            return dict(zip(sample_ids, donors))
    return {
        sample: next(donor for donor in donors if object_ids[sample] != object_ids[donor])
        for sample in sample_ids
    }


def trimesh_from_result(result):
    return trimesh.Trimesh(
        vertices=result.vertices.detach().float().cpu().numpy(),
        faces=result.faces.detach().cpu().numpy(),
        process=False,
    )


def normalize_mesh(mesh):
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    scale = 2.0 / (bounds[1] - bounds[0]).max()
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -scale * center
    mesh.apply_transform(transform)
    return mesh, transform


def sample_surface(mesh, count, seed):
    points, faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = np.asarray(mesh.face_normals[faces], dtype=np.float32)
    return points.astype(np.float32), normals


def proper_axis_rotations():
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3))
            rotation[range(3), permutation] = signs
            if np.linalg.det(rotation) > 0:
                rotations.append(rotation)
    return rotations


def principal_basis(points):
    covariance = np.cov(points, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def symmetric_distance(source, target):
    source_to_target = cKDTree(target).query(source, workers=-1)[0].mean()
    target_to_source = cKDTree(source).query(target, workers=-1)[0].mean()
    return float((source_to_target + target_to_source) / 2)


def align_mesh(prediction_mesh, target_mesh, prediction_points, target_points):
    source_basis = principal_basis(prediction_points)
    target_basis = principal_basis(target_points)
    source_center = prediction_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_radius = np.sqrt(np.mean(np.sum((prediction_points - source_center) ** 2, axis=1)))
    target_radius = np.sqrt(np.mean(np.sum((target_points - target_center) ** 2, axis=1)))
    scale = target_radius / source_radius
    candidates = []
    coarse_source = prediction_points[::5]
    coarse_target = target_points[::5]
    for axis_rotation in proper_axis_rotations():
        transform = np.eye(4)
        transform[:3, :3] = scale * target_basis @ axis_rotation @ source_basis.T
        transform[:3, 3] = target_center - transform[:3, :3] @ source_center
        transformed = trimesh.transform_points(coarse_source, transform)
        candidates.append((symmetric_distance(transformed, coarse_target), transform))

    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prediction_points))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
    refined = []
    for _, transform in sorted(candidates, key=lambda item: item[0])[:4]:
        registration = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            0.2,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        transformed = trimesh.transform_points(prediction_points, registration.transformation)
        refined.append((symmetric_distance(transformed, target_points), registration.transformation))

    error, transform = min(refined, key=lambda item: item[0])
    aligned = prediction_mesh.copy()
    aligned.apply_transform(transform)
    return aligned, transform, error


def voxelize_points(points, resolution=64):
    occupancy = np.zeros((resolution, resolution, resolution), dtype=bool)
    coords = np.floor((points + 1) * resolution / 2).astype(np.int64)
    if len(coords):
        coords = coords[np.all((coords >= 0) & (coords < resolution), axis=1)]
        occupancy[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return occupancy


def sinkhorn_emd(prediction, target, device, epsilon=0.01, iterations=100):
    prediction = torch.from_numpy(prediction).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(target).to(device=device, dtype=torch.float32)
    cost = torch.cdist(prediction, target).square()
    log_kernel = -cost / epsilon
    log_mass_prediction = -np.log(len(prediction))
    log_mass_target = -np.log(len(target))
    log_u = torch.zeros(len(prediction), device=device)
    log_v = torch.zeros(len(target), device=device)
    for _ in range(iterations):
        log_u = log_mass_prediction - torch.logsumexp(log_kernel + log_v[None], dim=1)
        log_v = log_mass_target - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    transport = torch.exp(log_kernel + log_u[:, None] + log_v[None])
    return float(torch.sqrt((transport * cost).sum()).cpu())


def mesh_metrics(prediction_points, prediction_normals, target_points, target_normals,
                 emd_points, device):
    target_tree = cKDTree(target_points)
    prediction_tree = cKDTree(prediction_points)
    prediction_distances, prediction_neighbors = target_tree.query(prediction_points, workers=-1)
    target_distances, target_neighbors = prediction_tree.query(target_points, workers=-1)

    threshold = 0.01
    f_precision = float(np.mean(prediction_distances < threshold))
    f_recall = float(np.mean(target_distances < threshold))
    fscore = 2 * f_precision * f_recall / max(f_precision + f_recall, 1e-12)
    normal_forward = np.abs(np.sum(prediction_normals * target_normals[prediction_neighbors], axis=1)).mean()
    normal_backward = np.abs(np.sum(target_normals * prediction_normals[target_neighbors], axis=1)).mean()

    prediction_voxels = voxelize_points(prediction_points)
    target_voxels = voxelize_points(target_points)
    intersection = np.logical_and(prediction_voxels, target_voxels).sum()
    union = np.logical_or(prediction_voxels, target_voxels).sum()
    emd_count = min(emd_points, len(prediction_points), len(target_points))
    return {
        "fscore_0.01": fscore,
        "f_precision_0.01": f_precision,
        "f_recall_0.01": f_recall,
        "voxel_iou_64": float(intersection / max(union, 1)),
        "chamfer": float((prediction_distances.mean() + target_distances.mean()) / 2),
        "normal_consistency": float((normal_forward + normal_backward) / 2),
        "emd": sinkhorn_emd(
            prediction_points[:emd_count], target_points[:emd_count], device
        ),
    }


def load_target_mesh(record, dataset, output_dir, surface_points, icp_points, save_points, seed):
    object_id = record["object_id"]
    mesh_path = dataset.root / "objects" / object_id / "model.obj"
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False, skip_materials=True)
    with np.load(dataset.path(record["object_transform_path"]), allow_pickle=False) as data:
        dataset_transform = data["T_normalized_from_source"].astype(np.float64)
    mesh.apply_transform(dataset_transform)
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
                             icp_error, points, normals, touch_centers, seed, save_points):
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
        points=points[:save_points],
        normals=normals[:save_points],
        stage1_seed=np.int64(seed),
        stage2_seed=np.int64(seed + 1_000_000),
        icp_seed=np.int64(seed + 1),
        surface_seed=np.int64(seed + 2),
    )
    return stage1_path, slat_path, raw_path, normalized_path, aligned_path, alignment_path


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


def evaluate_condition(name, pipeline, encoder, loader, touch_cache, donor_ids,
                       records, target_cache, completed, args, mode="correct"):
    rows = []
    voxel_dir = args.output_dir / "voxels" / safe_name(name)
    voxel_dir.mkdir(parents=True, exist_ok=True)
    dtype = pipeline.shape_model_dtype
    remaining = sum(
        (name, record["sample_id"]) not in completed for record in loader.dataset.records
    )
    print(
        f"evaluating {name} ({mode}): {remaining} new, "
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
            touch_xyz = None
            if encoder is not None and mode != "omitted":
                touch_id = donor_ids[sample_id] if mode == "shuffled" else sample_id
                touch_xyz = touch_cache[touch_id][0][None].to(args.device)
                touch_mask = touch_cache[touch_id][1][None].to(args.device)

            seed = stable_seed(args.seed, sample_id)
            voxel_path = voxel_dir / f"{safe_name(sample_id)}.npz"
            try:
                stage1_inputs = preprocess_batch(pipeline, batch["image"], batch["pointmap"])
                stage2_inputs = preprocess_stage2(pipeline, batch["image"])
                torch.manual_seed(seed)
                with torch.autocast(
                    device_type=torch.device(args.device).type,
                    dtype=dtype,
                    enabled=not args.no_amp and torch.device(args.device).type == "cuda",
                ):
                    condition_args, condition_kwargs = pipeline.get_condition_input(
                        pipeline.ss_condition_embedder,
                        stage1_inputs,
                        pipeline.ss_condition_input_mapping,
                    )
                    touch_tokens = encoder(touch_xyz, touch_mask) if touch_xyz is not None else None
                    prediction = sample_shape(
                        pipeline, condition_args, condition_kwargs, touch_tokens,
                        args.inference_steps, args.device,
                    )
                    predicted_voxels = decode_voxels(pipeline.models["ss_decoder"], prediction)
                    target_voxels = decode_voxels(
                        pipeline.models["ss_decoder"], batch["target_shape"].to(args.device)
                    )

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
                aligned_mesh, alignment, icp_error = align_mesh(
                    normalized_mesh, target["mesh"], prediction_icp_points, target["icp_points"]
                )
                points, normals = sample_surface(aligned_mesh, args.surface_points, seed + 2)
                metrics = mesh_metrics(
                    points, normals, target["points"], target["normals"],
                    args.emd_points, args.device,
                )

                old_metrics = voxel_metrics(predicted_voxels, target_voxels)
                latent_mse = float(
                    (prediction.float() - batch["target_shape"].to(args.device).float()).square().mean()
                )
                touch_centers = touch_cache[sample_id][0][:, 0].numpy()
                np.savez_compressed(
                    voxel_path,
                    prediction=predicted_voxels[0].cpu().numpy(),
                    target=target_voxels[0].cpu().numpy(),
                    touch_centers=touch_centers,
                )
                paths = save_generated_artifacts(
                    args.output_dir, name, sample_id, prediction, predicted_voxels[0],
                    target_voxels[0], coords_original, coords, downsample_factor, slat,
                    raw_mesh, normalized_mesh, aligned_mesh, normalization, alignment,
                    icp_error, points, normals, touch_centers, seed, args.save_points,
                )
                stage1_path, slat_path, raw_path, normalized_path, aligned_path, alignment_path = paths
                row = {
                    "condition": name,
                    "sample_id": sample_id,
                    "object_id": record["object_id"],
                    "view_id": record["view_id"],
                    "image_path": str(loader.dataset.path(record["image_path"])),
                    "voxel_path": relative(voxel_path, args.output_dir),
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
                    "latent_mse": latent_mse,
                    **{metric: float(values[0]) for metric, values in old_metrics.items()},
                    "error": "",
                }
            except Exception as error:
                row = {
                    "condition": name,
                    "sample_id": sample_id,
                    "object_id": record["object_id"],
                    "view_id": record["view_id"],
                    "image_path": str(loader.dataset.path(record["image_path"])),
                    "voxel_path": "",
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
                    "latent_mse": np.nan,
                    "iou": np.nan,
                    "dice": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "predicted_voxels": np.nan,
                    "target_voxels": np.nan,
                    "volume_error": np.nan,
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"{name}: {sample_id} failed: {row['error']}", flush=True)
            rows.append(row)
            append_metric(args.output_dir / "metrics.csv", row)
            if not row["error"]:
                completed.add(key)
            progress += 1
            print(f"{name}: {progress}/{remaining} new samples", flush=True)
    return rows


def summarize(rows, primary_conditions, diagnostic_conditions, no_touch, best_touch):
    metrics = [
        "fscore_0.01", "f_precision_0.01", "f_recall_0.01", "voxel_iou_64",
        "chamfer", "normal_consistency", "emd", "icp_error",
        "iou", "dice", "precision", "recall", "volume_error", "latent_mse",
    ]
    higher_is_better = {
        "fscore_0.01", "f_precision_0.01", "f_recall_0.01", "voxel_iou_64",
        "normal_consistency", "iou", "dice", "precision", "recall",
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
        summary["conditions"][condition] = {"failed": sum(bool(row["error"]) for row in samples.values())}
        for metric in metrics:
            values = np.asarray([row[metric] for row in samples.values()], dtype=float)
            summary["conditions"][condition][metric] = {
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
            }

    baselines = ["official"] + ([no_touch] if no_touch else [])
    for condition in primary_conditions + diagnostic_conditions:
        for baseline in baselines:
            if condition == baseline:
                continue
            common = sorted(set(by_condition[condition]) & set(by_condition[baseline]))
            comparison = {}
            for metric in metrics:
                pairs = [
                    (by_condition[condition][sample][metric], by_condition[baseline][sample][metric])
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


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Full Stage-2 mesh evaluation currently requires --batch-size 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_config = load_data_config(args.data_config)
    data_config["dataset"]["split"] = args.split
    data_config["touch"]["contacts"]["shuffle_after_selection"] = False
    loader = build_dataloader(data_config, 1, args.workers, shuffle=False)
    selection_path = args.output_dir / "selected_samples.yaml"
    previous_samples = set()
    if selection_path.exists():
        with open(selection_path) as file:
            previous_samples = {
                item["sample_id"] for item in (yaml.safe_load(file) or [])
            }
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
        loader.dataset, data_config, args.max_samples, args.selection_seed, args.selection
    )
    loader.dataset.records = selected
    views_dir = args.output_dir / "selected_views"
    views_dir.mkdir(parents=True, exist_ok=True)
    for record, details in zip(selected, selection_details):
        source = loader.dataset.path(record["image_path"])
        destination = views_dir / f"{safe_name(record['sample_id'])}{source.suffix}"
        shutil.copy2(source, destination)
        details["image_path"] = str(destination.relative_to(args.output_dir))
    with open(selection_path, "w") as file:
        yaml.safe_dump(selection_details, file, sort_keys=False)

    records = {record["sample_id"]: record for record in selected}
    object_ids = {sample_id: record["object_id"] for sample_id, record in records.items()}
    sample_ids = list(records)
    print(f"selected {len(selected)} views from {len(set(object_ids.values()))} objects", flush=True)
    for index, item in enumerate(selection_details, 1):
        hidden = f" hidden {item['hidden_fraction']:.4f}" if "hidden_fraction" in item else ""
        print(
            f"selected {index}/{len(selection_details)}: {item['sample_id']} "
            f"object {item['object_id']} view {item['view_id']}{hidden}",
            flush=True,
        )

    metrics_path = args.output_dir / "metrics.csv"
    rows_by_key = {}
    for row in load_metrics(metrics_path):
        rows_by_key[(row["condition"], row["sample_id"])] = row
    completed = {key for key, row in rows_by_key.items() if not row["error"]}
    selection_changed = bool(previous_samples and previous_samples != set(sample_ids))
    if selection_changed:
        completed = {
            key for key in completed
            if not key[0].endswith(("_shuffled", "_omitted"))
        }
    if rows_by_key:
        print(f"resume: found {len(completed)} completed evaluations", flush=True)
    if selection_changed:
        print("resume: sample set changed; diagnostic conditions will be refreshed", flush=True)

    print("loading Stage-1 and Stage-2 evaluation pipeline", flush=True)
    pipeline, pipeline_config = build_pipeline(args.pipeline_config, args.device)
    print("evaluation pipeline loaded", flush=True)
    pipeline.ss_generator.no_shortcut = True
    cross_attention_kv = [block.cross_attn["shape"].to_kv for block in pipeline.backbone.blocks]
    official_kv = [
        {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for module in cross_attention_kv
    ]

    print("preparing normalized touch inputs", flush=True)
    touch_cache = make_touch_cache(loader, pipeline, args.device)
    print("normalized touch inputs ready", flush=True)
    donor_ids = shuffled_ids(sample_ids, object_ids, args.seed + 1)
    target_cache = {}
    for index, record in enumerate(selected):
        print(f"preparing target meshes: {index + 1}/{len(selected)}", flush=True)
        target_cache[record["object_id"]] = load_target_mesh(
            record, loader.dataset, args.output_dir,
            args.surface_points, args.icp_points, args.save_points,
            stable_seed(args.seed, f"target:{record['object_id']}"),
        )
    print("target meshes ready", flush=True)

    primary_conditions = ["official"]
    diagnostic_conditions = []
    restore_official_kv(cross_attention_kv, official_kv)
    new_rows = evaluate_condition(
        "official", pipeline, None, loader, touch_cache, donor_ids,
        records, target_cache, completed, args,
    )
    for row in new_rows:
        rows_by_key[(row["condition"], row["sample_id"])] = row

    touch_runs = []
    no_touch = None
    for run_dir in args.run_dirs:
        name = safe_name(run_dir.name)
        encoder = load_run(run_dir, pipeline, cross_attention_kv, official_kv, args.device)
        new_rows = evaluate_condition(
            name, pipeline, encoder, loader, touch_cache, donor_ids,
            records, target_cache, completed, args,
        )
        for row in new_rows:
            rows_by_key[(row["condition"], row["sample_id"])] = row
        primary_conditions.append(name)
        if encoder is None:
            no_touch = name
        else:
            touch_runs.append((name, run_dir))

    rows = list(rows_by_key.values())
    means = {
        name: np.nanmean([
            float(row["fscore_0.01"]) for row in rows
            if row["condition"] == name and row["sample_id"] in sample_ids
        ])
        for name, _ in touch_runs
    }
    best_touch = max(means, key=means.get) if means else None
    if best_touch is not None:
        best_run = next(run_dir for name, run_dir in touch_runs if name == best_touch)
        encoder = load_run(best_run, pipeline, cross_attention_kv, official_kv, args.device)
        modes = ["omitted"] if donor_ids is None else ["shuffled", "omitted"]
        for mode in modes:
            name = f"{best_touch}_{mode}"
            new_rows = evaluate_condition(
                name, pipeline, encoder, loader, touch_cache, donor_ids,
                records, target_cache, completed, args, mode=mode,
            )
            for row in new_rows:
                rows_by_key[(row["condition"], row["sample_id"])] = row
            diagnostic_conditions.append(name)

    rows = list(rows_by_key.values())
    with open(metrics_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    active_conditions = set(primary_conditions + diagnostic_conditions)
    summary_rows = [
        row for row in rows
        if row["condition"] in active_conditions and row["sample_id"] in sample_ids
    ]
    summary = summarize(
        summary_rows, primary_conditions, diagnostic_conditions, no_touch, best_touch
    )
    with open(args.output_dir / "summary.yaml", "w") as file:
        yaml.safe_dump(summary, file, sort_keys=False)
    with open(args.output_dir / "config.yaml", "w") as file:
        yaml.safe_dump({
            "arguments": {
                key: [str(path) for path in value] if key == "run_dirs" else str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "selected_objects": len(selected),
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
                "fscore_threshold": 0.01,
                "voxel_resolution": 64,
                "chamfer": "symmetric mean Euclidean nearest-neighbor distance",
                "emd": "entropic Sinkhorn approximation of Wasserstein-2; epsilon 0.01; 100 iterations",
                "normal_consistency": "symmetric mean absolute nearest-neighbor normal dot product",
            },
            "pipeline_config": OmegaConf.to_container(pipeline_config, resolve=True),
            "data_config": data_config,
        }, file, sort_keys=False)
    print(f"saved evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
