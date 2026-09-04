import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch

from dataloader import build_dataloader, load_data_config
from train import (
    TouchTrainingModel,
    amp,
    build_optimizer,
    build_stage1_pipeline,
    load_trainable_state_dict,
    prepare_batch,
)

PARAMETER_PATTERNS = {
    "vecsetx_latents": "touch_encoder.encoder.latents.",
    "vecsetx_point_embed": "touch_encoder.encoder.point_embed.",
    "vecsetx_cross_attention": "touch_encoder.encoder.cross_attend_blocks.",
    "vecsetx_bottleneck": ("touch_encoder.encoder.bottleneck.pre_bottleneck_proj."),
    "touch_projection": "touch_encoder.output_projection.",
    "touch_position_projection": "touch_encoder.position_projection.",
    "touch_embedding": "touch_encoder.touch_embedding",
    "shape_cross_attention_kv": ".cross_attn.shape.to_kv.",
}
VECSETX_GROUPS = tuple(PARAMETER_PATTERNS)[:4]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data1.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batches", type=int, default=25)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    return parser.parse_args()


def heading(title):
    print(f"\n{title}")


def tensor_stats(tensor):
    tensor = tensor.detach().float()
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return f"shape={tuple(tensor.shape)} finite=0/{tensor.numel()}"
    return (
        f"shape={tuple(tensor.shape)} mean={finite.mean().item():.6g} "
        f"std={finite.std(unbiased=False).item():.6g} "
        f"rms={finite.square().mean().sqrt().item():.6g} "
        f"min={finite.min().item():.6g} max={finite.max().item():.6g} "
        f"finite={finite.numel()}/{tensor.numel()}"
    )


def in_group(name, group):
    pattern = PARAMETER_PATTERNS[group]
    return pattern in name if pattern.startswith(".") else name.startswith(pattern)


def parameter_group(name):
    return next(
        (group for group in PARAMETER_PATTERNS if in_group(name, group)), "other"
    )


def norm(values):
    return sum(value.detach().float().square().sum().item() for value in values) ** 0.5


def report_parameter_changes(model, checkpoint_state, reference_state):
    heading("Checkpoint parameters")
    for group in PARAMETER_PATTERNS:
        tensors = [
            value
            for name, value in checkpoint_state.items()
            if parameter_group(name) == group
        ]
        print(
            f"{group}: saved_tensors={len(tensors)} saved_parameters={sum(x.numel() for x in tensors):,}"
        )

    heading("Changes from reference initialization")
    current = dict(model.named_parameters())
    for group in (
        *VECSETX_GROUPS,
        "touch_projection",
        "touch_position_projection",
        "touch_embedding",
        "shape_cross_attention_kv",
    ):
        names = [name for name in reference_state if parameter_group(name) == group]
        if not names:
            print(f"{group}: unavailable")
            continue
        difference = norm(
            current[name].detach().cpu() - reference_state[name] for name in names
        )
        reference = norm(reference_state[name] for name in names)
        ratio = difference / reference if reference else float("nan")
        print(f"{group}: change_norm={difference:.6g} relative_change={ratio:.6g}")
    if model.touch_encoder is not None:
        heading("Touch adapter parameters")
        print(f"touch_embedding: {tensor_stats(model.touch_encoder.touch_embedding)}")
        for name, parameter in model.touch_encoder.output_projection.named_parameters():
            print(f"output_projection.{name}: {tensor_stats(parameter)}")
        for name, parameter in model.touch_encoder.position_projection.named_parameters():
            print(f"position_projection.{name}: {tensor_stats(parameter)}")


def report_raw_coordinates(record, dataset, verbose=True):
    # Same reconstruction and visibility test as sam-3d-touch-data/sample_touch.py.
    with np.load(
        dataset.resolve_path(record["touch_path"]), allow_pickle=False
    ) as touch:
        points = []
        labels = []
        for index, (start, end) in enumerate(
            zip(touch["offsets"][:-1], touch["offsets"][1:])
        ):
            start, end = int(start), int(end)
            points.append(
                touch["points_local"][start:end] @ touch["R_camera_from_local"][index].T
                + touch["centers_camera"][index]
            )
            labels.append(touch["point_visibility"][start:end])
        points = np.concatenate(points)
        labels = np.concatenate(labels)
        tolerance = float(json.loads(touch["method_args"].item())["tolerance"])

    depth = np.load(dataset.resolve_path(record["depth_path"]), allow_pickle=False)
    pointmap = np.load(
        dataset.resolve_path(record["pointmap_path"]), allow_pickle=False
    )
    with np.load(
        dataset.resolve_path(record["camera_path"]), allow_pickle=False
    ) as camera:
        intrinsics = camera["K"]

    points_opencv = points * np.array([-1.0, -1.0, 1.0])
    projected = points_opencv @ intrinsics.T
    pixels = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int64)
    height, width = depth.shape
    inside = (
        (points_opencv[:, 2] > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    observed = np.full(len(points), np.nan)
    observed[inside] = depth[pixels[inside, 1], pixels[inside, 0]]
    depth_difference = points_opencv[:, 2] - observed

    visible_error = np.abs(depth_difference[labels == 1])
    hidden_difference = depth_difference[labels == 0]
    visible_median = np.nanmedian(visible_error) if len(visible_error) else np.nan
    hidden_median = (
        np.nanmedian(hidden_difference) if len(hidden_difference) else np.nan
    )
    visible_max = np.nanmax(visible_error) if len(visible_error) else np.nan
    hidden_min = np.nanmin(hidden_difference) if len(hidden_difference) else np.nan
    pointmap_matches_depth = np.allclose(pointmap[..., 2], depth, equal_nan=True)
    passed = (
        (not len(visible_error) or visible_max <= tolerance + 1e-6)
        and (not len(hidden_difference) or hidden_min > tolerance)
        and (len(visible_error) + len(hidden_difference) > 0)
        and pointmap_matches_depth
    )

    if verbose:
        heading("Raw SAM/PyTorch3D camera coordinates")
        print(f"status: {'PASS' if passed else 'FAIL'}")
        print(f"visibility_tolerance: {tolerance:.6g}")
        print(
            f"visible_depth_error: median={visible_median:.6g} "
            f"max={visible_max:.6g}"
        )
        print(
            f"hidden_depth_difference: median={hidden_median:.6g} "
            f"min={hidden_min:.6g}"
        )
        print(f"pointmap_z_matches_depth: {pointmap_matches_depth}")
    return passed


def report_canonical_coordinates(record, dataset, batch, index=0, verbose=True):
    # The target builder voxelizes the normalized object inside [-0.5, 0.5]^3.
    mask = batch["touch_mask"][index].numpy()
    points = batch["touch_xyz"][index, mask].numpy()
    with np.load(
        dataset.resolve_path(record["camera_path"]), allow_pickle=False
    ) as camera:
        transform = np.diag([-1.0, -1.0, 1.0, 1.0]) @ camera["T_camera_from_object"]

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    canonical = (points - translation) @ np.linalg.inv(rotation).T
    minimum = canonical.min(axis=0)
    maximum = canonical.max(axis=0)
    center = (minimum + maximum) / 2
    inside_target_cube = np.abs(canonical).max() <= 0.501
    rotation_error = np.abs(rotation.T @ rotation - np.eye(3)).max()

    passed = inside_target_cube and rotation_error < 1e-5
    if verbose:
        heading("Canonical shape-target coordinates")
        print(f"status: {'PASS' if passed else 'FAIL'}")
        print(f"bounds: min={minimum} max={maximum}")
        print(f"bbox_center: {center}")
        print(f"max_radius: {np.linalg.norm(canonical, axis=1).max():.6g}")
        print(f"rotation_orthogonality_error: {rotation_error:.6g}")
    # TODO: Add an independent point-to-target-mesh distance once there is one
    # authoritative target-mesh loader shared by training and diagnostics.
    return passed


def report_vecsetx_coordinates(touch_encoder, points, point_mask, verbose=True):
    # Same bbox-center/max-radius normalization as VecSetX README and infer.py.
    points, point_mask, _, _ = touch_encoder.prepare_points(points, point_mask)
    passed_count = 0
    for index in range(len(points)):
        sample = points[index, point_mask[index]].detach().float()
        minimum = sample.min(dim=0).values
        maximum = sample.max(dim=0).values
        center = (minimum + maximum) / 2
        centered = sample - center
        radius = centered.norm(dim=1).max()
        expected = centered / radius
        error = (sample - expected).square().mean().sqrt()
        passed = center.norm().item() <= 1e-3 and abs(radius.item() - 1.0) <= 1e-3
        passed_count += passed

        if verbose and index == 0:
            heading("VecSetX pretraining-coordinate normalization")
            print(f"status: {'PASS' if passed else 'FAIL'}")
            print(f"bounds: min={minimum.cpu().numpy()} max={maximum.cpu().numpy()}")
            print(f"bbox_center: {center.cpu().numpy()}")
            print(f"max_radius_about_bbox_center: {radius.item():.6g}")
            print(f"rms_difference_from_vecsetx_normalization: {error.item():.6g}")
    return passed_count


def install_activation_hooks(model):
    activations = {}
    handles = []
    touch_state = {"tokens": 0}

    if model.touch_encoder is not None:

        def vecsetx_hook(_module, inputs):
            activations["vecsetx_tokens"] = tensor_stats(inputs[0])
            if inputs[0].requires_grad:
                inputs[0].register_hook(
                    lambda gradient: activations.__setitem__(
                        "vecsetx_token_gradient", tensor_stats(gradient)
                    )
                )
            else:
                activations["vecsetx_token_gradient"] = "not tracked (encoder frozen)"

        def projection_hook(_module, _inputs, output):
            activations["projected_touch_tokens"] = tensor_stats(output)

        def position_hook(_module, _inputs, output):
            activations["touch_position_embedding"] = tensor_stats(output)

        def touch_hook(_module, _inputs, output):
            touch_state["tokens"] = output.shape[1]
            activations["touch_tokens_with_embedding"] = tensor_stats(output)
            output.register_hook(
                lambda gradient: activations.__setitem__(
                    "touch_token_gradient", tensor_stats(gradient)
                )
            )

        handles.append(
            model.touch_encoder.output_projection.register_forward_pre_hook(
                vecsetx_hook
            )
        )
        handles.append(
            model.touch_encoder.output_projection.register_forward_hook(projection_hook)
        )
        handles.append(
            model.touch_encoder.position_projection.register_forward_hook(position_hook)
        )
        handles.append(model.touch_encoder.register_forward_hook(touch_hook))

    blocks = model.generator.reverse_fn.backbone.blocks
    for index in sorted({0, len(blocks) - 1}):
        block = blocks[index]

        def kv_hook(_module, _inputs, output, block_index=index):
            key, value = output.chunk(2, dim=-1)
            touch_count = touch_state["tokens"]
            image_end = key.shape[1] - touch_count
            activations[f"block_{block_index:02d}_image_key"] = tensor_stats(
                key[:, :image_end]
            )
            activations[f"block_{block_index:02d}_image_value"] = tensor_stats(
                value[:, :image_end]
            )
            if touch_count:
                activations[f"block_{block_index:02d}_touch_key"] = tensor_stats(
                    key[:, image_end:]
                )
                activations[f"block_{block_index:02d}_touch_value"] = tensor_stats(
                    value[:, image_end:]
                )

        handles.append(block.cross_attn["shape"].to_kv.register_forward_hook(kv_hook))

    return activations, handles


def report_gradients(model):
    named_parameters = list(model.named_parameters())
    heading("Gradients")
    for group in PARAMETER_PATTERNS:
        parameters = [
            parameter for name, parameter in named_parameters if in_group(name, group)
        ]
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        parameter_norm = norm(parameters) if parameters else 0.0
        gradient_norm = norm(gradients) if gradients else 0.0
        ratio = gradient_norm / parameter_norm if parameter_norm else float("nan")
        print(
            f"{group}: parameters={sum(x.numel() for x in parameters):,} "
            f"with_gradient={sum(x.numel() for x in gradients):,} "
            f"gradient_norm={gradient_norm:.6g} gradient_parameter_ratio={ratio:.6g}"
        )


def main():
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.batches < 1:
        raise ValueError(
            "batch size and batches must be positive; workers cannot be negative"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "diagnostics.py is single-GPU; run it with python, not torchrun"
        )

    device = torch.device(args.device)
    data_config = load_data_config(args.data_config)
    data_config["dataset"]["split"] = args.split
    seed = int(data_config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    loader = build_dataloader(
        data_config,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        distributed=False,
        include_touch=True,
    )
    records = {record["sample_id"]: record for record in loader.dataset.records}
    batches_to_run = min(args.batches, len(loader))
    examples_to_run = min(args.batch_size * batches_to_run, len(loader.dataset))

    pipeline = build_stage1_pipeline(args.pipeline_config, device)
    touch_encoder = None
    if checkpoint["touch_config"] is not None:
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder

        touch_encoder = TouchEncoder(**checkpoint["touch_config"]).to(device)

    model = TouchTrainingModel(pipeline.ss_generator, touch_encoder)
    build_optimizer(
        touch_encoder,
        pipeline.backbone,
        argparse.Namespace(learning_rate=0.0, cross_attention_learning_rate=1.0),
    )
    reference_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter_group(name) in {*VECSETX_GROUPS, "shape_cross_attention_kv"}
    }
    initial_adapter_path = args.checkpoint.parent / "initial_touch_adapter.pt"
    if initial_adapter_path.exists():
        reference_state.update(
            torch.load(initial_adapter_path, map_location="cpu", weights_only=False)
        )
    load_trainable_state_dict(model, checkpoint["model"])

    heading("Run")
    print(f"checkpoint: {args.checkpoint}")
    print(f"mode: {checkpoint['mode']}")
    print(f"touch_config: {checkpoint['touch_config']}")
    print(f"device: {device} (one process, one GPU)")
    print(f"split: {args.split}")
    print(f"precision: {args.precision}")
    print(f"batch_size: {args.batch_size}")
    print(f"workers: {args.workers}")
    print(f"batches: {batches_to_run}")
    print(f"examples: {examples_to_run}")

    report_parameter_changes(model, checkpoint["model"], reference_state)

    activations, handles = install_activation_hooks(model)
    model.zero_grad(set_to_none=True)
    raw_passes = canonical_passes = vecsetx_passes = 0
    total_loss = 0.0
    example_count = 0
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index == batches_to_run:
                break

            for index, sample_id in enumerate(batch["sample_id"]):
                verbose = batch_index == 0 and index == 0
                record = records[sample_id]
                if verbose:
                    print(f"sample_detail: {sample_id}")
                raw_passes += report_raw_coordinates(
                    record, loader.dataset, verbose=verbose
                )
                canonical_passes += report_canonical_coordinates(
                    record, loader.dataset, batch, index=index, verbose=verbose
                )

            prepared = prepare_batch(
                pipeline,
                batch,
                device,
                args.precision,
                touch_encoder is not None,
                checkpoint["mode"] == "image_touch_joint",
            )
            if touch_encoder is not None:
                vecsetx_passes += report_vecsetx_coordinates(
                    touch_encoder,
                    prepared[-2],
                    prepared[-1],
                    verbose=batch_index == 0,
                )

            batch_size = len(batch["sample_id"])
            with amp(device, args.precision):
                loss = model(*prepared)
            (loss * batch_size / examples_to_run).backward()
            total_loss += loss.item() * batch_size
            example_count += batch_size
    finally:
        for handle in handles:
            handle.remove()

    heading("Coordinate summary")
    print(f"raw_camera: {raw_passes}/{example_count} passed")
    print(f"canonical_target: {canonical_passes}/{example_count} passed")
    if touch_encoder is not None:
        print(
            f"vecsetx_pretraining_normalization: {vecsetx_passes}/{example_count} passed"
        )

    heading("Forward/backward summary")
    print(f"mean_loss: {total_loss / example_count:.6g}")
    print("activations_and_token_gradients: final batch")
    for name, stats in activations.items():
        print(f"{name}: {stats}")
    print("parameter_gradients: mean loss over all examples")
    report_gradients(model)


if __name__ == "__main__":
    main()
