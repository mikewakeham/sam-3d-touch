import argparse
import copy
from contextlib import contextmanager
import os
import random
import time
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence

from dataloader import build_dataloader, collate_touch_batch, load_data_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=Path("configs/data1.yaml"))
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/touch_stage1"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wandb-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs when --max-steps is not set")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Total optimizer steps, including resumed steps; overrides --epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cross-attention-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--train-scope", choices=["shape_cross_attention", "shape_full"],
        help="Generator weights to train (default: shape_cross_attention)",
    )
    parser.add_argument(
        "--cross-attention-scope", choices=["kv", "full"],
        help="Legacy cross-attention selection; full means shape_cross_attention, not shape_full",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--no-touch", action="store_true")
    parser.add_argument("--no-pointmap", action="store_true")
    parser.add_argument(
        "--shared-pointmap-normalization",
        action="store_true",
        help="Normalize both pointmap branches with the full surface center and radius",
    )
    parser.add_argument("--no-visual", action="store_true",
                        help="Permanently zero image, mask and pointmap conditioning, including validation")
    parser.add_argument("--oracle-point-frame", action="store_true")
    parser.add_argument("--visual-dropout", type=float, default=0.0,
                        help="Per-sample probability of zeroing image and pointmap tokens during training")
    parser.add_argument("--constant-touch", action="store_true",
                        help="Use one fixed training surface's VecSetX features for every example")
    parser.add_argument("--point-encoder", choices=["vecsetx", "craftsman", "triposg"], default="vecsetx")
    parser.add_argument("--point-encoder-checkpoint", type=Path,
                        help="Local official encoder/VAE checkpoint instead of the default pretrained file")
    parser.add_argument("--train-point-encoder", "--train-vecsetx", dest="train_vecsetx", action="store_true")
    parser.add_argument("--point-encoder-from-scratch", "--vecsetx-from-scratch", dest="vecsetx_from_scratch", action="store_true",
                        help="Skip pretrained weights; requires --train-point-encoder")
    parser.add_argument("--vecsetx-learn", action="store_true",
                        help="Use frozen VecSetX decoder features before touch projection")
    parser.add_argument("--joint-pointmap", action="store_true")
    parser.add_argument("--touch-position", dest="no_touch_position", action="store_false",
                        help="Enable touch position embeddings (disabled by default)")
    parser.add_argument("--no-touch-position", action="store_true",
                        help="Disable touch position embeddings (the default; kept for existing jobs)")
    parser.set_defaults(no_touch_position=True)
    parser.add_argument(
        "--local-rank", "--local_rank", type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),
    )
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be nonnegative")
    if args.max_steps:
        args.epochs = None
    elif args.epochs < 1:
        parser.error("--epochs must be positive when --max-steps is not set")
    if args.vecsetx_from_scratch and (not args.train_vecsetx or args.no_touch or args.vecsetx_learn):
        parser.error("--point-encoder-from-scratch requires --train-point-encoder, surface conditioning, and no --vecsetx-learn")
    if args.vecsetx_from_scratch and args.point_encoder_checkpoint:
        parser.error("Random initialization cannot load --point-encoder-checkpoint")
    if args.point_encoder != "vecsetx" and (args.vecsetx_learn or args.joint_pointmap or args.constant_touch or args.no_touch):
        parser.error("CraftsMan/TripoSG require full surfaces with normals; joint pointmap, constant touch, decoder features and --no-touch are unsupported")
    if args.train_scope is not None and args.cross_attention_scope is not None:
        parser.error("Use --train-scope or legacy --cross-attention-scope, not both")
    args.train_scope = resolve_train_scope(args.train_scope, args.cross_attention_scope or "full")
    args.cross_attention_scope = "kv" if args.train_scope == "shape_cross_attention_kv" else "full"
    return args


def resolve_train_scope(train_scope=None, cross_attention_scope="kv"):
    if train_scope is None:
        return {"kv": "shape_cross_attention_kv", "full": "shape_cross_attention"}[cross_attention_scope]
    if train_scope not in ("shape_cross_attention_kv", "shape_cross_attention", "shape_full"):
        raise ValueError(f"Unknown training scope: {train_scope}")
    return train_scope


def checkpoint_train_scope(checkpoint):
    return resolve_train_scope(checkpoint.get("train_scope"), checkpoint.get("cross_attention_scope", "kv"))


def setup_distributed(args):
    if args.local_rank < 0:
        return False, 0, 1, torch.device(args.device)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl")
    return (
        True,
        dist.get_rank(),
        dist.get_world_size(),
        torch.device("cuda", args.local_rank),
    )


def build_stage1_pipeline(config_path, device, no_pointmap=False, no_visual=False):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

    class Stage1TrainingPipeline(InferencePipelinePointMap):
        def __init__(self, config_path, device):
            config_path = Path(config_path).resolve()
            config = OmegaConf.load(config_path)
            self.device = torch.device(device)
            self.workspace_dir = str(config_path.parent)
            self.ss_condition_input_mapping = list(
                config.get("ss_condition_input_mapping", ["image"])
            )

            preprocessor = config.get("ss_preprocessor")
            preprocessor = instantiate(preprocessor) if preprocessor is not None else None
            self.ss_preprocessor = self.init_ss_preprocessor(
                preprocessor, config.ss_generator_config_path
            )
            self.ss_generator = self.init_ss_generator(
                config.ss_generator_config_path, config.ss_generator_ckpt_path
            )
            self.ss_condition_embedder = self.init_ss_condition_embedder(
                config.ss_generator_config_path, config.ss_generator_ckpt_path
            )

            self.ss_generator.requires_grad_(False)
            self.ss_generator.train()
            if self.ss_condition_embedder is not None:
                self.ss_condition_embedder.requires_grad_(False)
                self.ss_condition_embedder.eval()

            self.ss_generator.self_consistency_prob = 0.0
            if hasattr(self.ss_generator.reverse_fn, "p_unconditional"):
                self.ss_generator.reverse_fn.p_unconditional = 0.0

            self.backbone = self.ss_generator.reverse_fn.backbone
            self.backbone.eval()
            if hasattr(self.backbone.condition_embedder, "normalize_images"):
                self.backbone.condition_embedder.normalize_images = True
            self.ss_generator.loss_weights = {
                name: float(name == "shape") for name in self.backbone.latent_mapping
            }

    pipeline = Stage1TrainingPipeline(config_path, device)
    if no_pointmap or no_visual:
        condition_embedder = pipeline.ss_condition_embedder
        if condition_embedder is None:
            condition_embedder = pipeline.backbone.condition_embedder
        if no_visual:
            disable_visual_conditioning(condition_embedder)
        else:
            disable_pointmap_conditioning(condition_embedder)
    return pipeline


def disable_pointmap_conditioning(fuser):
    names = {name for _, inputs in fuser.embedder_list for name, _ in inputs}
    pointmaps = {"pointmap", "rgb_pointmap"}
    if not pointmaps.issubset(names):
        raise ValueError(f"Expected pointmap and rgb_pointmap conditioning, found {sorted(names)}")
    fuser.force_drop_modalities = sorted(set(fuser.force_drop_modalities or []) | pointmaps)


def disable_visual_conditioning(fuser):
    # Surface tokens are appended separately, after this visual fuser.
    names = {name for _, inputs in fuser.embedder_list for name, _ in inputs}
    fuser.force_drop_modalities = sorted(set(fuser.force_drop_modalities or []) | names)


def build_stage1_preprocessor(config_path):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_path = Path(config_path).resolve()
    config = OmegaConf.load(config_path)
    preprocessor = config.get("ss_preprocessor")
    if preprocessor is None:
        generator_config = OmegaConf.load(
            config_path.parent / config.ss_generator_config_path
        )
        preprocessor = generator_config["tdfy"]["val_preprocessor"]
    return instantiate(preprocessor)


class TouchTrainingModel(torch.nn.Module):
    def __init__(self, generator, touch_encoder=None, no_pointmap=False, oracle_point_frame=False,
                 visual_dropout=0.0, constant_touch=False, no_visual=False,
                 shared_pointmap_normalization=False, joint_pointmap_points=None):
        super().__init__()
        self.generator = generator
        self.touch_encoder = touch_encoder
        self.conditioning_config = {
            "no_pointmap": no_pointmap,
            "oracle_point_frame": oracle_point_frame,
        }
        # Omit the default to preserve legacy checkpoint metadata.
        if no_visual:
            self.conditioning_config["no_visual"] = True
        if shared_pointmap_normalization:
            self.conditioning_config["shared_pointmap_normalization"] = True
        if joint_pointmap_points is not None:
            self.conditioning_config['joint_pointmap_points'] = joint_pointmap_points
        self.training_config = {"visual_dropout": visual_dropout, "constant_touch": constant_touch}
        self.register_buffer("constant_touch_features", None, persistent=False)
        self.constant_touch_sample_id = None

    def get_touch_tokens(self, touch_xyz, touch_mask):
        if self.touch_encoder is None:
            return None
        if self.training_config["constant_touch"]:
            if self.constant_touch_features is None:
                raise RuntimeError("Constant touch features have not been initialized or restored")
            features = self.constant_touch_features.expand(len(touch_xyz), -1, -1)
            return self.touch_encoder.output_projection(features) + self.touch_encoder.touch_embedding
        return self.touch_encoder(touch_xyz, touch_mask)

    def load_constant_touch(self, state):
        if not self.training_config["constant_touch"]:
            if state is not None:
                raise ValueError("Unexpected constant touch state for a normal surface run")
            return
        if state is None or self.touch_encoder is None:
            raise ValueError("Constant touch checkpoint is missing its fixed feature bank")
        features = state["features"]
        if features.ndim != 3 or features.shape[0] != 1 or not torch.isfinite(features).all():
            raise ValueError("Invalid constant touch feature bank")
        self.constant_touch_features = features.detach().to(self.touch_encoder.touch_embedding.device)
        self.constant_touch_sample_id = state["sample_id"]

    def forward(self, targets, condition_args, condition_kwargs, touch_xyz, touch_mask,
                visual_drop_mask=None):
        if visual_drop_mask is not None:
            if len(condition_args) != 1 or condition_kwargs:
                raise ValueError("Visual dropout expects a single precomputed visual-token context")
            visual = condition_args[0]
            if visual_drop_mask.shape != (visual.shape[0],):
                raise ValueError("Visual dropout mask must contain one decision per sample")
            condition_args = (visual.masked_fill(visual_drop_mask[:, None, None], 0),)
        if self.touch_encoder is None:
            loss, _ = self.generator.loss(targets, *condition_args, **condition_kwargs)
        else:
            touch_tokens = self.get_touch_tokens(touch_xyz, touch_mask)
            loss, _ = self.generator.loss(
                targets, *condition_args, touch_tokens=touch_tokens, **condition_kwargs
            )
        return loss


def amp(device, precision):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and precision == "bf16",
    )


class SurfacePointmapNormalizer:
    """Use the full surface's VecSetX center/radius for pointmap normalization."""

    def __init__(self, surface):
        center = (surface.max(dim=0).values + surface.min(dim=0).values) / 2
        radius = torch.linalg.vector_norm(surface - center, dim=-1).max()
        if not torch.isfinite(radius) or radius <= 0:
            raise ValueError("Full surface must have a positive finite radius")
        self.center = center
        self.radius = radius

    def normalize(self, pointmap, mask, scale=None, shift=None):
        center = self.center.to(pointmap)
        radius = self.radius.to(pointmap)
        normalized = (pointmap - center[:, None, None]) / radius
        return normalized, radius.expand(3), center


def preprocess_batch(
    pipeline, images, pointmaps, touch_xyz=None, touch_mask=None,
    shared_pointmap_normalization=False,
):
    if shared_pointmap_normalization and (touch_xyz is None or touch_mask is None):
        raise ValueError("Shared pointmap normalization requires full-surface points")

    items = []
    for index, (image, pointmap) in enumerate(zip(images, pointmaps)):
        preprocessor = pipeline.ss_preprocessor
        if shared_pointmap_normalization:
            preprocessor = copy.copy(preprocessor)
            normalizer = SurfacePointmapNormalizer(
                touch_xyz[index, touch_mask[index]]
            )
            preprocessor.normalize_pointmap = True
            preprocessor.pointmap_normalizer = normalizer
            preprocessor.rgb_pointmap_normalizer = normalizer
        items.append(pipeline.preprocess_image(
            image.numpy(), preprocessor, pointmap=pointmap.permute(2, 0, 1)
        ))
    return {key: torch.cat([item[key] for item in items]) for key in items[0]}


def preprocess_pointmap_batch(preprocessor, images, pointmaps, device):
    from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import get_mask

    items = []
    for image, pointmap in zip(images, pointmaps):
        rgba = torch.from_numpy((image.numpy() / 255).astype(np.float32))
        rgba = rgba.permute(2, 0, 1).contiguous()
        item = preprocessor._process_image_mask_pointmap_mess(
            rgba[:3],
            get_mask(rgba, None, "ALPHA_CHANNEL"),
            pointmap.permute(2, 0, 1),
        )
        items.append({
            key: item[key][None].to(device)
            for key in ("mask", "pointmap", "pointmap_scale", "pointmap_shift")
        })
    return {key: torch.cat([item[key] for item in items]) for key in items[0]}


def normalize_touch_to_pointmap_frame(touch_xyz, touch_mask, inputs, preprocessor):
    from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
        _apply_metric_to_ssi,
    )

    if not preprocessor.normalize_pointmap:
        return touch_xyz

    normalizer = preprocessor.pointmap_normalizer
    normalized = []
    for points, mask, scale, shift in zip(
        touch_xyz, touch_mask, inputs["pointmap_scale"], inputs["pointmap_shift"]
    ):
        valid_points = points[mask]
        if hasattr(normalizer, "point_remapper"):
            valid_points = normalizer.point_remapper(valid_points)
        output = torch.zeros_like(points)
        output[mask] = _apply_metric_to_ssi(valid_points, scale, shift)
        normalized.append(output)
    return torch.stack(normalized)


def combine_pointmap_and_touch(inputs, touch_xyz, touch_mask):
    pointmap = inputs["pointmap"].permute(0, 2, 3, 1).flatten(1, 2)
    pointmap_mask = inputs["mask"].flatten(1) > 0.5
    pointmap_mask &= torch.isfinite(pointmap).all(dim=-1)

    clouds = [
        torch.cat((points[valid], touch[mask]), dim=0)
        for points, valid, touch, mask in zip(
            pointmap, pointmap_mask, touch_xyz, touch_mask
        )
    ]
    lengths = torch.tensor([len(cloud) for cloud in clouds], device=pointmap.device)
    clouds = pad_sequence(clouds, batch_first=True)
    mask = torch.arange(clouds.shape[1], device=clouds.device)[None] < lengths[:, None]
    return clouds, mask


def make_targets(shape, backbone):
    targets = {}
    for name, mapping in backbone.latent_mapping.items():
        if name == "shape":
            targets[name] = shape
        else:
            targets[name] = shape.new_zeros(
                shape.shape[0], mapping.pos_emb.shape[0], mapping.input_layer.in_features
            )
    return targets


def prepare_batch(
    pipeline, batch, device, precision, use_touch, joint_pointmap=False,
    oracle_point_frame=False, shared_pointmap_normalization=False, use_normals=False,
    return_inputs=False, fixed_joint_patches=True,
):
    inputs = preprocess_batch(
        pipeline, batch["image"], batch["pointmap"],
        batch.get("touch_xyz"), batch.get("touch_mask"),
        shared_pointmap_normalization,
    )
    with torch.no_grad(), amp(device, precision):
        condition_args, condition_kwargs = pipeline.get_condition_input(
            pipeline.ss_condition_embedder,
            inputs,
            pipeline.ss_condition_input_mapping,
        )

    shape = batch["target_shape"].to(device, non_blocking=True)
    touch_xyz = touch_mask = None
    if use_touch:
        touch_mask = batch["touch_mask"].to(device, non_blocking=True)
        with torch.no_grad():
            touch_xyz = batch["touch_xyz"].to(device, non_blocking=True)
            normals = batch["touch_normals"].to(device, non_blocking=True) if use_normals else None
            if use_normals and joint_pointmap:
                raise ValueError("Joint pointmap does not provide surface normals")
            if oracle_point_frame:
                if joint_pointmap:
                    raise ValueError("Oracle point frame cannot use joint pointmap")
                transform = batch["object_from_camera"].to(device)
                touch_xyz = touch_xyz @ transform[:, :3, :3].transpose(1, 2) + transform[:, None, :3, 3]
                touch_xyz = touch_xyz.masked_fill(~touch_mask[..., None], 0)
                if normals is not None:
                    normals = normals @ torch.linalg.inv(transform[:, :3, :3])
                    normals = torch.nn.functional.normalize(normals, dim=-1)
            elif not shared_pointmap_normalization and not use_normals and not (
                    joint_pointmap and fixed_joint_patches and 'touch_patch_count' in batch):
                touch_xyz = normalize_touch_to_pointmap_frame(
                    touch_xyz, touch_mask, inputs, pipeline.ss_preprocessor,
                )
            if joint_pointmap:
                if fixed_joint_patches and 'touch_patch_count' in batch:
                    if 'joint_xyz' not in batch:
                        raise ValueError('Saved joint inputs are missing; rerun make_touch_data.py with --joint-pointmap')
                    for key in ('pointmap_scale', 'pointmap_shift'):
                        if not torch.allclose(batch['joint_' + key].to(inputs[key]), inputs[key], atol=1e-5, rtol=1e-5):
                            raise ValueError('Saved joint preprocessing differs; regenerate joint inputs with the training pipeline config')
                    touch_xyz = batch['joint_xyz'].to(device, non_blocking=True)
                    touch_mask = torch.ones(touch_xyz.shape[:2], dtype=torch.bool, device=device)
                else:
                    touch_xyz, touch_mask = combine_pointmap_and_touch(inputs, touch_xyz, touch_mask)
            if normals is not None:
                # Keep the existing five-item batch interface; the new encoders take XYZ+normal.
                touch_xyz = torch.cat((touch_xyz, normals), dim=-1)

    result = make_targets(shape, pipeline.backbone), condition_args, condition_kwargs, touch_xyz, touch_mask
    return (*result, inputs) if return_inputs else result


def initialize_constant_touch(model, pipeline, dataset, device, precision):
    index = min(range(len(dataset)), key=lambda i: dataset.records[i]["sample_id"])
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            batch = collate_touch_batch([dataset[index]])
            _, _, _, points, mask = prepare_batch(
                pipeline, batch, device, precision, True, oracle_point_frame=True,
            )
            encoder = model.touch_encoder
            was_training = encoder.training
            try:
                encoder.eval()
                with amp(device, precision):
                    points, mask, _, _ = encoder.prepare_points(points, mask)
                    features = encoder.encoder.encode(points, mask)["x"]
            finally:
                encoder.train(was_training)
            model.load_constant_touch({"features": features, "sample_id": dataset.records[index]["sample_id"]})
            if dist.is_initialized():
                dist.broadcast(model.constant_touch_features, src=0)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def make_visual_drop_mask(batch_size, probability, device, seed, step, rank):
    if probability == 0:
        return None
    # Separate, step-addressed RNG: no consumption of flow noise/time randomness.
    generator = torch.Generator(device=device).manual_seed(seed + 1_000_003 * step + 1_000_033 * rank)
    return torch.rand(batch_size, device=device, generator=generator) < probability


def shape_path_module_groups(backbone):
    """Shape-specific modules plus shared timestep/modulation used by shape.

    Layout-specific modules and the visual encoder remain frozen. Fixed shape
    positional embeddings are buffers, not optimizer parameters.
    """
    groups = {
        "shape_self_attention": [],
        "shape_mlp": [],
        "shape_input_output": [backbone.latent_mapping["shape"]],
        "shape_time_modulation": [backbone.t_embedder],
    }
    for name in ("d_embedder", "adaLN_modulation"):
        if hasattr(backbone, name):
            groups["shape_time_modulation"].append(getattr(backbone, name))
    for block in backbone.blocks:
        if "shape" not in block.self_attn.protect_modality_list:
            raise ValueError("shape_full requires shape self-attention protected from layout")
        for name in ("to_qkv", "to_out", "q_rms_norm", "k_rms_norm"):
            if hasattr(block.self_attn, name):
                groups["shape_self_attention"].append(getattr(block.self_attn, name)["shape"])
        groups["shape_self_attention"].append(block.norm1["shape"])
        groups["shape_mlp"].extend((block.mlp["shape"], block.norm3["shape"]))
        if hasattr(block, "adaLN_modulation"):
            groups["shape_time_modulation"].append(block.adaLN_modulation)
    return groups


def build_optimizer(touch_encoder, backbone, args):
    scope = resolve_train_scope(getattr(args, "train_scope", None), getattr(args, "cross_attention_scope", "full"))
    backbone.requires_grad_(False)
    backbone.train_scope = scope
    groups = []
    if touch_encoder is not None:
        groups.append({
            "params": list(touch_encoder.get_trainable_parameters()),
            "lr": args.learning_rate,
            "name": "touch_encoder",
        })

    if args.cross_attention_learning_rate > 0:
        modules = [block.cross_attn["shape"].to_kv for block in backbone.blocks]
        if scope != "shape_cross_attention_kv":
            modules = [
                module
                for block in backbone.blocks
                for module in (block.cross_attn["shape"], block.norm2["shape"])
            ]
        if scope == "shape_full":
            modules.extend(module for group in shape_path_module_groups(backbone).values() for module in group)
        for module in modules:
            module.requires_grad_(True)
        groups.append({
            "params": list(dict.fromkeys(parameter for module in modules for parameter in module.parameters())),
            "lr": args.cross_attention_learning_rate,
            "name": "shape_full" if scope == "shape_full" else "cross_attention",
        })

    if not groups:
        raise ValueError("No trainable parameters were selected")

    parameters = [parameter for group in groups for parameter in group["params"]]
    return torch.optim.AdamW(groups, weight_decay=0), parameters


def trainable_state_dict(model):
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def touch_adapter_state_dict(model):
    prefixes = (
        "touch_encoder.output_projection.",
        "touch_encoder.position_projection.",
        "touch_encoder.touch_embedding",
    )
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    }


def gradient_norm(parameters):
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        total = value if total is None else total + value
    return 0.0 if total is None else total.sqrt().item()


def component_gradient_norms(model, include_parameters=False):
    backbone = model.generator.reverse_fn.backbone
    groups = {
        "shape_cross_attention_kv": (
            parameter
            for block in model.generator.reverse_fn.backbone.blocks
            for parameter in block.cross_attn["shape"].to_kv.parameters()
        ),
        "shape_cross_attention": (
            parameter
            for block in model.generator.reverse_fn.backbone.blocks
            for module in (block.cross_attn["shape"], block.norm2["shape"])
            for parameter in module.parameters()
        ),
    }
    if getattr(backbone, "train_scope", None) == "shape_full":
        groups.update({
            name: (parameter for module in modules for parameter in module.parameters())
            for name, modules in shape_path_module_groups(backbone).items()
        })
    if model.touch_encoder is not None:
        groups.update({
            "touch_output_projection": model.touch_encoder.output_projection.parameters(),
            "touch_position_projection": model.touch_encoder.position_projection.parameters(),
            "touch_embedding": (model.touch_encoder.touch_embedding,),
            f"{model.touch_encoder.encoder_name}_encoder": model.touch_encoder.encoder.parameters(),
        })
    metrics = {}
    for name, parameters in groups.items():
        parameters = list(parameters)
        metrics[f"gradients/{name}"] = gradient_norm(parameters)
        if include_parameters:
            trainable = [parameter for parameter in parameters if parameter.requires_grad]
            if trainable:
                total = sum(parameter.detach().float().square().sum() for parameter in trainable)
                metrics[f"parameters/{name}_trainable_norm"] = total.sqrt().item()
    return metrics


@contextmanager
def token_magnitudes(model, prepared=None, drop_mask=None, enabled=True, pipeline=None):
    """Observe conditioning preparation and the model forward without retaining activations."""
    metrics, handles, calls = {}, [], []
    fuser = getattr(pipeline, 'ss_condition_embedder', None)
    if fuser is None:
        fuser = getattr(getattr(pipeline, 'backbone', None), 'condition_embedder', None)

    def record(name, tensor):
        value = tensor.detach().float()
        metrics[f"tokens/{name}_rms"] = value.square().mean().sqrt().item()
        metrics[f"tokens/{name}_mean_token_norm"] = value.norm(dim=-1).mean().item()

    def visual_metrics(condition_args, mask):
        if len(condition_args) == 1 and torch.is_tensor(condition_args[0]) and condition_args[0].ndim == 3:
            visual = condition_args[0]
            record('visual', visual)
            if mask is not None:
                record('visual_after_dropout', visual.masked_fill(mask[:, None, None], 0))

    def model_hook(module, inputs, kwargs):
        visual_metrics(inputs[1], kwargs.get('visual_drop_mask'))

    def projection_hook(module, inputs, output):
        record('surface_before_projection', inputs[0])
        record('surface_after_projection', output)

    def encoder_hook(module, inputs, output):
        record('surface_conditioning', output)

    def fuser_hook(module, inputs, output):
        # Channel compression mixes modalities, so separate final tokens no longer exist.
        if getattr(module, 'compression_projection_multiplier', 0) > 0:
            return
        if not torch.is_tensor(output) or output.ndim != 3 or sum(n for _, n in calls) != output.shape[1]:
            return
        offset = 0
        for name, count in calls:
            if name in ('pointmap', 'rgb_pointmap'):
                record(name + '_conditioning', output[:, offset:offset + count])
            offset += count

    try:
        if enabled:
            if hasattr(fuser, 'embedder_list'):
                modules = {}
                for module, arguments in fuser.embedder_list:
                    modules.setdefault(module, []).extend(name for name, _ in arguments)
                for module, names in modules.items():
                    def observe(module, inputs, output, names=names, counter=[0]):
                        name = names[counter[0] % len(names)]
                        counter[0] += 1
                        if torch.is_tensor(output) and output.ndim == 3:
                            calls.append((name, output.shape[1]))
                            if name in ('pointmap', 'rgb_pointmap'):
                                record(name + '_before_projection', output)
                    handles.append(module.register_forward_hook(observe))
                handles.append(fuser.register_forward_hook(fuser_hook))
            if prepared is not None:
                visual_metrics(prepared[1], drop_mask)
            elif isinstance(model, torch.nn.Module):
                handles.append(model.register_forward_pre_hook(model_hook, with_kwargs=True))
            if model.touch_encoder is not None:
                handles.append(model.touch_encoder.output_projection.register_forward_hook(projection_hook))
                handles.append(model.touch_encoder.register_forward_hook(encoder_hook))
        yield metrics
    finally:
        for handle in handles:
            handle.remove()


def load_trainable_state_dict(model, state_dict):
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if parameters.keys() != state_dict.keys():
        raise ValueError("Checkpoint trainable parameters do not match this run")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state_dict[name].to(parameter))


def save_checkpoint(
    path, model, optimizer, epoch, step, best_loss, mode, cross_attention_scope="kv", train_scope=None
):
    torch.save({
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "mode": mode,
        "conditioning_config": model.conditioning_config,
        "training_config": model.training_config,
        "constant_touch": (
            {"sample_id": model.constant_touch_sample_id,
             "features": model.constant_touch_features.detach().cpu()}
            if model.constant_touch_features is not None else None
        ),
        "cross_attention_scope": cross_attention_scope,
        "train_scope": resolve_train_scope(train_scope, cross_attention_scope),
        "touch_config": (
            model.touch_encoder.get_config() if model.touch_encoder is not None else None
        ),
    }, path)


def load_checkpoint(path, model, optimizer, mode, cross_attention_scope="kv", train_scope=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("conditioning_config", {"no_pointmap": False, "oracle_point_frame": False}) != model.conditioning_config:
        raise ValueError("Checkpoint conditioning configuration does not match this run")
    if checkpoint_train_scope(checkpoint) != resolve_train_scope(train_scope, cross_attention_scope):
        raise ValueError("Checkpoint training scope does not match this run; --resume cannot change scope")
    if checkpoint.get("training_config", {"visual_dropout": 0.0, "constant_touch": False}) != model.training_config:
        raise ValueError("Checkpoint visual dropout or constant touch setting does not match this run")
    touch_config = (
        model.touch_encoder.get_config() if model.touch_encoder is not None else None
    )
    if checkpoint["mode"] != mode or checkpoint["touch_config"] != touch_config:
        raise ValueError("Checkpoint mode or touch configuration does not match this run")
    load_trainable_state_dict(model, checkpoint["model"])
    model.load_constant_touch(checkpoint.get("constant_touch"))
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"], checkpoint["step"], checkpoint["best_loss"]


def aggregate(total_loss, total_samples, device, distributed):
    totals = torch.tensor([total_loss, total_samples], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(totals)
    return totals.tolist()


def train_epoch(
    pipeline, model, raw_model, loader, optimizer, parameters,
    device, args, epoch, step, total_train_steps, world_size,
    distributed, main_process, run, seed=0,
):
    if distributed:
        loader.sampler.set_epoch(epoch)
    if raw_model.touch_encoder is not None:
        raw_model.touch_encoder.train()

    total_loss = 0.0
    total_samples = 0
    total_dropped = 0
    log_start_time = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        should_log = (
            step == 0
            or (step + 1) % args.log_every == 0
            or batch_index + 1 == len(loader)
        )
        with token_magnitudes(raw_model, enabled=main_process and should_log, pipeline=pipeline) as token_metrics:
            prepared = prepare_batch(
                pipeline, batch, device, args.precision,
                raw_model.touch_encoder is not None,
                args.joint_pointmap,
                args.oracle_point_frame,
                getattr(args, "shared_pointmap_normalization", False),
                use_normals=bool(raw_model.touch_encoder is not None and getattr(raw_model.touch_encoder, "requires_normals", False)),
            )
            optimizer.zero_grad(set_to_none=True)
            drop_mask = make_visual_drop_mask(
                len(batch["target_shape"]), raw_model.training_config["visual_dropout"],
                device, seed, step, dist.get_rank() if distributed else 0,
            )
            with amp(device, args.precision):
                loss = model(*prepared, visual_drop_mask=drop_mask)
        loss.backward()
        component_gradients = (
            component_gradient_norms(raw_model, include_parameters=True)
            if main_process and should_log
            else {}
        )
        gradient_norm = None
        if args.gradient_clip > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, args.gradient_clip
            )
        optimizer.step()

        step += 1
        batch_size = batch["target_shape"].shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        if drop_mask is not None:
            total_dropped += int(drop_mask.sum())

        if step == 1 or step % args.log_every == 0 or batch_index + 1 == len(loader):
            elapsed = torch.tensor(
                time.perf_counter() - log_start_time,
                dtype=torch.float64,
                device=device,
            )
            loss_sum, sample_count = aggregate(
                total_loss, total_samples, device, distributed
            )
            dropped_sum = 0
            if raw_model.training_config["visual_dropout"] > 0:
                dropped_sum, _ = aggregate(total_dropped, total_samples, device, distributed)
            if distributed:
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

            mean_loss = loss_sum / sample_count
            samples_per_second = sample_count / elapsed.item()
            eta_seconds = (
                max(total_train_steps - step, 0)
                * args.batch_size
                * world_size
                / samples_per_second
            )
            if main_process:
                progress = (
                    f"step {step}/{args.max_steps} epoch {epoch + 1}"
                    if args.max_steps
                    else f"epoch {epoch + 1}/{args.epochs} step {step}"
                )
                print(
                    f"{progress} "
                    f"loss {mean_loss:.6f} "
                    f"throughput {samples_per_second:.2f} samples/s "
                    f"train_eta {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}",
                    flush=True,
                )
                metrics = {
                    "global_step": step,
                    "loss/train": mean_loss,
                    "conditioning/visual_dropout_fraction": dropped_sum / sample_count,
                    "performance/samples_per_second": samples_per_second,
                    "performance/train_eta_seconds": eta_seconds,
                }
                if gradient_norm is not None:
                    metrics["optimization/gradient_norm"] = gradient_norm.item()
                metrics.update(component_gradients)
                metrics.update(token_metrics)
                surface_rms = token_metrics.get('tokens/surface_conditioning_rms')
                for branch in ('pointmap', 'rgb_pointmap'):
                    branch_rms = token_metrics.get(f'tokens/{branch}_conditioning_rms')
                    if surface_rms is not None and branch_rms is not None and branch_rms > 0:
                        metrics[f'tokens/surface_to_{branch}_rms_ratio'] = surface_rms / branch_rms
                pointmap_metrics = {key: value for key, value in token_metrics.items()
                                    if key in ('tokens/pointmap_conditioning_rms', 'tokens/rgb_pointmap_conditioning_rms')}
                if pointmap_metrics:
                    print('conditioning ' + ' '.join(
                        f'{name}={value:.4g}' for name, value in pointmap_metrics.items()
                        if name.endswith('_conditioning_rms')
                    ), flush=True)
                for group in optimizer.param_groups:
                    metrics[f"learning_rate/{group['name']}"] = group["lr"]
                run.log(metrics)
            total_loss = 0.0
            total_samples = 0
            total_dropped = 0
            log_start_time = time.perf_counter()

        if args.max_steps and step >= args.max_steps:
            break
    return step


def validate(pipeline, model, loader, device, args, seed, distributed, rank):
    was_training = model.touch_encoder is not None and model.touch_encoder.training
    if model.touch_encoder is not None:
        model.touch_encoder.eval()

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    total_loss = 0.0
    total_samples = 0

    with torch.random.fork_rng(devices=cuda_devices):
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        with torch.no_grad():
            for batch in loader:
                prepared = prepare_batch(
                    pipeline, batch, device, args.precision,
                    model.touch_encoder is not None,
                    args.joint_pointmap,
                    args.oracle_point_frame,
                    getattr(args, "shared_pointmap_normalization", False),
                    use_normals=bool(model.touch_encoder is not None and getattr(model.touch_encoder, "requires_normals", False)),
                )
                with amp(device, args.precision):
                    loss = model(*prepared)
                batch_size = batch["target_shape"].shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    if was_training:
        model.touch_encoder.train()
    total_loss, total_samples = aggregate(
        total_loss, total_samples, device, distributed
    )
    return total_loss / total_samples


def configure_encoder_data(config, encoder_name):
    if encoder_name == "vecsetx":
        return config
    touch = config.setdefault("touch", {})
    if touch.get("source") != "full_surface":
        raise ValueError("CraftsMan/TripoSG require touch.source: full_surface")
    touch["include_normals"] = True
    touch.setdefault("pool_points", {"craftsman": 16384, "triposg": 20480}[encoder_name])
    if int(touch["pool_points"]) < {"craftsman": 768, "triposg": 2048}[encoder_name]:
        raise ValueError("Surface pool must contain at least as many points as encoder queries")
    return config


def main():
    args = parse_args()
    if not 0 <= args.visual_dropout <= 1:
        raise ValueError("--visual-dropout must be between 0 and 1")
    if args.no_visual and (args.no_touch or args.joint_pointmap):
        raise ValueError("--no-visual requires surface conditioning without --joint-pointmap")
    if args.shared_pointmap_normalization and (
        args.no_pointmap or args.no_visual or args.oracle_point_frame or args.joint_pointmap
    ):
        raise ValueError(
            "--shared-pointmap-normalization requires camera-frame pointmap conditioning"
        )
    if args.constant_touch and (
        not args.oracle_point_frame or args.train_vecsetx or args.vecsetx_learn
    ):
        raise ValueError("--constant-touch requires oracle full surfaces and frozen raw VecSetX features")
    if args.oracle_point_frame and (args.no_touch or args.joint_pointmap or not args.no_touch_position):
        raise ValueError("Oracle point frame requires touch, disabled touch position, and no joint pointmap")
    if args.no_touch and args.train_vecsetx:
        raise ValueError("--train-vecsetx cannot be used with --no-touch")
    if args.no_touch and args.vecsetx_learn:
        raise ValueError("--vecsetx-learn cannot be used with --no-touch")
    if args.no_touch and args.joint_pointmap:
        raise ValueError("--joint-pointmap cannot be used with --no-touch")

    distributed, rank, world_size, device = setup_distributed(args)
    main_process = rank == 0
    torch.set_float32_matmul_precision("high")

    from loguru import logger
    if not main_process:
        logger.remove()
    else:
        import wandb

    data_config = configure_encoder_data(load_data_config(args.data_config), args.point_encoder)
    seed = int(data_config.get("seed", 0))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    train_loader = build_dataloader(
        data_config, args.batch_size, args.workers,
        distributed=distributed,
        include_touch=not args.no_touch or args.shared_pointmap_normalization,
        oracle_point_frame=args.oracle_point_frame,
        joint_pointmap=args.joint_pointmap,
    )
    if len(train_loader) == 0:
        raise ValueError("Training loader has no batches")
    val_config = copy.deepcopy(data_config)
    val_config["dataset"]["split"] = "val"
    val_loader = build_dataloader(
        val_config, args.batch_size, args.val_workers,
        shuffle=False, distributed=distributed,
        include_touch=not args.no_touch or args.shared_pointmap_normalization,
        oracle_point_frame=args.oracle_point_frame,
        joint_pointmap=args.joint_pointmap,
    )

    pipeline = build_stage1_pipeline(
        args.pipeline_config, device, no_pointmap=args.no_pointmap, no_visual=args.no_visual,
    )
    touch_encoder = None
    if not args.no_touch:
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
        touch_encoder = TouchEncoder(
            encoder_name=args.point_encoder,
            output_dim=pipeline.backbone.cond_channels,
            trainable=args.train_vecsetx,
            pretrained=not args.vecsetx_from_scratch,
            encoder_checkpoint=args.point_encoder_checkpoint,
            use_learn=args.vecsetx_learn,
            use_position=not args.no_touch_position,
            position_scale="log",
        ).to(device)

    model = TouchTrainingModel(
        pipeline.ss_generator, touch_encoder, args.no_pointmap,
        args.oracle_point_frame, args.visual_dropout, args.constant_touch,
        no_visual=args.no_visual,
        shared_pointmap_normalization=args.shared_pointmap_normalization,
        joint_pointmap_points=(1024 if args.joint_pointmap and data_config.get('touch', {}).get('source') == 'touch_patches' else None),
    )
    optimizer, parameters = build_optimizer(touch_encoder, pipeline.backbone, args)
    if args.no_touch:
        mode = "image"
    elif args.joint_pointmap:
        mode = "image_touch_joint"
    else:
        mode = "image_touch"
    start_epoch, step, best_loss = 0, 0, float("inf")
    if args.resume:
        start_epoch, step, best_loss = load_checkpoint(
            args.resume, model, optimizer, mode, args.cross_attention_scope, args.train_scope
        )
    elif args.constant_touch:
        initialize_constant_touch(model, pipeline, train_loader.dataset, device, args.precision)

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[device.index], broadcast_buffers=False
        )
    raw_model = model.module if distributed else model

    config_path = args.output_dir / "config.yaml"
    if config_path.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {config_path}")

    if main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            run_config = {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "mode": mode,
                "world_size": world_size,
                "global_batch_size": args.batch_size * world_size,
                "touch_config": (
                    touch_encoder.get_config() if touch_encoder is not None else None
                ),
                "data": data_config,
                "constant_touch_sample_id": raw_model.constant_touch_sample_id,
            }
            with open(config_path, "w") as file:
                yaml.safe_dump(run_config, file, sort_keys=False)

            if touch_encoder is not None and not args.resume:
                torch.save(
                    touch_adapter_state_dict(raw_model),
                    args.output_dir / "initial_touch_adapter.pt",
                )

        print(f"mode: {mode}")
        print(f"training scope: {args.train_scope}")
        print(f"precision: {args.precision}")
        print(f"touch position: {touch_encoder is not None and touch_encoder.use_position}")
        if touch_encoder is not None:
            print(f"point encoder: {args.point_encoder}; pretrained={touch_encoder.pretrained}; trainable={touch_encoder.encoder_trainable}")
        print(f"visual dropout: {args.visual_dropout} per sample (training only)")
        if args.shared_pointmap_normalization:
            print("pointmap normalization: full-surface center and radius")
        if args.no_visual:
            print("visual conditioning: permanently disabled (training and validation)")
        if args.constant_touch:
            print(f"constant touch reference: {raw_model.constant_touch_sample_id}")
        print(f"GPUs: {world_size}")
        print(
            f"batch size: {args.batch_size} per GPU, "
            f"{args.batch_size * world_size} global"
        )
        print(f"workers: {args.workers} train and {args.val_workers} val per GPU")
        print(f"train samples: {len(train_loader.dataset)}")
        print(f"val samples: {len(val_loader.dataset)}")
        print(f"trainable parameters: {sum(p.numel() for p in parameters):,}")

    if distributed:
        dist.barrier()

    run = None
    if main_process:
        run_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        run_config.update({
            "mode": mode,
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
            "constant_touch_sample_id": raw_model.constant_touch_sample_id,
            "touch_config": (
                touch_encoder.get_config() if touch_encoder is not None else None
            ),
        })
        run = wandb.init(
            project="sam-3d-touch",
            name=args.output_dir.name,
            dir=str(args.output_dir),
            config=run_config,
            id=args.wandb_id,
            resume="must" if args.wandb_id else None,
        )
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")

    total_train_steps = args.max_steps or args.epochs * len(train_loader)

    epoch = start_epoch
    while (step < args.max_steps if args.max_steps else epoch < args.epochs):
        step = train_epoch(
            pipeline, model, raw_model, train_loader, optimizer, parameters,
            device, args, epoch, step, total_train_steps, world_size,
            distributed, main_process, run, seed=seed,
        )
        val_loss = validate(
            pipeline, raw_model, val_loader, device, args,
            seed + 1, distributed, rank,
        )
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if main_process:
            run.log({"global_step": step, "loss/val": val_loss})
            progress = (
                f"step {step}/{args.max_steps} epoch {epoch + 1}"
                if args.max_steps
                else f"epoch {epoch + 1}/{args.epochs}"
            )
            print(
                f"{progress} val_loss {val_loss:.6f}"
                f"{' best' if improved else ''}",
                flush=True,
            )
            save_checkpoint(
                args.output_dir / "last.pt", raw_model, optimizer,
                epoch + 1, step, best_loss, mode,
                args.cross_attention_scope,
                args.train_scope,
            )
            if improved:
                save_checkpoint(
                    args.output_dir / "best.pt", raw_model, optimizer,
                    epoch + 1, step, best_loss, mode,
                    args.cross_attention_scope,
                    args.train_scope,
                )

        if distributed:
            dist.barrier()
        epoch += 1

    if main_process:
        run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
