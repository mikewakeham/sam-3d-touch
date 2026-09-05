import argparse
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

from dataloader import build_dataloader, load_data_config


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cross-attention-learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--cross-attention-scope", choices=["kv", "full"], default="kv",
        help="Train shape K/V only, or full shape cross-attention plus its input norm",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--no-touch", action="store_true")
    parser.add_argument("--train-vecsetx", action="store_true")
    parser.add_argument("--joint-pointmap", action="store_true")
    parser.add_argument("--no-touch-position", action="store_true")
    parser.add_argument(
        "--local-rank", "--local_rank", type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),
    )
    return parser.parse_args()


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


def build_stage1_pipeline(config_path, device):
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

    return Stage1TrainingPipeline(config_path, device)


class TouchTrainingModel(torch.nn.Module):
    def __init__(self, generator, touch_encoder=None):
        super().__init__()
        self.generator = generator
        self.touch_encoder = touch_encoder

    def forward(self, targets, condition_args, condition_kwargs, touch_xyz, touch_mask):
        if self.touch_encoder is None:
            loss, _ = self.generator.loss(targets, *condition_args, **condition_kwargs)
        else:
            touch_tokens = self.touch_encoder(touch_xyz, touch_mask)
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


def preprocess_batch(pipeline, images, pointmaps):
    items = [
        pipeline.preprocess_image(
            image.numpy(),
            pipeline.ss_preprocessor,
            pointmap=pointmap.permute(2, 0, 1),
        )
        for image, pointmap in zip(images, pointmaps)
    ]
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
    pipeline, batch, device, precision, use_touch, joint_pointmap=False
):
    inputs = preprocess_batch(pipeline, batch["image"], batch["pointmap"])
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
            touch_xyz = normalize_touch_to_pointmap_frame(
                batch["touch_xyz"].to(device, non_blocking=True),
                touch_mask,
                inputs,
                pipeline.ss_preprocessor,
            )
            if joint_pointmap:
                touch_xyz, touch_mask = combine_pointmap_and_touch(
                    inputs, touch_xyz, touch_mask
                )

    return make_targets(shape, pipeline.backbone), condition_args, condition_kwargs, touch_xyz, touch_mask


def build_optimizer(touch_encoder, backbone, args):
    groups = []
    if touch_encoder is not None:
        groups.append({
            "params": list(touch_encoder.get_trainable_parameters()),
            "lr": args.learning_rate,
            "name": "touch_encoder",
        })

    if args.cross_attention_learning_rate > 0:
        modules = [block.cross_attn["shape"].to_kv for block in backbone.blocks]
        if args.cross_attention_scope == "full":
            modules = [
                module
                for block in backbone.blocks
                for module in (block.cross_attn["shape"], block.norm2["shape"])
            ]
        for module in modules:
            module.requires_grad_(True)
        groups.append({
            "params": [parameter for module in modules for parameter in module.parameters()],
            "lr": args.cross_attention_learning_rate,
            "name": "cross_attention",
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


def component_gradient_norms(model):
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
    if model.touch_encoder is not None:
        groups.update({
            "touch_output_projection": model.touch_encoder.output_projection.parameters(),
            "touch_position_projection": model.touch_encoder.position_projection.parameters(),
            "touch_embedding": (model.touch_encoder.touch_embedding,),
            "vecsetx_encoder": model.touch_encoder.encoder.parameters(),
        })
    return {
        f"gradients/{name}": gradient_norm(parameters)
        for name, parameters in groups.items()
    }


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
    path, model, optimizer, epoch, step, best_loss, mode, cross_attention_scope="kv"
):
    torch.save({
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "mode": mode,
        "cross_attention_scope": cross_attention_scope,
        "touch_config": (
            model.touch_encoder.get_config() if model.touch_encoder is not None else None
        ),
    }, path)


def load_checkpoint(path, model, optimizer, mode, cross_attention_scope="kv"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("cross_attention_scope", "kv") != cross_attention_scope:
        raise ValueError("Checkpoint cross-attention scope does not match this run")
    touch_config = (
        model.touch_encoder.get_config() if model.touch_encoder is not None else None
    )
    if checkpoint["mode"] != mode or checkpoint["touch_config"] != touch_config:
        raise ValueError("Checkpoint mode or touch configuration does not match this run")
    load_trainable_state_dict(model, checkpoint["model"])
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
    distributed, main_process, run,
):
    if distributed:
        loader.sampler.set_epoch(epoch)
    if raw_model.touch_encoder is not None:
        raw_model.touch_encoder.train()

    total_loss = 0.0
    total_samples = 0
    log_start_time = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        prepared = prepare_batch(
            pipeline, batch, device, args.precision,
            raw_model.touch_encoder is not None,
            args.joint_pointmap,
        )
        optimizer.zero_grad(set_to_none=True)
        with amp(device, args.precision):
            loss = model(*prepared)
        loss.backward()
        should_log = (
            step == 0
            or (step + 1) % args.log_every == 0
            or batch_index + 1 == len(loader)
        )
        component_gradients = (
            component_gradient_norms(raw_model)
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

        if step == 1 or step % args.log_every == 0 or batch_index + 1 == len(loader):
            elapsed = torch.tensor(
                time.perf_counter() - log_start_time,
                dtype=torch.float64,
                device=device,
            )
            loss_sum, sample_count = aggregate(
                total_loss, total_samples, device, distributed
            )
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
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {step} "
                    f"loss {mean_loss:.6f} "
                    f"throughput {samples_per_second:.2f} samples/s "
                    f"train_eta {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}",
                    flush=True,
                )
                metrics = {
                    "global_step": step,
                    "loss/train": mean_loss,
                    "performance/samples_per_second": samples_per_second,
                    "performance/train_eta_seconds": eta_seconds,
                }
                if gradient_norm is not None:
                    metrics["optimization/gradient_norm"] = gradient_norm.item()
                metrics.update(component_gradients)
                for group in optimizer.param_groups:
                    metrics[f"learning_rate/{group['name']}"] = group["lr"]
                run.log(metrics)
            total_loss = 0.0
            total_samples = 0
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


def main():
    args = parse_args()
    if args.no_touch and args.train_vecsetx:
        raise ValueError("--train-vecsetx cannot be used with --no-touch")
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

    data_config = load_data_config(args.data_config)
    seed = int(data_config.get("seed", 0))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    train_loader = build_dataloader(
        data_config, args.batch_size, args.workers,
        distributed=distributed, include_touch=not args.no_touch,
    )
    val_config = load_data_config(args.data_config)
    val_config["dataset"]["split"] = "val"
    val_loader = build_dataloader(
        val_config, args.batch_size, args.val_workers,
        shuffle=False, distributed=distributed, include_touch=not args.no_touch,
    )

    pipeline = build_stage1_pipeline(args.pipeline_config, device)
    touch_encoder = None
    if not args.no_touch:
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
        touch_encoder = TouchEncoder(
            encoder_name="vecsetx",
            output_dim=pipeline.backbone.cond_channels,
            trainable=args.train_vecsetx,
            use_position=not args.no_touch_position,
            position_scale="log",
        ).to(device)

    model = TouchTrainingModel(pipeline.ss_generator, touch_encoder)
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
            args.resume, model, optimizer, mode, args.cross_attention_scope
        )

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
            }
            with open(config_path, "w") as file:
                yaml.safe_dump(run_config, file, sort_keys=False)

            if touch_encoder is not None and not args.resume:
                torch.save(
                    touch_adapter_state_dict(raw_model),
                    args.output_dir / "initial_touch_adapter.pt",
                )

        print(f"mode: {mode}")
        print(f"cross-attention scope: {args.cross_attention_scope}")
        print(f"precision: {args.precision}")
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

    total_train_steps = args.epochs * len(train_loader)
    if args.max_steps:
        total_train_steps = min(total_train_steps, args.max_steps)

    for epoch in range(start_epoch, args.epochs):
        step = train_epoch(
            pipeline, model, raw_model, train_loader, optimizer, parameters,
            device, args, epoch, step, total_train_steps, world_size,
            distributed, main_process, run,
        )
        val_loss = validate(
            pipeline, raw_model, val_loader, device, args,
            seed + 1, distributed, rank,
        )
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if main_process:
            run.log({"global_step": step, "loss/val": val_loss})
            print(
                f"epoch {epoch + 1}/{args.epochs} val_loss {val_loss:.6f}"
                f"{' best' if improved else ''}",
                flush=True,
            )
            save_checkpoint(
                args.output_dir / "last.pt", raw_model, optimizer,
                epoch + 1, step, best_loss, mode,
                args.cross_attention_scope,
            )
            if improved:
                save_checkpoint(
                    args.output_dir / "best.pt", raw_model, optimizer,
                    epoch + 1, step, best_loss, mode,
                    args.cross_attention_scope,
                )

        if distributed:
            dist.barrier()
        if args.max_steps and step >= args.max_steps:
            break

    if main_process:
        run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
