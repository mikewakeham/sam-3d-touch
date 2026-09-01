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

from dataloader import build_dataloader, load_data_config


def build_stage1_pipeline(config_path, device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

    class Stage1TrainingPipeline(InferencePipelinePointMap):
        """Stage-1-only setup using the existing inference pipeline methods."""

        def __init__(self, config_path, device):
            config_path = Path(config_path).resolve()
            config = OmegaConf.load(config_path)

            self.device = torch.device(device)
            self.workspace_dir = str(config_path.parent)
            self.ss_condition_input_mapping = list(config.get("ss_condition_input_mapping", ["image"]))

            preprocessor = config.get("ss_preprocessor")
            preprocessor = instantiate(preprocessor) if preprocessor is not None else None
            self.ss_preprocessor = self.init_ss_preprocessor(preprocessor, config.ss_generator_config_path)
            self.ss_generator = self.init_ss_generator(
                config.ss_generator_config_path, config.ss_generator_ckpt_path
            )
            self.ss_condition_embedder = self.init_ss_condition_embedder(
                config.ss_generator_config_path, config.ss_generator_ckpt_path
            )

            self.ss_generator.train()
            self.ss_generator.self_consistency_prob = 0.0
            if hasattr(self.ss_generator.reverse_fn, "p_unconditional"):
                self.ss_generator.reverse_fn.p_unconditional = 0.0

            self.backbone = self.ss_generator.reverse_fn.backbone
            self.backbone.eval()
            if hasattr(self.backbone.condition_embedder, "normalize_images"):
                self.backbone.condition_embedder.normalize_images = True

    return Stage1TrainingPipeline(config_path, device)


class TouchTrainingModel(torch.nn.Module):
    def __init__(self, generator, touch_encoder):
        super().__init__()
        self.generator = generator
        self.touch_encoder = touch_encoder

    def forward(self, targets, touch_xyz, touch_mask, condition_args, condition_kwargs):
        if self.touch_encoder is None:
            loss, _ = self.generator.loss(targets, *condition_args, **condition_kwargs)
        else:
            touch_tokens = self.touch_encoder(touch_xyz, touch_mask)
            loss, _ = self.generator.loss(
                targets, *condition_args, touch_tokens=touch_tokens, **condition_kwargs
            )
        return loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=Path("configs/data1.yaml"))
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/touch_stage1"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cross-attention-learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-touch", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", -1)))
    return parser.parse_args()


def setup_distributed(args):
    local_rank = args.local_rank
    if local_rank < 0:
        return False, 0, 1, torch.device(args.device)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    return True, rank, dist.get_world_size(), torch.device("cuda", local_rank)


def save_run_config(path, args, data_config, world_size):
    from omegaconf import OmegaConf

    if path.exists():
        return

    arguments = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    config = {
        "arguments": arguments,
        "distributed": {
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "workers_per_gpu": args.workers,
            "total_workers": args.workers * world_size,
            "val_workers_per_gpu": args.val_workers,
            "total_val_workers": args.val_workers * world_size,
        },
        "data_config": data_config,
        "pipeline_config": OmegaConf.to_container(
            OmegaConf.load(args.pipeline_config), resolve=True
        ),
    }
    with open(path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def preprocess_batch(pipeline, images, pointmaps):
    items = []
    for image, pointmap in zip(images, pointmaps):
        item = pipeline.preprocess_image(
            image.numpy(), pipeline.ss_preprocessor, pointmap=pointmap.permute(2, 0, 1)
        )
        items.append(item)
    return {key: torch.cat([item[key] for item in items]) for key in items[0]}


def normalize_touch(touch_xyz, inputs, preprocessor):
    from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import _apply_metric_to_ssi

    if not preprocessor.normalize_pointmap:
        return touch_xyz

    normalizer = preprocessor.pointmap_normalizer
    normalized = []
    for points, scale, shift in zip(touch_xyz, inputs["pointmap_scale"], inputs["pointmap_shift"]):
        shape = points.shape
        points = points.reshape(-1, 3)
        if hasattr(normalizer, "point_remapper"):
            points = normalizer.point_remapper(points)
        normalized.append(_apply_metric_to_ssi(points, scale, shift).reshape(shape))
    return torch.stack(normalized)


def make_targets(shape, backbone):
    targets = {}
    for name, mapping in backbone.latent_mapping.items():
        target_shape = (shape.shape[0], mapping.pos_emb.shape[0], mapping.input_layer.in_features)
        targets[name] = shape if name == "shape" else shape.new_zeros(target_shape)
    return targets


def validate(pipeline, model, loader, device, args, seed, distributed, rank):
    if model.touch_encoder is not None:
        model.touch_encoder.eval()
    python_state = random.getstate()
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    total_loss = 0
    total_samples = 0

    with torch.random.fork_rng(devices=cuda_devices):
        random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        with torch.no_grad():
            for batch in loader:
                inputs = preprocess_batch(pipeline, batch["image"], batch["pointmap"])
                touch_xyz = None
                touch_mask = None
                if model.touch_encoder is not None:
                    touch_xyz = normalize_touch(
                        batch["touch_xyz"].to(device), inputs, pipeline.ss_preprocessor
                    )
                    touch_mask = batch["touch_mask"].to(device)
                shape = batch["target_shape"].to(device)
                with torch.autocast(
                    device.type, dtype=torch.bfloat16,
                    enabled=not args.no_amp and device.type == "cuda"
                ):
                    condition_args, condition_kwargs = pipeline.get_condition_input(
                        pipeline.ss_condition_embedder, inputs, pipeline.ss_condition_input_mapping
                    )
                    loss = model(
                        make_targets(shape, pipeline.backbone), touch_xyz, touch_mask,
                        condition_args, condition_kwargs
                    )
                total_loss += loss.item() * shape.shape[0]
                total_samples += shape.shape[0]

    if distributed:
        totals = torch.tensor([total_loss, total_samples], dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        total_loss, total_samples = totals.tolist()

    random.setstate(python_state)
    if model.touch_encoder is not None:
        model.touch_encoder.train()
    return total_loss / total_samples


def save_checkpoint(path, encoder, cross_attention_kv, optimizer, epoch, step, best_val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"touch_encoder": encoder.state_dict() if encoder is not None else None,
         "shape_cross_attention_kv": [module.state_dict() for module in cross_attention_kv],
         "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step,
         "best_val_loss": best_val_loss}, path
    )


def main():
    args = parse_args()
    distributed, rank, world_size, device = setup_distributed(args)
    main_process = rank == 0

    from loguru import logger

    if not main_process:
        logger.remove()

    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder

    if main_process:
        import wandb

    config = load_data_config(args.data_config)
    seed = config.get("seed", 0)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    loader = build_dataloader(
        config, args.batch_size, args.workers, distributed=distributed
    )
    val_config = load_data_config(args.data_config)
    val_config["dataset"]["split"] = "val"
    val_config["touch"]["contacts"]["shuffle_after_selection"] = False
    val_loader = build_dataloader(
        val_config, args.batch_size, args.val_workers, shuffle=False, distributed=distributed
    )

    pipeline = build_stage1_pipeline(args.pipeline_config, device)
    encoder = None if args.no_touch else TouchEncoder(pipeline.backbone.cond_channels).to(device)
    cross_attention_kv = []
    if args.cross_attention_learning_rate:
        cross_attention_kv = [block.cross_attn["shape"].to_kv for block in pipeline.backbone.blocks]
        for module in cross_attention_kv:
            module.requires_grad_(True)

    optimizer_groups = []
    touch_group_index = None
    if encoder is not None:
        touch_group_index = len(optimizer_groups)
        optimizer_groups.append({"params": encoder.parameters(), "lr": args.learning_rate})
    cross_attention_group_index = None
    if cross_attention_kv:
        cross_attention_group_index = len(optimizer_groups)
        optimizer_groups.append({
            "params": [parameter for module in cross_attention_kv for parameter in module.parameters()],
            "lr": args.cross_attention_learning_rate,
        })
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0)
    trainable_parameters = [parameter for group in optimizer_groups for parameter in group["params"]]

    pipeline.ss_generator.loss_weights = {
        name: float(name == "shape") for name in pipeline.backbone.latent_mapping
    }

    start_epoch = 0
    step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if encoder is not None:
            encoder.load_state_dict(checkpoint["touch_encoder"])
        for module, state in zip(cross_attention_kv, checkpoint.get("shape_cross_attention_kv", [])):
            module.load_state_dict(state)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)

    model = TouchTrainingModel(pipeline.ss_generator, encoder)
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[device.index], broadcast_buffers=False
        )
    raw_model = model.module if distributed else model

    if main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_run_config(
            args.output_dir / "config.yaml", args, config, world_size
        )
    if distributed:
        dist.barrier()

    run = None
    if main_process:
        run_config = vars(args).copy()
        run_config["world_size"] = world_size
        run_config["global_batch_size"] = args.batch_size * world_size
        run = wandb.init(
            project="sam-3d-touch", name=args.output_dir.name,
            dir=str(args.output_dir), config=run_config
        )
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")
        print(f"train samples: {len(loader.dataset)}")
        print(f"val samples: {len(val_loader.dataset)}")
        print(f"GPUs: {world_size}")
        print(f"batch size: {args.batch_size} per GPU, {args.batch_size * world_size} global")
        print(f"workers: {args.workers} train, {args.val_workers} val per GPU")
        touch_parameters = sum(p.numel() for p in encoder.parameters()) if encoder is not None else 0
        print(f"touch parameters: {touch_parameters:,}")
    kv_parameters = sum(p.numel() for module in cross_attention_kv for p in module.parameters())
    if main_process:
        print(f"cross-attention K/V parameters: {kv_parameters:,}")
        print(f"condition width: {pipeline.backbone.cond_channels}")

    total_train_steps = args.epochs * len(loader)
    if args.max_steps:
        total_train_steps = min(total_train_steps, args.max_steps)

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            loader.sampler.set_epoch(epoch)
        if raw_model.touch_encoder is not None:
            raw_model.touch_encoder.train()
        train_loss = 0
        train_samples = 0
        log_start_time = time.perf_counter()
        for batch in loader:
            inputs = preprocess_batch(pipeline, batch["image"], batch["pointmap"])
            touch_xyz = None
            touch_mask = None
            if raw_model.touch_encoder is not None:
                touch_xyz = normalize_touch(
                    batch["touch_xyz"].to(device), inputs, pipeline.ss_preprocessor
                )
                touch_mask = batch["touch_mask"].to(device)
            shape = batch["target_shape"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp and device.type == "cuda"):
                with torch.no_grad():
                    condition_args, condition_kwargs = pipeline.get_condition_input(
                        pipeline.ss_condition_embedder, inputs, pipeline.ss_condition_input_mapping
                    )
                loss = model(
                    make_targets(shape, pipeline.backbone), touch_xyz, touch_mask,
                    condition_args, condition_kwargs
                )

            loss.backward()
            if args.gradient_clip:
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
            optimizer.step()
            step += 1
            train_loss += loss.item() * shape.shape[0]
            train_samples += shape.shape[0]

            if step == 1 or step % args.log_every == 0:
                elapsed = time.perf_counter() - log_start_time
                totals = torch.tensor([train_loss, train_samples], dtype=torch.float64, device=device)
                elapsed = torch.tensor(elapsed, dtype=torch.float64, device=device)
                if distributed:
                    dist.all_reduce(totals)
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

                global_loss, global_samples = totals.tolist()
                mean_train_loss = global_loss / global_samples
                samples_per_second = global_samples / elapsed.item()
                eta_seconds = (
                    max(total_train_steps - step, 0) * args.batch_size * world_size
                    / samples_per_second
                )
                if main_process:
                    train_eta = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    print(
                        f"epoch {epoch + 1}/{args.epochs} step {step} train_loss {mean_train_loss:.6f} "
                        f"throughput {samples_per_second:.2f} samples/s train_eta {train_eta}", flush=True
                    )
                    metrics = {
                        "global_step": step,
                        "loss/train": mean_train_loss,
                        "performance/samples_per_second": samples_per_second,
                        "performance/train_eta_seconds": eta_seconds,
                    }
                    if touch_group_index is not None:
                        metrics["learning_rate/touch_encoder"] = optimizer.param_groups[touch_group_index]["lr"]
                    if args.gradient_clip:
                        metrics["optimization/gradient_norm"] = gradient_norm.item()
                    if cross_attention_group_index is not None:
                        metrics["learning_rate/cross_attention"] = optimizer.param_groups[cross_attention_group_index]["lr"]
                    run.log(metrics)
                train_loss = 0
                train_samples = 0
                log_start_time = time.perf_counter()
            if main_process and args.save_every and step % args.save_every == 0:
                save_checkpoint(
                    args.output_dir / "last.pt", raw_model.touch_encoder, cross_attention_kv,
                    optimizer, epoch, step, best_val_loss
                )
            if args.max_steps and step >= args.max_steps:
                break

        val_loss = validate(
            pipeline, raw_model, val_loader, device, args, seed + 1, distributed, rank
        )
        if main_process:
            run.log({"global_step": step, "loss/val": val_loss})
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            if main_process:
                save_checkpoint(
                    args.output_dir / "best.pt", raw_model.touch_encoder, cross_attention_kv,
                    optimizer, epoch + 1, step, best_val_loss
                )
        if main_process:
            save_checkpoint(
                args.output_dir / "last.pt", raw_model.touch_encoder, cross_attention_kv,
                optimizer, epoch + 1, step, best_val_loss
            )
            suffix = " best" if improved else ""
            print(f"epoch {epoch + 1}/{args.epochs} step {step} val_loss {val_loss:.6f}{suffix}", flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    if main_process:
        run.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
