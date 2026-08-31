import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dataloader import build_dataloader, load_data_config
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import _apply_metric_to_ssi
from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
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
        self.ss_generator = self.init_ss_generator(config.ss_generator_config_path, config.ss_generator_ckpt_path)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=Path("configs/data1.yaml"))
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/touch_stage1"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cross-attention-learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def preprocess_batch(pipeline, images, pointmaps):
    items = []
    for image, pointmap in zip(images, pointmaps):
        item = pipeline.preprocess_image(
            image.numpy(), pipeline.ss_preprocessor, pointmap=pointmap.permute(2, 0, 1)
        )
        items.append(item)
    return {key: torch.cat([item[key] for item in items]) for key in items[0]}


def normalize_touch(touch_xyz, inputs, preprocessor):
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


def validate(pipeline, encoder, loader, device, args, seed):
    encoder.eval()
    python_state = random.getstate()
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    total_loss = 0
    total_samples = 0

    with torch.random.fork_rng(devices=cuda_devices):
        random.seed(seed)
        torch.manual_seed(seed)
        with torch.no_grad():
            for batch in loader:
                inputs = preprocess_batch(pipeline, batch["image"], batch["pointmap"])
                touch_xyz = normalize_touch(batch["touch_xyz"].to(device), inputs, pipeline.ss_preprocessor)
                touch_mask = batch["touch_mask"].to(device)
                shape = batch["target_shape"].to(device)
                with torch.autocast(
                    device.type, dtype=torch.bfloat16,
                    enabled=not args.no_amp and device.type == "cuda"
                ):
                    touch_tokens = encoder(touch_xyz, touch_mask)
                    condition_args, condition_kwargs = pipeline.get_condition_input(
                        pipeline.ss_condition_embedder, inputs, pipeline.ss_condition_input_mapping
                    )
                    loss, _ = pipeline.ss_generator.loss(
                        make_targets(shape, pipeline.backbone), *condition_args,
                        touch_tokens=touch_tokens, **condition_kwargs
                    )
                total_loss += loss.item() * shape.shape[0]
                total_samples += shape.shape[0]

    random.setstate(python_state)
    encoder.train()
    return total_loss / total_samples


def save_checkpoint(path, encoder, cross_attention_kv, optimizer, epoch, step, best_val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"touch_encoder": encoder.state_dict(),
         "shape_cross_attention_kv": [module.state_dict() for module in cross_attention_kv],
         "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step,
         "best_val_loss": best_val_loss}, path
    )


def main():
    args = parse_args()
    config = load_data_config(args.data_config)
    seed = config.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device)
    loader = build_dataloader(config, args.batch_size, args.workers)
    val_config = load_data_config(args.data_config)
    val_config["dataset"]["split"] = "val"
    val_config["touch"]["contacts"]["shuffle_after_selection"] = False
    val_loader = build_dataloader(val_config, args.batch_size, args.workers, shuffle=False)

    pipeline = Stage1TrainingPipeline(args.pipeline_config, device)
    encoder = TouchEncoder(pipeline.backbone.cond_channels).to(device)
    cross_attention_kv = []
    if args.cross_attention_learning_rate:
        cross_attention_kv = [block.cross_attn["shape"].to_kv for block in pipeline.backbone.blocks]
        for module in cross_attention_kv:
            module.requires_grad_(True)

    optimizer_groups = [{"params": encoder.parameters(), "lr": args.learning_rate}]
    if cross_attention_kv:
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
        encoder.load_state_dict(checkpoint["touch_encoder"])
        for module, state in zip(cross_attention_kv, checkpoint.get("shape_cross_attention_kv", [])):
            module.load_state_dict(state)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="sam-3d-touch", name=args.output_dir.name,
        dir=str(args.output_dir), config=vars(args)
    )
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")
    print(f"train samples: {len(loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
    print(f"touch parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    kv_parameters = sum(p.numel() for module in cross_attention_kv for p in module.parameters())
    print(f"cross-attention K/V parameters: {kv_parameters:,}")
    print(f"condition width: {pipeline.backbone.cond_channels}")

    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        train_loss = 0
        train_samples = 0
        for batch in loader:
            inputs = preprocess_batch(pipeline, batch["image"], batch["pointmap"])
            touch_xyz = normalize_touch(batch["touch_xyz"].to(device), inputs, pipeline.ss_preprocessor)
            touch_mask = batch["touch_mask"].to(device)
            shape = batch["target_shape"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp and device.type == "cuda"):
                touch_tokens = encoder(touch_xyz, touch_mask)
                with torch.no_grad():
                    condition_args, condition_kwargs = pipeline.get_condition_input(
                        pipeline.ss_condition_embedder, inputs, pipeline.ss_condition_input_mapping
                    )
                loss, _ = pipeline.ss_generator.loss(
                    make_targets(shape, pipeline.backbone), *condition_args,
                    touch_tokens=touch_tokens, **condition_kwargs
                )

            loss.backward()
            if args.gradient_clip:
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
            optimizer.step()
            step += 1
            train_loss += loss.item() * shape.shape[0]
            train_samples += shape.shape[0]

            if step == 1 or step % args.log_every == 0:
                mean_train_loss = train_loss / train_samples
                print(f"epoch {epoch + 1}/{args.epochs} step {step} train_loss {mean_train_loss:.6f}", flush=True)
                metrics = {
                    "global_step": step,
                    "loss/train": mean_train_loss,
                    "learning_rate/touch_encoder": optimizer.param_groups[0]["lr"],
                }
                if args.gradient_clip:
                    metrics["optimization/gradient_norm"] = gradient_norm.item()
                if cross_attention_kv:
                    metrics["learning_rate/cross_attention"] = optimizer.param_groups[1]["lr"]
                run.log(metrics)
                train_loss = 0
                train_samples = 0
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(
                    args.output_dir / "last.pt", encoder, cross_attention_kv,
                    optimizer, epoch, step, best_val_loss
                )
            if args.max_steps and step >= args.max_steps:
                break

        val_loss = validate(pipeline, encoder, val_loader, device, args, seed + 1)
        run.log({"global_step": step, "loss/val": val_loss})
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            save_checkpoint(
                args.output_dir / "best.pt", encoder, cross_attention_kv,
                optimizer, epoch + 1, step, best_val_loss
            )
        save_checkpoint(
            args.output_dir / "last.pt", encoder, cross_attention_kv,
            optimizer, epoch + 1, step, best_val_loss
        )
        suffix = " best" if improved else ""
        print(f"epoch {epoch + 1}/{args.epochs} step {step} val_loss {val_loss:.6f}{suffix}", flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    run.finish()


if __name__ == "__main__":
    main()
