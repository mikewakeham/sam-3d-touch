import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch
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


def save_checkpoint(path, encoder, optimizer, epoch, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"touch_encoder": encoder.state_dict(), "optimizer": optimizer.state_dict(),
         "epoch": epoch, "step": step}, path
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
    pipeline = Stage1TrainingPipeline(args.pipeline_config, device)
    encoder = TouchEncoder(pipeline.backbone.cond_channels).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=0)

    pipeline.ss_generator.loss_weights = {
        name: float(name == "shape") for name in pipeline.backbone.latent_mapping
    }

    start_epoch = 0
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        encoder.load_state_dict(checkpoint["touch_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        step = checkpoint["step"]

    print(f"samples: {len(loader.dataset)}")
    print(f"touch parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"condition width: {pipeline.backbone.cond_channels}")

    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        for batch in loader:
            inputs = preprocess_batch(pipeline, batch["image"], batch["pointmap"])
            touch_xyz = normalize_touch(batch["touch_xyz"].to(device), inputs, pipeline.ss_preprocessor)
            shape = batch["target_shape"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp and device.type == "cuda"):
                touch_tokens = encoder(touch_xyz)
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
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.gradient_clip)
            optimizer.step()
            step += 1

            if step == 1 or step % args.log_every == 0:
                print(f"epoch {epoch + 1}/{args.epochs} step {step} loss {loss.item():.6f}", flush=True)
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(args.output_dir / f"step_{step:08d}.pt", encoder, optimizer, epoch, step)
            if args.max_steps and step >= args.max_steps:
                break

        save_checkpoint(args.output_dir / "last.pt", encoder, optimizer, epoch + 1, step)
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
