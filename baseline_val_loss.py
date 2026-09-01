import argparse
import os
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import torch
import torch.distributed as dist
import yaml

from dataloader import build_dataloader, load_data_config
from train import TouchTrainingModel, build_stage1_pipeline, setup_distributed, validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/s1_official_baseline"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", -1)))
    return parser.parse_args()


def main():
    args = parse_args()
    distributed, rank, world_size, device = setup_distributed(args)
    main_process = rank == 0

    from loguru import logger

    if not main_process:
        logger.remove()

    config = load_data_config(args.data_config)
    seed = config.get("seed", 0)
    train_loader = build_dataloader(
        config, args.batch_size, args.workers, distributed=distributed
    )
    val_config = load_data_config(args.data_config)
    val_config["dataset"]["split"] = "val"
    val_config["touch"]["contacts"]["shuffle_after_selection"] = False
    val_loader = build_dataloader(
        val_config, args.batch_size, args.workers, shuffle=False, distributed=distributed
    )

    pipeline = build_stage1_pipeline(args.pipeline_config, device)
    pipeline.ss_generator.loss_weights = {
        name: float(name == "shape") for name in pipeline.backbone.latent_mapping
    }
    model = TouchTrainingModel(pipeline.ss_generator, None)

    if main_process:
        print(f"official SAM3D validation samples: {len(val_loader.dataset)}", flush=True)
    val_loss = validate(
        pipeline, model, val_loader, device, args, seed + 1, distributed, rank
    )

    total_steps = args.epochs * len(train_loader)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    if main_process:
        import wandb

        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "official_val_loss": val_loss,
            "global_batch_size": args.batch_size * world_size,
            "world_size": world_size,
            "steps_per_epoch": len(train_loader),
            "epochs": args.epochs,
            "total_steps": total_steps,
            "seed": seed,
        }
        with open(args.output_dir / "result.yaml", "w") as file:
            yaml.safe_dump(result, file, sort_keys=False)

        run = wandb.init(
            project="sam-3d-touch",
            name=args.output_dir.name,
            dir=str(args.output_dir),
            config=result,
            job_type="evaluation",
            tags=["official", "baseline"],
        )
        run.define_metric("global_step")
        run.define_metric("loss/val", step_metric="global_step")
        run.log({"global_step": 0, "loss/val": val_loss})
        run.log({"global_step": total_steps, "loss/val": val_loss})
        run.summary["official_val_loss"] = val_loss
        run.finish()
        print(f"official SAM3D val_loss {val_loss:.6f}", flush=True)
        print(f"saved baseline to {args.output_dir}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
