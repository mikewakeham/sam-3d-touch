"""Run the existing trainer, keeping a local metrics log and a step-zero checkpoint."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.encoders.prepare_data import experiment_path


def main():
    import torch
    import wandb
    import train

    args = train.parse_args()
    output = experiment_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ['WANDB_DIR'] = str(output)
    os.environ['WANDB_CACHE_DIR'] = str(output / 'wandb_cache')
    os.environ['WANDB_DATA_DIR'] = str(output / 'wandb_data')
    os.environ.setdefault('WANDB_MODE', 'offline')
    original_init = wandb.init

    def init_run(*values, **settings):
        run = original_init(*values, **settings)
        original_log = run.log

        def log(metrics, *values, **settings):
            metrics = dict(metrics)
            if torch.cuda.is_available():
                metrics['performance/peak_gpu_gib'] = torch.cuda.max_memory_allocated() / 1024**3
            with (output / 'metrics.jsonl').open('a') as file:
                file.write(json.dumps(metrics) + '\n')
            return original_log(metrics, *values, **settings)

        run.log = log
        return run

    original_epoch = train.train_epoch

    def train_epoch(pipeline, model, raw_model, loader, optimizer, parameters,
                    device, args, epoch, step, total_train_steps, world_size,
                    distributed, main_process, run, seed=0):
        if step == 0:
            data = train.configure_encoder_data(train.load_data_config(args.data_config), args.point_encoder)
            data['dataset']['split'] = 'val'
            validation = train.build_dataloader(
                data, args.batch_size, args.val_workers, shuffle=False, distributed=distributed,
                include_touch=not args.no_touch or args.shared_pointmap_normalization,
                oracle_point_frame=args.oracle_point_frame,
            )
            loss = train.validate(pipeline, raw_model, validation, device, args, seed + 1, distributed,
                                  torch.distributed.get_rank() if distributed else 0)
            if main_process:
                run.log({'global_step': 0, 'loss/val': loss})
                print(f'initial val_loss {loss:.6f}', flush=True)
                train.save_checkpoint(output / 'initial.pt', raw_model, optimizer, 0, 0, loss,
                                      'image' if args.no_touch else ('image_touch_joint' if args.joint_pointmap else 'image_touch'),
                                      args.cross_attention_scope, args.train_scope)
            del validation
        return original_epoch(pipeline, model, raw_model, loader, optimizer, parameters,
                              device, args, epoch, step, total_train_steps, world_size,
                              distributed, main_process, run, seed)

    wandb.init = init_run
    train.train_epoch = train_epoch
    train.main()


if __name__ == '__main__':
    main()
