"""Three matched camera-frame treatments: training and fixed-bank loss only.

One GPU per process; no decoder, rollouts, reconstruction metrics, or feature cache.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = next(p for p in Path(__file__).resolve().parents if (p/'train.py').is_file())
sys.path.insert(0, str(REPO))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation-dir', type=Path, required=True)
    p.add_argument('--arm', choices=['object_stock', 'camera_stock', 'camera_shared'], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--data-config', type=Path, default=Path('configs/data_full_surface.yaml'))
    p.add_argument('--pipeline-config', type=Path, default=Path('checkpoints/hf/pipeline.yaml'))
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=4, help='Microbatch per GPU')
    p.add_argument('--global-batch-size', type=int, default=4)
    p.add_argument('--validate-every', type=int, default=100)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--train-scope', choices=['shape_cross_attention', 'shape_full'], default='shape_cross_attention')
    p.add_argument('--visual-dropout', type=float, default=.5)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--cross-attention-learning-rate', type=float, default=1e-5)
    p.add_argument('--gradient-clip', type=float, default=1.)
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.global_batch_size, a.validate_every) < 1:
        p.error('Step and batch counts must be positive')
    if a.global_batch_size % a.batch_size or a.workers < 0:
        p.error('Global batch must be divisible by microbatch; workers must be nonnegative')
    if not 0 <= a.visual_dropout <= 1 or a.gradient_clip <= 0:
        p.error('Invalid dropout or gradient clip')
    return a


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def build_model(settings, device):
    """Shared constructor for training and subsequent experiment-specific evaluation."""
    from train import build_stage1_pipeline, TouchTrainingModel, build_optimizer
    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
    pipeline = build_stage1_pipeline(settings['pipeline_config'], device)
    encoder = TouchEncoder(encoder_name='vecsetx', output_dim=pipeline.backbone.cond_channels,
                           trainable=False, use_position=False, position_scale='log').to(device)
    model = TouchTrainingModel(pipeline.ss_generator, encoder,
                               visual_dropout=settings['visual_dropout'])
    optimizer, parameters = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(**settings))
    return pipeline, model, optimizer, parameters


def restore_for_evaluation(checkpoint_path, device, pipeline_config=None):
    """Restore with this experiment's prepare_batch, never vanilla evaluate.py.

    Returns pipeline, model, metadata; caller uses metadata['settings']['arm'].
    No optimizer state or frozen pretrained weights are duplicated in the file.
    """
    import torch
    from train import load_trainable_state_dict
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 'camera_frame_formal_v1':
        raise ValueError('Not a camera-frame formal checkpoint')
    settings = dict(checkpoint['metadata']['settings'])
    if pipeline_config is not None: settings['pipeline_config'] = str(pipeline_config)
    pipeline, model, optimizer, _ = build_model(settings, device)
    del optimizer
    load_trainable_state_dict(model, checkpoint['trainable_state'])
    model.touch_encoder.eval()
    return pipeline, model, checkpoint['metadata']


def main():
    args = arguments()
    if args.output_dir.exists(): raise FileExistsError('Use a fresh output directory')
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader, Subset
    from dataloader import TouchDataset, collate_touch_batch
    from train import amp, make_visual_drop_mask, trainable_state_dict
    from experiments.coordinate_system.scripts.camera_frame_formal.training_runtime import (
        ARMS, fixed_rng, prepare_batch, training_indices, validation_indices, accumulate_update)
    if not torch.cuda.is_available(): raise RuntimeError('Run on a cluster GPU')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device); torch.set_float32_matmul_precision('high')
    prep = args.preparation_dir.resolve()
    report = json.loads((prep/'preparation.json').read_text())
    if not report['complete']: raise ValueError('Preparation is incomplete')
    config = yaml.safe_load(args.data_config.read_text())
    if sha(args.data_config) != report['input_sha256'][report['settings']['data_config']]:
        raise ValueError('Data config changed since preparation')
    if sha(args.pipeline_config) != report['input_sha256'][report['settings']['pipeline_config']]:
        raise ValueError('Pipeline config changed since preparation')
    convention = ARMS[args.arm][0]
    datasets = {}
    for split in ('train', 'held_view', 'val'):
        if not report['selection'][split]:
            continue
        manifest = prep/f'{convention}_{split}.jsonl'
        if sha(manifest) != report['manifest_sha256'][manifest.name]:
            raise ValueError(f'Manifest changed: {manifest}')
        c = copy.deepcopy(config)
        c['dataset'].update(manifest=str(manifest), split_file=str(prep/'selected_splits.json'),
                            split='val' if split == 'val' else 'train')
        datasets[split] = TouchDataset(c)
        if [r['sample_id'] for r in datasets[split].records] != report['selection'][split]:
            raise ValueError(f'Selection differs: {split}')
    eval_datasets = dict(datasets)
    eval_datasets['train'] = Subset(datasets['train'], validation_indices(datasets['train'].records))
    eval_ids = {}
    for split, dataset in eval_datasets.items():
        records = datasets[split].records
        indices = dataset.indices if isinstance(dataset, Subset) else range(len(records))
        eval_ids[split] = [records[i]['sample_id'] for i in indices]
    loader_options = dict(batch_size=args.batch_size, collate_fn=collate_touch_batch,
                          pin_memory=True, num_workers=args.workers)
    if args.workers:
        loader_options.update(multiprocessing_context='spawn', persistent_workers=True)
    # Loader-owned RNG prevents worker setup from changing flow noise or initialization.
    eval_loaders = {s: DataLoader(d, shuffle=False, generator=torch.Generator().manual_seed(args.seed),
                                 **loader_options) for s, d in eval_datasets.items()}
    order = list(training_indices(len(datasets['train']), args.steps*args.global_batch_size, args.seed))
    train_loader = DataLoader(datasets['train'], sampler=order,
                             generator=torch.Generator().manual_seed(args.seed), **loader_options)
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings['cross_attention_scope'] = 'full'
    with fixed_rng(args.seed, device):
        pipeline, model, optimizer, parameters = build_model(settings, device)
    # Keep the same modes as production training; only the touch module switches for validation.
    metadata = dict(settings=settings, selection=report['selection'], validation_ids=eval_ids,
                    training_samples=len(datasets['train']), validation_counts={s: len(d) for s,d in eval_datasets.items()},
                    preparation_sha256=sha(prep/'preparation.json'),
                    training_order_sha256=hashlib.sha256(np.asarray(order, dtype='<i8').tobytes()).hexdigest(),
                    source_sha256={str(p.relative_to(REPO)): sha(p) for p in [REPO/'train.py', REPO/'dataloader.py',
                        REPO/'sam3d_objects/model/backbone/dit/embedder/touch.py', *Path(__file__).parent.glob('*.py')]},
                    trainable_parameters=sum(p.numel() for p in parameters),
                    runtime=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name()),
                    protocol='All arms raw camera surface -> native VecSetX normalization. '
                             'Only camera_shared replaces BOTH pointmap normalizers. Target unit cube unchanged. '
                             'Validation: visual present; fixed native flow noise/time, one draw per observation, sample-weighted loss. '
                             'Different target conventions need final reconstruction evaluation; losses alone do not prove success.')
    initial_hash = hashlib.sha256()
    for name, value in trainable_state_dict(model).items():
        initial_hash.update(name.encode()); initial_hash.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    metadata['initial_trainable_sha256'] = initial_hash.hexdigest()
    out = args.output_dir.resolve(); out.mkdir(parents=True)
    write_json(out/'config.json', metadata)
    result = dict(complete=False, metadata=metadata, validations=[])
    def log(metrics):
        with (out/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(metrics, allow_nan=False)+'\n')

    @torch.no_grad()
    def validate(step):
        was_training = model.touch_encoder.training
        model.touch_encoder.eval()
        metrics = {'global_step': step}
        try:
            for split_index, (split, loader) in enumerate(eval_loaders.items()):
                total, count = 0., 0
                for index, batch in enumerate(loader):
                    seed = args.seed + 10_000_019 + split_index*1_000_003 + index
                    with fixed_rng(seed, device):
                        prepared = prepare_batch(pipeline, batch, device, args.precision, args.arm)
                    # Separate context guarantees identical flow draws even if preprocessing consumes RNG.
                    with fixed_rng(seed + 100_000_007, device), amp(device, args.precision):
                        loss = model(*prepared)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite validation loss')
                    n = len(batch['target_shape']); total += float(loss)*n; count += n
                metrics[f'loss/{split}_fixed'] = total/count
        finally:
            model.touch_encoder.train(was_training)
        result['validations'].append(metrics); log(metrics)
        write_json(out/'results.json', result)
        print(f'Validation step {step}: '+json.dumps(metrics), flush=True)

    def checkpoint(step, final=False):
        # One rolling weights file; overwrite atomically. No repeated optimizer/frozen-weight copies.
        metadata['checkpoint_step'] = step
        temp = out/'checkpoint.tmp'
        torch.save(dict(format='camera_frame_formal_v1', metadata=metadata,
                        trainable_state=trainable_state_dict(model)), temp)
        temp.replace(out/('final.pt' if final else 'latest.pt'))
        if final and (out/'latest.pt').exists(): (out/'latest.pt').unlink()

    validate(0)
    iterator = iter(train_loader)
    accumulation = args.global_batch_size//args.batch_size
    model.touch_encoder.train()
    window_loss, update_times, started = [], [], time.perf_counter()
    for step in range(1, args.steps+1):
        micro_index = 0
        def loss_fn(batch):
            nonlocal micro_index
            address = (step-1)*accumulation + micro_index
            micro_index += 1
            with fixed_rng(args.seed+address+200_000_033, device):
                prepared = prepare_batch(pipeline, batch, device, args.precision, args.arm)
            drop = make_visual_drop_mask(len(batch['target_shape']), args.visual_dropout,
                                         device, args.seed, address, 0)
            with fixed_rng(args.seed+address+300_000_007, device), amp(device, args.precision):
                return model(*prepared, visual_drop_mask=drop)
        update_start = time.perf_counter()
        loss, grad = accumulate_update((next(iterator) for _ in range(accumulation)), loss_fn,
                                       optimizer, parameters, args.global_batch_size, args.gradient_clip)
        update_times.append(time.perf_counter()-update_start)
        window_loss.append(loss)
        if step == 1 or step % 10 == 0 or step == args.steps:
            metrics = dict(global_step=step, **{'loss/train': sum(window_loss)/len(window_loss),
                'optimization/gradient_norm': grad, 'exposure/epochs': step*args.global_batch_size/len(datasets['train']),
                'performance/elapsed_seconds': time.perf_counter()-started,
                'performance/seconds_per_update': sum(update_times)/len(update_times),
                'performance/remaining_training_seconds': (args.steps-step)*sum(update_times)/len(update_times)})
            log(metrics); window_loss.clear(); update_times.clear()
            print(f'{args.arm} step {step}/{args.steps} loss {metrics["loss/train"]:.6f} '
                  f'train ETA {metrics["performance/remaining_training_seconds"]/60:.1f} min', flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            checkpoint(step, final=step == args.steps)
            validate(step)
    result['complete'] = True
    result['checkpoint'] = 'final.pt'
    write_json(out/'results.json', result)



if __name__ == '__main__': main()
