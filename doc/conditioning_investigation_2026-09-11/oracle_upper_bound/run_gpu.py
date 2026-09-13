"""Full-data Stage-1 upper bound. One process/GPU per arm; no Stage 2.

Use --preflight first to compare this live point-conditioning path with the
historical successful oracle checkpoint. Training starts afresh, not from it.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INV = HERE.parent
sys.path[:0] = [str(HERE), str(INV), str(REPO)]
os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import TouchDataset, collate_touch_batch, load_data_config
from experiments.noisy_target.evaluate import tensor_digest
from train import (amp, build_optimizer, build_stage1_pipeline,
                   load_trainable_state_dict, prepare_batch, trainable_state_dict)
from tiny_fit_gpu import parameter_digest
from model import UpperBoundModel
from protocol import ARMS, arm_config, check_splits, digest, epoch_schedule, evaluation_groups


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def sources():
    paths = [HERE / n for n in ('run_gpu.py', 'model.py', 'protocol.py', 'assess.py')]
    paths += [REPO / n for n in ('train.py', 'dataloader.py',
              'sam3d_objects/model/backbone/dit/embedder/touch.py')]
    return {str(p.relative_to(REPO)): sha(p) for p in paths}


def build(args, arm):
    config = arm_config(arm)
    seed_all(args.seed)
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    pipeline = build_stage1_pipeline(args.pipeline_config, device)
    encoder = None
    if config['touch']:
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
        encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels,
                               trainable=False, use_position=False).to(device).eval()
    model = UpperBoundModel(pipeline.ss_generator, encoder, config['oracle'])
    opt, params = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope='full'))
    gen = pipeline.ss_generator
    gen.reverse_fn.training = True
    assert gen.reverse_fn.p_unconditional == gen.self_consistency_prob == gen.fm_eps_max == 0
    assert gen.loss_weights['shape'] == 1 and all(v == 0 for k, v in gen.loss_weights.items() if k != 'shape')
    return config, device, pipeline, model, opt, params


def datasets(args, config):
    data = load_data_config(args.data_config)
    if data.get('touch', {}).get('source') != 'full_surface':
        raise ValueError('This experiment requires full-surface observations')
    result = {}
    for split in ('train', 'val'):
        dc = copy.deepcopy(data); dc['dataset']['split'] = split
        result[split] = TouchDataset(dc, include_touch=config['touch'], oracle_point_frame=config['oracle'])
    check_splits(result['train'].records, result['val'].records)
    return result


def prepare(pipeline, batch, device, config):
    return prepare_batch(pipeline, batch, device, 'bf16', config['touch'], False, config['oracle'])


def preflight(args):
    """No optimization: historical loss plus live/direct forward/gradient parity."""
    from probe_alignment_tolerance_gpu import locate_dropout
    reference = json.loads((INV / 'visual_dropout_returned_manual/oracle/results.json').read_text())
    for path, expected in reference['source_sha256'].items():
        if sha(REPO / path) != expected:
            raise ValueError(f'Historical source changed: {path}')
    fit_dir = locate_dropout(args.oracle_fit_dir, reference)
    config, device, pipeline, model, optimizer, params = build(args, 'oracle_dropout')
    archive = torch.load(fit_dir / 'checkpoint_1000.pt', map_location='cpu', weights_only=True)
    assert archive['settings'] == reference['settings'] and archive['step'] == 1000
    assert archive['conditioning_config'] == model.conditioning_config
    load_trainable_state_dict(model, archive['model']); del archive
    assert parameter_digest(model.named_parameters()) == reference['final_all_parameters_sha256']
    ds = datasets(args, config)
    original_records = list(ds['train'].records)
    lookup = {r['sample_id']: r for r in original_records}
    report = dict(kind='oracle_upper_bound_preflight', complete=False,
                  source_sha256=sources(), settings={'seed': args.seed},
                  pipeline_sha256=sha(args.pipeline_config), data_config_sha256=sha(args.data_config),
                  historical_reference_sha256=sha(INV / 'visual_dropout_returned_manual/oracle/results.json'),
                  counts=check_splits(original_records, ds['val'].records), rows=[], gradient_parity=[],
                  checkpoint=str(fit_dir / 'checkpoint_1000.pt'),
                  checkpoint_sha256=sha(fit_dir / 'checkpoint_1000.pt'),
                  runtime={'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()})
    # One fitted and one reserved historical view group; no new fitting test.
    for group in (0, 4):
        metadata = reference['input_batches'][group]
        ds['train'].records = [lookup[s] for s in metadata['sample_ids']]
        preparation_start = time.perf_counter()
        batch = collate_touch_batch([ds['train'][i] for i in range(4)])
        prepared = prepare(pipeline, batch, device, config)
        torch.cuda.synchronize()
        preparation_seconds = time.perf_counter() - preparation_start
        targets, ca, kw, points, mask = prepared
        encoder = model.touch_encoder
        with torch.no_grad(), amp(device, 'bf16'):
            pp, mm, _, _ = encoder.prepare_points(points, mask)
            features = encoder.encoder.encode(pp, mm)['x'].detach()
        actual = dict(image_sha256=tensor_digest(batch['image']),
                      pointmap_sha256=tensor_digest(batch['pointmap']),
                      target_sha256=tensor_digest(targets['shape']), features_sha256=tensor_digest(features))
        assert all(actual[k] == metadata[k] for k in actual), 'Historical inputs/features changed'
        for drop in (False, True):
            state = 'visual_zero' if drop else 'visual_present'
            values = []
            for draw in range(8):
                seed_all(400000 + 29 + draw)
                with torch.no_grad(), amp(device, 'bf16'):
                    value = model(*prepared, drop_visual=drop)
                values.append(float(value))
            expected = reference['final_fresh'][state][group]['fresh_noise_native_losses']
            # Historical cross-device BF16 replay is numerical, not bitwise.
            np.testing.assert_allclose(values, expected, rtol=.001, atol=1e-6)
            report['rows'].append(dict(group=group, state=state, losses=values,
                historical_max_abs_error=float(np.max(np.abs(np.array(values) - expected)))))
            if group == 0:
                optimizer.zero_grad(set_to_none=True); seed_all(400029)
                torch.cuda.synchronize(); update_start = time.perf_counter()
                with amp(device, 'bf16'):
                    live = model(*prepared, drop_visual=drop)
                live.backward()
                torch.cuda.synchronize(); forward_backward_seconds = time.perf_counter() - update_start
                grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
                optimizer.zero_grad(set_to_none=True); seed_all(400029)
                with amp(device, 'bf16'):
                    tokens = encoder.output_projection(features) + encoder.touch_embedding
                    visual = torch.zeros_like(ca[0]) if drop else ca[0]
                    direct, _ = model.generator.loss(targets, visual, touch_tokens=tokens)
                direct.backward()
                torch.testing.assert_close(live, direct, rtol=1e-5, atol=1e-7)
                current = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
                assert set(current) == set(grads)
                for name, grad in current.items():
                    torch.testing.assert_close(grad.cpu(), grads[name], rtol=1e-4, atol=1e-7)
                assert not any(p.grad is not None for p in encoder.encoder.parameters())
                report['gradient_parity'].append(dict(state=state, tensors=len(grads), passed=True,
                    forward_backward_seconds=forward_backward_seconds))
                del grads, current
            write_json(args.output_dir / 'preflight.partial.json', report)
        report.setdefault('batch_preparation_seconds', []).append(preparation_seconds)
    assert parameter_digest(model.named_parameters()) == reference['final_all_parameters_sha256']
    assert not optimizer.state, 'Preflight must not update the optimizer'
    with torch.no_grad(), amp(device, 'bf16'):
        model.constant_features = features[:1].clone()
        constant_a = model.surface_tokens(points, mask)
        constant_b = model.surface_tokens(points.roll(1, 0) * 2, mask)
        expected = encoder.output_projection(features[:1].expand(4, -1, -1)) + encoder.touch_embedding
        torch.testing.assert_close(constant_a, constant_b, rtol=0, atol=0)
        torch.testing.assert_close(constant_a, expected, rtol=0, atol=0)
        model.constant_features = None
    report['constant_feature_invariance_passed'] = True
    report['timing_scope'] = ('Warm historical forward/backward excludes Adam, logging and assessment; '
                             'preparation includes data IO/DINO. Estimates are not measured full-run throughput.')
    report['complete'] = True
    write_json(args.output_dir / 'preflight.json', report)
    print('PASS: historical oracle replay and live/direct gradient parity. No updates.', flush=True)


def training(args):
    gate = json.loads(args.preflight_report.read_text())
    if not gate.get('complete') or gate.get('kind') != 'oracle_upper_bound_preflight' or gate['source_sha256'] != sources():
        raise ValueError('Need a completed preflight for these exact runtime sources')
    if gate['pipeline_sha256'] != sha(args.pipeline_config) or gate['data_config_sha256'] != sha(args.data_config):
        raise ValueError('Pipeline/data config differs from the preflight')
    config, device, pipeline, model, optimizer, params = build(args, args.arm)
    ds = datasets(args, config)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.pipeline_config)
    data = load_data_config(args.data_config)
    def resolve(path):
        p = Path(path)
        return p if p.is_absolute() else Path(data['dataset']['root']) / p
    evidence = dict(source_sha256=sources(), pipeline_yaml=args.pipeline_config.read_text(),
                    generator_yaml=(args.pipeline_config.parent / cfg.ss_generator_config_path).read_text(),
                    data_yaml=args.data_config.read_text(),
                    manifest_sha256=sha(resolve(data['dataset']['manifest'])),
                    splits_sha256=sha(resolve(data['dataset']['split_file'])),
                    records_sha256={k: digest(v.records) for k, v in ds.items()})
    contract = dict(arm=args.arm, seed=args.seed, batch_size=args.batch_size,
                    conditioning=config, precision='bf16', cross_attention_scope='full',
                    learning_rate=1e-4, cross_attention_learning_rate=1e-5,
                    evidence=evidence, protocol_version=1)
    # This encoder is identical across all surface arms; only its input differs.
    initial_ca = parameter_digest((n, p) for n, p in pipeline.backbone.named_parameters() if p.requires_grad)
    initial_adapter = parameter_digest((n, p) for n, p in model.named_parameters()
                                      if n.startswith('touch_encoder.') and p.requires_grad)
    constant_record = None
    if config['constant']:
        constant_ds = copy.copy(ds['train'])
        constant_record = min(ds['train'].records, key=lambda r: r['sample_id'])
        constant_ds.records = [constant_record]
        batch = collate_touch_batch([constant_ds[0]])
        _, _, _, points, mask = prepare(pipeline, batch, device, config)
        with torch.no_grad(), amp(device, 'bf16'):
            pp, mm, _, _ = model.touch_encoder.prepare_points(points, mask)
            model.constant_features = model.touch_encoder.encoder.encode(pp, mm)['x'].detach()
        contract['constant_sample_id'] = constant_record['sample_id']
        contract['constant_features_sha256'] = tensor_digest(model.constant_features)
    start_epoch = step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['contract'] != contract:
            raise ValueError('Resume contract/data/source mismatch')
        load_trainable_state_dict(model, saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start_epoch, step = saved['epoch'], saved['step']
        del saved
    selections = {split: evaluation_groups(v.records, args.seed) for split, v in ds.items()}
    report = dict(contract=contract, counts=check_splits(ds['train'].records, ds['val'].records),
                  initial_ca_sha256=initial_ca, initial_adapter_sha256=initial_adapter,
                  trainable_parameters=sum(p.numel() for p in params),
                  selection={k: [[r['sample_id'] for r in g] for g in groups] for k, groups in selections.items()},
                  constant_record=constant_record, epochs=args.epochs,
                  preflight_sha256=sha(args.preflight_report),
                  runtime={'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()})
    write_json(args.output_dir / 'config.json', report)
    print(json.dumps({'arm': args.arm, **report['counts'], 'epochs': args.epochs,
                      'batch_size': args.batch_size, 'parameters': report['trainable_parameters']}), flush=True)
    for epoch in range(start_epoch, args.epochs):
        batches, drops = epoch_schedule(len(ds['train']), args.batch_size, args.seed, epoch)
        loader = DataLoader(ds['train'], batch_sampler=batches, num_workers=args.workers,
                            collate_fn=collate_touch_batch, pin_memory=True,
                            generator=torch.Generator().manual_seed(1300000 + args.seed + epoch))
        model.generator.reverse_fn.training = True
        losses = {False: [], True: []}; start = time.perf_counter()
        for bi, batch in enumerate(loader):
            prepared = prepare(pipeline, batch, device, config)
            dropped = bool(drops[bi] and config['visual_dropout'])
            seed_all(1400000 + args.seed + step)
            optimizer.zero_grad(set_to_none=True)
            with amp(device, 'bf16'):
                loss = model(*prepared, drop_visual=dropped)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step(); step += 1
            row = dict(epoch=epoch + 1, step=step, visual_dropped=dropped, loss=float(loss),
                       gradient_norm=float(norm), sample_ids=batch['sample_id'])
            with (args.output_dir / 'training.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            losses[dropped].append(float(loss))
            if bi == 0 or step % 20 == 0:
                elapsed = time.perf_counter() - start
                remaining = (args.epochs - epoch) * len(batches) - bi - 1
                print(args.arm, 'epoch', epoch + 1, 'step', step, 'loss', float(loss),
                      'train ETA hours', round(remaining * elapsed / (bi + 1) / 3600, 2), flush=True)
        checkpoint = dict(model=trainable_state_dict(model), optimizer=optimizer.state_dict(),
                          epoch=epoch + 1, step=step, contract=contract,
                          conditioning_config=model.conditioning_config,
                          touch_config=model.touch_encoder.get_config() if model.touch_encoder else None)
        temp = args.output_dir / 'last.tmp.pt'; torch.save(checkpoint, temp); temp.replace(args.output_dir / 'last.pt')
        if epoch + 1 in set(args.assess_epochs + [args.epochs]):
            # Preserve the exact epoch endpoint; never select solely by validation loss.
            torch.save(checkpoint, args.output_dir / f'epoch_{epoch + 1}.pt')
            del checkpoint
            from assess import assess
            assess(args.output_dir / f'assessment_{epoch + 1}', pipeline, model, ds, selections,
                   config, device, args.seed, epoch + 1, step, args.pipeline_config)
        else:
            del checkpoint
    write_json(args.output_dir / 'complete.json', dict(complete=True, epoch=args.epochs, step=step, contract=contract))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--oracle-fit-dir', type=Path)
    p.add_argument('--preflight-report', type=Path)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--pipeline-config', type=Path, default=Path('checkpoints/hf/pipeline.yaml'))
    p.add_argument('--data-config', type=Path, default=Path('configs/data_full_surface.yaml'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--assess-epochs', type=int, nargs='*', default=[1, 10, 20])
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    if a.seed != 29 and a.preflight:
        p.error('Historical preflight requires seed 29')
    if min(a.epochs, a.batch_size) < 1 or a.workers < 0:
        p.error('Invalid training dimensions')
    if not a.preflight and (a.arm is None or a.preflight_report is None):
        p.error('Training requires --arm and --preflight-report')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        p.error('Use launch.py, not torchrun: each arm has one GPU and a fixed batch size')
    if a.output_dir.exists() and not a.resume:
        raise FileExistsError(a.output_dir)
    if a.resume and (a.preflight or a.resume.resolve() != (a.output_dir / 'last.pt').resolve()):
        p.error('Resume only the same run from its last.pt epoch boundary')
    a.output_dir.mkdir(parents=True, exist_ok=bool(a.resume))
    if a.preflight:
        preflight(a)
    else:
        training(a)


if __name__ == '__main__':
    main()
