"""Matched 1000-update camera/oracle fit with balanced 50% visual dropout.

Retain frozen VecSetX and original projector/full shape CA. Replay original
initial/final losses and evaluate the original fitted checkpoint on fresh draws
before restoring initialization and training. No decoding or Stage 2.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from omegaconf import OmegaConf
from dataloader import build_dataloader, collate_touch_batch, load_data_config
from experiments.noisy_target.evaluate import tensor_digest
from probe_oracle_visual_streams_gpu import locate_fit
from probe_tiny_fit_views import choose_views
from tiny_fit_gpu import parameter_digest
from train import (TouchTrainingModel, amp, build_optimizer, build_stage1_pipeline,
                   component_gradient_norms, prepare_batch, trainable_state_dict)
from visual_dropout_protocol import dropout_schedule, schedule_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['camera', 'oracle'], required=True)
    parser.add_argument('--baseline-fit-dir', type=Path)
    parser.add_argument('--reference-root', type=Path, default=HERE)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    single_path = args.reference_root / f'tiny_fit_returned_46083371/{args.arm}/results.json'
    multi_path = args.reference_root / f'multiple_view_returned_46083371/{args.arm}.json'
    single, multi = (json.loads(p.read_text()) for p in (single_path, multi_path))
    assert multi['source_reference_sha256'] == sha(single_path)
    assert multi['driver_sha256'] == sha(HERE / 'fit_multiple_views_gpu.py')
    assert multi['view_selector_sha256'] == sha(HERE / 'probe_tiny_fit_views.py')
    assert multi['settings'] == dict(arm=args.arm, steps=1000, seed=29, precision='bf16',
                                   objects=4, fit_views_per_object=4, reserved_views_per_object=3)
    try:
        fit_dir = locate_fit(args.baseline_fit_dir, multi)
    except ValueError:
        raise ValueError(f'Cannot locate one matching original {args.arm} multi-view fit. '
                         'Pass --baseline-fit-dir /path/to/that/arm containing results.json '
                         'and fitted_parameters.pt. Do not rerun the baseline training.') from None
    settings = single['settings']
    seed, precision = settings['seed'], settings['precision']
    assert seed == 29 and precision == 'bf16'
    for source, expected in single['source_sha256'].items():
        assert sha(REPO / source) == expected, source
    pipeline_path, data_path = Path(settings['pipeline_config']), Path(settings['data_config'])
    assert pipeline_path.read_text() == single['pipeline_yaml']
    assert data_path.read_text() == single['data_yaml']
    cfg = OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent / cfg.ss_generator_config_path).read_text() == single['generator_yaml']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda')
    pipeline = build_stage1_pipeline(pipeline_path, device)
    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
    encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels,
                           trainable=False, use_position=False).to(device).eval()
    model = TouchTrainingModel(pipeline.ss_generator, encoder, False, args.arm == 'oracle')
    optimizer, optimized = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope='full'))
    initial_hash = parameter_digest(model.named_parameters())
    assert initial_hash == multi['initial_all_parameters_sha256']
    assert parameter_digest((n, p) for n, p in pipeline.backbone.named_parameters() if p.requires_grad) == multi['initial_trainable_ca_sha256']
    initial_trainable = {n: p.clone() for n, p in trainable_state_dict(model).items()}
    parameters = dict(model.named_parameters())
    gen = pipeline.ss_generator
    gen.reverse_fn.training = True
    assert gen.reverse_fn.p_unconditional == 0 and gen.self_consistency_prob == 0 and gen.fm_eps_max == 0
    assert not pipeline.ss_condition_embedder.training
    data = load_data_config(data_path); data['dataset']['split'] = 'train'
    assert data.get('touch', {}).get('source') == 'full_surface'
    ds = build_dataloader(data, 4, 0, shuffle=False, include_touch=True,
                          oracle_point_frame=args.arm == 'oracle').dataset
    groups = choose_views(ds.records, single['sample_ids'], count=6)
    prepared, metadata = [], []
    for group, records in enumerate(groups):
        ds.records = records
        batch = collate_touch_batch([ds[i] for i in range(4)])
        targets, ca, kw, points, mask = prepare_batch(
            pipeline, batch, device, precision, True, False, args.arm == 'oracle')
        assert len(ca) == 1 and torch.is_tensor(ca[0]) and not kw
        assert mask.all() and points.shape[1] == encoder.num_points == 8192
        with torch.no_grad(), amp(device, precision):
            pp, mm, _, _ = encoder.prepare_points(points, mask)
            features = encoder.encoder.encode(pp, mm)['x'].detach()
        entry = dict(group=group, split='fit' if group < 4 else 'reserved_view',
                     sample_ids=[r['sample_id'] for r in records],
                     image_sha256=tensor_digest(batch['image']), pointmap_sha256=tensor_digest(batch['pointmap']),
                     target_sha256=tensor_digest(targets['shape']), features_sha256=tensor_digest(features))
        assert entry == multi['input_batches'][group], f'Input mismatch at group {group}'
        prepared.append((targets, ca[0], features)); metadata.append(entry)
    schedule = dropout_schedule(seed)
    report = dict(settings={**multi['settings'], 'visual_dropout_fraction': .5,
                            'dropout_unit': 'whole_batch', 'fresh_bank_base': 400000,
                            'baseline_fit_dir': str(fit_dir)},
                  reference_sha256={'single': sha(single_path), 'multiple': sha(multi_path)},
                  source_sha256={**single['source_sha256'], **{str(p.relative_to(REPO)): sha(p) for p in
                      (Path(__file__), HERE / 'visual_dropout_protocol.py', HERE / 'probe_oracle_visual_streams_gpu.py',
                       HERE / 'probe_tiny_fit_views.py')}},
                  initial_all_parameters_sha256=initial_hash,
                  initial_trainable_ca_sha256=multi['initial_trainable_ca_sha256'],
                  baseline_final_parameters_sha256=multi['final_all_parameters_sha256'],
                  dropout_schedule=schedule, dropout_schedule_sha256=schedule_digest(schedule),
                  input_batches=metadata, baseline_preflight={}, baseline_fresh={},
                  interface_checks={}, assessments=[], training=[], final_fresh={}, complete=False,
                  runtime={'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()},
                  scope='Four objects, four fitted and three reserved views. Balanced visual dropout; '
                        'all-input evaluation is primary. Original trained baseline freshly evaluated '
                        'without updates. Frozen VecSetX, unchanged shape CA scope. No decoding.')
    args.output_dir.mkdir(parents=True)

    def save():
        (args.output_dir / 'results.partial.json').write_text(json.dumps(report, indent=2) + '\n')

    def loss_at(group, drop=False, swap=False):
        targets, visual, features = prepared[group]
        tokens = encoder.output_projection(features) + encoder.touch_embedding
        if swap:
            tokens = tokens.roll(1, 0)
        visual = torch.zeros_like(visual) if drop else visual
        key = 'visual_zero_surface_present' if drop else 'visual_present_surface_present'
        handle = None
        if key not in report['interface_checks']:
            # Verify the actual context at shape cross-attention, not just the
            # arguments before classifier-free/condition handling.
            def check_context(module, inputs):
                context = inputs[0]
                expected = torch.cat((visual, tokens.to(visual)), dim=1).to(context)
                torch.testing.assert_close(context, expected, rtol=0, atol=0)
                assert torch.count_nonzero(context[:, visual.shape[1]:]).item() > 0
                if drop:
                    assert torch.count_nonzero(context[:, :visual.shape[1]]).item() == 0
                report['interface_checks'][key] = {'passed': True, 'context_shape': list(context.shape),
                    'visual_tokens': visual.shape[1], 'surface_tokens': tokens.shape[1]}
            handle = pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check_context)
        try:
            loss, _ = gen.loss(targets, visual, touch_tokens=tokens)
        finally:
            if handle is not None:
                handle.remove()
        assert key in report['interface_checks'], 'Shape context hook did not execute'
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite native loss')
        return loss

    def assess(bank=100000, drop=False):
        py = random.getstate()
        rows = []
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            for group in range(7):
                values, wrong = [], []
                for draw in range(8):
                    for swap in (False, True):
                        torch.manual_seed(bank + seed + draw); random.seed(bank + seed + draw)
                        with amp(device, precision):
                            value = float(loss_at(group, drop, swap))
                        (wrong if swap else values).append(value)
                rows.append(dict(group=group, split=metadata[group]['split'],
                                 fresh_noise_native_losses=values, swapped_surface_native_losses=wrong))
        random.setstate(py)
        return rows

    def require_replay(rows, expected):
        for actual, old in zip(rows, expected):
            for key in ('fresh_noise_native_losses', 'swapped_surface_native_losses'):
                np.testing.assert_array_equal(actual[key], old[key])

    initial_rows = assess()
    require_replay(initial_rows, multi['assessments'][0]['rows'])
    report['baseline_preflight']['initial'] = initial_rows; save()
    archive = torch.load(fit_dir / 'fitted_parameters.pt', map_location='cpu', weights_only=True)
    assert archive['step'] == 1000 and archive['settings'] == multi['settings']
    assert archive['conditioning_config'] == model.conditioning_config
    assert set(archive['model']) == set(initial_trainable)
    with torch.no_grad():
        for name, value in archive['model'].items():
            parameters[name].copy_(value)
    del archive
    assert parameter_digest(model.named_parameters()) == multi['final_all_parameters_sha256']
    final_rows = assess()
    require_replay(final_rows, multi['assessments'][-1]['rows'])
    report['baseline_preflight']['final'] = final_rows; save()
    for name, drop in [('visual_present', False), ('visual_zero', True)]:
        report['baseline_fresh'][name] = assess(400000, drop); save()
    # The trained checkpoint is only a read-only comparator, not a warm start.
    with torch.no_grad():
        for name, value in initial_trainable.items():
            parameters[name].copy_(value)
    del initial_trainable
    assert parameter_digest(model.named_parameters()) == initial_hash and not optimizer.state
    report['baseline_preflight_passed'] = True
    report['initial_parameters_restored'] = True
    report['assessments'].append({'step': 0, 'rows': initial_rows}); save()
    for step in range(1, 1001):
        group = (step - 1) % 4
        drop = schedule[step - 1]
        torch.manual_seed(seed + step); random.seed(seed + step)
        optimizer.zero_grad(set_to_none=True)
        with amp(device, precision):
            loss = loss_at(group, drop)
        loss.backward()
        assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
        gradients = component_gradient_norms(model) if step == 1 or step % 20 == 0 else {}
        norm = torch.nn.utils.clip_grad_norm_(optimized, 1., error_if_nonfinite=True)
        optimizer.step()
        report['training'].append(dict(step=step, group=group, visual_dropped=drop,
                                       loss=float(loss), preclip_gradient_norm=float(norm), **gradients))
        if step % 20 == 0:
            print(args.arm, step, 'visual_dropped', drop, 'loss', float(loss), flush=True)
        if step in (100, 300, 1000):
            rows = assess()
            report['assessments'].append({'step': step, 'rows': rows}); save()
            print(args.arm, step, 'all-input reserved loss', np.mean(
                [r['fresh_noise_native_losses'] for r in rows[4:]]), flush=True)
        if step in (300, 1000):
            torch.save(dict(model=trainable_state_dict(model), optimizer=optimizer.state_dict(),
                            step=step, settings=report['settings'], conditioning_config=model.conditioning_config,
                            dropout_schedule_sha256=report['dropout_schedule_sha256'],
                            source_sha256=report['source_sha256']), args.output_dir / f'checkpoint_{step}.pt')
    report['final_all_parameters_sha256'] = parameter_digest(model.named_parameters())
    assert report['final_all_parameters_sha256'] != initial_hash
    for name, drop in [('visual_present', False), ('visual_zero', True)]:
        report['final_fresh'][name] = assess(400000, drop); save()
    assert parameter_digest(model.named_parameters()) == report['final_all_parameters_sha256']
    report['complete'] = True
    (args.output_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
