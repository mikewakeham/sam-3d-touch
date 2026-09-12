"""Bounded constructive test: separate surface attention around an image anchor."""
import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from dataloader import TouchDataset, collate_touch_batch, load_data_config
from experiments.noisy_target.evaluate import tensor_digest
from object_transfer_protocol import digest_json, make_schedule, validate_plan
from analyze_object_transfer import validate
from separate_surface_attention import SeparateSurfaceAttention, verify_module

MODEL_KEYS = ('separate_oracle', 'separate_constant', 'separate_camera', 'joint_oracle')
from tiny_fit_gpu import parameter_digest
from train import (TouchTrainingModel, amp, build_optimizer, build_stage1_pipeline,
                   gradient_norm, prepare_batch, trainable_state_dict)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_gradient_norms(model):
    blocks = model.generator.reverse_fn.backbone.blocks
    attention = [block.cross_attn['shape'] for block in blocks]
    adapted = [module.surface if isinstance(module, SeparateSurfaceAttention) else module
               for module in attention]
    groups = {
        'shape_cross_attention_kv': (p for module in adapted for p in module.to_kv.parameters()),
        'shape_cross_attention': (p for block in blocks
                                  for module in (block.cross_attn['shape'], block.norm2['shape'])
                                  for p in module.parameters()),
        'shape_visual_attention': (p for module in attention if isinstance(module, SeparateSurfaceAttention)
                                   for p in module.visual.parameters()),
    }
    encoder = model.touch_encoder
    if encoder is not None:
        groups.update(touch_output_projection=encoder.output_projection.parameters(),
                      touch_position_projection=encoder.position_projection.parameters(),
                      touch_embedding=(encoder.touch_embedding,),
                      vecsetx_encoder=encoder.encoder.parameters())
    return {f'gradients/{name}': gradient_norm(parameters) for name, parameters in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key', choices=MODEL_KEYS, required=True)
    parser.add_argument('--image-fit-dir', type=Path, default=Path('outputs/conditioning_investigation/object_transfer/manual/image'))
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.cuda.set_device(args.device_index)
    device = torch.device('cuda', args.device_index)
    plan_path = HERE / 'object_transfer_plan.json'
    plan = json.loads(plan_path.read_text()); validate_plan(plan)
    seed, precision = plan['seed'], 'bf16'
    reference_path = HERE / 'tiny_fit_returned_46083371/oracle/results.json'
    reference = json.loads(reference_path.read_text())
    image_report_path = HERE/'object_transfer_returned_manual/image/results.json'
    image_report = json.loads(image_report_path.read_text()); validate(image_report)
    assert sha(args.image_fit_dir/'results.json') == sha(image_report_path)
    assert image_report['plan'] == plan and image_report['plan_sha256'] == sha(plan_path)
    assert image_report['reference_sha256'] == sha(reference_path)
    for name, digest in image_report['source_sha256'].items():
        assert sha(REPO/name) == digest, name
    frame_report_path = HERE/'object_transfer_returned_manual'/('camera' if args.model_key == 'separate_camera' else 'oracle')/'results.json'
    frame_report = json.loads(frame_report_path.read_text()); validate(frame_report)
    for source, digest in reference['source_sha256'].items():
        assert sha(REPO / source) == digest, source
    pipeline_path, data_path = (Path(reference['settings'][k]) for k in ('pipeline_config', 'data_config'))
    assert pipeline_path.read_text() == reference['pipeline_yaml']
    assert data_path.read_text() == reference['data_yaml']
    cfg = OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent / cfg.ss_generator_config_path).read_text() == reference['generator_yaml']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pipeline = build_stage1_pipeline(pipeline_path, device)
    oracle = args.model_key != 'separate_camera'
    separate = args.model_key.startswith('separate_')
    constant = args.model_key == 'separate_constant'
    encoder = None
    if args.model_key != 'image':
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
        encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels,
                               trainable=False, use_position=False).to(device).eval()
    model = TouchTrainingModel(pipeline.ss_generator, encoder, False, oracle)
    optimizer, optimized = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope='full'))
    # Restore the existing image-only adaptation before introducing any surface
    # branch. Encoder initialization remains matched to the original point arms.
    archive_path = args.image_fit_dir/'checkpoint_1024.pt'
    archive = torch.load(archive_path, map_location='cpu', weights_only=True)
    assert archive['step'] == 1024 and archive['settings'] == image_report['settings']
    assert archive['conditioning_config'] == {'no_pointmap': False, 'oracle_point_frame': False}
    for field in ('plan_sha256', 'schedule_sha256', 'source_sha256'):
        assert archive[field] == image_report[field]
    parameters = dict(model.named_parameters())
    assert set(archive['model']) == {n for n, p in parameters.items() if n.startswith('generator.') and p.requires_grad}
    with torch.no_grad():
        for name, value in archive['model'].items():
            assert parameters[name].shape == value.shape
            parameters[name].copy_(value)
    del archive, optimizer
    anchor_hash = parameter_digest((f'generator.{n}', p) for n, p in model.generator.named_parameters())
    assert anchor_hash == image_report['final_model_sha256']
    assert parameter_digest(encoder.named_parameters()) == frame_report['initial_encoder_sha256']
    model.generator.requires_grad_(False)
    module_checks = verify_module(pipeline.backbone.blocks[0].cross_attn['shape'], device)
    if separate:
        attention_parameters = []
        for block in pipeline.backbone.blocks:
            wrapped = SeparateSurfaceAttention(block.cross_attn['shape'])
            block.cross_attn['shape'] = wrapped
            attention_parameters.extend(wrapped.surface.parameters())
        groups = [dict(name='touch_encoder', params=list(encoder.get_trainable_parameters()), lr=1e-4),
                  dict(name='surface_attention', params=attention_parameters, lr=1e-5)]
        optimizer = torch.optim.AdamW(groups, weight_decay=0)
        optimized = [p for group in optimizer.param_groups for p in group['params']]
    else:
        optimizer, optimized = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
            learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope='full'))
    assert {id(p) for p in optimized} == {id(p) for p in model.parameters() if p.requires_grad}
    frozen_hash = parameter_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    initial_generator = parameter_digest(model.generator.named_parameters())
    initial_model = parameter_digest(model.named_parameters())
    initial_ca = parameter_digest((n, p) for n, p in pipeline.backbone.named_parameters() if p.requires_grad)
    gen = pipeline.ss_generator
    gen.reverse_fn.training = True
    assert gen.reverse_fn.p_unconditional == 0 and gen.self_consistency_prob == 0 and gen.fm_eps_max == 0
    assert gen.loss_weights['shape'] == 1 and all(v == 0 for k, v in gen.loss_weights.items() if k != 'shape')
    data = load_data_config(data_path)
    assert data.get('touch', {}).get('source') == 'full_surface'
    datasets, lookup, split_of = [], {}, {}
    for split in ('train', 'val'):
        config = copy.deepcopy(data); config['dataset']['split'] = split
        ds = TouchDataset(config, include_touch=encoder is not None, oracle_point_frame=oracle)
        datasets.append(ds)
        for r in ds.records:
            assert r['sample_id'] not in lookup
            lookup[r['sample_id']] = r
            assert split_of.setdefault(r['object_id'], split) == split
    ds = datasets[0]
    assert all(split_of[o] == 'train' for o in plan['training_object_ids'])
    assert all(split_of[o] == 'val' for o in plan['held_object_ids'])
    prepared, metadata, target_digests = [], [], {}
    for batch_plan in plan['batches']:
        records = [lookup[s] for s in batch_plan['sample_ids']]
        assert [r['object_id'] for r in records] == batch_plan['object_ids']
        ds.records = records
        batch = collate_touch_batch([ds[i] for i in range(4)])
        targets, ca, kw, points, mask = prepare_batch(pipeline, batch, device, precision,
                                                    encoder is not None, False, oracle)
        assert len(ca) == 1 and torch.is_tensor(ca[0]) and not kw
        features = None
        if encoder is not None:
            assert mask.all() and points.shape[1] == encoder.num_points == 8192
            with torch.no_grad(), amp(device, precision):
                pp, mm, _, _ = encoder.prepare_points(points, mask)
                features = encoder.encoder.encode(pp, mm)['x'].detach()
        target_hashes = [tensor_digest(t) for t in targets['shape']]
        for oid, digest in zip(batch_plan['object_ids'], target_hashes):
            assert target_digests.setdefault(oid, digest) == digest, 'View-dependent target'
        entry = {**batch_plan, 'image_sha256': tensor_digest(batch['image']),
                 'pointmap_sha256': tensor_digest(batch['pointmap']), 'visual_sha256': tensor_digest(ca[0]),
                 'target_sha256': tensor_digest(targets['shape']), 'object_target_sha256': target_hashes}
        if features is not None:
            entry['features_sha256'] = tensor_digest(features)
        prepared.append((targets, ca[0], features)); metadata.append(entry)
    assert metadata == frame_report['input_batches'], 'Original feature/input mismatch'
    constant_features = prepared[0][2][0:1].clone().detach()
    assert tuple(constant_features.shape) == (1, 1024, 32)
    schedule = make_schedule(seed)
    report = dict(settings={'model_key': args.model_key, 'seed': seed, 'precision': precision, 'steps': 1024,
                            'visual_dropout_fraction': 0,
                            'monitor_bank_base': 600000, 'monitor_draws': 4,
                            'final_bank_base': 700000, 'final_draws': 8},
                  plan=plan, plan_sha256=sha(plan_path), schedule=schedule, schedule_sha256=digest_json(schedule),
                  reference_sha256=sha(reference_path), source_sha256={**image_report['source_sha256'], **{
                      str(p.relative_to(REPO)): sha(p) for p in (Path(__file__), HERE/'separate_surface_attention.py',
                                                               HERE/'analyze_object_transfer.py', HERE/'analyze_separate_surface.py')}},
                  initial_generator_sha256=initial_generator, initial_model_sha256=initial_model,
                  initial_trainable_ca_sha256=initial_ca,
                  initial_encoder_sha256=parameter_digest(encoder.named_parameters()) if encoder is not None else None,
                  trainable_parameters={g['name']: sum(p.numel() for p in g['params']) for g in optimizer.param_groups},
                  input_batches=metadata, assessments=[], training=[], final_rows=[], dropout_context_verified=False,
                  anchor_report_sha256=sha(image_report_path), frame_report_sha256=sha(frame_report_path),
                  anchor_checkpoint_sha256=sha(archive_path), anchor_model_sha256=anchor_hash,
                  initial_frozen_parameters_sha256=frozen_hash, module_checks=module_checks,
                  initial_removed_replay=[], final_removed_rows=[], interface_context_checks={},
                  constant_bank=dict(batch=0, row=0, sample_id=metadata[0]['sample_ids'][0],
                      object_id=metadata[0]['object_ids'][0], sha256=tensor_digest(constant_features)),
                  runtime={'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device),
                           'device_index': args.device_index}, complete=False,
                  scope='16 fitted identities: four fitted and two held views each. 16 held identities, '
                        'two views each. Diagnostic development pool, seed 37. Native Stage-1 losses; '
                        '1024 additional updates from the completed image-1024 anchor. Separate arms freeze '
                        'the entire image generator and add independent zero-output-initialized surface CA. '
                        'Joint oracle controls warm start with original concatenated/shared CA training. '
                        'One seed/development pool; no production fix is established by fitting alone.')
    args.output_dir.mkdir(parents=True)

    def save():
        (args.output_dir/'results.partial.json').write_text(json.dumps(report, indent=2)+'\n')

    def forward_loss(index, drop=False, swap=False, per_object=False, removed=False):
        targets, visual, features = prepared[index]
        if constant:
            features = constant_features.expand(visual.shape[0], -1, -1)
        tokens = None
        if encoder is not None and not removed:
            tokens = encoder.output_projection(features) + encoder.touch_embedding
            if swap:
                tokens = tokens.roll(1, 0)
        assert not swap or tokens is not None
        if drop:
            assert tokens is not None and args.model_key == 'oracle_dropout'
            visual = torch.zeros_like(visual)
        cond = {} if tokens is None else {'touch_tokens': tokens}
        handles = []
        if not removed and not report['interface_context_checks']:
            if constant:
                torch.testing.assert_close(tokens, tokens[:1].expand_as(tokens), rtol=0, atol=0)
            targets_to_check = ([('visual', pipeline.backbone.blocks[0].cross_attn['shape'].visual.to_kv, visual),
                                 ('surface', pipeline.backbone.blocks[0].cross_attn['shape'].surface.to_kv, tokens.to(visual))]
                                if separate else [('joint', pipeline.backbone.blocks[0].cross_attn['shape'].to_kv,
                                                   torch.cat((visual, tokens.to(visual)), dim=1))])
            for label, module, expected in targets_to_check:
                def check_context(module, inputs, label=label, expected=expected):
                    torch.testing.assert_close(inputs[0], expected.to(inputs[0]), rtol=0, atol=0)
                    report['interface_context_checks'][label] = inputs[0].shape[1]
                handles.append(module.register_forward_pre_hook(check_context))
        values = []
        original = gen.loss_fn
        functions = dict(original) if isinstance(original, dict) else {k: original for k in gen.loss_weights}
        shape_fn = functions['shape']
        def capture(pred, target):
            native = shape_fn(pred, target)
            losses = F.mse_loss(pred.float(), target.float(), reduction='none').flatten(1).mean(1)
            torch.testing.assert_close(losses.mean(), native.float(), rtol=1e-5, atol=1e-7)
            values.append(losses.detach())
            return native
        if per_object:
            functions['shape'] = capture
        try:
            # Training keeps the original loss_fn untouched. Assessment captures
            # per-object values while returning the original native scalar.
            with patch.object(gen, 'loss_fn', functions if per_object else original):
                loss, _ = gen.loss(targets, visual, **cond)
        finally:
            for handle in handles:
                handle.remove()
        assert torch.isfinite(loss)
        if per_object:
            assert len(values) == 1 and values[0].shape == (4,)
            torch.testing.assert_close(values[0].mean(), loss.float(), rtol=1e-5, atol=1e-7)
            return float(loss), values[0].cpu().tolist()
        return loss

    def assess(bank, draws, swaps, removed=False):
        rows = []
        py = random.getstate()
        with torch.random.fork_rng(devices=[args.device_index]), torch.no_grad():
            for index, meta in enumerate(metadata):
                for swap in ([False, True] if swaps and encoder is not None else [False]):
                    matrix, scalar = [], []
                    for draw in range(draws):
                        draw_seed = bank+seed+100*(index % 4)+draw
                        torch.manual_seed(draw_seed); random.seed(draw_seed)
                        with amp(device, precision):
                            native, values = forward_loss(index, swap=swap, per_object=True, removed=removed)
                        scalar.append(native); matrix.append(values)
                    rows.append({'batch': index, 'split': meta['split'], 'swapped_surface': swap,
                                 'scalar_losses': scalar, 'per_object_losses': matrix})
        random.setstate(py)
        return rows

    report['initial_removed_replay'] = assess(600000, 4, False, removed=True)
    assert report['initial_removed_replay'] == image_report['assessments'][-1]['rows']
    report['initial_removed_replay_passed'] = True
    initial_rows = assess(600000, 4, False)
    if separate:
        assert initial_rows == image_report['assessments'][-1]['rows'], 'Zero branch changed image function'
    report['initial_zero_residual_exact'] = initial_rows == image_report['assessments'][-1]['rows']
    report['assessments'].append({'step': 0, 'rows': initial_rows}); save()
    for step, item in enumerate(schedule, 1):
        index = item['batch']
        assert metadata[index]['split'] == 'fit'
        drop = False
        torch.manual_seed(seed+step); random.seed(seed+step)
        optimizer.zero_grad(set_to_none=True)
        with amp(device, precision):
            loss = forward_loss(index, drop=drop)
        loss.backward()
        assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
        gradients = experiment_gradient_norms(model) if step == 1 or step % 32 == 0 else {}
        norm = torch.nn.utils.clip_grad_norm_(optimized, 1., error_if_nonfinite=True)
        optimizer.step()
        report['training'].append(dict(step=step, batch=index, visual_dropped=drop,
                                       loss=float(loss), preclip_gradient_norm=float(norm), **gradients))
        if step % 32 == 0:
            print(args.model_key, step, float(loss), 'drop', drop, flush=True)
        if step in (256, 512, 1024):
            report['assessments'].append({'step': step, 'rows': assess(600000, 4, False)})
            torch.save(dict(model=trainable_state_dict(model), optimizer=optimizer.state_dict(), step=step,
                            settings=report['settings'], conditioning_config=model.conditioning_config,
                            plan_sha256=report['plan_sha256'], schedule_sha256=report['schedule_sha256'],
                            source_sha256=report['source_sha256']), args.output_dir/f'checkpoint_{step}.pt')
            save()
    report['final_model_sha256'] = parameter_digest(model.named_parameters())
    assert report['final_model_sha256'] != initial_model
    report['final_rows'] = assess(700000, 8, True)
    assert parameter_digest(model.named_parameters()) == report['final_model_sha256']
    assert report['interface_context_checks'] == ({'visual': 7528, 'surface': 1024} if separate else {'joint': 8552})
    report['final_removed_rows'] = assess(700000, 8, False, removed=True)
    if separate:
        assert report['final_removed_rows'] == image_report['final_rows'], 'Frozen image fallback changed'
    report['final_removed_anchor_exact'] = report['final_removed_rows'] == image_report['final_rows']
    final_frozen = parameter_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    assert final_frozen == frozen_hash
    report['final_frozen_parameters_sha256'] = final_frozen
    assert tensor_digest(constant_features) == report['constant_bank']['sha256']
    if constant:
        rows = {(r['batch'], r['swapped_surface']): r for r in report['final_rows']}
        for index in range(32):
            for name in ('scalar_losses', 'per_object_losses'):
                assert rows[index, False][name] == rows[index, True][name]
    report['complete'] = True
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
