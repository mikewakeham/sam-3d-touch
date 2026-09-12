"""Matched geometry-free control: one fixed training surface for every sample.

Retains the existing VecSetX/projector/shared-attention path. No new architecture.
The original real-oracle initialization, inputs and initial losses must replay.
"""
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
from tiny_fit_gpu import parameter_digest
from train import (TouchTrainingModel, amp, build_optimizer, build_stage1_pipeline,
                   component_gradient_norms, prepare_batch, trainable_state_dict)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.model_key = 'oracle_constant'
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.cuda.set_device(args.device_index)
    device = torch.device('cuda', args.device_index)
    plan_path = HERE / 'object_transfer_plan.json'
    plan = json.loads(plan_path.read_text()); validate_plan(plan)
    seed, precision = plan['seed'], 'bf16'
    reference_path = HERE / 'tiny_fit_returned_46083371/oracle/results.json'
    reference = json.loads(reference_path.read_text())
    baseline_path = HERE / 'object_transfer_returned_manual/oracle/results.json'
    baseline = json.loads(baseline_path.read_text()); validate(baseline)
    assert baseline['plan'] == plan and baseline['plan_sha256'] == sha(plan_path)
    assert baseline['reference_sha256'] == sha(reference_path)
    for name, digest in baseline['source_sha256'].items():
        assert sha(REPO / name) == digest, name
    for source, digest in reference['source_sha256'].items():
        assert sha(REPO / source) == digest, source
    pipeline_path, data_path = (Path(reference['settings'][k]) for k in ('pipeline_config', 'data_config'))
    assert pipeline_path.read_text() == reference['pipeline_yaml']
    assert data_path.read_text() == reference['data_yaml']
    cfg = OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent / cfg.ss_generator_config_path).read_text() == reference['generator_yaml']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pipeline = build_stage1_pipeline(pipeline_path, device)
    oracle = args.model_key.startswith('oracle')
    encoder = None
    if args.model_key != 'image':
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
        encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels,
                               trainable=False, use_position=False).to(device).eval()
    model = TouchTrainingModel(pipeline.ss_generator, encoder, False, oracle)
    optimizer, optimized = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope='full'))
    initial_generator = parameter_digest(model.generator.named_parameters())
    initial_model = parameter_digest(model.named_parameters())
    initial_ca = parameter_digest((n, p) for n, p in pipeline.backbone.named_parameters() if p.requires_grad)
    assert initial_ca == reference['initial_trainable_ca_sha256']
    assert initial_generator == baseline['initial_generator_sha256']
    assert initial_model == baseline['initial_model_sha256']
    assert initial_ca == baseline['initial_trainable_ca_sha256']
    assert parameter_digest(encoder.named_parameters()) == baseline['initial_encoder_sha256']
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
    assert metadata == baseline['input_batches'], 'Actual oracle inputs differ'
    # Fix one existing training record before looking at any result. Preserve
    # the complete native token distribution instead of inventing an average code.
    constant_features = prepared[0][2][0:1].clone().detach()
    assert metadata[0]['split'] == 'fit' and not constant_features.requires_grad
    assert tuple(constant_features.shape) == (1, 1024, 32)
    schedule = make_schedule(seed)
    report = dict(settings={'model_key': args.model_key, 'seed': seed, 'precision': precision, 'steps': 1024,
                            'visual_dropout_fraction': .5 if args.model_key == 'oracle_dropout' else 0,
                            'monitor_bank_base': 600000, 'monitor_draws': 4,
                            'final_bank_base': 700000, 'final_draws': 8},
                  plan=plan, plan_sha256=sha(plan_path), schedule=schedule, schedule_sha256=digest_json(schedule),
                  reference_sha256=sha(reference_path), source_sha256={**baseline['source_sha256'], **{
                      str(p.relative_to(REPO)): sha(p) for p in (Path(__file__), HERE/'analyze_object_transfer.py',
                                                               HERE/'analyze_constant_surface.py')}},
                  initial_generator_sha256=initial_generator, initial_model_sha256=initial_model,
                  initial_trainable_ca_sha256=initial_ca,
                  initial_encoder_sha256=parameter_digest(encoder.named_parameters()) if encoder is not None else None,
                  trainable_parameters={g['name']: sum(p.numel() for p in g['params']) for g in optimizer.param_groups},
                  input_batches=metadata, assessments=[], training=[], final_rows=[], dropout_context_verified=False,
                  baseline_report_sha256=sha(baseline_path), initial_real_replay=[], initial_real_replay_passed=False,
                  constant_context_verified=False, constant_token_counts=None,
                  constant_bank=dict(batch=0, row=0, sample_id=metadata[0]['sample_ids'][0],
                      object_id=metadata[0]['object_ids'][0], shape=list(constant_features.shape),
                      dtype=str(constant_features.dtype), sha256=tensor_digest(constant_features)),
                  runtime={'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device),
                           'device_index': args.device_index}, complete=False,
                  scope='16 fitted identities: four fitted and two held views each. 16 held identities, '
                        'two views each. Diagnostic development pool, seed 37. Native Stage-1 losses; '
                        '1024 updates, 256 presentations per fitted identity. Every sample receives the SAME '
                        'fixed training-surface feature bank; no sample-specific surface information. '
                        'This is a matched training control, not a proposed touch representation.')
    args.output_dir.mkdir(parents=True)

    def save():
        (args.output_dir/'results.partial.json').write_text(json.dumps(report, indent=2)+'\n')

    def forward_loss(index, drop=False, swap=False, per_object=False, real_surface=False):
        targets, visual, features = prepared[index]
        if not real_surface:
            features = constant_features.expand(visual.shape[0], -1, -1)
        tokens = None
        if encoder is not None:
            tokens = encoder.output_projection(features) + encoder.touch_embedding
            if swap:
                tokens = tokens.roll(1, 0)
        assert not swap or tokens is not None
        if drop:
            assert tokens is not None and args.model_key == 'oracle_dropout'
            visual = torch.zeros_like(visual)
        cond = {} if tokens is None else {'touch_tokens': tokens}
        handle = None
        if not real_surface and not report['constant_context_verified']:
            def check_context(module, inputs):
                context = inputs[0]
                expected = torch.cat((visual, tokens.to(visual)), dim=1).to(context)
                torch.testing.assert_close(context, expected, rtol=0, atol=0)
                torch.testing.assert_close(tokens, tokens[:1].expand_as(tokens), rtol=0, atol=0)
                assert visual.shape[1] == 7528 and tokens.shape[1] == 1024
                report['constant_context_verified'] = True
                report['constant_token_counts'] = dict(visual=visual.shape[1], surface=tokens.shape[1])
            handle = pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check_context)
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
            if handle is not None:
                handle.remove()
        assert torch.isfinite(loss)
        if per_object:
            assert len(values) == 1 and values[0].shape == (4,)
            torch.testing.assert_close(values[0].mean(), loss.float(), rtol=1e-5, atol=1e-7)
            return float(loss), values[0].cpu().tolist()
        return loss

    def assess(bank, draws, swaps, real_surface=False):
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
                            native, values = forward_loss(index, swap=swap, per_object=True, real_surface=real_surface)
                        scalar.append(native); matrix.append(values)
                    rows.append({'batch': index, 'split': meta['split'], 'swapped_surface': swap,
                                 'scalar_losses': scalar, 'per_object_losses': matrix})
        random.setstate(py)
        return rows

    report['initial_real_replay'] = assess(600000, 4, False, real_surface=True)
    assert report['initial_real_replay'] == baseline['assessments'][0]['rows']
    assert parameter_digest(model.named_parameters()) == initial_model
    report['initial_real_replay_passed'] = True
    report['assessments'].append({'step': 0, 'rows': assess(600000, 4, False)}); save()
    for step, item in enumerate(schedule, 1):
        index = item['batch']
        assert metadata[index]['split'] == 'fit'
        drop = args.model_key == 'oracle_dropout' and item['drop_visual']
        torch.manual_seed(seed+step); random.seed(seed+step)
        optimizer.zero_grad(set_to_none=True)
        with amp(device, precision):
            loss = forward_loss(index, drop=drop)
        loss.backward()
        assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
        gradients = component_gradient_norms(model) if step == 1 or step % 32 == 0 else {}
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
    assert report['constant_context_verified']
    assert tensor_digest(constant_features) == report['constant_bank']['sha256']
    rows = {(r['batch'], r['swapped_surface']): r for r in report['final_rows']}
    for index in range(32):
        for name in ('scalar_losses', 'per_object_losses'):
            assert rows[index, False][name] == rows[index, True][name], 'Constant surface swap changed loss'
    report['complete'] = True
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
