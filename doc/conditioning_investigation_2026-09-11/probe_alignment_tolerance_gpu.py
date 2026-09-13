"""Reuse fitted oracle models: reconstruct GT support and measure rotation tolerance."""
import argparse
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
from omegaconf import OmegaConf
from alignment_tolerance_protocol import cases, rotation
from dataloader import build_dataloader, collate_touch_batch, load_data_config
from experiments.noisy_target.evaluate import tensor_digest
from probe_tiny_fit_views import choose_views
from rollout_gpu import occupancy, geometry_metrics
from tiny_fit_gpu import parameter_digest
from train import TouchTrainingModel, amp, build_optimizer, build_stage1_pipeline, prepare_batch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else stream_sha(stream)


def stream_sha(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024*1024), b''):
        value.update(chunk)
    return value.hexdigest()


def locate_dropout(explicit, reference):
    roots = [explicit] if explicit else [p.parent for p in (REPO/'outputs/conditioning_investigation/visual_dropout').rglob('results.json')]
    matches = [p for p in roots if p is not None and (p/'results.json').is_file()
               and (p/'checkpoint_1000.pt').is_file() and json.loads((p/'results.json').read_text()) == reference]
    if len(matches) != 1:
        raise ValueError(f'Need one matching existing dropout fit; found {matches}. Pass --dropout-fit-dir. Do not retrain.')
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, choices=range(4), required=True)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--dropout-fit-dir', type=Path)
    parser.add_argument('--original-fit-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    single_path = HERE/'tiny_fit_returned_46083371/oracle/results.json'
    multi_path = HERE/'multiple_view_returned_46083371/oracle.json'
    drop_path = HERE/'visual_dropout_returned_manual/oracle/results.json'
    single, multi, drop = [json.loads(p.read_text()) for p in (single_path, multi_path, drop_path)]
    assert drop['complete'] and drop['settings']['steps'] == 1000
    assert drop['reference_sha256'] == {'single': sha(single_path), 'multiple': sha(multi_path)}
    assert multi['source_reference_sha256'] == sha(single_path)
    assert multi['driver_sha256'] == sha(HERE/'fit_multiple_views_gpu.py')
    assert multi['view_selector_sha256'] == sha(HERE/'probe_tiny_fit_views.py')
    for name, value in drop['source_sha256'].items():
        assert sha(REPO/name) == value, name
    drop_dir = locate_dropout(args.dropout_fit_dir, drop)
    original_dir = args.original_fit_dir or Path(drop['settings']['baseline_fit_dir'])
    if args.shard == 0:
        assert json.loads((original_dir/'results.json').read_text()) == multi
    pipeline_path = Path(single['settings']['pipeline_config'])
    data_path = Path(single['settings']['data_config'])
    assert pipeline_path.read_text() == single['pipeline_yaml']
    assert data_path.read_text() == single['data_yaml']
    cfg = OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent/cfg.ss_generator_config_path).read_text() == single['generator_yaml']
    torch.cuda.set_device(args.device_index)
    device = torch.device('cuda', args.device_index)
    random.seed(29); np.random.seed(29); torch.manual_seed(29)
    pipeline = build_stage1_pipeline(pipeline_path, device)
    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
    encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels, trainable=False, use_position=False).to(device).eval()
    model = TouchTrainingModel(pipeline.ss_generator, encoder, False, True)
    optimizer, _ = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope='full'))
    del optimizer
    assert parameter_digest(model.named_parameters()) == multi['initial_all_parameters_sha256']
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    parameters = dict(model.named_parameters())
    report = {'settings': {'shard': args.shard, 'seed': 29, 'precision': 'bf16', 'steps': 25,
                           'cfg': 0, 'sampling_bank': 200000, 'sampling_draws': 2,
                           'cases': cases(args.shard)},
              'reference_sha256': {p.name if p == multi_path else str(p.relative_to(HERE)): sha(p)
                                   for p in (single_path, multi_path, drop_path)},
              'source_sha256': {**drop['source_sha256'], **{str(p.relative_to(REPO)): sha(p) for p in
                  (Path(__file__), HERE/'alignment_tolerance_protocol.py', HERE/'analyze_alignment_tolerance.py', HERE/'rollout_gpu.py', HERE/'tiny_fit_gpu.py')}},
              'runtime': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device)},
              'checkpoint_checks': {}, 'input_batches': [], 'replay': [], 'rows': [], 'coordinate_checks': [],
              'scope': 'No training. Known residual rotations before VecSetX normalization/encoding. '
                       'Four already fitted identities, four fitted and three reserved views. '
                       'No new-object inference or deployed pose estimate. Stage 1 only.', 'complete': False}
    args.output_dir.mkdir(parents=True)
    def save():
        (args.output_dir/'results.partial.json').write_text(json.dumps(report, indent=2)+'\n')
    gen = pipeline.ss_generator
    assert gen.reverse_fn.p_unconditional == 0 and gen.self_consistency_prob == 0 and gen.fm_eps_max == 0
    assert gen.loss_weights['shape'] == 1 and all(v == 0 for k, v in gen.loss_weights.items() if k != 'shape')
    data = load_data_config(data_path); data['dataset']['split'] = 'train'
    ds = build_dataloader(data, 4, 0, shuffle=False, include_touch=True, oracle_point_frame=True).dataset
    groups = choose_views(ds.records, single['sample_ids'], count=6)
    prepared = []
    for group, records in enumerate(groups):
        ds.records = records
        batch = collate_touch_batch([ds[i] for i in range(4)])
        targets, ca, kw, points, mask = prepare_batch(pipeline, batch, device, 'bf16', True, False, True)
        assert len(ca) == 1 and not kw and mask.all() and points.shape[1] == encoder.num_points == 8192
        with torch.no_grad(), amp(device, 'bf16'):
            pp, mm, _, _ = encoder.prepare_points(points, mask)
            features = encoder.encoder.encode(pp, mm)['x'].detach()
        entry = dict(group=group, split='fit' if group < 4 else 'reserved_view', sample_ids=[r['sample_id'] for r in records],
                     image_sha256=tensor_digest(batch['image']), pointmap_sha256=tensor_digest(batch['pointmap']),
                     target_sha256=tensor_digest(targets['shape']), features_sha256=tensor_digest(features))
        assert entry == multi['input_batches'][group] == drop['input_batches'][group]
        report['input_batches'].append(entry)
        prepared.append((targets, ca[0], points, mask, features))
    decoder = pipeline.init_ss_decoder(cfg.ss_decoder_config_path, cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    with torch.no_grad(), amp(device, 'bf16'):
        target_support = [occupancy(decoder, prepared[0][0]['shape'][i:i+1]) for i in range(4)]
    assert all(torch.equal(p[0]['shape'], prepared[0][0]['shape']) for p in prepared)
    current = None
    for policy, condition, axis, degrees in cases(args.shard):
        if current != policy:
            if current is not None:
                assert parameter_digest(model.named_parameters()) == report['checkpoint_checks'][current]['parameter_sha256']
            path = (original_dir/'fitted_parameters.pt') if policy == 'original' else (drop_dir/'checkpoint_1000.pt')
            archive = torch.load(path, map_location='cpu', weights_only=True)
            expected = multi if policy == 'original' else drop
            assert archive['step'] == 1000 and archive['settings'] == expected['settings']
            assert archive['conditioning_config'] == {'no_pointmap': False, 'oracle_point_frame': True}
            assert set(archive['model']) == trainable_names
            for n, value in archive['model'].items():
                assert value.shape == parameters[n].shape
            with torch.no_grad():
                for n, value in archive['model'].items():
                    parameters[n].copy_(value)
            del archive
            final_hash = parameter_digest(model.named_parameters())
            assert final_hash == expected['final_all_parameters_sha256']
            model.requires_grad_(False)
            report['checkpoint_checks'][policy] = {'path': str(path), 'checkpoint_sha256': sha(path), 'parameter_sha256': final_hash}
            current = policy
            gen.reverse_fn.training = True
            old_rows = (drop['baseline_fresh'] if policy == 'original' else drop['final_fresh'])['visual_present']
            for group, (targets, visual, points, mask, features) in enumerate(prepared):
                with torch.no_grad(), amp(device, 'bf16'):
                    tokens = encoder.output_projection(features)+encoder.touch_embedding
                    for wrong in (False, True):
                        values = []
                        for draw in range(8):
                            torch.manual_seed(400000+29+draw); random.seed(400000+29+draw)
                            loss, _ = gen.loss(targets, visual, touch_tokens=tokens.roll(1, 0) if wrong else tokens)
                            values.append(float(loss))
                        key = 'swapped_surface_native_losses' if wrong else 'fresh_noise_native_losses'
                        np.testing.assert_allclose(values, old_rows[group][key], rtol=1e-5, atol=1e-7)
                        report['replay'].append(dict(policy=policy, group=group, wrong=wrong, losses=values))
            save()
        gen.no_shortcut = True; gen.inference_steps = 25
        gen.rescale_t = float(cfg.get('ss_rescale_t', 3))
        gen.reverse_fn.interval = list(cfg.get('ss_cfg_interval', [0, 500]))
        gen.reverse_fn.strength = 0; gen.reverse_fn.training = False
        r = torch.tensor(rotation(axis, degrees), device=device, dtype=torch.float32)
        torch.testing.assert_close(r.T@r, torch.eye(3, device=device), rtol=0, atol=1e-6)
        torch.testing.assert_close(torch.det(r), r.new_tensor(1), rtol=0, atol=1e-6)
        for group, (targets, visual, points, mask, features) in enumerate(prepared):
            with torch.no_grad():
                rotated = points if degrees == 0 else points@r.T
                torch.testing.assert_close(rotated@r, points, rtol=1e-5, atol=1e-6)
                with amp(device, 'bf16'):
                    pp, mm, _, _ = encoder.prepare_points(rotated, mask)
                    condition_features = features if degrees == 0 else encoder.encoder.encode(pp, mm)['x']
                    tokens = encoder.output_projection(condition_features)+encoder.touch_embedding
                    if condition == 'wrong_surface':
                        tokens = tokens.roll(1, 0)
                report['coordinate_checks'].append(dict(policy=policy, condition=condition, group=group,
                    rotation=rotation(axis, degrees), actual_angle_degrees=float(torch.rad2deg(torch.acos(((torch.trace(r)-1)/2).clamp(-1,1)))),
                    object_frame_points_sha256=tensor_digest(points), normalized_points_sha256=tensor_digest(pp),
                    features_sha256=tensor_digest(condition_features),
                    point_displacement_rms=torch.sqrt((rotated-points).square().sum(-1).mean(-1)).cpu().tolist()))
                context_seen = []
                def check_context(module, inputs):
                    expected_context = torch.cat((visual, tokens.to(visual)), 1).to(inputs[0])
                    torch.testing.assert_close(inputs[0], expected_context, rtol=0, atol=0)
                    context_seen.append(True)
                hook = pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check_context)
                try:
                    for draw in range(2):
                        torch.manual_seed(200000+29+draw)
                        noise = gen._generate_x0(targets)
                        with amp(device, 'bf16'), patch.object(gen, '_generate_noise', side_effect=lambda *a, **k: {n:v.clone() for n,v in noise.items()}):
                            pred = gen({n:tuple(v.shape) for n,v in targets.items()}, device, visual, touch_tokens=tokens)['shape']
                        assert torch.isfinite(pred).all()
                        supports = []
                        for i, sid in enumerate(report['input_batches'][group]['sample_ids']):
                            with amp(device, 'bf16'):
                                support = occupancy(decoder, pred[i:i+1])
                            supports.append(support)
                            report['rows'].append(dict(policy=policy, condition=condition, axis=axis, degrees=degrees,
                                group=group, split='fit' if group < 4 else 'reserved_view', sample_id=sid, object_id=groups[group][i]['object_id'],
                                draw=draw, noise_sha256={n:tensor_digest(v[i]) for n,v in noise.items()},
                                latent_mse=float((pred[i].float()-targets['shape'][i].float()).square().mean()),
                                **geometry_metrics(support, target_support[i])))
                        np.savez_compressed(args.output_dir/f'{policy}_{condition}_g{group}_d{draw}.npz',
                            prediction=pred.float().cpu().numpy(), target=targets['shape'].cpu().numpy(),
                            predicted_occupancy=np.stack(supports), target_occupancy=np.stack(target_support))
                finally:
                    hook.remove()
                assert context_seen
            save(); print('shard', args.shard, policy, condition, 'group', group, 'done', flush=True)
    assert parameter_digest(model.named_parameters()) == report['checkpoint_checks'][current]['parameter_sha256']
    report['parameters_unchanged'] = True; report['complete'] = True
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
