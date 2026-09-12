"""Complete the rotation x normalization experiment with frozen VecSetX.

Reproduce historical camera/oracle features, inputs, weights and initial losses
before fitting either mixed coordinate treatment. Stage-1 shape training only.
"""
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
from dataloader import build_dataloader, load_data_config, collate_touch_batch
from train import build_stage1_pipeline, prepare_batch, build_optimizer, amp, TouchTrainingModel, trainable_state_dict, component_gradient_norms
from tiny_fit_gpu import parameter_digest
from probe_tiny_fit_views import choose_views
from rollout_gpu import occupancy, geometry_metrics
from experiments.noisy_target.evaluate import tensor_digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mixed_coordinates(object_points, camera_points, object_from_camera_rotation):
    """Inputs are already normalized. Do not call prepare_points on outputs."""
    inverse = object_from_camera_rotation.float()
    forward = torch.linalg.inv(inverse)
    rotation_only = torch.bmm(object_points.float(), forward.transpose(1, 2))
    normalization_only = torch.bmm(camera_points.float(), inverse.transpose(1, 2))
    torch.testing.assert_close(torch.bmm(rotation_only, inverse.transpose(1, 2)), object_points.float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.bmm(normalization_only, forward.transpose(1, 2)), camera_points.float(), rtol=1e-5, atol=1e-6)
    return {'rotation_only': rotation_only, 'normalization_only': normalization_only}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['rotation_only', 'normalization_only'], required=True)
    p.add_argument('--reference-root', type=Path, default=HERE)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    single, multiple, reference_hashes = {}, {}, {}
    for arm in ['camera', 'oracle']:
        sp = a.reference_root / 'tiny_fit_returned_46083371' / arm / 'results.json'
        mp = a.reference_root / 'multiple_view_returned_46083371' / f'{arm}.json'
        single[arm], multiple[arm] = json.loads(sp.read_text()), json.loads(mp.read_text())
        reference_hashes[arm] = {'single': sha(sp), 'multiple': sha(mp)}
        old = multiple[arm]
        assert old['source_reference_sha256'] == sha(sp)
        assert old['driver_sha256'] == sha(HERE / 'fit_multiple_views_gpu.py')
        assert old['view_selector_sha256'] == sha(HERE / 'probe_tiny_fit_views.py')
        assert len(old['training']) == 1000 and old['training'][-1]['step'] == 1000
        assert [v['step'] for v in old['assessments']] == [0, 100, 300, 1000]
        for rel, digest in old['source_sha256'].items():
            assert sha(REPO / rel) == digest, rel
    reference = single['oracle']
    for key in ['sample_ids', 'initial_all_parameters_sha256', 'initial_trainable_ca_sha256',
                'pipeline_yaml', 'generator_yaml', 'data_yaml', 'source_sha256']:
        assert reference[key] == single['camera'][key], key
    settings = reference['settings']
    seed, precision = settings['seed'], settings['precision']
    assert seed == 29 and precision == 'bf16' and settings['steps'] == 1000 and settings['objects'] == 4
    for arm in multiple:
        assert multiple[arm]['settings'] == {**multiple['oracle']['settings'], 'arm': arm}
    pipeline_path, data_path = Path(settings['pipeline_config']), Path(settings['data_config'])
    assert pipeline_path.read_text() == reference['pipeline_yaml']
    assert data_path.read_text() == reference['data_yaml']
    cfg = OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent / cfg.ss_generator_config_path).read_text() == reference['generator_yaml']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda')
    pipeline = build_stage1_pipeline(pipeline_path, device)
    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
    encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels,
                           trainable=False, use_position=False).to(device).eval()
    model = TouchTrainingModel(pipeline.ss_generator, encoder, False, True)
    optimizer, parameters = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope='full'))
    initial_hash = parameter_digest(model.named_parameters())
    assert initial_hash == reference['initial_all_parameters_sha256']
    assert parameter_digest((n, v) for n, v in pipeline.backbone.named_parameters() if v.requires_grad) == reference['initial_trainable_ca_sha256']
    gen = pipeline.ss_generator
    assert gen.reverse_fn.p_unconditional == 0 and gen.self_consistency_prob == 0 and gen.fm_eps_max == 0
    data = load_data_config(data_path); data['dataset']['split'] = 'train'
    assert data.get('touch', {}).get('source') == 'full_surface'
    ds = build_dataloader(data, 4, 0, shuffle=False, include_touch=True, oracle_point_frame=True).dataset
    groups = choose_views(ds.records, reference['sample_ids'], count=6)
    assert len(groups) == 7
    prepared, metadata = [], []
    for group, records in enumerate(groups):
        ds.records = records
        batch = collate_touch_batch([ds[i] for i in range(4)])
        targets, ca, kw, po, mask = prepare_batch(pipeline, batch, device, precision, True, False, True)
        tc, cac, kwc, pc, mc = prepare_batch(pipeline, batch, device, precision, True, False, False)
        assert torch.equal(mask, mc) and mask.all() and po.shape[1] == encoder.num_points == 8192
        # Only the xyz frame should change. Compare all precomputed condition
        # leaves, including non-tensor metadata, instead of assuming equality.
        from torch.utils import _pytree
        leaves_o, tree_o = _pytree.tree_flatten((targets, ca, kw))
        leaves_c, tree_c = _pytree.tree_flatten((tc, cac, kwc))
        assert tree_o == tree_c
        for x, y in zip(leaves_o, leaves_c):
            if torch.is_tensor(x):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            else:
                assert x == y
        with torch.no_grad():
            pp_o, mm_o, _, _ = encoder.prepare_points(po, mask)
            pp_c, mm_c, _, _ = encoder.prepare_points(pc, mask)
            assert torch.equal(mm_o, mm_c)
            mixed = mixed_coordinates(pp_o, pp_c, batch['object_from_camera'][:, :3, :3].to(device))
        features = {}
        with torch.no_grad(), amp(device, precision):
            for name, pp in [('oracle', pp_o), ('camera', pp_c), (a.arm, mixed[a.arm])]:
                features[name] = encoder.encoder.encode(pp, mm_o)['x'].detach()
                assert torch.isfinite(features[name]).all()
        entry = {'group': group, 'split': 'fit' if group < 4 else 'reserved_view',
                 'sample_ids': [r['sample_id'] for r in records],
                 'image_sha256': tensor_digest(batch['image']), 'pointmap_sha256': tensor_digest(batch['pointmap']),
                 'target_sha256': tensor_digest(targets['shape']),
                 'features_sha256': {name: tensor_digest(value) for name, value in features.items()},
                 'mixed_coordinates_sha256': tensor_digest(mixed[a.arm]),
                 'object_from_camera_sha256': tensor_digest(batch['object_from_camera'])}
        for arm in ['camera', 'oracle']:
            old = multiple[arm]['input_batches'][group]
            for key in ['group', 'split', 'sample_ids', 'image_sha256', 'pointmap_sha256', 'target_sha256']:
                assert entry[key] == old[key], (arm, group, key)
            assert entry['features_sha256'][arm] == old['features_sha256'], (arm, group, 'features')
        prepared.append((targets, ca, kw, features)); metadata.append(entry)
    report = {'settings': {'arm': a.arm, 'steps': 1000, 'seed': seed, 'precision': precision,
                          'objects': 4, 'fit_views_per_object': 4, 'reserved_views_per_object': 3,
                          'output_dir': str(a.output_dir)},
              'initial_all_parameters_sha256': initial_hash,
              'initial_trainable_ca_sha256': reference['initial_trainable_ca_sha256'],
              'reference_sha256': reference_hashes, 'source_sha256': reference['source_sha256'],
              'driver_sha256': sha(Path(__file__)), 'input_batches': metadata,
              'trainable_parameters': {g['name']: sum(v.numel() for v in g['params']) for g in optimizer.param_groups},
              'training': [], 'assessments': [], 'baseline_preflight': {}, 'sampled': [],
              'runtime': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()},
              'scope': 'Two missing coordinate-factor treatments. Frozen VecSetX; fixed projector/full shape CA scope. '
                       'Four objects, four trained and three reserved views. Known transforms are privileged. '
                       'Historical endpoint reuse requires exact input/features and initial-loss reproduction. No Stage 2.'}
    a.output_dir.mkdir(parents=True)

    def save():
        (a.output_dir / 'results.partial.json').write_text(json.dumps(report, indent=2) + '\n')

    def conditions(group, arm, swap=False):
        targets, ca, kw, features = prepared[group]
        x = features[arm].roll(1, 0) if swap else features[arm]
        tokens = encoder.output_projection(x) + encoder.touch_embedding
        return targets, ca, {**kw, 'touch_tokens': tokens}

    def assess(arm):
        py = random.getstate()
        rows = []
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            gen.reverse_fn.training = True
            for group in range(7):
                values, wrong = [], []
                for draw in range(8):
                    for swap in [False, True]:
                        torch.manual_seed(100000 + seed + draw); random.seed(100000 + seed + draw)
                        with amp(device, precision):
                            targets, ca, kw = conditions(group, arm, swap)
                            loss, _ = gen.loss(targets, *ca, **kw)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite assessment')
                        (wrong if swap else values).append(float(loss))
                rows.append({'group': group, 'split': metadata[group]['split'],
                             'fresh_noise_native_losses': values, 'swapped_surface_native_losses': wrong})
        random.setstate(py)
        return rows

    # These are real replayed model checks, not merely comparison of old JSONs.
    for arm in ['camera', 'oracle']:
        rows = assess(arm)
        expected = multiple[arm]['assessments'][0]['rows']
        report['baseline_preflight'][arm] = rows; save()
        for x, y in zip(rows, expected):
            for key in ['fresh_noise_native_losses', 'swapped_surface_native_losses']:
                np.testing.assert_array_equal(x[key], y[key], err_msg=f'{arm}: historical initial loss changed')
    report['baseline_preflight_passed'] = True
    report['assessments'].append({'step': 0, 'rows': assess(a.arm)}); save()
    for step in range(1, 1001):
        group = (step - 1) % 4
        torch.manual_seed(seed + step); random.seed(seed + step)
        optimizer.zero_grad(set_to_none=True)
        with amp(device, precision):
            targets, ca, kw = conditions(group, a.arm)
            loss, _ = gen.loss(targets, *ca, **kw)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training')
        loss.backward()
        assert not any(v.grad is not None for v in model.parameters() if not v.requires_grad)
        gradients = component_gradient_norms(model) if step == 1 or step % 20 == 0 else {}
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        report['training'].append({'step': step, 'group': group, 'loss': float(loss),
                                   'preclip_gradient_norm': float(norm), **gradients})
        if step % 20 == 0:
            print(a.arm, step, float(loss), flush=True)
        if step in [100, 300, 1000]:
            report['assessments'].append({'step': step, 'rows': assess(a.arm)}); save()
        if step in [300, 1000]:
            torch.save({'model': trainable_state_dict(model), 'optimizer': optimizer.state_dict(),
                        'step': step, 'settings': report['settings'], 'reference_sha256': reference_hashes,
                        'driver_sha256': report['driver_sha256']}, a.output_dir / f'checkpoint_{step}.pt')
    report['final_all_parameters_sha256'] = parameter_digest(model.named_parameters())
    assert report['final_all_parameters_sha256'] != initial_hash
    # Same conditional-only Stage-1 rollout control as the historical experiment.
    decoder = pipeline.init_ss_decoder(cfg.ss_decoder_config_path, cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    with torch.no_grad(), amp(device, precision):
        support_targets = [occupancy(decoder, prepared[0][0]['shape'][i:i+1]) for i in range(4)]
    gen.no_shortcut = True; gen.inference_steps = 25
    gen.rescale_t = float(cfg.get('ss_rescale_t', 3))
    gen.reverse_fn.interval = list(cfg.get('ss_cfg_interval', [0, 500]))
    gen.reverse_fn.strength = 0; gen.reverse_fn.training = False
    report['sample_noise_sha256'] = []
    with torch.no_grad():
        for draw in [0, 1]:
            torch.manual_seed(200000 + seed + draw)
            noise = gen._generate_x0(prepared[0][0])
            hashes = {k: tensor_digest(v) for k, v in noise.items()}
            assert hashes == multiple['oracle']['sample_noise_sha256'][draw]
            report['sample_noise_sha256'].append(hashes)
            for group in range(7):
                with amp(device, precision):
                    targets, ca, kw = conditions(group, a.arm)
                    shapes = {k: tuple(v.shape) for k, v in targets.items()}
                    with patch.object(gen, '_generate_noise', side_effect=lambda *a, **k: {n: v.clone() for n, v in noise.items()}):
                        pred = gen(shapes, device, *ca, **kw)['shape']
                if not torch.isfinite(pred).all():
                    raise FloatingPointError('Nonfinite rollout')
                for i, sid in enumerate(metadata[group]['sample_ids']):
                    with amp(device, precision):
                        support = occupancy(decoder, pred[i:i+1])
                    artifact = f'{sid}_draw{draw}.npz'
                    np.savez_compressed(a.output_dir / artifact, prediction=pred[i].float().cpu().numpy(),
                        target=targets['shape'][i].float().cpu().numpy(), predicted_occupancy=support, target_occupancy=support_targets[i])
                    report['sampled'].append({'group': group, 'split': metadata[group]['split'], 'sample_id': sid,
                        'noise_draw': draw, 'cfg': 0, 'artifact': artifact,
                        'latent_mse': float((pred[i].float() - targets['shape'][i].float()).square().mean()),
                        **geometry_metrics(support, support_targets[i])})
                save()
    report['complete'] = True
    (a.output_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
