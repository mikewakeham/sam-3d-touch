"""Locate the loss of surface utility at saved steps 256/1024; no training."""
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
from analyze_object_transfer import validate
from tiny_fit_gpu import parameter_digest
from train import TouchTrainingModel, amp, build_optimizer, build_stage1_pipeline, prepare_batch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key', choices=('image', 'camera', 'oracle'), required=True)
    parser.add_argument('--fit-root', type=Path, default=Path('outputs/conditioning_investigation/object_transfer/manual'))
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.partial.json').exists():
        raise FileExistsError(args.output)
    fit_dir = args.fit_root / args.model_key
    fit_path = fit_dir / 'results.json'
    archived = HERE / 'object_transfer_returned_manual' / args.model_key / 'results.json'
    assert sha(fit_path) == sha(archived), 'Different source run; do not retrain'
    fit = json.loads(fit_path.read_text()); validate(fit)
    for name, digest in fit['source_sha256'].items():
        assert sha(REPO / name) == digest, name
    assert sha(HERE / 'object_transfer_plan.json') == fit['plan_sha256']
    reference_path = HERE / 'tiny_fit_returned_46083371/oracle/results.json'
    assert sha(reference_path) == fit['reference_sha256']
    reference = json.loads(reference_path.read_text())
    for step in (256, 1024):
        if not (fit_dir / f'checkpoint_{step}.pt').is_file():
            raise FileNotFoundError(fit_dir / f'checkpoint_{step}.pt')
    pipeline_path, data_path = (Path(reference['settings'][k]) for k in ('pipeline_config', 'data_config'))
    assert pipeline_path.read_text() == reference['pipeline_yaml']
    assert data_path.read_text() == reference['data_yaml']
    cfg = OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent / cfg.ss_generator_config_path).read_text() == reference['generator_yaml']
    seed, precision = 37, 'bf16'
    torch.cuda.set_device(args.device_index)
    device = torch.device('cuda', args.device_index)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pipeline = build_stage1_pipeline(pipeline_path, device)
    oracle = args.model_key == 'oracle'
    encoder = None
    if args.model_key != 'image':
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
        encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels,
                               trainable=False, use_position=False).to(device).eval()
    model = TouchTrainingModel(pipeline.ss_generator, encoder, False, oracle)
    optimizer, _ = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope='full'))
    del optimizer
    assert parameter_digest(model.named_parameters()) == fit['initial_model_sha256']
    parameters = dict(model.named_parameters())
    trained_names = {n for n, p in parameters.items() if p.requires_grad}
    frozen_hash = parameter_digest((n, p) for n, p in parameters.items() if n not in trained_names)
    model.requires_grad_(False)
    gen = pipeline.ss_generator
    gen.reverse_fn.training = True
    assert gen.reverse_fn.p_unconditional == 0 and gen.self_consistency_prob == 0 and gen.fm_eps_max == 0
    assert gen.loss_weights['shape'] == 1 and all(v == 0 for k, v in gen.loss_weights.items() if k != 'shape')
    data = load_data_config(data_path)
    assert data.get('touch', {}).get('source') == 'full_surface'
    data = copy.deepcopy(data); data['dataset']['split'] = 'val'
    ds = TouchDataset(data, include_touch=encoder is not None, oracle_point_frame=oracle)
    lookup = {r['sample_id']: r for r in ds.records}
    prepared, metadata = {}, []
    for meta in fit['input_batches'][24:]:
        assert meta['split'] == 'held_object'
        records = [lookup[s] for s in meta['sample_ids']]
        assert [r['object_id'] for r in records] == meta['object_ids']
        ds.records = records
        batch = collate_touch_batch([ds[i] for i in range(4)])
        targets, ca, kw, points, mask = prepare_batch(pipeline, batch, device, precision,
                                                    encoder is not None, False, oracle)
        assert len(ca) == 1 and not kw
        features = None
        if encoder is not None:
            assert mask.all() and points.shape[1] == encoder.num_points == 8192
            with torch.no_grad(), amp(device, precision):
                pp, mm, _, _ = encoder.prepare_points(points, mask)
                features = encoder.encoder.encode(pp, mm)['x'].detach()
        actual = dict(image_sha256=tensor_digest(batch['image']), pointmap_sha256=tensor_digest(batch['pointmap']),
                      visual_sha256=tensor_digest(ca[0]), target_sha256=tensor_digest(targets['shape']),
                      object_target_sha256=[tensor_digest(t) for t in targets['shape']])
        if features is not None:
            actual['features_sha256'] = tensor_digest(features)
        assert all(meta[k] == v for k, v in actual.items()), 'Input drift'
        prepared[meta['batch']] = (targets, ca[0], features)
        metadata.append(meta)
    report = dict(settings={'model_key': args.model_key, 'fit_root': str(args.fit_root),
                            'seed': seed, 'precision': precision, 'steps': [256, 1024],
                            'fresh_bank_base': 800000, 'draws': 8},
                  fit_report_sha256=sha(fit_path), input_batches=metadata,
                  source_sha256={**fit['source_sha256'], **{str(p.relative_to(REPO)): sha(p) for p in (
                      Path(__file__), HERE/'analyze_object_transfer.py', HERE/'analyze_transfer_checkpoints.py')}},
                  checkpoints=[], rows=[], context_checks=[], complete=False,
                  runtime={'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device)},
                  scope='Frozen early/late checkpoints; same 16 held identities and two views. '
                        'Correct versus three other objects within each batch; removing surface tokens '
                        'is a diagnostic distribution change, not a deployable fix. Native Stage-1 loss only.')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.with_suffix('.partial.json').write_text(json.dumps(report, indent=2)+'\n')

    checked = set()

    def measure(step, index, condition, bank, draws):
        targets, visual, features = prepared[index]
        scalar, matrix = [], []
        for draw in range(draws):
            draw_seed = bank + seed + 100*(index % 4) + draw
            torch.manual_seed(draw_seed); random.seed(draw_seed)
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

            functions['shape'] = capture
            with torch.no_grad(), amp(device, precision):
                tokens = None
                if condition != 'removed':
                    tokens = encoder.output_projection(features) + encoder.touch_embedding
                    if condition.startswith('wrong_'):
                        tokens = tokens.roll(int(condition[-1]), 0)
                cond = {} if tokens is None else {'touch_tokens': tokens}
                handle = None
                if (step, condition) not in checked:
                    def check_context(module, inputs):
                        context = inputs[0]
                        expected = visual if tokens is None else torch.cat((visual, tokens.to(visual)), dim=1)
                        torch.testing.assert_close(context, expected.to(context), rtol=0, atol=0)
                        report['context_checks'].append(dict(step=step, condition=condition,
                            visual_tokens=visual.shape[1], surface_tokens=0 if tokens is None else tokens.shape[1]))
                        checked.add((step, condition))
                    handle = pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check_context)
                try:
                    with patch.object(gen, 'loss_fn', functions):
                        loss, _ = gen.loss(targets, visual, **cond)
                finally:
                    if handle is not None:
                        handle.remove()
            assert len(values) == 1 and values[0].shape == (4,)
            torch.testing.assert_close(values[0].mean(), loss.float(), rtol=1e-5, atol=1e-7)
            assert torch.isfinite(loss) and torch.isfinite(values[0]).all()
            scalar.append(float(loss)); matrix.append(values[0].cpu().tolist())
        return dict(step=step, batch=index, condition=condition, scalar_losses=scalar, per_object_losses=matrix)

    for step in (256, 1024):
        path = fit_dir / f'checkpoint_{step}.pt'
        archive = torch.load(path, map_location='cpu', weights_only=True)
        assert archive['step'] == step and archive['settings'] == fit['settings']
        assert archive['conditioning_config'] == model.conditioning_config
        for key in ('source_sha256', 'plan_sha256', 'schedule_sha256'):
            assert archive[key] == fit[key]
        assert set(archive['model']) == trained_names
        with torch.no_grad():
            for name, value in archive['model'].items():
                assert value.shape == parameters[name].shape
                parameters[name].copy_(value)
        del archive
        model_hash = parameter_digest(model.named_parameters())
        if step == 1024:
            assert model_hash == fit['final_model_sha256']
        # Exact replay covers every held object, at both checkpoints. The late
        # replay also covers the original wrong-surface condition.
        expected_rows = fit['assessments'][1]['rows'] if step == 256 else fit['final_rows']
        replay = []
        for expected in expected_rows:
            if expected['split'] != 'held_object':
                continue
            condition = ('removed' if encoder is None else
                         'wrong_1' if expected['swapped_surface'] else 'correct')
            actual = measure(step, expected['batch'], condition, 600000 if step == 256 else 700000,
                             4 if step == 256 else 8)
            for name in ('scalar_losses', 'per_object_losses'):
                np.testing.assert_array_equal(actual[name], expected[name])
            replay.append(actual)
        checkpoint = dict(step=step, path=str(path), checkpoint_sha256=sha(path),
                          model_sha256=model_hash, replay=replay, replay_passed=True)
        report['checkpoints'].append(checkpoint)
        conditions = ('removed',) if encoder is None else ('correct', 'wrong_1', 'wrong_2', 'wrong_3', 'removed')
        for index in prepared:
            for condition in conditions:
                report['rows'].append(measure(step, index, condition, 800000, 8))
            save()
            print(args.model_key, step, 'held batch', index-23, '/8', flush=True)
        assert parameter_digest(model.named_parameters()) == model_hash
        assert parameter_digest((n, p) for n, p in parameters.items() if n not in trained_names) == frozen_hash
        checkpoint['parameters_unchanged'] = True
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
