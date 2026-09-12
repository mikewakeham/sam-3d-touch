"""Evaluate existing tiny-fit checkpoints on 32 disjoint objects; no training.

Per-object losses are captured inside the unchanged native scalar loss call,
with numerical agreement checks. The selection predates the dropout result.
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
from probe_oracle_visual_streams_gpu import locate_fit
from tiny_fit_gpu import parameter_digest
from train import TouchTrainingModel, amp, build_optimizer, build_stage1_pipeline, prepare_batch

MODEL_KEYS = ('image_original', 'camera_original', 'camera_dropout', 'oracle_original', 'oracle_dropout')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key', choices=MODEL_KEYS, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reference-root', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.partial.json').exists():
        raise FileExistsError(args.output)
    arm, policy = args.model_key.split('_')
    single_path = args.reference_root / f'tiny_fit_returned_46083371/{arm}/results.json'
    multi_path = args.reference_root / f'multiple_view_returned_46083371/{arm}.json'
    single, multi = (json.loads(p.read_text()) for p in (single_path, multi_path))
    trained_path = (args.reference_root / f'visual_dropout_returned_manual/{arm}/results.json'
                    if policy == 'dropout' else multi_path)
    trained = json.loads(trained_path.read_text())
    assert multi['source_reference_sha256'] == sha(single_path)
    assert multi['driver_sha256'] == sha(HERE / 'fit_multiple_views_gpu.py')
    assert multi['view_selector_sha256'] == sha(HERE / 'probe_tiny_fit_views.py')
    for name, expected in trained['source_sha256'].items():
        assert sha(REPO / name) == expected, name
    if policy == 'dropout':
        assert trained['complete'] and trained['baseline_preflight_passed'] and trained['initial_parameters_restored']
        assert trained['reference_sha256']['multiple'] == sha(multi_path)
        assert trained['reference_sha256']['single'] == sha(single_path)
    checkpoint = args.checkpoint
    if checkpoint is None:
        if policy == 'dropout':
            checkpoint = REPO / f'outputs/conditioning_investigation/visual_dropout/manual/{arm}/checkpoint_1000.pt'
        else:
            checkpoint = locate_fit(None, multi) / 'fitted_parameters.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{checkpoint}; pass --checkpoint with the existing {args.model_key} checkpoint. Do not retrain.')
    settings = single['settings']
    assert settings['seed'] == 29 and settings['precision'] == 'bf16' and settings['objects'] == 4
    seed, precision = settings['seed'], settings['precision']
    pipeline_path, data_path = Path(settings['pipeline_config']), Path(settings['data_config'])
    assert pipeline_path.read_text() == single['pipeline_yaml']
    assert data_path.read_text() == single['data_yaml']
    cfg = OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent / cfg.ss_generator_config_path).read_text() == single['generator_yaml']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda')
    pipeline = build_stage1_pipeline(pipeline_path, device)
    encoder = None
    if arm != 'image':
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
        encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels,
                               trainable=False, use_position=False).to(device).eval()
    model = TouchTrainingModel(pipeline.ss_generator, encoder, False, arm == 'oracle')
    optimizer, _ = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope='full'))
    del optimizer
    assert parameter_digest(model.named_parameters()) == trained['initial_all_parameters_sha256']
    archive = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert archive['step'] == 1000 and archive['settings'] == trained['settings']
    assert archive['conditioning_config'] == model.conditioning_config
    parameters = dict(model.named_parameters())
    assert set(archive['model']) == {n for n, p in parameters.items() if p.requires_grad}
    with torch.no_grad():
        for name, value in archive['model'].items():
            parameters[name].copy_(value)
    del archive
    final_hash = parameter_digest(model.named_parameters())
    assert final_hash == trained['final_all_parameters_sha256']
    model.requires_grad_(False)
    gen = pipeline.ss_generator
    gen.reverse_fn.training = True
    assert gen.reverse_fn.p_unconditional == 0 and gen.self_consistency_prob == 0 and gen.fm_eps_max == 0
    assert gen.loss_weights['shape'] == 1 and all(v == 0 for k, v in gen.loss_weights.items() if k != 'shape')
    data = load_data_config(data_path)
    datasets = []
    for split in ('train', 'val'):
        config = copy.deepcopy(data); config['dataset']['split'] = split
        datasets.append(TouchDataset(config, include_touch=encoder is not None, oracle_point_frame=arm == 'oracle'))
    ds = datasets[0]
    lookup, split_of = {}, {}
    for split, dataset in zip(('train', 'val'), datasets):
        for record in dataset.records:
            assert record['sample_id'] not in lookup
            lookup[record['sample_id']] = record
            previous = split_of.setdefault(record['object_id'], split)
            assert previous == split
    selection_path = args.reference_root / 'unseen_object_selection.json'
    selection = json.loads(selection_path.read_text())
    fitted = {lookup[s]['object_id'] for s in single['sample_ids']}
    selected = [lookup[s] for s in selection['sample_ids']]
    by_object = {}
    for record in selected:
        assert record['object_id'] not in fitted
        by_object.setdefault(record['object_id'], []).append(record)
    assert len(by_object) == 32 and all(len(v) == 2 for v in by_object.values())
    assert len({r['sample_id'] for r in selected}) == 64
    object_ids = sorted(by_object)
    # Same four distinct identities occupy each batch in both selected views.
    batches = [[by_object[o][view] for o in object_ids[start:start+4]]
               for view in (0, 1) for start in range(0, 32, 4)]

    def prepare(records):
        ds.records = records
        batch = collate_touch_batch([ds[i] for i in range(4)])
        targets, ca, kw, points, mask = prepare_batch(pipeline, batch, device, precision,
                                                    encoder is not None, False, arm == 'oracle')
        assert len(ca) == 1 and not kw
        touch = features = None
        if encoder is not None:
            assert mask.all() and points.shape[1] == encoder.num_points == 8192
            with torch.no_grad(), amp(device, precision):
                pp, mm, _, _ = encoder.prepare_points(points, mask)
                features = encoder.encoder.encode(pp, mm)['x']
                touch = encoder.output_projection(features) + encoder.touch_embedding
        metadata = dict(sample_ids=[r['sample_id'] for r in records],
                        object_ids=[r['object_id'] for r in records],
                        image_sha256=tensor_digest(batch['image']), pointmap_sha256=tensor_digest(batch['pointmap']),
                        target_sha256=tensor_digest(targets['shape']), visual_sha256=tensor_digest(ca[0]))
        if features is not None:
            metadata['features_sha256'] = tensor_digest(features)
        return (targets, ca[0], touch), metadata

    def measured_loss(prepared, draw_seed, swap=False):
        targets, visual, touch = prepared
        cond = {} if touch is None else {'touch_tokens': touch.roll(1, 0) if swap else touch}
        original_fn = gen.loss_fn
        functions = dict(original_fn) if isinstance(original_fn, dict) else {k: original_fn for k in gen.loss_weights}
        shape_fn = functions['shape']
        captured = []

        def capture(pred, target):
            native = shape_fn(pred, target)
            per_object = F.mse_loss(pred.float(), target.float(), reduction='none').flatten(1).mean(1)
            torch.testing.assert_close(per_object.mean(), native.float(), rtol=1e-5, atol=1e-7)
            captured.append(per_object.detach())
            return native

        functions['shape'] = capture
        torch.manual_seed(draw_seed); random.seed(draw_seed)
        with torch.no_grad(), amp(device, precision), patch.object(gen, 'loss_fn', functions):
            loss, _ = gen.loss(targets, visual, **cond)
        assert len(captured) == 1 and captured[0].shape == (4,)
        torch.testing.assert_close(captured[0].mean(), loss.float(), rtol=1e-5, atol=1e-7)
        assert torch.isfinite(loss) and torch.isfinite(captured[0]).all()
        return float(loss), captured[0].cpu().tolist()

    anchor, anchor_meta = prepare([lookup[s] for s in single['sample_ids']])
    expected_meta = multi['input_batches'][0]
    for key in ('sample_ids', 'image_sha256', 'pointmap_sha256', 'target_sha256'):
        assert anchor_meta[key] == expected_meta[key], key
    if encoder is not None:
        assert anchor_meta['features_sha256'] == expected_meta['features_sha256']
    expected_anchor = (trained['final_fresh']['visual_present'][0] if policy == 'dropout'
                       else trained['assessments'][-1]['rows'][0])
    replay_bank = 400000 if policy == 'dropout' else 100000
    replay = {}
    for swap in ([False, True] if encoder is not None else [False]):
        key = 'swapped_surface_native_losses' if swap else 'fresh_noise_native_losses'
        replay[key] = [measured_loss(anchor, replay_bank + seed + draw, swap)[0] for draw in range(8)]
        np.testing.assert_array_equal(replay[key], expected_anchor[key])
    del anchor
    report = dict(settings={'model_key': args.model_key, 'checkpoint': str(checkpoint), 'draws': 8,
                            'seed': seed, 'precision': precision, 'unseen_bank_base': 500000},
                  reference_sha256={'single': sha(single_path), 'multiple': sha(multi_path), 'trained': sha(trained_path)},
                  selection=selection, selection_sha256=sha(selection_path),
                  fitted_object_ids=sorted(fitted), final_parameters_sha256=final_hash,
                  source_sha256={str(p.relative_to(REPO)): sha(p) for p in
                                 (Path(__file__), REPO/'train.py', REPO/'dataloader.py',
                                  REPO/'sam3d_objects/model/backbone/generator/shortcut/model.py')},
                  anchor_replay=replay, anchor_replay_passed=True, input_batches=[], rows=[], complete=False,
                  runtime={'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()},
                  scope='32 identities unseen by these four-object finetunes, two fixed views each. '
                        'Diagnostic selection, not a population benchmark or a claim of absence from '
                        'pretraining. Frozen checkpoints; no optimization or decoding.')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.with_suffix('.partial.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    for index, records in enumerate(batches):
        prepared, metadata = prepare(records)
        report['input_batches'].append({'batch': index, **metadata})
        for swap in ([False, True] if encoder is not None else [False]):
            matrix, scalar = [], []
            # Both views of an object receive the same time/noise draws, as do
            # all checkpoints and both surface treatments.
            for draw in range(8):
                value, per_object = measured_loss(prepared, 500000 + seed + 100 * (index % 8) + draw, swap)
                scalar.append(value); matrix.append(per_object)
            for i, record in enumerate(records):
                report['rows'].append({'sample_id': record['sample_id'], 'object_id': record['object_id'],
                    'dataset_split': split_of[record['object_id']], 'view_index': index // 8,
                    'batch': index, 'swapped_surface': swap,
                    'surface_object_id': None if encoder is None else (
                        records[(i-1) % 4]['object_id'] if swap else record['object_id']),
                    'losses': [v[i] for v in matrix]})
            report.setdefault('batch_scalar_losses', []).append({'batch': index, 'swapped_surface': swap,
                                                                 'losses': scalar})
        save()
        print(args.model_key, 'completed unseen batch', index + 1, '/ 16', flush=True)
    assert parameter_digest(model.named_parameters()) == final_hash
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('Wrote', args.output, flush=True)


if __name__ == '__main__':
    main()
