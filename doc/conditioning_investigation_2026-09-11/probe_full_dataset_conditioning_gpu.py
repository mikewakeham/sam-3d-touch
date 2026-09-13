"""Check surface utility in existing dataset-wide checkpoints; no training."""
import argparse
import copy
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
from tiny_fit_gpu import parameter_digest
from train import TouchTrainingModel, amp, build_optimizer, build_stage1_pipeline, load_trainable_state_dict, prepare_batch
from analyze_full_dataset_conditioning import sha, validate_checkpoint_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key', choices=('image', 'surface'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.partial.json').exists():
        raise FileExistsError(args.output)
    run_path = HERE/'full_dataset_probe_references'/f'{args.model_key}.json'
    expected_path = run_path.with_name(f'{args.model_key}_expected.json')
    run = json.loads(run_path.read_text()); config = run['config']
    expected = json.loads(expected_path.read_text())
    checkpoint_path = args.checkpoint or Path(config['output_dir'])/'best.pt'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'{checkpoint_path}; supply --checkpoint for the existing {expected["run_id"]} best checkpoint. Do not retrain.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    checkpoint_meta = {k: checkpoint[k] for k in ('epoch', 'step', 'best_loss', 'mode', 'cross_attention_scope', 'touch_config')}
    checkpoint_meta['conditioning_config'] = checkpoint.get('conditioning_config', {'no_pointmap': False, 'oracle_point_frame': False})
    validate_checkpoint_metadata(checkpoint_meta, config, expected)
    reference_path = HERE/'tiny_fit_returned_46083371/oracle/results.json'
    reference = json.loads(reference_path.read_text())
    for name, digest in reference['source_sha256'].items():
        assert sha(REPO/name) == digest, name
    pipeline_path = Path(config['pipeline_config'])
    assert pipeline_path.read_text() == reference['pipeline_yaml']
    cfg = OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent/cfg.ss_generator_config_path).read_text() == reference['generator_yaml']
    data_path = Path(reference['settings']['data_config'])
    assert data_path.read_text() == reference['data_yaml']
    frame_path = HERE/'object_transfer_returned_manual'/('camera' if args.model_key == 'surface' else 'image')/'results.json'
    frame = json.loads(frame_path.read_text())
    torch.cuda.set_device(args.device_index); device = torch.device('cuda', args.device_index)
    random.seed(37); np.random.seed(37); torch.manual_seed(37)
    pipeline = build_stage1_pipeline(pipeline_path, device)
    encoder = None
    if args.model_key == 'surface':
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
        encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels, trainable=False, use_position=False).to(device).eval()
        assert encoder.get_config() == checkpoint_meta['touch_config']
    model = TouchTrainingModel(pipeline.ss_generator, encoder, False, False)
    optimizer, _ = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope='full'))
    del optimizer
    loaded_names = sorted(checkpoint['model'])
    for name, value in checkpoint['model'].items():
        assert dict(model.named_parameters())[name].shape == value.shape
    load_trainable_state_dict(model, checkpoint['model'])
    del checkpoint
    model_hash = parameter_digest(model.named_parameters())
    model.requires_grad_(False)
    gen = pipeline.ss_generator
    gen.reverse_fn.training = True
    assert gen.reverse_fn.p_unconditional == 0 and gen.self_consistency_prob == 0 and gen.fm_eps_max == 0
    assert gen.loss_weights['shape'] == 1 and all(v == 0 for k, v in gen.loss_weights.items() if k != 'shape')
    data = load_data_config(data_path); assert data['touch']['source'] == 'full_surface'
    datasets, lookup, split_of = [], {}, {}
    for split in ('train', 'val'):
        dc = copy.deepcopy(data); dc['dataset']['split'] = split
        ds = TouchDataset(dc, include_touch=encoder is not None, oracle_point_frame=False)
        datasets.append(ds)
        for record in ds.records:
            assert record['sample_id'] not in lookup
            lookup[record['sample_id']] = record
            assert split_of.setdefault(record['object_id'], split) == split
    ds = datasets[0]
    report = dict(settings={'model_key': args.model_key, 'seed': 37, 'precision': 'bf16', 'draws': 8, 'bank_base': 900000},
                  run_reference_sha256=sha(run_path), expected_reference_sha256=sha(expected_path),
                  frame_reference_sha256=sha(frame_path), checkpoint=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
                  checkpoint_metadata=checkpoint_meta, loaded_parameter_names=loaded_names,
                  current_dataset_object_counts={split: sum(s == split for s in split_of.values()) for split in ('train', 'val')},
                  model_sha256=model_hash, input_batches=[], rows=[], context_checks={}, complete=False,
                  source_sha256={**reference['source_sha256'], **{str(p.relative_to(REPO)): sha(p) for p in
                      (Path(__file__), HERE/'analyze_full_dataset_conditioning.py', HERE/'tiny_fit_gpu.py')}},
                  runtime={'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device)},
                  scope='Existing dataset-wide image/surface best checkpoints. 16 current dataset-train and '
                        '16 dataset-validation identities, two views each. No training or decoding. Metadata '
                        'matches exported run config/best step/loss; historical checkpoint lacks source/data '
                        'fingerprints, so exact historical training-data replay cannot be certified.')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.with_suffix('.partial.json').write_text(json.dumps(report, indent=2)+'\n')

    for index, previous in enumerate(frame['input_batches'][16:32]):
        split = 'train' if index < 8 else 'val'
        records = [lookup[s] for s in previous['sample_ids']]
        assert [r['object_id'] for r in records] == previous['object_ids']
        assert all(split_of[r['object_id']] == split for r in records)
        ds.records = records
        batch = collate_touch_batch([ds[i] for i in range(4)])
        targets, ca, kw, points, mask = prepare_batch(pipeline, batch, device, 'bf16', encoder is not None, False, False)
        assert len(ca) == 1 and not kw
        features = tokens = None
        if encoder is not None:
            assert mask.all() and points.shape[1] == encoder.num_points == 8192
            with torch.no_grad(), amp(device, 'bf16'):
                pp, mm, _, _ = encoder.prepare_points(points, mask)
                features = encoder.encoder.encode(pp, mm)['x']
                tokens = encoder.output_projection(features)+encoder.touch_embedding
        actual = dict(image_sha256=tensor_digest(batch['image']), pointmap_sha256=tensor_digest(batch['pointmap']),
                      visual_sha256=tensor_digest(ca[0]), target_sha256=tensor_digest(targets['shape']),
                      object_target_sha256=[tensor_digest(t) for t in targets['shape']])
        if features is not None:
            actual['features_sha256'] = tensor_digest(features)
        assert all(previous[k] == v for k, v in actual.items()), 'Data/feature drift from established diagnostic inputs'
        report['input_batches'].append(dict(batch=index, dataset_split=split, sample_ids=previous['sample_ids'],
                                            object_ids=previous['object_ids'], **actual))
        conditions = ('removed',) if tokens is None else ('correct', 'wrong_1', 'wrong_2', 'wrong_3', 'removed')
        for condition in conditions:
            touch = None if condition == 'removed' else (tokens.roll(int(condition[-1]), 0) if condition.startswith('wrong_') else tokens)
            cond = {} if touch is None else {'touch_tokens': touch}
            matrix, scalar = [], []
            for draw in range(8):
                draw_seed = 900000+37+100*(index % 4)+400*(index // 8)+draw
                torch.manual_seed(draw_seed); random.seed(draw_seed)
                captured = []
                original = gen.loss_fn
                functions = dict(original) if isinstance(original, dict) else {k: original for k in gen.loss_weights}
                shape_fn = functions['shape']
                def capture(pred, target):
                    native = shape_fn(pred, target)
                    per_object = F.mse_loss(pred.float(), target.float(), reduction='none').flatten(1).mean(1)
                    torch.testing.assert_close(per_object.mean(), native.float(), rtol=1e-5, atol=1e-7)
                    captured.append(per_object.detach())
                    return native
                functions['shape'] = capture
                handle = None
                if condition not in report['context_checks']:
                    def check_context(module, inputs):
                        expected_context = ca[0] if touch is None else torch.cat((ca[0], touch.to(ca[0])), dim=1)
                        torch.testing.assert_close(inputs[0], expected_context.to(inputs[0]), rtol=0, atol=0)
                        report['context_checks'][condition] = inputs[0].shape[1]
                    handle = pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check_context)
                try:
                    with torch.no_grad(), amp(device, 'bf16'), patch.object(gen, 'loss_fn', functions):
                        loss, _ = gen.loss(targets, ca[0], **cond)
                finally:
                    if handle is not None:
                        handle.remove()
                assert len(captured) == 1 and captured[0].shape == (4,)
                torch.testing.assert_close(captured[0].mean(), loss.float(), rtol=1e-5, atol=1e-7)
                assert torch.isfinite(loss) and torch.isfinite(captured[0]).all()
                scalar.append(float(loss)); matrix.append(captured[0].cpu().tolist())
            report['rows'].append(dict(batch=index, condition=condition, scalar_losses=scalar, per_object_losses=matrix))
        save(); print(args.model_key, 'completed batch', index+1, '/16', flush=True)
    assert parameter_digest(model.named_parameters()) == model_hash
    report['parameters_unchanged'] = True
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
