"""No-update Stage-1 probe of completed production oracle/constant checkpoints."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (_source_path(p, 'train.py')).is_file() and (p / 'sam3d_objects').is_dir())
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import yaml
from dataloader import TouchDataset, collate_touch_batch
from train import amp, build_stage1_pipeline, prepare_batch
from evaluate import restore_run
from experiments.coordinate_system.scripts.full_checkpoints.protocol import check_splits, evaluation_groups
from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_probe_core import condition_tokens, paired_loss, tensor_sha

ARMS = {
    'oracle': 'stage1_full_surface_oracle_dropout',
    'constant': 'stage1_full_surface_oracle_constant_dropout',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def make_bank(gen, targets, seed):
    """Four native times/draws and two draws at each of three fixed times."""
    device = targets['shape'].device
    bank = []
    original_generator = gen.random_generator
    try:
        with torch.random.fork_rng(devices=[device.index]):
            gen.random_generator = torch.Generator().manual_seed(seed)
            for draw in range(4):
                torch.manual_seed(seed + draw)
                times = gen._generate_t(targets)
                noise = gen._generate_x0(targets)
                bank.append(('native', draw, times, noise))
            for draw in range(2):
                torch.manual_seed(seed + 100 + draw)
                noise = gen._generate_x0(targets)
                for time in (0.05, 0.5, 0.95):
                    times = torch.full((len(targets['shape']),), time, device=device)
                    bank.append((f't_{time:g}', draw, times, noise))
    finally:
        gen.random_generator = original_generator
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--objects-per-split', type=int, default=16)
    args = parser.parse_args()
    if args.objects_per_split < 4 or args.objects_per_split % 4:
        raise ValueError('Object count must be a positive multiple of four')
    if args.output.exists():
        raise FileExistsError(f'{args.output}: use a new output directory')
    run_dir = args.run_dir or Path('outputs/conditioning_investigation') / ARMS[args.arm]
    checkpoint_path = run_dir / 'last.pt'
    run_config_path = run_dir / 'config.yaml'
    config = yaml.safe_load(run_config_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    expected_training = {'visual_dropout': 0.5, 'constant_touch': args.arm == 'constant'}
    expected_conditioning = {'no_pointmap': False, 'oracle_point_frame': True}
    assert checkpoint['step'] == 14660 and checkpoint['epoch'] == 20
    assert checkpoint['training_config'] == expected_training
    assert checkpoint['conditioning_config'] == expected_conditioning
    assert checkpoint['cross_attention_scope'] == 'full' and checkpoint['mode'] == 'image_touch'
    assert checkpoint['touch_config'] == config['touch_config']
    assert not checkpoint['touch_config']['trainable'] and not checkpoint['touch_config']['use_position']
    assert not checkpoint['touch_config'].get('use_learn', False)
    assert config['data']['touch']['source'] == 'full_surface'
    for key, value in {**expected_training, **expected_conditioning}.items():
        assert config['arguments'][key] == value, key
    assert config['arguments']['joint_pointmap'] is False
    pipeline_path = Path(config['arguments']['pipeline_config'])
    metadata_keys = ('step', 'epoch', 'best_loss', 'mode', 'training_config',
                     'conditioning_config', 'touch_config', 'cross_attention_scope')
    report = {
        'complete': False, 'settings': {**vars(args), 'run_dir': str(run_dir),
            'output': str(args.output), 'seed': 29, 'precision': 'bf16'},
        'checkpoint_metadata': {key: checkpoint[key] for key in metadata_keys},
        'checkpoint_sha256': sha(checkpoint_path), 'config_sha256': sha(run_config_path),
        'data_config': config['data'], 'inputs': [], 'banks': [], 'rows': [],
        'scope': '16 train and 16 identity-disjoint validation objects by default, two views each. '
                 'Native and fixed-time shape velocity MSE; no updates, decoding, or Stage2. '
                 'Wrong surfaces are cyclic distractors from three other identities in the batch.',
    }
    source_paths = [Path(__file__), _source_path(HERE, 'checkpoint_probe_core.py'), _source_path(HERE, 'protocol.py'),
                    _source_path(REPO, 'train.py'), _source_path(REPO, 'evaluate.py'), _source_path(REPO, 'dataloader.py'),
                    _source_path(REPO, 'sam3d_objects/model/backbone/dit/embedder/touch.py'),
                    _source_path(REPO, 'sam3d_objects/model/backbone/generator/shortcut/model.py'),
                    _source_path(REPO, 'sam3d_objects/model/backbone/generator/flow_matching/model.py')]
    report['source_sha256'] = {str(p.relative_to(REPO)): sha(p) for p in source_paths}
    report['pipeline_sha256'] = sha(pipeline_path)
    torch.cuda.set_device(args.device_index)
    device = torch.device('cuda', args.device_index)
    random.seed(29); np.random.seed(29); torch.manual_seed(29)
    pipeline = build_stage1_pipeline(pipeline_path, device)
    model = restore_run(pipeline, checkpoint, device)
    loaded_names = sorted(checkpoint['model'])
    # Store compact digests, release the checkpoint's Adam state before probing.
    expected_weights = {name: tensor_sha(value) for name, value in checkpoint['model'].items()}
    parameters = dict(model.named_parameters())
    assert all(tensor_sha(parameters[name]) == digest for name, digest in expected_weights.items())
    del checkpoint
    model.requires_grad_(False)
    model.touch_encoder.eval()
    gen = pipeline.ss_generator
    gen.reverse_fn.training = True
    assert gen.reverse_fn.p_unconditional == gen.self_consistency_prob == gen.fm_eps_max == 0
    report['runtime'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                         'gpu': torch.cuda.get_device_name(device)}
    report['generator'] = {'sigma_min': gen.sigma_min, 'time_scale': gen.time_scale,
                           'time_sampler': repr(gen.training_time_sampler_fn)}
    report['restored_parameter_names'] = loaded_names
    report['constant_bank_sha256'] = tensor_sha(model.constant_touch_features) if args.arm == 'constant' else None
    report['constant_sample_id'] = model.constant_touch_sample_id
    datasets = {}
    for split in ('train', 'val'):
        data = copy.deepcopy(config['data']); data['dataset']['split'] = split
        datasets[split] = TouchDataset(data, include_touch=True, oracle_point_frame=True)
    report['split_counts'] = check_splits(datasets['train'].records, datasets['val'].records)
    report['dataset_sha256'] = {
        key: sha(datasets['train'].resolve_path(config['data']['dataset'][key]))
        for key in ('manifest', 'split_file')}
    selections = {split: evaluation_groups(ds.records, 29, args.objects_per_split, 2)
                  for split, ds in datasets.items()}
    assert all(sum(len(group) for group in groups) == args.objects_per_split * 2
               for groups in selections.values())
    args.output.mkdir(parents=True)
    write(args.output / 'results.partial.json', report)
    with torch.no_grad():
        for si, (split, groups) in enumerate(selections.items()):
            ds = copy.copy(datasets[split])
            for gi, records in enumerate(groups):
                ds.records = records
                batch = collate_touch_batch([ds[i] for i in range(4)])
                targets, ca, kw, points, mask = prepare_batch(
                    pipeline, batch, device, 'bf16', True, False, True)
                assert len(ca) == 1 and not kw
                assert len({r['object_id'] for r in records}) == 4
                with amp(device, 'bf16'):
                    tokens = model.get_touch_tokens(points, mask)
                assert torch.isfinite(tokens).all()
                if args.arm == 'constant':
                    torch.testing.assert_close(tokens, tokens[:1].expand_as(tokens), rtol=0, atol=0)
                report['inputs'].append({
                    'split': split, 'group': gi, 'sample_ids': batch['sample_id'],
                    'object_ids': [r['object_id'] for r in records],
                    'image_sha256': tensor_sha(batch['image']),
                    'pointmap_sha256': tensor_sha(batch['pointmap']),
                    'visual_sha256': tensor_sha(ca[0]), 'points_sha256': tensor_sha(points),
                    'mask_sha256': tensor_sha(mask), 'tokens_sha256': tensor_sha(tokens),
                    'target_sha256': {name: tensor_sha(t) for name, t in targets.items()}})
                bank = make_bank(gen, targets, 2300000 + si * 10000 + gi * 1000 + 29)
                for time_kind, draw, times, noise in bank:
                    report['banks'].append({'split': split, 'group': gi, 'time_kind': time_kind,
                        'draw': draw, 'times': times.cpu().tolist(), 'time_sha256': tensor_sha(times),
                        'noise_sha256': {name: tensor_sha(t) for name, t in noise.items()}})
                for visual_state in ('present', 'zero'):
                    visual = ca[0] if visual_state == 'present' else torch.zeros_like(ca[0])
                    for shift in (range(1) if args.arm == 'constant' else range(4)):
                        conditioned = condition_tokens(tokens, shift)
                        expected_context = torch.cat((visual, conditioned.to(visual)), 1)
                        seen = []
                        def context_check(module, inputs):
                            torch.testing.assert_close(inputs[0], expected_context.to(inputs[0]), rtol=0, atol=0)
                            seen.append(True)
                        handle = pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(context_check)
                        try:
                            for time_kind, draw, times, noise in bank:
                                with amp(device, 'bf16'):
                                    losses = paired_loss(gen, targets, visual, conditioned, times, noise)
                                report['rows'].append({'split': split, 'group': gi,
                                    'visual': visual_state, 'surface_shift': shift,
                                    'time_kind': time_kind, 'draw': draw, 'losses': losses})
                            assert len(seen) == len(bank)
                        finally:
                            handle.remove()
                write(args.output / 'results.partial.json', report)
                print(args.arm, split, f'batch {gi + 1}/{len(groups)} complete', flush=True)
    assert all(tensor_sha(parameters[name]) == digest for name, digest in expected_weights.items())
    report['adapted_parameters_unchanged'] = True
    report['complete'] = True
    write(args.output / 'results.json', report)
    print('Complete:', args.output / 'results.json', flush=True)


if __name__ == '__main__':
    main()
