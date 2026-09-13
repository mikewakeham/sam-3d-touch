"""Sample the completed F18 checkpoints, replaying their exact selected observations."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import yaml
from dataloader import TouchDataset, collate_touch_batch
from train import amp, build_stage1_pipeline, prepare_batch
from evaluate import restore_run
from checkpoint_probe_gpu import sha, write
from checkpoint_probe_core import condition_tokens, tensor_sha
from checkpoint_rollout_core import sample_from_noise, support_metrics


def decode_support(decoder, shape):
    x = shape if decoder.reshape_input_to_cube else decoder.flat_to_cube(shape)
    logits = decoder(x)
    assert torch.isfinite(logits).all(), 'Nonfinite decoder output'
    return (logits[0, 0] > 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-root', type=Path, required=True)
    parser.add_argument('--arm', choices=('oracle', 'constant'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-index', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output}: use a new output directory')
    reference_path = args.probe_root / args.arm / 'results.json'
    ref = json.loads(reference_path.read_text())
    assert ref['complete'] and ref['adapted_parameters_unchanged']
    assert ref['settings']['arm'] == args.arm and ref['checkpoint_metadata']['step'] == 14660
    for name, digest in ref['source_sha256'].items():
        assert sha(REPO / name) == digest, f'Source changed since F18: {name}'
    run_dir = Path(ref['settings']['run_dir'])
    checkpoint_path = run_dir / 'last.pt'
    assert sha(checkpoint_path) == ref['checkpoint_sha256'], 'Checkpoint changed since F18'
    assert sha(run_dir / 'config.yaml') == ref['config_sha256']
    config = yaml.safe_load((run_dir / 'config.yaml').read_text())
    pipeline_path = Path(config['arguments']['pipeline_config'])
    assert sha(pipeline_path) == ref['pipeline_sha256']
    assert config['data'] == ref['data_config']
    torch.cuda.set_device(args.device_index)
    device = torch.device('cuda', args.device_index)
    random.seed(29); np.random.seed(29); torch.manual_seed(29)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert {k: checkpoint[k] for k in ref['checkpoint_metadata']} == ref['checkpoint_metadata']
    pipeline = build_stage1_pipeline(pipeline_path, device)
    model = restore_run(pipeline, checkpoint, device)
    expected_weights = {name: tensor_sha(value) for name, value in checkpoint['model'].items()}
    parameters = dict(model.named_parameters())
    assert all(tensor_sha(parameters[k]) == v for k, v in expected_weights.items())
    del checkpoint
    model.requires_grad_(False)
    model.touch_encoder.eval()
    constant_hash = tensor_sha(model.constant_touch_features) if args.arm == 'constant' else None
    assert constant_hash == ref['constant_bank_sha256']
    cfg = yaml.safe_load(pipeline_path.read_text())
    decoder = pipeline.init_ss_decoder(cfg['ss_decoder_config_path'], cfg['ss_decoder_ckpt_path']).eval().requires_grad_(False)
    gen = pipeline.ss_generator
    gen.no_shortcut = True
    gen.inference_steps = 25
    gen.rescale_t = float(cfg.get('ss_rescale_t', 3))
    gen.reverse_fn.interval = list(cfg.get('ss_cfg_interval', [0, 500]))
    gen.reverse_fn.strength = 0
    gen.reverse_fn.training = False
    # This is the conditional-only sampler used in the successful earlier probes.
    # It is not an assertion about the optimal guidance or integration settings.
    t_seq, d = gen._prepare_t_and_d()
    assert d == 0 and gen.reverse_fn.p_unconditional == 0
    datasets, lookup = {}, {}
    for split in ('train', 'val'):
        data = copy.deepcopy(config['data']); data['dataset']['split'] = split
        datasets[split] = TouchDataset(data, include_touch=True, oracle_point_frame=True)
        lookup[split] = {r['sample_id']: r for r in datasets[split].records}
    dataset_hashes = {key: sha(datasets['train'].resolve_path(config['data']['dataset'][key]))
                      for key in ('manifest', 'split_file')}
    assert dataset_hashes == ref['dataset_sha256']
    decoder_paths = [Path(cfg[key]) for key in ('ss_decoder_config_path', 'ss_decoder_ckpt_path')]
    decoder_paths = [p if p.is_absolute() else pipeline_path.parent / p for p in decoder_paths]
    report = {'complete': False, 'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'checkpoint_metadata': ref['checkpoint_metadata'], 'checkpoint_sha256': ref['checkpoint_sha256'],
              'reference_sha256': sha(reference_path), 'dataset_sha256': dataset_hashes,
              'pipeline_sha256': ref['pipeline_sha256'], 'source_sha256': ref['source_sha256'],
              'additional_source_sha256': {str(p.relative_to(REPO)): sha(p) for p in
                  (Path(__file__), HERE / 'checkpoint_rollout_core.py')},
              'decoder_sha256': {str(p): sha(p) for p in decoder_paths},
              'sampler': {'steps': 25, 'cfg_strength': 0, 'no_shortcut': True, 'rescale_t': gen.rescale_t,
                          'reversed_timestamp': gen.reversed_timestamp, 'time_sequence': t_seq.tolist(),
                          'solver': gen._solver_method, 'precision': 'bf16', 'draws': 2},
              'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device)},
              'inputs': [], 'banks': [], 'samples': [], 'files_sha256': {},
              'scope': 'Exactly the F18 observations and checkpoints. Pure-noise Stage1 generation. '
                       'Correct or one fixed cyclic wrong surface; present/zero visuals. '
                       'Target latents only decode into reference support; no target values are passed to sampling. '
                       'Raw metrics are pose sensitive; saved supports allow separate rigid analysis. No Stage2 or updates.'}
    args.output.mkdir(parents=True)
    write(args.output / 'results.partial.json', report)
    with torch.no_grad():
        for previous in ref['inputs']:
            split, gi = previous['split'], previous['group']
            records = [lookup[split][sid] for sid in previous['sample_ids']]
            assert [r['object_id'] for r in records] == previous['object_ids']
            ds = copy.copy(datasets[split]); ds.records = records
            batch = collate_touch_batch([ds[i] for i in range(4)])
            targets, ca, kw, points, mask = prepare_batch(pipeline, batch, device, 'bf16', True, False, True)
            assert len(ca) == 1 and not kw
            with amp(device, 'bf16'):
                tokens = model.get_touch_tokens(points, mask)
            actual = {'split': split, 'group': gi, 'sample_ids': batch['sample_id'], 'object_ids': previous['object_ids'],
                      'image_sha256': tensor_sha(batch['image']), 'pointmap_sha256': tensor_sha(batch['pointmap']),
                      'visual_sha256': tensor_sha(ca[0]), 'points_sha256': tensor_sha(points), 'mask_sha256': tensor_sha(mask),
                      'tokens_sha256': tensor_sha(tokens), 'target_sha256': {k: tensor_sha(v) for k, v in targets.items()}}
            assert actual == previous, f'Observation/token replay failed: {split}/{gi}'
            report['inputs'].append(actual)
            with amp(device, 'bf16'):
                target_support = np.stack([decode_support(decoder, t[None]) for t in targets['shape']])
            assert all(np.any(s) for s in target_support)
            actual['target_support_sha256'] = tensor_sha(torch.from_numpy(target_support))
            target_file = f'{split}_g{gi}_targets.npz'
            np.savez_compressed(args.output / target_file, target_occupancy=target_support,
                                target_shape=targets['shape'].float().cpu().numpy(), sample_ids=np.array(batch['sample_id']))
            report['files_sha256'][target_file] = sha(args.output / target_file)
            banks = []
            for draw in range(2):
                seed = 2400029 + (0 if split == 'train' else 10000) + gi * 100 + draw
                with torch.random.fork_rng(devices=[device.index]):
                    torch.manual_seed(seed)
                    # Target values are deliberately unavailable to the noise sampler.
                    noise = gen._generate_noise({k: tuple(v.shape) for k, v in targets.items()}, device)
                banks.append(noise)
                report['banks'].append({'split': split, 'group': gi, 'draw': draw, 'seed': seed,
                    'noise_sha256': {k: tensor_sha(v) for k, v in noise.items()}})
            for visual_state in ('present', 'zero'):
                visual = ca[0] if visual_state == 'present' else torch.zeros_like(ca[0])
                for shift in ((0,) if args.arm == 'constant' else (0, 1)):
                    conditioned = condition_tokens(tokens, shift)
                    expected_context = torch.cat((visual, conditioned.to(visual)), 1)
                    seen = []
                    def context_check(module, inputs):
                        torch.testing.assert_close(inputs[0], expected_context.to(inputs[0]), rtol=0, atol=0)
                        seen.append(True)
                    hook = pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(context_check)
                    try:
                        for draw, noise in enumerate(banks):
                            with amp(device, 'bf16'):
                                predicted = sample_from_noise(gen, noise, visual, conditioned)
                                supports = np.stack([decode_support(decoder, p[None]) for p in predicted])
                            filename = f'{split}_g{gi}_{visual_state}_s{shift}_d{draw}.npz'
                            np.savez_compressed(args.output / filename, predicted_occupancy=supports,
                                                predicted_shape=predicted.float().cpu().numpy(), sample_ids=np.array(batch['sample_id']))
                            report['files_sha256'][filename] = sha(args.output / filename)
                            for index, record in enumerate(records):
                                report['samples'].append({'split': split, 'group': gi, 'visual': visual_state,
                                    'surface_shift': shift, 'draw': draw, 'sample_id': record['sample_id'],
                                    'object_id': record['object_id'], 'prediction_file': filename, 'array_index': index,
                                    'target_file': target_file, **support_metrics(supports[index], target_support[index])})
                        assert seen, 'Shape cross-attention context was never observed'
                    finally:
                        hook.remove()
            write(args.output / 'results.partial.json', report)
            print(args.arm, split, f'group {gi + 1}/8 done', flush=True)
    assert all(tensor_sha(parameters[k]) == v for k, v in expected_weights.items())
    report['adapted_parameters_unchanged'] = True
    report['complete'] = True
    write(args.output / 'results.json', report)
    print('Complete:', args.output / 'results.json', flush=True)


if __name__ == '__main__':
    main()
