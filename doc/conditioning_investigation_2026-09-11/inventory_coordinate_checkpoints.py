"""Read existing run configs/checkpoint metadata on CPU; never execute a model."""
import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_RUNS = {
    'camera': 'outputs/stage1_full_surface_full_cross_attention',
    'oracle': 'outputs/stage1_full_surface_oracle_full_cross_attention',
    'image': 'outputs/stage1_image_full_cross_attention',
    'camera_no_pointmap': 'outputs/stage1_full_surface_no_pointmap_full_cross_attention',
    'oracle_no_pointmap': 'outputs/stage1_full_surface_oracle_no_pointmap_full_cross_attention',
}
IGNORED_ARGUMENTS = {
    'output_dir', 'wandb_id', 'workers', 'val_workers', 'local_rank',
    'log_every', 'device',
}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_checkpoint(path, loader=None):
    if not path.is_file():
        return {'status': 'missing', 'path': str(path)}
    if loader is None:
        import torch
        loader = lambda p: torch.load(p, map_location='cpu', weights_only=True, mmap=True)
    # mmap reads tensor metadata without copying all model/optimizer storage.
    # Do not fall back to arbitrary-pickle loading if this fails.
    checkpoint = loader(path)
    metadata = {k: checkpoint.get(k) for k in (
        'epoch', 'step', 'best_loss', 'mode', 'cross_attention_scope',
        'touch_config', 'conditioning_config',
    )}
    model = checkpoint['model']
    schema = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in sorted(model.items())}
    groups = []
    for group in checkpoint.get('optimizer', {}).get('param_groups', []):
        groups.append({**{k: v for k, v in group.items() if k != 'params'},
                       'parameter_tensors': len(group.get('params', []))})
    stat = path.stat()
    return {
        'status': 'read', 'path': str(path.resolve()), 'size_bytes': stat.st_size,
        'mtime_ns': stat.st_mtime_ns, 'metadata': metadata,
        'model_parameter_tensors': len(schema),
        'model_parameter_count': sum(v.numel() for v in model.values()),
        'parameter_schema_sha256': digest_bytes(json.dumps(schema, sort_keys=True).encode()),
        'optimizer_groups': groups,
        'note': 'Schema hash is not a weight-content hash. Missing metadata remains unknown.',
    }


def file_record(path):
    if not path.is_file():
        return {'path': str(path), 'status': 'missing'}
    content = path.read_bytes()
    return {'path': str(path.resolve()), 'status': 'read',
            'sha256': digest_bytes(content), 'text': content.decode()}


def compare_configs(a, b):
    """Report differences, not a causal-equivalence verdict."""
    if a is None or b is None:
        return {'status': 'config_missing'}
    aa, bb = a.get('arguments', {}), b.get('arguments', {})
    # Keep an absent argument distinct from an explicitly saved false/default.
    missing = {'not_recorded': True}
    differences = {
        k: [aa.get(k, missing), bb.get(k, missing)]
        for k in sorted(set(aa) | set(bb))
        if k not in IGNORED_ARGUMENTS and aa.get(k, missing) != bb.get(k, missing)
    }
    other = {k: [a.get(k, missing), b.get(k, missing)]
             for k in sorted((set(a) | set(b)) - {'arguments'})
             if a.get(k, missing) != b.get(k, missing)}
    return {'status': 'compared', 'argument_differences': differences,
            'other_config_differences': other, 'ignored_arguments': sorted(IGNORED_ARGUMENTS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-dir', action='append', default=[], metavar='KEY=PATH',
                        help='Override a known run directory if it has moved.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runs = dict(DEFAULT_RUNS)
    for item in args.run_dir:
        key, separator, value = item.partition('=')
        if not separator or key not in runs or not value:
            parser.error('--run-dir must be a known KEY=PATH')
        runs[key] = value
    import yaml
    report = {'scope': 'CPU metadata inventory only. No model construction, tensor inference, '
                       'training, CUDA, or checkpoint writes. Current config/data fingerprints '
                       'do not certify historical training inputs or pretrained base weights.',
              'script_sha256': digest_bytes(Path(__file__).read_bytes()), 'runs': {}}
    configs = {}
    for key, directory in runs.items():
        root = Path(directory)
        entry = {'directory': str(root), 'config': file_record(root/'config.yaml'), 'checkpoints': {}}
        try:
            config = yaml.safe_load(entry['config']['text']) if entry['config']['status'] == 'read' else None
            if config is not None and not isinstance(config, dict):
                raise ValueError('Run config must be a mapping')
            configs[key] = config
            entry['parsed_config'] = config
        except Exception as error:
            entry['config_error'] = f'{type(error).__name__}: {error}'
            configs[key] = None
        for name in ('best.pt', 'last.pt'):
            try:
                entry['checkpoints'][name] = read_checkpoint(root/name)
            except Exception as error:
                entry['checkpoints'][name] = {'status': 'error', 'path': str(root/name),
                                             'error': f'{type(error).__name__}: {error}'}
        report['runs'][key] = entry
        print(key, {k: v['status'] for k, v in entry['checkpoints'].items()}, flush=True)
    report['camera_oracle_config_comparison'] = compare_configs(configs['camera'], configs['oracle'])
    report['no_pointmap_config_comparison'] = compare_configs(configs['camera_no_pointmap'], configs['oracle_no_pointmap'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, default=str, allow_nan=False)
        stream.write('\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
