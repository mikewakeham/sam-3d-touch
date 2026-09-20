"""Select small, fixed experiments from the existing object split; copy no assets."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def experiment_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError(f"Experiment outputs must be inside {HERE}")
    return path


def prepare(config_path, output_dir, train_objects=256, val_objects=64, overfit_objects=16, seed=29):
    output_dir = experiment_path(output_dir)
    config = yaml.safe_load(Path(config_path).read_text())
    root = Path(config['dataset']['root']).resolve()
    manifest = root / config['dataset']['manifest']
    splits = json.loads((root / config['dataset']['split_file']).read_text())
    assigned = [obj for group in splits.values() for obj in group]
    if len(set(assigned)) != len(assigned):
        raise ValueError('Source object splits overlap or contain duplicates')
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len({row['sample_id'] for row in records}) != len(records):
        raise ValueError('Duplicate sample IDs in source manifest')
    grouped = {}
    for row in records:
        grouped.setdefault(row['object_id'], []).append(row)
    if min(train_objects, val_objects, overfit_objects) < 1 or overfit_objects > train_objects:
        raise ValueError('Counts must be positive; overfit count cannot exceed training count')

    # Same seeded object ranking as data_generation/general/create_splits.py.
    selected = {}
    for split, count in [('train', train_objects), ('val', val_objects)]:
        candidates = [obj for obj in splits[split] if len(grouped.get(obj, [])) == 8]
        candidates.sort(key=lambda obj: (hashlib.sha256(f'{seed}:{obj}'.encode()).hexdigest(), obj))
        if len(candidates) < count:
            raise ValueError(f'Need {count} {split} objects with eight views; found {len(candidates)}')
        selected[split] = candidates[:count]

    files = {}
    for stage in ['small', 'overfit']:
        membership = selected if stage == 'small' else {
            'train': selected['train'][:overfit_objects],
            # Deliberately measure memorization on exactly the training examples.
            'val': selected['train'][:overfit_objects],
        }
        rows = []
        for obj in sorted(set(membership['train'] + membership['val'])):
            views = sorted(grouped[obj], key=lambda row: row['sample_id'])
            rows.extend(views if stage == 'small' else views[:1])
        files[f'{stage}.jsonl'] = ''.join(json.dumps(row) + '\n' for row in rows)
        files[f'{stage}_splits.json'] = json.dumps({**membership, 'test': []}, indent=2) + '\n'
        data = {
            'seed': seed,
            'dataset': {'root': str(root), 'manifest': str(output_dir / f'{stage}.jsonl'),
                        'split_file': str(output_dir / f'{stage}_splits.json'), 'split': 'train'},
            'touch': {'source': 'full_surface'},
        }
        files[f'{stage}.yaml'] = yaml.safe_dump(data, sort_keys=False)
    # Re-running preparation must not silently change an ongoing comparison.
    for name, content in files.items():
        path = output_dir / name
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f'{path} differs; use another experiment data directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = output_dir / name
        if not path.exists():
            path.write_text(content)
    (HERE / 'logs').mkdir(exist_ok=True)
    print(f'Small: {train_objects} train / {val_objects} held-out objects, eight views each')
    print(f'Overfit: {overfit_objects} objects, one view; val intentionally equals train')
    print(f'Configs: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', type=Path, default=REPO / 'configs/data_zeroverse_5000_8views_full_surface.yaml')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'data')
    parser.add_argument('--train-objects', type=int, default=256)
    parser.add_argument('--val-objects', type=int, default=64)
    parser.add_argument('--overfit-objects', type=int, default=16)
    parser.add_argument('--seed', type=int, default=29)
    args = parser.parse_args()
    prepare(args.data_config, args.output_dir, args.train_objects, args.val_objects, args.overfit_objects, args.seed)
