"""Descriptive full-run audit; no inference or training, standard library only."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

RUNS = {
    'oracle': 'cps50h2l', 'constant': 'hwz5ifqt', 'camera': 'mpd9bbwh',
    'image_pointmap': 'xun3al7m', 'image_only': 'fl7b2znc',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'runs': {}, 'source_sha256': {}, 'scope':
              'Descriptive exported aggregate losses, one run per arm. No per-object '
              'inference, significance test, equivalence test, or training replay.'}
    histories = {}
    for key, run_id in RUNS.items():
        folder = args.exports / run_id
        run = json.loads((folder / 'run.json').read_text())
        rows = list(csv.DictReader((folder / 'history.csv').open()))
        report['source_sha256'][run_id] = {
            name: sha(folder / name) for name in ('run.json', 'history.csv')}
        vals = [(int(float(r['global_step'])), float(r['loss/val']))
                for r in rows if r.get('loss/val')]
        train = [r for r in rows if r.get('loss/train')]
        assert run['state'] == 'finished' and len(vals) == 20
        assert [s for s, _ in vals] == list(range(733, 14661, 733))
        intervals, previous = [], 0
        for r in train:
            step = int(float(r['global_step']))
            assert step > previous
            loss = float(r['loss/train'])
            assert math.isfinite(loss)
            intervals.append((step, step - previous, loss,
                              float(r.get('conditioning/visual_dropout_fraction') or 0)))
            previous = step
        assert previous == 14660
        def weighted(column, low=0, high=14660):
            selected = [r for r in intervals if low < r[0] <= high]
            return sum(r[1] * r[column] for r in selected) / sum(r[1] for r in selected)
        # Weight by update counts, not CSV row counts: endpoint intervals are shorter.
        # Exact sample weights cannot be recovered from these CSVs alone.
        gradients = {}
        for name in train[0]:
            if not name.startswith('gradients/'):
                continue
            values = [float(r[name]) for r in train if r.get(name)]
            assert values and all(math.isfinite(v) for v in values)
            gradients[name] = {'min': min(values), 'max': max(values),
                               'nonzero_records': sum(v != 0 for v in values),
                               'records': len(values)}
        best = min(vals, key=lambda row: row[1])
        report['runs'][key] = {
            'run_id': run_id, 'name': run['name'], 'state': run['state'],
            'config': run['config'], 'final_step': previous,
            'validation': [{'epoch': s // 733, 'step': s, 'loss': v} for s, v in vals],
            'final_validation': vals[-1][1],
            'best_validation': {'epoch': best[0] // 733, 'step': best[0], 'loss': best[1]},
            'train_epoch1_update_weighted': weighted(2, 0, 733),
            'train_epochs19_20_update_weighted': weighted(2, 13194),
            'dropout_update_weighted': weighted(3),
            'gradient_logging': gradients,
        }
        histories[key] = {'val': dict(vals), 'dropout': [(r[0], r[3]) for r in intervals]}
    report['identical_dropout_log_series'] = (
        histories['oracle']['dropout'] == histories['constant']['dropout'] == histories['camera']['dropout'])
    report['oracle_comparisons'] = {}
    for key in ('constant', 'camera', 'image_pointmap', 'image_only'):
        other = report['runs'][key]
        oracle = report['runs']['oracle']
        difference = oracle['final_validation'] - other['final_validation']
        report['oracle_comparisons'][key] = {
            'final_validation_oracle_minus_reference': difference,
            'final_validation_relative_percent': 100 * difference / other['final_validation'],
            'epochs_oracle_lower_validation': sum(
                value < histories[key]['val'][step] for step, value in histories['oracle']['val'].items()),
            'late_training_relative_percent': 100 * (
                oracle['train_epochs19_20_update_weighted'] / other['train_epochs19_20_update_weighted'] - 1),
        }
    report['limitations'] = [
        'Training averages are update-weighted estimates; logs internally weight samples and final batches may be smaller.',
        'New training losses mix visual-present and visual-zero examples; historical visual baselines do not.',
        'Validation keeps visuals present; no visual-zero or per-time or per-object losses were exported.',
        'Nonzero gradients prove a connected optimization path, not useful geometry interpretation.',
        'Epochs reuse the same validation data and seeded evaluation; they are not independent replicates.',
        'Historical image+pointmap manifest differs by path; equality with the full-surface manifest is not verified locally.',
        'Equal seeds and dropout counts do not by themselves prove identical latent noise or all training input tensors.',
        'No reconstruction or coordinate-impossibility claim follows from aggregate native losses.',
    ]
    report['analysis_source_sha256'] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'comparisons': report['oracle_comparisons'],
                      'identical_dropout': report['identical_dropout_log_series']}, indent=2))


if __name__ == '__main__':
    main()
