"""Independently reproduce support metrics and measure proper-rigid witnesses locally."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from pose_shape_geometry import points, metrics, register


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def work(task):
    root, arm, sample = task
    folder = Path(root) / arm
    with np.load(folder / sample['prediction_file'], allow_pickle=False) as pf, \
         np.load(folder / sample['target_file'], allow_pickle=False) as qf:
        index = sample['array_index']
        assert str(pf['sample_ids'][index]) == str(qf['sample_ids'][index]) == sample['sample_id']
        p_grid, q_grid = pf['predicted_occupancy'][index], qf['target_occupancy'][index]
    p, q = points(p_grid), points(q_grid)
    iou = float(np.count_nonzero(p_grid & q_grid) / np.count_nonzero(p_grid | q_grid))
    assert iou == sample['iou'] and len(p) == sample['predicted_count'] and len(q) == sample['target_count']
    raw = metrics(p, q)
    for key in ('precision_2v', 'recall_2v', 'fscore_2v'):
        assert abs(raw[key] - sample[key]) < 1e-12, key
    # Identity is already a maximal proximity witness when both P and R are 1.
    if raw['precision_2v'] == raw['recall_2v'] == 1:
        rigid = {'rotation': np.eye(3).tolist(), 'translation': [0., 0., 0.],
                 'metrics': raw, 'winner': 'identity_already_maximal_2v'}
    else:
        rigid = register(p, q)
    candidate = rigid['metrics']
    choose_rigid = (min(candidate['precision_2v'], candidate['recall_2v']), candidate['fscore_2v']) > (
        min(raw['precision_2v'], raw['recall_2v']), raw['fscore_2v'])
    return {'arm': arm, **sample, 'raw': raw, 'rigid': rigid,
            'witness': candidate if choose_rigid else raw,
            'witness_transform': 'rigid' if choose_rigid else 'identity'}


def summarize(rows):
    out = {}
    for split in ('train', 'val'):
        out[split] = {}
        for visual in ('present', 'zero'):
            cells = {}
            for label, arm, shift in (('oracle', 'oracle', 0), ('oracle_wrong', 'oracle', 1), ('constant', 'constant', 0)):
                selected = [r for r in rows if (r['split'], r['visual'], r['arm'], r['surface_shift']) == (split, visual, arm, shift)]
                objects = {}
                for oid in sorted({r['object_id'] for r in selected}):
                    group = [r for r in selected if r['object_id'] == oid]
                    objects[oid] = {kind: {k: mean(r[kind][k] for r in group) for k in
                        ('precision_2v', 'recall_2v', 'fscore_2v')} for kind in ('raw', 'witness')}
                cells[label] = {'objects': objects}
                for kind in ('raw', 'witness'):
                    cells[label][kind] = {
                        **{k: mean(o[kind][k] for o in objects.values()) for k in ('precision_2v', 'recall_2v', 'fscore_2v')},
                        'objects_95_precision_and_recall': sum(min(o[kind]['precision_2v'], o[kind]['recall_2v']) >= .95 for o in objects.values()),
                        'samples_95_precision_and_recall': sum(min(r[kind]['precision_2v'], r[kind]['recall_2v']) >= .95 for r in selected),
                        'objects_count': len(objects), 'samples_count': len(selected)}
            for reference in ('constant', 'oracle_wrong'):
                cells['oracle_minus_' + reference] = {}
                for kind in ('raw', 'witness'):
                    differences = {oid: v[kind]['fscore_2v'] - cells[reference]['objects'][oid][kind]['fscore_2v']
                                   for oid, v in cells['oracle']['objects'].items()}
                    cells['oracle_minus_' + reference][kind] = {'mean_difference': mean(differences.values()),
                        'objects_positive': sum(v > 0 for v in differences.values()), 'per_object': differences}
            out[split][visual] = cells
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    reports = {arm: json.loads((args.root / arm / 'results.json').read_text()) for arm in ('oracle', 'constant')}
    # Known rigid positive controls on four deterministically chosen new targets.
    controls_path = args.output / 'controls.json'
    controls = json.loads(controls_path.read_text()) if args.resume and controls_path.exists() else []
    for item in ([] if controls else reports['oracle']['inputs'][:4]):
        file = args.root / 'oracle' / f"{item['split']}_g{item['group']}_targets.npz"
        with np.load(file, allow_pickle=False) as a:
            q = points(a['target_occupancy'][0])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        r = register(q @ rotation.T + np.array([.07, -.02, .03]), q)
        assert min(r['metrics']['precision_2v'], r['metrics']['recall_2v']) >= .95
        controls.append({'sample_id': item['sample_ids'][0], 'rigid': r})
    (args.output / 'controls.json').write_text(json.dumps(controls, indent=2) + '\n')
    tasks = [(str(args.root), arm, s) for arm, report in reports.items() for s in report['samples']]
    # Prioritize correct oracle outputs across both splits, then matched controls.
    tasks.sort(key=lambda t: (t[1] != 'oracle' or t[2]['surface_shift'] != 0, t[2]['visual'] != 'present', t[2]['group']))
    progress_path = args.output / 'predictions.jsonl'
    rows = [json.loads(line) for line in progress_path.read_text().splitlines()] if args.resume and progress_path.exists() else []
    def key(arm, sample):
        return (arm, sample['split'], sample['group'], sample['sample_id'], sample['visual'], sample['surface_shift'], sample['draw'])
    expected = {key(arm, sample): sample for _, arm, sample in tasks}
    done = set()
    for row in rows:
        k = key(row['arm'], row)
        assert k in expected and k not in done and all(row[n] == v for n, v in expected[k].items())
        done.add(k)
    pending = [t for t in tasks if key(t[1], t[2]) not in done]
    print(f'Resuming {len(done)} verified rows; {len(pending)} remain', flush=True)
    start = time.monotonic()
    with progress_path.open('a' if args.resume else 'x') as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(work, pending), len(rows) + 1):
            rows.append(row); f.write(json.dumps(row) + '\n'); f.flush()
            if index % 32 == 0:
                print(f'Verified/registered {index}/{len(tasks)} in {time.monotonic()-start:.1f}s', flush=True)
    assert len(rows) == 768
    result = {'summary': summarize(rows), 'rows': len(rows), 'all_raw_metrics_reproduced': True,
        'source_sha256': {p.name: sha(p) for p in (Path(__file__), HERE.parent / 'pose_shape_geometry.py')},
        'report_sha256': {arm: sha(args.root / arm / 'results.json') for arm in reports},
        'scope': 'Identity or fixed proper-rigid RMS-registration candidate, selecting higher minimum P/R at two voxels. '
                 'No scale/reflection. Positive witness is valid; a poor witness is not proof no rigid transform can work.'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Complete', args.output / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
