"""Read deduplicated historical evidence without unpacking the archives.

--verify checks immutable retained original bytes; --metrics additionally
recomputes the 1,280 completed full-training rollout IoUs/counts.
Only --extract writes a requested original file, and never overwrites a file.
"""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

DEFAULT_RESULTS = _repo.parent / 'coordinate_system_results'


class Evidence:
    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir or os.environ.get('SAM3D_COORDINATE_RESULTS', DEFAULT_RESULTS)).expanduser().resolve()
        index = self.results_dir / 'index.json'
        if not index.is_file():
            raise FileNotFoundError(f'{index}: results are stored outside the repository; set SAM3D_COORDINATE_RESULTS or pass --results-dir')
        self.manifest = json.loads(index.read_text())
        self.entries = {e['original']: e for e in self.manifest['entries']}
        self.archives = {}

    def close(self):
        for archive in self.archives.values():
            archive.close()

    def read(self, original):
        entry = self.entries[original]
        path = self.results_dir / Path(entry['path']).relative_to('results')
        if 'member' not in entry:
            return path.read_bytes()
        if path not in self.archives:
            self.archives[path] = zipfile.ZipFile(path)
        return self.archives[path].read(entry['member'])

    def verify(self):
        seen = set()
        original_count = 0
        for name, entry in self.entries.items():
            original_count += 1
            location = (entry['path'], entry.get('member'))
            if location in seen:
                continue
            data = self.read(name)
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError(f'Retained evidence differs: {name}')
            seen.add(location)
        return {'immutable_originals_resolved': original_count, 'unique_payloads_verified': len(seen)}

    def metrics(self):
        import numpy as np
        historical = 'doc/conditioning_investigation_2026-09-11/oracle_upper_bound/'
        roots = [historical+'rollouts_returned_20260913_143515/'+arm for arm in ('oracle','constant')]
        roots += ['outputs/objaverse/conditioning_investigation/full_frame_probe_20260913_172925/camera']
        count = 0
        cells = {}
        for root in roots:
            report = json.loads(self.read(root+'/results.json'))
            assert report['complete']
            cache = {}
            for row in report['samples']:
                for field, array in [('prediction_file','predicted_occupancy'),('target_file','target_occupancy')]:
                    name = row[field]
                    if name not in cache:
                        with np.load(io.BytesIO(self.read(root+'/'+name)), allow_pickle=False) as data:
                            cache[name] = (data[array].copy(), list(data['sample_ids']))
                i = row['array_index']
                pred, ids = cache[row['prediction_file']]
                target, target_ids = cache[row['target_file']]
                assert ids[i] == target_ids[i] == row['sample_id']
                a, b = pred[i], target[i]
                iou = float((a & b).sum()/(a | b).sum())
                assert abs(iou-row['iou']) < 1e-12
                assert int(a.sum()) == row['predicted_count'] and int(b.sum()) == row['target_count']
                count += 1
                key = '/'.join(map(str,[root.split('/')[-1],row['split'],row['visual'],row['surface_shift']]))
                cells.setdefault(key, []).append(iou)
        assert count == 1280
        return {'occupancy_iou_and_counts_recomputed': count,
                'mean_iou': {k: sum(v)/len(v) for k,v in cells.items()},
                'scope': 'Existing paired outputs only; no model execution or new registration.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, help='External result directory; defaults to SAM3D_COORDINATE_RESULTS or ../coordinate_system_results')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--metrics', action='store_true')
    parser.add_argument('--list', metavar='SUBSTRING')
    parser.add_argument('--extract', metavar='ORIGINAL_PATH')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    evidence = Evidence(args.results_dir)
    try:
        if args.list is not None:
            for name, entry in evidence.entries.items():
                if args.list in name:
                    print(name, entry['path'])
        if args.verify:
            print(json.dumps(evidence.verify(), indent=2))
        if args.metrics:
            print(json.dumps(evidence.metrics(), indent=2))
        if args.extract:
            if args.output is None:
                parser.error('--extract requires --output')
            data = evidence.read(args.extract)
            with args.output.open('xb') as file:
                file.write(data)
    finally:
        evidence.close()


if __name__ == '__main__':
    main()
