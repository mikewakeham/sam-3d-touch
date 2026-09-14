"""Pure-Python schedule for the matched visual-dropout experiment."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import hashlib
import json
import random


def dropout_schedule(seed=29):
    # Each group occurs 250 times. Balance within groups, not alternating global
    # steps (which would permanently drop visual input for only two groups).
    per_group = []
    for group in range(4):
        flags = [False] * 125 + [True] * 125
        random.Random(710000 + seed + group).shuffle(flags)
        per_group.append(flags)
    return [per_group[i % 4][i // 4] for i in range(1000)]


def schedule_digest(schedule):
    return hashlib.sha256(json.dumps(schedule, separators=(',', ':')).encode()).hexdigest()
