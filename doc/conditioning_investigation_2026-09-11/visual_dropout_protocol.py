"""Pure-Python schedule for the matched visual-dropout experiment."""
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
