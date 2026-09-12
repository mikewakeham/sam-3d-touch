"""CPU protocol for the bounded 16-training-object coordinate comparison."""
import hashlib
import json
import random

MODEL_KEYS = ('image', 'camera', 'oracle', 'oracle_dropout')


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_schedule(seed=37):
    flags = {}
    for batch in range(16):
        values = [False] * 32 + [True] * 32
        random.Random(810000 + seed + batch).shuffle(values)
        flags[batch] = values
    schedule = []
    for epoch in range(64):
        order = list(range(16))
        random.Random(820000 + seed + epoch).shuffle(order)
        schedule.extend({'batch': b, 'drop_visual': flags[b][epoch]} for b in order)
    return schedule


def validate_plan(plan):
    assert plan['seed'] == 37 and plan['steps'] == 1024
    assert len(plan['training_object_ids']) == len(plan['held_object_ids']) == 16
    assert len(set(plan['training_object_ids'])) == len(set(plan['held_object_ids'])) == 16
    assert not set(plan['training_object_ids']) & set(plan['held_object_ids'])
    assert len(plan['batches']) == 32
    seen = set()
    for index, batch in enumerate(plan['batches']):
        assert batch['batch'] == index and len(batch['sample_ids']) == len(batch['object_ids']) == 4
        assert len(set(batch['object_ids'])) == 4
        assert not seen & set(batch['sample_ids'])
        seen.update(batch['sample_ids'])
        split = 'fit' if index < 16 else 'held_view' if index < 24 else 'held_object'
        assert batch['split'] == split
        expected = plan['held_object_ids'] if split == 'held_object' else plan['training_object_ids']
        assert set(batch['object_ids']) <= set(expected)
    for split, ids, count in [('fit', plan['training_object_ids'], 4),
                              ('held_view', plan['training_object_ids'], 2),
                              ('held_object', plan['held_object_ids'], 2)]:
        actual = [o for b in plan['batches'] if b['split'] == split for o in b['object_ids']]
        assert all(actual.count(o) == count for o in ids)
