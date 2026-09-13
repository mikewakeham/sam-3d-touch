"""CPU-only choices for the aligned full-data upper-bound experiment."""
import hashlib
import json
import random

ARMS = ('oracle_dropout', 'constant_dropout', 'camera_dropout', 'image')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def arm_config(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return dict(oracle=arm in ('oracle_dropout', 'constant_dropout'),
                touch=arm != 'image', constant=arm == 'constant_dropout',
                visual_dropout=0.0 if arm == 'image' else 0.5)


def epoch_schedule(size, batch_size, seed, epoch):
    """Same order/noise/dropout across arms; no global RNG consumption."""
    if size < 1 or batch_size < 1:
        raise ValueError('Need nonempty data and positive batch size')
    order = list(range(size))
    random.Random(1100000 + seed + epoch).shuffle(order)
    batches = [order[i:i + batch_size] for i in range(0, size, batch_size)]
    drops = [True] * (len(batches) // 2) + [False] * (len(batches) - len(batches) // 2)
    random.Random(1200000 + seed + epoch).shuffle(drops)
    return batches, drops


def evaluation_groups(records, seed, limit=32, views=2):
    """Select by metadata before outcomes; each group has distinct identities."""
    by_object = {}
    for record in records:
        by_object.setdefault(record['object_id'], []).append(record)
    ids = sorted(by_object, key=lambda oid: digest([seed, oid]))[:limit]
    if len(ids) < 4:
        raise ValueError('At least four identities per split are required')
    # Groups of four allow fixed nonidentity cyclic distractors.
    ids = ids[:len(ids) // 4 * 4]
    groups = []
    for offset in range(0, len(ids), 4):
        objects = ids[offset:offset + 4]
        choices = {oid: sorted(by_object[oid], key=lambda r: r['sample_id']) for oid in objects}
        if any(len(v) < views for v in choices.values()):
            raise ValueError('Not enough views for fixed evaluation selection')
        for view in range(views):
            groups.append([choices[oid][view * (len(choices[oid]) - 1) // max(views - 1, 1)]
                           for oid in objects])
    return groups


def check_splits(train, val):
    tids = {r['object_id'] for r in train}
    vids = {r['object_id'] for r in val}
    if not tids or not vids or tids & vids:
        raise ValueError('Training and validation identities must be nonempty and disjoint')
    samples = [r['sample_id'] for r in train + val]
    if len(samples) != len(set(samples)):
        raise ValueError('Duplicate sample IDs')
    return dict(train_objects=len(tids), val_objects=len(vids),
                train_samples=len(train), val_samples=len(val))
