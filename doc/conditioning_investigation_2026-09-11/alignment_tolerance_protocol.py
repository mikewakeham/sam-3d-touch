"""Prespecified coordinate-error probe and target-referenced reporting."""
import math
import statistics


def cases(shard):
    if shard == 0:
        return [('original', 'aligned', None, 0), ('dropout', 'aligned', None, 0),
                ('dropout', 'wrong_surface', None, 0)]
    angle = {1: 5, 2: 15, 3: 30}[shard]
    return [('dropout', f'{axis}_{sign*angle:+d}', axis, sign*angle)
            for axis in ('x', 'y', 'z') for sign in (-1, 1)]


def rotation(axis, degrees):
    """Active column-vector rotation. Row-vector points multiply its transpose."""
    r = [[float(i == j) for j in range(3)] for i in range(3)]
    if axis is None:
        assert degrees == 0
        return r
    i, j = {'x': (1, 2), 'y': (2, 0), 'z': (0, 1)}[axis]
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    r[i][i] = r[j][j] = c
    r[i][j], r[j][i] = -s, s
    return r


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row['policy'], row['condition'], row['split'], row['object_id'])
        value = row['voxel_iou']
        assert math.isfinite(value) and 0 <= value <= 1
        grouped.setdefault(key, []).append(value)
    cells = {}
    for (policy, condition, split, obj), values in grouped.items():
        cells.setdefault((policy, condition, split), {})[obj] = statistics.mean(values)
    summary = {}
    for (policy, condition, split), objects in cells.items():
        summary.setdefault(split, {})[policy + '/' + condition] = {
            'mean_object_iou': statistics.mean(objects.values()),
            'minimum_object_mean_iou': min(objects.values()), 'object_iou': objects}
    baseline = cells[('dropout', 'aligned', 'reserved_view')]
    # Engineering gates for this fitting/reference task, not a population claim.
    reference_accurate = statistics.mean(baseline.values()) >= .95 and min(baseline.values()) >= .90
    tolerance = {}
    for angle in (5, 15, 30):
        entries = {}
        for _, name, axis, signed in cases({5: 1, 15: 2, 30: 3}[angle]):
            objects = cells[('dropout', name, 'reserved_view')]
            assert objects.keys() == baseline.keys()
            delta = {obj: objects[obj] - baseline[obj] for obj in baseline}
            entries[name] = {'mean_iou_change': statistics.mean(delta.values()),
                             'minimum_object_mean_iou': min(objects.values()),
                             'object_iou_change': delta,
                             'passes_sampled_direction': min(delta.values()) >= -.02 and min(objects.values()) >= .90}
        tolerance[str(angle)] = {'directions': entries,
                                'passes_all_sampled_directions': reference_accurate and all(
                                    e['passes_sampled_direction'] for e in entries.values())}
    return {'cells': summary, 'aligned_reference_meets_95_mean_90_each': reference_accurate,
            'rotation_tolerance': tolerance,
            'scope': 'Four fitted identities and their reserved views, CFG 0, two sampling seeds. '
                     'Engineering thresholds were fixed before results: aligned mean IoU >= .95 '
                     'and each object mean >= .90; tested rotation directions lose <= .02 IoU '
                     'for each object and retain >= .90. Not proof for all rotations, objects, '
                     'seeds, or learned pose errors. No equivalence or generalization claim.'}
