"""Validate and summarize the frozen visual-stream factorial; standard library only."""
import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import mean

STREAMS = ('rgb', 'silhouette', 'pointmap')
BITS = tuple(itertools.product((0, 1), repeat=3))


def analyze(report):
    assert report['complete'] and report['preflight_passed']
    settings = report['settings']
    assert settings['arm'] == 'oracle' and settings['draws'] == 8
    assert settings['probe_bank_base'] == 300000 and settings['replay_bank_base'] == 100000
    assert len(report['input_batches']) == 7
    assert len({r['target_sha256'] for r in report['input_batches']}) == 1
    replay = report['preflight']
    assert len(replay) == 14
    assert {(r['group'], r['swap']) for r in replay} == set(itertools.product(range(7), (False, True)))
    for row in replay:
        assert len(row['losses']) == 8 and all(math.isfinite(x) and x >= 0 for x in row['losses'])
        assert math.isfinite(row['max_abs_error']) and row['max_abs_error'] >= 0
    table = {}
    for row in report['rows']:
        assert set(row['other_view_streams']) == set(STREAMS)
        bits = tuple(row['other_view_streams'][s] for s in STREAMS)
        assert bits in BITS and isinstance(row['swapped_surface'], bool)
        group = row['group']
        assert row['split'] == ('fit' if group < 4 else 'reserved_view')
        assert len(row['losses']) == 8 and all(math.isfinite(x) and x >= 0 for x in row['losses'])
        key = group, bits, row['swapped_surface']
        assert key not in table, 'Duplicate cell'
        table[key] = row['losses']
    assert set(table) == set(itertools.product(range(1, 7), BITS, (False, True))), 'Missing cells'
    for group in range(2, 7):
        for swap in (False, True):
            assert table[group, (0, 0, 0), swap] == table[1, (0, 0, 0), swap]

    def cell(group, bits, quantity):
        correct = table[group, bits, False]
        if quantity == 'correct_surface_loss':
            return correct
        return [wrong - right for wrong, right in zip(table[group, bits, True], correct)]

    summary = {}
    for split, groups in [('fit_view_changes', range(1, 4)), ('reserved_view_changes', range(4, 7))]:
        out = {}
        for quantity in ('correct_surface_loss', 'surface_benefit_wrong_minus_correct'):
            cells = {''.join(map(str, bits)): mean(x for g in groups for x in cell(g, bits, quantity))
                     for bits in BITS}
            effects = {}
            for index, stream in enumerate(STREAMS):
                conditional = []
                for bits in BITS:
                    if bits[index] == 1:
                        continue
                    changed = list(bits)
                    changed[index] = 1
                    delta = [b - a for g in groups for a, b in
                             zip(cell(g, bits, quantity), cell(g, tuple(changed), quantity))]
                    conditional.append({'other_stream_settings': {s: bits[i] for i, s in enumerate(STREAMS)
                                                                   if i != index},
                                        'mean_delta': mean(delta)})
                effects[stream] = conditional
            out[quantity] = {'cells_rgb_silhouette_pointmap': cells, 'conditional_effects': effects,
                             'coherent_other_minus_anchor': cells['111'] - cells['000']}
        summary[split] = out
    return {'summary': summary,
            'limits': 'Four fitted objects and one checkpoint. Changes in mismatched view streams '
                      'measure model dependence, not physical inconsistency of natural inputs. '
                      'Do not assign additive blame percentages. Draws and views share objects; '
                      'no independent-sample significance claim. A positive wrong-minus-correct '
                      'surface gap establishes relative preference, not superiority to image-only.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(json.loads(args.report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
