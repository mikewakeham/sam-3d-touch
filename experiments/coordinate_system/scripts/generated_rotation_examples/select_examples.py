"""Summarize all fixed examples and select illustrative orientation-error cases."""
import argparse
import json
from pathlib import Path


def quantities(row):
    fit = row['rotation_fit']
    latent = row['latent_mse']
    raw = fit['raw']; aligned = fit['aligned_voxelized']
    latent_gain = (latent['decoded_unaligned_reencoded_to_stored_target']-
                   latent['decoded_aligned_reencoded_to_stored_target'])
    return {'geometry_f2_gain': aligned['fscore_2v']-raw['fscore_2v'],
            'aligned_fscore_2v': aligned['fscore_2v'],
            'aligned_min_precision_recall_2v': min(aligned['precision_2v'], aligned['recall_2v']),
            'reencoded_latent_mse_reduction': latent_gain,
            'reencoded_latent_relative_reduction': latent_gain/latent['decoded_unaligned_reencoded_to_stored_target']
                if latent['decoded_unaligned_reencoded_to_stored_target'] else 0.,
            'rotation_degrees': fit['angle_degrees'],
            'dropped_fraction': fit['aligned_rasterization']['dropped_fraction']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top', type=int, default=6)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    if not report.get('complete'):
        raise ValueError('Measurement report is incomplete')
    rows = []
    for row in report['examples']:
        values = quantities(row)
        both_improve = (values['geometry_f2_gain'] > 0 and
                        values['reencoded_latent_mse_reduction'] > 0)
        candidate = (both_improve and values['rotation_degrees'] >= 5 and
                     values['dropped_fraction'] <= .01 and
                     row['rotation_fit']['raw']['fscore_2v'] <= .6 and
                     values['aligned_min_precision_recall_2v'] >= .8 and
                     values['geometry_f2_gain'] >= .2 and
                     values['reencoded_latent_relative_reduction'] >= .1)
        rows.append({**row, 'derived': values, 'both_improve': both_improve,
                     'orientation_error_candidate': candidate})
    candidates = [row for row in rows if row['orientation_error_candidate']]
    candidates.sort(key=lambda row: (row['derived']['aligned_min_precision_recall_2v'],
                                     row['derived']['geometry_f2_gain'],
                                     row['derived']['reencoded_latent_relative_reduction']), reverse=True)
    selected, object_counts = [], {}
    for row in candidates:
        oid = row['object_id']
        if object_counts.get(oid, 0) >= 2: continue
        selected.append(row); object_counts[oid] = object_counts.get(oid, 0)+1
        if len(selected) == args.top: break
    summary = {'total_predictions': len(rows),
        'geometry_improves': sum(row['derived']['geometry_f2_gain'] > 0 for row in rows),
        'reencoded_latent_mse_improves': sum(row['derived']['reencoded_latent_mse_reduction'] > 0 for row in rows),
        'both_improve': sum(row['both_improve'] for row in rows),
        'orientation_error_candidates': len(candidates), 'selected_count': len(selected),
        'selection': 'Post-hoc candidate criterion: raw F@2 <=0.6; voxelized aligned min(precision,recall)@2 >=0.8; F@2 gain >=0.2; re-encoded latent MSE relative reduction >=10%; fitted angle >=5 degrees; <=1% aligned points outside target cube. Rank by aligned min(precision,recall)@2, then F@2 gain and relative latent-MSE reduction; at most two per object.',
        'selected': selected, 'all_examples': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: summary[k] for k in ['total_predictions','geometry_improves',
        'reencoded_latent_mse_improves','both_improve','orientation_error_candidates','selected_count']}, indent=2))


if __name__ == '__main__':
    main()
