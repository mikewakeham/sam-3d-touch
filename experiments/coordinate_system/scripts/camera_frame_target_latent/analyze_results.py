"""Plot three completed local results.json files; no model loading or GPU."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir', type=Path, help='Contains <arm>/results.json')
    args = parser.parse_args()
    arms = ['object_stock', 'camera_stock', 'camera_shared']
    runs = {arm: json.loads((args.results_dir/arm/'results.json').read_text()) for arm in arms}
    reference = runs[arms[0]]['metadata']
    match_keys = ['selection', 'validation_ids', 'source_sha256', 'preparation_sha256',
                  'training_order_sha256', 'initial_trainable_sha256', 'runtime', 'trainable_parameters']
    reference_settings = {k: v for k,v in reference['settings'].items() if k not in ('arm', 'output_dir')}
    for arm, run in runs.items():
        assert run['complete'] and run['metadata']['settings']['arm'] == arm
        assert all(run['metadata'][k] == reference[k] for k in match_keys)
        settings = {k: v for k,v in run['metadata']['settings'].items() if k not in ('arm', 'output_dir')}
        assert settings == reference_settings
        assert [row['global_step'] for row in run['validations']] == list(range(0, 1001, 100))
    train_ids = set(reference['selection']['train'])
    held_ids = set(reference['selection']['held_view'])
    assert len(train_ids) == 128 and len(held_ids) == 64 and not train_ids & held_ids
    train_objects = {s.rsplit('_', 1)[0] for s in train_ids}
    held_objects = {s.rsplit('_', 1)[0] for s in held_ids}
    assert train_objects == held_objects and len(train_objects) == 16
    assert not reference['selection']['val']
    final = {arm: run['validations'][-1] for arm,run in runs.items()}
    summary = dict(recorded_pairing_verified=True, matched_fields=match_keys,
        settings=reference_settings, objects=16, training_views=128, held_views=64,
        final=final, comparisons={})
    for key in ['loss/train_fixed', 'loss/held_view_fixed']:
        a,b,c = [final[arm][key] for arm in arms]
        summary['comparisons'][key] = dict(camera_vs_object_percent=100*(b/a-1),
            shared_vs_stock_percent=100*(c/b-1), shared_minus_stock=c-b,
            camera_stock_improvement_500_to_1000_percent=100*(1-b/runs['camera_stock']['validations'][5][key]))
    summary['limits'] = ['One seed; aggregate losses only, no independent-sample confidence intervals.',
        'Object/camera target distributions differ; raw loss ratios are not reconstruction-quality ratios.',
        'Training-reference and held-view sets use different observations/noise banks; absolute train/held gaps are not controlled view effects.',
        'No preparation figures/report or generated checkpoint reconstructions included in these three result files.',
        'All runs use shape cross-attention scope and visual dropout 0.5; other scopes/policies are untested here.']
    (args.results_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2563a6', '#de7c20', '#208c6b']
    labels = ['Object target / stock pointmap', 'Camera target / stock pointmap', 'Camera target / shared pointmap']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), sharey=True)
    for ax,key,title in zip(axes, ['loss/train_fixed','loss/held_view_fixed'],
                            ['Training reference: 32 views', 'Held-out views: 64 views of the same objects']):
        for arm, color, label in zip(arms, colors, labels):
            rows = runs[arm]['validations'][1:]
            ax.plot([r['global_step'] for r in rows], [r[key] for r in rows],
                    label=label, color=color, linestyle='--' if arm=='camera_shared' else '-', marker='o', markersize=3)
        ax.set_title(title, fontsize=11); ax.set_xlabel('Optimizer update')
        ax.set_xticks([100, 300, 500, 700, 900, 1000]); ax.grid(alpha=.2)
        ax.set_ylim(.055, .175); ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Fixed-bank Stage-1 flow-matching loss')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5,.95), fontsize=9)
    fig.suptitle('Camera-frame target experiment: shared pointmap normalization gives no clear loss benefit', fontsize=12, y=.98)
    fig.text(.5, .01, '16 objects; 1,000 updates; shape cross-attention training. Both measurements keep visual inputs present.\n'
             'Steps 100–1,000 shown; the initial step-0 loss drop is omitted for readability. One seed; no reconstruction measurements.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.09,1,.87)); fig.savefig(args.results_dir/'loss_curves.png', dpi=170); plt.close(fig)
    print(json.dumps(summary['comparisons'], indent=2))


if __name__ == '__main__': main()
