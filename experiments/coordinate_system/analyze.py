from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT = Path('experiments/coordinate_system/results')


def rotation_matrix(transform):
    # Saved transforms use column vectors: R @ diag(scale).
    linear = transform[:3, :3].astype(float)
    return linear / np.linalg.norm(linear, axis=0)


def angular_distance(a, b):
    relative = np.asarray(a, dtype=float) @ np.swapaxes(np.asarray(b, dtype=float), -1, -2)
    return np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def rotation_metrics(prediction, target, references, camera_frame=False):
    pred_distances = angular_distance(prediction, references)
    gt_distances = angular_distance(target, references)
    return dict(
        rotation_error_deg=float(angular_distance(prediction, np.eye(3) if camera_frame else target)),
        prediction_nearest_old_deg=float(pred_distances.min()),
        gt_nearest_old_deg=float(gt_distances.min()),
        predicted_old_index=int(pred_distances.argmin()), gt_old_index=int(gt_distances.argmin()),
        nearest_old_agreement=int(pred_distances.argmin() == gt_distances.argmin()),
    )


def analyze(output=OUTPUT):
    data = pd.read_csv(output / 'metrics.csv')
    folder = output / 'analysis'
    folder.mkdir(exist_ok=True)
    with np.load(output / 'reference_cameras.npz') as reference:
        view_ids = reference['view_ids']
    columns = ['rotation_error_deg', 'prediction_nearest_old_deg', 'gt_nearest_old_deg', 'nearest_old_agreement']
    data.groupby(['model', 'camera_set'])[columns].agg(['count', 'mean', 'median']).to_csv(folder / 'summary.csv')
    sns.set_theme()

    def save(name):
        plt.tight_layout()
        plt.savefig(folder / f'{name}.png', dpi=200)
        plt.savefig(folder / f'{name}.pdf')
        plt.close()

    for model in ['base', 'old', 'new']:
        subset = data[data.model == model]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, camera_set in zip(axes, ['old', 'sampled']):
            views = subset[subset.camera_set == camera_set]
            counts = pd.crosstab(views.gt_old_index, views.predicted_old_index).reindex(
                index=range(8), columns=range(8), fill_value=0)
            counts.to_csv(folder / f'{model}_{camera_set}_counts.csv')
            sns.heatmap(counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0),
                        annot=True, fmt='.2f', vmin=0, vmax=1, ax=ax,
                        xticklabels=view_ids, yticklabels=view_ids)
            ax.set(title=f'{model}, {camera_set} cameras', xlabel='Predicted nearest old view', ylabel='GT nearest old view')
        save(f'{model}_confusion')

        plt.figure(figsize=(6, 4))
        sns.ecdfplot(data=subset, x='prediction_nearest_old_deg', hue='camera_set')
        plt.xlabel('Distance to nearest old rotation (degrees)')
        plt.title(model)
        save(f'{model}_distance')

        sampled = subset[subset.camera_set == 'sampled']
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=sampled, x='gt_nearest_old_deg', y='rotation_error_deg')
        if model != 'new':
            # Best error possible if restricted to the eight old rotations.
            distance = np.linspace(0, sampled.gt_nearest_old_deg.max(), 100)
            sns.lineplot(x=distance, y=distance, label='Nearest-eight oracle')
        plt.xlabel('GT distance to nearest old rotation (degrees)')
        plt.ylabel('Rotation from identity (degrees)' if model == 'new' else 'Rotation error (degrees)')
        plt.title(model)
        save(f'{model}_error')

        plt.figure(figsize=(6, 4))
        sns.countplot(data=subset, x='predicted_old_index', hue='camera_set', order=range(8))
        plt.xticks(range(8), view_ids)
        plt.xlabel('Predicted nearest old view')
        plt.title(model)
        save(f'{model}_selections')


if __name__ == '__main__':
    analyze()
