import argparse
import csv
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline")
    parser.add_argument("--touch")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_metrics(path):
    numeric = {
        "latent_mse", "iou", "dice", "precision", "recall",
        "predicted_voxels", "target_voxels", "volume_error",
    }
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def error_projection(prediction, target, axis):
    true_positive = (prediction & target).max(axis=axis)
    false_positive = (prediction & ~target).max(axis=axis)
    false_negative = (~prediction & target).max(axis=axis)
    image = np.zeros(true_positive.shape + (3,), dtype=np.float32)
    image[..., 0] = false_positive
    image[..., 1] = true_positive
    image[..., 2] = false_negative
    return image


def plot_example(sample_id, category, rank, conditions, rows, evaluation_dir, output_dir,
                 baseline, touch):
    sample_rows = {condition: rows[condition][sample_id] for condition in conditions}
    first = sample_rows[conditions[0]]
    first_voxels = np.load(evaluation_dir / first["voxel_path"], allow_pickle=False)
    target = first_voxels["target"]
    touch_centers = first_voxels["touch_centers"]

    figure, axes = plt.subplots(len(conditions) + 2, 3, figsize=(10, 2.2 * (len(conditions) + 2)))
    image = np.asarray(Image.open(first["image_path"]).convert("RGBA"))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input image")
    axes[0, 1].scatter(touch_centers[:, 0], touch_centers[:, 1], s=8)
    axes[0, 1].set_title("Touch centers XY")
    axes[0, 2].scatter(touch_centers[:, 0], touch_centers[:, 2], s=8)
    axes[0, 2].set_title("Touch centers XZ")

    projection_axes = [0, 1, 2]
    projection_names = ["YZ", "XZ", "XY"]
    for column, (axis, title) in enumerate(zip(projection_axes, projection_names)):
        axes[1, column].imshow(target.max(axis=axis), cmap="gray", origin="lower")
        axes[1, column].set_title(f"Target {title}")

    for row_index, condition in enumerate(conditions, start=2):
        values = sample_rows[condition]
        with np.load(evaluation_dir / values["voxel_path"], allow_pickle=False) as data:
            prediction = data["prediction"]
        for column, axis in enumerate(projection_axes):
            axes[row_index, column].imshow(
                error_projection(prediction, target, axis), origin="lower"
            )
        axes[row_index, 0].set_ylabel(
            f"{condition}\nIoU {values['iou']:.3f}", rotation=0, labelpad=55, va="center"
        )

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    delta = sample_rows[touch]["iou"] - sample_rows[baseline]["iou"]
    figure.suptitle(
        f"{sample_id}  object {first['object_id']}  {touch} - {baseline} IoU: {delta:+.3f}\n"
        "green: correct   red: false positive   blue: missed",
        y=0.995,
    )
    figure.tight_layout(rect=(0.06, 0, 1, 0.97))
    path = output_dir / f"{category}_{rank:02d}_{safe_name(sample_id)}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_summary(primary_conditions, rows, baseline, touch, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    values = [[row["iou"] for row in rows[condition].values()]
              for condition in primary_conditions]
    axes[0].boxplot(values, showfliers=False)
    axes[0].set_xticks(range(1, len(primary_conditions) + 1), primary_conditions)
    axes[0].set_ylabel("Voxel IoU")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Validation IoU")

    sample_ids = sorted(set(rows[baseline]) & set(rows[touch]))
    deltas = [rows[touch][sample]["iou"] - rows[baseline][sample]["iou"]
              for sample in sample_ids]
    axes[1].hist(deltas, bins=25)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel(f"{touch} - {baseline} IoU")
    axes[1].set_ylabel("Examples")
    axes[1].set_title("Per-example touch improvement")
    figure.tight_layout()
    figure.savefig(output_dir / "summary.png", dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = args.output_dir or args.evaluation_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.evaluation_dir / "summary.yaml") as file:
        summary = yaml.safe_load(file)
    metric_rows = read_metrics(args.evaluation_dir / "metrics.csv")
    rows = {}
    for row in metric_rows:
        rows.setdefault(row["condition"], {})[row["sample_id"]] = row

    baseline = args.baseline or summary["no_touch"] or "official"
    touch = args.touch or summary["best_touch"]
    primary_conditions = summary["primary_conditions"]
    conditions = primary_conditions + summary["diagnostic_conditions"]
    sample_ids = sorted(set(rows[baseline]) & set(rows[touch]))
    deltas = {sample: rows[touch][sample]["iou"] - rows[baseline][sample]["iou"]
              for sample in sample_ids}

    ordered = sorted(sample_ids, key=deltas.get)
    median = float(np.median(list(deltas.values())))
    selections = {
        "best": list(reversed(ordered[-args.count:])),
        "worst": ordered[:args.count],
        "typical": sorted(sample_ids, key=lambda sample: abs(deltas[sample] - median))[:args.count],
        "random": random.Random(args.seed).sample(sample_ids, min(args.count, len(sample_ids))),
    }

    plot_summary(primary_conditions, rows, baseline, touch, output_dir)
    for category, selected in selections.items():
        for rank, sample_id in enumerate(selected, start=1):
            plot_example(
                sample_id, category, rank, conditions, rows, args.evaluation_dir,
                output_dir, baseline, touch
            )

    with open(output_dir / "selections.yaml", "w") as file:
        yaml.safe_dump({
            "baseline": baseline,
            "touch": touch,
            "samples": selections,
        }, file, sort_keys=False)
    print(f"saved figures to {output_dir}")


if __name__ == "__main__":
    main()
