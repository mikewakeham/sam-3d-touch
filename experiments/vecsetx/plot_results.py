import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reconstruction-metrics",
        type=Path,
        default=Path("experiments/vecsetx/outputs/reconstruction/metrics.json"),
    )
    parser.add_argument(
        "--latent-metrics",
        type=Path,
        default=Path("experiments/vecsetx/outputs/latent_space/metrics.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/vecsetx/outputs/plots"),
    )
    return parser.parse_args()


def load_json(path):
    with path.open() as file:
        return json.load(file)


def save_figure(figure, path):
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_reconstruction(report, output_dir):
    samples = report["samples"]
    resolution = report["settings"]["resolution"]
    grid_spacing = 2.0 / resolution

    fields = {
        "Input → mesh\nmean": "reference_to_reconstruction_mean",
        "Input → mesh\n95th percentile": "reference_to_reconstruction_p95",
        "Mesh → input\nmean": "reconstruction_to_reference_mean",
        "Mesh → input\n95th percentile": "reconstruction_to_reference_p95",
    }
    distances = {
        label: [
            sample["full_surface"]["metrics"][field] / grid_spacing
            for sample in samples
        ]
        for label, field in fields.items()
    }
    chamfer = np.array(
        [sample["full_surface"]["metrics"]["chamfer_l2"] for sample in samples]
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=pd.DataFrame(distances), color="#4C78A8", ax=axes[0])
    axes[0].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel(f"Distance in {resolution}³ SDF grid cells")
    axes[0].set_xlabel("")
    axes[0].set_title("Surface distance")

    sns.ecdfplot(x=chamfer, color="#4C78A8", linewidth=2.5, ax=axes[1])
    median = float(np.median(chamfer))
    percentile_90 = float(np.quantile(chamfer, 0.9))
    axes[1].axvline(median, color="#F58518", linestyle="--", linewidth=1.5)
    axes[1].axvline(percentile_90, color="#E45756", linestyle="--", linewidth=1.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Chamfer-L2 in VecSetX coordinates")
    axes[1].set_ylabel("Fraction of samples")
    axes[1].set_title("Error distribution")
    axes[1].text(median, 0.48, f" median\n {median:.2e}", color="#F58518")
    axes[1].text(
        percentile_90,
        0.82,
        f" 90th percentile\n {percentile_90:.2e}",
        color="#E45756",
    )

    figure.suptitle(
        f"VecSetX full-surface reconstruction ({len(samples)} samples)",
        fontweight="bold",
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "reconstruction_quantitative")


def plot_latent(report, output_dir):
    alignment = report["source_alignment"]
    samples = report["samples"]
    sample_count = len(samples)
    object_counts = Counter(sample["object_id"] for sample in samples)
    exact_chance = 1.0 / sample_count
    object_chance = sum(count**2 for count in object_counts.values()) / sample_count**2

    retrieval = pd.DataFrame(
        [
            {
                "Input": source.title(),
                "Criterion": criterion,
                "Accuracy": alignment[source][field],
            }
            for source in ("touch", "joint")
            for criterion, field in (
                ("Exact sample", "nearest_full_exact_sample_accuracy"),
                ("Same object", "nearest_full_same_object_accuracy"),
            )
        ]
        + [
            {"Input": "Chance", "Criterion": "Exact sample", "Accuracy": exact_chance},
            {"Input": "Chance", "Criterion": "Same object", "Accuracy": object_chance},
        ]
    )

    distances = pd.DataFrame(
        {
            "Comparison": [
                "Joint ↔ matching full",
                "Touch ↔ matching full",
                "Unrelated full ↔ full",
            ],
            "Median cosine distance": [
                alignment["joint"]["full_to_source_distance_median"],
                alignment["touch"]["full_to_source_distance_median"],
                alignment["different_object_full_surface_distance_median"],
            ],
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(
        data=retrieval,
        x="Criterion",
        y="Accuracy",
        hue="Input",
        palette={"Touch": "#F58518", "Joint": "#54A24B", "Chance": "#BAB0AC"},
        ax=axes[0],
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Top-1 retrieval accuracy")
    axes[0].set_xlabel("")
    axes[0].set_title(f"Retrieval from {sample_count:,} full codes")
    axes[0].legend(title="")
    for container in axes[0].containers:
        labels = [
            f"{bar.get_height() * 100:.1f}%"
            if bar.get_height() >= 0.01
            else f"{bar.get_height() * 100:.3f}%"
            for bar in container
        ]
        axes[0].bar_label(container, labels=labels, fontsize=10, padding=2)

    sns.barplot(
        data=distances,
        x="Comparison",
        y="Median cosine distance",
        hue="Comparison",
        palette=["#54A24B", "#F58518", "#B279A2"],
        legend=False,
        ax=axes[1],
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Median cosine distance")
    axes[1].set_title("Encoder-space correspondence")
    axes[1].tick_params(axis="x", rotation=15)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.4f", fontsize=10, padding=2)

    figure.suptitle(
        "VecSetX bottlenecks preserve shape correspondence",
        fontweight="bold",
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "latent_quantitative")


def select_qualitative_samples(report, output_dir):
    samples = report["samples"]
    errors = np.array(
        [sample["full_surface"]["metrics"]["chamfer_l2"] for sample in samples]
    )
    selected = []
    for label, quantile in (
        ("10th percentile", 0.1),
        ("median", 0.5),
        ("90th percentile", 0.9),
        ("worst", 1.0),
    ):
        target = np.quantile(errors, quantile)
        index = int(np.argmin(np.abs(errors - target)))
        selected.append(
            {
                "level": label,
                "sample_id": samples[index]["sample_id"],
                "chamfer_l2": float(errors[index]),
            }
        )

    with (output_dir / "qualitative_samples.json").open("w") as file:
        json.dump(selected, file, indent=2)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    reconstruction = load_json(args.reconstruction_metrics)
    latent = load_json(args.latent_metrics)
    plot_reconstruction(reconstruction, args.output_dir)
    plot_latent(latent, args.output_dir)
    select_qualitative_samples(reconstruction, args.output_dir)
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
