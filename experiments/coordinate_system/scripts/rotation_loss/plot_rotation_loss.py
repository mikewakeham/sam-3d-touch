"""Aggregate by object and make slide figures from either rotation probe."""
import argparse
import csv
import json
from pathlib import Path
import sys
import textwrap

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def object_means(rows, metric):
    values = {}
    for row in rows:
        values.setdefault(row["object_id"], []).append(float(row[metric]))
    return {oid: float(np.mean(v)) for oid, v in values.items()}


def estimate(values):
    # Resample objects, after averaging their repeated views/draws.
    values = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(29)
    samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(1)
    low, high = np.quantile(samples, [.025, .975])
    return {"objects": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)), "ci_low": float(low), "ci_high": float(high)}


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def rotation_figures(root, output, summary):
    rows = read_csv(root/"measurements.csv")
    metadata = json.loads((root/"results.json").read_text())
    table = []
    keys = sorted({(r["split"], r["part"], r["axis"], float(r["degrees"])) for r in rows
                   if r["split"] != "example"})
    for split, part, axis, angle in keys:
        cell = [r for r in rows if (r["split"], r["part"], r["axis"], float(r["degrees"])) == (split, part, axis, angle)]
        table.append(dict(split=split, part=part, axis=axis, degrees=angle,
                          **estimate(object_means(cell, "latent_mse").values())))
    summary["rotation_targets"] = table
    summary["numerical_floor"] = metadata["checks"]
    summary["different_object_reference"] = metadata["different_object_reference"]
    splits = ["val"] if any(r["split"] == "val" for r in table) else ["train"]
    fig, axes = plt.subplots(1, len(splits), figsize=(6.6, 4.8), squeeze=False)
    for ax, split in zip(axes[0], splits):
        for axis, color in zip("xyz", ["#c74a46", "#27864b", "#3478bd"]):
            cells = sorted([r for r in table if r["split"] == split and r["part"] == "padded"
                            and r["axis"] in (axis, "none")], key=lambda r: r["degrees"])
            angle = [r["degrees"] for r in cells]
            ax.plot(angle, [r["mean"] for r in cells], "o-", color=color, label=axis.upper())
            ax.fill_between(angle, [r["ci_low"] for r in cells], [r["ci_high"] for r in cells], color=color, alpha=.12)
        floor = [c["zero_repeat_mse"] for c in metadata["checks"] if c["split"] == split and c["part"] == "padded"]
        ax.axhline(np.mean(floor), color="gray", linestyle=":", label="Repeat-zero floor")
        ax.set(title=f"{split}: same shape, fixed padded scale", xlabel="Rotation (degrees)", ylabel="SS target mean MSE")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.22),
                  ncol=4, fontsize=9)
        ax.grid(alpha=.15)
    save_figure(fig, output/"rotation_angle_mse")
    summary["native_controls"] = [r for r in table if r["part"] == "native"]

    # Reuse accepted input-plot style. These are encoder INPUTS, not decodings.
    from experiments.coordinate_system.scripts.presentation.render_conditioning import plot
    for example in (root/"examples").glob("*.npz"):
        with np.load(example) as data:
            names = ["original", "rotated", "restored"]
            for name in names:
                points = (np.argwhere(data[name])+.5)/64-.5
                plot(output/f"{example.stem}_{name}.png", [(points, "#337bc4", .9)],
                     np.zeros(3), .55, grid_spacing=.5, object_axes=None)
            axis = str(data["axis"].item()).upper() if "axis" in data else "Z"
            angle = int(data["degrees"].item()) if "degrees" in data else 90
            captions = ["Original encoder input", f"Same input, exact {axis} {angle}°", "Known inverse restored"]
            scores = [0., float(data["latent_mse"]), float(data["restored_latent_mse"])]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
        for ax, name, caption, score in zip(axes, names, captions, scores):
            ax.imshow(Image.open(output/f"{example.stem}_{name}.png"))
            ax.set_axis_off()
            ax.set_title(caption)
            ax.text(.5, -.03, f"Encoded mean MSE: {score:.6g}", transform=ax.transAxes, ha="center")
        save_figure(fig, output/f"{example.stem}_rotation_comparison")


def checkpoint_figures(root, output, summary):
    losses = read_csv(root/"velocity_losses.csv")
    endpoints = read_csv(root/"endpoint_scores.csv")
    metadata = json.loads((root/"results.json").read_text())
    models = [m["name"] for m in metadata["models"]]
    splits = sorted({r["split"] for r in losses})
    table = []
    for model in models:
        for split in splits:
            for surface, kind in sorted({(r["surface"], r["time_kind"]) for r in losses}):
                cell = [r for r in losses if (r["model"], r["split"], r["surface"], r["time_kind"]) == (model, split, surface, kind)]
                if cell:
                    table.append(dict(model=model, split=split, metric="velocity_mse", surface=surface,
                                      time_kind=kind, **estimate(object_means(cell, "velocity_mse").values())))
            cell = [r for r in endpoints if (r["model"], r["split"]) == (model, split)]
            for metric in ("identity_mse", "best_mse"):
                table.append(dict(model=model, split=split, metric=metric, surface="correct",
                                  time_kind="sampled_endpoint", **estimate(object_means(cell, metric).values())))
    summary["checkpoints"] = table
    # Matched raw score contrasts, rather than treating rotated-reference minima as learning gains.
    summary["paired_model_differences"] = []
    for split in splits:
        for metric, source in (("velocity_mse", losses), ("identity_mse", endpoints)):
            cells = {}
            for model in models:
                cell = [r for r in source if r["model"] == model and r["split"] == split]
                if metric == "velocity_mse":
                    cell = [r for r in cell if r["surface"] == "correct" and r["time_kind"] == "native"]
                cells[model] = object_means(cell, metric)
            base = cells[models[0]]
            for model in models[1:]:
                common = sorted(base.keys() & cells[model].keys())
                summary["paired_model_differences"].append(dict(split=split, metric=metric,
                    comparison=f"{model} minus {models[0]}",
                    **estimate([cells[model][oid]-base[oid] for oid in common])))
    for metrics, filename, title in ((["velocity_mse"], "native_velocity_loss", "Observed native velocity MSE"),
                                     (["identity_mse", "best_mse"], "endpoint_latent_scores", "Fixed endpoint latent MSE")):
        fig, axes = plt.subplots(1, len(splits), figsize=(6*len(splits), 4.8), squeeze=False)
        for ax, split in zip(axes[0], splits):
            for j, metric in enumerate(metrics):
                cells = [next(r for r in table if r["model"] == m and r["split"] == split
                              and r["metric"] == metric and r["surface"] == "correct"
                              and (metric != "velocity_mse" or r["time_kind"] == "native")) for m in models]
                x = np.arange(len(models))+(j-(len(metrics)-1)/2)*.35
                means = np.array([r["mean"] for r in cells])
                errors = np.maximum(0, [means-[r["ci_low"] for r in cells], [r["ci_high"] for r in cells]-means])
                label = {"velocity_mse": "Native times", "identity_mse": "Original target", "best_mse": "Best tested rotation"}[metric]
                ax.bar(x, means, width=.35, yerr=errors, capsize=3, label=label)
            labels = [textwrap.fill(m.removeprefix("stage1_").removesuffix("_full_cross_attention").replace("_", " "), 20)
                      for m in models]
            ax.set_xticks(np.arange(len(models)), labels, fontsize=9)
            ax.set(title=f"{split}: {title}", ylabel="MSE")
            ax.margins(y=.25)
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False, loc="upper left")
        save_figure(fig, output/filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.rotation_dir and not args.checkpoint_dir:
        parser.error("Supply --rotation-dir, --checkpoint-dir, or both")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    if args.rotation_dir:
        rotation_figures(args.rotation_dir, args.output_dir, summary)
    if args.checkpoint_dir:
        checkpoint_figures(args.checkpoint_dir, args.output_dir, summary)
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("Saved:", args.output_dir)
