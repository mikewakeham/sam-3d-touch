"""Paired latent error reduction: positive means the comparison model is better."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/noisy_target/outputs"))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()
    filename = args.sample_id.replace("/", "_") + ".npz"
    with np.load(args.output_dir / args.baseline / filename) as data:
        baseline, times = data["errors"], data["times"]
    with np.load(args.output_dir / args.comparison / filename) as data:
        if not np.array_equal(times, data["times"]) or baseline.shape != data["errors"].shape:
            raise ValueError("Maps must have matching times and noise draws")
        difference = (baseline - data["errors"]).mean(axis=1)
    bound = max(float(np.abs(difference).max()), 1e-8)
    figure, axes = plt.subplots(len(times), 3, squeeze=False, figsize=(10, 3 * len(times)))
    for row, time in enumerate(times):
        for axis in range(3):
            plane = np.take(difference[row], difference.shape[axis + 1] // 2, axis=axis)
            image = axes[row, axis].imshow(plane.T, origin="lower", cmap="RdBu", vmin=-bound, vmax=bound)
            axes[row, axis].set_title(f"t={time:g}; {'xyz'[axis]} middle slice")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Baseline − comparison velocity MSE")
    figure.suptitle(f"{args.sample_id}: latent-grid error reduction (blue improves, red worsens)")
    figure.savefig(args.output_dir / f"{args.comparison}_vs_{args.baseline}_{filename[:-4]}.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
