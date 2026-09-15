"""Save a few actual encoder inputs with the largest exact-rotation penalties."""
import argparse
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import numpy as np
from dataloader import load_data_config
from experiments.coordinate_system.scripts.rotation_loss.rotation_utils import (
    load_mesh, voxelize, native_rotations, rotate_grid,
)


def export_examples(probe_dir, data_config, count):
    with (probe_dir/"measurements.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    report = json.loads((probe_dir/"results.json").read_text())
    root = Path(load_data_config(data_config)["dataset"]["root"])
    # One worst exact rotation per identity; do not select three rotations of
    # the same object. Native rotations avoid mesh revoxelization artifacts.
    worst = {}
    for row in rows:
        if row["split"] != "val" or row["part"] != "native" or float(row["degrees"]) == 0:
            continue
        oid = row["object_id"]
        if oid not in worst or float(row["latent_mse"]) > float(worst[oid]["latent_mse"]):
            worst[oid] = row
    selected = sorted(worst.values(), key=lambda row: (-float(row["latent_mse"]), row["object_id"]))[:count]
    folder = probe_dir/"examples"
    folder.mkdir(exist_ok=True)
    for row in selected:
        oid, axis, angle = row["object_id"], row["axis"], int(float(row["degrees"]))
        matrix = next(m for _, a, d, m in native_rotations() if a == axis and d == angle)
        original = voxelize(load_mesh(root, oid), stock=True)
        assert int(original.sum()) == int(row["occupied"])
        rotated = rotate_grid(original, matrix)
        restored = rotate_grid(rotated, matrix.T)
        np.testing.assert_array_equal(restored, original)
        floor = next(c["zero_repeat_mse"] for c in report["checks"]
                     if c["object_id"] == oid and c["part"] == "native")
        np.savez_compressed(folder/f"{oid}.npz", original=original, rotated=rotated,
                            restored=restored, axis=axis, degrees=angle,
                            latent_mse=float(row["latent_mse"]), restored_latent_mse=floor)
        print(f"{oid}: {axis.upper()} {angle} degrees, latent MSE {float(row['latent_mse']):.6f}")
    (probe_dir/"high_error_examples.json").write_text(json.dumps({
        "selection": "Largest native 90/180-degree target latent MSE per validation object; top distinct identities. Selected extremes, not representative examples.",
        "geometry": "Rebuilt using original target voxelizer, then exact grid permutations. Scores copied from completed GPU probe; no model encoding or decoding here.",
        "examples": selected,
    }, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=REPO/"configs/data_full_surface.yaml")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    export_examples(args.probe_dir, args.data_config, args.top)
