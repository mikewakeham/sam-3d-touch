"""Reproduce run summaries and geometry invariants without the SAM3D runtime.

Requires NumPy only. Algebra below is an independent implementation, not an
execution of the torch preprocessing path. No neural-network claim is made.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def normalize(p):
    centered = p - (p.max(0) + p.min(0)) / 2
    return centered / np.linalg.norm(centered, axis=1).max()


def main():
    runs = []
    for path in sorted((REPO.parent / "wandb-results").glob("*/run.json")):
        run = json.loads(path.read_text())
        with path.with_name("history.csv").open() as f:
            rows = list(csv.DictReader(f))
        vals = [(float(r["loss/val"]), int(r["global_step"]))
                for r in rows if r.get("loss/val")]
        # Interval-weighted estimates: logging windows have variable lengths.
        epochs = defaultdict(lambda: [0., 0])
        previous = 0
        for r in rows:
            if not r.get("loss/train"):
                continue
            step = int(r["global_step"])
            epoch = (step - 1) // 733 + 1
            width = step - previous
            epochs[epoch][0] += float(r["loss/train"]) * width
            epochs[epoch][1] += width
            previous = step
        runs.append({"id": run["id"], "name": run["name"], "state": run["state"],
                     "config": run["config"], "best_val_and_step": min(vals) if vals else None,
                     "approx_epoch_train": {k: s / n for k, (s, n) in epochs.items()},
                     "run_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    root = REPO / "data_generation/objaverse-dexonomy"
    ids = json.loads((REPO / "outputs/diagnostics/sample_selection.json").read_text())["sample_ids"]
    records = {r["sample_id"]: r for r in map(json.loads,
               (root / "generated_data/samples_full_surface.jsonl").read_text().splitlines())}
    canonical = defaultdict(list)
    rows = []
    for sid in ids:
        rec = records[sid]
        with np.load(root / rec["camera_path"], allow_pickle=False) as c:
            K = c["K"].astype(float)
            T = np.diag([-1., -1., 1., 1.]) @ c["T_camera_from_object"]
        with np.load(root / rec["full_surface_path"], allow_pickle=False) as s:
            p = s["points_camera"].astype(float)
            hidden = float(np.mean(s["point_visibility"] == 0))
        obj = (p - T[:3, 3]) @ np.linalg.inv(T[:3, :3]).T
        canonical[rec["object_id"]].append(obj)
        pm = np.load(root / rec["pointmap_path"], allow_pickle=False).astype(float)
        yy, xx = np.indices(pm.shape[:2])
        valid = np.isfinite(pm).all(-1) & (pm[..., 2] > 0)
        pixels = np.stack((xx, yy), -1)[valid]
        errors = []
        for signs in ([-1, -1, 1], [1, 1, 1]):
            projected = (pm[valid] * signs) @ K.T
            errors.append(np.linalg.norm(projected[:, :2] / projected[:, 2:] - pixels, axis=-1))
        # Positive scalar SSI is exactly cancelled by subsequent bbox/radius
        # normalization. Rotation is not cancelled; bbox centering also changes.
        affine = (p - np.array([.37, -.8, 1.9])) / .23
        rows.append({"sample_id": sid, "object_id": rec["object_id"],
                     "point_count": len(p), "hidden_fraction": hidden,
                     "reprojection_p99_pixels": float(np.quantile(errors[0], .99)),
                     "wrong_sign_median_pixels": float(np.median(errors[1])),
                     "ssi_cancellation_max": float(np.max(np.abs(normalize(affine) - normalize(p)))),
                     "bbox_rotation_noncommutation_max": float(np.max(np.abs(normalize(p) - normalize(obj) @ T[:3, :3].T)))})
    result = {"runs": runs, "geometry": rows, "summary": {
        "runs": len(runs), "samples": len(rows), "objects": len(canonical),
        "cross_view_object_coordinate_max": max(float(np.max(np.abs(v[0] - p)))
            for v in canonical.values() for p in v[1:]),
        "worst_reprojection_p99_pixels": max(r["reprojection_p99_pixels"] for r in rows),
        "median_wrong_sign_pixels": float(np.median([r["wrong_sign_median_pixels"] for r in rows])),
        "ssi_cancellation_max": max(r["ssi_cancellation_max"] for r in rows),
        "median_hidden_fraction": float(np.median([r["hidden_fraction"] for r in rows]))},
        "limitations": ["Selected local subset, not population inference.",
                        "No VAE, source-mesh alignment, encoder, or GPU execution.",
                        "Epoch estimates assume 733 optimizer steps per epoch; not exact sample weights.",
                        "SSI cancellation applies to positive isotropic affine transforms only, before position side-channel."]}
    (HERE / "local_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
