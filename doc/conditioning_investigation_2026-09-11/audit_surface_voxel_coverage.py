"""Point-hit coverage of decoded target occupancy; no learned model execution.

Inputs to point voxelization are recorded surfaces and camera transforms only.
The target is used for scoring, never to construct the point grid.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

HERE = Path(__file__).resolve().parent


def main():
    bundle = HERE / "geometry_bundle"
    metadata = json.loads((bundle / "bundle_manifest.json").read_text())
    report = json.loads((bundle / "results.json").read_text())
    rows = []
    for sid, record in metadata["records"].items():
        camera_path = bundle / "data" / record["camera_path"]
        surface_path = bundle / "data" / record["full_surface_path"]
        with np.load(camera_path, allow_pickle=False) as data:
            transform = np.diag([-1., -1., 1., 1.]) @ data["T_camera_from_object"]
        with np.load(surface_path, allow_pickle=False) as data:
            points = data["points_camera"].astype(float)
        points = (points - transform[:3, 3]) @ np.linalg.inv(transform[:3, :3]).T
        assert np.isfinite(points).all()
        assert np.max(np.abs(points)) < .50001
        available = [row for run in report["runs"] for row in run["rows"]
                     if row["sample_id"] == sid and (bundle / "rollouts" / row["artifact"]).is_file()]
        target = None
        for row in available:
            with np.load(bundle / "rollouts" / row["artifact"], allow_pickle=False) as data:
                if target is None:
                    target = data["target_occupancy"].astype(bool)
                else:
                    np.testing.assert_array_equal(target, data["target_occupancy"])
        assert target is not None and target.shape == (64, 64, 64)
        indices = np.floor((np.clip(points, -.5 + 1e-6, .5 - 1e-6) + .5) * 64).astype(int)
        grid = np.zeros_like(target)
        grid[tuple(indices.T)] = True
        def metrics(g):
            intersection = int((g & target).sum())
            return {"occupied": int(g.sum()), "iou": intersection / int((g | target).sum()),
                    "precision": intersection / int(g.sum()), "recall": intersection / int(target.sum())}
        rows.append({"sample_id": sid, "point_count": len(points), "target_occupied": int(target.sum()),
                     "raw_point_hits": metrics(grid),
                     "six_neighbor_one_voxel_dilation": metrics(binary_dilation(grid)),
                     "source_sha256": {str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in [camera_path, surface_path, bundle / "rollouts" / available[0]["artifact"]]}})
    output = {"rows": rows, "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": "Five previously selected diagnostic objects, not population prevalence. Target is decoded occupancy, not independently voxelized original mesh. Uses oracle camera inversion. Coverage is not an information-theoretic bound on reconstruction; interpolation may infer unsampled surface. Dilation is a fixed geometric comparison, not an adopted preprocessing fix. No VAE execution."}
    (HERE / "surface_voxel_coverage.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(row["sample_id"][:8], row["raw_point_hits"])


if __name__ == "__main__":
    main()
