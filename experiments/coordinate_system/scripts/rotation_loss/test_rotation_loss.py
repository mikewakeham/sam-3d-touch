"""CPU checks of geometry, fixed latent scoring, and the encoder-only runner."""
import json
import csv
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("LIDRA_SKIP_INIT", "true")
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import trimesh
from experiments.coordinate_system.scripts.rotation_loss import rotation_utils as utils
from experiments.coordinate_system.scripts.rotation_loss import run_rotation_loss_gpu as runner
from experiments.coordinate_system.scripts.rotation_loss import plot_rotation_loss as plotting


class SpatialEncoder(torch.nn.Module):
    """Spatial test features only; not evidence about the pretrained SS encoder."""
    def forward(self, x):
        return {"mean": torch.nn.functional.avg_pool3d(x, 4).repeat(1, 8, 1, 1, 1)}


class RotationTests(unittest.TestCase):
    def test_native_rotations_map_physical_centers(self):
        grid = np.zeros((7, 7, 7), dtype=bool)
        location = np.array([1, 4, 5])
        grid[tuple(location)] = True
        for _, _, _, matrix in utils.native_rotations():
            rotated = utils.rotate_grid(grid, matrix)
            expected = (location-3)@matrix.T+3
            np.testing.assert_array_equal(np.argwhere(rotated)[0], expected)
            np.testing.assert_array_equal(utils.rotate_grid(rotated, matrix.T), grid)

    def test_fixed_prediction_scores_use_reencoded_references(self):
        grid = np.zeros((64, 64, 64), dtype=bool)
        grid[5:13, 20:36, 36:48] = True
        encoder = SpatialEncoder()
        zero = utils.encode(encoder, grid, torch.device("cpu"))
        rotated = utils.encode(encoder, utils.rotate_grid(grid, utils.native_rotations()[5][3]), torch.device("cpu"))
        prediction = rotated.copy()
        scores = utils.endpoint_scores(prediction, {"identity": zero, "z90": rotated})
        self.assertGreater(scores["identity"], 0.)
        self.assertEqual(scores["z90"], 0.)
        np.testing.assert_array_equal(prediction, rotated)

    def test_object_weighting_not_repeat_weighting(self):
        rows = [{"object_id": "a", "x": 1.}]*9+[{"object_id": "b", "x": 9.}]
        means = plotting.object_means(rows, "x")
        self.assertEqual(plotting.estimate(means.values())["mean"], 5.)

    def test_checkpoint_tables_and_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            losses, endpoints = [], []
            for split in ("train", "val"):
                for model, value in (("camera", 3.), ("oracle", 2.)):
                    for oid in range(4):
                        for draw in range(2):
                            losses.append(dict(model=model, split=split, object_id=str(oid),
                                surface="correct", time_kind="native", velocity_mse=value, draw=draw))
                            endpoints.append(dict(model=model, split=split, object_id=str(oid),
                                identity_mse=value, best_mse=value/2, draw=draw))
            for name, rows in (("velocity_losses.csv", losses), ("endpoint_scores.csv", endpoints)):
                with (root/name).open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            (root/"results.json").write_text(json.dumps({"models": [{"name": "camera"}, {"name": "oracle"}]}))
            summary = {}
            plotting.checkpoint_figures(root, root, summary)
            self.assertTrue((root/"native_velocity_loss.png").is_file())
            self.assertTrue((root/"endpoint_latent_scores.png").is_file())
            self.assertTrue(all(r["mean"] == -1 for r in summary["paired_model_differences"]))

    def test_encoder_runner_without_sam_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mesh = trimesh.creation.box(extents=[1, .6, .35])
            grid = utils.voxelize(mesh, stock=True)
            # Explicitly fake features, but real Open3D geometry and disk I/O.
            mean = utils.encode(SpatialEncoder(), grid, torch.device("cpu")).reshape(16, 16, 16, 8).transpose(3, 0, 1, 2)
            records = []
            for i in range(4):
                oid = str(i)
                (root/"objects"/oid).mkdir(parents=True)
                mesh.export(root/"objects"/oid/"model.obj")
                dest = root/"generated_data"/oid
                dest.mkdir(parents=True)
                np.savez(dest/"object_transform.npz", T_normalized_from_source=np.eye(4))
                np.savez(dest/"target_latent.npz", mean=mean)
                for view in range(2):
                    records.append({"object_id": oid, "sample_id": f"{oid}_{view}",
                                    "target_path": f"generated_data/{oid}/target_latent.npz"})
            (root/"samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            (root/"splits.json").write_text(json.dumps({"train": [str(i) for i in range(4)], "val": []}))
            (root/"data.yaml").write_text(f"dataset:\n  root: {root}\n  manifest: samples.jsonl\n  split_file: splits.json\n  split: train\n")
            (root/"pipeline.yaml").write_text("ss_generator_config_path: generator.yaml\n")
            (root/"generator.yaml").write_text("module:\n  generator:\n    backbone:\n      sigma_min: 0\n")
            args = SimpleNamespace(data_config=root/"data.yaml", pipeline_config=root/"pipeline.yaml",
                encoder_checkpoint=root/"unused.ckpt", train_objects=4, val_objects=0,
                angles=[0, 30, 90], seed=29, device="cpu", output_dir=root/"output", example_object="0")
            with patch.object(runner, "load_encoder", return_value=SpatialEncoder()):
                runner.run(args)
            report = json.loads((root/"output/results.json").read_text())
            self.assertTrue(report["complete"])
            rows = plotting.read_csv(root/"output/measurements.csv")
            self.assertEqual(len(rows), 4*(7+7))
            self.assertTrue(all(float(r["inverse_vertex_error"]) < 1e-14 for r in rows if r["part"] == "padded"))
            self.assertTrue(all(float(r["inverse_grid_iou"]) == 1 for r in rows if r["part"] == "native"))
            self.assertGreater(max(float(r["latent_mse"]) for r in rows), 0.)
            figures = root/"figures"
            figures.mkdir()
            summary = {}
            plotting.rotation_figures(root/"output", figures, summary)
            self.assertTrue((figures/"0_rotation_comparison.png").is_file())
            self.assertEqual(len(summary["rotation_targets"]), 14)
            self.assertFalse((root/"output/target_latents.npz").exists())


if __name__ == "__main__":
    unittest.main()
