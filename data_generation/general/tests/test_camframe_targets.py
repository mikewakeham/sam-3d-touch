import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_target_latents as targets
from dataloader import TouchDataset
from sample_full_surface import sam_camera_transform, transform_points


class SmallEncoder(torch.nn.Module):
    def forward(self, occupancy):
        return {"mean": torch.nn.functional.avg_pool3d(occupancy, 4).repeat(1, 8, 1, 1, 1)}


class ImmediatePool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, *args):
        from concurrent.futures import Future
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future


class CameraTargetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.generated = self.root / "generated_data"
        self.object_dir = self.generated / "object_a"
        self.object_dir.mkdir(parents=True)
        self.mesh = trimesh.Trimesh(
            vertices=[[-.5, -.3, -.2], [.5, -.3, -.2], [-.2, .3, -.2], [.1, .1, .2]],
            faces=[[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], process=False)
        np.savez(self.object_dir / "mesh.npz", vertices=self.mesh.vertices,
                 faces=self.mesh.faces, coordinate_frame="normalized_object")
        np.savez(self.object_dir / "object_transform.npz", T_normalized_from_source=np.eye(4))
        targets.save_target(np.zeros((8, 16, 16, 16), dtype=np.float32),
                            self.object_dir / "target_latent.npz")
        self.records = []
        for index, angle in enumerate([0, .8]):
            view = self.object_dir / "views" / f"{index:03d}"
            view.mkdir(parents=True)
            transform = trimesh.transformations.rotation_matrix(angle, [1, 2, 3])
            transform[:3, 3] = [.2, -.4, 3]
            np.savez(view / "camera.npz", K=np.eye(3),
                     T_camera_from_object=np.diag([-1., -1., 1., 1.]) @ transform)
            np.save(view / "depth.npy", np.ones((4, 4), dtype=np.float32))
            Image.fromarray(np.full((4, 4, 4), 255, dtype=np.uint8)).save(view / "image.png")
            points = transform_points(self.mesh.vertices, transform).astype(np.float32)
            np.savez(view / "full_surface.npz", points_camera=points, format_version=1,
                     data_kind="full_surface", coordinate_frame="sam_camera",
                     point_ids=np.arange(len(points)), point_visibility=np.zeros(len(points), dtype=int),
                     visibility_tolerance=.005)
            (view / "touches.npz").write_bytes(b"existing touch data")
            row = dict(sample_id=f"object_a_{index:03d}", object_id="object_a",
                       view_id=f"{index:03d}", split="train", touch_path=None,
                       mesh_path="generated_data/object_a/mesh.npz",
                       target_path="generated_data/object_a/target_latent.npz")
            for key, filename in [("camera_path", "camera.npz"), ("image_path", "image.png"),
                                  ("depth_path", "depth.npy"), ("full_surface_path", "full_surface.npz")]:
                row[key] = str((view / filename).relative_to(self.root))
            self.records.append(row)
        self.write_rows("samples.jsonl", self.records)
        touches = [dict(row, touch_path=str(Path(row["camera_path"]).with_name("touches.npz")))
                   for row in self.records]
        self.write_rows("samples_simulated_touches.jsonl", touches)
        (self.generated / "splits.json").write_text(json.dumps(dict(train=["object_a"], val=[], test=[])))
        (self.generated / "settings.json").write_text('{"seed": 29}')
        (self.generated / "target_latents.json").write_text('{"coordinate_frame": "normalized_object"}')
        self.checkpoint = self.root / "encoder.ckpt"
        self.checkpoint.write_bytes(b"test encoder")

    def write_rows(self, name, rows):
        (self.generated / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def run_camera(self, *extra):
        generate = targets.generate_camera_target
        def cpu_generate(row, root, mesh, encoder, device):
            return generate(row, root, mesh, encoder, "cpu")
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0,1"), \
                patch("concurrent.futures.ProcessPoolExecutor", ImmediatePool), \
                patch("torch.cuda.is_available", return_value=True), \
                patch.object(targets, "load_encoder", return_value=SmallEncoder()), \
                patch.object(targets, "generate_camera_target", side_effect=cpu_generate):
            targets.main(["--target-frame", "camera", "--data-root", str(self.root),
                          "--encoder-checkpoint", str(self.checkpoint), *extra])

    def test_frame_roundtrip_and_camera_translation_invariance(self):
        rng = np.random.default_rng(29)
        path = self.root / self.records[0]["camera_path"]
        for _ in range(30):
            camera = trimesh.transformations.rotation_matrix(rng.uniform(-np.pi, np.pi), rng.normal(size=3))
            object_rotation = trimesh.transformations.rotation_matrix(rng.uniform(-np.pi, np.pi), rng.normal(size=3))
            camera = camera @ object_rotation
            camera[:3, 3] = rng.normal(size=3)
            np.savez(path, T_camera_from_object=np.diag([-1., -1., 1., 1.]) @ camera)
            mesh, metadata = targets.camera_target_mesh(self.mesh, path)
            np.testing.assert_allclose(mesh.bounds.mean(0), 0, atol=1e-14)
            self.assertAlmostEqual(mesh.extents.max(), 1)
            np.testing.assert_allclose(metadata["T_camera_from_target"] @ metadata["T_target_from_object"], camera, atol=1e-12)
            np.testing.assert_allclose(transform_points(mesh.vertices, metadata["T_camera_from_target"]),
                                       transform_points(self.mesh.vertices, camera), atol=1e-12)
            camera[:3, 3] += [2, 4, 8]
            np.savez(path, T_camera_from_object=np.diag([-1., -1., 1., 1.]) @ camera)
            moved, _ = targets.camera_target_mesh(self.mesh, path)
            np.testing.assert_array_equal(mesh.vertices, moved.vertices)

    def test_identity_and_invalid_camera(self):
        path = self.root / self.records[0]["camera_path"]
        mesh, _ = targets.camera_target_mesh(self.mesh, path)
        np.testing.assert_array_equal(mesh.vertices, self.mesh.vertices)
        for bad in [np.diag([2., 1., 1., 1.]), np.diag([-1., 1., 1., 1.]), np.full((4, 4), np.nan)]:
            np.savez(path, T_camera_from_object=bad)
            with self.assertRaises(ValueError):
                targets.camera_target_mesh(self.mesh, path)

    def test_pose_loader_uses_saved_camera_target_and_rejects_mismatch(self):
        from dataloader import collate_touch_batch
        rows, expected = [], []
        for row in self.records:
            _, metadata = targets.camera_target_mesh(self.mesh, self.root / row['camera_path'])
            path = self.root / targets.camera_target_path(row)
            targets.save_target(np.zeros((8, 16, 16, 16), dtype=np.float32), path, **metadata)
            rows.append(dict(row, target_path=str(path.relative_to(self.root))))
            expected.append(metadata['T_camera_from_target'])
        self.write_rows('samples_camframe.jsonl', rows)
        config = dict(dataset=dict(root=str(self.root), manifest='generated_data/samples_camframe.jsonl',
                                   split_file='generated_data/splits.json', split='train', target_frame='camera'),
                      touch=dict(source='full_surface'))
        dataset = TouchDataset(config, include_touch=False, include_pose_targets=True)
        batch = collate_touch_batch([dataset[0], dataset[1]])
        np.testing.assert_allclose(batch['camera_from_target'].numpy(), expected, atol=1e-6)
        self.assertEqual(batch['target_shape'].shape, (2, 4096, 8))
        self.assertNotIn('touch_xyz', batch)
        self.assertNotIn('camera_from_target', TouchDataset(config, include_touch=False)[0])
        path = self.root / rows[0]['target_path']
        with np.load(path) as data:
            broken = dict(data)
        broken['T_camera_from_target'] = broken['T_camera_from_target'].copy()
        broken['T_camera_from_target'][0, 3] += .25
        np.savez(path, **broken)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            dataset[0]
        config['dataset']['target_frame'] = 'object'
        with self.assertRaisesRegex(ValueError, 'target_frame'):
            TouchDataset(config, include_touch=False, include_pose_targets=True)

    def test_encoding_manifests_loader_and_original_files(self):
        before = {p: targets.checkpoint_sha256(p) for p in self.root.rglob("*") if p.is_file()}
        self.run_camera()
        for path, digest in before.items():
            self.assertEqual(targets.checkpoint_sha256(path), digest, str(path))
        for name in ["samples", "samples_simulated_touches"]:
            original = [json.loads(line) for line in (self.generated / (name + ".jsonl")).read_text().splitlines()]
            new = [json.loads(line) for line in (self.generated / (name + "_camframe.jsonl")).read_text().splitlines()]
            for old, row in zip(original, new):
                self.assertEqual(row, dict(old, target_path=str(targets.camera_target_path(old))))
        dataset = TouchDataset(dict(dataset=dict(root=str(self.root), manifest="generated_data/samples_camframe.jsonl",
                                                split_file="generated_data/splits.json", split="train"),
                                    touch=dict(source="full_surface")))
        for index, row in enumerate(self.records):
            mean = targets.validate_target(self.root / targets.camera_target_path(row), camera_frame=True)
            np.testing.assert_array_equal(dataset[index]["target_shape"], mean.transpose(1, 2, 3, 0).reshape(4096, 8))
            self.assertEqual(dataset[index]["touch_xyz"].shape, (4, 3))
        self.assertFalse(torch.equal(dataset[0]["target_shape"], dataset[1]["target_shape"]))

    def test_resume_skips_meshes_and_encoder_and_repairs_missing_view(self):
        self.run_camera()
        paths = [self.root / targets.camera_target_path(row) for row in self.records]
        before = paths[0].read_bytes()
        with patch.object(targets, "load_encoder", side_effect=AssertionError("loaded encoder")), \
                patch.object(targets, "load_normalized_mesh", side_effect=AssertionError("loaded mesh")):
            targets.encode_camera_objects([self.records], self.root, self.checkpoint, "0")
        paths[1].unlink()
        self.run_camera()
        self.assertEqual(paths[0].read_bytes(), before)
        self.assertTrue(paths[1].is_file())
        paths[1].write_bytes(b"")
        self.run_camera()
        targets.validate_target(paths[1], camera_frame=True)

    def test_subset_then_resume_and_changed_encoder_rejected(self):
        import shutil
        destination = self.generated / "object_b"
        shutil.copytree(self.object_dir, destination)
        extra_rows = [json.loads(json.dumps(row).replace("object_a", "object_b")) for row in self.records]
        self.records += extra_rows
        self.write_rows("samples.jsonl", self.records)
        self.write_rows("samples_simulated_touches.jsonl", [
            dict(row, touch_path=str(Path(row["camera_path"]).with_name("touches.npz"))) for row in self.records])
        self.run_camera("--limit", "1")
        self.assertFalse((self.generated / "samples_camframe.jsonl").exists())
        self.assertFalse((self.root / targets.camera_target_path(extra_rows[0])).exists())
        self.run_camera()
        self.assertTrue((self.root / targets.camera_target_path(extra_rows[0])).is_file())
        self.checkpoint.write_bytes(b"different encoder")
        with self.assertRaisesRegex(ValueError, "settings or source manifest"):
            self.run_camera()

    def test_failure_preserves_completed_files_without_publishing(self):
        camera = self.root / self.records[1]["camera_path"]
        original = camera.read_bytes()
        np.savez(camera, T_camera_from_object=np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            self.run_camera()
        self.assertTrue((self.root / targets.camera_target_path(self.records[0])).is_file())
        self.assertFalse((self.generated / "samples_camframe.jsonl").exists())
        camera.write_bytes(original)
        self.run_camera()
        self.assertTrue((self.generated / "samples_camframe.jsonl").is_file())

    def test_inconsistent_touch_manifest_does_not_publish(self):
        self.write_rows("samples_simulated_touches.jsonl", [])
        with self.assertRaisesRegex(ValueError, "same samples"):
            self.run_camera()
        self.assertFalse((self.generated / "samples_camframe.jsonl").exists())

    def test_original_entrypoint_and_canonical_target_unchanged(self):
        import make_data
        with patch.object(make_data, "main") as main:
            targets.main(["--data-root", str(self.root)])
            main.assert_called_once_with(["--stage", "latents", "--data-root", str(self.root)])
        with patch.object(make_data, "main") as main:
            targets.main(["--target-frame=object", "--data-root", str(self.root)])
            main.assert_called_once_with(["--stage", "latents", "--data-root", str(self.root)])
        path = self.object_dir / "target_latent.npz"
        original = path.read_bytes()
        _, created = targets.generate_target("object_a", self.root, SmallEncoder(), "cpu", False)
        self.assertFalse(created)
        self.assertEqual(path.read_bytes(), original)
        targets.generate_target("object_a", self.root, SmallEncoder(), "cpu", True)
        targets.validate_target(path)
        with np.load(path) as saved:
            self.assertEqual(saved.files, ["mean"])

    def test_cli_resume_with_spawned_worker(self):
        import subprocess
        from make_data import save_json

        checkpoint = targets.checkpoint_sha256(self.checkpoint)
        metadata = dict(coordinate_frame="normalized_sam_camera", normalization="aabb_center_longest_extent",
                        voxel_resolution=64, latent_shape=[8, 16, 16, 16], format_version=1,
                        encoder_checkpoint_sha256=checkpoint,
                        source_manifest_sha256=targets.checkpoint_sha256(self.generated / "samples.jsonl"))
        for name in ("settings.json", "target_latents.json"):
            metadata[name] = targets.checkpoint_sha256(self.generated / name)
        save_json(metadata, self.generated / "target_latents_camframe.json")
        for row in self.records:
            _, frame = targets.camera_target_mesh(self.mesh, self.root / row["camera_path"])
            targets.save_target(np.zeros((8, 16, 16, 16), dtype=np.float32),
                                self.root / targets.camera_target_path(row), **frame)
        result = subprocess.run([sys.executable, str(Path(targets.__file__).resolve()),
                                 "--target-frame=camera", "--data-root", str(self.root),
                                 "--encoder-checkpoint", str(self.checkpoint)],
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("2 reused, 0 encoded", result.stdout)
        self.assertTrue((self.generated / "samples_camframe.jsonl").is_file())

    def test_invalid_encoding_is_not_published(self):
        row = self.records[0]
        with patch.object(SmallEncoder, "forward", return_value={"mean": torch.full((1, 8, 16, 16, 16), float("nan"))}):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                targets.generate_camera_target(row, self.root, self.mesh, SmallEncoder(), "cpu")
        path = self.root / targets.camera_target_path(row)
        self.assertFalse(path.exists())
        targets.generate_camera_target(row, self.root, self.mesh, SmallEncoder(), "cpu")
        targets.validate_target(path, camera_frame=True)

    def test_source_manifest_change_and_duplicate_views_rejected(self):
        self.run_camera()
        self.write_rows("samples.jsonl", self.records[:1])
        with self.assertRaisesRegex(ValueError, "settings or source manifest"):
            self.run_camera()
        duplicate = dict(self.records[0], sample_id="other_sample")
        self.write_rows("samples.jsonl", [self.records[0], duplicate])
        with self.assertRaisesRegex(ValueError, "own view directory"):
            self.run_camera()

    def test_without_touch_manifest_and_explicit_overwrite(self):
        (self.generated / "samples_simulated_touches.jsonl").unlink()
        self.run_camera()
        self.assertFalse((self.generated / "samples_simulated_touches_camframe.jsonl").exists())
        path = self.root / targets.camera_target_path(self.records[0])
        path.write_bytes(b"replace me")
        self.run_camera("--overwrite")
        targets.validate_target(path, camera_frame=True)

    def test_real_small_vae_camera_target(self):
        from sparse_structure_vae import SparseStructureEncoderTdfyWrapper
        encoder = SparseStructureEncoderTdfyWrapper(in_channels=1, latent_channels=8, channels=[32, 32, 32],
                                                   num_res_blocks=1, num_res_blocks_middle=1,
                                                   sample_posterior=False, return_raw=True).eval()
        row = self.records[1]
        path = targets.generate_camera_target(row, self.root, self.mesh, encoder, "cpu")
        mean = targets.validate_target(path, camera_frame=True)
        self.assertEqual(mean.shape, (8, 16, 16, 16))
        with np.load(path) as saved:
            np.testing.assert_allclose(saved["T_camera_from_target"] @ saved["T_target_from_object"],
                                       sam_camera_transform(self.root / row["camera_path"]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
