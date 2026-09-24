"""CPU geometry, migration and training-loader checks for surface normals/pools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

from backfill_normals import backfill_object
from generate_target_latents import checkpoint_sha256, load_normalized_mesh
from make_data import object_complete, package_object, parse_args
from sample_full_surface import sam_camera_transform, transform_normals, transform_points, validate_surface
from surface_pool import make_surface_pool, select_surface_pool, validate_surface_pool


class NormalsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.generated = self.root / 'generated_data'
        self.object_dir = self.generated / 'box'
        self.object_dir.mkdir(parents=True)
        mesh = trimesh.creation.box(extents=[1, .6, .4])
        np.savez(self.object_dir / 'mesh.npz', vertices=mesh.vertices.astype(np.float32),
                 faces=mesh.faces.astype(np.int32), coordinate_frame='normalized_object')
        self.mesh = load_normalized_mesh(self.object_dir / 'mesh.npz')
        self.mesh_hash = checkpoint_sha256(self.object_dir / 'mesh.npz')
        self.args = parse_args(['--data-root', str(self.root), '--num-views', '2',
                                '--resolution', '32', '--num-points', '256', '--surface-pool-points', '1024'])
        self.saved = {}
        for index in range(2):
            view = self.object_dir / 'views' / f'{index:03d}'
            view.mkdir(parents=True)
            rotation = trimesh.transformations.rotation_matrix(index * np.pi / 2, [0, 1, 0])
            rotation[:3, 3] = [0, 0, 3]
            np.savez(view / 'camera.npz', K=np.array([[32., 0, 16], [0, 32, 16], [0, 0, 1]]),
                     T_camera_from_object=rotation)
            np.save(view / 'depth.npy', np.full((32, 32), 3, dtype=np.float32))
            rgba = np.zeros((32, 32, 4), dtype=np.uint8)
            rgba[2:-2, 2:-2] = 255
            Image.fromarray(rgba).save(view / 'image.png')
        self.records = package_object({'object_id': 'box'}, 'train', self.args)
        (self.object_dir / 'render_complete.json').write_text('{}')
        (self.object_dir / 'render_metadata.json').write_text('{}')
        np.savez(self.object_dir / 'object_transform.npz', transform=np.eye(4))
        (self.generated / 'settings.json').write_text(json.dumps({'num_views': 2, 'full_surface_format_version': 2}))
        (self.generated / 'objects.json').write_text(json.dumps([{'object_id': 'box'}]))
        self.paths = [self.root / row['full_surface_path'] for row in self.records]

    def downgrade(self):
        (self.generated / 'settings.json').write_text(json.dumps({'num_views': 2}))
        for path in self.paths:
            with np.load(path) as data:
                arrays = dict(data)
            for key in ['normals_camera', 'face_indices', 'normal_method', 'mesh_sha256',
                        'normal_numpy_version', 'normal_trimesh_version']:
                arrays.pop(key)
            arrays['format_version'] = np.int64(1)
            self.saved[path] = arrays
            np.savez_compressed(path, **arrays)
        (self.object_dir / 'surface_pool.npz').unlink()

    def test_default_generation_and_triangle_alignment(self):
        self.assertEqual(parse_args(['--data-root', str(self.root)]).surface_pool_points, 20480)
        self.assertTrue(object_complete(self.object_dir, self.args))
        reference_points = reference_normals = None
        for path in self.paths:
            with np.load(path) as data:
                validate_surface(data, require_normals=True)
                transform = sam_camera_transform(path.with_name('camera.npz'))
                normals = transform_normals(data['normals_camera'], np.linalg.inv(transform))
                expected = self.mesh.face_normals[data['face_indices']]
                np.testing.assert_allclose(normals, expected, atol=1e-6)
                points = transform_points(data['points_camera'], np.linalg.inv(transform))
                # A centered convex box must have normals pointing away from its center.
                self.assertTrue(np.all(np.sum(points * normals, axis=1) > 0))
                if reference_points is not None:
                    np.testing.assert_allclose(points, reference_points, atol=1e-6)
                    np.testing.assert_allclose(normals, reference_normals, atol=1e-6)
                reference_points, reference_normals = points, normals

    def test_backfill_preserves_every_old_array_and_resumes(self):
        self.downgrade()
        before = {path: path.read_bytes() for path in self.paths}
        result = backfill_object(self.object_dir, dry_run=True, pool_points=1024)
        self.assertEqual(result['updated'], 2)
        for path in self.paths:
            self.assertEqual(path.read_bytes(), before[path])
        self.assertFalse((self.object_dir / 'surface_pool.npz').exists())
        result = backfill_object(self.object_dir, pool_points=1024)
        self.assertEqual(result['updated'], 2)
        for path in self.paths:
            with np.load(path) as data:
                for key, value in self.saved[path].items():
                    if key != 'format_version':
                        np.testing.assert_array_equal(data[key], value)
        paths = self.paths + [self.object_dir / 'surface_pool.npz']
        timestamps = [path.stat().st_mtime_ns for path in paths]
        self.assertEqual(backfill_object(self.object_dir, pool_points=1024)['updated'], 0)
        backfill_object(self.object_dir, check_only=True, pool_points=1024)
        self.assertEqual(timestamps, [path.stat().st_mtime_ns for path in paths])

    def test_later_view_mismatch_prevents_all_writes(self):
        self.downgrade()
        arrays = dict(self.saved[self.paths[1]])
        arrays['points_camera'] = arrays['points_camera'] + np.float32(.01)
        np.savez(self.paths[1], **arrays)
        before = [path.read_bytes() for path in self.paths]
        with self.assertRaisesRegex(ValueError, 'Sampling replay differs'):
            backfill_object(self.object_dir, pool_points=1024)
        self.assertEqual(before, [path.read_bytes() for path in self.paths])
        self.assertFalse((self.object_dir / 'surface_pool.npz').exists())

    def test_overwrite_changes_only_normal_arrays(self):
        self.downgrade()
        backfill_object(self.object_dir, pool_points=1024)
        paths = self.paths + [self.object_dir / 'surface_pool.npz']
        for legacy in (False, True):
            saved = {}
            for path in paths:
                with np.load(path) as data:
                    arrays = dict(data)
                if legacy:
                    arrays.pop('normal_source')
                key = 'normals_object' if path.name == 'surface_pool.npz' else 'normals_camera'
                arrays[key][:] = np.nan
                np.savez_compressed(path, **arrays)
                saved[path] = arrays
            before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
            backfill_object(self.object_dir, pool_points=1024, overwrite_normals=True, dry_run=True)
            self.assertEqual(before, {path: path.read_bytes() for path in before})
            result = backfill_object(self.object_dir, pool_points=1024, overwrite_normals=True)
            self.assertEqual(result['updated'], 2)
            self.assertTrue(result['pool_updated'])
            self.assertFalse(result['pool_created'])
            for path, old in saved.items():
                with np.load(path) as data:
                    self.assertEqual(set(data), set(old))
                    for key, value in old.items():
                        if key not in ('normals_camera', 'normals_object'):
                            np.testing.assert_array_equal(data[key], value)
            for path in before.keys() - saved.keys():
                self.assertEqual(path.read_bytes(), before[path])
            backfill_object(self.object_dir, check_only=True, pool_points=1024)

    def test_overwrite_rejects_new_generation_and_unknown_provenance(self):
        for settings_present in (True, False):
            if not settings_present:
                (self.generated / 'settings.json').unlink()
            before = [path.read_bytes() for path in self.paths]
            with self.assertRaisesRegex(ValueError, 'Cannot identify normals'):
                backfill_object(self.object_dir, pool_points=1024, overwrite_normals=True)
            self.assertEqual(before, [path.read_bytes() for path in self.paths])

    def test_overwrite_skips_missing_data_and_rejects_changed_geometry(self):
        self.downgrade()
        before = [path.read_bytes() for path in self.paths]
        result = backfill_object(self.object_dir, pool_points=1024, overwrite_normals=True)
        self.assertEqual(result['updated'], 0)
        self.assertEqual(before, [path.read_bytes() for path in self.paths])
        self.assertFalse((self.object_dir / 'surface_pool.npz').exists())
        backfill_object(self.object_dir, pool_points=1024)
        pool_path = self.object_dir / 'surface_pool.npz'
        with np.load(pool_path) as data:
            arrays = dict(data)
        arrays['points_object'] += np.float32(.001)
        np.savez_compressed(pool_path, **arrays)
        before = [path.read_bytes() for path in self.paths + [pool_path]]
        with self.assertRaises(ValueError):
            backfill_object(self.object_dir, pool_points=1024, overwrite_normals=True)
        self.assertEqual(before, [path.read_bytes() for path in self.paths + [pool_path]])

    def test_cli_progress_resume_overwrite_and_check(self):
        self.downgrade()
        command = [sys.executable, str(Path(__file__).resolve().parent.parent / 'backfill_normals.py'),
                   '--data-root', str(self.root), '--workers', '2', '--pool-points', '1024']
        for flags, expected in (([], 'updated 2 views'), ([], 'already complete'),
                                (['--overwrite-normals'], 'pool normals overwritten'),
                                (['--check-only'], 'already complete')):
            result = subprocess.run(command + flags, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('[1/1] box:', result.stdout)
            self.assertIn(expected, result.stdout)
            self.assertIn('0 failed', result.stdout)

    def test_corrupted_normals_and_faces_are_detected(self):
        for key in ('normals_camera', 'face_indices'):
            with np.load(self.paths[0]) as data:
                original = dict(data)
            broken = dict(original)
            broken[key] = -original[key] if key == 'normals_camera' else (original[key] + 1) % len(self.mesh.faces)
            np.savez(self.paths[0], **broken)
            with self.assertRaisesRegex(ValueError, 'Saved normals or triangle mapping'):
                backfill_object(self.object_dir, pool_points=1024)
            np.savez(self.paths[0], **original)

    def test_interrupted_write_can_resume(self):
        from sample_full_surface import save_surface
        self.downgrade()
        calls = 0
        def interrupted(path, arrays, overwrite):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('interrupted')
            save_surface(path, arrays, overwrite)
        with patch('backfill_normals.save_surface', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                backfill_object(self.object_dir, pool_points=1024)
        self.assertEqual(backfill_object(self.object_dir, pool_points=1024)['updated'], 1)
        backfill_object(self.object_dir, check_only=True, pool_points=1024)

    def test_interrupted_compression_preserves_original_and_resumes(self):
        self.downgrade()
        before = [path.read_bytes() for path in self.paths]
        def interrupted(path, **arrays):
            Path(path).write_bytes(b'partial compressed file')
            raise OSError('interrupted during compression')
        with patch('sample_full_surface.np.savez_compressed', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted during compression'):
                backfill_object(self.object_dir, pool_points=1024)
        self.assertEqual(before, [path.read_bytes() for path in self.paths])
        self.assertTrue(self.paths[0].with_suffix('.tmp.npz').exists())
        backfill_object(self.object_dir, pool_points=1024)
        backfill_object(self.object_dir, check_only=True, pool_points=1024)
        self.assertFalse(self.paths[0].with_suffix('.tmp.npz').exists())

    def test_interrupted_pool_write_resumes_after_views_completed(self):
        from surface_pool import save_surface_pool
        self.downgrade()
        def interrupted_compression(temporary, **arrays):
            Path(temporary).write_bytes(b'partial pool')
            raise OSError('interrupted pool')
        def interrupted(path, arrays):
            with patch('surface_pool.np.savez_compressed', side_effect=interrupted_compression):
                save_surface_pool(path, arrays)
        with patch('backfill_normals.save_surface_pool', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted pool'):
                backfill_object(self.object_dir, pool_points=1024)
        for path in self.paths:
            with np.load(path) as data:
                validate_surface(data, require_normals=True)
        self.assertFalse((self.object_dir / 'surface_pool.npz').exists())
        self.assertTrue((self.object_dir / 'surface_pool.tmp.npz').exists())
        result = backfill_object(self.object_dir, pool_points=1024)
        self.assertEqual(result['updated'], 0)
        self.assertTrue(result['pool_created'])
        backfill_object(self.object_dir, check_only=True, pool_points=1024)
        self.assertFalse((self.object_dir / 'surface_pool.tmp.npz').exists())

    def test_pool_replay_subsets_and_corruption(self):
        pool = make_surface_pool(self.mesh, 4096, 42, self.mesh_hash)
        validate_surface_pool(pool, self.mesh, self.mesh_hash)
        points, normals, faces = select_surface_pool(pool, 512, 29)
        np.testing.assert_allclose(normals, self.mesh.face_normals[faces], atol=1e-6)
        again = select_surface_pool(pool, 512, 29)
        np.testing.assert_array_equal(points, again[0])
        self.assertEqual(len(np.unique(points, axis=0)), 512)
        with self.assertRaises(ValueError):
            select_surface_pool(pool, 5000, 29)
        pool['points_object'][0] = 0
        with self.assertRaisesRegex(ValueError, 'replay differs'):
            validate_surface_pool(pool, self.mesh, self.mesh_hash)

    def test_backfill_default_is_20480(self):
        self.downgrade()
        result = backfill_object(self.object_dir)
        self.assertEqual(result['pool_points'], 20480)
        with np.load(self.object_dir / 'surface_pool.npz') as pool:
            self.assertEqual(pool['points_object'].shape, (20480, 3))
            self.assertEqual(pool['normals_object'].shape, (20480, 3))
            validate_surface_pool(pool, self.mesh, self.mesh_hash)
        backfill_object(self.object_dir, check_only=True)

    def test_resample_pool_only_preserves_other_files_and_resumes(self):
        pool_path = self.object_dir / 'surface_pool.npz'
        # Start with the previous production pool size.
        from surface_pool import save_surface_pool
        with np.load(pool_path) as saved:
            seed = int(saved['source_sample_seed'])
        save_surface_pool(pool_path, make_surface_pool(self.mesh, 16384, seed, self.mesh_hash))
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        with self.assertRaisesRegex(ValueError, 'use --resample-pool'):
            backfill_object(self.object_dir)
        result = backfill_object(self.object_dir, resample_pool=True, dry_run=True)
        self.assertTrue(result['pool_pending'])
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        result = backfill_object(self.object_dir, resample_pool=True)
        self.assertEqual(result['updated'], 0)
        self.assertTrue(result['pool_updated'])
        for path in before:
            if path != pool_path:
                self.assertEqual(path.read_bytes(), before[path])
        with np.load(pool_path) as saved:
            self.assertEqual(saved['points_object'].shape, (20480, 3))
            validate_surface_pool(saved, self.mesh, self.mesh_hash)
            for count in (16384, 20480):
                points, normals, faces = select_surface_pool(saved, count, 29)
                self.assertEqual(len(points), count)
                np.testing.assert_allclose(normals, self.mesh.face_normals[faces], atol=1e-6)
        timestamp = pool_path.stat().st_mtime_ns
        self.assertFalse(backfill_object(self.object_dir, resample_pool=True)['pool_updated'])
        self.assertEqual(timestamp, pool_path.stat().st_mtime_ns)
        backfill_object(self.object_dir, check_only=True)

    def test_resample_interruption_preserves_old_pool(self):
        from surface_pool import save_surface_pool
        pool_path = self.object_dir / 'surface_pool.npz'
        before = pool_path.read_bytes()
        def interrupted_compression(temporary, **arrays):
            Path(temporary).write_bytes(b'partial larger pool')
            raise OSError('interrupted pool resize')
        def interrupted(path, arrays):
            with patch('surface_pool.np.savez_compressed', side_effect=interrupted_compression):
                save_surface_pool(path, arrays)
        with patch('backfill_normals.save_surface_pool', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted pool resize'):
                backfill_object(self.object_dir, resample_pool=True)
        self.assertEqual(before, pool_path.read_bytes())
        backfill_object(self.object_dir, resample_pool=True)
        self.assertFalse(pool_path.with_suffix('.tmp.npz').exists())
        backfill_object(self.object_dir, check_only=True)

    def test_resample_does_not_upgrade_old_views(self):
        self.downgrade()
        before = [path.read_bytes() for path in self.paths]
        result = backfill_object(self.object_dir, resample_pool=True)
        self.assertEqual(result['updated'], 0)
        self.assertTrue(result['pool_created'])
        self.assertEqual(before, [path.read_bytes() for path in self.paths])

    def test_resample_rejects_wrong_mesh_and_conflicting_modes(self):
        for options in ({'overwrite_normals': True}, {'check_only': True}, {'pool_points': 0}):
            with self.assertRaisesRegex(ValueError, '--resample-pool requires'):
                backfill_object(self.object_dir, resample_pool=True, **options)
        path = self.object_dir / 'surface_pool.npz'
        with np.load(path) as saved:
            arrays = dict(saved)
        arrays['mesh_sha256'] = 'different mesh'
        np.savez_compressed(path, **arrays)
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'different mesh'):
            backfill_object(self.object_dir, resample_pool=True)
        self.assertEqual(before, path.read_bytes())

    def test_resample_cli_progress_and_resume(self):
        command = [sys.executable, str(Path(__file__).resolve().parent.parent / 'backfill_normals.py'),
                   '--data-root', str(self.root), '--workers', '2', '--resample-pool']
        for expected in ('resampled to 20480 points', 'already complete'):
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('[1/1] box:', result.stdout)
            self.assertIn(expected, result.stdout)

    def test_inverse_transpose_and_translation(self):
        transform = np.diag([2., 3., 4., 1.])
        transform[:3, 3] = [20, 30, 40]
        normal = np.array([[1., 1., 0.]]) / np.sqrt(2)
        result = transform_normals(normal, transform)
        tangent = np.array([[1., -1., 0.]]) @ transform[:3, :3].T
        self.assertAlmostEqual(float(np.sum(result * tangent)), 0, places=6)
        transform[:3, 3] = 0
        np.testing.assert_array_equal(result, transform_normals(normal, transform))
        with self.assertRaises(ValueError):
            transform_normals(np.zeros((1, 3)), transform)

    def test_loader_both_formats_and_aligned_normals(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from dataloader import TouchDataset, collate_touch_batch
        for row in self.records:
            row['target_path'] = 'target.npz'
        np.savez(self.root / 'target.npz', mean=np.zeros((8, 16, 16, 16), dtype=np.float32))
        (self.root / 'samples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.records))
        (self.root / 'splits.json').write_text(json.dumps({'train': ['box'], 'val': [], 'test': []}))
        config = {'seed': 29, 'dataset': {'root': str(self.root), 'manifest': 'samples.jsonl',
                  'split_file': 'splits.json', 'split': 'train'}, 'touch': {'source': 'full_surface'}}
        dataset = TouchDataset(config)
        expected = dataset[0]['touch_xyz'].clone()
        self.downgrade()
        np.testing.assert_array_equal(dataset[0]['touch_xyz'], expected)
        config['touch']['include_normals'] = True
        with self.assertRaisesRegex(ValueError, 'no normals'):
            TouchDataset(config)[0]
        backfill_object(self.object_dir, pool_points=1024)
        dataset = TouchDataset(config)
        batch = collate_touch_batch([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch['touch_normals'].shape), (2, 256, 3))
        np.testing.assert_array_equal(batch['touch_xyz'][0], expected)
        config['touch']['pool_points'] = 512
        dataset = TouchDataset(config)
        first, second = dataset[0], dataset[1]
        self.assertEqual(tuple(first['touch_xyz'].shape), (512, 3))
        for key, transform_fn in [('touch_xyz', transform_points), ('touch_normals', transform_normals)]:
            a = transform_fn(first[key].numpy(), np.linalg.inv(sam_camera_transform(self.paths[0].with_name('camera.npz'))))
            b = transform_fn(second[key].numpy(), np.linalg.inv(sam_camera_transform(self.paths[1].with_name('camera.npz'))))
            np.testing.assert_allclose(a, b, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
