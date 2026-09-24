"""CPU checks for geometry, saved data, and the existing 8192-point training input."""

import ast
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import trimesh

from make_data import package_object, parse_args as data_args
from make_touch_data import main, make_object, parse_args
from sample_full_surface import sam_camera_transform, transform_points
from sample_touch_patches import adaptive_region, farthest_centers, sample_region, select_patch_indices

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


class GeometryTests(unittest.TestCase):
    def test_wall_corner_and_sparse_support(self):
        rng = np.random.default_rng(29)
        points = np.column_stack([rng.uniform(-.08, .08, (2000, 2)), np.zeros(2000)])
        normals = np.tile([0., 0., 1.], (len(points), 1))
        basis, radii, _ = adaptive_region(points, normals, np.zeros(3), .04, .2)
        np.testing.assert_allclose(radii, [.008, .04, .04])
        self.assertAlmostEqual(abs(basis[2, 0]), 1)
        corner = np.concatenate([points, points[:, [0, 2, 1]]])
        normals = np.concatenate([normals, normals[:, [0, 2, 1]]])
        _, radii, _ = adaptive_region(corner, normals, np.zeros(3), .04, .2)
        np.testing.assert_allclose(radii, [.04] * 3)
        _, radii, _ = adaptive_region(points[:3], normals[:3], np.zeros(3), .04, .2)
        np.testing.assert_allclose(radii, [.04] * 3)

    def test_large_crossing_triangle_and_uniform_disk(self):
        # No triangle vertex is anywhere near the patch; area is sampled without subdivision.
        mesh = trimesh.Trimesh(vertices=[[-10, -10, 0], [10, -10, 0], [0, 10, 0]],
                               faces=[[0, 1, 2]], process=False)
        before = (mesh.vertices.copy(), mesh.faces.copy())
        points, faces = sample_region(mesh, np.zeros(3), np.eye(3), [.04, .04, .008], 20000, 29)
        np.testing.assert_array_equal(faces, 0)
        np.testing.assert_array_equal(points[:, 2], 0)
        radial = np.sum(points[:, :2] ** 2, axis=1) / .04 ** 2
        self.assertLessEqual(radial.max(), 1 + 1e-10)
        self.assertAlmostEqual(radial.mean(), .5, delta=.01)
        self.assertLess(np.abs(points.mean(axis=0)).max(), .001)
        for original, current in zip(before, (mesh.vertices, mesh.faces)):
            np.testing.assert_array_equal(original, current)

    def test_original_face_membership_and_area_weighting(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [.01, 0, 0], [0, .01, 0],
                                        [-.02, 0, 0], [0, -.02, 0]],
                               faces=[[0, 1, 2], [0, 3, 4]], process=False)
        points, faces = sample_region(mesh, np.zeros(3), np.eye(3), [.1] * 3, 10000, 29)
        self.assertAlmostEqual(np.mean(faces == 1), .8, delta=.02)
        triangles = mesh.triangles[faces]
        barycentric = trimesh.triangles.points_to_barycentric(triangles, points)
        self.assertGreaterEqual(barycentric.min(), -1e-10)
        np.testing.assert_allclose((triangles * barycentric[..., None]).sum(axis=1), points, atol=1e-12)
        again, again_faces = sample_region(mesh, np.zeros(3), np.eye(3), [.1] * 3, 10000, 29)
        np.testing.assert_array_equal(points, again)
        np.testing.assert_array_equal(faces, again_faces)

    def test_seeded_fps_prefix(self):
        points = np.random.default_rng(4).normal(size=(1000, 3))
        all_centers = farthest_centers(points, 32, 29)
        np.testing.assert_array_equal(all_centers[:8], farthest_centers(points, 8, 29))
        self.assertEqual(len(np.unique(all_centers)), 32)
        for index in range(1, 32):
            distances = np.sum((points[:, None] - points[all_centers[:index]]) ** 2, axis=2).min(axis=1)
            self.assertEqual(distances.argmax(), all_centers[index])
        with self.assertRaises(ValueError):
            farthest_centers(points[:2], 32, 29)

    def test_corner_sampling_keeps_both_sides(self):
        mesh = trimesh.creation.box()
        points, faces = sample_region(mesh, np.array([.5, .5, 0]), np.eye(3), [.04] * 3, 4096, 29)
        on_x = np.isclose(points[:, 0], .5)
        on_y = np.isclose(points[:, 1], .5)
        self.assertTrue((on_x | on_y).all())
        self.assertAlmostEqual(on_x.mean(), .5, delta=.03)
        self.assertAlmostEqual(on_y.mean(), .5, delta=.03)
        self.assertEqual(len(np.unique(mesh.face_normals[faces], axis=0)), 2)


class IntegrationTests(unittest.TestCase):
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
        self.mesh = mesh
        args = data_args(['--data-root', str(self.root), '--num-views', '2', '--num-points', '8192',
                          '--surface-pool-points', '0', '--resolution', '32'])
        for index in range(2):
            view = self.object_dir / 'views' / f'{index:03d}'
            view.mkdir(parents=True)
            rotation = trimesh.transformations.rotation_matrix(index * np.pi / 2, [0, 1, 0])
            rotation[:3, 3] = [0, 0, 3]
            np.savez(view / 'camera.npz', K=np.array([[32., 0, 16], [0, 32, 16], [0, 0, 1]]),
                     T_camera_from_object=rotation)
            np.save(view / 'depth.npy', np.full((32, 32), 3, dtype=np.float32))
            rgba = np.zeros((32, 32, 4), np.uint8)
            rgba[2:-2, 2:-2] = 255
            Image.fromarray(rgba).save(view / 'image.png')
        self.records = package_object({'object_id': 'box'}, 'train', args)
        target = self.object_dir / 'target.npz'
        np.savez(target, mean=np.zeros((8, 16, 16, 16), np.float32))
        for record in self.records:
            record['target_path'] = str(target.relative_to(self.root))
        self.manifest = self.generated / 'samples.jsonl'
        self.manifest.write_text(''.join(json.dumps(record) + '\n' for record in self.records))
        (self.generated / 'splits_train_val.json').write_text(json.dumps({'train': ['box'], 'val': []}))
        self.args = parse_args(['--data-root', str(self.root)])

    def test_generation_loader_prefixes_and_source_immutability(self):
        from dataloader import TouchDataset, collate_touch_batch
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        main(['--data-root', str(self.root)])
        output = self.generated / 'samples_simulated_touches.jsonl'
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        for record, original in zip(rows, self.records):
            self.assertEqual({**record, 'touch_path': original['touch_path']}, original)
            with np.load(self.root / record['touch_path']) as bank:
                self.assertEqual(bank['points_camera'].shape, (32768, 3))
                np.testing.assert_array_equal(bank['point_visibility'][bank['offsets'][:-1]], 0)
                transform = sam_camera_transform(self.root / record['camera_path'])
                points = transform_points(bank['points_camera'], np.linalg.inv(transform))
                triangles = self.mesh.triangles[bank['face_indices']]
                barycentric = trimesh.triangles.points_to_barycentric(triangles, points)
                self.assertGreaterEqual(barycentric.min(), -1e-5)
                np.testing.assert_allclose((triangles * barycentric[..., None]).sum(axis=1), points, atol=2e-7)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        path = self.root / rows[0]['touch_path']
        stamp = path.stat().st_mtime_ns
        make_object((self.args, self.records))
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        selected = []
        for count, per_contact in [(32, 256), (16, 512), (8, 1024)]:
            import yaml
            config = yaml.safe_load((REPO / f'configs/data_zeroverse_5000_8views_touch_{count}x{per_contact}.yaml').read_text())
            config['dataset']['root'] = str(self.root)
            dataset = TouchDataset(config)
            batch = collate_touch_batch([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch['touch_xyz'].shape), (2, 8192, 3))
            self.assertTrue(batch['touch_mask'].all())
            self.assertEqual(tuple(batch['target_shape'].shape), (2, 4096, 8))
            selected.append(batch['touch_xyz'][0].numpy().reshape(count, per_contact, 3))
        np.testing.assert_array_equal(selected[0][:16], selected[1][:, :256])
        np.testing.assert_array_equal(selected[1][:8], selected[2][:, :512])
        # Pilots cannot replace the complete manifest.
        content = output.read_bytes()
        main(['--data-root', str(self.root), '--object-id', 'box'])
        self.assertEqual(output.read_bytes(), content)

    def test_viewer_uses_identical_loader_points_and_region_boundaries(self):
        from dataloader import TouchDataset
        from view_data import load_touch_view, parse_args as viewer_args
        make_object((self.args, self.records))
        path = (self.root / self.records[0]['full_surface_path']).with_name('simulated_touches.npz')
        for count, per_contact in [(32, 256), (16, 512), (8, 1024)]:
            args = viewer_args(['--data-root', str(self.root), '--object-id', 'box',
                                '--contacts', str(count)])
            points, colors, centers, lines = load_touch_view(args)
            dataset = object.__new__(TouchDataset)
            dataset.contact_count, dataset.points_per_contact = count, per_contact
            np.testing.assert_array_equal(points, dataset.load_touch_patches(path))
            self.assertEqual(colors.shape, points.shape)
            np.testing.assert_array_equal(centers, points[::per_contact])
            self.assertEqual(lines.shape, (count * 3 * 64, 2, 3))
            with np.load(path) as saved:
                for index, segments in enumerate(lines.reshape(count, -1, 2, 3)):
                    local = (segments.reshape(-1, 3) - centers[index]) @ saved['bases_camera'][index] / saved['radii'][index]
                    np.testing.assert_allclose(np.sum(local ** 2, axis=1), 1, atol=1e-5)

    def test_evaluation_hidden_fraction_and_saved_joint_view(self):
        import torch
        from dataloader import TouchDataset
        from evaluation.evaluate import hidden_fraction, save_generated_artifacts
        from view_data import load_joint_view
        make_object((self.args, self.records))
        path = (self.root / self.records[0]['full_surface_path']).with_name('simulated_touches.npz')
        dataset = object.__new__(TouchDataset)
        dataset.root, dataset.contact_count, dataset.points_per_contact = self.root, 32, 256
        touch = dataset.load_touch_patches(path)
        fraction = hidden_fraction(dict(touch_path=str(path)), dataset,
                                   dict(touch=dict(source='touch_patches', contacts=dict(count=32),
                                                   point_sampling=dict(points_per_contact=256))))
        with np.load(path) as data:
            selected = select_patch_indices(data, 32, 256)
            self.assertEqual(fraction, np.mean(data['point_visibility'][selected] == 0))
        # An actual evaluation artifact writer -> viewer round trip, without model inference.
        camera_points = np.concatenate([touch[:64] + .1, touch])
        tensor = torch.zeros(1)
        mesh = trimesh.creation.box()
        paths = save_generated_artifacts(
            self.root, 'test', 'box_000', tensor, tensor, tensor, tensor, tensor, 1,
            SimpleNamespace(coords=tensor, feats=tensor), mesh, mesh, mesh, np.eye(4), np.eye(4),
            0., 1., 0., touch, touch, camera_points, 29, 10,
            joint_input=dict(encoder_input_camera=camera_points, touch_count=np.int64(8192)))
        args = SimpleNamespace(joint_input=paths[0], contacts=32, input_device='cpu')
        points, indices = load_joint_view(args, touch)
        self.assertEqual(points.shape, (8192, 3))
        np.testing.assert_array_equal(points, camera_points[indices + 64])
        with self.assertRaisesRegex(ValueError, 'does not match'):
            load_joint_view(args, touch + .01)

    def test_v1_read_only_replay_and_mismatch(self):
        for record in self.records:
            path = self.root / record['full_surface_path']
            with np.load(path) as saved:
                arrays = dict(saved)
            for key in ['normals_camera', 'face_indices', 'normal_method', 'mesh_sha256']:
                arrays.pop(key, None)
            arrays['format_version'] = np.int64(1)
            np.savez_compressed(path, **arrays)
        before = [(self.root / r['full_surface_path']).read_bytes() for r in self.records]
        make_object((self.args, self.records))
        self.assertEqual(before, [(self.root / r['full_surface_path']).read_bytes() for r in self.records])
        arrays['points_camera'] += .01
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'Sampling replay differs'):
            make_object((self.args, self.records))

    def test_live_joint_viewer_budgets_toggles_and_cache(self):
        import torch
        import view_data
        make_object((self.args, self.records))
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        args = view_data.parse_args(['--data-root', str(self.root), '--object-id', 'box',
                                     '--joint-pointmap'])
        mesh, rgba, pointmap, surface, K, pool = view_data.load_view(args)
        # Supply a small processed pointmap, exercising the real combination and CPU FPS.
        inputs = dict(pointmap=torch.from_numpy(pointmap[::4, ::4].copy()).permute(2, 0, 1)[None],
                      mask=torch.from_numpy((rgba[::4, ::4, 3] / 255).copy())[None, None])
        inputs['pointmap'][0, :, 1, 1] = float('nan')
        prepared = (SimpleNamespace(normalize_pointmap=False), inputs)
        valid = (inputs['mask'][0, 0] > .5) & inputs['pointmap'][0].isfinite().all(dim=0)
        expected_pointmap = inputs['pointmap'][0].permute(1, 2, 0)[valid].numpy()

        class Handle(SimpleNamespace):
            def on_update(self, callback):
                self.callback = callback
                return callback

            def on_click(self, callback):
                return callback

        gui, scene = {}, {}

        def add_gui(label, **kwargs):
            handle = Handle(value=kwargs.pop('initial_value', None), **kwargs)
            gui[label] = handle
            return handle

        def add_scene(name, **kwargs):
            handle = Handle(**{'visible': True, **kwargs})
            scene[name] = handle
            return handle

        server = Mock()
        server.atomic.side_effect = nullcontext
        for method in ('add_checkbox', 'add_dropdown', 'add_button'):
            getattr(server.gui, method).side_effect = add_gui
        server.gui.add_markdown.side_effect = lambda text: add_gui('counts', content=text)
        for method in ('add_mesh_simple', 'add_point_cloud', 'add_line_segments', 'add_camera_frustum'):
            getattr(server.scene, method).side_effect = add_scene
        # This unnormalized fixture never calls SSI; avoid its unavailable local GPU dependencies.
        transforms = ModuleType('sam3d_objects.data.dataset.tdfy.img_and_mask_transforms')
        transforms._apply_metric_to_ssi = Mock(side_effect=AssertionError('Unexpected SSI call'))
        with patch.dict(sys.modules, {transforms.__name__: transforms}), \
                patch.object(view_data, 'prepare_joint_preview', return_value=prepared) as prepare, \
                patch.object(view_data, 'load_joint_cloud', wraps=view_data.load_joint_cloud) as compute:
            view_data.build_viewer(server, args, mesh, rgba, pointmap, surface, K, pool)
            selection = gui['Touch patch budget']
            for count in (32, 16, 8, 32):
                selection.value = f'{count} x {8192 // count} = 8192'
                selection.callback(None)
                expected_touch = view_data.load_touch_view(args)[0]
                np.testing.assert_array_equal(scene['/joint_before'].points[:len(expected_pointmap)], expected_pointmap)
                np.testing.assert_array_equal(scene['/joint_before'].points[-8192:], expected_touch)
                self.assertEqual(scene['/joint_fps'].points.shape, (8192, 3))
                self.assertEqual(scene['/joint_fps'].colors.shape, (8192, 3))
                self.assertIn('= 8,192', gui['counts'].content)
                self.assertFalse(selection.disabled)
            self.assertEqual(compute.call_count, 3)
            prepare.assert_called_once()
        toggle = gui['Joint after FPS (8192 points)']
        toggle.value = False
        toggle.callback(None)
        self.assertFalse(scene['/joint_fps'].visible)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_cli_parallel_worker_reproducibility(self):
        main(['--data-root', str(self.root)])
        rows = make_object((self.args, self.records))
        before = [(self.root / row['touch_path']).read_bytes() for row in rows]
        subprocess.run([sys.executable, str(Path(__file__).with_name('make_touch_data.py')),
                        '--data-root', str(self.root), '--workers', '2'], check=True, capture_output=True)
        self.assertEqual(before, [(self.root / row['touch_path']).read_bytes() for row in rows])

    def test_interrupt_stops_workers_releases_lock_and_allows_restart(self):
        # Exercise the real pool/lock lifecycle with one busy and one idle worker.
        driver = self.root / 'interrupt_driver.py'
        driver.write_text(f'''import os
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
import make_touch_data as generator

def initialize():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    Path({str(self.root)!r}, f"worker-{{os.getpid()}}").touch()

def slow_object(task):
    Path({str(self.root)!r}, "busy").touch()
    time.sleep(60)
    raise RuntimeError("Test should interrupt before this point")

generator.initialize_worker = initialize
generator.make_object = slow_object
if __name__ == "__main__":
    try:
        generator.main()
    except KeyboardInterrupt:
        raise SystemExit(130)
''')
        command = [sys.executable, str(driver), '--data-root', str(self.root), '--workers', '2']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            deadline = time.monotonic() + 20
            while len(list(self.root.glob('worker-*'))) < 2 or not (self.root / 'busy').exists():
                self.assertIsNone(process.poll(), 'Generator exited before workers started')
                self.assertLess(time.monotonic(), deadline, 'Workers did not start')
                time.sleep(.05)
            worker_pids = [int(path.name.split('-')[1]) for path in self.root.glob('worker-*')]
            os.killpg(process.pid, signal.SIGINT)
            stdout, stderr = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 130, stdout + stderr)
            for pid in worker_pids:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            self.assertFalse((self.generated / 'samples_simulated_touches.jsonl').exists())
            with (self.generated / '.build.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                blocked = subprocess.run(
                    [sys.executable, str(Path(__file__).with_name('make_touch_data.py')),
                     '--data-root', str(self.root)], capture_output=True, text=True, timeout=15)
                self.assertNotEqual(blocked.returncode, 0)
                self.assertIn('Dataset is locked', blocked.stderr)
                self.assertNotIn('Traceback', blocked.stderr)
            # Reuse the actual CLI after cancellation, including manifest publication.
            subprocess.run([sys.executable, str(Path(__file__).with_name('make_touch_data.py')),
                            '--data-root', str(self.root), '--workers', '2'],
                           check=True, capture_output=True, timeout=30)
            self.assertTrue((self.generated / 'samples_simulated_touches.jsonl').exists())
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.communicate(timeout=15)


class JointInputTests(unittest.TestCase):
    def test_existing_joint_path_and_fixed_encoder_budget(self):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        from sam3d_objects.model.backbone.dit.embedder.surface_encoder_utils import farthest_point_indices
        from evaluation.input_visualizations import surface_indices
        # Execute the production functions without importing GPU model construction.
        tree = ast.parse((REPO / 'train.py').read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'combine_pointmap_and_touch']
        namespace = dict(torch=torch, pad_sequence=pad_sequence)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'train.py', 'exec'), namespace)
        tree = ast.parse((REPO / 'sam3d_objects/model/backbone/dit/embedder/touch.py').read_text())
        encoder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TouchEncoder')
        methods = [node for node in encoder.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('prepare_points', 'normalize_points_for_vecsetx')]
        for method in methods:
            method.decorator_list = []
        exec(compile(ast.Module(body=methods, type_ignores=[]), 'touch.py', 'exec'), namespace)
        model = SimpleNamespace(num_points=8192)
        model.normalize_points_for_vecsetx = lambda *args: namespace['normalize_points_for_vecsetx'](model, *args)
        prepare = lambda *args: namespace['prepare_points'](model, *args)
        generator = torch.Generator().manual_seed(29)
        touch = torch.rand(1, 8192, 3, generator=generator)
        touch_mask = torch.ones(1, 8192, dtype=torch.bool)
        inputs = dict(pointmap=torch.rand(1, 3, 8, 8, generator=generator), mask=torch.ones(1, 1, 8, 8))
        inputs['pointmap'][0, :, 0, 0] = float('nan')
        inputs['mask'][0, :, 0, 1] = 0
        joint, joint_mask = namespace['combine_pointmap_and_touch'](inputs, touch, touch_mask)
        self.assertEqual(joint.shape, (1, 8192 + 62, 3))
        torch.testing.assert_close(joint[:, -8192:], touch)
        # Only substitute PyTorch3D's unavailable native extension with the existing CPU FPS reference.
        ops = ModuleType('pytorch3d.ops')
        calls = []
        def fps(points, lengths, K, random_start_point):
            calls.append(K)
            indices = farthest_point_indices(points, K, random_start_point)
            return points.gather(1, indices[..., None].expand(-1, -1, 3)), indices
        ops.sample_farthest_points = fps
        with patch.dict(sys.modules, {'pytorch3d': ModuleType('pytorch3d'), 'pytorch3d.ops': ops}):
            selected, mask, _, _ = prepare(touch, touch_mask)
            self.assertEqual(calls, [])
            selected, mask, _, _ = prepare(joint, joint_mask)
            self.assertEqual(calls, [8192])
        self.assertEqual(selected.shape, (1, 8192, 3))
        self.assertTrue(mask.all())
        indices = surface_indices(joint[0].numpy(), 'vecsetx', 'cpu')
        normalized, _, _ = model.normalize_points_for_vecsetx(joint, joint_mask)
        torch.testing.assert_close(selected[0], normalized[0, indices])

    def test_joint_visualization_inverse(self):
        import torch
        from evaluation.input_visualizations import joint_camera_points
        points = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        scale, shift = torch.tensor([[2., 3., 4.]]), torch.tensor([[.2, .3, .4]])
        class Normalizer:
            def denormalize(self, p, s, t):
                return p * s[:, None, None] + t[:, None, None]
        preprocessor = SimpleNamespace(normalize_pointmap=True, pointmap_normalizer=Normalizer())
        result = joint_camera_points((points - shift) / scale,
                                     dict(pointmap_scale=scale, pointmap_shift=shift), preprocessor)
        np.testing.assert_allclose(result, points.numpy(), atol=1e-6)




if __name__ == '__main__':
    unittest.main()
