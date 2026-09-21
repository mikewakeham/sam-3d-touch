"""CPU checks: python -m unittest evaluation.test_evaluation -v"""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image
import trimesh

from evaluation.geometry import (
    load_mesh, load_geometry, load_dataset_mesh, load_dataset_view,
    load_evaluation, load_reference_camera, camera_fit, scene_bounds,
)
from evaluation.metrics import sample_surface, mesh_metrics, normalize_mesh, align_mesh
from evaluation.make_orbit import orbit_eye, input_orbit_extrinsic, save_frames


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mesh = trimesh.creation.box(extents=(.4, .6, 1.))
        self.mesh_path = self.root / 'mesh.npz'
        np.savez(self.mesh_path, vertices=self.mesh.vertices, faces=self.mesh.faces,
                 coordinate_frame='normalized_object')

    def tearDown(self):
        self.temporary.cleanup()

    def test_normalized_mesh_is_not_transformed_twice(self):
        transform = np.eye(4)
        transform[:3, :3] *= 7
        transform[:3, 3] = (1, 2, 3)
        np.savez(self.root / 'object_transform.npz', T_normalized_from_source=transform)
        dataset = argparse.Namespace(root=self.root, resolve_path=lambda path: self.root / path)
        record = {'object_id': 'box', 'mesh_path': 'mesh.npz', 'object_transform_path': 'object_transform.npz'}
        mesh, applied = load_dataset_mesh(record, dataset)
        np.testing.assert_allclose(mesh.vertices, self.mesh.vertices)
        np.testing.assert_allclose(applied, np.eye(4))
        self.mesh.export(self.root / 'source.obj')
        record['mesh_path'] = 'source.obj'
        mesh, applied = load_dataset_mesh(record, dataset)
        np.testing.assert_allclose(mesh.vertices, trimesh.transform_points(self.mesh.vertices, transform))
        np.testing.assert_allclose(applied, transform)

    def test_glb_scene_keeps_instances_and_transforms(self):
        scene = trimesh.Scene()
        scene.add_geometry(self.mesh, node_name='first')
        transform = np.eye(4)
        transform[0, 3] = 3.
        scene.add_geometry(self.mesh, node_name='second', transform=transform)
        path = self.root / 'scene.glb'
        scene.export(path)
        loaded = load_mesh(path)
        self.assertEqual(len(loaded.faces), len(self.mesh.faces) * 2)
        np.testing.assert_allclose(loaded.bounds, scene.bounds, atol=1e-6)

    def test_points_normals_and_latent_rejection(self):
        points, normals = sample_surface(self.mesh, 40, 29)
        path = self.root / 'pool.npz'
        np.savez(path, points_object=points, normals_object=normals, coordinate_frame='normalized_object')
        item = load_geometry(path)
        np.testing.assert_array_equal(item['points'], points)
        np.testing.assert_array_equal(item['normals'], normals)
        self.assertEqual(item['frame'], 'normalized_object')
        np.savez(path, shape=np.zeros((4096, 8)))
        with self.assertRaisesRegex(ValueError, 'Select --points-key'):
            load_geometry(path)
        np.savez(path, points=points, points_camera=points)
        with self.assertRaises(ValueError):
            load_geometry(path)
        np.testing.assert_array_equal(load_geometry(path, points_key='points_camera')['points'], points)

    def test_voxel_coordinates(self):
        grid = np.zeros((64, 64, 64), dtype=bool)
        grid[16, 32, 48] = True
        path = self.root / 'stage1.npz'
        np.savez(path, prediction=grid)
        item = load_geometry(path, voxel_key='prediction')
        np.testing.assert_allclose(item['mesh'].bounds.mean(axis=0), [-.25, 0., .25])
        np.testing.assert_allclose(item['mesh'].extents, [.9 / 64] * 3)

    def test_shared_camera_preserves_geometry(self):
        shifted = self.mesh.copy()
        shifted.apply_translation([3, 2, 1])
        items = [{'mesh': self.mesh}, {'mesh': shifted}]
        before = shifted.vertices.copy()
        center, radius = camera_fit(items, aspect=.5)
        bounds = scene_bounds(items)
        np.testing.assert_allclose(center, bounds.mean(axis=0))
        self.assertGreater(radius, np.linalg.norm(bounds[1] - bounds[0]) / 2)
        np.testing.assert_array_equal(shifted.vertices, before)
        np.testing.assert_allclose(orbit_eye(center, radius, 0, 0, 'y') - center, [0, 0, -radius])
        np.testing.assert_allclose(orbit_eye(center, radius, 0, 0, 'z') - center, [0, -radius, 0])

    def test_dataset_view_shared_sam_frame(self):
        directory = self.root / 'generated_data' / 'box'
        view = directory / 'views' / '000'
        view.mkdir(parents=True)
        (directory / 'mesh.npz').write_bytes(self.mesh_path.read_bytes())
        transform = np.eye(4)
        transform[2, 3] = 2
        np.savez(view / 'camera.npz', K=np.eye(3), T_camera_from_object=transform)
        Image.fromarray(np.full((2, 2, 4), 255, dtype=np.uint8)).save(view / 'image.png')
        np.save(view / 'depth.npy', np.ones((2, 2), dtype=np.float32))
        points, normals = sample_surface(self.mesh, 40, 29)
        np.savez(directory / 'surface_pool.npz', points_object=points, normals_object=normals)
        items, images = load_dataset_view(self.root, 'box', 0)
        sam = np.diag([-1., -1., 1., 1.]) @ transform
        np.testing.assert_allclose(items[0]['mesh'].vertices, trimesh.transform_points(self.mesh.vertices, sam))
        np.testing.assert_allclose(items[2]['points'], trimesh.transform_points(points, sam))
        np.testing.assert_allclose(items[2]['normals'], normals @ sam[:3, :3].T)
        np.testing.assert_allclose(items[1]['points'][1], [-1, 0, 1])
        self.assertTrue(images[0].exists())
        args = argparse.Namespace(inputs=None, evaluation_dir=None, data_root=self.root,
                                  object_id='box', view_id=0, width=768, height=768)
        camera = load_reference_camera(args)
        projected = trimesh.transform_points(items[1]['points'], input_orbit_extrinsic(camera, 0))
        pixels = projected @ np.asarray(camera['intrinsics']).T
        np.testing.assert_allclose(pixels[:, :2] / pixels[:, 2:], [[0, 0], [384, 0], [0, 384], [384, 384]])

    def make_evaluation(self):
        self.mesh.export(self.root / 'mesh.ply')
        np.savez(self.root / 'points.npz', surface_seed=29)
        np.savez(self.root / 'alignment.npz', surface_seed=29,
                 prediction_normalization=np.eye(4), icp_transform=np.eye(4))
        grid = np.zeros((64, 64, 64), dtype=bool)
        grid[32, 32, 32] = True
        np.savez(self.root / 'stage1.npz', prediction=grid, downsample_factor=1)
        row = dict(condition='frozen', sample_id='box_000', object_id='box', error='',
                   target_mesh_path='mesh.ply', mesh_aligned_path='mesh.ply', image_path='missing.png',
                   target_points_path='points.npz', alignment_path='alignment.npz', stage1_path='stage1.npz')
        with (self.root / 'metrics.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return row

    def test_evaluation_has_no_dataset_or_decoded_gt_dependency(self):
        self.make_evaluation()
        items, images = load_evaluation(self.root, 'box_000', modes=('mesh', 'voxel'))
        self.assertEqual([item['name'] for item in items], ['mesh_ground_truth', 'mesh_frozen', 'voxel_frozen'])
        self.assertEqual(images, [])
        np.testing.assert_allclose(items[2]['mesh'].bounds.mean(axis=0), 0)
        with self.assertRaises(ValueError):
            load_evaluation(self.root, 'box_000', conditions=['missing'])

    def test_batch_orbits_use_unique_completed_samples(self):
        from evaluation.make_orbit import render_all

        row = self.make_evaluation()
        with (self.root / 'metrics.csv').open('a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writerow(dict(row, condition='official'))
            writer.writerow(dict(row, sample_id='other_001'))
            writer.writerow(dict(row, sample_id='failed_000', error='No mesh'))
        args = argparse.Namespace(evaluation_dir=self.root, sample_id=None, conditions=None,
                                  max_samples=0, selection_seed=29, output_dir=None)
        arguments = ['--evaluation-dir', str(self.root), '--all-samples', '--frames', '12']
        with patch('evaluation.make_orbit.subprocess.run') as run:
            render_all(args, arguments)
        self.assertEqual(run.call_count, 2)
        self.assertEqual([call.args[0][-1] for call in run.call_args_list], ['box_000', 'other_001'])
        for call in run.call_args_list:
            self.assertNotIn('--all-samples', call.args[0])
            self.assertEqual(call.args[0][-4:-2], ['--frames', '12'])
            self.assertTrue(call.kwargs['check'])
        args.sample_id = 'box_000'
        with self.assertRaises(ValueError):
            render_all(args, arguments)

    def test_batch_orbits_random_subset_is_reproducible(self):
        from evaluation.make_orbit import render_all

        row = self.make_evaluation()
        with (self.root / 'metrics.csv').open('a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            for index in range(99):
                writer.writerow(dict(row, sample_id=f'object_{index:03d}'))
            writer.writerow(dict(row, condition='official'))
            writer.writerow(dict(row, sample_id='failed', error='No mesh'))
        args = argparse.Namespace(evaluation_dir=self.root, sample_id=None, conditions=None,
                                  max_samples=10, selection_seed=29, output_dir=None)
        arguments = ['--evaluation-dir', str(self.root), '--all-samples', '--max-samples', '10']
        selections = []
        for count in (10, 10, 20):
            args.max_samples = count
            with patch('evaluation.make_orbit.subprocess.run') as run:
                render_all(args, arguments)
            ids = [call.args[0][-1] for call in run.call_args_list]
            self.assertEqual(len(ids), count)
            self.assertEqual(len(set(ids)), count)
            self.assertNotIn('failed', ids)
            saved = json.loads((self.root / 'orbits' / 'selected_samples.json').read_text())
            self.assertEqual(saved['sample_ids'], ids)
            self.assertEqual(saved['available_samples'], 100)
            selections.append(ids)
        self.assertEqual(selections[0], selections[1])
        self.assertEqual(selections[0], selections[2][:10])
        self.assertNotEqual(selections[0], sorted(selections[0]))

    def test_voxel_orbits_show_decoded_output_before_downsampling(self):
        self.make_evaluation()
        grid = np.zeros((64, 64, 64), dtype=bool)
        grid[16, 32, 48] = True
        np.savez(self.root / 'stage1.npz', prediction=grid, downsample_factor=2, coords=[[0, 32, 32, 32]])
        items, _ = load_evaluation(self.root, 'box_000', modes=('voxel',))
        np.testing.assert_allclose(items[0]['mesh'].bounds.mean(axis=0), [-.25, 0., .25])

    def test_input_orbits_use_actual_surface_cloud_and_shared_frame(self):
        import yaml
        from evaluation.input_visualizations import load_input_visualizations
        from data_generation.general.sample_full_surface import sample_full_surface

        row = self.make_evaluation()
        with (self.root / 'metrics.csv').open('a', newline='') as file:
            csv.DictWriter(file, fieldnames=list(row)).writerow(dict(row, condition='scratch'))
        object_dir = self.root / 'generated_data' / 'box'
        view = object_dir / 'views' / '000'
        view.mkdir(parents=True)
        (object_dir / 'mesh.npz').write_bytes(self.mesh_path.read_bytes())
        # Include roll, off-center intrinsics and translated normalization: a
        # symmetric, axis-aligned fixture would miss camera flips/frame errors.
        camera = trimesh.transformations.euler_matrix(.2, .3, .4)
        camera[:3, 3] = [.1, -.2, 3]
        K = np.array([[2.4, 0, .7], [0, 2.7, .9], [0, 0, 1]])
        np.savez(view / 'camera.npz', K=K, T_camera_from_object=camera)
        rgba = np.full((2, 2, 4), 255, dtype=np.uint8)
        rgba[0, 0, 3] = 0
        Image.fromarray(rgba).save(view / 'image.png')
        np.save(view / 'depth.npy', np.ones((2, 2), dtype=np.float32))
        points, faces = trimesh.sample.sample_surface(self.mesh, 8192, seed=29)
        surface = sample_full_surface(points, view / 'camera.npz', view / 'depth.npy', .01,
                                      self.mesh.face_normals[faces], faces)
        np.savez(view / 'full_surface.npz', **surface)
        # A newer/larger pool exists, but these VecSetX checkpoints never used it.
        np.savez(object_dir / 'surface_pool.npz', points_object=np.ones((20480, 3)))
        grid = np.zeros((64, 64, 64), dtype=bool)
        grid[32, 32, 32] = True
        np.savez(self.root / 'stage1.npz', touch_centers=surface['points_camera'] * 2 + 1,
                 prediction=grid, downsample_factor=1)
        normalization = np.diag([2., 2., 2., 1.])
        normalization[:3, 3] = [.4, -.3, .2]
        np.savez(self.root / 'points.npz', evaluation_normalization=normalization)
        target = self.mesh.copy()
        target.apply_transform(normalization)
        target.export(self.root / 'mesh.ply')
        record = {'sample_id': 'box_000', 'object_id': 'box', 'view_id': '000',
                  'mesh_path': str(object_dir / 'mesh.npz'), 'camera_path': str(view / 'camera.npz'),
                  'image_path': str(view / 'image.png'), 'depth_path': str(view / 'depth.npy'),
                  'full_surface_path': str(view / 'full_surface.npz')}
        (self.root / 'samples.jsonl').write_text(json.dumps(record) + '\n')
        (self.root / 'splits.json').write_text(json.dumps({'val': ['box'], 'train': [], 'test': []}))
        data = {'dataset': {'root': str(self.root), 'manifest': 'samples.jsonl', 'split_file': 'splits.json', 'split': 'val'},
                'touch': {'source': 'full_surface'}}
        run = {'data': data, 'touch_config': {'encoder_name': 'vecsetx'}, 'mode': 'image_touch'}
        config = {'selection_data_config': data, 'runs': {'frozen': run, 'scratch': run}}
        (self.root / 'config.yaml').write_text(yaml.safe_dump(config))
        args = argparse.Namespace(evaluation_dir=self.root, sample_id='box_000', conditions=None,
                                  input_device='cpu', blender='unused')
        with patch('evaluation.input_visualizations.textured_mesh_path', return_value=self.root / 'textured.glb'):
            groups, images, details = load_input_visualizations(args)
        self.assertEqual({name for name, _ in groups}, {
            'input_mesh_textured', 'input_mesh', 'input_mesh_pointmap', 'input_mesh_surface',
            'input_mesh_pointmap_surface', 'input_pointmap', 'input_surface', 'input_pointmap_surface'})
        surface_item = dict(groups)['input_surface'][0]
        np.testing.assert_allclose(surface_item['points'], trimesh.transform_points(points, normalization), atol=5e-7)
        self.assertEqual(len(surface_item['points']), 8192)
        self.assertEqual(details['surface_groups'], [['frozen', 'scratch']])
        self.assertEqual(details['frozen']['encoder_valid_points'], 8192)
        self.assertEqual(len(dict(groups)['input_pointmap'][0]['points']), 3)
        self.assertEqual(images, [view / 'image.png'])
        np.testing.assert_allclose(dict(groups)['input_mesh_textured'][0]['transform'], normalization)
        args.inputs = None
        args.width, args.height = 768, 768
        reference = load_reference_camera(args)
        extrinsic = input_orbit_extrinsic(reference, 0)
        projection = np.asarray(reference['intrinsics'])
        # Every visible point must land back on its own source pixel, including
        # the SAM left/up -> OpenCV right/down conversion.
        pointmap = dict(groups)['input_pointmap'][0]['points']
        pixels = trimesh.transform_points(pointmap, extrinsic) @ projection.T
        np.testing.assert_allclose(pixels[:, :2] / pixels[:, 2:], [[384, 0], [0, 384], [384, 384]], atol=1e-4)
        # Full surface and mesh vertices use the very same input projection.
        for original, displayed in [(points, surface_item['points']),
                                    (self.mesh.vertices, target.vertices)]:
            expected = trimesh.transform_points(original, camera) @ K.T
            actual = trimesh.transform_points(displayed, extrinsic) @ projection.T
            np.testing.assert_allclose(actual[:, :2] / actual[:, 2:],
                                       384 * expected[:, :2] / expected[:, 2:], atol=1e-4)
        # Camera motion closes after one turn and preserves object-Z elevation.
        initial_pose = np.linalg.inv(extrinsic)
        pivot = np.asarray(reference['pivot'])
        for angle in [0, np.pi / 2, np.pi, 2 * np.pi]:
            pose = np.linalg.inv(input_orbit_extrinsic(reference, angle))
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)
            self.assertAlmostEqual(np.linalg.norm(pose[:3, 3] - pivot), np.linalg.norm(initial_pose[:3, 3] - pivot))
            self.assertAlmostEqual(pose[2, 3], initial_pose[2, 3])
        np.testing.assert_allclose(input_orbit_extrinsic(reference, 2 * np.pi), extrinsic, atol=1e-7)
        from evaluation.make_orbit import main
        argv = ['make_orbit', '--evaluation-dir', str(self.root), '--sample-id', 'box_000',
                '--with-inputs', '--modes', 'mesh', 'voxel', '--input-device', 'cpu']
        with patch('sys.argv', argv), patch('evaluation.make_orbit.render') as render, patch(
            'evaluation.input_visualizations.textured_mesh_path', return_value=self.root / 'textured.glb'
        ):
            main()
        self.assertEqual(render.call_count, 12)  # Eight inputs + mesh/voxel for two conditions.
        first_center = render.call_args_list[0].args[1]
        first_radius = render.call_args_list[0].args[2]
        for call in render.call_args_list:
            np.testing.assert_array_equal(call.args[1], first_center)
            self.assertEqual(call.args[2], first_radius)
            self.assertEqual(call.kwargs['reference_camera'], reference)
        output = self.root / 'orbits' / 'box_000'
        self.assertEqual((output / 'input_view.png').read_bytes(), (view / 'image.png').read_bytes())
        settings = json.loads((output / 'orbit_settings.json').read_text())
        self.assertEqual(settings['input_details']['frozen']['encoder_valid_points'], 8192)
        self.assertIn('voxel_frozen', settings['geometry_names'])
        self.assertEqual(settings['reference_camera'], reference)
        # Explicit generic framing still works, even for a moved evaluation
        # whose source camera/dataset is no longer available.
        (view / 'camera.npz').unlink()
        with patch('sys.argv', ['make_orbit', '--evaluation-dir', str(self.root), '--sample-id', 'box_000',
                                '--fit-camera', '--overwrite']), patch('evaluation.make_orbit.render') as render:
            main()
        self.assertTrue(all(call.kwargs['reference_camera'] is None for call in render.call_args_list))

    def test_encoder_surface_selection_keeps_context_count(self):
        from evaluation.input_visualizations import surface_indices

        for name, count in [('vecsetx', 8192), ('craftsman', 16384), ('triposg', 20480)]:
            np.testing.assert_array_equal(surface_indices(np.zeros((count, 3)), name, 'cpu'), np.arange(count))
        with self.assertRaises(ValueError):
            surface_indices(np.zeros((100, 3)), 'unknown')

    def test_renderer_passes_saved_intrinsics_and_pose_to_open3d(self):
        from evaluation.make_orbit import render

        camera = dict(intrinsics=[[700., 0, 383.5], [0, 710., 383.5], [0, 0, 1]],
                      camera_to_world=trimesh.transformations.translation_matrix([1., -3., 2.]).tolist(),
                      pivot=[0., 0., 0.], up=[0., 0., 1.])
        args = argparse.Namespace(width=768, height=768, light_strength=1., up='y', frames=2)
        renderer = MagicMock()
        renderer.render_to_image.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
        renderer.render_to_depth_image.return_value = np.zeros((2, 2))
        open3d = MagicMock()
        open3d.visualization.rendering.OffscreenRenderer.return_value = renderer
        with patch.dict(sys.modules, open3d=open3d), patch('evaluation.make_orbit.initialize_lighting'), \
             patch('evaluation.make_orbit.save_frames'):
            render([], np.zeros(3), 3., args, self.root / 'orbit', 1., reference_camera=camera)
        calls = renderer.setup_camera.call_args_list
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(calls[0].args[0], camera['intrinsics'])
        np.testing.assert_allclose(calls[0].args[1], np.linalg.inv(camera['camera_to_world']))
        self.assertEqual(calls[0].args[2:], (768, 768))
        self.assertFalse(np.allclose(calls[1].args[1], calls[0].args[1]))

    def test_surface_fps_uses_saved_encoder_coordinates(self):
        import types
        import torch
        from evaluation.input_visualizations import surface_indices

        points = np.random.default_rng(29).normal(size=(9000, 3)).astype(np.float32)
        expected = np.arange(8999, 807, -1)
        def fps(normalized, lengths, K, random_start_point):
            centered = torch.from_numpy(points) - (torch.from_numpy(points).amax(0) + torch.from_numpy(points).amin(0)) / 2
            centered *= 1 / torch.linalg.vector_norm(centered, dim=1).amax()
            torch.testing.assert_close(normalized[0], centered, rtol=0, atol=0)
            self.assertEqual(K, 8192)
            self.assertFalse(random_start_point)
            self.assertEqual(lengths.tolist(), [9000])
            return normalized[:, expected], torch.from_numpy(expected)[None]
        ops = types.ModuleType('pytorch3d.ops')
        ops.sample_farthest_points = fps
        with patch.dict(sys.modules, {'pytorch3d.ops': ops}):
            selected = surface_indices(points, 'vecsetx', 'cpu')
        np.testing.assert_array_equal(selected, expected)

    def test_rescore_saved_outputs_without_inference(self):
        self.make_evaluation()
        result = subprocess.run([sys.executable, '-m', 'evaluation.metrics', '--evaluation-dir', str(self.root),
                                 '--surface-points', '500'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / 'metrics_rescored.csv').open() as file:
            row = next(csv.DictReader(file))
        self.assertEqual(row['error'], '')
        self.assertAlmostEqual(float(row['chamfer']), 0)
        self.assertAlmostEqual(float(row['fscore_0.01']), 1)
        settings = json.loads((self.root / 'metrics_rescored.json').read_text())
        self.assertEqual(settings['summary']['frozen']['completed'], 1)

    def test_metric_definitions_and_deterministic_sampling(self):
        points, normals = sample_surface(self.mesh, 100, 29)
        np.testing.assert_array_equal(sample_surface(self.mesh, 100, 29)[0], points)
        metrics = mesh_metrics(points, normals, points, -normals, 0, 'cpu', 1)
        self.assertEqual(metrics['chamfer'], 0)
        self.assertEqual(metrics['fscore_0.01'], 1)
        self.assertEqual(metrics['normal_consistency'], 1)
        self.assertTrue(np.isnan(metrics['emd']))
        point = np.array([[0., 0., 0.]])
        normal = np.array([[1., 0., 0.]])
        metrics = mesh_metrics(point, normal, point + [.02, 0, 0], normal, 0, 'cpu', 1)
        self.assertAlmostEqual(metrics['chamfer'], .02)
        self.assertEqual(metrics['fscore_0.01'], 0)
        normalized, transform = normalize_mesh(self.mesh)
        self.assertAlmostEqual(normalized.extents.max(), 2)
        np.testing.assert_allclose(trimesh.transform_points(self.mesh.vertices, transform), normalized.vertices)

    @unittest.skipUnless(importlib.util.find_spec('open3d'), 'Open3D is optional for CPU mesh metrics')
    def test_similarity_alignment_recovers_transformed_shape(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.apply_scale([.3, .7, 1.])
        points, _ = sample_surface(mesh, 1000, 29)
        transform = trimesh.transformations.rotation_matrix(.6, [1, 2, 3])
        transform[:3, :3] *= 1.3
        transform[:3, 3] = [.4, -.2, .7]
        prediction = mesh.copy()
        prediction.apply_transform(transform)
        transformed = trimesh.transform_points(points, transform)
        aligned, result, error, _, _ = align_mesh(prediction, mesh, transformed, points, 1)
        self.assertLess(error, 1e-5)
        np.testing.assert_allclose(aligned.bounds, mesh.bounds, atol=1e-5)

    def test_resume_rejects_changed_inputs_and_missing_artifacts(self):
        from evaluation.evaluate import validate_resume, artifacts_complete

        for name in ('checkpoint.pt', 'config.yaml', 'pipeline.yaml', 'samples.jsonl', 'splits.json'):
            (self.root / name).write_text('fixture')
        args = argparse.Namespace(output_dir=self.root, data_config=None,
                                  pipeline_config=self.root / 'pipeline.yaml', split='val',
                                  selection='random', max_samples=2, selection_seed=29, seed=29,
                                  inference_steps=25, stage2_inference_steps=25,
                                  surface_points=1000, icp_points=100, emd_points=0,
                                  save_points=100, no_amp=False)
        data = {'dataset': {'root': str(self.root), 'manifest': 'samples.jsonl', 'split_file': 'splits.json'}}
        checkpoints = [self.root / 'checkpoint.pt']
        selected = [{'sample_id': 'box_000'}]
        validate_resume(args, checkpoints, data, selected, 0)
        validate_resume(args, checkpoints, data, selected, 0)
        # Changing parallelism is safe when resuming the same selected evaluation.
        with patch.dict(os.environ, WORLD_SIZE='4', LOCAL_RANK='1'):
            args.device = 'cuda:1'
            args.workers = 4
            args.metric_workers = 16
            validate_resume(args, checkpoints, data, selected, 1)
        args.inference_steps = 30
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_resume(args, checkpoints, data, selected, 0)
        args.inference_steps = 25
        (self.root / 'checkpoint.pt').write_text('different checkpoint')
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_resume(args, checkpoints, data, selected, 0)
        row = {'error': ''}
        for key in ('stage1_path', 'slat_path', 'mesh_raw_path', 'mesh_normalized_path',
                    'mesh_aligned_path', 'alignment_path', 'target_mesh_path', 'target_points_path'):
            row[key] = 'checkpoint.pt'
        self.assertTrue(artifacts_complete(row, self.root))
        row['stage1_path'] = 'missing.npz'
        self.assertFalse(artifacts_complete(row, self.root))
        row['stage1_path'] = 'checkpoint.pt'
        row['error'] = 'interrupted'
        self.assertFalse(artifacts_complete(row, self.root))

    def test_summary_without_official_or_decoded_gt(self):
        from evaluation.evaluate import summarize

        metrics = dict.fromkeys(['fscore_0.01', 'f_precision_0.01', 'f_recall_0.01', 'voxel_iou_64',
                                 'chamfer', 'normal_consistency', 'emd', 'icp_error', 'icp_fitness',
                                 'icp_inlier_rmse', 'stage1_aligned_chamfer', 'stage1_aligned_voxel_iou_64'], .1)
        rows = [dict(metrics, condition=name, sample_id='box_000', error='') for name in ('base', 'frozen')]
        summary = summarize(rows, ['base', 'frozen'], [], 'base', 'frozen')
        self.assertIn('frozen_vs_base', summary['comparisons'])
        self.assertNotIn('frozen_vs_official', summary['comparisons'])

    @unittest.skipUnless(importlib.util.find_spec('open3d'), 'Open3D rendering dependency unavailable')
    def test_render_geometry_and_material(self):
        from evaluation.make_orbit import mesh_geometry, material

        geometry = mesh_geometry(self.mesh)
        np.testing.assert_allclose(np.asarray(geometry.vertices), self.mesh.vertices)
        self.assertTrue(geometry.has_vertex_normals())
        self.assertEqual(material((.7, .7, .7)).shader, 'defaultLit')

    def test_optional_sinkhorn_on_cpu(self):
        from evaluation.metrics import sinkhorn_emd

        source = np.array([[0., 0., 0.]], dtype=np.float32)
        target = np.array([[.25, 0., 0.]], dtype=np.float32)
        self.assertAlmostEqual(sinkhorn_emd(source, target, 'cpu'), .25, places=6)

    def test_gif_preserves_aspect(self):
        args = argparse.Namespace(mp4=False, fps=20, gif_fps=10, gif_size=64)
        frames = [np.full((80, 160, 3), value, dtype=np.uint8) for value in (0, 50, 100, 150)]
        save_frames(frames, self.root / 'orbit', args)
        with Image.open(self.root / 'orbit.gif') as image:
            self.assertEqual(image.size, (64, 32))
            self.assertEqual(image.n_frames, 2)
        self.assertTrue((self.root / 'orbit.png').exists())

    def test_metrics_import_does_not_load_training(self):
        result = subprocess.run([sys.executable, '-c',
                                 'import evaluation.metrics, evaluation.geometry, sys; '
                                 'assert "torch" not in sys.modules; assert "train" not in sys.modules'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
