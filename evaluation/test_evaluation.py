"""CPU checks: python -m unittest evaluation.test_evaluation -v"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh

from evaluation.geometry import (
    load_mesh, load_geometry, load_dataset_mesh, load_dataset_view,
    load_evaluation, camera_fit, scene_bounds,
)
from evaluation.metrics import sample_surface, mesh_metrics, normalize_mesh, align_mesh
from evaluation.make_orbit import orbit_eye, save_frames


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
