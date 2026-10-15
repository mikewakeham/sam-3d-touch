"""Analytic metric checks and CPU integration of pose capture with shape evaluation."""
import ast
from contextlib import nullcontext
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
import trimesh
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_quaternion

from evaluation import evaluate
from evaluation import pose_metrics as pose
from evaluation.pose_metrics import re, chamfer_distance, icp
from sam3d_objects.data.dataset.tdfy.pose_target import PoseTargetConverter

ROOT = Path(__file__).resolve().parents[1]


def official_decoder():
    # Execute the actual decoder unchanged, without importing GPU renderers.
    tree = ast.parse((ROOT / 'sam3d_objects/pipeline/inference_utils.py').read_text())
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == 'pose_decoder')
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith('ROTATION_6D_')
                                                 for t in n.targets))]
    namespace = dict(torch=torch, matrix_to_quaternion=matrix_to_quaternion,
                     PoseTargetConverter=PoseTargetConverter)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'inference_utils.py', 'exec'), namespace)
    return namespace


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh = trimesh.creation.box(extents=[.8, .45, .25])
        self.pose_path = self.root / 'pose.npz'

    def camera(self, transform):
        path = self.root / 'camera.npz'
        np.savez(path, rotation=transform[:3, :3], translation=transform[:3, 3], scale=np.ones(3),
                 T_camera_from_object=np.diag([-1., -1., 1., 1.]) @ transform,
                 pose_source_frame='normalized_object', pose_coordinate_frame='sam_camera',
                 pose_rotation_convention='column_vectors')
        return path

    def test_copied_functions_match_upstream_code(self):
        expected = {
            'axis_aligned_bbox_overlaps_3d': '0029b1bfad6683ead520db5f27410cac5900dd734bab3e178bb69bc6c20b27a8',
            'chamfer_distance': '94203f92572d3a792b04d9edc8a2458128f5cad1f0df4ebbe105200f189ee428',
            're': '40598db126d41551c6e9e8018d43fa3042b8b1f7adf135272095667c1e1468cf',
            'calc_pts_diameter': '50cbc4e7e2582422fa6c9b5b3620da916945b959f8f3b078644b1f40e63f19b2',
            'procrustes': '215fe021cae281617f90fe9987e8b28d1fa4e788d38069f59d03e6fd8b3805e3',
            'icp': 'e7fac11753caca26073f4b8cb86536e8c113b620d2d337ba67c72fe04a4faf83',
        }
        raw = (ROOT / 'evaluation/pose_metrics.py').read_bytes()
        nodes = {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
        for name, digest in expected.items():
            node = nodes[name]
            self.assertEqual(hashlib.sha256(ast.dump(node).encode()).hexdigest(), digest)

    def test_add_s_uses_unsquared_symmetric_distances(self):
        a = np.array([[0., 0, 0], [2, 0, 0], [5, 0, 0]])
        b = np.array([[.25, 0, 0], [1, 0, 0]])
        distances = cdist(a, b)
        expected = distances.min(axis=0).mean() + distances.min(axis=1).mean()
        self.assertAlmostEqual(chamfer_distance(a, b), expected)
        self.assertAlmostEqual(chamfer_distance(b, a), expected)

    def test_box_iou_identity_overlap_disjoint(self):
        a = torch.tensor([[0., 0, 0, 1, 1, 1]], dtype=torch.float64)
        for offset, expected in [(0., 1.), (.5, 1 / 3), (1., 0.), (2., 0.)]:
            b = a + torch.tensor([[offset, 0, 0, offset, 0, 0]])
            self.assertAlmostEqual(pose.axis_aligned_bbox_overlaps_3d(a, b).item(), expected)

    def test_icp_recovers_known_correction_not_canonical_rotation(self):
        points = np.random.default_rng(7).normal(size=(2000, 3)) * [1, .6, .3]
        rotation = Rotation.from_euler('xyz', [2, -3, 5], degrees=True).as_matrix()
        moved = points @ rotation.T + [.01, -.02, .03]
        correction, aligned, _ = icp(moved, points, scale=False, reflection=False,
                                     threshold=1e-12, max_iterations=100)
        np.testing.assert_allclose(aligned, points, atol=1e-7)
        self.assertAlmostEqual(re(correction[:3, :3], np.eye(3)),
                               np.degrees(Rotation.from_matrix(rotation).magnitude()), places=6)

    def test_posed_mesh_metrics_and_inputs_unchanged(self):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', [20, 35, 10], degrees=True).as_matrix()
        transform[:3, 3] = [.2, -.1, 3]
        np.savez(self.pose_path, prediction_transform=transform, target_transform=transform)
        before = self.mesh.vertices.copy()
        scores = pose.score_pose(self.mesh, self.mesh, self.pose_path, 10000, 4000, 29)
        self.assertAlmostEqual(scores['pose_box_iou_3d'], 1.)
        self.assertLess(scores['pose_add_s'], .01)
        self.assertLess(scores['pose_icp_rot_deg'], 1.)
        np.testing.assert_array_equal(before, self.mesh.vertices)
        # A different canonical orientation, compensated by pose, is the same placed object.
        canonical = trimesh.transformations.rotation_matrix(.7, [0, 0, 1])
        rotated = self.mesh.copy().apply_transform(canonical)
        np.savez(self.pose_path, prediction_transform=transform @ np.linalg.inv(canonical), target_transform=transform)
        rotated_scores = pose.score_pose(rotated, self.mesh, self.pose_path, 10000, 4000, 29)
        for key in scores:
            self.assertAlmostEqual(scores[key], rotated_scores[key], places=5)

    def test_translation_is_not_removed_from_overlap_or_add_s(self):
        target = np.eye(4)
        target[:3, 3] = [0, 0, 3]
        prediction = target.copy()
        prediction[0, 3] = 2
        np.savez(self.pose_path, prediction_transform=prediction, target_transform=target)
        scores = pose.score_pose(self.mesh, self.mesh, self.pose_path, 3000, 1000, 29)
        self.assertEqual(scores['pose_box_iou_3d'], 0.)
        self.assertGreater(scores['pose_add_s'], 1.)

    def test_official_decoder_normalization_rotation_and_downsampling(self):
        ns = official_decoder()
        rotation = torch.tensor(Rotation.from_euler('xyz', [35, -20, 70], degrees=True).as_matrix(), dtype=torch.float32)
        translation = torch.tensor([[.3, -.2, 4.]])
        scene_scale, scene_shift = torch.full((1, 3), 2.5), torch.tensor([[.4, -.6, 2.]])
        quat = matrix_to_quaternion(rotation.T)[None]
        target = PoseTargetConverter.dicts_instance_pose_to_pose_target(
            'ScaleShiftInvariant', instance_scale_l2c=torch.ones(1, 3),
            instance_position_l2c=translation, instance_quaternion_l2c=quat,
            scene_scale=scene_scale, scene_shift=scene_shift,
        )
        # The released decoder stacks the two 6D vectors as columns.
        rot6d = torch.cat((rotation.T[:, 0], rotation.T[:, 1]))[None, None]
        outputs = dict(shape=torch.zeros(1, 1, 2),
                       **{'6drotation_normalized': (rot6d - ns['ROTATION_6D_MEAN']) / ns['ROTATION_6D_STD']},
                       translation=target['x_instance_translation'][:, None],
                       scale=target['x_instance_scale'][:, None].log(),
                       translation_scale=target['x_translation_scale'][:, None].log())
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation.numpy(), translation[0].numpy()
        pipeline = NS(pose_decoder=ns['pose_decoder']('ScaleShiftInvariant'))
        inputs = dict(pointmap_scale=scene_scale, pointmap_shift=scene_shift)
        original = {key: value.clone() for key, value in outputs.items()}
        torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
        for factor in (1, 2):
            pose.save_pose(self.pose_path, pipeline, outputs, inputs, factor, self.camera(transform))
            with np.load(self.pose_path) as saved:
                expected = transform.copy()
                expected[:3, :3] *= factor
                np.testing.assert_allclose(saved['prediction_transform'], expected, atol=2e-5)
                np.testing.assert_allclose(saved['target_transform'], transform, atol=1e-6)
                np.testing.assert_allclose(saved['pointmap_shift'], scene_shift)
                self.assertIn('raw_6drotation_normalized', saved)
                shrunk = self.mesh.copy()
                shrunk.vertices /= factor
                placed = shrunk.copy().apply_transform(saved['prediction_transform'])
                correct = self.mesh.copy().apply_transform(transform)
                np.testing.assert_allclose(placed.vertices, correct.vertices, atol=2e-5)
        for key in outputs:
            torch.testing.assert_close(outputs[key], original[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])

    def test_capture_failure_keeps_raw_pose(self):
        inputs = dict(pointmap_scale=torch.ones(1), pointmap_shift=torch.zeros(1, 3))
        outputs = dict(shape=torch.zeros(1, 1, 2), translation=torch.ones(1, 1, 3))
        with self.assertRaises(KeyError):
            pose.save_pose(self.pose_path, NS(pose_decoder=lambda *a, **kw: {}), outputs, inputs, 1, 'missing')
        with np.load(self.pose_path) as saved:
            np.testing.assert_array_equal(saved['raw_translation'], outputs['translation'].numpy())

    def test_summary_counts_failures_and_compares_metrics_in_correct_direction(self):
        rows = [dict(condition='official', sample_id='a', error='', pose_error='',
                     pose_box_iou_3d=.5, pose_add_s=.1, pose_icp_rot_deg=20.),
                dict(condition='touch', sample_id='a', error='', pose_error='',
                     pose_box_iou_3d=.7, pose_add_s=.05, pose_icp_rot_deg=10.),
                dict(condition='touch', sample_id='b', error='', pose_error='invalid pose',
                     pose_box_iou_3d=np.nan, pose_add_s=np.nan, pose_icp_rot_deg=np.nan),
                dict(condition='decoded_gt', sample_id='a', error='', pose_error='',
                     pose_box_iou_3d=np.nan, pose_add_s=np.nan, pose_icp_rot_deg=np.nan)]
        summary = evaluate.summarize(rows, ['official', 'touch', 'decoded_gt'], [], [], 'touch')
        self.assertEqual(summary['conditions']['touch']['pose_completed'], 1)
        evaluate.write_json(self.root / 'summary.json', summary)
        restored = json.loads((self.root / 'summary.json').read_text())
        self.assertIs(type(restored['conditions']['touch']['pose_completed']), int)
        self.assertIsNone(restored['conditions']['decoded_gt']['pose_add_s']['mean'])
        self.assertEqual(summary['conditions']['touch']['pose_failed'], 1)
        self.assertEqual(summary['conditions']['decoded_gt']['pose_completed'], 0)
        for key in ("pose_box_iou_3d", "pose_add_s", "pose_icp_rot_deg"):
            self.assertEqual(summary['comparisons']['touch_vs_official'][key]['improved_fraction'], 1.)

    def test_pose_is_automatic_and_cannot_mix_resume_protocols(self):
        with patch.object(sys, 'argv', ['evaluate', '--checkpoints', 'fake.pt']):
            args = evaluate.parse_args()
        self.assertFalse(hasattr(args, 'pose_metrics'))
        args.output_dir = self.root
        args.pipeline_config = self.root / 'pipeline.yaml'
        args.pipeline_config.write_text('{}')
        data = dict(dataset=dict(root=str(self.root)))
        with patch.dict('sys.modules', {'train': NS(checkpoint_train_scope=lambda _: None)}):
            _, settings = evaluate.validate_resume(args, [], data, [], {}, data)
            self.assertIs(settings['pose_metrics'], True)
            evaluate.validate_resume(args, [], data, [], {}, data)
            settings['pose_metrics'] = {'version': 1, 'sources': 'evaluation/vendor/sources.json'}
            (self.root / 'evaluation_settings.json').write_text(json.dumps(settings))
            evaluate.validate_resume(args, [], data, [], {}, data)
            del settings['pose_metrics']
            (self.root / 'evaluation_settings.json').write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, 'Evaluation protocol changed'):
                evaluate.validate_resume(args, [], data, [], {}, data)

    def test_evaluator_preserves_shape_and_handles_pose_failure_and_decoded_gt(self):
        camera = self.camera(np.eye(4))
        record = dict(sample_id='s', object_id='o', view_id='000', camera_path=str(camera), image_path='image.png')
        class Dataset:
            records = [record]
            resolve_path = staticmethod(Path)
            def __len__(self): return 1
        class Loader:
            dataset = Dataset()
            def __iter__(self):
                yield dict(sample_id=['s'], target_shape=torch.zeros(1, 1, 2), image=[torch.zeros(8, 8, 4)])
        class Generator:
            inference_steps = 25
            def __call__(self, shapes, device, *a, **kw):
                return {name: torch.randn(shape) for name, shape in shapes.items()}
        generator = Generator()
        mesh = self.mesh
        def sample_slat(*a, **kw):
            return NS(coords=torch.zeros(1, 4, dtype=torch.int32), feats=torch.randn(1, 8))
        pipeline = NS(ss_generator=generator, shape_model_dtype=torch.float32, downsample_ss_dist=0,
                      backbone=NS(latent_mapping={name: NS(pos_emb=torch.zeros(1), input_layer=NS(in_features=dim))
                                                 for name, dim in [('shape', 2), ('translation', 3)]}),
                      models={'ss_decoder': None}, sample_slat=sample_slat,
                      decode_slat=lambda *a: {'mesh': [NS(success=True, vertices=torch.tensor(mesh.vertices),
                                                         faces=torch.tensor(mesh.faces))]},
                      pose_decoder=lambda *a, **kw: dict(rotation=torch.tensor([[1., 0, 0, 0]]),
                                                          translation=torch.zeros(1, 3), scale=torch.ones(1, 3)))
        def prepare(*a, **kw):
            return None, (), {}, None, None, dict(pointmap_scale=torch.ones(1), pointmap_shift=torch.zeros(1, 3))
        fake_modules = {'train': NS(prepare_batch=prepare),
                        'sam3d_objects.pipeline.inference_utils': NS(downsample_sparse_structure=lambda x: (x, 1),
                                                                    prune_sparse_structure=lambda x, _: x)}
        target = evaluate.load_target_mesh
        rows, directories = [], []
        with patch.dict('sys.modules', fake_modules), \
             patch.object(evaluate, 'preprocess_stage2', return_value={}), \
             patch.object(evaluate, 'decode_voxels', return_value=torch.ones(1, 2, 2, 2, dtype=torch.bool)), \
             patch.object(evaluate, 'load_dataset_mesh', return_value=(mesh.copy(), np.eye(4))):
            for enabled, invalid, use_gt in [(False, False, False), (True, False, False),
                                             (True, True, False), (True, False, True)]:
                out = self.root / str(len(rows)); out.mkdir()
                args = NS(output_dir=out, device='cpu', seed=29, no_amp=True,
                          inference_steps=10, stage2_inference_steps=10, icp_points=600,
                          surface_points=1000, emd_points=0, save_points=100, metric_workers=1)
                cache = {'o': target(record, Loader.dataset, out, 1000, 600, 100, 29)}
                if invalid:
                    np.savez(camera, K=np.eye(3))
                elif enabled:
                    self.camera(np.eye(4))
                # Bypass pose work for the shape-invariance regression baseline only.
                baseline = (nullcontext() if enabled else patch.object(
                    pose, 'save_pose', side_effect=RuntimeError('pose bypassed for regression baseline')))
                with baseline:
                    row = evaluate.evaluate_condition('decoded_gt' if use_gt else 'test', pipeline, None, Loader(),
                                                     {'s': record}, cache, set(), out / 'metrics.csv', args,
                                                     use_gt_latent=use_gt)[0]
                self.assertEqual(row['error'], '')
                self.assertTrue(evaluate.artifacts_complete(row, out))
                rows.append(row); directories.append(out)
        for row in rows[1:3]:
            for key in ['fscore_0.01', 'chamfer', 'voxel_iou_64', 'normal_consistency']:
                self.assertEqual(rows[0][key], row[key])
        for index in (1, 2):
            for key in ('stage1_path', 'slat_path'):
                with np.load(directories[0] / rows[0][key]) as a, np.load(directories[index] / rows[index][key]) as b:
                    for field in a.files: np.testing.assert_array_equal(a[field], b[field])
            self.assertEqual((directories[0] / rows[0]['mesh_raw_path']).read_bytes(),
                             (directories[index] / rows[index]['mesh_raw_path']).read_bytes())
        self.assertAlmostEqual(rows[1]['pose_box_iou_3d'], 1., places=6)
        self.assertTrue(rows[2]['pose_error'])
        self.assertTrue(np.isnan(rows[2]['pose_add_s']))
        self.assertEqual(rows[3]['pose_path'], '')
        self.assertTrue(np.isnan(rows[3]['pose_add_s']))
        self.assertEqual(generator.inference_steps, 25)
        (directories[1] / rows[1]['pose_path']).unlink()
        self.assertFalse(evaluate.artifacts_complete(rows[1], directories[1]))


if __name__ == '__main__':
    unittest.main()
