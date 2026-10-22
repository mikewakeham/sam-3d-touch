"""Analytic checks of pose metric definitions and coordinate invariance."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import trimesh
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from evaluation import pose_metrics as pose
from evaluation.pose_metrics import re, chamfer_distance, icp


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh = trimesh.creation.box(extents=[.8, .45, .25])
        self.pose_path = self.root / 'pose.npz'

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


if __name__ == '__main__':
    unittest.main()
