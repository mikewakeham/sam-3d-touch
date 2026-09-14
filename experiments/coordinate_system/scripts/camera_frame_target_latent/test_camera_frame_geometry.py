"""CPU coordinate-contract tests; synthetic shapes are test fixtures, not evidence."""
from pathlib import Path
import sys
import unittest
import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p/'train.py').is_file())
sys.path.insert(0, str(REPO))
from experiments.coordinate_system.scripts.camera_frame_target_latent.camera_frame_geometry import (
    affine, make_frames, normalization, select_records)


class CameraFrameTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(29)
        self.points = rng.normal(size=(1024, 3))*[.5, .2, .1]
        m, _, _ = normalization(self.points, 'extent')
        self.points = affine(self.points, m)
        angle = .7
        self.camera = np.eye(4)
        self.camera[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                               [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        self.camera[:3, 3] = [2, -1, 4]

    def test_target_cube_and_exact_frame_conversion(self):
        surface = affine(self.points, self.camera)
        f = make_frames(self.points, surface, self.camera)
        target = affine(self.points, f['target_from_object'])
        np.testing.assert_allclose((target.min(0)+target.max(0))/2, 0, atol=1e-12)
        self.assertAlmostEqual(np.ptp(target, axis=0).max(), 1)
        vec = affine(surface, f['vec_from_camera'])
        np.testing.assert_allclose(affine(vec, f['target_from_vec']), target, atol=1e-12)
        self.assertFalse(np.allclose(vec, target))

    def test_camera_translation_cancels_but_rotation_is_retained(self):
        first = make_frames(self.points, affine(self.points, self.camera), self.camera)
        translated = self.camera.copy(); translated[:3, 3] += [3, 4, 5]
        second = make_frames(self.points, affine(self.points, translated), translated)
        np.testing.assert_allclose(first['target_from_object'], second['target_from_object'], atol=1e-12)
        identity = make_frames(self.points, self.points, np.eye(4))
        self.assertFalse(np.allclose(first['target_from_object'], identity['target_from_object']))

    def test_visible_subset_must_use_full_surface_normalization(self):
        surface = affine(self.points, self.camera)
        visible = surface[surface[:, 0] > np.median(surface[:, 0])]
        frame, _, _ = normalization(surface, 'radius')
        independent, _, _ = normalization(visible, 'radius')
        self.assertGreater(np.abs(affine(visible, frame)-affine(visible, independent)).max(), .01)
        # A surface point at a pixel gets exactly the same shared coordinate.
        np.testing.assert_allclose(affine(surface[:8], frame), affine(surface, frame)[:8])

    def test_no_silent_nonrigid_camera_or_degenerate_geometry(self):
        bad = self.camera.copy(); bad[0, :3] *= 1.01
        with self.assertRaises(ValueError): make_frames(self.points, affine(self.points, bad), bad)
        with self.assertRaises(ValueError): normalization(np.zeros((10, 3)), 'radius')
        bad = self.camera.copy(); bad[0, :3] *= -1
        with self.assertRaises(ValueError): make_frames(self.points, affine(self.points, bad), bad)

    def test_selection_uses_all_remaining_views_and_disjoint_objects(self):
        records = [dict(object_id=f'o{i}', sample_id=f'o{i}_{j:02}') for i in range(170) for j in range(16)]
        splits = dict(train=[f'o{i}' for i in range(135)], val=[f'o{i}' for i in range(135, 170)])
        groups, ids = select_records(records, splits, train_objects=128, val_objects=32, held_views=2, train_views=0)
        self.assertEqual(len(groups['train']), 128*14)
        self.assertEqual(len(groups['held_view']), 128*2)
        self.assertEqual(len(groups['val']), 32*16)
        self.assertFalse(set(ids['train']) & set(ids['val']))
        all_ids = [r['sample_id'] for values in groups.values() for r in values]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        reverse, _ = select_records(records[::-1], splits, train_objects=128, val_objects=32, held_views=2, train_views=0)
        self.assertEqual(groups, reverse)

    def test_quick_split_same_objects_disjoint_views_no_unseen_objects(self):
        records = [dict(object_id=f'o{i}', sample_id=f'o{i}_{j:02}') for i in range(25) for j in range(16)]
        splits = dict(train=[f'o{i}' for i in range(20)], val=[f'o{i}' for i in range(20, 25)])
        groups, ids = select_records(records, splits)
        self.assertEqual(len(groups['train']), 16*8)
        self.assertEqual(len(groups['held_view']), 16*4)
        self.assertEqual(groups['val'], [])
        self.assertEqual(ids['val'], [])
        self.assertEqual({r['object_id'] for r in groups['train']},
                         {r['object_id'] for r in groups['held_view']})
        self.assertFalse({r['sample_id'] for r in groups['train']} &
                         {r['sample_id'] for r in groups['held_view']})
        self.assertEqual(groups, select_records(records[::-1], splits)[0])


if __name__ == '__main__': unittest.main()
