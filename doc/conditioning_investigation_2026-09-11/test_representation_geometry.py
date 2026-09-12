"""CPU checks for geometry used in the GPU handoff."""
import importlib.util
import unittest

import numpy as np
from representation_geometry import point_grid, overlap, sample_mesh, surface_agreement, zero_surface


class GeometryTests(unittest.TestCase):
    def test_grid_axes_and_boundaries(self):
        points = np.array([[-.5, -.5, -.5], [.5, .5, .5], [-.25, 0, .25]])
        grid = point_grid(points)
        self.assertEqual(int(grid.sum()), 3)
        self.assertTrue(grid[0, 0, 0] and grid[63, 63, 63] and grid[16, 32, 48])
        with self.assertRaises(ValueError):
            point_grid(np.array([[.51, 0, 0]]))
        with self.assertRaises(ValueError):
            point_grid(np.array([[np.nan, 0, 0]]))

    def test_overlap_distinguishes_missing_and_extra(self):
        a = np.zeros((3, 3, 3), dtype=bool); a[0, 0, 0] = True
        b = a.copy(); b[1, 1, 1] = True
        self.assertEqual(overlap(a, b)['precision'], 1.)
        self.assertEqual(overlap(a, b)['recall'], .5)
        self.assertEqual(overlap(b, a)['precision'], .5)
        self.assertEqual(overlap(b, a)['recall'], 1.)

    def test_triangle_sampling_and_surface_metric(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0]])
        faces = np.array([[0, 1, 2]])
        p = sample_mesh(vertices, faces, count=10000, seed=3)
        np.testing.assert_array_equal(p, sample_mesh(vertices, faces, count=10000, seed=3))
        self.assertTrue((p >= 0).all() and (p[:, :2].sum(1) <= 1 + 1e-12).all())
        np.testing.assert_allclose(p.mean(0), [1 / 3, 1 / 3, 0], atol=.01)
        self.assertEqual(surface_agreement(p, p)['fscore'], 1.)
        self.assertEqual(surface_agreement(p + [0, 0, 1], p)['fscore'], 0.)

    @unittest.skipUnless(importlib.util.find_spec('skimage'), 'scikit-image is available on the cluster, absent in this local test environment')
    def test_isosurface_axis_order_and_scale(self):
        axis = np.linspace(-1.05, 1.05, 33)
        x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
        vertices, _ = zero_surface((x - .23).astype(np.float32))
        np.testing.assert_allclose(vertices[:, 0], .23, atol=1e-6)
        np.testing.assert_allclose(vertices[:, 1:].min(0), [-1.05, -1.05], atol=1e-6)
        np.testing.assert_allclose(vertices[:, 1:].max(0), [1.05, 1.05], atol=1e-6)
        self.assertIsNone(zero_surface(np.ones((4, 4, 4))))


if __name__ == '__main__':
    unittest.main()
