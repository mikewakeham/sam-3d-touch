"""Reference-gate regressions including thin and collapsed triangles."""
import unittest
import numpy as np
from frame_probe_core import triangle_witness, sampled_mesh_check


class MeshTests(unittest.TestCase):
    def test_known_triangle_distance_and_collapsed_faces(self):
        t = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]],
                      [[0., 0, 0], [1, 0, 0], [0, 0, 0]],
                      [[0., 0, 0], [0, 0, 0], [0, 0, 0]]])
        p = np.array([[.25, .25, .3], [.5, .4, 0], [.3, .4, 0]])
        distances = np.linalg.norm(p-triangle_witness(p, t), axis=1)
        np.testing.assert_allclose(distances, [.3, .4, .5], atol=1e-15)

    def test_thin_triangles_with_transform_and_float32_storage(self):
        rng = np.random.default_rng(29)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        t = np.array([[[0., 0, 0], [1, 0, 0], [.3, width, 0]]
                      for width in (1., 1e-3, 1e-6, 1e-9, 1e-12)]) @ rotation.T
        weights = np.tile([.1, .3, .6], (len(t), 1))
        sampled = np.sum(weights[:, :, None]*t, axis=1)
        camera = (sampled @ rotation.T + [0, 0, 2]).astype(np.float32)
        recovered = (camera.astype(float)-[0, 0, 2]) @ rotation
        result, witness = sampled_mesh_check(recovered, sampled, t)
        self.assertTrue(result['point_to_mesh_passed'])
        self.assertLess(result['point_to_mesh_max_distance'], 2e-7)
        self.assertTrue(np.isfinite(witness).all())

    def test_wrong_frames_and_wrong_seed_correspondence_rejected(self):
        t = np.tile([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]], (2, 1, 1))
        p = np.array([[.2, .1, 0], [.6, .2, 0]])
        self.assertTrue(sampled_mesh_check(p, p, t)[0]['point_to_mesh_passed'])
        for wrong in (p+[0, 0, .001], p*1.1, p[:, [1, 0, 2]], p[::-1]):
            self.assertFalse(sampled_mesh_check(wrong, p, t)[0]['point_to_mesh_passed'])
        # Identical recorded samples cannot excuse a point lying off the actual face.
        wrong = p+[0, 0, .001]
        self.assertFalse(sampled_mesh_check(wrong, wrong, t)[0]['point_to_mesh_passed'])


if __name__ == '__main__':
    unittest.main()
