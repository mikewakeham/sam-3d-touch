"""Cluster preflight: spatial ordering, raw-set symmetry, and gradient reach."""
import unittest
import torch
from fit_feature_readout import SpatialReadout


class ReadoutTests(unittest.TestCase):
    def test_shared_initialization_and_spatial_order(self):
        torch.manual_seed(29); a = SpatialReadout(32)
        torch.manual_seed(29); b = SpatialReadout(1024)
        for name, value in a.named_parameters():
            if not name.startswith('input.'):
                torch.testing.assert_close(value, dict(b.named_parameters())[name], rtol=0, atol=0)
        torch.testing.assert_close(a.xyz_features[1, :3] - a.xyz_features[0, :3], torch.tensor([0., 0., 1/16]))
        torch.testing.assert_close(a.xyz_features[16, :3] - a.xyz_features[0, :3], torch.tensor([0., 1/16, 0.]))
        torch.testing.assert_close(a.xyz_features[256, :3] - a.xyz_features[0, :3], torch.tensor([1/16, 0., 0.]))

    def test_raw_permutation_symmetry_and_slot_sensitivity(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(29)
        model = SpatialReadout(32, width=32).to(device).eval()
        x = torch.randn(1, 1024, 32, device=device)
        with torch.no_grad():
            normal = model(x); permuted = model(x.flip(1))
            torch.testing.assert_close(normal, permuted, rtol=1e-4, atol=2e-6)
            model.slot_identity = True
            delta = (model(x) - model(x.flip(1))).abs().max()
            self.assertGreater(float(delta), 1e-5)
        self.assertEqual(tuple(normal.shape), (1, 4096, 8))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = model.input[1].weight.detach().clone()
        model(x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertGreater(float((before - model.input[1].weight).abs().max()), 0.)


if __name__ == '__main__':
    unittest.main()
