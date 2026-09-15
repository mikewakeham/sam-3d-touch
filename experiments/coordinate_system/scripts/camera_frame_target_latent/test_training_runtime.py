"""CPU tests for effective batch size, RNG isolation, and both actual preprocessor paths."""
import ast
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Callable, Optional
import unittest
import warnings

import numpy as np
import torch

REPO = next(p for p in Path(__file__).resolve().parents if (p/'train.py').is_file())
sys.path.insert(0, str(REPO))
from experiments.coordinate_system.scripts.camera_frame_target_latent.training_runtime import (
    SurfaceNormalizer, preprocess_inputs, fixed_rng, training_indices, accumulate_update)
from train import preprocess_batch as preprocess_production_batch


def source_class(path, name, namespace):
    # Execute the actual class with lightweight import substitutes, not a copied implementation.
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class RuntimeTests(unittest.TestCase):
    def test_accumulation_equals_batch16_and_one_update(self):
        torch.manual_seed(4)
        x, y = torch.randn(16, 3), torch.randn(16, 2)
        whole = torch.nn.Linear(3, 2)
        micro = torch.nn.Linear(3, 2); micro.load_state_dict(whole.state_dict())
        opt1 = torch.optim.AdamW(whole.parameters(), lr=.01, weight_decay=0)
        opt2 = torch.optim.AdamW(micro.parameters(), lr=.01, weight_decay=0)
        loss = (whole(x)-y).square().mean(); loss.backward()
        torch.nn.utils.clip_grad_norm_(whole.parameters(), 1.); opt1.step()
        batches = [dict(x=x[i:i+4], target_shape=y[i:i+4]) for i in range(0, 16, 4)]
        measured, _ = accumulate_update(batches, lambda b: (micro(b['x'])-b['target_shape']).square().mean(),
                                       opt2, list(micro.parameters()), 16, 1.)
        self.assertAlmostEqual(measured, float(loss.detach()), places=6)
        for a, b in zip(whole.parameters(), micro.parameters()): torch.testing.assert_close(a, b)
        self.assertTrue(all(s['step'] == 1 for s in opt2.state.values()))

    def test_evaluation_preserves_rng_even_on_exception(self):
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        py, npstate, ts = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with fixed_rng(29, torch.device('cpu')):
                random.random(); np.random.rand(); torch.rand(12)
                raise RuntimeError('fixture')
        self.assertEqual(py, random.getstate())
        np.testing.assert_equal(npstate, np.random.get_state())
        torch.testing.assert_close(ts, torch.get_rng_state())
        with fixed_rng(37, torch.device('cpu')): a = torch.rand(12)
        torch.rand(99)
        with fixed_rng(37, torch.device('cpu')): b = torch.rand(12)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_sampler_full_epochs_and_exact_sample_budget(self):
        a = list(training_indices(7, 32, 29))
        self.assertEqual(a, list(training_indices(7, 32, 29)))
        self.assertEqual(len(a), 32)
        for start in range(0, 28, 7): self.assertEqual(sorted(a[start:start+7]), list(range(7)))

    def test_both_pointmap_paths_use_surface_frame_and_restore_stock(self):
        class Stock:
            def normalize(self, pm, mask, scale=None, shift=None):
                return pm+20, torch.ones(3), torch.full((3,), -20.)
        ns = dict(torch=torch, dataclass=dataclass, Callable=Callable, Optional=Optional,
                  warnings=warnings, SSIPointmapNormalizer=Stock,
                  logger=SimpleNamespace(warning=lambda *a: None))
        cls = source_class(REPO/'sam3d_objects/data/dataset/tdfy/preprocessor.py', 'PreProcessor', ns)
        # A real spatial crop must act AFTER normalization; full-image XYZ stays on its own grid.
        def crop(rgb, mask, pointmap): return rgb[:, 1:, 1:], mask[:, 1:, 1:], pointmap[:, 1:, 1:]
        processor = cls(normalize_pointmap=True, pointmap_normalizer=Stock(),
                        rgb_pointmap_normalizer=Stock(), img_mask_pointmap_joint_transform=[crop])
        class Pipeline:
            ss_preprocessor = processor
            def preprocess_image(self, image, preprocessor, pointmap):
                rgb = torch.from_numpy(image).permute(2, 0, 1).float()/255
                return {k: v[None] for k, v in preprocessor._process_image_mask_pointmap_mess(
                    rgb[:3], rgb[3:], pointmap).items()}
        surface = torch.tensor([[-1., 0., 2.], [3., 2., 4.], [0., -2., 3.]])
        pm = torch.arange(36, dtype=torch.float32).reshape(3, 4, 3)/10
        batch = dict(image=torch.full((1, 3, 4, 4), 255, dtype=torch.uint8),
                     pointmap=pm[None], touch_xyz=surface[None], touch_mask=torch.ones((1, 3), dtype=torch.bool))
        before = preprocess_inputs(Pipeline(), batch, False)
        shared = preprocess_inputs(Pipeline(), batch, True)
        production = preprocess_production_batch(
            Pipeline(), batch['image'], batch['pointmap'],
            batch['touch_xyz'], batch['touch_mask'], True,
        )
        after = preprocess_inputs(Pipeline(), batch, False)
        center = (surface.min(0).values+surface.max(0).values)/2
        radius = torch.linalg.vector_norm(surface-center, dim=1).max()
        expected = ((pm-center)/radius).permute(2, 0, 1)
        torch.testing.assert_close(shared['pointmap'][0], expected[:, 1:, 1:])
        torch.testing.assert_close(shared['rgb_pointmap'][0], expected)
        for key in before: torch.testing.assert_close(before[key], after[key])
        for key in shared: torch.testing.assert_close(shared[key], production[key])
        for key in ('image', 'rgb_image', 'mask', 'rgb_image_mask'):
            torch.testing.assert_close(before[key], shared[key])
        torch.testing.assert_close(surface, batch['touch_xyz'][0])
        # Check against the REAL VecSetX normalization method, without loading weights/CUDA ops.
        source = REPO/'sam3d_objects/model/backbone/dit/embedder/touch.py'
        tree = ast.parse(source.read_text())
        touch = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TouchEncoder')
        method = next(n for n in touch.body if isinstance(n, ast.FunctionDef) and n.name == 'normalize_points_for_vecsetx')
        ns = {'torch': torch}; exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
        vec, _, _ = ns[method.name](None, surface[None], batch['touch_mask'])
        norm = SurfaceNormalizer(surface)
        normalized, _, _ = norm.normalize(surface.T[:, None, :], None)
        torch.testing.assert_close(vec[0], normalized[:, 0, :].T)

    def test_degenerate_normalization_rejects(self):
        with self.assertRaises(ValueError): SurfaceNormalizer(torch.zeros(4, 3))
        with self.assertRaises(ValueError): SurfaceNormalizer(torch.full((4, 3), float('nan')))


if __name__ == '__main__': unittest.main()
