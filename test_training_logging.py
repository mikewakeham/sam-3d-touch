"""CPU checks that scalar diagnostics preserve forwards, gradients, and RNG state."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from train import component_gradient_norms, token_magnitudes


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_name = 'test'
        self.encoder = nn.Linear(3, 2)
        self.output_projection = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 4))
        self.position_projection = nn.Linear(3, 4)
        self.position_projection.requires_grad_(False)
        self.touch_embedding = nn.Parameter(torch.ones(1, 1, 4))

    def forward(self, points, mask=None):
        return self.output_projection(self.encoder(points)) + self.touch_embedding


def model_fixture():
    encoder = Encoder()
    attention = nn.Module()
    attention.to_kv = nn.Linear(4, 4)
    backbone = SimpleNamespace(blocks=[SimpleNamespace(
        cross_attn={'shape': attention}, norm2={'shape': nn.LayerNorm(4)})])
    return SimpleNamespace(touch_encoder=encoder,
                           generator=SimpleNamespace(reverse_fn=SimpleNamespace(backbone=backbone)))


class LoggingTests(unittest.TestCase):
    def test_training_loop_logs_scalars_at_existing_intervals(self):
        import train
        fixture = model_fixture()
        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.reverse_fn = fixture.generator.reverse_fn
            def loss(self, targets, *args, touch_tokens, **kwargs):
                return touch_tokens.square().mean(), {}
        model = train.TouchTrainingModel(Generator(), fixture.touch_encoder)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD([{'params': parameters, 'name': 'touch_encoder'}], lr=.01)
        args = SimpleNamespace(precision='fp32', joint_pointmap=False, oracle_point_frame=False,
                               log_every=3, gradient_clip=1., batch_size=2, epochs=1, max_steps=0)
        prepared = ({}, (torch.ones(2, 7, 4),), {}, torch.randn(2, 5, 3), torch.ones(2, 5, dtype=torch.bool))
        rows = []
        with patch('train.prepare_batch', return_value=prepared):
            step = train.train_epoch(None, model, model, [{'target_shape': torch.zeros(2, 1)}] * 4,
                                     optimizer, parameters, torch.device('cpu'), args, 0, 0, 4, 1,
                                     False, True, SimpleNamespace(log=rows.append))
        self.assertEqual(step, 4)
        self.assertEqual([row['global_step'] for row in rows], [1, 3, 4])
        for row in rows:
            self.assertIn('tokens/surface_before_projection_rms', row)
            self.assertIn('tokens/surface_after_projection_rms', row)
            self.assertIn('tokens/visual_rms', row)
            self.assertIn('parameters/touch_output_projection_trainable_norm', row)
            self.assertIn('gradients/touch_output_projection', row)
        self.assertFalse(model.touch_encoder._forward_hooks)

    def test_rms_and_unchanged_forward_gradients_rng(self):
        torch.manual_seed(29)
        model = model_fixture()
        reference = copy.deepcopy(model.touch_encoder)
        points = torch.randn(2, 5, 3)
        visual = torch.full((2, 7, 4), 3.)
        prepared = (None, (visual,), {})
        before = torch.get_rng_state().clone()
        with token_magnitudes(model, prepared, torch.tensor([False, True])) as metrics:
            output = model.touch_encoder(points)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        expected = reference(points)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.square().mean().backward()
        expected.square().mean().backward()
        for actual, original in zip(model.touch_encoder.parameters(), reference.parameters()):
            if original.grad is not None:
                torch.testing.assert_close(actual.grad, original.grad, rtol=0, atol=0)
        raw = reference.encoder(points)
        projected = reference.output_projection(raw)
        self.assertAlmostEqual(metrics['tokens/surface_before_projection_rms'], raw.square().mean().sqrt().item())
        self.assertAlmostEqual(metrics['tokens/surface_after_projection_rms'], projected.square().mean().sqrt().item())
        self.assertAlmostEqual(metrics['tokens/surface_conditioning_rms'], expected.square().mean().sqrt().item())
        self.assertEqual(metrics['tokens/visual_rms'], 3.)
        self.assertAlmostEqual(metrics['tokens/visual_after_dropout_rms'], (9 / 2)**.5, places=6)
        self.assertTrue(all(isinstance(value, float) for value in metrics.values()))
        self.assertFalse(model.touch_encoder._forward_hooks)
        self.assertFalse(model.touch_encoder.output_projection._forward_hooks)

    def test_disabled_image_only_and_exception_cleanup(self):
        model = model_fixture()
        prepared = (None, (torch.ones(1, 2, 4),), {})
        with token_magnitudes(model, prepared, enabled=False) as metrics:
            model.touch_encoder(torch.ones(1, 2, 3))
            self.assertFalse(model.touch_encoder._forward_hooks)
        self.assertEqual(metrics, {})
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with token_magnitudes(model, prepared):
                raise RuntimeError('test failure')
        self.assertFalse(model.touch_encoder._forward_hooks)
        self.assertFalse(model.touch_encoder.output_projection._forward_hooks)
        model.touch_encoder = None
        with token_magnitudes(model, prepared) as metrics:
            pass
        self.assertEqual(metrics, {'tokens/visual_rms': 1.})

    def test_parameter_norms_match_trainable_weights_and_gradients(self):
        model = model_fixture()
        model.touch_encoder.encoder.requires_grad_(False)
        model.touch_encoder(torch.ones(1, 2, 3)).square().mean().backward()
        gradient_only = component_gradient_norms(model)
        metrics = component_gradient_norms(model, include_parameters=True)
        for key, value in gradient_only.items():
            self.assertEqual(metrics[key], value)
        weights = list(model.touch_encoder.output_projection.parameters())
        expected = torch.cat([parameter.detach().flatten() for parameter in weights]).norm().item()
        self.assertAlmostEqual(metrics['parameters/touch_output_projection_trainable_norm'], expected, places=6)
        self.assertNotIn('parameters/test_encoder_trainable_norm', metrics)
        self.assertNotIn('parameters/touch_position_projection_trainable_norm', metrics)
        self.assertEqual(metrics['gradients/test_encoder'], 0.)


if __name__ == '__main__':
    unittest.main()
