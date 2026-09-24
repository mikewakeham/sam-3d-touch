"""CPU checks of encoder initialization and saved trainable weights."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ast
from typing import Optional
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent


def load_touch_class(constructor=None):
    # Exercise production methods without importing GPU-only attention dependencies.
    namespace = dict(torch=torch, nn=nn, F=F, Optional=Optional)
    exec((ROOT / 'sam3d_objects/model/layers/llama3/ff.py').read_text(), namespace)
    tree = ast.parse((ROOT / 'sam3d_objects/model/backbone/dit/embedder/touch.py').read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) or
                   (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id == 'VECSETX_ENCODE_PARAMETER_PREFIXES' for t in node.targets))]
    namespace['ENCODERS'] = {'vecsetx': {'constructor': constructor or SmallEncoder}}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<production touch>', 'exec'), namespace)
    return namespace['TouchEncoder']



def load_vecsetx_namespace(directory):
    import math
    import numpy as np
    from einops import rearrange, repeat
    namespace = dict(torch=torch, nn=nn, F=F, np=np, math=math,
                     rearrange=rearrange, repeat=repeat, einsum=torch.einsum)
    # Skip imports of unavailable CUDA packages; tests use the real masked SDPA path.
    for filename in ['utils.py', 'bottleneck.py', 'autoencoder.py']:
        tree = ast.parse((directory / filename).read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(directory / filename), 'exec'), namespace)
    return namespace


def small_real_encoder():
    namespace = load_vecsetx_namespace(ROOT / 'sam3d_objects/model/backbone/dit/embedder/vecsetx')
    return namespace['create_autoencoder'](
        depth=2, dim=64, M=8, N=16, query_type='learnable',
        bottleneck=namespace['NormalizedBottleneck'],
        bottleneck_args={'dim': 64, 'latent_dim': 8},
    )


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_inputs = 8
        self.point_embed = nn.Linear(3, 4)
        self.bottleneck = nn.Module()
        self.bottleneck.pre_bottleneck_proj = nn.Linear(4, 2)
        self.unused_decoder = nn.Linear(2, 4)

    def encode(self, points, mask):
        return {'x': self.bottleneck.pre_bottleneck_proj(self.point_embed(points))}


class InitializationTests(unittest.TestCase):
    def test_scratch_trains_and_restores_without_pretrained_file(self):
        TouchEncoder = load_touch_class()
        with patch('torch.load', side_effect=AssertionError('Scratch must not load pretrained weights')):
            model = TouchEncoder(output_dim=4, trainable=True, pretrained=False, use_position=False)
            points = torch.randn(2, 8, 3)
            optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=.01)
            before = model.encoder.point_embed.weight.detach().clone()
            model(points).square().mean().backward()
            self.assertIsNotNone(model.encoder.point_embed.weight.grad)
            self.assertFalse(model.encoder.unused_decoder.weight.requires_grad)
            optimizer.step()
            self.assertFalse(torch.equal(before, model.encoder.point_embed.weight))
            state = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
            restored = TouchEncoder(**model.get_config())
            with torch.no_grad():
                for name, parameter in restored.named_parameters():
                    if parameter.requires_grad:
                        parameter.copy_(state[name])
            torch.testing.assert_close(model(points), restored(points), rtol=0, atol=0)
            self.assertIs(model.get_config()['pretrained'], False)

    def test_default_loads_pretrained_and_freezes_encoder(self):
        TouchEncoder = load_touch_class()
        weights = SmallEncoder().state_dict()
        with patch('torch.load', return_value={'model': weights}) as load:
            model = TouchEncoder(output_dim=4, use_position=False)
        load.assert_called_once()
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(model.output_projection[1].w1.weight.requires_grad)
        torch.testing.assert_close(model.encoder.point_embed.weight, weights['point_embed.weight'])
        self.assertNotIn('pretrained', model.get_config())  # Old checkpoint metadata stays compatible.

    def test_actual_vecsetx_encoder_gradients_and_restore(self):
        torch.set_num_threads(1)
        TouchEncoder = load_touch_class(small_real_encoder)
        with patch('torch.load', side_effect=AssertionError('Unexpected pretrained load')):
            model = TouchEncoder(output_dim=16, trainable=True, pretrained=False, use_position=False)
            points = torch.randn(2, 16, 3)
            output = model(points)
            self.assertTrue(torch.isfinite(output).all())
            (output - torch.randn_like(output)).square().mean().backward()
            for name, parameter in model.encoder.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                else:
                    self.assertIsNone(parameter.grad, name)
            self.assertGreater(model.encoder.latents.weight.grad.abs().sum().item(), 0)
            state = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
            restored = TouchEncoder(**model.get_config())
            with torch.no_grad():
                for name, parameter in restored.named_parameters():
                    if parameter.requires_grad:
                        parameter.copy_(state[name])
            torch.testing.assert_close(model(points), restored(points), rtol=0, atol=0)
            encoder = model.encoder
            self.assertEqual(torch.count_nonzero(encoder.to_outputs[1].weight).item(), 0)
            self.assertEqual(torch.count_nonzero(encoder.to_outputs[1].bias).item(), 0)
            self.assertFalse(encoder.bottleneck.pre_bottleneck_norm.elementwise_affine)

    def test_invalid_scratch_modes(self):
        TouchEncoder = load_touch_class()
        for kwargs in [dict(trainable=False), dict(trainable=True, use_learn=True)]:
            with self.assertRaises(ValueError):
                TouchEncoder(pretrained=False, **kwargs)


if __name__ == '__main__':
    unittest.main()
