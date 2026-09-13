"""CPU checks of permanent visual removal; no pretrained weights or CUDA imports."""
import ast
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import typing
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from test_source_integration import REPO, SmallGenerator, SmallTouch, optimizer, prepared
import train
import evaluate


def native_nodes(relative_path, names, namespace, parent=None):
    tree = ast.parse((REPO / relative_path).read_text())
    nodes = tree.body
    if parent:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == parent).body
    selected = [n for n in nodes if getattr(n, 'name', None) in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(REPO / relative_path), 'exec'), namespace)
    return [namespace[name] for name in names]


# Execute the native classes/methods without importing unrelated GPU dependencies.
FeedForward, = native_nodes('sam3d_objects/model/layers/llama3/ff.py', ['FeedForward'],
                           {'nn': nn, 'F': F, 'Optional': typing.Optional})
EmbedderFuser, = native_nodes('sam3d_objects/model/backbone/dit/embedder/embedder_fuser.py',
    ['EmbedderFuser'], {'torch': torch, 'nn': nn, 'math': math, 'FeedForward': FeedForward,
    'logger': SimpleNamespace(warning=lambda *a: None),
    **{name: getattr(typing, name) for name in ('Optional', 'Tuple', 'List', 'Literal', 'Dict')}})
get_condition_input, embed_condition = native_nodes('sam3d_objects/pipeline/inference_pipeline.py',
    ['get_condition_input', 'embed_condition'], {}, parent='InferencePipeline')


class VisualEmbedder(nn.Module):
    embed_dim = 3

    def forward(self, x):
        return x


NAMES = ('image', 'rgb_image', 'mask', 'rgb_image_mask', 'pointmap', 'rgb_pointmap')


class Pipeline:
    get_condition_input = get_condition_input
    embed_condition = embed_condition

    @staticmethod
    def map_input_keys(inputs, mapping):
        return tuple(inputs[k] for k in mapping)

    def __init__(self, embedded=False):
        self.ss_generator = SmallGenerator()
        self.backbone = self.ss_generator.backbone
        fuser = EmbedderFuser([(VisualEmbedder(), [(name, name) for name in NAMES])],
                             projection_net_hidden_dim_multiplier=1.0)
        fuser.requires_grad_(False).eval()
        self.ss_condition_embedder = None if embedded else fuser
        self.backbone.condition_embedder = fuser if embedded else None

    @property
    def fuser(self):
        return self.ss_condition_embedder if self.ss_condition_embedder is not None else self.backbone.condition_embedder

    def condition(self, inputs):
        return self.get_condition_input(self.fuser, inputs, [])[0]


class NoVisualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.inputs = {name: torch.randn(4, 2, 3) for name in NAMES}

    def test_cli_and_invalid_input_combinations(self):
        base = ['train.py', '--pipeline-config', 'unused.yaml']
        with patch.object(sys, 'argv', base):
            self.assertFalse(train.parse_args().no_visual)
        with patch.object(sys, 'argv', base + ['--no-visual']):
            self.assertTrue(train.parse_args().no_visual)
        for invalid in ('--no-touch', '--joint-pointmap'):
            with patch.object(sys, 'argv', base + ['--no-visual', invalid]):
                with self.assertRaisesRegex(ValueError, 'requires surface conditioning'):
                    train.main()

    def test_native_fuser_training_validation_and_surface_gradients(self):
        pipeline = Pipeline()
        baseline = pipeline.condition(self.inputs)[0]
        self.assertGreater(torch.count_nonzero(baseline).item(), 0)
        train.disable_visual_conditioning(pipeline.fuser)
        model = train.TouchTrainingModel(pipeline.ss_generator, SmallTouch(),
                                         oracle_point_frame=True, no_visual=True)
        batch = list(prepared())
        batch[1] = pipeline.condition(self.inputs)
        self.assertEqual(batch[1][0].shape, baseline.shape)
        self.assertEqual(torch.count_nonzero(batch[1][0]).item(), 0)
        torch.testing.assert_close(batch[1][0], pipeline.condition(
            {name: x * 17 + 53 for name, x in self.inputs.items()})[0], rtol=0, atol=0)
        model(*batch).backward()
        self.assertGreater(model.touch_encoder.output_projection.weight.grad.norm().item(), 0)
        self.assertGreater(torch.count_nonzero(model.generator.tokens).item(), 0)
        self.assertIsNone(model.touch_encoder.encoder.weight.grad)
        args = SimpleNamespace(precision='fp32', joint_pointmap=False, oracle_point_frame=True)
        with patch.object(train, 'prepare_batch', return_value=batch):
            train.validate(pipeline, model, [{'target_shape': batch[0]['shape']}],
                           torch.device('cpu'), args, 29, False, 0)
        self.assertEqual(torch.count_nonzero(model.generator.visual).item(), 0)
        original = model(*batch).detach()
        batch[3] = batch[3] * 5
        self.assertFalse(torch.equal(original, model(*batch).detach()))

    def test_checkpoint_resume_and_inference_context_in_both_embedder_locations(self):
        touch_stub = ModuleType('sam3d_objects.model.backbone.dit.embedder.touch')
        touch_stub.TouchEncoder = SmallTouch
        for no_visual in (False, True):
            model = train.TouchTrainingModel(SmallGenerator(), SmallTouch(),
                                             oracle_point_frame=True, no_visual=no_visual)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'last.pt'
                train.save_checkpoint(path, model, optimizer(model), 1, 3, .2, 'image_touch', 'full')
                checkpoint = torch.load(path, weights_only=False)
                self.assertEqual(checkpoint['conditioning_config'].get('no_visual', False), no_visual)
                if not no_visual:
                    self.assertNotIn('no_visual', checkpoint['conditioning_config'])
                other = train.TouchTrainingModel(SmallGenerator(), SmallTouch(),
                                                 oracle_point_frame=True, no_visual=no_visual)
                train.load_checkpoint(path, other, optimizer(other), 'image_touch', 'full')
                incompatible = train.TouchTrainingModel(SmallGenerator(), SmallTouch(),
                                                        oracle_point_frame=True, no_visual=not no_visual)
                with self.assertRaisesRegex(ValueError, 'conditioning configuration'):
                    train.load_checkpoint(path, incompatible, optimizer(incompatible), 'image_touch', 'full')
            for embedded in (False, True):
                pipeline = Pipeline(embedded)
                before = pipeline.condition(self.inputs)[0].clone()
                # The embedded fuser is frozen in production; only enable the fake backbone weights.
                def enable(encoder, backbone, args):
                    backbone.weight.requires_grad_(True)
                    backbone.bias.requires_grad_(True)
                with patch.dict(sys.modules, {touch_stub.__name__: touch_stub}), patch.object(
                        evaluate, 'build_optimizer', side_effect=enable):
                    restored = evaluate.restore_run(pipeline, checkpoint, torch.device('cpu'))
                after = pipeline.condition(self.inputs)[0]
                if no_visual:
                    self.assertEqual(torch.count_nonzero(after).item(), 0)
                else:
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
                batch = list(prepared()); batch[1] = (after,)
                restored(*batch)
                self.assertGreater(torch.count_nonzero(restored.generator.tokens).item(), 0)


if __name__ == '__main__':
    unittest.main()
