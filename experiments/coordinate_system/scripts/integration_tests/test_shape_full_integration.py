"""CPU checks using SAM3D's actual dense transformer at reduced width.

No pretrained weights/CUDA are loaded. Optional logging/config/tree packages
are stubbed only for import; the exercised math uses the source model + Torch.
Run this file directly, separately from suites needing those optional packages.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'train.py').is_file())
sys.path.insert(0, str(REPO))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import train

for name in ('optree', 'astor', 'loguru', 'omegaconf'):
    if importlib.util.find_spec(name) is None:
        module = ModuleType(name)
        if name == 'loguru':
            module.logger = SimpleNamespace(info=lambda *args: None)
        if name == 'omegaconf':
            module.OmegaConf = SimpleNamespace()
        sys.modules[name] = module
# Skip the models package's eager imports of unrelated model families.
package = 'sam3d_objects.model.backbone.tdfy_dit.models'
module = ModuleType(package)
module.__path__ = [str(REPO / 'sam3d_objects/model/backbone/tdfy_dit/models')]
sys.modules[package] = module
from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import SparseStructureFlowTdfyWrapper
from sam3d_objects.model.backbone.tdfy_dit.models.mm_latent import Latent
from experiments.coordinate_system.scripts.integration_tests.test_source_integration import SmallTouch


class Touch(SmallTouch):
    def get_trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)


class Generator(nn.Module):
    def __init__(self, share_mod=False):
        super().__init__()
        self.reverse_fn = nn.Module()
        self.reverse_fn.backbone = SparseStructureFlowTdfyWrapper(
            latent_mapping={name: Latent(3, 12, lambda: torch.zeros(n, 12))
                            for name, n in [('shape', 8), ('translation', 1)]},
            in_channels=3, out_channels=3, model_channels=12, cond_channels=3,
            num_blocks=2, num_heads=2, qk_rms_norm=True, qk_rms_norm_cross=True,
            is_shortcut_model=True, share_mod=share_mod,
        )
        # Pretrained modulation is not identically zero; exercise upstream grads.
        for name, p in self.reverse_fn.backbone.named_parameters():
            if 'adaLN_modulation' in name or name.startswith('d_embedder.mlp.2'):
                nn.init.normal_(p, std=.1)

    def loss(self, targets, visual, touch_tokens=None):
        backbone = self.reverse_fn.backbone
        prediction = backbone({k: v * .5 + .1 for k, v in targets.items()},
                              torch.tensor([.2, .8]), visual,
                              d=torch.tensor([.1, .3]), touch_tokens=touch_tokens)
        return (prediction['shape'] - targets['shape']).square().mean(), {}


def setup(scope='shape_full', constant=False, image=False, share_mod=False):
    model = train.TouchTrainingModel(Generator(share_mod), None if image else Touch(),
                                    no_pointmap=True, oracle_point_frame=not image,
                                    constant_touch=constant, visual_dropout=0. if image else .5)
    if constant:
        model.load_constant_touch({'features': torch.randn(1, 5, 2), 'sample_id': 'constant'})
    args = SimpleNamespace(train_scope=scope, cross_attention_scope='full',
                           learning_rate=.001, cross_attention_learning_rate=.001)
    optimizer, parameters = train.build_optimizer(model.touch_encoder, model.generator.reverse_fn.backbone, args)
    return model, optimizer, parameters


def batch():
    return ({'shape': torch.randn(2, 8, 3), 'translation': torch.randn(2, 1, 3)},
            (torch.randn(2, 7, 3),), {}, torch.randn(2, 5, 3), torch.ones(2, 5, dtype=torch.bool))


def ddp_worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        for image, constant in [(True, False), (False, False), (False, True)]:
            torch.manual_seed(29 + rank)
            model, opt, parameters = setup(constant=constant, image=image)
            if constant:
                dist.broadcast(model.constant_touch_features, src=0)
            wrapped = nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
            for step in range(3):
                opt.zero_grad(set_to_none=True)
                drop = train.make_visual_drop_mask(2, 0. if image else .5, torch.device('cpu'), 29, step, rank)
                wrapped(*batch(), visual_drop_mask=drop).backward()
                opt.step()
            weights = torch.cat([p.detach().flatten() for p in parameters])
            gathered = [torch.empty_like(weights) for _ in range(2)]
            dist.all_gather(gathered, weights)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class ShapeFullTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_cli_and_legacy_scope(self):
        base = ['train.py', '--pipeline-config', 'unused']
        for flags, expected in [([], 'shape_cross_attention'),
                                (['--train-scope', 'shape_full'], 'shape_full'),
                                (['--cross-attention-scope', 'full'], 'shape_cross_attention'),
                                (['--cross-attention-scope', 'kv'], 'shape_cross_attention_kv')]:
            with patch.object(sys, 'argv', base + flags):
                self.assertEqual(train.parse_args().train_scope, expected)
        self.assertEqual(train.checkpoint_train_scope({}), 'shape_cross_attention_kv')
        self.assertEqual(train.checkpoint_train_scope({'cross_attention_scope': 'full'}), 'shape_cross_attention')

    def test_selected_parameters_gradients_and_frozen_layout(self):
        for scope in ('shape_cross_attention_kv', 'shape_cross_attention', 'shape_full'):
            model, opt, parameters = setup(scope)
            backbone = model.generator.reverse_fn.backbone
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            self.assertEqual(len(parameters), len(set(map(id, parameters))))
            for name, p in backbone.named_parameters():
                if scope == 'shape_full':
                    expected = '.shape.' in name or name.startswith(('t_embedder.', 'd_embedder.')) or '.adaLN_modulation.' in name
                elif scope == 'shape_cross_attention':
                    expected = '.cross_attn.shape.' in name or '.norm2.shape.' in name
                else:
                    expected = '.cross_attn.shape.to_kv.' in name
                self.assertEqual(p.requires_grad, expected, name)
            data = batch()
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                model(*data).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
                opt.step()
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    torch.testing.assert_close(before[name], p, rtol=0, atol=0)
            if scope == 'shape_full':
                for name, modules in train.shape_path_module_groups(backbone).items():
                    self.assertGreater(train.gradient_norm(p for m in modules for p in m.parameters()), 0, name)

    def test_all_three_variants_checkpoint_and_evaluation_restoration(self):
        import evaluate
        for image, constant in [(True, False), (False, False), (False, True)]:
            model, opt, _ = setup(constant=constant, image=image)
            data = batch()
            model(*data).backward(); opt.step(); opt.zero_grad(set_to_none=True)
            mode = 'image' if image else 'image_touch'
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'last.pt'
                train.save_checkpoint(path, model, opt, 1, 3, .2, mode, 'full', 'shape_full')
                saved = torch.load(path, weights_only=False)
                self.assertEqual(saved['train_scope'], 'shape_full')
                restored, other, _ = setup(constant=constant, image=image)
                self.assertEqual(train.load_checkpoint(path, restored, other, mode, 'full', 'shape_full'), (1, 3, .2))
                torch.testing.assert_close(model(*data), restored(*data), rtol=0, atol=0)
                self.assertEqual(len(opt.state), len(other.state))
                with self.assertRaisesRegex(ValueError, 'scope'):
                    train.load_checkpoint(path, restored, other, mode, 'full')
                # A fresh copy supplies identical frozen pretrained parameters.
                generator = copy.deepcopy(model.generator)
                for p in generator.parameters():
                    if p.requires_grad:
                        nn.init.zeros_(p)
                fuser = SimpleNamespace(embedder_list=[(None, [('pointmap', None), ('rgb_pointmap', None)])], force_drop_modalities=[])
                pipeline = SimpleNamespace(ss_generator=generator, backbone=generator.reverse_fn.backbone, ss_condition_embedder=fuser)
                stub = ModuleType('sam3d_objects.model.backbone.dit.embedder.touch'); stub.TouchEncoder = Touch
                with patch.dict(sys.modules, {'sam3d_objects.model.backbone.dit.embedder.touch': stub}):
                    evaluated = evaluate.restore_run(pipeline, saved, torch.device('cpu'))
                torch.testing.assert_close(model(*data), evaluated(*data), rtol=0, atol=0)
                model(*data).backward(); opt.step()
                restored(*data).backward(); other.step()
                torch.testing.assert_close(model(*data), restored(*data), rtol=0, atol=0)

    def test_shared_modulation_and_shape_protection(self):
        model, opt, _ = setup(share_mod=True)
        backbone = model.generator.reverse_fn.backbone
        self.assertTrue(backbone.adaLN_modulation[1].weight.requires_grad)
        data = batch()
        prediction = model(*data)
        changed = ({**data[0], 'translation': data[0]['translation'] * 100}, *data[1:])
        torch.testing.assert_close(prediction, model(*changed), rtol=0, atol=0)
        prediction.backward(); opt.step()
        backbone.blocks[0].self_attn.protect_modality_list = []
        with self.assertRaisesRegex(ValueError, 'protected'):
            train.shape_path_module_groups(backbone)

    def test_two_rank_shape_full_all_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(ddp_worker, args=(str(Path(tmp) / 'rendezvous'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
