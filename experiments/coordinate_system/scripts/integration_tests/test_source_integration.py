"""Real Torch tests of production training plumbing; no SAM3D weights/GPU needed."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

REPO = next(p for p in Path(__file__).resolve().parents if (_source_path(p, 'train.py')).is_file() and (p / 'sam3d_objects').is_dir())
sys.path.insert(0, str(REPO))
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import train


class SmallVec(nn.Linear):
    def __init__(self):
        super().__init__(3, 2, bias=False)
        # Model restoration reuses identical pretrained frozen weights.
        with torch.no_grad():
            self.weight.copy_(torch.arange(6).reshape(2, 3) / 6)
        self.requires_grad_(False)

    def encode(self, points, mask):
        return {'x': self(points)}


class SmallTouch(nn.Module):
    def __init__(self, use_position=False):
        super().__init__()
        self.encoder = SmallVec()
        self.output_projection = nn.Linear(2, 3)
        self.touch_embedding = nn.Parameter(torch.ones(1, 1, 3))

    def prepare_points(self, points, mask):
        return points, mask, None, None

    def forward(self, points, mask):
        return self.output_projection(self.encoder.encode(points, mask)['x']) + self.touch_embedding

    def get_config(self):
        return {'use_position': False}


class SmallGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 1)
        self.reverse_fn = SimpleNamespace(backbone=self.backbone)

    def loss(self, targets, visual, touch_tokens=None):
        self.visual = visual.detach().clone()
        self.tokens = touch_tokens.detach().clone() if touch_tokens is not None else None
        context = visual.mean(1)
        if touch_tokens is not None:
            context = context + touch_tokens.mean(1)
        return (self.backbone(context) - targets['shape']).square().mean(), {}


def make_model(constant=False, dropout=0.):
    model = train.TouchTrainingModel(SmallGenerator(), SmallTouch(), oracle_point_frame=True,
                                    visual_dropout=dropout, constant_touch=constant)
    if constant:
        model.load_constant_touch({'sample_id': 'reference_000', 'features': torch.randn(1, 5, 2)})
    return model


def prepared():
    return ({'shape': torch.randn(4, 1)}, (torch.randn(4, 7, 3),), {},
            torch.randn(4, 5, 3), torch.ones(4, 5, dtype=torch.bool))


def optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)


def ddp_worker(rank, init_file):
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=2)
    try:
        for constant in (False, True):
            torch.manual_seed(29 + rank)
            model = make_model(constant, .5)
            if constant:
                dist.broadcast(model.constant_touch_features, src=0)
            wrapped = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
            opt = optimizer(model)
            for step in range(3):
                opt.zero_grad(set_to_none=True)
                batch = prepared()
                mask = train.make_visual_drop_mask(4, .5, torch.device('cpu'), 29, step, rank)
                wrapped(*batch, visual_drop_mask=mask).backward(); opt.step()
            weights = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
            gathered = [torch.empty_like(weights) for _ in range(2)]
            dist.all_gather(gathered, weights)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
            if constant:
                gathered = [torch.empty_like(model.constant_touch_features) for _ in range(2)]
                dist.all_gather(gathered, model.constant_touch_features)
                torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
            del wrapped, model, opt
    finally:
        dist.destroy_process_group()


class SourceIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_new_cli_defaults_to_full_scope(self):
        with patch.object(sys, 'argv', ['train.py', '--pipeline-config', 'unused.yaml']):
            args = train.parse_args()
        self.assertEqual(args.cross_attention_scope, 'full')
        self.assertEqual(args.visual_dropout, 0.)
        self.assertFalse(args.constant_touch)
        with patch.object(sys, 'argv', ['train.py', '--pipeline-config', 'unused.yaml', '--cross-attention-scope', 'kv']):
            self.assertEqual(train.parse_args().cross_attention_scope, 'kv')

    def test_mask_is_per_sample_and_preserves_rng(self):
        state = torch.random.get_rng_state().clone()
        a = train.make_visual_drop_mask(1024, .5, torch.device('cpu'), 29, 200, 0)
        torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)
        b = train.make_visual_drop_mask(1024, .5, torch.device('cpu'), 29, 200, 0)
        torch.testing.assert_close(a, b)
        self.assertTrue(a.any() and (~a).any())
        self.assertFalse(torch.equal(a, train.make_visual_drop_mask(1024, .5, torch.device('cpu'), 29, 200, 1)))
        self.assertIsNone(train.make_visual_drop_mask(4, 0., torch.device('cpu'), 29, 0, 0))
        self.assertTrue(train.make_visual_drop_mask(4, 1., torch.device('cpu'), 29, 0, 0).all())

    def test_mixed_visual_dropout_preserves_surface_and_gradients(self):
        model, batch = make_model(dropout=.5), prepared()
        visual = batch[1][0].clone()
        expected_tokens = model.get_touch_tokens(batch[3], batch[4]).detach()
        mask = torch.tensor([True, False, True, False])
        loss = model(*batch, visual_drop_mask=mask); loss.backward()
        self.assertEqual(torch.count_nonzero(model.generator.visual[mask]).item(), 0)
        torch.testing.assert_close(model.generator.visual[~mask], visual[~mask])
        torch.testing.assert_close(model.generator.tokens, expected_tokens)
        torch.testing.assert_close(batch[1][0], visual)
        self.assertIsNotNone(model.touch_encoder.output_projection.weight.grad)
        self.assertIsNone(model.touch_encoder.encoder.weight.grad)
        # Validation-style call stays visual-present even though module.training is true.
        model(*batch)
        torch.testing.assert_close(model.generator.visual, visual)

    def test_default_forward_and_gradients_match_previous_path(self):
        model, batch = make_model(), prepared()
        baseline = copy.deepcopy(model)
        current = model(*batch); current.backward()
        tokens = baseline.touch_encoder(batch[3], batch[4])
        old, _ = baseline.generator.loss(batch[0], *batch[1], touch_tokens=tokens)
        old.backward()
        torch.testing.assert_close(current, old, rtol=0, atol=0)
        for (name, p), (old_name, q) in zip(model.named_parameters(), baseline.named_parameters()):
            self.assertEqual(name, old_name)
            if p.grad is None:
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_constant_ignores_points_but_projector_trains(self):
        model, batch = make_model(True, .5), prepared()
        bank = model.constant_touch_features.clone()
        a = model.get_touch_tokens(batch[3], batch[4])
        b = model.get_touch_tokens(batch[3] * 123, batch[4])
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        opt = optimizer(model); model(*batch).backward(); opt.step()
        self.assertIsNone(model.touch_encoder.encoder.weight.grad)
        torch.testing.assert_close(bank, model.constant_touch_features, rtol=0, atol=0)
        self.assertFalse(torch.equal(a, model.get_touch_tokens(batch[3], batch[4])))

    def test_checkpoint_resume_and_old_defaults(self):
        for constant in (False, True):
            model, batch = make_model(constant, .5), prepared()
            opt = optimizer(model); model(*batch).backward(); opt.step()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'checkpoint.pt'
                train.save_checkpoint(path, model, opt, 2, 10, .2, 'image_touch', 'full')
                restored = make_model(constant, .5); other_opt = optimizer(restored)
                self.assertEqual(train.load_checkpoint(path, restored, other_opt, 'image_touch', 'full'), (2, 10, .2))
                torch.testing.assert_close(model(*batch), restored(*batch), rtol=0, atol=0)
                self.assertEqual(len(opt.state), len(other_opt.state))
                with self.assertRaisesRegex(ValueError, 'dropout or constant'):
                    incompatible = make_model(constant, 0.)
                    train.load_checkpoint(path, incompatible, optimizer(incompatible), 'image_touch', 'full')
                if constant:
                    self.assertEqual(restored.constant_touch_sample_id, 'reference_000')
                    torch.testing.assert_close(model.constant_touch_features, restored.constant_touch_features)
                else:
                    # Emulate an old checkpoint predating both optional fields.
                    saved = torch.load(path, weights_only=False)
                    saved.pop('training_config'); saved.pop('constant_touch'); torch.save(saved, path)
                    old_model = make_model()
                    train.load_checkpoint(path, old_model, optimizer(old_model), 'image_touch', 'full')
                    torch.testing.assert_close(model(*batch), old_model(*batch), rtol=0, atol=0)
                    saved.pop('cross_attention_scope'); torch.save(saved, path)
                    train.load_checkpoint(path, old_model, optimizer(old_model), 'image_touch', 'kv')
                    with self.assertRaisesRegex(ValueError, 'scope'):
                        train.load_checkpoint(path, old_model, optimizer(old_model), 'image_touch', 'full')

    def test_initializer_chooses_fixed_training_record_without_rng_drift(self):
        model = train.TouchTrainingModel(SmallGenerator(), SmallTouch(), oracle_point_frame=True, constant_touch=True)
        class Dataset:
            records = [{'sample_id': 'z_000'}, {'sample_id': 'a_000'}]
            def __len__(self): return len(self.records)
            def __getitem__(self, index):
                self.selected = index
                random.random(); np.random.rand(); torch.rand(1)
                return index
        ds = Dataset(); batch = prepared()
        py, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
        # Single reference encoding, not all four fake examples.
        ref = (batch[0], batch[1], {}, batch[3][:1], batch[4][:1])
        with patch.object(train, 'collate_touch_batch', side_effect=lambda x: x), patch.object(train, 'prepare_batch', return_value=ref):
            train.initialize_constant_touch(model, None, ds, torch.device('cpu'), 'fp32')
        self.assertEqual(ds.selected, 1); self.assertEqual(model.constant_touch_sample_id, 'a_000')
        self.assertEqual(random.getstate(), py)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)

    def test_evaluation_restores_constant_bank(self):
        import evaluate
        model, batch = make_model(True, .5), prepared()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            train.save_checkpoint(path, model, optimizer(model), 1, 1, .2, 'image_touch', 'full')
            checkpoint = torch.load(path, weights_only=False)
        generator = SmallGenerator()
        pipeline = SimpleNamespace(ss_generator=generator, backbone=generator.backbone)
        stub = ModuleType('sam3d_objects.model.backbone.dit.embedder.touch'); stub.TouchEncoder = SmallTouch
        def enable(encoder, backbone, args):
            backbone.requires_grad_(True)
        with patch.dict(sys.modules, {'sam3d_objects.model.backbone.dit.embedder.touch': stub}), patch.object(evaluate, 'build_optimizer', side_effect=enable):
            restored = evaluate.restore_run(pipeline, checkpoint, torch.device('cpu'))
        torch.testing.assert_close(model(*batch), restored(*batch), rtol=0, atol=0)
        torch.testing.assert_close(restored.get_touch_tokens(batch[3], batch[4]),
                                   restored.get_touch_tokens(batch[3] * 99, batch[4]), rtol=0, atol=0)

    def test_validate_never_drops_visuals(self):
        model, batch = make_model(True, 1.), prepared()
        args = SimpleNamespace(precision='fp32', joint_pointmap=False, oracle_point_frame=True)
        with patch.object(train, 'prepare_batch', return_value=batch):
            loss = train.validate(None, model, [{'target_shape': batch[0]['shape']}],
                                  torch.device('cpu'), args, 29, False, 0)
        self.assertTrue(np.isfinite(loss))
        torch.testing.assert_close(model.generator.visual, batch[1][0], rtol=0, atol=0)

    def test_two_rank_ddp_normal_and_constant(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(ddp_worker, args=(str(Path(tmp) / 'rendezvous'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
