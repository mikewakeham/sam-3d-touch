"""Core surface-encoder gradients, checkpoint loading and normalization."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from sam3d_objects.model.backbone.dit.embedder import craftsman, triposg, touch
from sam3d_objects.model.backbone.dit.embedder.surface_encoder_utils import load_surface_encoder_weights
import train


def small_encoder(name):
    cls = craftsman.CraftsManEncoder if name == 'craftsman' else triposg.TripoSGPointEncoder
    return cls(width=32, heads=4, layers=2, num_latents=8, latent_dim=4)


def surfaces(batch=2, size=32):
    points = torch.randn(batch, size, 3)
    normals = F.normalize(torch.randn_like(points), dim=-1)
    return torch.cat((points, normals), dim=-1)


class PointEncoderTests(unittest.TestCase):
    def test_shapes_gradients_and_repeatable_eval(self):
        for name in ('craftsman', 'triposg'):
            model = small_encoder(name)
            x = surfaces()
            tokens = model.encode(x)
            self.assertEqual(tokens.shape, (2, 8, 4))
            tokens.square().mean().backward()
            for key, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, key)
                self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, key)
            model.eval()
            torch.testing.assert_close(model.encode(x), model.encode(x), rtol=0, atol=0)

    def test_scratch_adapter_restore_and_frozen_pretrained(self):
        for name in ('craftsman', 'triposg'):
            with patch.dict(touch.ENCODERS, {name: {'constructor': lambda: small_encoder(name)}}):
                with patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('scratch download')):
                    model = touch.TouchEncoder(name, output_dim=16, trainable=True, pretrained=False, use_position=False)
                x = surfaces()
                optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=.01)
                before = next(model.encoder.parameters()).detach().clone()
                model(x).square().mean().backward()
                optimizer.step()
                self.assertFalse(torch.equal(before, next(model.encoder.parameters())))
                restored = touch.TouchEncoder(**model.get_config())
                train.load_trainable_state_dict(restored, train.trainable_state_dict(model))
                model.eval(); restored.eval()
                torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / ('model.ckpt' if name == 'craftsman' else 'model.safetensors')
                    if name == 'craftsman':
                        torch.save({'state_dict': {'shape_model.' + key: value for key, value in model.encoder.state_dict().items()}}, path)
                    else:
                        from safetensors.torch import save_file
                        save_file(model.encoder.state_dict(), path)
                    frozen = touch.TouchEncoder(name, output_dim=16, encoder_checkpoint=path, use_position=False)
                    frozen.train()
                    self.assertFalse(frozen.encoder.training)
                    self.assertTrue(all(not p.requires_grad for p in frozen.encoder.parameters()))
                    frozen(x).square().mean().backward()
                    self.assertTrue(all(p.grad is None for p in frozen.encoder.parameters()))
                    self.assertIsNotNone(frozen.output_projection[1].w1.weight.grad)
                    frozen.set_trainable(True)
                    self.assertTrue(frozen.encoder.training)
                    frozen(x).square().mean().backward()
                    self.assertTrue(all(p.grad is not None for p in frozen.encoder.parameters()))

    def test_normalization_preserves_normals_and_rejects_bad_inputs(self):
        with patch.dict(touch.ENCODERS, {'craftsman': {'constructor': lambda: small_encoder('craftsman')}}):
            model = touch.TouchEncoder('craftsman', pretrained=False, trainable=True)
            x = surfaces()
            prepared, shift, scale = model.prepare_surface(x)
            torch.testing.assert_close(prepared[..., 3:], x[..., 3:], rtol=0, atol=0)
            torch.testing.assert_close(prepared[..., :3] / scale[:, None] + shift[:, None], x[..., :3])
            torch.testing.assert_close(prepared[..., :3].abs().amax(dim=(1,2)), torch.ones(2))
            for broken in (x[..., :3], x * float('nan'), torch.zeros_like(x)):
                with self.assertRaises(ValueError): model(broken)
            with self.assertRaises(ValueError): model(x, torch.zeros(2,32,dtype=torch.bool))
            with self.assertRaises(ValueError): model(x[:, :4])

    def test_strict_pretrained_keys(self):
        for name in ('craftsman', 'triposg'):
            model = small_encoder(name)
            state = model.state_dict()
            state.pop(next(iter(state)))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'weights'
                if name == 'craftsman': torch.save({'state_dict': state}, path)
                else:
                    from safetensors.torch import save_file
                    save_file(state, path)
                with self.assertRaises(RuntimeError): load_surface_encoder_weights(model, name, path)


if __name__ == '__main__':
    unittest.main()
