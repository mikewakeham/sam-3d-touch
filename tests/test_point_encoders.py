"""CPU tests for the encoder ports and the existing training/evaluation interface.

Set POINT_ENCODER_REFERENCES to external CraftsMan3D/ and TripoSG/ checkouts
for upstream numerical parity. Set POINT_ENCODER_WEIGHTS to a directory with
craftsman-model.ckpt and triposg-vae.safetensors to also check released weights.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ast
import contextlib
import io
import json
import math
import os
import tempfile
import types
from typing import Any, Dict, Optional
import unittest
from unittest.mock import patch

os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat

from sam3d_objects.model.backbone.dit.embedder import craftsman, triposg, touch
from sam3d_objects.model.backbone.dit.embedder.surface_encoder_utils import farthest_point_indices, load_surface_encoder_weights
import train


def small_encoder(name):
    cls = craftsman.CraftsManEncoder if name == 'craftsman' else triposg.TripoSGPointEncoder
    return cls(width=32, heads=4, layers=2, num_latents=8, latent_dim=4)


def surfaces(batch=2, size=32):
    points = torch.randn(batch, size, 3)
    normals = F.normalize(torch.randn_like(points), dim=-1)
    return torch.cat((points, normals), dim=-1)


def reference_definitions(path, namespace, names):
    # Execute the original encoder classes, without importing their GPU rendering frameworks.
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    for node in tree.body:
        node.decorator_list = []
    exec(compile(tree, str(path), 'exec'), namespace)


def reference_encoder(name, root, width=32, heads=4, layers=2, num_latents=8, latent_dim=4):
    ns = dict(torch=torch, nn=nn, F=F, math=math, repeat=repeat, Optional=Optional, Any=Any, Dict=Dict)
    if name == 'craftsman':
        base = root / 'CraftsMan3D/craftsman'
        ns['checkpoint'] = lambda function, inputs, parameters, enabled: function(*inputs)
        reference_definitions(base / 'models/transformers/utils.py', ns, {'init_linear', 'MLP'})
        reference_definitions(base / 'models/transformers/attention.py', ns, {'MultiheadAttention', 'QKVMultiheadAttention', 'ResidualAttentionBlock', 'MultiheadCrossAttention', 'QKVMultiheadCrossAttention', 'ResidualCrossAttentionBlock'})
        reference_definitions(base / 'models/transformers/perceiver_1d.py', ns, {'Perceiver'})
        reference_definitions(base / 'models/autoencoders/michelangelo_autoencoder.py', ns, {'FourierEmbedder', 'PerceiverCrossAttentionEncoder'})
        model = nn.Module()
        model.encoder = ns['PerceiverCrossAttentionEncoder'](
            use_downsample=True, num_latents=num_latents, embedder=ns['FourierEmbedder'](num_freqs=8, include_pi=False),
            point_feats=3, embed_point_feats=False, width=width, heads=heads, layers=layers,
            init_scale=.25 / math.sqrt(width), qkv_bias=False, use_ln_post=True, use_flash=True,
        )
        model.pre_kl = nn.Linear(width, latent_dim * 2)
        model.encode = lambda x: model.pre_kl(model.encoder(x[..., :3], x[..., 3:])).chunk(2, dim=-1)[0]
    else:
        base = root / 'TripoSG/triposg/models'
        ns.update(Attention=triposg.Attention, FeedForward=triposg.FeedForward,
                  FP32LayerNorm=triposg.FP32LayerNorm, LayerNorm=triposg.LayerNorm,
                  apply_rotary_emb=triposg.apply_rotary_emb)
        reference_definitions(base / 'embeddings.py', ns, {'FrequencyPositionalEmbedding'})
        reference_definitions(base / 'attention_processor.py', ns, {'TripoSGAttnProcessor2_0'})
        reference_definitions(base / 'transformers/triposg_transformer.py', ns, {'DiTBlock'})
        reference_definitions(base / 'autoencoders/autoencoder_kl_triposg.py', ns, {'TripoSGEncoder'})
        model = nn.Module()
        model.embedder = ns['FrequencyPositionalEmbedding'](num_freqs=8, include_pi=False)
        model.encoder = ns['TripoSGEncoder'](in_channels=54, dim=width, num_attention_heads=heads, num_layers=layers)
        model.quant = nn.Linear(width, latent_dim * 2)
        # Execute upstream _encode and _sample_features verbatim as bound methods.
        tree = ast.parse((base / 'autoencoders/autoencoder_kl_triposg.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TripoSGVAEModel')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_encode', '_sample_features')]
        ns.update(np=np, fps=reference_fps)
        exec(compile(ast.Module(body=methods, type_ignores=[]), '<upstream TripoSG methods>', 'exec'), ns)
        for key in ('_encode', '_sample_features'):
            setattr(model, key, types.MethodType(ns[key], model))
        model.config = types.SimpleNamespace(in_channels=3)
        model.encode = lambda x: model._encode(x, num_tokens=num_latents, seed=0).chunk(2, dim=-1)[0]
    return model


def reference_fps(points, batch, ratio, random_start=True):
    # Same query indices for both networks: isolate architecture/weights from RNG backend.
    batch_size = int(batch.max()) + 1
    points = points.reshape(batch_size, -1, 3)
    indices = farthest_point_indices(points, round(points.shape[1] * ratio), random_start=False)
    return (indices + torch.arange(batch_size)[:, None] * points.shape[1]).flatten()


class PointEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

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

    def test_training_normal_plumbing_and_oracle_rotation(self):
        x = surfaces()
        transform = torch.eye(4).repeat(2,1,1)
        transform[:, :3, :3] = torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]])
        transform[:, :3, 3] = torch.tensor([1.,2,3])
        batch = dict(image=None, pointmap=None, target_shape=torch.zeros(2,1), touch_xyz=x[..., :3],
                     touch_normals=x[..., 3:], touch_mask=torch.ones(2,32,dtype=torch.bool), object_from_camera=transform)
        pipeline = types.SimpleNamespace(ss_condition_embedder=None, ss_condition_input_mapping=None,
                    get_condition_input=lambda *a: ((), {}), backbone=types.SimpleNamespace(latent_mapping={}))
        with patch('train.preprocess_batch', return_value={}):
            prepared = train.prepare_batch(pipeline,batch,torch.device('cpu'),'fp32',True,use_normals=True)
            torch.testing.assert_close(prepared[3], x)
            prepared = train.prepare_batch(pipeline,batch,torch.device('cpu'),'fp32',True,oracle_point_frame=True,use_normals=True)
            expected_xyz = x[..., :3] @ transform[:, :3, :3].transpose(1,2) + transform[:, None, :3,3]
            expected_n = x[..., 3:] @ transform[:, :3, :3].transpose(1,2)
            torch.testing.assert_close(prepared[3][..., :3], expected_xyz)
            torch.testing.assert_close(prepared[3][..., 3:], expected_n)

    def test_cli_and_dataset_options(self):
        for name in ('craftsman','triposg'):
            argv=['train.py','--pipeline-config','unused','--point-encoder',name,'--point-encoder-from-scratch','--train-point-encoder']
            with patch.object(sys,'argv',argv): args=train.parse_args()
            self.assertTrue(args.vecsetx_from_scratch and args.train_vecsetx and args.no_touch_position)
            config=train.configure_encoder_data({'touch': {'source':'full_surface'}},name)
            self.assertEqual(config['touch']['pool_points'],16384 if name == 'craftsman' else 20480)
            self.assertTrue(config['touch']['include_normals'])
            for flag in ('--joint-pointmap','--vecsetx-learn','--constant-touch','--no-touch'):
                with patch.object(sys,'argv',argv+[flag]), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit): train.parse_args()

    def test_generated_pool_through_loader_training_and_checkpoint(self):
        # Reuse the existing synthetic mesh/camera/normal fixture, without Blender or GPU.
        general = Path(__file__).resolve().parent.parent / 'data_generation/general/tests'
        with patch.object(sys, 'path', [str(general), *sys.path]):
            from test_normals import NormalsTests
            fixture = NormalsTests('test_default_generation_and_triangle_alignment')
            fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for record in fixture.records:
            record['target_path'] = 'target.npz'
        np.savez(fixture.root / 'target.npz', mean=np.zeros((8,16,16,16), dtype=np.float32))
        (fixture.root / 'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in fixture.records))
        (fixture.root / 'splits.json').write_text(json.dumps({'train':['box'], 'val':[], 'test':[]}))
        from dataloader import TouchDataset, collate_touch_batch
        config = {'seed':29, 'dataset':{'root':str(fixture.root), 'manifest':'samples.jsonl',
                  'split_file':'splits.json', 'split':'train'},
                  'touch':{'source':'full_surface', 'include_normals':True, 'pool_points':1024}}
        dataset = TouchDataset(config)
        batch = collate_touch_batch([dataset[0],dataset[1]])
        class Generator(nn.Module):
            def loss(self, targets, *args, touch_tokens, **kwargs):
                return touch_tokens.square().mean(), {}
        pipeline = types.SimpleNamespace(ss_condition_embedder=None, ss_condition_input_mapping=None,
                    get_condition_input=lambda *a: ((), {}), backbone=types.SimpleNamespace(latent_mapping={}))
        for name in ('craftsman','triposg'):
            with patch.dict(touch.ENCODERS, {name:{'constructor':lambda:small_encoder(name)}}):
                encoder=touch.TouchEncoder(name, output_dim=16, trainable=True, pretrained=False, use_position=False)
                model=train.TouchTrainingModel(Generator(),encoder)
                optimizer=torch.optim.AdamW(encoder.get_trainable_parameters(),lr=.01)
                with patch('train.preprocess_batch',return_value={}):
                    prepared=train.prepare_batch(pipeline,batch,torch.device('cpu'),'fp32',True,use_normals=True)
                model(*prepared).backward(); optimizer.step()
                path=fixture.root / (name+'.pt')
                train.save_checkpoint(path,model,optimizer,1,2,.3,'image_touch')
                restored=train.TouchTrainingModel(Generator(),touch.TouchEncoder(**encoder.get_config()))
                new_optimizer=torch.optim.AdamW(restored.touch_encoder.get_trainable_parameters(),lr=.01)
                self.assertEqual(train.load_checkpoint(path,restored,new_optimizer,'image_touch'),(1,2,.3))
                model.eval();restored.eval()
                torch.testing.assert_close(model.get_touch_tokens(*prepared[3:]),restored.get_touch_tokens(*prepared[3:]),rtol=0,atol=0)

    @unittest.skipUnless(os.environ.get('POINT_ENCODER_REFERENCES'), 'external upstream checkouts not configured')
    def test_upstream_small_forward_and_gradient_parity(self):
        root=Path(os.environ['POINT_ENCODER_REFERENCES'])
        for name in ('craftsman','triposg'):
            model=small_encoder(name).eval()
            reference=reference_encoder(name,root).eval()
            reference.load_state_dict(model.state_dict(), strict=True)
            x=surfaces()
            with patch.dict(sys.modules, {'torch_cluster': types.SimpleNamespace(fps=reference_fps)}):
                actual, expected=model.encode(x),reference.encode(x)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            actual.square().mean().backward();expected.square().mean().backward()
            for (key,p),(ref_key,q) in zip(model.named_parameters(),reference.named_parameters()):
                self.assertEqual(key,ref_key)
                torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)

    @unittest.skipUnless(os.environ.get('POINT_ENCODER_WEIGHTS') and os.environ.get('POINT_ENCODER_REFERENCES'), 'released weights not configured')
    def test_released_weights_and_upstream_parity(self):
        root=Path(os.environ['POINT_ENCODER_REFERENCES']); weights=Path(os.environ['POINT_ENCODER_WEIGHTS'])
        for name,cls,width,heads,filename in (
            ('craftsman',craftsman.CraftsManEncoder,768,12,'craftsman-model.ckpt'),
            ('triposg',triposg.TripoSGPointEncoder,512,8,'triposg-vae.safetensors')):
            # Query count is parameter-free; reduced here to keep this a CPU test.
            model=cls(num_latents=8).eval()
            load_surface_encoder_weights(model,name,weights/filename)
            reference=reference_encoder(name,root,width=width,heads=heads,layers=8,num_latents=8,latent_dim=64).eval()
            reference.load_state_dict(model.state_dict(),strict=True)
            x=surfaces(batch=1)
            with torch.no_grad(),patch.dict(sys.modules,{'torch_cluster':types.SimpleNamespace(fps=reference_fps)}):
                torch.testing.assert_close(model.encode(x),reference.encode(x),rtol=1e-5,atol=1e-6)
            del model,reference


if __name__ == '__main__':
    unittest.main()
