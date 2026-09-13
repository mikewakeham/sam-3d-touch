"""CPU checks for the treatment boundary, balanced exposure and GPU allocation."""
import unittest
from shared_orientation_scope_protocol import additional_shape_parameter,continuation_schedule,next_noise_seed
from run_shared_orientation_scope_pair import allocation_devices


class ScopeTests(unittest.TestCase):
    def test_shape_path_included(self):
        for n in ('t_embedder.mlp.0.weight','d_embedder.mlp.2.bias',
                  'latent_mapping.shape.input_layer.weight','latent_mapping.shape.out_layer.bias',
                  'blocks.23.adaLN_modulation.1.weight',
                  'blocks.0.self_attn.to_qkv.shape.weight','blocks.0.self_attn.to_out.shape.bias',
                  'blocks.9.self_attn.q_rms_norm.shape.gamma','blocks.9.self_attn.k_rms_norm.shape.gamma',
                  'blocks.9.mlp.shape.mlp.0.weight','blocks.9.mlp.shape.mlp.2.bias'):
            self.assertTrue(additional_shape_parameter(n),n)

    def test_layout_and_existing_groups_excluded(self):
        for n in ('latent_mapping.translation.input_layer.weight','latent_mapping.shape.pos_emb',
                  'condition_embedder.backbone.mlp.weight','blocks.0.cross_attn.shape.to_q.weight',
                  'blocks.0.norm2.shape.weight','blocks.0.self_attn.to_qkv.6drotation_normalized.weight',
                  'blocks.0.mlp.6drotation_normalized.mlp.0.weight','touch_encoder.output_projection.0.weight',
                  'encoder.latents.weight'):
            self.assertFalse(additional_shape_parameter(n),n)

    def test_exposure_and_noise(self):
        flags=continuation_schedule();self.assertEqual(len(flags),1000)
        for group in range(4):self.assertEqual(sum(flags[group::4]),125)
        self.assertEqual(next_noise_seed(1),1030);self.assertEqual(next_noise_seed(1000),2029)
        self.assertEqual(len({next_noise_seed(i) for i in range(1,1001)}),1000)
        for i in (0,1001):
            with self.assertRaises(ValueError):next_noise_seed(i)

    def test_allocation_mask_preserved(self):
        self.assertEqual(allocation_devices('3,7',2),['3','7'])
        self.assertEqual(allocation_devices('GPU-a,GPU-b',2),['GPU-a','GPU-b'])
        self.assertEqual(allocation_devices('MIG-a',1),['MIG-a'])
        self.assertEqual(allocation_devices(None,4),['0','1'])
        with self.assertRaises(ValueError):allocation_devices('',0)


if __name__=='__main__':unittest.main()
