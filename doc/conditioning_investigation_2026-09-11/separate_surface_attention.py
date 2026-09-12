"""Experimental shape-attention wrapper; production modules stay unchanged."""
import copy
import torch
from torch import nn


class SeparateSurfaceAttention(nn.Module):
    def __init__(self, visual_attention, visual_tokens=7528, surface_tokens=1024):
        super().__init__()
        self.visual_tokens = visual_tokens
        self.surface_tokens = surface_tokens
        self.visual = visual_attention.requires_grad_(False)
        self.surface = copy.deepcopy(visual_attention).requires_grad_(True)
        # A zero residual preserves the image function at initialization.
        nn.init.zeros_(self.surface.to_out.weight)
        nn.init.zeros_(self.surface.to_out.bias)

    def forward(self, x, context):
        if context.shape[1] not in (self.visual_tokens, self.visual_tokens+self.surface_tokens):
            raise ValueError(f'Unexpected conditioning length: {context.shape[1]}')
        visual = context[:, :self.visual_tokens]
        result = self.visual(x, visual)
        if context.shape[1] == self.visual_tokens:
            return result
        # Separate softmax normalization and independent trainable projections.
        # Frozen visual computation remains differentiable w.r.t. hidden states.
        return result + self.surface(x, context[:, self.visual_tokens:])


def verify_module(visual_attention, device):
    """Run actual CUDA attention/gradient checks on a disposable module copy."""
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(910037)
        probe = SeparateSurfaceAttention(copy.deepcopy(visual_attention), 5, 7)
        x = torch.randn(2, 8, visual_attention.channels, device=device, requires_grad=True)
        visual = torch.randn(2, 5, visual_attention.ctx_channels, device=device)
        surface = torch.randn(2, 7, visual_attention.ctx_channels, device=device)
        joined = torch.cat((visual, surface), dim=1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            reference = probe.visual(x, visual)
            actual = probe(x, joined)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(probe(x, visual), reference, rtol=0, atol=0)
            actual.float().square().mean().backward()
        assert probe.surface.to_out.weight.grad is not None
        assert torch.count_nonzero(probe.surface.to_out.weight.grad).item() > 0
        assert all(p.grad is None for p in probe.visual.parameters())
        assert x.grad is not None and torch.isfinite(x.grad).all()
        with torch.no_grad():
            probe.surface.to_out.load_state_dict(probe.visual.to_out.state_dict())
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            expected = probe.visual(x, visual) + probe.surface(x, surface)
            torch.testing.assert_close(probe(x, joined), expected, rtol=0, atol=0)
            torch.testing.assert_close(probe(x, visual), probe.visual(x, visual), rtol=0, atol=0)
        try:
            probe(x, joined[:, :6])
        except ValueError:
            pass
        else:
            raise AssertionError('Unexpected context length accepted')
    return dict(zero_residual_exact=True, absent_surface_exact=True,
                independent_sum_exact=True, output_gradient_nonzero=True,
                visual_parameters_no_grad=True, hidden_state_gradient_finite=True,
                wrong_length_rejected=True)
