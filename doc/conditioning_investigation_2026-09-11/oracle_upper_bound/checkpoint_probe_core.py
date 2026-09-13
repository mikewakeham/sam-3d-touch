"""Small paired-loss helpers, independently testable without SAM3D imports."""
import hashlib
from unittest.mock import patch

import torch
import torch.nn.functional as F


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def clone_tree(values):
    return {name: value.clone() for name, value in values.items()}


def paired_loss(generator, targets, visual, tokens, times, noise):
    """Run the original scalar objective, capturing per-object shape losses.

    Patch only the noise/time/shortcut draw. The trained forward path and shape
    loss remain the generator's own. All loss weights except shape must be zero.
    """
    assert generator.loss_weights['shape'] == 1
    assert all(v == 0 for k, v in generator.loss_weights.items() if k != 'shape')
    assert targets.keys() == noise.keys()
    batch = targets['shape'].shape[0]
    assert times.shape == (batch,) and torch.isfinite(times).all()
    for key in targets:
        assert targets[key].shape == noise[key].shape
    original = generator.loss_fn
    functions = dict(original) if isinstance(original, dict) else {
        key: original for key in generator.loss_weights}
    shape_fn = functions['shape']
    captured = []

    def capture(pred, target):
        scalar = shape_fn(pred, target)
        per_object = F.mse_loss(pred.float(), target.float(), reduction='none').flatten(1).mean(1)
        torch.testing.assert_close(per_object.mean(), scalar.float(), rtol=1e-5, atol=1e-7)
        captured.append(per_object.detach())
        return scalar

    functions['shape'] = capture
    kwargs = {} if tokens is None else {'touch_tokens': tokens}
    with patch.object(generator, '_generate_t', side_effect=lambda _: times.clone()) as tm, \
         patch.object(generator, '_generate_x0', side_effect=lambda _: clone_tree(noise)) as nm, \
         patch.object(generator, '_generate_d', side_effect=lambda _: torch.zeros_like(times)) as dm, \
         patch.object(generator, 'loss_fn', functions):
        loss, _ = generator.loss(targets, visual, **kwargs)
    assert tm.call_count == nm.call_count == dm.call_count == 1
    assert len(captured) == 1 and captured[0].shape == (batch,)
    assert torch.isfinite(captured[0]).all() and torch.isfinite(loss)
    torch.testing.assert_close(captured[0].mean(), loss.float(), rtol=1e-5, atol=1e-7)
    return captured[0].cpu().tolist()


def condition_tokens(tokens, shift):
    if not 0 <= shift < len(tokens):
        raise ValueError('Cyclic distractor shift must be in [0, batch size)')
    return tokens if shift == 0 else tokens.roll(shift, 0)
