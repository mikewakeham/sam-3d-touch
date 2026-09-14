"""Experiment-only conditioning and reproducible sampling. No production patches."""
import copy
from contextlib import contextmanager
import random

import numpy as np
import torch

ARMS = {
    'object_stock': ('object', False),
    'camera_stock': ('camera', False),
    'camera_shared': ('camera', True),
}


class SurfaceNormalizer:
    """Replace BOTH pointmap normalizers with one observed-surface affine."""
    def __init__(self, surface):
        if surface.ndim != 2 or surface.shape[-1] != 3 or not len(surface):
            raise ValueError('Expected a nonempty XYZ surface')
        if not torch.isfinite(surface).all():
            raise ValueError('Nonfinite surface')
        self.center = (surface.max(0).values + surface.min(0).values) / 2
        self.radius = torch.linalg.vector_norm(surface-self.center, dim=-1).max()
        if not torch.isfinite(self.radius) or self.radius <= 0:
            raise ValueError('Degenerate surface')

    def normalize(self, pointmap, mask, scale=None, shift=None):
        # Ignore any stock/full-image moments: both branches use this same affine.
        center = self.center.to(pointmap)
        radius = self.radius.to(pointmap)
        return ((pointmap-center[:, None, None]) / radius,
                radius.expand(3), center)


def preprocess_inputs(pipeline, batch, shared):
    items = []
    for image, pointmap, points, mask in zip(
            batch['image'], batch['pointmap'], batch['touch_xyz'], batch['touch_mask']):
        processor = pipeline.ss_preprocessor
        if shared:
            processor = copy.copy(processor)
            normalizer = SurfaceNormalizer(points[mask])
            processor.normalize_pointmap = True
            processor.pointmap_normalizer = normalizer
            processor.rgb_pointmap_normalizer = normalizer
        items.append(pipeline.preprocess_image(
            image.numpy(), processor, pointmap=pointmap.permute(2, 0, 1)))
    return {key: torch.cat([item[key] for item in items]) for key in items[0]}


def prepare_batch(pipeline, batch, device, precision, arm):
    from train import amp, make_targets
    inputs = preprocess_inputs(pipeline, batch, ARMS[arm][1])
    with torch.no_grad(), amp(device, precision):
        condition_args, condition_kwargs = pipeline.get_condition_input(
            pipeline.ss_condition_embedder, inputs, pipeline.ss_condition_input_mapping)
    # Identical raw camera surface in all arms. VecSetX applies its own normalizer.
    # This avoids a redundant SSI -> radius normalization and prevents C changing
    # surface features as well as pointmap features. No oracle transform here.
    return (make_targets(batch['target_shape'].to(device), pipeline.backbone),
            condition_args, condition_kwargs, batch['touch_xyz'].to(device),
            batch['touch_mask'].to(device))


@contextmanager
def fixed_rng(seed, device):
    """Evaluation and condition preprocessing cannot advance training RNG streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)


def training_indices(size, count, seed):
    """Repeat shuffled full epochs, carrying partial batches across epoch boundaries."""
    rng = np.random.default_rng(seed)
    written = 0
    while written < count:
        order = rng.permutation(size)
        for index in order[:min(size, count-written)]:
            yield int(index)
        written += min(size, count-written)


def validation_indices(records, train_views=2):
    """Two deterministic training views per object; other splits use all views."""
    by_object = {}
    for index, row in enumerate(records):
        by_object.setdefault(row['object_id'], []).append(index)
    return [i for indices in by_object.values()
            for i in sorted(indices, key=lambda i: records[i]['sample_id'])[:train_views]]


def accumulate_update(batches, loss_fn, optimizer, parameters, total_samples, gradient_clip):
    """Accumulate sample-weighted means; exactly one optimizer update per global batch."""
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    observed = 0
    for batch in batches:
        count = len(batch['target_shape'])
        loss = loss_fn(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        (loss * (count / total_samples)).backward()
        total += float(loss.detach()) * count
        observed += count
    if observed != total_samples:
        raise ValueError(f'Incomplete optimizer batch: {observed} != {total_samples}')
    norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    return total / total_samples, float(norm)
