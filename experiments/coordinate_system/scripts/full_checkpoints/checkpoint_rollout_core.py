"""Noise-only generation and Stage-1 support metrics; no training targets fed to sampling."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
import torch
from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_probe_core import clone_tree


def sample_from_noise(generator, noise, visual, tokens):
    shapes = {name: tuple(value.shape) for name, value in noise.items()}
    device = noise['shape'].device
    with patch.object(generator, '_generate_noise', side_effect=lambda *a, **k: clone_tree(noise)) as draw:
        prediction = generator(shapes, device, visual, touch_tokens=tokens)
    assert draw.call_count == 1, 'Expected exactly one initial-noise draw'
    assert prediction['shape'].shape == noise['shape'].shape
    assert torch.isfinite(prediction['shape']).all(), 'Nonfinite generated shape'
    return prediction['shape']


def support_metrics(pred, target):
    assert pred.shape == target.shape == (64, 64, 64)
    assert pred.dtype == target.dtype == np.bool_
    p, q = np.argwhere(pred), np.argwhere(target)
    if not len(q):
        raise ValueError('Empty decoded target cannot be a fidelity reference')
    intersection = np.count_nonzero(pred & target)
    union = np.count_nonzero(pred | target)
    precision = float(np.mean(cKDTree(q).query(p)[0] <= 2.00000001)) if len(p) else 0.
    recall = float(np.mean(cKDTree(p).query(q)[0] <= 2.00000001)) if len(p) else 0.
    return {'iou': float(intersection / union), 'precision_2v': precision, 'recall_2v': recall,
            'fscore_2v': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'predicted_count': len(p), 'target_count': len(q)}
