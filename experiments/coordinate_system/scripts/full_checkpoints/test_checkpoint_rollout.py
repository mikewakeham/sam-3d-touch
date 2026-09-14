"""CPU checks: noise-only generation, support metrics, and complete paired summaries."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import copy
import unittest

import numpy as np
import torch
from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_rollout_core import sample_from_noise, support_metrics
from experiments.coordinate_system.scripts.full_checkpoints.analyze_checkpoint_rollouts import analyze


class Generator:
    def _generate_noise(self, shapes, device):
        raise AssertionError('Uncontrolled sampling noise')

    def __call__(self, shapes, device, visual, touch_tokens):
        assert all(isinstance(v, tuple) and all(isinstance(i, int) for i in v) for v in shapes.values())
        self.received_shapes = shapes
        noise = self._generate_noise(shapes, device)
        noise['shape'].add_(visual + touch_tokens)
        return noise


def reports():
    output = {}
    for arm in ('oracle', 'constant'):
        inputs = [dict(split=s, group=0, sample_ids=[s+str(i)+'_000' for i in range(4)],
                       object_ids=[s+str(i) for i in range(4)], tokens_sha256=arm) for s in ('train', 'val')]
        samples = []
        for item in inputs:
            for sid, oid in zip(item['sample_ids'], item['object_ids']):
                for visual in ('present', 'zero'):
                    for shift in ((0, 1) if arm == 'oracle' else (0,)):
                        for draw in range(2):
                            value = .99 if arm == 'oracle' and shift == 0 else .2
                            samples.append(dict(split=item['split'], group=0, sample_id=sid, object_id=oid,
                                                visual=visual, surface_shift=shift, draw=draw,
                                                **{k: value for k in ('iou', 'precision_2v', 'recall_2v', 'fscore_2v')}))
        output[arm] = dict(complete=True, adapted_parameters_unchanged=True, settings=dict(arm=arm),
                           checkpoint_metadata=dict(step=14660), inputs=inputs, samples=samples,
                           **{k: {} for k in ('source_sha256', 'additional_source_sha256', 'dataset_sha256',
                                             'pipeline_sha256', 'decoder_sha256', 'sampler', 'banks')})
    return output


class RolloutTests(unittest.TestCase):
    def test_sampling_receives_noise_shapes_not_target_values(self):
        gen = Generator(); noise = {'shape': torch.ones(4, 2), 'layout': torch.zeros(4, 1)}
        visual, tokens = torch.ones(4, 2), torch.full((4, 2), 2.)
        with torch.no_grad():
            result = sample_from_noise(gen, noise, visual, tokens)
            second = sample_from_noise(gen, noise, visual * 0, tokens)
        torch.testing.assert_close(result, torch.full((4, 2), 4.))
        torch.testing.assert_close(second, torch.full((4, 2), 3.))
        torch.testing.assert_close(noise['shape'], torch.ones(4, 2))
        self.assertEqual(gen.received_shapes, {'shape': (4, 2), 'layout': (4, 1)})

    def test_reference_empty_and_distance_metrics(self):
        target = np.zeros((64, 64, 64), dtype=bool); target[25:35, 25:35, 25:35] = True
        self.assertEqual(support_metrics(target, target)['iou'], 1.)
        self.assertEqual(support_metrics(target, target)['fscore_2v'], 1.)
        empty = np.zeros_like(target)
        self.assertEqual(support_metrics(empty, target)['fscore_2v'], 0.)
        self.assertEqual(support_metrics(empty, target)['iou'], 0.)
        with self.assertRaises(ValueError):
            support_metrics(target, empty)
        shifted = np.roll(target, 2, axis=0)
        self.assertEqual(support_metrics(shifted, target)['fscore_2v'], 1.)
        self.assertLess(support_metrics(shifted, target)['iou'], 1.)

    def test_summary_endpoint_and_control_direction(self):
        result = analyze(reports())
        cell = result['summary']['val']['present']
        self.assertTrue(cell['oracle']['raw_endpoint_met'])
        self.assertFalse(cell['constant']['raw_endpoint_met'])
        self.assertAlmostEqual(cell['oracle_minus_constant']['mean_fscore_difference'], .79)

    def test_missing_duplicate_unpaired_samples_rejected(self):
        for kind in ('missing', 'duplicate', 'noise', 'observation'):
            data = reports()
            if kind == 'missing': data['oracle']['samples'].pop()
            elif kind == 'duplicate': data['oracle']['samples'].append(data['oracle']['samples'][0])
            elif kind == 'noise': data['constant']['banks'] = {'wrong': True}
            else: data['constant']['inputs'][0]['group'] = 1
            with self.assertRaises(AssertionError): analyze(data)


if __name__ == '__main__':
    unittest.main()
