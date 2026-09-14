"""CPU tests of intervention pairing and report completeness, not a GPU smoke test."""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import copy
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_probe_core import paired_loss, condition_tokens, tensor_sha
from experiments.coordinate_system.scripts.full_checkpoints.analyze_checkpoint_probe import analyze


class Generator:
    loss_weights = {'shape': 1.0}
    loss_fn = {'shape': F.mse_loss}

    def _generate_t(self, targets):
        raise AssertionError('Uncontrolled time draw')

    def _generate_x0(self, targets):
        raise AssertionError('Uncontrolled noise draw')

    def _generate_d(self, targets):
        raise AssertionError('Uncontrolled shortcut draw')

    def loss(self, targets, visual, touch_tokens=None):
        times = self._generate_t(targets)
        noise = self._generate_x0(targets)
        assert torch.count_nonzero(self._generate_d(targets)) == 0
        self.last_times = times.clone(); self.last_noise = noise['shape'].clone()
        target = targets['shape'] - noise['shape']
        # Synthetic objective deliberately depends on every experimental input.
        pred = visual + touch_tokens + times[:, None] + noise['shape']
        scalar = self.loss_fn['shape'](pred, target)
        noise['shape'].zero_()  # Must not mutate the shared bank.
        return scalar, {}


def reports():
    bank = [dict(split=s, group=0, time_kind=t, draw=d) for s in ('train', 'val')
            for t in ('native', 't_0.05', 't_0.5', 't_0.95') for d in range(2)]
    result = {}
    for arm in ('oracle', 'constant'):
        inputs = [dict(split=s, group=0, sample_ids=[s+str(i)+'_000' for i in range(4)],
                       object_ids=[s+str(i) for i in range(4)], **{k: 'same' for k in
                       ('image_sha256', 'pointmap_sha256', 'visual_sha256', 'points_sha256', 'mask_sha256', 'target_sha256')})
                  for s in ('train', 'val')]
        rows = [dict(**b, visual=v, surface_shift=shift,
                     losses=[(1 if arm == 'oracle' else 2) + shift] * 4)
                for b in bank for v in ('present', 'zero')
                for shift in (range(4) if arm == 'oracle' else range(1))]
        result[arm] = dict(complete=True, adapted_parameters_unchanged=True,
                           settings=dict(arm=arm, objects_per_split=4), checkpoint_metadata=dict(step=14660),
                           source_sha256={}, pipeline_sha256='same', dataset_sha256={}, data_config={},
                           banks=copy.deepcopy(bank), inputs=inputs, rows=rows)
    return result


class ProbeTests(unittest.TestCase):
    def test_same_actual_bank_and_native_scalar(self):
        gen = Generator()
        targets = {'shape': torch.arange(8, dtype=torch.float32).reshape(4, 2)}
        times = torch.tensor([.05, .2, .5, .95])
        noise = {'shape': torch.ones(4, 2)}
        visual = torch.ones(4, 2); tokens = torch.arange(4.)[:, None].expand(4, 2)
        old_fn = gen.loss_fn
        expected = ((visual + tokens + times[:, None] + noise['shape']) - (targets['shape'] - noise['shape'])).square().mean(1)
        with torch.no_grad():
            values = paired_loss(gen, targets, visual, tokens, times, noise)
        torch.testing.assert_close(torch.tensor(values), expected)
        self.assertIs(gen.loss_fn, old_fn)
        torch.testing.assert_close(noise['shape'], torch.ones(4, 2))
        for shift in (1, 2, 3):
            with torch.no_grad():
                paired_loss(gen, targets, visual * 0, condition_tokens(tokens, shift), times, noise)
            torch.testing.assert_close(gen.last_noise, noise['shape'], rtol=0, atol=0)
            torch.testing.assert_close(gen.last_times, times, rtol=0, atol=0)
        with self.assertRaisesRegex(AssertionError, 'Uncontrolled time'):
            gen._generate_t(targets)

    def test_distractors_and_constant_invariance(self):
        tokens = torch.arange(4.)[:, None]
        for shift in (1, 2, 3):
            self.assertTrue((condition_tokens(tokens, shift) != tokens).all())
            torch.testing.assert_close(condition_tokens(torch.ones(4, 2), shift), torch.ones(4, 2))
        with self.assertRaises(ValueError):
            condition_tokens(tokens, 4)
        self.assertNotEqual(tensor_sha(tokens), tensor_sha(tokens.to(torch.bfloat16)))

    def test_analyzer_has_correct_direction_and_rejects_missing_cells(self):
        data = reports(); result = analyze(data)
        cell = result['summary']['val']['zero']['native']
        self.assertEqual(cell['mean_constant_minus_oracle'], 1)
        self.assertEqual(cell['mean_wrong_minus_correct'], 2)
        self.assertEqual(cell['objects_oracle_better_than_constant'], 4)
        data['oracle']['rows'].pop()
        with self.assertRaisesRegex(AssertionError, 'Missing'):
            analyze(data)

    def test_analyzer_rejects_unpaired_inputs_noise_and_duplicates(self):
        for mutation in ('inputs', 'banks', 'duplicate'):
            data = reports()
            if mutation == 'inputs':
                data['constant']['inputs'][0]['visual_sha256'] = 'different'
            elif mutation == 'banks':
                data['constant']['banks'][0]['noise_sha256'] = 'different'
            else:
                data['oracle']['rows'].append(data['oracle']['rows'][0])
            with self.assertRaises(AssertionError):
                analyze(data)


if __name__ == '__main__':
    unittest.main()
