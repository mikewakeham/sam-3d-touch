"""CPU behavior tests; real Torch forward/gradient equivalence is the GPU gate."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from analyze import analyze
from protocol import ARMS

HERE = Path(__file__).resolve().parent


class FakeBase:
    def __init__(self, generator, encoder, no_pointmap, oracle):
        self.generator, self.touch_encoder = generator, encoder

    def register_buffer(self, name, value, persistent):
        setattr(self, name, value)


class Features:
    def __init__(self, x):
        self.x = x

    def expand(self, batch, *shape):
        return np.broadcast_to(self.x, (batch, *self.x.shape[1:]))


class BehaviorTests(unittest.TestCase):
    def test_visual_dropout_and_constant_semantics(self):
        tree = ast.parse((HERE / 'model.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        namespace = {'torch': SimpleNamespace(zeros_like=np.zeros_like), 'TouchTrainingModel': FakeBase}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(HERE / 'model.py'), 'exec'), namespace)
        class Encoder:
            touch_embedding = 1
            def __call__(self, points, mask):
                return self.output_projection(points) + self.touch_embedding
            def output_projection(self, features):
                return features * 2
        class Generator:
            def loss(self, targets, visual, **kwargs):
                self.visual, self.tokens = visual, kwargs.get('touch_tokens')
                return 0., {}
        gen = Generator(); model = namespace['UpperBoundModel'](gen, Encoder(), True)
        visual = np.ones((4, 5, 3)); points = np.ones((4, 2, 3))
        model.forward({}, (visual,), {}, points, None, drop_visual=True)
        self.assertFalse(gen.visual.any()); self.assertTrue((gen.tokens == 3).all())
        self.assertTrue((visual == 1).all())
        model.forward({}, (visual,), {}, points, None)
        self.assertTrue((gen.visual == 1).all())
        model.constant_features = Features(np.full((1, 2, 3), 7.))
        a = model.surface_tokens(points, None)
        b = model.surface_tokens(points * 91, None)
        np.testing.assert_array_equal(a, b); self.assertTrue((a == 15).all())

    def make_reports(self, root):
        evidence = {'sources': 'same'}
        for arm in ARMS:
            p = root / arm / 'assessment_20'; p.mkdir(parents=True)
            contract = dict(seed=29, batch_size=4, evidence=evidence, cross_attention_scope='full',
                            learning_rate=1e-4, cross_attention_learning_rate=1e-5)
            config = dict(contract=contract, initial_ca_sha256='same', initial_adapter_sha256='same')
            (p.parent / 'config.json').write_text(json.dumps(config))
            report = dict(complete=True, epoch=20, step=100, inputs=[], native=[], samples=[])
            for split in ('train', 'val'):
                report['inputs'].append(dict(split=split, group=0, targets_sha256='same', tokens_sha256=arm))
                for condition in (['correct', 'wrong_1'] if arm in ('oracle_dropout', 'camera_dropout') else ['correct']):
                    for oid in range(4):
                        value = .99 if arm == 'oracle_dropout' and condition == 'correct' else .5
                        report['samples'].append(dict(split=split, sample_id=f'{split}_{oid}', object_id=str(oid),
                            draw=0, condition=condition, noise_sha256={'shape': str(oid)}, iou=value,
                            precision_2v=value, recall_2v=value, fscore_2v=value))
            (p / 'results.json').write_text(json.dumps(report))

    def test_pairing_and_object_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_reports(root)
            result = analyze(root, 20)
            self.assertTrue(result['raw']['val']['oracle_dropout']['raw_endpoint_met'])
            self.assertAlmostEqual(result['paired']['val']['oracle_minus_image']['mean_difference'], .49)
            path = root / 'image/assessment_20/results.json'
            report = json.loads(path.read_text()); report['samples'][0]['noise_sha256'] = {'shape': 'wrong'}
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'Unpaired sampling noise'):
                analyze(root, 20)


if __name__ == '__main__':
    unittest.main()
