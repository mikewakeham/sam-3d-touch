"""Prevent evaluation results from being reused with different weights, data or settings."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate import register_runs


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'evaluation'
        self.out.mkdir()
        self.data = {'dataset': {'root': str(self.root), 'split': 'val'}}
        self.protocol = {'samples': ['sample'], 'seed': 29}

    def checkpoint(self, name, step=50):
        folder = self.root / name
        folder.mkdir(exist_ok=True)
        (folder / 'config.yaml').write_text(json.dumps({'data': self.data}))
        path = folder / 'best.pt'
        path.write_text(str(step))
        return path

    def describe(self, path):
        return {'step': int(path.read_text()), 'checkpoint': str(path), 'data': self.data,
                'touch_config': {}, 'conditioning_config': {'no_pointmap': False, 'oracle_point_frame': False},
                'mode': 'image_touch', 'train_scope': 'shape_cross_attention'}

    def register(self, paths, previous=None):
        return register_runs(self.out, self.protocol, paths, self.describe, previous or {}, self.data)

    def test_append_subset_reorder_and_overwritten_checkpoint(self):
        a, b = self.checkpoint('a'), self.checkpoint('b', 60)
        self.assertEqual(self.register([a])[0], ['a'])
        self.assertEqual(self.register([b, a])[0], ['b', 'a'])
        self.assertEqual(set(self.register([b])[1]['entries']), {'a', 'b'})
        a.write_text('100')
        names, settings = self.register([a])
        self.assertTrue(names[0].startswith('a_step100_'))
        self.assertEqual(settings['entries']['a']['metadata']['step'], 50)
        self.assertEqual(self.register([a])[0], names)

    def test_copy_reuses_content_and_config_change_creates_entry(self):
        a = self.checkpoint('a')
        self.register([a])
        copied = self.checkpoint('copy')
        self.assertEqual(self.register([copied])[0], ['a'])
        (copied.parent / 'config.yaml').write_text('{"changed": true}')
        self.assertEqual(self.register([copied])[0], ['copy'])

    def test_changed_effective_data_cannot_reuse_results(self):
        a = self.checkpoint('a')
        self.register([a])
        path = self.out / 'evaluation_settings.json'
        before = path.read_bytes()
        self.data = dict(self.data, touch={'source': 'touch_patches'})
        with self.assertRaisesRegex(ValueError, 'Data configuration changed for a'):
            self.register([a])
        self.assertEqual(path.read_bytes(), before)

    def test_protocol_mismatch_does_not_write(self):
        a = self.checkpoint('a')
        self.register([a])
        before = (self.out / 'evaluation_settings.json').read_bytes()
        self.protocol['seed'] = 30
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            self.register([a])
        self.assertEqual((self.out / 'evaluation_settings.json').read_bytes(), before)

    def test_results_without_settings_are_not_reinterpreted(self):
        (self.out / 'metrics.csv').write_text('old results')
        with self.assertRaisesRegex(ValueError, 'no resume settings'):
            self.register([self.checkpoint('a')])


if __name__ == '__main__':
    unittest.main()
