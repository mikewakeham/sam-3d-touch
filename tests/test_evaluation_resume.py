"""Exercise cumulative bookkeeping and the evaluator loop without GPU dependencies."""
import argparse
import ast
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluator_functions():
    source = ast.parse((ROOT / 'evaluation/evaluate.py').read_text())
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(globals(), re=__import__('re'))
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'evaluate.py', 'exec'), namespace)
    return namespace


_bookkeeping = evaluator_functions()
file_identity = _bookkeeping['file_identity']
register_runs = _bookkeeping['register_runs']


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

    def test_two_checkpoints_same_directory(self):
        best = self.checkpoint('a')
        last = best.with_name('last.pt')
        last.write_text('75')
        names, _ = self.register([best, last])
        self.assertEqual(len(set(names)), 2)

    def legacy(self, checkpoint, previous=True):
        old = dict(self.protocol, format_version=1,
                   checkpoints=[file_identity(checkpoint)],
                   run_configs=[file_identity(checkpoint.parent / 'config.yaml')])
        (self.out / 'evaluation_settings.json').write_text(json.dumps(old))
        metadata = self.describe(checkpoint)
        return {'data_config': self.data, 'runs': {'a': metadata}} if previous else {}

    def test_legacy_unchanged_keeps_name_and_backup(self):
        a = self.checkpoint('a')
        previous = self.legacy(a)
        before = (self.out / 'evaluation_settings.json').read_bytes()
        self.assertEqual(self.register([a], previous)[0], ['a'])
        self.assertEqual((self.out / 'evaluation_settings.v1.json').read_bytes(), before)

    def test_legacy_overwritten_or_missing_not_bound_to_new_weights(self):
        a = self.checkpoint('a')
        previous = self.legacy(a)
        a.write_text('100')
        names, settings = self.register([a], previous)
        self.assertNotEqual(names[0], 'a')
        self.assertIsNone(settings['entries']['a']['checkpoint_sha256'])
        self.assertEqual(settings['entries']['a']['metadata']['step'], 50)
        a.unlink()
        b = self.checkpoint('b', 75)
        self.assertIn('a', self.register([b])[1]['entries'])

    def test_interrupted_legacy_can_recover_metadata_from_unchanged_checkpoint(self):
        a = self.checkpoint('a')
        self.legacy(a, previous=False)
        self.assertEqual(self.register([a])[0], ['a'])

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

    def test_csv_append_preserves_extended_header_and_success_over_failed_shard(self):
        ns = evaluator_functions()
        row = {'condition': 'a', 'sample_id': 's', 'error': '', 'stage1': 0.2}
        path = self.out / 'metrics.csv'
        ns['write_metrics'](path, [row])
        ns['append_metric'](path, {'condition': 'b', 'sample_id': 's', 'error': ''})
        rows = ns['load_metrics'](path)
        self.assertEqual(rows[1]['stage1'], 'nan')
        shard = self.out / 'metrics_rank0.csv'
        ns['write_metrics'](shard, [dict(row, error='failed')])
        ns['artifacts_complete'] = lambda r, _: not r['error']
        merged = ns['merged_metrics']([path, shard], self.out)
        self.assertEqual(merged[('a', 's')]['error'], '')

    def test_existing_stage1_metrics_do_not_require_reference_or_recomputation(self):
        ns = evaluator_functions()
        row = {'condition': 'a', 'sample_id': 's', 'error': '',
               'stage1_aligned_chamfer': 0., 'stage1_aligned_voxel_iou_64': 1.,
               'stage1_downsample_factor': 1}
        ns['aligned_stage1_points'] = Mock(side_effect=AssertionError('must reuse metrics'))
        ns['add_stage1_metrics']([row], self.out, 1)
        self.assertEqual(row['stage1_aligned_chamfer'], 0.)

    def test_stale_rank_shard_cannot_erase_derived_metrics(self):
        ns = evaluator_functions()
        ns['artifacts_complete'] = lambda r, _: not r['error']
        row = {'condition': 'a', 'sample_id': 's', 'error': '', 'stage1_aligned_chamfer': .1}
        canonical, shard = self.out / 'metrics.csv', self.out / 'metrics_rank3.csv'
        ns['write_metrics'](canonical, [row])
        ns['write_metrics'](shard, [dict(row, stage1_aligned_chamfer='nan')])
        merged = ns['merged_metrics']([canonical, shard], self.out)
        self.assertEqual(float(merged[('a', 's')]['stage1_aligned_chamfer']), .1)

    def test_summary_multiple_baselines_and_empty_entry(self):
        ns = evaluator_functions()
        rows = [{'condition': n, 'sample_id': 's', 'error': '', 'fscore_0.01': score}
                for n, score in [('a', .2), ('b', .3), ('touch', .4)]]
        summary = ns['summarize'](rows, ['touch', 'b', 'a', 'empty'], [], ['a', 'b'], 'touch')
        self.assertEqual(summary['conditions']['empty']['completed'], 0)
        self.assertIsNone(summary['no_touch'])
        self.assertEqual(summary['comparisons']['touch_vs_a']['fscore_0.01']['paired_samples'], 1)
        self.assertIn('touch_vs_b', summary['comparisons'])

    def test_main_add_subset_repeat_and_later_checkpoint(self):
        ns = evaluator_functions()
        a, b = self.checkpoint('a'), self.checkpoint('b', 75)
        image = self.root / 'image.png'
        image.write_bytes(b'image')
        pipeline = self.root / 'pipeline.yaml'
        pipeline.write_text('{}')
        record = {'sample_id': 's', 'object_id': 'o', 'view_id': 'v', 'image_path': str(image)}
        args = NS(checkpoints=[a], run_dirs=None, output_dir=self.out, data_config=None,
                  pipeline_config=pipeline, selection_data_config=None, split='val', selection='random',
                  max_samples=1, selection_seed=29, seed=29, inference_steps=25, stage2_inference_steps=25,
                  surface_points=10, icp_points=10, emd_points=0, save_points=10, no_amp=False,
                  batch_size=1, metric_workers=1, workers=0, device='cpu')
        ns['parse_args'] = lambda: args
        ns['torch'] = NS(manual_seed=lambda _: None, cuda=NS(is_available=lambda: False))
        ns['yaml'] = NS(safe_load=lambda value: json.loads(value.read() if hasattr(value, "read") else value), safe_dump=lambda data, file, **kw: json.dump(data, file))
        ns['select_records'] = lambda *a: ([record], [dict(record)])
        ns['read_run'] = lambda path, *_: (self.describe(path), {}, json.loads(json.dumps(self.data)))
        ns['build_pipeline'] = Mock(side_effect=lambda *_: (NS(ss_generator=NS()), {}))
        ns['restore_run'] = lambda *_: NS(touch_encoder=object(), get_touch_tokens=None)
        ns['load_target_mesh'] = Mock(return_value={})
        dependent_conditions = []
        ns['add_stage1_metrics'] = lambda rows, *_: dependent_conditions.append({r['condition'] for r in rows})
        calls = []
        interrupt_once = [True]

        def evaluate(name, pipeline, encoder, loader, records, targets, completed, path, args, **kw):
            calls.append(name)
            row = {'condition': name, 'sample_id': 's', 'object_id': 'o', 'view_id': 'v',
                   'error': '', 'fscore_0.01': .5}
            for key in ('stage1_path', 'slat_path', 'mesh_raw_path', 'mesh_normalized_path',
                        'mesh_aligned_path', 'alignment_path', 'target_mesh_path', 'target_points_path'):
                artifact = args.output_dir / f'{name}_{key}'
                artifact.write_bytes(b'valid')
                row[key] = artifact.name
            ns['append_metric'](path, row)
            completed.add((name, 's'))
            if interrupt_once.pop() if interrupt_once else False:
                raise RuntimeError('simulated interruption after committed sample')
            return [row]

        ns['evaluate_condition'] = evaluate
        def loader(*args, **kwargs):
            return NS(dataset=NS(records=[record], resolve_path=Path))
        modules = {'dataloader': NS(build_dataloader=loader, load_data_config=lambda p: self.data),
                   'train': NS(checkpoint_train_scope=lambda _: 'shape_cross_attention', configure_encoder_data=lambda d, _: d),
                   'omegaconf': NS(OmegaConf=NS(to_container=lambda d, **kw: d))}
        # Encoder metadata is used by the real loop when selecting input configuration.
        original_describe = self.describe
        self.describe = lambda p: dict(original_describe(p), touch_config={'encoder_name': 'test'})
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {'WORLD_SIZE': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                ns['main']()
            ns['main']()
            self.assertEqual(calls, ['official', 'decoded_gt', 'a'])
            args.checkpoints = [b]
            ns['main']()
            self.assertEqual(calls, ['official', 'decoded_gt', 'a', 'b'])
            self.assertEqual(dependent_conditions[-1], {'official', 'decoded_gt', 'a', 'b'})
            ns['main']()
            self.assertEqual(len(calls), 4)
            a.write_text('100')
            args.checkpoints = [a]
            ns['main']()
        rows = ns['load_metrics'](self.out / 'metrics.csv')
        self.assertEqual(len(rows), 5)
        config = json.loads((self.out / 'config.yaml').read_text())
        self.assertEqual(len(config['runs']), 3)
        self.assertEqual(config['runs']['a']['step'], 50)
        self.assertEqual(ns['build_pipeline'].call_count, 5)


if __name__ == '__main__':
    unittest.main()
