"""Check sweep inputs and reference compatibility without loading GPU models."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)
    return namespace


class PatchCountTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ns = functions('evaluation/evaluate.py', ['file_identity'], {'Path': Path})
        self.ns.update(np=np, json=json, yaml=yaml)
        functions('experiments/vecsetx/sam3d_patch_count.py',
                  ['check_reference', 'check_selected_inputs', 'encode_patches'], self.ns)

    def reference(self):
        pipeline = self.root / 'pipeline.yaml'
        pipeline.write_text('pipeline')
        manifest = self.root / 'manifest.jsonl'
        manifest.write_text('sample')
        data = {'dataset': {'root': str(self.root)}}
        selected = [{'sample_id': 'object_000', 'object_id': 'object', 'view_id': '000'}]
        settings = dict(seed=29, selection_data=data, pipeline=self.ns['file_identity'](pipeline),
                        dataset_files=[self.ns['file_identity'](manifest)])
        (self.root / 'evaluation_settings.json').write_text(json.dumps(settings))
        args = NS(reference_evaluation=self.root, pipeline_config=pipeline, seed=29)
        return args, selected, data, manifest

    def test_current_json_and_legacy_yaml_references(self):
        args, selected, data, _ = self.reference()
        for filename, serialize in [('selected_samples.json', json.dumps),
                                    ('selected_samples.yaml', yaml.safe_dump)]:
            with self.subTest(filename=filename):
                path = self.root / filename
                path.write_text(serialize(selected))
                self.ns['check_reference'](args, selected, data)
                with self.assertRaisesRegex(ValueError, 'objects/views differ'):
                    self.ns['check_reference'](args, [dict(selected[0], view_id='001')], data)
                path.unlink()

    def test_reference_rejects_changed_protocol_and_dataset(self):
        args, selected, data, manifest = self.reference()
        (self.root / 'selected_samples.json').write_text(json.dumps(selected))
        args.seed = 30
        with self.assertRaisesRegex(ValueError, 'seed differs'):
            self.ns['check_reference'](args, selected, data)
        args.seed = 29
        manifest.write_text('changed manifest')
        with self.assertRaisesRegex(ValueError, 'dataset file changed'):
            self.ns['check_reference'](args, selected, data)

    def test_camera_targets_can_differ_but_observations_cannot(self):
        record = dict(sample_id='a', mesh_path='mesh', image_path='image',
                      camera_path='camera', depth_path='depth', target_path='canonical_target')
        records = {'a': dict(record, target_path='camera_target')}
        dataset = NS(resolve_path=lambda path: self.root / path)
        canonical = {'dataset': {}}
        camera = {'dataset': {'target_frame': 'camera'}}
        check = self.ns['check_selected_inputs']
        check([record], dataset, records, dataset, canonical, camera)
        with self.assertRaisesRegex(ValueError, 'target_path'):
            check([record], dataset, records, dataset, camera, camera)
        records['a']['image_path'] = 'other_image'
        with self.assertRaisesRegex(ValueError, 'image_path'):
            check([record], dataset, records, dataset, canonical, camera)

    def test_sparse_inputs_keep_pointmap_and_whole_nested_patches(self):
        subsets = functions('experiments/vecsetx/point_count_sweep.py', ['subset_indices'], {'np': np})
        self.ns['subset_indices'] = lambda points, count: subsets['subset_indices'](
            points, count, baseline_points_per_patch=224, prefix_points=1024)
        points = np.arange(8192 * 3).reshape(1, 8192, 3)
        mask = np.ones((1, 8192), dtype=bool)
        shift, scale = object(), object()
        encoder = NS(prepare_points=Mock(return_value=(points, mask, shift, scale)))
        model = NS(touch_encoder=encoder, get_touch_tokens=Mock(return_value='baseline'))
        encode = Mock(return_value='sparse')
        self.ns['touch_tokens'] = encode
        self.assertEqual(self.ns['encode_patches'](model, 32, points, mask), 'baseline')
        encoder.prepare_points.assert_not_called()
        for count in (24, 16, 12, 8, 4, 1):
            with self.subTest(count=count):
                self.assertEqual(self.ns['encode_patches'](model, count, points, mask), 'sparse')
                args = encode.call_args.args
                np.testing.assert_array_equal(args[1], points[:, :1024 + count * 224])
                self.assertIs(args[2], shift)
                self.assertIs(args[3], scale)


if __name__ == '__main__':
    unittest.main()
