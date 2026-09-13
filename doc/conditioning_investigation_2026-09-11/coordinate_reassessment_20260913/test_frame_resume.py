"""Exercise resume against the actual returned artifacts without meshes or CUDA."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import frame_probe_gpu as probe

ROOT = probe.REPO/'outputs/conditioning_investigation/full_frame_probe_20260913_170408/camera'


class ResumeTests(unittest.TestCase):
    def run_resume(self, changed=None):
        saved = json.loads((ROOT/'results.partial.json').read_text())
        records = []
        hashes = {}
        objects = {x['object_id']: x for x in saved['geometry_objects']}
        for x in saved['geometry_samples']:
            sid, oid = x['sample_id'], x['object_id']
            record = {'sample_id': sid, 'object_id': oid}
            for key, digest in [('object_transform_path', objects[oid]['object_transform_sha256']),
                                ('target_path', objects[oid]['target_file_sha256']),
                                ('full_surface_path', x['surface_sha256']),
                                ('camera_path', x['camera_sha256'])]:
                record[key] = f'/mock/{sid}/{key}'
                hashes[Path(record[key])] = digest
            hashes[Path('/mock/objects')/oid/'model.obj'] = objects[oid]['mesh_sha256']
            records.append(record)
        hashes[Path('/mock/encoder')] = saved['target_encoder_sha256']
        hashes[probe.REPO/'data_generation/objaverse-dexonomy/generate_target_latents.py'] = saved['target_generation_source_sha256']
        if changed == 'target':
            hashes[Path(records[0]['target_path'])] = 'changed'
        original_sha = probe.sha
        def digest(path):
            path = Path(path)
            if changed == 'report' and path == ROOT/'results.partial.json':
                return 'changed'
            return hashes[path] if path in hashes else original_sha(path)
        ds = SimpleNamespace(root=Path('/mock'), records=records, resolve_path=Path)
        report = copy.deepcopy(saved)
        for key in ('geometry_objects', 'geometry_samples'):
            report[key] = []
        report['files_sha256'] = {}
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(resume_geometry_from=ROOT, encoder_checkpoint=Path('/mock/encoder'), output=Path(folder))
            with patch.object(probe, 'sha', side_effect=digest):
                probe.geometry_reference(args, {'train': ds},
                                         [{'split': 'train', 'sample_ids': [r['sample_id'] for r in records]}],
                                         report, None)
            self.assertTrue(report['coordinate_reference_passed'])
            self.assertFalse(report['geometry_reuse']['mesh_queries_rerun'])
            self.assertFalse(report['geometry_reuse']['target_encodings_rerun'])
            self.assertFalse(report['geometry_reuse']['original_nearest_query_gate_passed'])
            self.assertEqual(len(report['geometry_samples']), 64)
            self.assertEqual(report['files_sha256'], saved['files_sha256'])
            for name, expected in report['files_sha256'].items():
                self.assertEqual(original_sha(Path(folder)/name), expected)

    def test_actual_artifacts_resume_without_cuda_or_open3d(self):
        self.run_resume()

    def test_changed_current_target_rejected(self):
        with self.assertRaises(AssertionError):
            self.run_resume('target')

    def test_unaudited_report_rejected(self):
        with self.assertRaises(AssertionError):
            self.run_resume('report')


if __name__ == '__main__':
    unittest.main()
