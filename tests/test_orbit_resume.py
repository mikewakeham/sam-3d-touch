"""Exercise incremental orbit rendering without Blender, Open3D or GPU work."""
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import make_orbit as orbit
from evaluation.input_visualizations import input_groups, saved_surface_names


class OrbitResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'orbits' / 'sample'
        self.image = self.root / 'image.png'
        self.image.write_bytes(b'image')
        self.conditions = ['baseline', 'full']
        self.rendered = []
        self.fail_on = None
        self.camera = {'orbit_axis': 'input_camera_up', 'pivot': [0, 0, 0],
                       'camera_to_world': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]}
        for target, replacement in (
            ('evaluation.make_orbit.load_inputs', self.load_inputs),
            ('evaluation.make_orbit.load_reference_camera', lambda _: self.camera),
            ('evaluation.make_orbit.render', self.render),
            ('evaluation.input_visualizations.load_input_visualizations', self.load_visualizations),
        ):
            mock = patch(target, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def item(self, name, size=1):
        return {'name': name, 'mesh': trimesh.creation.box(extents=[size] * 3)}

    def selected(self, args):
        return args.conditions or self.conditions

    def load_inputs(self, args):
        items = [self.item('mesh_ground_truth')]
        items.extend(self.item(f'{mode}_{name}', 2 if name == 'patch' else 1)
                     for mode in args.modes for name in self.selected(args))
        return items, [self.image]

    def load_visualizations(self, args, previous):
        surfaces = [(self.item('surface'), [name]) for name in self.selected(args) if name != 'baseline']
        names = saved_surface_names(previous)
        groups = input_groups(self.item('input_mesh'), self.item('textured'), self.item('pointmap'), surfaces, names)
        return groups, [self.image], {'surface_names': names, 'surface_groups': [members for _, members in surfaces]}

    def render(self, items, center, radius, args, output, scale, reference_camera=None):
        self.rendered.append(output.name)
        for extension in ('.gif', '.png', '.mp4') if args.mp4 else ('.gif', '.png'):
            output.with_suffix(extension).write_bytes(f'{output.name}:{args.width}'.encode())
        if output.name == self.fail_on:
            raise RuntimeError('interrupted render')

    def argv(self, extra=()):
        return ['orbit', '--evaluation-dir', str(self.root), '--sample-id', 'sample',
                '--with-inputs', '--modes', 'mesh', 'voxel', *extra]

    def run_orbits(self, extra=()):
        self.rendered.clear()
        with patch.object(sys, 'argv', self.argv(extra)):
            orbit.main()
        return list(self.rendered)

    def args(self, extra=()):
        with patch.object(sys, 'argv', self.argv(extra)):
            return orbit.parse_args()

    def snapshots(self):
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.output.iterdir() if p.suffix in ('.gif', '.png')}

    def make_legacy(self):
        self.run_orbits()
        settings = orbit.read_orbit_settings(self.output)
        for filename in self.output.glob('input*_full.*'):
            filename.rename(filename.with_name(filename.name.replace('_full.', '.')))
        settings['geometry_names'] = [name.removesuffix('_full') if name.startswith('input') else name
                                      for name in settings['geometry_names']]
        settings.pop('renders')
        settings.pop('scale')
        settings['input_details'].pop('surface_names')
        orbit.write_orbit_settings(self.output, settings)

    def test_add_patch_to_legacy_preserves_files_names_and_framing(self):
        self.make_legacy()
        before = self.snapshots()
        self.conditions.append('patch')
        rendered = self.run_orbits()
        self.assertEqual(set(rendered), {'mesh_patch', 'voxel_patch', 'input_mesh_surface_patch',
                                        'input_mesh_pointmap_surface_patch', 'input_surface_patch',
                                        'input_pointmap_surface_patch'})
        after = self.snapshots()
        self.assertTrue(all(after[name] == value for name, value in before.items()))
        settings = orbit.read_orbit_settings(self.output)
        self.assertEqual(settings['scale'], 1.)  # New patch mesh is twice as large.
        self.assertEqual(settings['input_details']['surface_names'], {'full': '', 'patch': '_patch'})
        self.assertTrue(orbit.orbits_complete(self.output, self.args(), self.conditions, ['full', 'patch']))
        self.assertEqual(self.run_orbits(), [])
        self.assertEqual(self.run_orbits(['--conditions', 'patch']), [])
        self.assertIn('mesh_full', orbit.read_orbit_settings(self.output)['renders'])

    def test_interrupted_render_keeps_other_completed_variants(self):
        self.run_orbits()
        self.conditions.append('patch')
        self.fail_on = 'input_mesh_pointmap_surface_patch'
        with self.assertRaisesRegex(RuntimeError, 'interrupted render'):
            self.run_orbits()
        records = orbit.read_orbit_settings(self.output)['renders']
        self.assertIn('input_mesh_surface_patch', records)
        self.assertNotIn(self.fail_on, records)
        self.fail_on = None
        rendered = self.run_orbits()
        self.assertNotIn('input_mesh_surface_patch', rendered)
        self.assertNotIn('mesh_baseline', rendered)
        self.assertIn('input_mesh_pointmap_surface_patch', rendered)

    def test_incomplete_files_changed_settings_and_overwrite(self):
        self.run_orbits()
        (self.output / 'mesh_full.gif').write_bytes(b'')
        self.assertEqual(self.run_orbits(), ['mesh_full'])
        old = self.snapshots()
        changed = self.run_orbits(['--conditions', 'full', '--width', '1024'])
        self.assertIn('mesh_full', changed)
        self.assertNotIn('mesh_baseline', changed)
        self.assertEqual(self.snapshots()['mesh_baseline.png'], old['mesh_baseline.png'])
        self.assertFalse(orbit.orbits_complete(self.output, self.args(['--width', '1024']), self.conditions, ['full']))
        self.assertEqual(set(self.run_orbits(['--width', '1024'])), {'mesh_baseline', 'voxel_baseline'})
        self.assertEqual(self.run_orbits(['--width', '1024']), [])
        self.assertEqual(len(self.run_orbits(['--width', '1024', '--overwrite'])), 12)
        self.assertEqual(len(self.run_orbits(['--width', '1024', '--mp4'])), 12)
        self.assertEqual(self.run_orbits(['--width', '1024', '--mp4']), [])

    def test_batch_skips_complete_samples_without_forcing_overwrite(self):
        self.run_orbits()
        with (self.root / 'metrics.csv').open('w') as file:
            writer = csv.DictWriter(file, fieldnames=['sample_id', 'condition', 'error'])
            writer.writeheader()
            for name in self.conditions:
                writer.writerow({'sample_id': 'sample', 'condition': name, 'error': ''})
        (self.root / 'config.yaml').write_text(json.dumps({'runs': {'full': {'touch_config': {'encoder_name': 'test'}}}}))
        args = self.args()
        args.sample_id = None
        args.all_samples = True
        arguments = ['--evaluation-dir', str(self.root), '--all-samples', '--with-inputs', '--modes', 'mesh', 'voxel']
        with patch.object(orbit.subprocess, 'run') as child:
            orbit.render_all(args, arguments)
            child.assert_not_called()
            (self.output / 'mesh_full.png').unlink()
            orbit.render_all(args, arguments)
            self.assertEqual(child.call_count, 1)
            self.assertNotIn('--overwrite', child.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
