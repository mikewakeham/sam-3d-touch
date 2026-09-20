"""CPU tests; set BLENDER_BIN to include real Blender import/render checks."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import trimesh
from PIL import Image

from generate_target_latents import load_normalized_mesh, save_target, validate_target, voxelize_mesh
from make_data import get_objects, make_splits, parse_args, render_object
from pointmaps import depth_to_pointmap
from sample_full_surface import classify_visibility, transform_points

HERE = Path(__file__).resolve().parent
BLENDER = os.environ.get("BLENDER_BIN")


class GeometryTests(unittest.TestCase):
    def test_projection_and_visibility(self):
        depth = np.full((5, 5), 2, dtype=np.float32)
        depth[0, 0] = np.nan
        K = np.array([[4, 0, 2], [0, 4, 2], [0, 0, 1]], dtype=np.float32)
        points = depth_to_pointmap(depth, K)
        np.testing.assert_array_equal(points[2, 2], [0, 0, 2])
        np.testing.assert_array_equal(points[2, 3], [-.5, 0, 2])
        np.testing.assert_array_equal(points[3, 2], [0, -.5, 2])
        self.assertTrue(np.isnan(points[0, 0]).all())
        labels = classify_visibility(np.array([[0, 0, 2], [0, 0, 3], [0, 0, -1], [20, 0, 2]]),
                                     K, np.eye(4), depth, .005)
        np.testing.assert_array_equal(labels, [1, 0, -1, -1])

    def test_discovery_ids_and_split_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ['a/model.obj', 'b/model.obj', 'shape/cup weird.GLb']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            objects = get_objects(root)
            self.assertEqual(len({o['object_id'] for o in objects}), 3)
            subset = get_objects(root, 'a/*.obj')
            self.assertEqual(subset[0]['object_id'], objects[0]['object_id'])
            splits = make_splits(objects, 29, .5, .25)
            self.assertEqual(sum(map(len, splits.values())), 3)
            self.assertFalse(set(splits['train']) & set(splits['test']))

    def test_splits_do_not_depend_on_arrival_order(self):
        objects = [{'object_id': f'object-{i}'} for i in range(100)]
        first = make_splits(objects[:70], 29, .8, .1)
        grown = make_splits(objects, 29, .8, .1, first)
        self.assertEqual(grown, make_splits(list(reversed(objects)), 29, .8, .1))
        for split in first:
            self.assertTrue(set(first[split]).issubset(grown[split]))
        # Preserve existing published assignments, including datasets made before hash splitting.
        preserved = make_splits(objects, 29, .8, .1, {'train': ['object-0'], 'val': ['object-1'], 'test': ['object-2']})
        for split, object_id in [('train', 'object-0'), ('val', 'object-1'), ('test', 'object-2')]:
            self.assertIn(object_id, preserved[split])

    def test_invalid_latent_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'target_latent.npz'
            with self.assertRaises(ValueError):
                save_target(np.full((8, 16, 16, 16), np.nan, dtype=np.float32), target)
            self.assertFalse(target.exists())

    def test_gpu_process_isolation(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(['--data-root', temporary, '--render-device', 'OPTIX'])
            with patch('make_data.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run:
                render_object({'model_path': '/input/object.glb'}, Path(temporary), args, 'GPU-test-uuid')
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--device') + 1], 'OPTIX')
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-test-uuid')
            self.assertNotEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), 'GPU-test-uuid')

    def test_standalone_vae_checkpoint_roundtrip(self):
        import torch
        from sparse_structure_vae import SparseStructureEncoderTdfyWrapper
        torch.set_num_threads(1)
        # Small random model exercises the actual copied implementation on CPU.
        # It does not validate the unavailable pretrained checkpoint's accuracy.
        options = dict(in_channels=1, latent_channels=8, channels=[32, 32, 32],
                       num_res_blocks=1, num_res_blocks_middle=1,
                       sample_posterior=False, return_raw=True)
        encoder = SparseStructureEncoderTdfyWrapper(**options).eval()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'test_encoder.ckpt'
            torch.save(encoder.state_dict(), checkpoint)
            restored = SparseStructureEncoderTdfyWrapper(pretrained_ckpt_path=str(checkpoint), **options).eval()
            occupancy = voxelize_mesh(trimesh.creation.box()).unsqueeze(0)
            with torch.inference_mode():
                expected = encoder(occupancy)['mean']
                actual = restored(occupancy)['mean']
            self.assertEqual(tuple(actual.shape), (1, 8, 16, 16, 16))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(any(name.startswith('sam3d_objects') for name in sys.modules))
            self.assertNotIn('spconv', sys.modules)
            path = Path(temporary) / 'target.npz'
            save_target(actual[0].numpy(), path)
            np.testing.assert_array_equal(validate_target(path), expected[0].numpy())

    def test_surface_voxelization(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CPU torch is needed for occupancy tensor checks')
        mesh = trimesh.creation.box()
        occupancy = voxelize_mesh(mesh).numpy()[0]
        self.assertEqual(occupancy.shape, (64, 64, 64))
        self.assertEqual(occupancy.dtype, np.float32)
        self.assertEqual(occupancy[32, 32, 32], 0)  # Surface, not solid occupancy.
        self.assertEqual(occupancy.sum(), 64**3 - 62**3)
        for axis in range(3):
            self.assertTrue(np.take(occupancy, 0, axis=axis).all())
            self.assertTrue(np.take(occupancy, 63, axis=axis).all())


@unittest.skipUnless(BLENDER, 'Set BLENDER_BIN to run CPU Blender integration tests')
class BlenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='general_pipeline_test_')
        cls.root = Path(cls.temp.name)
        cls.inputs = cls.root / 'inputs'
        cls.inputs.mkdir()
        mesh = trimesh.creation.box(extents=[1, .7, .5])
        mesh.apply_translation([2, 3, 4])
        for extension in ('obj', 'ply', 'stl'):
            mesh.export(cls.inputs / f'box.{extension}')
        # Texture + transformed multi-mesh hierarchy.
        uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]] * 2)
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=Image.new('RGB', (8, 8), (210, 35, 15)))
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name='box', transform=trimesh.transformations.translation_matrix([1, 0, 0]))
        scene.add_geometry(trimesh.creation.icosphere(subdivisions=1, radius=.3), node_name='sphere',
                           transform=trimesh.transformations.translation_matrix([3, 3, 4.3]))
        scene.export(cls.inputs / 'hierarchy.glb')
        # Exercise a modifier, nonuniform scale and a collection instance in .blend.
        fixture_script = cls.root / 'fixture.py'
        fixture_script.write_text('''import bpy, sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--') + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(location=(2, 3, 4))
obj = bpy.context.object
obj.scale = (1, .7, .4)
modifier = obj.modifiers.new('bevel', 'BEVEL')
modifier.width = .12
modifier.segments = 2
collection = bpy.data.collections.new('asset')
bpy.context.scene.collection.children.link(collection)
for owner in list(obj.users_collection): owner.objects.unlink(obj)
collection.objects.link(obj)
instance = bpy.data.objects.new('instance', None)
instance.instance_type = 'COLLECTION'
instance.instance_collection = collection
instance.location = (2, 0, 0)
bpy.context.scene.collection.objects.link(instance)
bpy.ops.wm.save_as_mainfile(filepath=str(root / 'modifier.blend'))
bpy.ops.export_scene.gltf(filepath=str(root / 'scene.gltf'), export_format='GLTF_SEPARATE')
bpy.ops.export_scene.fbx(filepath=str(root / 'scene.fbx'))
bpy.ops.wm.usd_export(filepath=str(root / 'scene.usdc'))
''')
        result = subprocess.run([BLENDER, '-b', '--factory-startup', '--python-exit-code', '1',
                                 '-P', str(fixture_script), '--', str(cls.inputs)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.output = cls.root / 'dataset'
        cls.command = [sys.executable, str(HERE / 'make_data.py'), '--objects-root', str(cls.inputs),
                       '--data-root', str(cls.output), '--stage', 'render', '--render-device', 'CPU',
                       '--blender', BLENDER, '--num-views', '2', '--resolution', '64', '--samples', '4',
                       '--num-points', '256', '--workers', '2']
        result = subprocess.run(cls.command, capture_output=True, text=True)
        if result.returncode:
            logs = '\n'.join(path.read_text()[-3000:] for path in cls.output.rglob('blender.log'))
            raise AssertionError(result.stdout + result.stderr + logs)
        cls.records = [json.loads(line) for line in (cls.output / 'generated_data/samples_rendered.jsonl').read_text().splitlines()]

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('KEEP_PIPELINE_TEST_OUTPUT'):
            print(f'Test outputs: {cls.root}')
            cls.temp._finalizer.detach()
        else:
            cls.temp.cleanup()

    def test_alignment_all_formats(self):
        self.assertEqual(len(self.records), 16)
        cameras = {}
        intrinsics = {}
        for record in self.records:
            mesh = load_normalized_mesh(self.output / record['mesh_path'])
            with np.load(self.output / record['camera_path']) as camera:
                pointmap = depth_to_pointmap(np.load(self.output / record['depth_path']), camera['K'])
                T = np.diag([-1, -1, 1, 1]) @ camera['T_camera_from_object']
                # The saved camera must frame the unit cube's enclosing sphere,
                # using TRELLIS.2's distance/FOV relation without extra padding.
                half_fov = np.arctan(pointmap.shape[1] / (2 * camera['K'][0, 0]))
                radius = np.linalg.norm(T[:3, 3])
                self.assertAlmostEqual(radius * np.sin(half_fov), np.sqrt(3) / 2, places=6)
                previous_K = intrinsics.setdefault(record['view_id'], camera['K'])
                np.testing.assert_array_equal(camera['K'], previous_K)
                previous = cameras.setdefault(record['view_id'], T)
                np.testing.assert_array_equal(T, previous)
            points = pointmap[np.isfinite(pointmap).all(-1)]
            self.assertGreater(len(points), 30)
            object_points = transform_points(points, np.linalg.inv(T))
            _, distance, _ = trimesh.proximity.closest_point_naive(mesh, object_points[::4])
            self.assertLess(np.percentile(distance, 95), .001, record['sample_id'])
            with np.load(self.output / record['full_surface_path']) as data:
                self.assertEqual(data['points_camera'].dtype, np.float32)
                self.assertEqual(data['points_camera'].shape, (256, 3))
                object_surface = transform_points(data['points_camera'], np.linalg.inv(T))
                _, distance, _ = trimesh.proximity.closest_point_naive(mesh, object_surface)
                self.assertLess(distance.max(), 1e-5)
        self.assertFalse(np.array_equal(intrinsics['000'], intrinsics['001']))
        self.assertEqual((self.output / 'generated_data/samples.jsonl').read_text(), '')
        self.assertFalse(list(self.output.rglob('touches.npz')))
        self.assertFalse(list(self.output.rglob('pointmap.npy')))
        self.assertTrue(all('pointmap_path' not in row for row in self.records))

    def test_texture_survives_baking(self):
        record = next(row for row in self.records if row['object_id'].startswith('hierarchy-'))
        rgba = np.asarray(Image.open(self.output / record['image_path']))
        self.assertGreater(np.sum((rgba[..., 0].astype(float) > rgba[..., 1] * 1.5) & (rgba[..., 3] == 255)), 30)

    def test_eight_view_data_and_metadata_contract(self):
        import hashlib
        import torch
        from generate_target_latents import generate_target, save_metadata
        from make_data import write_manifests
        from sparse_structure_vae import SparseStructureEncoderTdfyWrapper
        sys.path.insert(0, str(HERE.parents[1]))
        from dataloader import TouchDataset

        output = self.root / 'eight_views'
        command = list(self.command)
        command[command.index('--data-root') + 1] = str(output)
        command += ['--limit', '1', '--num-views', '8', '--resolution', '768',
                    '--samples', '32', '--num-points', '8192',
                    '--train-fraction', '1', '--val-fraction', '0']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        generated = output / 'generated_data'
        settings = json.loads((generated / 'settings.json').read_text())
        self.assertEqual([settings[key] for key in ('num_views', 'resolution', 'samples', 'num_points')],
                         [8, 768, 32, 8192])
        self.assertEqual(len(settings['pipeline_sha256']), 64)
        objects = json.loads((generated / 'objects.json').read_text())
        rows = [json.loads(line) for line in (generated / 'samples_rendered.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertEqual({row['view_id'] for row in rows}, {f'{i:03d}' for i in range(8)})
        reference_surface = None
        for row in rows:
            for key in ('image_path', 'depth_path', 'camera_path', 'full_surface_path',
                        'object_transform_path', 'mesh_path'):
                self.assertTrue((output / row[key]).is_file(), key)
            rgba = np.asarray(Image.open(output / row['image_path']))
            depth = np.load(output / row['depth_path'])
            self.assertEqual(rgba.shape, (768, 768, 4))
            self.assertEqual(depth.dtype, np.float32)
            self.assertTrue(np.isnan(depth[rgba[..., 3] == 0]).all())
            with np.load(output / row['camera_path']) as camera, np.load(output / row['full_surface_path']) as surface:
                self.assertEqual(surface['points_camera'].shape, (8192, 3))
                self.assertEqual(surface['coordinate_frame'].item(), 'sam_camera')
                self.assertEqual(surface['point_visibility'].shape, (8192,))
                self.assertTrue(np.isin(surface['point_visibility'], [-1, 0, 1]).all())
                np.testing.assert_array_equal(surface['point_ids'], np.arange(8192))
                transform = np.diag([-1., -1., 1., 1.]) @ camera['T_camera_from_object']
                points = transform_points(surface['points_camera'], np.linalg.inv(transform))
                if reference_surface is None:
                    reference_surface = points
                np.testing.assert_allclose(points, reference_surface, atol=2e-6)

        # Exercise mesh -> encoder -> saved target -> manifest -> training loader.
        # Random small CPU weights verify the contract, not pretrained accuracy.
        torch.set_num_threads(1)
        encoder = SparseStructureEncoderTdfyWrapper(
            in_channels=1, latent_channels=8, channels=[32, 32, 32],
            num_res_blocks=1, num_res_blocks_middle=1,
            sample_posterior=False, return_raw=True,
        ).eval()
        checkpoint = self.root / 'audit_encoder.ckpt'
        torch.save(encoder.state_dict(), checkpoint)
        save_metadata(generated, checkpoint)
        metadata = json.loads((generated / 'target_latents.json').read_text())
        self.assertEqual(metadata['encoder_checkpoint_sha256'], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        target, created = generate_target(rows[0]['object_id'], output, encoder, 'cpu', False)
        self.assertTrue(created)
        before = target.stat().st_mtime_ns
        self.assertFalse(generate_target(rows[0]['object_id'], output, encoder, 'cpu', False)[1])
        self.assertEqual(target.stat().st_mtime_ns, before)
        rendered, ready = write_manifests(parse_args(command[2:]), objects)
        self.assertEqual((len(rendered), len(ready)), (8, 8))
        self.assertEqual(len({row['target_path'] for row in ready}), 1)
        dataset = TouchDataset({'dataset': {'root': str(output), 'manifest': 'generated_data/samples.jsonl',
                                           'split_file': 'generated_data/splits.json', 'split': 'train'},
                                'touch': {'source': 'full_surface'}})
        self.assertEqual(len(dataset), 8)
        sample = dataset[0]
        self.assertEqual(tuple(sample['target_shape'].shape), (4096, 8))
        self.assertEqual(tuple(sample['touch_xyz'].shape), (8192, 3))
        self.assertEqual(tuple(sample['pointmap'].shape), (768, 768, 3))
        checkpoint.write_bytes(b'different checkpoint')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            save_metadata(generated, checkpoint)

    def test_resume_repair_and_settings_guard(self):
        record = self.records[0]
        image = self.output / record['image_path']
        before = image.stat().st_mtime_ns
        surface = self.output / record['full_surface_path']
        original = np.load(surface)['points_camera']
        surface.unlink()
        result = subprocess.run(self.command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(image.stat().st_mtime_ns, before)
        np.testing.assert_array_equal(np.load(surface)['points_camera'], original)
        result = subprocess.run([*self.command, '--seed', '30'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Inputs/settings differ', result.stderr)
        self.assertEqual(image.stat().st_mtime_ns, before)

    def test_bad_object_does_not_stop_other_objects(self):
        import shutil
        inputs = self.root / 'bad_inputs'
        inputs.mkdir()
        shutil.copy2(self.inputs / 'hierarchy.glb', inputs / 'good.glb')
        (inputs / 'bad.glb').write_text('invalid GLB fixture')
        command = list(self.command)
        command[command.index('--objects-root') + 1] = str(inputs)
        output = self.root / 'failure_dataset'
        command[command.index('--data-root') + 1] = str(output)
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        failures = json.loads((output / 'generated_data/failed_objects.json').read_text())
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0]['object_id'].startswith('bad-'))
        lines = (output / 'generated_data/samples_rendered.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(json.loads(line)['object_id'].startswith('good-') for line in lines))

    def test_limit_then_resume_full_dataset(self):
        command = list(self.command)
        output = self.root / 'limited_dataset'
        command[command.index('--data-root') + 1] = str(output)
        first = subprocess.run([*command, '--limit', '1'], capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        manifest = output / 'generated_data/samples_rendered.jsonl'
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        image = output / rows[0]['image_path']
        mtime = image.stat().st_mtime_ns
        splits = (output / 'generated_data/splits.json').read_bytes()
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(len(manifest.read_text().splitlines()), 16)
        self.assertEqual(image.stat().st_mtime_ns, mtime)
        self.assertEqual((output / 'generated_data/splits.json').read_bytes(), splits)

    def test_growing_inputs_and_failed_import_retry(self):
        import shutil
        inputs = self.root / 'growing_inputs'
        inputs.mkdir()
        source = self.inputs / 'hierarchy.glb'
        for name, status in [('first', 'complete'), ('later', 'generating')]:
            shutil.copy2(source, inputs / f'{name}.glb')
            (inputs / f'{name}_generation.json').write_text(json.dumps({'status': status}))
        output = self.root / 'growing_dataset'
        command = list(self.command)
        command[command.index('--objects-root') + 1] = str(inputs)
        command[command.index('--data-root') + 1] = str(output)
        command += ['--ready-marker', '{stem}_generation.json']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = output / 'generated_data/samples_rendered.jsonl'
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        old_image = output / rows[0]['image_path']
        old_mtime = old_image.stat().st_mtime_ns
        old_split = rows[0]['split']
        target = old_image.parents[2] / 'target_latent.npz'
        # Synthetic target checks retention/publication only.
        save_target(np.zeros((8, 16, 16, 16), dtype=np.float32), target)
        target_mtime = target.stat().st_mtime_ns
        (inputs / 'later_generation.json').write_text(json.dumps({'status': 'complete'}))
        (inputs / 'broken.glb').write_bytes(b'incomplete GLB')
        (inputs / 'broken_generation.json').write_text(json.dumps({'status': 'complete'}))
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(len(manifest.read_text().splitlines()), 4)
        # A failed import can be replaced with its finished source and retried.
        shutil.copy2(source, inputs / 'broken.glb')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows_after = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual(len(rows_after), 6)
        self.assertEqual(old_image.stat().st_mtime_ns, old_mtime)
        self.assertEqual(target.stat().st_mtime_ns, target_mtime)
        self.assertEqual(next(row['split'] for row in rows_after if row['sample_id'] == rows[0]['sample_id']), old_split)
        ready = output / 'generated_data/samples.jsonl'
        self.assertEqual(len(ready.read_text().splitlines()), 2)
        # Mutating an already rendered source must not silently mix old and new geometry.
        with (inputs / 'first.glb').open('ab') as file:
            file.write(b'changed')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Previously rendered input changed', result.stderr)
        self.assertEqual(old_image.stat().st_mtime_ns, old_mtime)

    def test_fixed_scene_and_repeatable_lighting(self):
        outputs = []
        for name in ('fixed_first', 'fixed_repeat'):
            output = self.root / name
            command = list(self.command)
            command[command.index('--data-root') + 1] = str(output)
            result = subprocess.run([*command, '--limit', '1', '--camera-mode', 'fixed',
                                     '--lighting', 'fixed'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = [json.loads(line) for line in (output / 'generated_data/samples_rendered.jsonl').read_text().splitlines()]
            outputs.append((output, rows))
        left, right = outputs
        for a, b in zip(left[1], right[1]):
            np.testing.assert_array_equal(np.asarray(Image.open(left[0] / a['image_path'])),
                                          np.asarray(Image.open(right[0] / b['image_path'])))
            with np.load(left[0] / a['camera_path']) as camera:
                self.assertAlmostEqual(float(camera['K'][0, 0]), 32 / np.tan(np.deg2rad(20)), places=4)

    def test_training_loader_contract(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CPU torch is needed to exercise the real training loader')
        sys.path.insert(0, str(HERE.parents[1]))
        from dataloader import TouchDataset
        # Synthetic targets test serialization/loading only, never encoder accuracy.
        rows = []
        for original in self.records:
            row = dict(original)
            path = self.output / 'generated_data' / row['object_id'] / 'test_target.npz'
            np.savez(path, mean=np.zeros((8, 16, 16, 16), dtype=np.float32))
            validate_target(path)
            row['target_path'] = str(path.relative_to(self.output))
            rows.append(row)
        manifest = self.output / 'test_samples.jsonl'
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        config = {'dataset': {'root': str(self.output), 'manifest': 'test_samples.jsonl',
                             'split_file': 'generated_data/splits.json', 'split': 'train'},
                  'touch': {'source': 'full_surface'}}
        dataset = TouchDataset(config, oracle_point_frame=True)
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        self.assertEqual(tuple(sample['target_shape'].shape), (4096, 8))
        self.assertEqual(tuple(sample['touch_xyz'].shape), (256, 3))
        self.assertEqual(tuple(sample['pointmap'].shape), (64, 64, 3))
        self.assertTrue(torch.isfinite(sample['touch_xyz']).all())
        self.assertNotIn('touch_xyz', TouchDataset(config, include_touch=False)[0])
        # An existing dataset's saved pointmap must remain equivalent to depth loading.
        row = dict(dataset.records[0])
        saved = self.output / 'saved_pointmap.npy'
        np.save(saved, sample['pointmap'].numpy())
        row['pointmap_path'] = str(saved)
        np.testing.assert_array_equal(dataset.load_pointmap(row), sample['pointmap'].numpy())


if __name__ == '__main__':
    unittest.main()
