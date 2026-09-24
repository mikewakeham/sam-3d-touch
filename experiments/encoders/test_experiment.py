import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from experiments.encoders import prepare_data, compare, run


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / 'experiment'
        self.output.mkdir()
        self.patch = patch.object(prepare_data, 'HERE', self.output)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        os.environ['MPLCONFIGDIR'] = str(self.output / '.matplotlib')
        os.environ['XDG_CACHE_HOME'] = str(self.output / '.cache')
        self.records = [{'object_id': f'object{i}', 'sample_id': f'object{i}/view{v}', 'view_id': v,
                         'image_path': f'object{i}/view{v}.png'} for i in range(7) for v in range(8)]
        (self.root / 'samples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.records))
        (self.root / 'splits.json').write_text(json.dumps({'train': [f'object{i}' for i in range(4)],
                                                        'val': ['object4', 'object5'], 'test': ['object6']}))
        self.config = self.root / 'source.yaml'
        self.config.write_text(yaml.safe_dump({'dataset': {'root': str(self.root), 'manifest': 'samples.jsonl',
                                                           'split_file': 'splits.json'}}))

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_data.prepare(self.config, self.output / 'data', 3, 2, 2)

    def test_fixed_subsets_loader_and_source_preservation(self):
        from dataloader import TouchDataset
        before = {p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        self.prepare()
        data_dir = self.output / 'data'
        contents = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in data_dir.iterdir()}
        self.prepare()
        self.assertEqual(contents, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in data_dir.iterdir()})
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()})
        data = yaml.safe_load((data_dir / 'small.yaml').read_text())
        train = TouchDataset(data)
        data['dataset']['split'] = 'val'
        val = TouchDataset(data)
        self.assertEqual((len(train), len(val)), (24, 16))
        self.assertFalse({r['object_id'] for r in train.records} & {r['object_id'] for r in val.records})
        data = yaml.safe_load((data_dir / 'overfit.yaml').read_text())
        train = TouchDataset(data)
        data['dataset']['split'] = 'val'
        val = TouchDataset(data)
        self.assertEqual(train.records, val.records)
        self.assertEqual(len(train), 2)
        self.assertEqual(len({r['object_id'] for r in train.records}), 2)

    def test_rejects_changed_selection_leakage_and_output_escape(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            prepare_data.prepare(self.config, self.output / 'data', 4, 2, 2)
        splits = json.loads((self.root / 'splits.json').read_text())
        splits['val'].append(splits['train'][0])
        (self.root / 'splits.json').write_text(json.dumps(splits))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.prepare()
        with self.assertRaisesRegex(ValueError, 'inside'):
            prepare_data.experiment_path(self.root / 'outside')

    def test_shuffle_keeps_xyz_normals_mask_paired_and_visuals_unchanged(self):
        original = {key: torch.ones(1, 5, 3) for key in ['touch_xyz', 'touch_normals', 'touch_mask', 'image', 'pointmap', 'target_shape']}
        donor = {key: value * 2 for key, value in original.items()}
        shuffled = compare.replace_surface(original, donor)
        for key in ['touch_xyz', 'touch_normals', 'touch_mask']:
            self.assertIs(shuffled[key], donor[key])
        for key in ['image', 'pointmap', 'target_shape']:
            self.assertIs(shuffled[key], original[key])

    def test_plotting_artifacts(self):
        target = np.zeros((12, 12, 12), dtype=bool)
        target[3:8, 3:8, 3:8] = True
        compare.plot_voxels(self.output / 'preview.png', np.full((32, 32, 4), 255, np.uint8),
                            target, {'matched': target, 'shuffled': np.roll(target, 3, axis=0)})
        (self.output / 'metrics.jsonl').write_text(json.dumps({'global_step': 0, 'loss/val': 1}) + '\n' +
                                                json.dumps({'global_step': 10, 'loss/train': .5, 'loss/val': .7}) + '\n')
        compare.plot_losses([self.output], self.output)
        from PIL import Image
        for name in ['preview.png', 'losses.png']:
            with Image.open(self.output / name) as image:
                self.assertGreater(image.width, 500)
                self.assertGreater(np.asarray(image).std(), 1)

    def test_training_wrapper_initial_checkpoint_and_local_logging(self):
        import train
        import wandb
        arguments = ['run.py', '--data-config', str(self.output / 'data/small.yaml'),
                     '--pipeline-config', 'unused.yaml', '--output-dir', str(self.output / 'run'),
                     '--point-encoder', 'craftsman', '--train-scope', 'shape_cross_attention']
        self.prepare()
        logged = []
        fake_run = SimpleNamespace(log=lambda row, **kwargs: logged.append(row))
        saved = []
        def fake_main():
            args = train.parse_args()
            logger = wandb.init()
            result = train.train_epoch(None, None, None, None, None, None, torch.device('cpu'), args,
                                       0, 0, 10, 1, False, True, logger, seed=29)
            self.assertEqual(result, 10)
            logger.log({'global_step': 10, 'loss/train': .2})
        with patch.object(sys, 'argv', arguments), patch.object(train, 'main', fake_main), \
             patch.object(train, 'validate', return_value=.6), patch.object(train, 'build_dataloader', return_value=[]), \
             patch.object(train, 'save_checkpoint', side_effect=lambda *args: saved.append(args)), \
             patch.object(train, 'train_epoch', return_value=10), patch.object(wandb, 'init', return_value=fake_run), \
             patch.dict(os.environ):
            run.main()
        self.assertEqual(saved[0][0], self.output / 'run/initial.pt')
        self.assertEqual(saved[0][6], 'image_touch')
        self.assertEqual(logged[0], {'global_step': 0, 'loss/val': .6})
        local = [json.loads(line) for line in (self.output / 'run/metrics.jsonl').read_text().splitlines()]
        self.assertEqual(local, logged)

    def test_job_variants_flags_steps_and_paths(self):
        job = Path(__file__).with_name('jobs') / 'train.sh'
        text = job.read_text()
        # Execute the actual shell flag selection while replacing only cluster paths/launcher.
        capture = self.output / 'capture.py'
        capture.write_text('import json, sys; print(json.dumps(sys.argv[1:]))\n')
        launcher = '/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/torchrun'
        text = '\n'.join(line for line in text.splitlines() if not line.startswith('cd '))
        text = text.replace(launcher, f'{sys.executable} {capture}')
        script = self.output / 'train.sh'
        script.write_text(text)
        for variant in ['image', *[f'{name}_{mode}' for name in ['vecsetx', 'craftsman', 'triposg'] for mode in ['frozen', 'scratch']]]:
            for stage, steps in [('overfit', '200'), ('small', '1024')]:
                gpus = '1' if stage == 'overfit' else '4'
                result = subprocess.run(['bash', str(script), variant, stage], check=True, capture_output=True, text=True,
                                        env={**os.environ, 'SLURM_GPUS_ON_NODE': gpus})
                flags = json.loads(result.stdout)
                self.assertIn(f'--nproc_per_node={gpus}', flags)
                self.assertEqual(flags[flags.index('--max-steps') + 1], steps)
                self.assertIn(f'experiments/encoders/outputs/{stage}_{variant}', flags)
                self.assertEqual('--point-encoder-from-scratch' in flags, variant.endswith('_scratch'))
                self.assertEqual('--no-touch' in flags, variant == 'image')
        result = subprocess.run(['bash', str(script), 'unknown', 'small'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_compare_cpu_control_flow_paired_noise_and_outputs(self):
        # Exercise the comparison driver without pretending to run unavailable SAM3D weights.
        import csv
        import dataloader
        from evaluation import evaluate
        import train
        class Dataset(torch.utils.data.Dataset):
            def __init__(self):
                self.records = [row for row in self_records if row['view_id'] == 0]
            def __len__(self):
                return len(self.records)
            def __getitem__(self, index):
                row = self.records[index]
                value = int(row['object_id'][-1]) + 1
                return {'sample_id': row['sample_id'], 'image': torch.full((8, 8, 4), 255, dtype=torch.uint8),
                        'pointmap': torch.ones(3, 8, 8), 'target_shape': torch.ones(4096, 8),
                        'touch_xyz': torch.full((8, 3), float(value)),
                        'touch_normals': torch.tensor([[1., 0., 0.]]).repeat(8, 1)}
        class Decoder(torch.nn.Module):
            reshape_input_to_cube = True
            def forward(self, latent):
                return latent.mean(dim=-1).reshape(-1, 1, 16, 16, 16)
        draws = []
        class Model(torch.nn.Module):
            conditioning_config = {'oracle_point_frame': False}
            touch_encoder = SimpleNamespace(requires_normals=True)
            def get_touch_tokens(self, points, mask):
                return points.mean(dim=1, keepdim=True)
            def forward(self, targets, cond_args, cond_kwargs, points, mask):
                noise = torch.rand(())
                draws.append(float(noise))
                return points[..., :3].square().mean() + noise
        def prepare(pipeline, batch, *args, **kwargs):
            return ({'shape': batch['target_shape']}, (), {},
                    torch.cat([batch['touch_xyz'], batch['touch_normals']], dim=-1), batch['touch_mask'])
        self_records = self.records[:16]
        checkpoint = {'step': 10, 'mode': 'image_touch', 'touch_config': {'encoder_name': 'craftsman'}}
        config = self.output / 'pipeline.yaml'
        config.write_text('ss_decoder_config_path: unused\nss_decoder_ckpt_path: unused\n')
        checkpoint_path = self.output / 'run/best.pt'
        checkpoint_path.parent.mkdir()
        args = ['compare.py', '--checkpoints', str(checkpoint_path), '--pipeline-config', str(config),
                '--output-dir', str(self.output / 'comparison'), '--device', 'cpu', '--objects', '2']
        def loader(*args, **kwargs):
            return torch.utils.data.DataLoader(Dataset(), batch_size=1, collate_fn=dataloader.collate_touch_batch)
        with patch.object(sys, 'argv', args), \
             patch.object(evaluate, 'read_run', return_value=(checkpoint, {}, {'touch': {'source': 'full_surface'}})), \
             patch.object(dataloader, 'build_dataloader', side_effect=loader), \
             patch.object(train, 'build_stage1_pipeline', return_value=SimpleNamespace(
                 ss_generator=SimpleNamespace(random_generator=torch.Generator()), init_ss_decoder=lambda *args: Decoder())), \
             patch.object(evaluate, 'restore_run', return_value=Model()), \
             patch.object(train, 'prepare_batch', side_effect=prepare), \
             patch.object(evaluate, 'sample_shape', side_effect=lambda *args: torch.randn(1, 4096, 8)):
            compare.main()
        self.assertEqual(draws[::2], draws[1::2])
        with (self.output / 'comparison/metrics.csv').open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 4)
        for row in rows:
            if row['treatment'] == 'shuffled':
                self.assertNotEqual(row['sample_id'], row['donor_sample_id'])
        for path in (self.output / 'comparison/run_best').glob('*.npz'):
            with np.load(path) as arrays:
                np.testing.assert_array_equal(arrays['matched'], arrays['shuffled'])
                self.assertEqual(arrays['target'].shape, (16, 16, 16))
        self.assertTrue((self.output / 'comparison/summary.json').is_file())
        self.assertEqual(len(list((self.output / 'comparison/run_best').glob('*.png'))), 2)


if __name__ == '__main__':
    unittest.main()
