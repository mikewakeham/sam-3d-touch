"""CPU checks for paired exposure, split integrity and control semantics."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from protocol import arm_config, check_splits, epoch_schedule, evaluation_groups


HERE = Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def test_schedule_coverage_balance_and_resume(self):
        for epoch in range(20):
            batches, drops = epoch_schedule(1003, 4, 29, epoch)
            self.assertEqual(sorted(i for b in batches for i in b), list(range(1003)))
            self.assertEqual(sum(drops), len(batches) // 2)
            self.assertEqual((batches, drops), epoch_schedule(1003, 4, 29, epoch))
        self.assertNotEqual(epoch_schedule(1003, 4, 29, 0), epoch_schedule(1003, 4, 29, 1))

    def test_controls(self):
        oracle, constant, camera = (arm_config(a) for a in ('oracle_dropout', 'constant_dropout', 'camera_dropout'))
        self.assertEqual({k: v for k, v in oracle.items() if k != 'constant'},
                         {k: v for k, v in constant.items() if k != 'constant'})
        self.assertEqual({k: v for k, v in oracle.items() if k != 'oracle'},
                         {k: v for k, v in camera.items() if k != 'oracle'})
        self.assertEqual(arm_config('image')['visual_dropout'], 0)

    def test_selection_and_leakage(self):
        records = [{'object_id': str(i), 'sample_id': f'{i}_{v:03}'} for i in range(40) for v in range(16)]
        groups = evaluation_groups(records, 29)
        self.assertEqual(len(groups), 16)
        self.assertEqual(len({r['object_id'] for g in groups for r in g}), 32)
        self.assertTrue(all(len({r['object_id'] for r in g}) == 4 for g in groups))
        with self.assertRaises(ValueError):
            check_splits(records, records[:4])
        self.assertEqual(check_splits(records[:16], records[16:])['train_objects'], 1)

    def test_launcher_allocation(self):
        for count in (1, 2, 4):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-a,GPU-b,GPU-c,GPU-d')
            run = subprocess.run([sys.executable, str(HERE / 'launch.py'), '--gpus', str(count),
                                  '--preflight-report', 'unused.json', '--output-dir', '/tmp/unused-upper-bound',
                                  '--dry-run'], env=env, text=True, capture_output=True, check=True)
            result = json.loads(run.stdout)
            self.assertEqual(len(result['gpus']), count)
            self.assertEqual(len(result['waves']), 4 // count)
            for wave in result['waves']:
                for arm, command in wave:
                    self.assertEqual(command[command.index('--batch-size') + 1], '4')

    def test_no_historical_source_mutation_and_no_stage2(self):
        # Historical-source parity is enforced on GPU. Here guard the new runner's
        # boundary: only the Stage-1 builder, with no Stage-2 entry point.
        for name in ('run_gpu.py', 'model.py', 'assess.py'):
            tree = ast.parse((HERE / name).read_text())
            attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            self.assertFalse(attributes & {'init_slat_decoder', 'init_slat_generator', 'slat_generator'})


if __name__ == '__main__':
    unittest.main()
