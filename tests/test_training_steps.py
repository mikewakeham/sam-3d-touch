"""Exercise the actual training-loop control flow without loading GPU models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import ast
import contextlib
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = ast.parse((Path(__file__).resolve().parent.parent / "train.py").read_text())


class TrainingStepsTests(unittest.TestCase):
    def parse(self, *flags):
        functions = [node for node in SOURCE.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("parse_args", "resolve_train_scope")]
        namespace = {"argparse": argparse, "Path": Path, "os": os}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "train.py", "exec"), namespace)
        with patch("sys.argv", ["train.py", "--pipeline-config", "pipeline.yaml", *flags]):
            return namespace["parse_args"]()

    def loop(self, max_steps, epochs, step=0, start_epoch=0):
        main = next(node for node in SOURCE.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        start = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "total_train_steps" for target in node.targets))
        # Run the production loop and checkpoint calls, replacing model work only.
        body = main.body[start:-2]
        args = SimpleNamespace(max_steps=max_steps, epochs=epochs, output_dir=Path("unused"),
                               cross_attention_scope="full", train_scope="shape_cross_attention")
        trained_epochs = []

        def train_epoch(pipeline, model, raw_model, loader, optimizer, parameters, device,
                        args, epoch, step, total_train_steps, *unused, **kwargs):
            trained_epochs.append(epoch)
            return min(step + len(loader), max_steps) if max_steps else step + len(loader)

        namespace = dict(args=args, step=step, start_epoch=start_epoch, train_loader=range(3),
                         pipeline=None, model=None, raw_model=None, optimizer=None, parameters=None,
                         device=None, world_size=1, distributed=False, main_process=True,
                         run=Mock(), seed=29, val_loader=None, rank=0, best_loss=float("inf"),
                         mode="image", train_epoch=train_epoch, validate=Mock(return_value=.3),
                         save_checkpoint=Mock())
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=body, type_ignores=[]), "train.py", "exec"), namespace)
        return namespace, trained_epochs

    def test_steps_override_epochs_and_clear_unused_config(self):
        args = self.parse("--max-steps", "20000", "--epochs", "1")
        self.assertEqual(args.max_steps, 20000)
        self.assertIsNone(args.epochs)
        self.assertEqual(self.parse().epochs, 20)

    def test_reaches_step_target_beyond_epoch_cap(self):
        state, epochs = self.loop(max_steps=20, epochs=1)
        self.assertEqual(state["step"], 20)
        self.assertEqual(state["total_train_steps"], 20)
        self.assertEqual(epochs, list(range(7)))
        last = state["save_checkpoint"].call_args
        self.assertEqual(last.args[3:5], (7, 20))

    def test_resume_past_old_epoch_limit(self):
        state, epochs = self.loop(max_steps=20, epochs=None, step=14, start_epoch=25)
        self.assertEqual(state["step"], 20)
        self.assertEqual(epochs, [25, 26])

    def test_completed_resume_does_not_take_extra_step(self):
        for step in (20, 21):
            state, epochs = self.loop(max_steps=20, epochs=None, step=step, start_epoch=25)
            self.assertEqual(state["step"], step)
            self.assertEqual(epochs, [])
            state["save_checkpoint"].assert_not_called()

    def test_epoch_only_jobs_keep_their_existing_limit(self):
        state, epochs = self.loop(max_steps=0, epochs=3, step=3, start_epoch=1)
        self.assertEqual(state["step"], 9)
        self.assertEqual(epochs, [1, 2])


if __name__ == "__main__":
    unittest.main()
