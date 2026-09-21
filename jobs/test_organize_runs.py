import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from organize_runs import check_moves, plan_moves, run_destination


class OrganizeRunsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        (self.repo / "train.py").touch()

    def run_folder(self, name, no_pointmap=False, data_config="data_zeroverse_5000_8views_full_surface.yaml"):
        folder = self.repo / "outputs" / name
        folder.mkdir(parents=True)
        (folder / "config.yaml").write_text(yaml.safe_dump({"arguments": {
            "data_config": "configs/" + data_config,
            "no_pointmap": no_pointmap,
            "cross_attention_learning_rate": 1e-5,
        }}))
        (folder / "last.pt").write_bytes(b"checkpoint fixture; preserve exactly")
        return folder

    def command(self, *args):
        return subprocess.run([sys.executable, str(Path(__file__).with_name("organize_runs.py")), *args],
                              cwd=self.repo, capture_output=True, text=True)

    def test_preview_apply_preserves_contents_and_is_repeatable(self):
        source = self.run_folder("zeroverse_8views_image_surface_frozen_lr3e5")
        contents = (source / "config.yaml").read_bytes()
        (source / "wandb" / "run-example").mkdir(parents=True)
        (source / "wandb" / "latest-run").symlink_to("run-example", target_is_directory=True)
        destination = self.repo / "outputs/zeroverse/zeroverse_pointmap_surface_frozen_lr3e5"
        self.assertEqual(self.command().returncode, 0)
        self.assertTrue(source.exists())
        self.assertFalse(destination.exists())
        result = self.command("--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(source.exists())
        self.assertEqual((destination / "config.yaml").read_bytes(), contents)
        self.assertEqual((destination / "last.pt").read_bytes(), b"checkpoint fixture; preserve exactly")
        self.assertTrue((destination / "wandb/latest-run").is_dir())
        self.assertEqual(plan_moves(self.repo), ([], []))
        self.assertEqual(self.command("--apply").returncode, 0)
        self.assertTrue((self.repo / "logs/objaverse").is_dir())

    def test_intermediate_names_and_true_image_only(self):
        examples = [
            ("zeroverse/zeroverse_8views_image", True, "zeroverse_image"),
            ("zeroverse_8views/pointmap_surface_frozen_lr1e5", False, "zeroverse_pointmap_surface_frozen"),
            ("zeroverse/zeroverse_8views_craftsman_frozen", False, "zeroverse_pointmap_craftsman_frozen"),
            ("zeroverse_8views_image", False, "zeroverse_pointmap"),
        ]
        for name, no_pointmap, expected in examples:
            with self.subTest(name=name):
                source = self.run_folder(name, no_pointmap)
                config = yaml.safe_load((source / "config.yaml").read_text())
                self.assertEqual(run_destination(source, config, self.repo / "outputs").name, expected)

    def test_collision_aborts_before_any_moves(self):
        source = self.run_folder("zeroverse_8views_image")
        other = self.run_folder("stage1_image_full_cross_attention", data_config="data1.yaml")
        self.run_folder("zeroverse/zeroverse_pointmap")
        result = self.command("--apply")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to overwrite", result.stderr)
        self.assertTrue(source.exists())
        self.assertTrue(other.exists())

    def test_two_sources_to_one_destination(self):
        self.run_folder("zeroverse_8views_image")
        self.run_folder("zeroverse_8views_pointmap")
        with self.assertRaises(FileExistsError):
            check_moves(plan_moves(self.repo)[0])

    def test_objaverse_nested_outputs_logs_and_unknowns(self):
        self.run_folder("stage1_image_full_cross_attention", data_config="data1.yaml")
        nested = self.repo / "outputs/conditioning_investigation/trial"
        nested.mkdir(parents=True)
        (nested / "result.json").write_text("{}")
        unknown = self.run_folder("unrelated", data_config="unknown.yaml")
        logs = self.repo / "logs"
        logs.mkdir()
        for name in ("zv_image-123.out", "s1_image_full_ca-456.err", "unrelated.log"):
            (logs / name).write_text("keep")
        result = self.command("--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.repo / "outputs/objaverse/stage1_image_full_cross_attention/last.pt").is_file())
        self.assertTrue((self.repo / "outputs/objaverse/conditioning_investigation/trial/result.json").is_file())
        self.assertTrue(unknown.exists())
        self.assertTrue((logs / "zeroverse/zv_image-123.out").exists())
        self.assertTrue((logs / "objaverse/s1_image_full_ca-456.err").exists())
        self.assertTrue((logs / "unrelated.log").exists())

    def test_missing_condition_flag_is_not_guessed(self):
        config = {"arguments": {"data_config": "configs/data_zeroverse_5000_8views_full_surface.yaml"}}
        self.assertIsNone(run_destination(Path("zeroverse_image"), config, self.repo / "outputs"))


if __name__ == "__main__":
    unittest.main()
