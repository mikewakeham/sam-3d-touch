import copy
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock

from organize_wandb import organize


def run(name="zeroverse_8views_image", **config):
    return SimpleNamespace(
        id="existing-id", name=name, group="", state="finished",
        config={"output_dir": "outputs/" + name,
                "data_config": "configs/data_zeroverse_5000_8views_full_surface.yaml",
                "no_pointmap": False, **config},
        summary={"val/loss": 0.32}, update=Mock(),
    )


class OrganizeWandbTests(unittest.TestCase):
    def organize(self, runs, **kwargs):
        with redirect_stdout(io.StringIO()) as output:
            organize(runs, **kwargs)
        return output.getvalue()

    def test_preview_apply_and_repeat(self):
        sample = run()
        config = copy.deepcopy(sample.config)
        self.organize([sample])
        sample.update.assert_not_called()
        self.assertEqual(sample.name, "zeroverse_8views_image")
        self.organize([sample], apply=True)
        self.assertEqual((sample.name, sample.group), ("zeroverse_pointmap", "zeroverse"))
        self.assertEqual(sample.id, "existing-id")
        self.assertEqual(sample.config, config)
        self.assertEqual(sample.summary, {"val/loss": 0.32})
        self.organize([sample], apply=True)
        sample.update.assert_called_once_with()

    def test_image_only_and_objaverse_names(self):
        image = run(no_pointmap=True)
        objaverse = run("custom-objaverse-name", data_config="configs/data1.yaml")
        self.organize([image, objaverse], apply=True)
        self.assertEqual(image.name, "zeroverse_image")
        self.assertEqual((objaverse.name, objaverse.group), ("custom-objaverse-name", "objaverse"))

    def test_active_and_unidentified_runs_are_untouched(self):
        active = run()
        active.state = "running"
        unknown = run(data_config="unknown.yaml")
        output = self.organize([active, unknown], apply=True)
        self.assertIn("SKIP active", output)
        self.assertIn("REVIEW", output)
        active.update.assert_not_called()
        unknown.update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
