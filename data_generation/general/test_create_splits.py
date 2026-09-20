import json
from pathlib import Path
import tempfile
import unittest

from create_splits import create_splits


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.generated = self.root / "generated_data"
        self.generated.mkdir()

    def write_manifest(self, objects):
        rows = [{"object_id": obj, "view_id": view, "split": "test"}
                for obj in objects for view in range(8)]
        (self.generated / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    def test_counts_grouping_and_original_preserved(self):
        objects = [f"object-{index}" for index in range(5000)]
        self.write_manifest(objects)
        original = self.generated / "splits.json"
        original.write_text('{"original": true}\n')
        result = create_splits(self.root)
        self.assertEqual(len(result["train"]), 4750)
        self.assertEqual(len(result["val"]), 250)
        self.assertEqual(result["test"], [])
        self.assertFalse(set(result["train"]) & set(result["val"]))
        self.assertEqual(set(result["train"] + result["val"]), set(objects))
        self.assertEqual(original.read_text(), '{"original": true}\n')
        self.assertEqual(json.loads((self.generated / "splits_train_val.json").read_text()), result)
        (self.generated / "splits_train_val.json").unlink()
        self.write_manifest(reversed(objects))
        self.assertEqual(create_splits(self.root), result)

    def test_rerun_freezes_validation_and_adds_new_training_objects(self):
        self.write_manifest(["a", "b", "c", "d"])
        initial = create_splits(self.root, val_objects=2)
        self.assertEqual(create_splits(self.root, val_objects=2), initial)
        self.write_manifest(["a", "b", "c", "d", "new"])
        grown = create_splits(self.root, val_objects=2, seed=99)
        self.assertEqual(grown["val"], initial["val"])
        self.assertEqual(set(grown["train"]), set(initial["train"]) | {"new"})
        with self.assertRaises(ValueError):
            create_splits(self.root, val_objects=1)
        self.write_manifest(["a", "b", "c", "d"])
        with self.assertRaises(ValueError):
            create_splits(self.root, val_objects=2)

    def test_requires_nonempty_train_and_validation(self):
        self.write_manifest(["a", "b"])
        for count in [-1, 0, 2, 3]:
            with self.assertRaises(ValueError):
                create_splits(self.root, val_objects=count)
        self.assertFalse((self.generated / "splits_train_val.json").exists())


if __name__ == "__main__":
    unittest.main()
