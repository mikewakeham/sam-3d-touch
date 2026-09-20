import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def create_splits(data_root, val_objects=250, seed=29):
    generated = Path(data_root) / "generated_data"
    counts = Counter()
    with (generated / "samples.jsonl").open() as file:
        for line in file:
            if line.strip():
                counts[json.loads(line)["object_id"]] += 1

    if not 0 < val_objects < len(counts):
        raise ValueError("Validation count must be positive and leave at least one training object")

    path = generated / "splits_train_val.json"
    if path.exists():
        existing = json.loads(path.read_text())
        assigned = existing["train"] + existing["val"] + existing["test"]
        if len(set(assigned)) != len(assigned) or existing["test"]:
            raise ValueError("Existing train/val split has duplicate assignments or test objects")
        if set(assigned) - counts.keys():
            raise ValueError("Previously assigned objects are missing from the ready manifest")
        if len(existing["val"]) != val_objects:
            raise ValueError("Validation count differs from the existing split; keep it fixed across runs")
        # Preserve validation membership; newly completed objects go into training.
        validation = set(existing["val"])
    else:
        # Same seeded object hash as make_data.py, ranked for an exact count.
        ranked = sorted(counts, key=lambda object_id: (
            hashlib.sha256(f"{seed}:{object_id}".encode()).hexdigest(), object_id
        ))
        validation = set(ranked[:val_objects])

    splits = {
        "train": sorted(counts.keys() - validation),
        "val": sorted(validation),
        "test": [],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(splits, indent=2) + "\n")
    temporary.replace(path)

    print(f"Split file: {path}")
    for name, objects in splits.items():
        print(f"{name}: {len(objects)} objects, {sum(counts[obj] for obj in objects)} views")
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign ready objects to train/val without regenerating data.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--val-objects", type=int, default=250)
    parser.add_argument("--seed", type=int, default=29, help="Seed for initial selection; reruns preserve validation objects")
    args = parser.parse_args()
    create_splits(args.data_root, args.val_objects, args.seed)
