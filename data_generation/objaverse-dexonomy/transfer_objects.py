import json
import shutil
from pathlib import Path


OLD_ROOT = Path(
    "/n/holylabs/qianqian_lab/Lab/mwakeham/"
    "visuotactile-objects/dexonomy-data"
)

NEW_ROOT = Path(
    "/n/holylabs/qianqian_lab/Lab/mwakeham/"
    "visuotactile-objects/sam-3d-touch-data/objaverse-dexonomy"
)

MANIFEST = OLD_ROOT / "output/objaverse_seed29/objects.jsonl"
OBJECTS_DIR = NEW_ROOT / "objects"

EXPECTED_OBJECTS = 1024


def main():
    with MANIFEST.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    object_ids = set()

    for row in rows:
        object_id = row["object_id"]

        if object_id in object_ids:
            raise ValueError(f"Duplicate object ID: {object_id}")

        source_obj = OLD_ROOT / "data" / row["source_visual_mesh_path"]
        source_dir = source_obj.parent

        required_files = [
            source_obj,
            source_dir / "material.mtl",
            source_dir / "material_0.png",
        ]

        for path in required_files:
            if not path.is_file():
                raise FileNotFoundError(path)

        object_ids.add(object_id)

    if len(object_ids) != EXPECTED_OBJECTS:
        raise ValueError(
            f"Found {len(object_ids)} objects; expected {EXPECTED_OBJECTS}"
        )

    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        object_id = row["object_id"]

        source_obj = OLD_ROOT / "data" / row["source_visual_mesh_path"]
        source_dir = source_obj.parent

        destination_dir = OBJECTS_DIR / object_id
        destination_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_obj, destination_dir / "model.obj")
        shutil.copy2(
            source_dir / "material.mtl",
            destination_dir / "material.mtl",
        )
        shutil.copy2(
            source_dir / "material_0.png",
            destination_dir / "material_0.png",
        )

        if index % 100 == 0 or index == len(rows):
            print(f"Copied {index}/{len(rows)} objects")

    for object_id in object_ids:
        object_dir = OBJECTS_DIR / object_id

        for filename in [
            "model.obj",
            "material.mtl",
            "material_0.png",
        ]:
            if not (object_dir / filename).is_file():
                raise FileNotFoundError(object_dir / filename)

    print()
    print("Transfer complete")
    print(f"Objects copied: {len(object_ids)}")
    print(f"Output: {OBJECTS_DIR}")


if __name__ == "__main__":
    main()