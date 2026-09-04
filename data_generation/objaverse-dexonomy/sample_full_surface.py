"""Export globally sampled surfaces for existing rendered samples; no rendering."""

import argparse
import json
from pathlib import Path

import numpy as np

from sample_touch import (
    DEFAULT_DATA_ROOT,
    classify_visibility,
    prepare_surface,
    transform_points,
)


def surface_settings(touch):
    if int(touch["format_version"]) != 4:
        raise ValueError("Expected existing format-4 touch data")
    return {
        "density": float(touch["density"]),
        "seed": int(touch["surface_seed_parts"][0]),
        "max_edge": float(touch["max_edge"]),
    }


def sample_full_surface(surface, point_ids, camera_path, depth_path, tolerance):
    with np.load(camera_path, allow_pickle=False) as camera:
        K = camera["K"]
        camera_transform = camera["T_camera_from_object"]
    depth = np.load(depth_path, allow_pickle=False)
    points = surface["points"][point_ids]
    # Same OpenCV -> SAM camera transform as sample_touch.pack_touch().
    T_sam_from_object = np.diag([-1.0, -1.0, 1.0, 1.0]) @ camera_transform
    points_camera = transform_points(points, T_sam_from_object).astype(np.float32)
    if not len(points_camera) or not np.isfinite(points_camera).all():
        raise ValueError(f"Empty or non-finite surface for {camera_path}")
    return {
        "points_camera": points_camera,
        "point_visibility": classify_visibility(
            points, K, camera_transform, depth, tolerance
        ),
        "point_ids": point_ids,
        "keep_priority": surface["keep_priority"][point_ids],
        "format_version": np.int64(1),
        "data_kind": "full_surface",
        "coordinate_frame": "sam_camera",
        "density": surface["density"],
        "max_edge": surface["max_edge"],
        "surface_point_count": len(surface["points"]),
        "surface_area": surface["mesh"].area,
        "surface_seed_parts": surface["seed_parts"],
        "visibility_tolerance": tolerance,
    }


def save_surface(path, arrays, overwrite):
    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[key], value) for key, value in arrays.items()
            ):
                raise ValueError(f"Existing surface differs: {path}; use --overwrite")
        return
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("generated_data/samples.jsonl"))
    parser.add_argument("--object-id")
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_points < 1:
        parser.error("--num-points must be positive")

    args.data_root = args.data_root.resolve()
    manifest_path = args.data_root / args.manifest
    suffix = f"_{args.object_id}" if args.object_id else ""
    output_manifest = manifest_path.with_name(f"samples_full_surface{suffix}.jsonl")
    if output_manifest.resolve() == manifest_path.resolve():
        raise ValueError("Input and output manifests must differ")
    with manifest_path.open() as file:
        records = [json.loads(line) for line in file if line.strip()]
    if args.object_id is not None:
        records = [record for record in records if record["object_id"] == args.object_id]
    if not records:
        raise ValueError("No matching samples in the manifest")
    objects = {}
    for record in records:
        objects.setdefault(record["object_id"], []).append(record)

    for object_index, (object_id, views) in enumerate(objects.items(), 1):
        first = views[0]
        with np.load(args.data_root / first["touch_path"], allow_pickle=False) as touch:
            settings = surface_settings(touch)
        surface = prepare_surface(
            args.data_root / "objects" / object_id / "model.obj",
            args.data_root / first["object_transform_path"],
            # These objects are already accepted; do not reapply the touch component limit.
            max_added_components=None,
            **settings,
        )
        # Same stable priority ranking as dataloader.load_touch(), but globally.
        point_ids = np.argsort(surface["keep_priority"], kind="stable")[:args.num_points].astype(np.int64)
        for record in views:
            if record["object_transform_path"] != first["object_transform_path"]:
                raise ValueError(f"Object transform differs between views: {object_id}")
            with np.load(args.data_root / record["touch_path"], allow_pickle=False) as touch:
                # Match sampling settings, not historical random points or face IDs.
                # Exported point_ids refer only to this newly generated pool.
                if (
                    surface_settings(touch) != settings
                    or not np.array_equal(touch["surface_seed_parts"], surface["seed_parts"])
                    or int(touch["surface_point_count"]) != len(surface["points"])
                    or not np.isclose(float(touch["surface_area"]), surface["mesh"].area)
                ):
                    raise ValueError(f"Surface settings, seed, count or area differ from {record['touch_path']}")
                tolerance = float(json.loads(touch["method_args"].item())["tolerance"])
            camera_path = args.data_root / record["camera_path"]
            arrays = sample_full_surface(
                surface, point_ids, camera_path,
                args.data_root / record["depth_path"], tolerance,
            )
            arrays["requested_point_count"] = args.num_points
            output_path = camera_path.with_name("full_surface.npz")
            save_surface(output_path, arrays, args.overwrite)
            record["full_surface_path"] = str(output_path.relative_to(args.data_root))
        print(
            f"[{object_index}/{len(objects)}] {object_id}: "
            f"{len(point_ids)} points/view, {len(views)} views", flush=True,
        )

    temporary = output_manifest.with_suffix(".tmp.jsonl")
    with temporary.open("w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    temporary.replace(output_manifest)
    print(f"Saved {output_manifest}: {len(records)} samples", flush=True)


if __name__ == "__main__":
    main()
