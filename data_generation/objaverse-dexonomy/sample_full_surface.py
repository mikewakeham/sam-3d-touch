"""Sample mesh surfaces directly for existing rendered samples; no rendering."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from generate_target_latents import load_normalized_mesh
from sample_touch import (
    DEFAULT_DATA_ROOT,
    classify_visibility,
    transform_points,
)


def sample_full_surface(points, camera_path, depth_path, tolerance):
    with np.load(camera_path, allow_pickle=False) as camera:
        K = camera["K"]
        camera_transform = camera["T_camera_from_object"]
    depth = np.load(depth_path, allow_pickle=False)
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
        # IDs identify this object's new sample, not the historical touch pool.
        "point_ids": np.arange(len(points), dtype=np.int64),
        "format_version": np.int64(1),
        "data_kind": "full_surface",
        "coordinate_frame": "sam_camera",
        "sampling_method": "area_weighted_direct",
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
            seed_parts = touch["surface_seed_parts"].astype(np.int64)
        mesh = load_normalized_mesh(
            args.data_root / "objects" / object_id / "model.obj",
            args.data_root / first["object_transform_path"],
        )
        if not np.isfinite(mesh.area) or mesh.area <= 0:
            raise ValueError(f"Mesh must have positive finite surface area: {object_id}")
        sample_seed = int(np.random.default_rng(seed_parts).integers(2**31))
        points, _ = trimesh.sample.sample_surface(mesh, args.num_points, seed=sample_seed)
        for record in views:
            if record["object_transform_path"] != first["object_transform_path"]:
                raise ValueError(f"Object transform differs between views: {object_id}")
            with np.load(args.data_root / record["touch_path"], allow_pickle=False) as touch:
                if int(touch["format_version"]) != 4:
                    raise ValueError("Expected existing format-4 touch data")
                if not np.array_equal(touch["surface_seed_parts"], seed_parts):
                    raise ValueError(f"Object seed differs between views: {object_id}")
                tolerance = float(json.loads(touch["method_args"].item())["tolerance"])
            camera_path = args.data_root / record["camera_path"]
            arrays = sample_full_surface(
                points, camera_path,
                args.data_root / record["depth_path"], tolerance,
            )
            arrays["requested_point_count"] = args.num_points
            arrays["sample_seed_parts"] = seed_parts
            arrays["sample_seed"] = sample_seed
            arrays["surface_area"] = mesh.area
            output_path = camera_path.with_name("full_surface.npz")
            save_surface(output_path, arrays, args.overwrite)
            record["full_surface_path"] = str(output_path.relative_to(args.data_root))
        print(
            f"[{object_index}/{len(objects)}] {object_id}: "
            f"{len(points)} points/view, {len(views)} views", flush=True,
        )

    temporary = output_manifest.with_suffix(".tmp.jsonl")
    with temporary.open("w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    temporary.replace(output_manifest)
    print(f"Saved {output_manifest}: {len(records)} samples", flush=True)


if __name__ == "__main__":
    main()
