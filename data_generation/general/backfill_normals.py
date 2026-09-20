"""Recover triangle normals for existing full surfaces, without rendering or encoding."""

import argparse
import fcntl
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh

from generate_target_latents import checkpoint_sha256, load_normalized_mesh
from sample_full_surface import (
    NORMAL_METHOD, classify_visibility, sam_camera_transform, save_surface,
    transform_normals, transform_points, validate_surface,
)
from surface_pool import DEFAULT_POOL_POINTS, make_surface_pool, save_surface_pool, validate_surface_pool


def backfill_object(object_dir, dry_run=False, check_only=False, pool_points=DEFAULT_POOL_POINTS):
    object_dir = Path(object_dir)
    mesh_path = object_dir / "mesh.npz"
    mesh = load_normalized_mesh(mesh_path)
    mesh_hash = checkpoint_sha256(mesh_path)
    records = json.loads((object_dir / "samples.json").read_text())
    view_ids = [record["view_id"] for record in records]
    if not view_ids or len(set(view_ids)) != len(view_ids):
        raise ValueError("Object must have a nonempty set of unique views")
    settings_path = object_dir.parent / "settings.json"
    if settings_path.exists():
        count = int(json.loads(settings_path.read_text())["num_views"])
        if set(view_ids) != {f"{index:03d}" for index in range(count)}:
            raise ValueError("Object views do not match the dataset settings")

    pending = []
    points = face_indices = normals = None
    first_seed = first_count = None
    max_error = 0.0
    for record in records:
        view_dir = object_dir / "views" / record["view_id"]
        path = view_dir / "full_surface.npz"
        with np.load(path, allow_pickle=False) as saved:
            arrays = dict(saved)
        validate_surface(arrays)
        count = int(arrays["requested_point_count"])
        seed = int(arrays["sample_seed"])
        if count != len(arrays["points_camera"]):
            raise ValueError(f"Point count mismatch: {path}")
        if str(arrays["sampling_method"]) != "area_weighted_direct":
            raise ValueError(f"Unsupported sampling method: {path}")
        if not np.array_equal(arrays["point_ids"], np.arange(count)):
            raise ValueError(f"Point ordering was changed: {path}")
        if points is None:
            first_seed, first_count = seed, count
            points, face_indices = trimesh.sample.sample_surface(mesh, count, seed=seed)
            normals = mesh.face_normals[face_indices]
        elif seed != first_seed or count != first_count:
            raise ValueError(f"Views must share one sampled object surface: {path}")

        transform = sam_camera_transform(view_dir / "camera.npz")
        replayed = transform_points(points, transform).astype(np.float32)
        error = float(np.abs(replayed - arrays["points_camera"]).max())
        max_error = max(max_error, error)
        # Coordinates are in a unit-scale object scene; allow only float32 roundoff.
        if error > 1e-6:
            raise ValueError(f"Sampling replay differs by {error:.6g}: {path}. "
                             "Use the original NumPy/Trimesh versions; no normals were assigned.")
        with np.load(view_dir / "camera.npz", allow_pickle=False) as camera:
            labels = classify_visibility(points, camera["K"], camera["T_camera_from_object"],
                                         np.load(view_dir / "depth.npy", allow_pickle=False),
                                         float(arrays["visibility_tolerance"]))
        if not np.array_equal(labels, arrays["point_visibility"]):
            raise ValueError(f"Saved visibility does not match the mesh/camera/depth: {path}")
        normals_camera = transform_normals(normals, transform)
        if int(arrays["format_version"]) == 2:
            if (str(arrays.get("mesh_sha256", "")) != mesh_hash
                    or not np.array_equal(arrays["face_indices"], face_indices)
                    or not np.allclose(arrays["normals_camera"], normals_camera, atol=1e-6, rtol=0)):
                raise ValueError(f"Saved normals or triangle mapping do not match the mesh: {path}")
            continue
        if check_only:
            raise ValueError(f"Normals have not been backfilled: {path}")
        arrays.update(format_version=np.int64(2), normals_camera=normals_camera,
                      face_indices=np.asarray(face_indices, dtype=np.int64), normal_method=NORMAL_METHOD,
                      mesh_sha256=mesh_hash, normal_numpy_version=np.__version__,
                      normal_trimesh_version=str(trimesh.__version__ or "unknown"))
        validate_surface(arrays, require_normals=True)
        pending.append((path, arrays))

    pool_path = object_dir / "surface_pool.npz"
    pool = None
    if pool_points:
        if pool_path.exists():
            with np.load(pool_path, allow_pickle=False) as saved:
                validate_surface_pool(saved, mesh, mesh_hash)
                if (len(saved["points_object"]) != pool_points
                        or int(saved["source_sample_seed"]) != first_seed):
                    raise ValueError("Existing pool size/seed differs; keep the same --pool-points on resume")
        elif check_only:
            raise ValueError(f"Missing surface pool: {pool_path}")
        else:
            pool = make_surface_pool(mesh, pool_points, first_seed, mesh_hash)

    # Validate every view and pool before replacing any. Mid-write interruption is resumable.
    if not dry_run:
        for path, arrays in pending:
            save_surface(path, arrays, overwrite=True)
        if pool is not None:
            save_surface_pool(pool_path, pool)
    return {"object_id": object_dir.name, "views": len(records), "updated": len(pending),
            "max_point_error": max_error, "pool_points": pool_points,
            "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
            "signed_volume": float(mesh.volume),
            "pool_created": pool is not None and not dry_run, "dry_run": dry_run}


def process_object(arguments):
    object_dir, dry_run, check_only, pool_points = arguments
    try:
        return backfill_object(object_dir, dry_run, check_only, pool_points)
    except Exception as error:
        return {"object_id": object_dir.name, "error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--object-id")
    parser.add_argument("--pool-points", type=int, default=DEFAULT_POOL_POINTS,
                        help="Additional shared point/normal pool; 0 only backfills existing points")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Verify replay without modifying surfaces")
    mode.add_argument("--check-only", action="store_true", help="Require and verify normals on every selected view")
    args = parser.parse_args()
    if args.workers < 1 or args.pool_points < 0 or (args.limit is not None and args.limit < 1):
        parser.error("Workers and limit must be positive; pool points must be nonnegative")
    generated = args.data_root.expanduser().resolve() / "generated_data"
    objects = json.loads((generated / "objects.json").read_text())
    if args.object_id:
        objects = [obj for obj in objects if obj["object_id"] == args.object_id]
        if not objects:
            parser.error("Unknown object ID")
    # samples.json marks packaged objects, including ones still awaiting target latents.
    directories = [generated / obj["object_id"] for obj in objects
                   if (generated / obj["object_id"] / "samples.json").is_file()]
    if args.limit is not None:
        directories = directories[:args.limit]
    if not directories:
        parser.error("No packaged objects found")
    print(f"NumPy {np.__version__}; Trimesh {trimesh.__version__}; {len(directories)} objects", flush=True)
    failures = 0
    # Share the builder's lock so packaging and backfill cannot overwrite each other.
    with (generated / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        arguments = [(path, args.dry_run, args.check_only, args.pool_points) for path in directories]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(process_object, arguments):
                print(json.dumps(result), flush=True)
                failures += "error" in result
    print(f"Checked {len(directories)} objects; failures: {failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
