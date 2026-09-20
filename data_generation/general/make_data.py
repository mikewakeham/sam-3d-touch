"""Build SAM 3D training data from local object files. Run --help for stage options."""

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from generate_target_latents import (
    DEFAULT_ENCODER_CHECKPOINT, encode_objects,
    load_normalized_mesh, save_metadata, validate_target, checkpoint_sha256,
)
from sample_full_surface import sample_full_surface, save_surface, validate_surface
from surface_pool import DEFAULT_POOL_POINTS, make_surface_pool, save_surface_pool, validate_surface_pool


FORMATS = {".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl", ".blend",
           ".usd", ".usda", ".usdc", ".usdz", ".dae", ".abc"}
SETTING_NAMES = ("num_views", "resolution", "seed", "camera_radius", "fov_degrees",
                 "samples", "frame", "rotation", "num_points", "visibility_tolerance",
                 "train_fraction", "val_fraction", "render_device", "camera_mode", "lighting", "ready_marker",
                 "surface_pool_points")


def pipeline_sha256():
    digest = hashlib.sha256()
    for name in ("make_data.py", "render_blender.py", "sample_full_surface.py", "surface_pool.py", "generate_target_latents.py", "pointmaps.py", "sparse_structure_vae.py", "vae_utils.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-root", type=Path, help="Input directory (or one mesh file)")
    parser.add_argument("--pattern", default="**/*", help="Relative glob, e.g. '**/*.glb' for Zeroverse")
    parser.add_argument("--ready-marker", help="Only use objects whose adjacent JSON marker has status=complete, e.g. '{stem}_generation.json'")
    parser.add_argument("--data-root", type=Path, required=True, help="Output dataset root")
    parser.add_argument("--stage", choices=["all", "render", "latents"], default="all",
                        help="render includes depth and full surfaces; all also encodes targets")
    parser.add_argument("--limit", type=int, help="Process only the first N objects; omit to resume the full dataset")
    parser.add_argument("--object-id", help="Restrict processing to an ID from objects.json")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--render-device", choices=["CUDA", "OPTIX", "CPU"], default="CUDA")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--render-workers-per-gpu", type=int, default=1)
    parser.add_argument("--blender-threads", type=int, default=4)
    parser.add_argument("--num-views", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--camera-mode", choices=["trellis2", "fixed"], default="trellis2",
                        help="Shared seeded FOV/distance variation or fixed FOV/distance")
    parser.add_argument("--lighting", choices=["trellis2", "fixed"], default="trellis2",
                        help="Shared seeded TRELLIS.2 lighting or fixed three-light scene")
    parser.add_argument("--seed", type=int, default=29)
    # TRELLIS frames the unit cube using sqrt(3)/2 / sin(FOV/2); 2.7 adds margin at 40 degrees.
    parser.add_argument("--camera-radius", type=float, default=2.7)
    parser.add_argument("--fov-degrees", type=float, default=40.0)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--rotation", type=float, nargs=3, default=[0, 0, 0],
                        help="XYZ degrees applied after Blender's format-aware import")
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--surface-pool-points", type=int, default=DEFAULT_POOL_POINTS,
                        help="Shared XYZ/normal pool per object; 0 disables the optional larger pool")
    parser.add_argument("--visibility-tolerance", type=float, default=0.005)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_ENCODER_CHECKPOINT)
    parser.add_argument("--dry-run", action="store_true", help="Discover inputs and print settings without generating data")
    args = parser.parse_args(argv)
    if args.surface_pool_points < 0:
        parser.error("--surface-pool-points must be nonnegative")
    if min(args.workers, args.render_workers_per_gpu, args.blender_threads,
           args.num_views, args.resolution, args.samples, args.num_points) < 1:
        parser.error("Worker, view, resolution, sample and point counts must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if not 0 <= args.train_fraction <= 1 or not 0 <= args.val_fraction <= 1 or args.train_fraction + args.val_fraction > 1:
        parser.error("Split fractions must be nonnegative and sum to at most one")
    if not np.isfinite([args.camera_radius, args.fov_degrees, args.visibility_tolerance, *args.rotation]).all():
        parser.error("Camera, tolerance and rotation values must be finite")
    if not 0.97 < args.camera_radius < 99 or not 0 < args.fov_degrees < 180 or args.visibility_tolerance < 0:
        parser.error("Use 0.97 < radius < 99, 0 < FOV < 180 and nonnegative visibility tolerance")
    args.data_root = args.data_root.expanduser().resolve()
    args.encoder_checkpoint = args.encoder_checkpoint.expanduser().resolve()
    return args


def get_objects(objects_root, pattern="**/*", ready_marker=None):
    objects_root = objects_root.expanduser().resolve()
    if objects_root.is_file():
        paths, root = [objects_root], objects_root.parent
    elif objects_root.is_dir():
        paths = sorted(path for path in objects_root.glob(pattern) if path.is_file() and path.suffix.lower() in FORMATS)
        root = objects_root
    else:
        raise FileNotFoundError(objects_root)
    if not paths:
        raise ValueError(f"No supported object files under {objects_root} matching {pattern!r}")
    objects = []
    for path in paths:
        if ready_marker:
            marker = path.parent / ready_marker.format(stem=path.stem, name=path.name)
            try:
                if json.loads(marker.read_text()).get("status") != "complete":
                    continue
            except (OSError, ValueError):
                continue
        if path.suffix.lower() not in FORMATS:
            raise ValueError(f"Unsupported object format: {path}")
        relative = path.relative_to(root).as_posix()
        # Stable across root moves, input subsets and discovery order; no hex-ID assumption.
        digest = hashlib.sha256(relative.encode()).hexdigest()[:12]
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")[:64] or "object"
        objects.append({"object_id": f"{name}-{digest}", "model_path": str(path),
                        "source_relative_path": relative,
                        "source_size": path.stat().st_size, "source_mtime_ns": path.stat().st_mtime_ns})
    if not objects:
        raise ValueError("No completed objects found; rerun after generation finishes an object")
    if len({obj["object_id"] for obj in objects}) != len(objects):
        raise ValueError("Duplicate object IDs")
    return objects


def save_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def make_splits(objects, seed, train_fraction, val_fraction, existing=None):
    # Keep published assignments; new objects are assigned independently of arrival order.
    splits = {name: list((existing or {}).get(name, [])) for name in ("train", "val", "test")}
    assigned = {object_id for ids in splits.values() for object_id in ids}
    for obj in objects:
        object_id = obj["object_id"]
        if object_id in assigned:
            continue
        digest = hashlib.sha256(f"{seed}:{object_id}".encode()).hexdigest()
        value = int(digest[:16], 16) / 2**64
        split = "train" if value < train_fraction else "val" if value < train_fraction + val_fraction else "test"
        splits[split].append(object_id)
    return {name: sorted(ids) for name, ids in splits.items()}


def check_existing_objects(objects, previous, generated):
    current = {obj["object_id"]: obj for obj in objects}
    for old in previous:
        obj = current.get(old["object_id"])
        if obj is None:
            raise ValueError(f"Previously registered object is missing or no longer ready: {old['model_path']}")
        if obj != old and (generated / old["object_id"] / "render_complete.json").is_file():
            raise ValueError(f"Previously rendered input changed: {old['model_path']}; use a new --data-root")
        # Failed/incomplete imports may have been captured while their source was being written.


def format_time(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_object(obj, output_dir, args, gpu_id):
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    command = [
        args.blender, "--background", "--factory-startup", "--disable-autoexec",
        "--threads", str(args.blender_threads), "--python-exit-code", "1",
        "--python", str(Path(__file__).with_name("render_blender.py")), "--",
        "--object", obj["model_path"], "--output", str(output_dir),
        "--num-views", str(args.num_views), "--resolution", str(args.resolution),
        "--seed", str(args.seed), "--camera-radius", str(args.camera_radius),
        "--fov-degrees", str(args.fov_degrees), "--samples", str(args.samples),
        "--device", args.render_device, "--frame", str(args.frame),
        "--rotation", *map(str, args.rotation),
        "--camera-mode", args.camera_mode, "--lighting", args.lighting,
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "blender.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"Blender exited {result.returncode}; see {log_path}")


def render_complete(object_dir, num_views):
    if not (object_dir / "render_complete.json").is_file():
        return False
    files = [object_dir / name for name in ("mesh.npz", "object_transform.npz", "render_metadata.json")]
    files += [object_dir / "views" / f"{i:03d}" / name
              for i in range(num_views) for name in ("image.png", "depth.npy", "camera.npz")]
    if not all(path.is_file() and path.stat().st_size > 0 for path in files):
        return False
    metadata = json.loads((object_dir / "render_metadata.json").read_text())
    for asset in metadata.get("external_assets", []):
        path = Path(asset["path"])
        if not path.is_file() or path.stat().st_size != asset["size"] or path.stat().st_mtime_ns != asset["mtime_ns"]:
            raise ValueError(f"External asset changed or is missing: {path}; use a new --data-root")
    return True


def package_object(obj, split, args):
    object_dir = args.data_root / "generated_data" / obj["object_id"]
    mesh = load_normalized_mesh(object_dir / "mesh.npz")
    seed_parts = [args.seed, int(hashlib.sha256(obj["object_id"].encode()).hexdigest()[:8], 16)]
    sample_seed = int(np.random.default_rng(seed_parts).integers(2**31))
    points, face_indices = trimesh.sample.sample_surface(mesh, args.num_points, seed=sample_seed)
    normals = mesh.face_normals[face_indices]
    mesh_hash = checkpoint_sha256(object_dir / "mesh.npz")
    if args.surface_pool_points:
        pool = make_surface_pool(mesh, args.surface_pool_points, sample_seed, mesh_hash)
        save_surface_pool(object_dir / "surface_pool.npz", pool)
    records = []
    for index in range(args.num_views):
        view_id = f"{index:03d}"
        view_dir = object_dir / "views" / view_id
        depth = np.load(view_dir / "depth.npy", allow_pickle=False)
        if depth.shape != (args.resolution, args.resolution):
            raise ValueError(f"Unexpected depth shape in {view_dir}: {depth.shape}")
        with Image.open(view_dir / "image.png") as image:
            rgba = np.asarray(image.convert("RGBA"))
        if rgba.shape != (*depth.shape, 4) or not np.any(rgba[..., 3]):
            raise ValueError(f"Empty or mismatched render: {view_dir}")
        alpha = rgba[..., 3]
        if any(np.any(edge > 0) for edge in (alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1])):
            raise ValueError(f"Object touches image boundary: {view_dir}; use a larger camera radius in a new dataset")
        # Ignore transparent background while retaining NaNs expected by SAM 3D.
        depth[rgba[..., 3] == 0] = np.nan
        if not np.any(np.isfinite(depth) & (depth > 0)):
            raise ValueError(f"No finite foreground depth: {view_dir}")
        np.save(view_dir / "depth.npy", depth)
        arrays = sample_full_surface(points, view_dir / "camera.npz", view_dir / "depth.npy",
                                     args.visibility_tolerance, normals, face_indices)
        arrays.update(requested_point_count=args.num_points, sample_seed_parts=seed_parts,
                      sample_seed=sample_seed, surface_area=mesh.area, mesh_sha256=mesh_hash,
                      normal_numpy_version=np.__version__, normal_trimesh_version=str(trimesh.__version__ or "unknown"))
        save_surface(view_dir / "full_surface.npz", arrays, overwrite=True)
        record = {"sample_id": f"{obj['object_id']}_{view_id}", "object_id": obj["object_id"],
                  "view_id": view_id, "split": split, "touch_path": None, "target_path": None}
        for key, path in {
            "image_path": view_dir / "image.png", "depth_path": view_dir / "depth.npy",
            "camera_path": view_dir / "camera.npz",
            "full_surface_path": view_dir / "full_surface.npz",
            "object_transform_path": object_dir / "object_transform.npz", "mesh_path": object_dir / "mesh.npz",
        }.items():
            record[key] = str(path.relative_to(args.data_root))
        records.append(record)
        if args.surface_pool_points:
            record["surface_pool_path"] = str((object_dir / "surface_pool.npz").relative_to(args.data_root))
    save_json(records, object_dir / "samples.json")
    return records


def object_complete(object_dir, args):
    if not render_complete(object_dir, args.num_views) or not (object_dir / "samples.json").is_file():
        return False
    try:
        records = json.loads((object_dir / "samples.json").read_text())
        complete = len(records) == args.num_views and all(
            (args.data_root / record[key]).is_file()
            for record in records for key in ("depth_path", "camera_path", "full_surface_path")
        )
        if not complete:
            return False
        if args.surface_pool_points:
            with np.load(object_dir / "surface_pool.npz", allow_pickle=False) as pool:
                validate_surface_pool(pool)
                if len(pool["points_object"]) != args.surface_pool_points:
                    return False
        for record in records:
            with np.load(args.data_root / record["full_surface_path"], allow_pickle=False) as surface:
                validate_surface(surface, require_normals=True)
                if len(surface["points_camera"]) != args.num_points:
                    return False
        return True
    except (ValueError, KeyError, OSError, EOFError):
        return False


def process_object(obj, split, args, gpu_slots):
    start = time.monotonic()
    object_dir = args.data_root / "generated_data" / obj["object_id"]
    try:
        if object_complete(object_dir, args):
            return {"object_id": obj["object_id"], "status": "already complete"}
        if not render_complete(object_dir, args.num_views):
            object_dir.mkdir(parents=True, exist_ok=True)
            for name in ("render_complete.json", "samples.json", "target_latent.npz"):
                (object_dir / name).unlink(missing_ok=True)
            gpu_id = gpu_slots.get()
            try:
                render_object(obj, object_dir, args, gpu_id)
                source = Path(obj["model_path"]).stat()
                if source.st_size != obj["source_size"] or source.st_mtime_ns != obj["source_mtime_ns"]:
                    raise ValueError("Input changed during rendering; rerun after generation finishes")
                save_json({"num_views": args.num_views}, object_dir / "render_complete.json")
            finally:
                gpu_slots.put(gpu_id)
        package_object(obj, split, args)
        return {"object_id": obj["object_id"], "status": "created", "seconds": round(time.monotonic() - start, 2)}
    except Exception as error:
        return {"object_id": obj["object_id"], "stage": "render", "error": f"{type(error).__name__}: {error}"}


def write_manifests(args, objects):
    generated = args.data_root / "generated_data"
    rendered, ready = [], []
    for obj in objects:
        object_dir = generated / obj["object_id"]
        if not object_complete(object_dir, args):
            continue
        rows = json.loads((object_dir / "samples.json").read_text())
        target = object_dir / "target_latent.npz"
        if target.is_file():
            validate_target(target)
            for row in rows:
                row["target_path"] = str(target.relative_to(args.data_root))
            ready.extend(rows)
        rendered.extend(rows)
    for filename, rows in (("samples_rendered.jsonl", rendered), ("samples.jsonl", ready)):
        path = generated / filename
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
        temporary.replace(path)
    return rendered, ready


def run(args):
    generated = args.data_root / "generated_data"
    settings_path = generated / "settings.json"
    objects_path = generated / "objects.json"
    if args.stage == "latents":
        objects = json.loads(objects_path.read_text())
        settings = json.loads(settings_path.read_text())
        if settings["pipeline_sha256"] != pipeline_sha256():
            raise ValueError("Pipeline code changed since rendering; use the same code revision")
        for key in SETTING_NAMES:
            setattr(args, key, settings[key])
    else:
        if args.objects_root is None:
            raise ValueError("--objects-root is required for render/all")
        objects = get_objects(args.objects_root, args.pattern, args.ready_marker)
        if args.data_root == args.objects_root.resolve() or args.objects_root.resolve() in args.data_root.parents:
            raise ValueError("Keep --data-root outside --objects-root to avoid discovering generated files")
        version = subprocess.check_output([args.blender, "--version"], text=True)
        blender_version = next(line for line in version.splitlines() if line.startswith("Blender "))
        settings = {name: getattr(args, name) for name in SETTING_NAMES}
        settings.update(format_version=2, pointmap_storage="reconstruct_from_depth",
                        full_surface_format_version=2, normal_method="triangle_face_winding",
                        persistent_data=True, blender_version=blender_version,
                        voxel_resolution=64, latent_shape=[8, 16, 16, 16],
                        camera_schedule="shared_hammersley", color_transform="Filmic",
                        normalization="evaluated_blender_world_centered_unit_cube")
        settings["pipeline_sha256"] = pipeline_sha256()
        if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
            raise ValueError(f"Inputs/settings differ from {settings_path}; use a new --data-root")
        if objects_path.exists():
            check_existing_objects(objects, json.loads(objects_path.read_text()), generated)
        save_json(settings, settings_path)
        save_json(objects, objects_path)
    splits_path = generated / "splits.json"
    existing_splits = json.loads(splits_path.read_text()) if splits_path.exists() else None
    splits = make_splits(objects, args.seed, args.train_fraction, args.val_fraction, existing_splits)
    save_json(splits, splits_path)
    lookup = {object_id: split for split, ids in splits.items() for object_id in ids}
    selected = objects
    if args.object_id:
        selected = [obj for obj in objects if obj["object_id"] == args.object_id]
        if not selected:
            raise ValueError(f"Unknown --object-id: {args.object_id}")
    if args.limit is not None:
        selected = selected[:args.limit]
    gpu_ids = [gpu.strip() for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu.strip() and gpu.strip() != "-1"]
    failures = []
    if args.stage in {"all", "render"}:
        pending = [obj for obj in selected if not object_complete(generated / obj["object_id"], args)]
        if pending:
            if args.render_device != "CPU" and not gpu_ids:
                raise RuntimeError("Set CUDA_VISIBLE_DEVICES to allocated GPU IDs (or use --render-device CPU for a smoke test)")
            slots = gpu_ids * args.render_workers_per_gpu if args.render_device != "CPU" else [None] * args.workers
            with mp.Manager() as manager:
                queue = manager.Queue()
                for slot in slots:
                    queue.put(slot)
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
                    futures = [executor.submit(process_object, obj, lookup[obj["object_id"]], args, queue) for obj in pending]
                    for index, future in enumerate(as_completed(futures), 1):
                        result = future.result()
                        print(f"[{index}/{len(pending)}] {json.dumps(result)}", flush=True)
                        if "error" in result:
                            failures.append(result)
    rendered, ready = write_manifests(args, objects)
    if args.stage in {"all", "latents"}:
        eligible = {row["object_id"] for row in rendered}
        for obj in selected:
            if obj["object_id"] not in eligible:
                failures.append({"object_id": obj["object_id"], "stage": "latents", "error": "Render/package incomplete"})
        save_metadata(generated, args.encoder_checkpoint)
        pending_ids = [obj["object_id"] for obj in selected if obj["object_id"] in eligible
                       and not (generated / obj["object_id"] / "target_latent.npz").is_file()]
        if pending_ids:
            if not gpu_ids:
                raise RuntimeError("Target encoding requires CUDA_VISIBLE_DEVICES; use --stage render for CPU tests")
            with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=mp.get_context("spawn")) as executor:
                futures = [executor.submit(encode_objects, pending_ids[i::len(gpu_ids)], args.data_root,
                                           args.encoder_checkpoint, gpu) for i, gpu in enumerate(gpu_ids)
                           if pending_ids[i::len(gpu_ids)]]
                for future in as_completed(futures):
                    failures.extend(future.result())
        rendered, ready = write_manifests(args, objects)
    save_json(failures, generated / "failed_objects.json")
    print(f"Rendered samples: {len(rendered)}; training-ready samples: {len(ready)}; failures: {len(failures)}")
    print(f"Training manifest: {generated / 'samples.jsonl'}", flush=True)
    if failures:
        raise SystemExit(1)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        if args.objects_root is None:
            raise ValueError("--dry-run requires --objects-root")
        objects = get_objects(args.objects_root, args.pattern, args.ready_marker)
        print(json.dumps({"objects": len(objects), "examples": objects[:5],
                          "settings": {key: getattr(args, key) for key in SETTING_NAMES}}, indent=2))
        return
    if args.stage in {"all", "latents"} and not args.encoder_checkpoint.is_file():
        raise FileNotFoundError(f"Encoder checkpoint: {args.encoder_checkpoint}; use --stage render to prepare geometry first")
    generated = args.data_root / "generated_data"
    generated.mkdir(parents=True, exist_ok=True)
    # Prevent two independent jobs from racing on this dataset's manifests.
    # Parallelism within one job uses independent per-object outputs and GPU slots.
    with (generated / ".build.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another builder is writing {args.data_root}") from None
        run(args)


if __name__ == "__main__":
    main()
