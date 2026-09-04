import time
import argparse
import json
import random
import subprocess
from pathlib import Path
import numpy as np
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from sample_touch import SAMPLERS, prepare_surface, sample_touch

class BlenderError(RuntimeError):
    pass

DEFAULT_DATA_ROOT = Path(
    "/n/holylabs/qianqian_lab/Lab/mwakeham/"
    "visuotactile-objects/sam-3d-touch-data/objaverse-dexonomy"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--render-script", type=Path, default=Path(__file__).with_name("render_blender.py"))
    parser.add_argument("--num-views", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--camera-radius", type=float, default=2.0)
    parser.add_argument("--fov-degrees", type=float, default=40.0)
    parser.add_argument("--touch-method", choices=SAMPLERS, default="geodesic_ball")
    parser.add_argument("--touch-density", type=float, default=200000)
    parser.add_argument("--touch-args", type=json.loads, default={})
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--render-workers-per-gpu", type=int, default=1)
    

    return parser.parse_args()


def get_objects(objects_dir):
    objects = []

    for object_dir in sorted(objects_dir.iterdir()):
        if not object_dir.is_dir():
            continue

        model_path = object_dir / "model.obj"
        material_path = object_dir / "material.mtl"
        texture_path = object_dir / "material_0.png"

        for path in [model_path, material_path, texture_path]:
            if not path.is_file():
                raise FileNotFoundError(path)

        objects.append(
            {
                "object_id": object_dir.name,
                "model_path": model_path,
            }
        )

    return objects


def make_splits(objects, seed, train_fraction, val_fraction):
    object_ids = [obj["object_id"] for obj in objects]

    random.Random(seed).shuffle(object_ids)

    num_objects = len(object_ids)
    num_train = int(train_fraction * num_objects)
    num_val = int(val_fraction * num_objects)

    return {
        "train": sorted(object_ids[:num_train]),
        "val": sorted(object_ids[num_train:num_train + num_val]),
        "test": sorted(object_ids[num_train + num_val:]),
    }


def save_splits(splits, path):
    with path.open("w") as file:
        json.dump(splits, file, indent=2)
        file.write("\n")


def get_split_lookup(splits):
    return {
        object_id: split
        for split, object_ids in splits.items()
        for object_id in object_ids
    }

def format_time(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_object(obj, output_dir, args, gpu_id):
    object_seed = args.seed ^ int(obj["object_id"][:8], 16)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    process = subprocess.Popen(
        [
            args.blender, "--background",
            "--threads", "4",
            "--python-exit-code", "1",
            "--python", str(args.render_script),
            "--",
            "--object", str(obj["model_path"]),
            "--output", str(output_dir),
            "--num-views", str(args.num_views),
            "--resolution", str(args.resolution),
            "--seed", str(object_seed),
            "--camera-radius", str(args.camera_radius),
            "--fov-degrees", str(args.fov_degrees),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    output = []

    for line in process.stdout:
        output.append(line)

        if line.startswith("Rendered view "):
            print(f"{obj['object_id']} | {line.strip()}", flush=True)

    returncode = process.wait()

    if returncode != 0:
        print("".join(output))
        raise BlenderError(f"Blender failed with exit code {returncode}")


def depth_to_pointmap(depth, K):
    """
    Back-project camera-axis depth into the camera convention expected by
    SAM 3D's external pointmap path.

    Blender/OpenCV camera:
        +x right, +y down, +z forward

    SAM 3D/PyTorch3D pointmap:
        +x left, +y up, +z forward
    """
    height, width = depth.shape

    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    pointmap = np.stack([-x, -y, z], axis=-1).astype(np.float32)

    valid = np.isfinite(depth) & (depth > 0)
    pointmap[~valid] = np.nan

    return pointmap


def make_touch_sample(obj, view_dir, object_transform_path, surface, args):
    touch_path = view_dir / "touches.npz"

    sample_touch(
        model_path=obj["model_path"],
        object_transform_path=object_transform_path,
        camera_path=view_dir / "camera.npz",
        depth_path=view_dir / "depth.npy",
        output_path=touch_path,
        method=args.touch_method,
        density=args.touch_density,
        seed=args.seed,
        surface=surface,
        **args.touch_args,
    )

    if not touch_path.is_file():
        raise FileNotFoundError(touch_path)

    return touch_path


def make_stage1_target(
    obj,
    object_output_dir,
    object_transform_path,
):
    """
    Stage-1 target preparation will go here later.

    It should eventually:

        load model.obj
        apply T_normalized_from_source
        voxelize the complete object to 64^3
        run the frozen SAM 3D sparse-structure encoder
        save the posterior mean to target_latent.npz

    One target is created per object, not per rendered view.
    """
    return None


def relative_path(path, root):
    if path is None:
        return None

    return str(path.relative_to(root))


def package_object(obj, split, generated_dir, data_root, completed_sample_ids, args,):
    object_output_dir = generated_dir / obj["object_id"]

    object_transform_path = (
        object_output_dir / "object_transform.npz"
    )

    if not object_transform_path.is_file():
        raise FileNotFoundError(object_transform_path)

    target_path = make_stage1_target(
        obj=obj,
        object_output_dir=object_output_dir,
        object_transform_path=object_transform_path,
    )

    surface = prepare_surface(
        model_path=obj["model_path"],
        object_transform_path=object_transform_path,
        density=args.touch_density,
        seed=args.seed,
    )

    for view_index in range(args.num_views):
        view_id = f"{view_index:03d}"
        sample_id = f"{obj['object_id']}_{view_id}"

        if sample_id in completed_sample_ids:
            continue

        view_dir = object_output_dir / "views" / view_id

        image_path = view_dir / "image.png"
        depth_path = view_dir / "depth.npy"
        camera_path = view_dir / "camera.npz"
        pointmap_path = view_dir / "pointmap.npy"

        for path in [image_path, depth_path, camera_path]:
            if not path.is_file():
                raise FileNotFoundError(path)

        depth = np.load(depth_path).astype(np.float32)

        with np.load(camera_path) as camera:
            K = camera["K"].astype(np.float32)

        pointmap = depth_to_pointmap(depth, K)
        np.save(pointmap_path, pointmap)

        touch_path = make_touch_sample(
            obj=obj,
            view_dir=view_dir,
            object_transform_path=object_transform_path,
            surface=surface,
            args=args,
        )

        yield {
            "sample_id": sample_id,
            "object_id": obj["object_id"],
            "view_id": view_id,
            "split": split,
            "image_path": relative_path(image_path, data_root),
            "depth_path": relative_path(depth_path, data_root),
            "pointmap_path": relative_path(pointmap_path, data_root),
            "camera_path": relative_path(camera_path, data_root),
            "object_transform_path": relative_path(
                object_transform_path,
                data_root,
            ),
            "touch_path": relative_path(touch_path, data_root),
            "target_path": relative_path(target_path, data_root),
        }

def load_failed_object_ids(path):
    if not path.is_file():
        return set()

    object_ids = set()

    with path.open() as file:
        for line in file:
            record = json.loads(line)
            object_ids.add(record["object_id"])

    return object_ids

def load_sample_ids(path):
    if not path.is_file():
        return set()

    sample_ids = set()

    with path.open() as file:
        for line in file:
            record = json.loads(line)
            sample_ids.add(record["sample_id"])

    return sample_ids


def append_jsonl(records, path):
    with path.open("a") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def process_object(
    obj,
    split,
    generated_dir,
    data_root,
    completed_sample_ids,
    args,
    write_lock,
    progress,
    completion_times,
    completed_views_at_start,
    pending_views,
    total_views,
    start_time,
    manifest_path,
    failures_path,
):
    touch_start_time = time.monotonic()
    print(f"Touch processing {obj['object_id']}", flush=True)

    try:
        for record in package_object(
            obj=obj,
            split=split,
            generated_dir=generated_dir,
            data_root=data_root,
            completed_sample_ids=completed_sample_ids,
            args=args,
        ):
            with write_lock:
                append_jsonl([record], manifest_path)

                progress.value += 1
                views_added = progress.value - completed_views_at_start
                remaining_views = pending_views - views_added
                now = time.monotonic()
                elapsed = now - start_time

                completion_times.append(now)

                if len(completion_times) > 129:
                    completion_times.pop(0)

                if len(completion_times) >= 32:
                    window_elapsed = completion_times[-1] - completion_times[0]
                    views_in_window = len(completion_times) - 1
                    views_per_second = views_in_window / window_elapsed
                    eta_text = format_time(remaining_views / views_per_second)
                    rate_text = f"{views_per_second:.2f} views/s"
                else:
                    eta_text = "warming up"
                    rate_text = "--"

                print(
                    f"[{progress.value}/{total_views} views] packaged | "
                    f"elapsed {format_time(elapsed)} | "
                    f"rate {rate_text} | "
                    f"ETA {eta_text}",
                    flush=True,
                )

    except BlenderError:
        raise

    except Exception as error:
        failure = {
            "object_id": obj["object_id"],
            "error": f"{type(error).__name__}: {error}",
        }

        with write_lock:
            append_jsonl([failure], failures_path)

            print(
                f"Failed {obj['object_id']}: {failure['error']}",
                flush=True,
            )
    
    elapsed = time.monotonic() - touch_start_time

    print(
        f"Touch finished {obj['object_id']} in {format_time(elapsed)}",
        flush=True,
    )

    return obj["object_id"]


def render_job(obj, generated_dir, args, gpu_id):
    start_time = time.monotonic()

    print(f"Rendering {obj['object_id']} on GPU {gpu_id}", flush=True)

    render_object(
        obj=obj,
        output_dir=generated_dir / obj["object_id"],
        args=args,
        gpu_id=gpu_id,
    )

    elapsed = time.monotonic() - start_time

    print(
        f"Rendered {obj['object_id']} in {format_time(elapsed)}",
        flush=True,
    )

    return obj


def main():
    args = parse_args()

    objects_dir = args.data_root / "objects"
    generated_dir = args.data_root / "generated_data"
    splits_path = generated_dir / "splits.json"
    manifest_path = generated_dir / "samples.jsonl"
    failures_path = generated_dir / "failed_objects.jsonl"

    generated_dir.mkdir(parents=True, exist_ok=True)

    objects = get_objects(objects_dir)

    splits = make_splits(
        objects=objects,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    save_splits(splits, splits_path)

    split_lookup = get_split_lookup(splits)

    if args.object_id is not None:
        objects = [
            obj
            for obj in objects
            if obj["object_id"] == args.object_id
        ]

        if not objects:
            raise ValueError(f"Unknown object ID: {args.object_id}")

    if args.retry_failed:
        failed_object_ids = set()
    else:
        failed_object_ids = load_failed_object_ids(failures_path)

    failed_count = sum(
        obj["object_id"] in failed_object_ids
        for obj in objects
    )

    objects = [
        obj
        for obj in objects
        if obj["object_id"] not in failed_object_ids
    ]

    print(f"Previously failed objects skipped: {failed_count}")

    completed_sample_ids = load_sample_ids(manifest_path)
    pending_objects = []

    total_views = len(objects) * args.num_views
    completed_views = 0

    for obj in objects:
        expected_sample_ids = {
            f"{obj['object_id']}_{view_index:03d}"
            for view_index in range(args.num_views)
        }

        completed_views += len(
            expected_sample_ids & completed_sample_ids
        )

        if not expected_sample_ids.issubset(completed_sample_ids):
            pending_objects.append(obj)

    pending_views = total_views - completed_views

    print(f"Objects already complete: {len(objects) - len(pending_objects)}")
    print(f"Objects remaining: {len(pending_objects)}")
    print(f"Views already complete: {completed_views}/{total_views}")
    print(f"Views remaining: {pending_views}", flush=True)

    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [
        gpu_id.strip()
        for gpu_id in visible_gpus.split(",")
        if gpu_id.strip()
    ]

    if not gpu_ids:
        raise RuntimeError("CUDA_VISIBLE_DEVICES contains no allocated GPUs")

    print(f"Workers: {args.workers}")
    print(f"GPUs: {gpu_ids}")
    print(f"Concurrent Blender processes per GPU: {args.render_workers_per_gpu}")

    start_time = time.monotonic()
    completed_views_at_start = completed_views

    with mp.Manager() as manager:
        write_lock = manager.Lock()
        progress = manager.Value("i", completed_views)
        completion_times = manager.list()

        render_executors = [
            ThreadPoolExecutor(
                max_workers=args.render_workers_per_gpu
            )
            for _ in gpu_ids
        ]

        try:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=mp.get_context("spawn"),
            ) as touch_executor:
                render_futures = []

                for object_index, obj in enumerate(pending_objects):
                    gpu_index = object_index % len(gpu_ids)

                    future = render_executors[gpu_index].submit(
                        render_job,
                        obj,
                        generated_dir,
                        args,
                        gpu_ids[gpu_index],
                    )

                    render_futures.append(future)

                touch_futures = []

                for render_future in as_completed(render_futures):
                    obj = render_future.result()

                    object_completed_sample_ids = {
                        f"{obj['object_id']}_{view_index:03d}"
                        for view_index in range(args.num_views)
                        if f"{obj['object_id']}_{view_index:03d}"
                        in completed_sample_ids
                    }

                    future = touch_executor.submit(
                        process_object,
                        obj,
                        split_lookup[obj["object_id"]],
                        generated_dir,
                        args.data_root,
                        object_completed_sample_ids,
                        args,
                        write_lock,
                        progress,
                        completion_times,
                        completed_views_at_start,
                        pending_views,
                        total_views,
                        start_time,
                        manifest_path,
                        failures_path,
                    )

                    touch_futures.append(future)

                for object_index, future in enumerate(
                    as_completed(touch_futures),
                    start=1,
                ):
                    object_id = future.result()

                    print(
                        f"Touch objects finished: "
                        f"{object_index}/{len(touch_futures)} "
                        f"({object_id})",
                        flush=True,
                    )

        finally:
            for executor in render_executors:
                executor.shutdown()

        completed_views = progress.value
        views_added = completed_views - completed_views_at_start
    print(f"Objects considered: {len(objects)}")
    print(f"Views added: {views_added}")
    print(f"Views complete: {completed_views}/{total_views}")
    print(f"Manifest: {manifest_path}")
    print(f"Failures: {failures_path}")

if __name__ == "__main__":
    main()