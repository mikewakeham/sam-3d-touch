import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

import os
os.environ["LIDRA_SKIP_INIT"] = "true"


DEFAULT_ENCODER_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints/hf/ss_encoder.ckpt"


def checkpoint_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def load_normalized_mesh(model_path):
    with np.load(model_path, allow_pickle=False) as data:
        if data["coordinate_frame"].item() != "normalized_object":
            raise ValueError(f"Unexpected mesh coordinate frame: {model_path}")
        mesh = trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh has no vertices or faces")
    if not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("Mesh must have finite vertices and positive surface area")
    center = mesh.bounds.mean(axis=0)
    extent = np.ptp(mesh.bounds, axis=0).max()
    if not np.allclose(center, 0, atol=1e-4) or not np.isclose(extent, 1, atol=1e-4):
        raise ValueError(f"Expected centered unit-cube mesh: center={center}, extent={extent}")
    return mesh


def voxelize_mesh(mesh, resolution=64):
    import open3d as o3d
    import torch

    vertices = np.clip(
        np.asarray(mesh.vertices),
        -0.5 + 1e-6,
        0.5 - 1e-6,
    )

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        o3d_mesh,
        voxel_size=1 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )

    coords = np.asarray(
        [voxel.grid_index for voxel in voxel_grid.get_voxels()],
        dtype=np.int64,
    )

    if len(coords) == 0:
        raise ValueError("Voxelization produced no occupied voxels")

    if np.any(coords < 0) or np.any(coords >= resolution):
        raise ValueError("Voxel coordinates are outside the 64^3 grid")

    occupancy = torch.zeros(
        1,
        resolution,
        resolution,
        resolution,
        dtype=torch.float32,
    )

    occupancy[
        0,
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
    ] = 1

    return occupancy


def load_encoder(checkpoint_path, device):
    from sparse_structure_vae import (
        SparseStructureEncoderTdfyWrapper,
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    encoder = SparseStructureEncoderTdfyWrapper(
        sample_posterior=False,
        return_raw=True,
        in_channels=1,
        latent_channels=8,
        channels=[32, 128, 512],
        num_res_blocks=2,
        num_res_blocks_middle=2,
        pretrained_ckpt_path=str(checkpoint_path),
    )

    return encoder.eval().to(device)


def validate_target(path, camera_frame=False):
    with np.load(path, allow_pickle=False) as data:
        fields = {"mean"}
        if camera_frame:
            fields |= {"coordinate_frame", "T_target_from_object", "T_camera_from_target"}
        if set(data.files) != fields:
            raise ValueError(f"Unexpected target fields: {data.files}")
        if camera_frame:
            if str(data["coordinate_frame"]) != "normalized_sam_camera":
                raise ValueError(f"Unexpected target frame: {path}")
            for key in ("T_target_from_object", "T_camera_from_target"):
                transform = data[key]
                if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                        or not np.allclose(transform[3], [0, 0, 0, 1])
                        or np.linalg.det(transform[:3, :3]) <= 0):
                    raise ValueError(f"Invalid {key}: {path}")
        mean = data["mean"]

    if mean.shape != (8, 16, 16, 16):
        raise ValueError(f"Unexpected target shape: {mean.shape}")

    if mean.dtype != np.float32:
        raise ValueError(f"Unexpected target dtype: {mean.dtype}")

    if not np.isfinite(mean).all():
        raise ValueError("Target contains non-finite values")

    return mean


def save_target(mean, output_path, **metadata):
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    with temporary_path.open("wb") as file:
        np.savez_compressed(file, mean=mean, **metadata)

    validate_target(temporary_path, camera_frame=bool(metadata))
    temporary_path.replace(output_path)


def generate_target(object_id, data_root, encoder, device, overwrite):
    import torch

    object_dir = data_root / "generated_data" / object_id
    model_path = object_dir / "mesh.npz"
    object_transform_path = object_dir / "object_transform.npz"
    target_path = object_dir / "target_latent.npz"

    for path in [model_path, object_transform_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if target_path.is_file() and not overwrite:
        validate_target(target_path)
        return target_path, False

    mesh = load_normalized_mesh(
        model_path=model_path,
    )

    occupancy = voxelize_mesh(mesh)

    with torch.inference_mode():
        output = encoder(occupancy.unsqueeze(0).to(device))
        mean = output["mean"][0].float().cpu().numpy()

    save_target(mean, target_path)

    return target_path, True


def save_metadata(generated_dir, encoder_checkpoint):
    metadata_path = generated_dir / "target_latents.json"

    metadata = {
        "format_version": 1,
        "coordinate_frame": "normalized_object",
        "voxelization": "surface_fixed_bounds",
        "voxel_resolution": 64,
        "latent_shape": [8, 16, 16, 16],
        "latent_dtype": "float32",
        "encoder_checkpoint_sha256": checkpoint_sha256(encoder_checkpoint),
    }

    if metadata_path.is_file():
        with metadata_path.open() as file:
            existing = json.load(file)

        if existing != metadata:
            raise ValueError(
                f"Existing target metadata does not match this run: "
                f"{metadata_path}"
            )

        return

    temporary_path = metadata_path.with_name(metadata_path.name + ".tmp")

    with temporary_path.open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    temporary_path.replace(metadata_path)



def encode_objects(object_ids, data_root, checkpoint, gpu_id):
    # One model per allocated GPU, reused across all its objects.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Target encoding requires a CUDA GPU; run this stage on the cluster")
    encoder = load_encoder(checkpoint, torch.device("cuda:0"))
    failures = []
    for index, object_id in enumerate(object_ids, 1):
        try:
            generate_target(object_id, data_root, encoder, torch.device("cuda:0"), False)
            print(f"[GPU {gpu_id}: {index}/{len(object_ids)}] latent {object_id}", flush=True)
        except Exception as error:
            failures.append({"object_id": object_id, "stage": "latents", "error": str(error)})
    return failures


def camera_target_mesh(mesh, camera_path):
    from sample_full_surface import sam_camera_transform

    camera = sam_camera_transform(camera_path)
    vertices = np.asarray(mesh.vertices) @ camera[:3, :3].T
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) / 2
    extent = (upper - lower).max()
    if not np.isfinite(vertices).all() or extent <= 0:
        raise ValueError(f"Invalid camera-frame mesh: {camera_path}")
    target = mesh.copy()
    target.vertices = (vertices - center) / extent
    target_from_object = np.eye(4)
    target_from_object[:3, :3] = camera[:3, :3] / extent
    target_from_object[:3, 3] = -center / extent
    camera_from_target = np.eye(4)
    camera_from_target[:3, :3] *= extent
    camera_from_target[:3, 3] = center + camera[:3, 3]
    return target, dict(coordinate_frame="normalized_sam_camera",
                       T_target_from_object=target_from_object,
                       T_camera_from_target=camera_from_target)


def camera_target_path(record):
    return Path(record["camera_path"]).with_name("target_latent_camframe.npz")


def generate_camera_target(record, data_root, mesh, encoder, device):
    import torch

    target, metadata = camera_target_mesh(mesh, data_root / record["camera_path"])
    occupancy = voxelize_mesh(target)
    with torch.inference_mode():
        mean = encoder(occupancy.unsqueeze(0).to(device))["mean"][0].float().cpu().numpy()
    path = data_root / camera_target_path(record)
    save_target(mean, path, **metadata)
    return path


def encode_camera_objects(objects, data_root, checkpoint, gpu_id, overwrite=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    import torch
    torch.set_num_threads(1)
    encoder = None
    for index, records in enumerate(objects, 1):
        pending = [row for row in records if overwrite or not (data_root / camera_target_path(row)).is_file()
                   or (data_root / camera_target_path(row)).stat().st_size == 0]
        if pending:
            if encoder is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("Target encoding requires a CUDA GPU; run this stage on the cluster")
                encoder = load_encoder(checkpoint, torch.device("cuda:0"))
            mesh = load_normalized_mesh(data_root / records[0]["mesh_path"])
            for row in pending:
                generate_camera_target(row, data_root, mesh, encoder, torch.device("cuda:0"))
        print(f"[GPU {gpu_id}: {index}/{len(objects)}] {records[0]['object_id']}: "
              f"{len(records) - len(pending)} reused, {len(pending)} encoded", flush=True)


def write_camera_manifests(data_root, records):
    generated = data_root / "generated_data"
    by_id = {row["sample_id"]: row for row in records}
    sources = [(generated / "samples.jsonl", records)]
    touch_path = generated / "samples_simulated_touches.jsonl"
    if touch_path.is_file():
        touches = [json.loads(line) for line in touch_path.read_text().splitlines() if line.strip()]
        if len(touches) != len(records) or {row["sample_id"] for row in touches} != set(by_id):
            raise ValueError("Touch manifest must contain the same samples as samples.jsonl")
        for row in touches:
            original = by_id[row["sample_id"]]
            if any(row[key] != value for key, value in original.items() if key != "touch_path"):
                raise ValueError(f"Touch manifest differs from samples.jsonl: {row['sample_id']}")
        sources.append((touch_path, touches))
    for source, rows in sources:
        destination = source.with_name(source.stem + "_camframe.jsonl")
        temporary = destination.with_suffix(".tmp.jsonl")
        with temporary.open("w") as file:
            for row in rows:
                file.write(json.dumps(dict(row, target_path=str(camera_target_path(row)))) + "\n")
        temporary.replace(destination)
        print(f"Wrote {destination} ({len(rows)} views)", flush=True)


def camera_main(argv):
    import argparse
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from make_data import save_json

    parser = argparse.ArgumentParser(description="Encode camera-frame targets from existing renders")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_ENCODER_CHECKPOINT)
    parser.add_argument("--limit", type=int, help="Encode only the first N objects; do not publish manifests")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.data_root = args.data_root.resolve()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    generated = args.data_root / "generated_data"
    print(f"Reading manifest and encoder metadata from {generated}", flush=True)
    source = generated / "samples.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not records or len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("Expected a nonempty manifest with unique sample IDs")
    if len({camera_target_path(row) for row in records}) != len(records):
        raise ValueError("Each sample must have its own view directory")
    groups = {}
    for row in records:
        groups.setdefault(row["object_id"], []).append(row)
    for rows in groups.values():
        if len({row["mesh_path"] for row in rows}) != 1:
            raise ValueError(f"Views must share one mesh: {rows[0]['object_id']}")
    metadata = dict(coordinate_frame="normalized_sam_camera", normalization="aabb_center_longest_extent",
                    voxel_resolution=64, latent_shape=[8, 16, 16, 16], format_version=1,
                    encoder_checkpoint_sha256=checkpoint_sha256(args.encoder_checkpoint),
                    source_manifest_sha256=checkpoint_sha256(source))
    for name in ("settings.json", "target_latents.json"):
        if (generated / name).is_file():
            metadata[name] = checkpoint_sha256(generated / name)
    metadata_path = generated / "target_latents_camframe.json"
    if metadata_path.is_file():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Camera-frame encoding settings or source manifest changed")
    else:
        save_json(metadata, metadata_path)
    objects = list(groups.values())[:args.limit]
    gpu_ids = [gpu.strip() for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if gpu.strip() and gpu.strip() != "-1"]
    if not gpu_ids:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to allocated GPU IDs")
    print(f"Processing {len(objects)} objects ({sum(map(len, objects))} views). "
          + ("Replacing camera-frame targets." if args.overwrite else
             "Reusing existing targets; encoding missing views."), flush=True)
    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(encode_camera_objects, objects[i::len(gpu_ids)], args.data_root,
                                   args.encoder_checkpoint, gpu, args.overwrite)
                   for i, gpu in enumerate(gpu_ids) if objects[i::len(gpu_ids)]]
        for future in futures:
            future.result()
    if args.limit is None:
        write_camera_manifests(args.data_root, records)
    else:
        print("Subset complete; rerun without --limit to finish and publish manifests.", flush=True)


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target-frame", choices=["object", "camera"], default="object")
    args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if args.target_frame == "camera":
        camera_main(remaining)
    else:
        from make_data import main as make_data_main
        make_data_main(["--stage", "latents", *remaining])


if __name__ == "__main__":
    main()
