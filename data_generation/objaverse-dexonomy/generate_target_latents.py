import time
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh

import os
os.environ["LIDRA_SKIP_INIT"] = "true"

from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import (
    SparseStructureEncoderTdfyWrapper,
)


DEFAULT_DATA_ROOT = Path(
    "/n/holylabs/qianqian_lab/Lab/mwakeham/"
    "visuotactile-objects/sam-3d-touch-data/objaverse-dexonomy"
)

DEFAULT_ENCODER_CHECKPOINT = Path(
    "/n/holylabs/qianqian_lab/Lab/mwakeham/"
    "visuotactile-objects/sam-3d-touch/checkpoints/hf/ss_encoder.ckpt"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_ENCODER_CHECKPOINT)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def format_time(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def checkpoint_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def get_object_ids(manifest_path, object_id):
    object_ids = set()

    with manifest_path.open() as file:
        for line in file:
            record = json.loads(line)
            object_ids.add(record["object_id"])

    object_ids = sorted(object_ids)

    if object_id is not None:
        if object_id not in object_ids:
            raise ValueError(f"Object is not in manifest: {object_id}")

        return [object_id]

    return object_ids


def load_normalized_mesh(model_path, object_transform_path):
    mesh = trimesh.load(
        str(model_path),
        force="mesh",
        process=False,
        skip_materials=True,
    )

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh has no vertices or faces")

    with np.load(object_transform_path, allow_pickle=False) as data:
        transform = data["T_normalized_from_source"].astype(np.float64)

    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Invalid T_normalized_from_source")

    mesh.apply_transform(transform)

    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Normalized mesh has non-finite vertices")

    bounds_min = mesh.vertices.min(axis=0)
    bounds_max = mesh.vertices.max(axis=0)
    center = (bounds_min + bounds_max) / 2
    max_extent = (bounds_max - bounds_min).max()

    if not np.allclose(center, 0, atol=1e-4):
        raise ValueError(f"Normalized mesh is not centered: {center}")

    if not np.isclose(max_extent, 1, atol=1e-4):
        raise ValueError(f"Normalized mesh extent is not 1: {max_extent}")

    return mesh


def voxelize_mesh(mesh, resolution=64):
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


def validate_target(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"mean"}:
            raise ValueError(f"Unexpected target fields: {data.files}")

        mean = data["mean"]

    if mean.shape != (8, 16, 16, 16):
        raise ValueError(f"Unexpected target shape: {mean.shape}")

    if mean.dtype != np.float32:
        raise ValueError(f"Unexpected target dtype: {mean.dtype}")

    if not np.isfinite(mean).all():
        raise ValueError("Target contains non-finite values")

    return mean


def save_target(mean, output_path):
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    with temporary_path.open("wb") as file:
        np.savez_compressed(file, mean=mean)

    temporary_path.replace(output_path)
    validate_target(output_path)


def generate_target(object_id, data_root, encoder, device, overwrite):
    model_path = data_root / "objects" / object_id / "model.obj"
    object_dir = data_root / "generated_data" / object_id
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
        object_transform_path=object_transform_path,
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


def update_manifest(manifest_path, data_root):
    records = []
    updated = 0

    with manifest_path.open() as file:
        for line in file:
            record = json.loads(line)
            target_path = (
                data_root
                / "generated_data"
                / record["object_id"]
                / "target_latent.npz"
            )

            if target_path.is_file():
                validate_target(target_path)
                relative_target_path = str(target_path.relative_to(data_root))

                if record.get("target_path") != relative_target_path:
                    record["target_path"] = relative_target_path
                    updated += 1

            records.append(record)

    temporary_path = manifest_path.with_name(manifest_path.name + ".tmp")

    with temporary_path.open("w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    temporary_path.replace(manifest_path)

    return updated


def main():
    args = parse_args()

    generated_dir = args.data_root / "generated_data"
    manifest_path = generated_dir / "samples.jsonl"

    for path in [manifest_path, args.encoder_checkpoint]:
        if not path.is_file():
            raise FileNotFoundError(path)

    object_ids = get_object_ids(
        manifest_path=manifest_path,
        object_id=args.object_id,
    )

    device = torch.device(args.device)
    encoder = load_encoder(args.encoder_checkpoint, device)

    save_metadata(
        generated_dir=generated_dir,
        encoder_checkpoint=args.encoder_checkpoint,
    )

    print(f"Objects: {len(object_ids)}")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder_checkpoint}")

    start_time = time.monotonic()
    created = 0
    skipped = 0

    for index, object_id in enumerate(object_ids, start=1):
        target_path, was_created = generate_target(
            object_id=object_id,
            data_root=args.data_root,
            encoder=encoder,
            device=device,
            overwrite=args.overwrite,
        )

        if was_created:
            created += 1
            status = "created"
        else:
            skipped += 1
            status = "already complete"

        elapsed = time.monotonic() - start_time
        seconds_per_object = elapsed / index
        eta = seconds_per_object * (len(object_ids) - index)

        print(
            f"[{index}/{len(object_ids)}] {object_id} | {status} | "
            f"elapsed {format_time(elapsed)} | ETA {format_time(eta)}",
            flush=True,
        )

    updated = update_manifest(
        manifest_path=manifest_path,
        data_root=args.data_root,
    )

    print()
    print(f"Targets created: {created}")
    print(f"Targets skipped: {skipped}")
    print(f"Manifest rows updated: {updated}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()