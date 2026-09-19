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

    validate_target(temporary_path)
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


if __name__ == "__main__":
    import sys
    from make_data import main
    main(["--stage", "latents", *sys.argv[1:]])
