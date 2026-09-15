"""Shared target preparation for the two rotation probes. No model decoding."""
import copy
from pathlib import Path

import numpy as np
import torch

from dataloader import TouchDataset
from experiments.coordinate_system.scripts.full_checkpoints.protocol import evaluation_groups
from experiments.coordinate_system.scripts.coordinate_audit.frame_contract import rotate_grid
from experiments.coordinate_system.scripts.alignment_robustness.alignment_tolerance_protocol import rotation


def select_groups(data, train_objects, val_objects, views, seed):
    groups = []
    for split, count in (("train", train_objects), ("val", val_objects)):
        if count == 0:
            continue
        config = copy.deepcopy(data)
        config["dataset"]["split"] = split
        dataset = TouchDataset(config, include_touch=False)
        for records in evaluation_groups(dataset.records, seed, count, views):
            groups.append((split, records))
    return groups


def native_rotations():
    variants = [("identity", None, 0, np.eye(3, dtype=int))]
    for axis in "xyz":
        for angle in (90, 180):
            matrix = np.rint(rotation(axis, angle)).astype(int)
            variants.append((f"{axis}{angle}", axis, angle, matrix))
    return variants


def load_mesh(root, object_id):
    # Copied from generate_target_latents.py::load_normalized_mesh. That source
    # lives in ignored data-generation directories and may not exist on cluster.
    import trimesh
    mesh = trimesh.load(str(Path(root)/"objects"/object_id/"model.obj"),
                        force="mesh", process=False, skip_materials=True)
    with np.load(Path(root)/"generated_data"/object_id/"object_transform.npz") as data:
        mesh.apply_transform(data["T_normalized_from_source"].astype(np.float64))
    bounds = mesh.bounds
    np.testing.assert_allclose(bounds.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose((bounds[1]-bounds[0]).max(), 1, atol=1e-4)
    return mesh


def voxelize(mesh, stock=False):
    # Production Open3D voxelization/indexing, copied from the target builder.
    # Only native zero uses its boundary clipping. Padded rotations never clip.
    import open3d as o3d
    vertices = np.asarray(mesh.vertices)
    if stock:
        vertices = np.clip(vertices, -.5+1e-6, .5-1e-6)
    elif np.max(np.abs(vertices)) >= .5:
        raise ValueError("Rotated mesh does not fit the grid; do not clip it")
    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
    voxels = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        geometry, voxel_size=1/64, min_bound=(-.5, -.5, -.5), max_bound=(.5, .5, .5))
    coords = np.array([v.grid_index for v in voxels.get_voxels()], dtype=int)
    if len(coords) == 0 or np.any(coords < 0) or np.any(coords >= 64):
        raise ValueError("Empty or out-of-bounds target occupancy")
    grid = np.zeros((64, 64, 64), dtype=bool)
    grid[tuple(coords.T)] = True
    return grid


def load_encoder(checkpoint, device):
    # Same deterministic SS encoder construction as generate_target_latents.py.
    from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import SparseStructureEncoderTdfyWrapper
    return SparseStructureEncoderTdfyWrapper(
        sample_posterior=False, return_raw=True, in_channels=1, latent_channels=8,
        channels=[32, 128, 512], num_res_blocks=2, num_res_blocks_middle=2,
        pretrained_ckpt_path=str(checkpoint)).eval().requires_grad_(False).to(device)


@torch.no_grad()
def encode(encoder, grid, device):
    x = torch.from_numpy(np.ascontiguousarray(grid)).float()[None, None].to(device)
    mean = encoder(x)["mean"][0].float().cpu().numpy()
    return np.ascontiguousarray(mean.transpose(1, 2, 3, 0).reshape(4096, 8))


def load_original(root, record):
    path = Path(record["target_path"])
    path = path if path.is_absolute() else Path(root)/path
    # Same mean loading and spatial flattening as dataloader.py::load_target.
    with np.load(path, allow_pickle=False) as data:
        mean = data["mean"]
    if mean.shape != (8, 16, 16, 16) or mean.dtype != np.float32 or not np.isfinite(mean).all():
        raise ValueError(f"Expected finite float32 SS mean [8,16,16,16]: {path}")
    return np.ascontiguousarray(mean.transpose(1, 2, 3, 0).reshape(4096, 8))


def mse(a, b):
    return float(np.mean((np.asarray(a, dtype=np.float64)-b)**2))


def native_means(encoder, root, record, device):
    grid = voxelize(load_mesh(root, record["object_id"]), stock=True)
    means = {name: encode(encoder, rotate_grid(grid, matrix), device)
             for name, _, _, matrix in native_rotations()}
    stored = load_original(root, record)
    np.testing.assert_allclose(means["identity"], stored, rtol=1e-4, atol=1e-5,
                               err_msg=f"Target regeneration differs: {record['object_id']}")
    means["identity"] = stored  # Score the actual original training label.
    return means


def endpoint_scores(prediction, candidates):
    return {name: mse(prediction, mean) for name, mean in candidates.items()}
