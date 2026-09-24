import json
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
    default_collate,
)


from data_generation.general.pointmaps import depth_to_pointmap
from data_generation.general.sample_full_surface import validate_surface, sam_camera_transform, transform_points, transform_normals
from data_generation.general.surface_pool import select_surface_pool
from data_generation.general.sample_touch_patches import select_patch_indices


def load_data_config(path):
    with open(path) as file:
        return yaml.safe_load(file)


class TouchDataset(Dataset):
    def __init__(self, config, include_touch=True, oracle_point_frame=False, joint_pointmap=False):
        if isinstance(config, (str, Path)):
            config = load_data_config(config)

        dataset_config = config["dataset"]
        self.root = Path(dataset_config["root"])
        self.include_touch = include_touch
        self.joint_pointmap = joint_pointmap
        self.oracle_point_frame = oracle_point_frame
        self.point_source = config.get("touch", {}).get("source", "touch")
        self.include_normals = bool(config.get("touch", {}).get("include_normals", False))
        self.surface_pool_count = config.get("touch", {}).get("pool_points")
        self.surface_subset_seed = int(config.get("seed", 0))
        if (self.include_normals or self.surface_pool_count is not None) and self.point_source != "full_surface":
            raise ValueError("Normals and surface pools require full-surface data")
        if self.surface_pool_count is not None:
            self.surface_pool_count = int(self.surface_pool_count)
            if self.surface_pool_count < 1:
                raise ValueError("touch.pool_points must be positive")
        if oracle_point_frame and (not include_touch or self.point_source != "full_surface"):
            raise ValueError("Oracle point frame requires full-surface conditioning")
        if self.point_source not in ("touch", "full_surface", "touch_patches"):
            raise ValueError(f"Unknown point source: {self.point_source}")
        if include_touch and self.point_source == "touch":
            touch_config = config["touch"]
            self.contact_count = int(touch_config["contacts"]["count"])
            self.radius = float(touch_config["neighborhood"]["max_geodesic_distance"])
            self.points_per_contact = int(
                touch_config["point_sampling"]["points_per_contact"]
            )
            if self.contact_count < 1 or self.points_per_contact < 1:
                raise ValueError("Contact and point counts must be positive")
            if self.radius < 0:
                raise ValueError("Neighborhood radius must be non-negative")
        if include_touch and self.point_source == "touch_patches":
            self.contact_count = int(config['touch']['contacts']['count'])
            self.points_per_contact = int(config['touch']['point_sampling']['points_per_contact'])
            if (self.contact_count, self.points_per_contact) not in ((32, 256), (16, 512), (8, 1024)):
                raise ValueError('Adaptive touch uses 32x256, 16x512, or 8x1024 (8192 total)')

        split = dataset_config["split"]
        with open(self.resolve_path(dataset_config["split_file"])) as file:
            object_ids = set(json.load(file)[split])

        manifest_path = self.resolve_path(dataset_config["manifest"])
        self.records = []
        with open(manifest_path) as file:
            for line in file:
                record = json.loads(line)
                if record["object_id"] in object_ids:
                    self.records.append(record)

    def resolve_path(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def __len__(self):
        return len(self.records)

    def load_touch(self, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["format_version"]) != 4:
                raise ValueError(f"Unsupported touch format in {path}")

            available_contacts = len(data["offsets"]) - 1
            if available_contacts < self.contact_count:
                raise ValueError(
                    f"Requested {self.contact_count} contacts from {path}, "
                    f"but only {available_contacts} are available"
                )

            point_clouds = []
            for contact_index in range(self.contact_count):
                start, end = data["offsets"][contact_index : contact_index + 2]
                start, end = int(start), int(end)

                point_ids = data["point_ids"][start:end]
                center_indices = np.flatnonzero(
                    point_ids == data["center_point_ids"][contact_index]
                )
                if len(center_indices) != 1:
                    raise ValueError(
                        f"Contact {contact_index} in {path} does not have exactly one center"
                    )

                center_index = int(center_indices[0])
                eligible = np.flatnonzero(
                    data["geodesic_distance"][start:end] <= self.radius
                )
                others = eligible[eligible != center_index]
                priorities = data["keep_priority"][start + others]
                others = others[
                    np.argsort(priorities, kind="stable")[: self.points_per_contact - 1]
                ]
                selected = start + np.concatenate(([center_index], others))

                points = data["points_local"][selected]
                rotation = data["R_camera_from_local"][contact_index]
                center = data["centers_camera"][contact_index]
                point_clouds.append(points @ rotation.T + center)

        points = np.concatenate(point_clouds).astype(np.float32, copy=False)
        if len(points) == 0 or not np.isfinite(points).all():
            raise ValueError(f"Touch point cloud in {path} is empty or non-finite")
        return np.ascontiguousarray(points)

    def load_target(self, path):
        with np.load(path, allow_pickle=False) as data:
            mean = data["mean"]

        if mean.shape != (8, 16, 16, 16):
            raise ValueError(f"Expected target [8,16,16,16] in {path}, got {mean.shape}")
        if mean.dtype != np.float32 or not np.isfinite(mean).all():
            raise ValueError(f"Target in {path} must be finite float32")

        return np.ascontiguousarray(mean.transpose(1, 2, 3, 0).reshape(4096, 8))

    def load_touch_patches(self, path):
        with np.load(path, allow_pickle=False) as data:
            indices = select_patch_indices(data, self.contact_count, self.points_per_contact)
            return np.ascontiguousarray(data['points_camera'][indices])

    def load_full_surface(self, path, include_normals=False):
        with np.load(path, allow_pickle=False) as data:
            validate_surface(data, require_normals=include_normals)
            points = data["points_camera"]
            if include_normals:
                return np.ascontiguousarray(points), np.ascontiguousarray(data["normals_camera"])
        return np.ascontiguousarray(points)

    def load_surface_pool(self, record):
        # Backfilled datasets keep their original manifests; the pool is beside mesh.npz.
        path = (self.resolve_path(record["surface_pool_path"]) if record.get("surface_pool_path")
                else self.resolve_path(record["mesh_path"]).with_name("surface_pool.npz"))
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run data_generation/general/backfill_normals.py")
        object_seed = int(hashlib.sha256(record["object_id"].encode()).hexdigest()[:8], 16)
        # Same fixed random subset for all views of an object and for every epoch.
        with np.load(path, allow_pickle=False) as pool:
            points, normals, _ = select_surface_pool(pool, self.surface_pool_count,
                                                     [self.surface_subset_seed, object_seed])
        transform = sam_camera_transform(self.resolve_path(record["camera_path"]))
        return transform_points(points, transform).astype(np.float32), transform_normals(normals, transform)

    def load_pointmap(self, record):
        if record.get("pointmap_path"):
            return np.load(self.resolve_path(record["pointmap_path"]), allow_pickle=False).astype(np.float32, copy=False)
        depth = np.load(self.resolve_path(record["depth_path"]), allow_pickle=False)
        with np.load(self.resolve_path(record["camera_path"]), allow_pickle=False) as camera:
            return depth_to_pointmap(depth, camera["K"])

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(self.resolve_path(record["image_path"])) as image:
            image = np.array(image.convert("RGBA"), dtype=np.uint8)

        pointmap = self.load_pointmap(record)
        target_shape = self.load_target(self.resolve_path(record["target_path"]))

        sample = {
            "image": torch.from_numpy(image.copy()),
            "pointmap": torch.from_numpy(np.ascontiguousarray(pointmap)),
            "target_shape": torch.from_numpy(target_shape),
            "sample_id": record["sample_id"],
        }
        if self.include_touch:
            if self.point_source == "full_surface":
                if self.surface_pool_count is not None:
                    touch_xyz, normals = self.load_surface_pool(record)
                elif self.include_normals:
                    touch_xyz, normals = self.load_full_surface(self.resolve_path(record["full_surface_path"]), True)
                else:
                    touch_xyz = self.load_full_surface(self.resolve_path(record["full_surface_path"]))
                if self.include_normals:
                    sample["touch_normals"] = torch.from_numpy(np.ascontiguousarray(normals))
            elif self.point_source == 'touch_patches':
                touch_xyz = self.load_touch_patches(self.resolve_path(record['touch_path']))
                sample['touch_patch_count'] = self.contact_count
                if self.joint_pointmap:
                    from data_generation.general.sample_touch_patches import saved_joint_points
                    with np.load(self.resolve_path(record['touch_path']), allow_pickle=False) as data:
                        sample['joint_xyz'] = torch.from_numpy(saved_joint_points(data, self.contact_count))
                        for key in ('pointmap_scale', 'pointmap_shift'):
                            sample['joint_' + key] = torch.from_numpy(data['joint_' + key].copy())
            else:
                touch_xyz = self.load_touch(self.resolve_path(record["touch_path"]))
            sample["touch_xyz"] = torch.from_numpy(touch_xyz)
            if self.oracle_point_frame:
                with np.load(self.resolve_path(record["camera_path"]), allow_pickle=False) as camera:
                    transform = np.diag([-1., -1., 1., 1.]) @ camera["T_camera_from_object"]
                if transform.shape != (4, 4) or not np.isfinite(transform).all():
                    raise ValueError("Invalid camera transform")
                sample["object_from_camera"] = torch.from_numpy(
                    np.linalg.inv(transform).astype(np.float32)
                )
        return sample


def collate_touch_batch(samples):
    if "touch_xyz" not in samples[0]:
        return default_collate(samples)

    touch_xyz = [sample["touch_xyz"] for sample in samples]
    lengths = torch.tensor([len(points) for points in touch_xyz])

    batch = default_collate(
        [
            {key: value for key, value in sample.items() if key not in ("touch_xyz", "touch_normals")}
            for sample in samples
        ]
    )
    batch["touch_xyz"] = pad_sequence(touch_xyz, batch_first=True)
    if "touch_normals" in samples[0]:
        normals = [sample["touch_normals"] for sample in samples]
        if any(normal.shape != points.shape for normal, points in zip(normals, touch_xyz)):
            raise ValueError("Normals must align with surface points")
        batch["touch_normals"] = pad_sequence(normals, batch_first=True)
    batch["touch_mask"] = (
        torch.arange(batch["touch_xyz"].shape[1])[None] < lengths[:, None]
    )
    return batch


class DistributedEvalSampler(Sampler):
    """Shard validation data across ranks without adding duplicate samples."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = (
            dist.get_world_size() if num_replicas is None else num_replicas
        )
        self.rank = dist.get_rank() if rank is None else rank

        shard_size, remainder = divmod(len(dataset), self.num_replicas)
        begin = self.rank * shard_size + min(self.rank, remainder)
        end = begin + shard_size + int(self.rank < remainder)
        self.indices = range(begin, end)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def build_dataloader(
    config,
    batch_size,
    num_workers,
    shuffle=True,
    distributed=False,
    include_touch=True,
    oracle_point_frame=False,
    joint_pointmap=False,
):
    if isinstance(config, (str, Path)):
        config = load_data_config(config)

    dataset = TouchDataset(config, include_touch=include_touch, oracle_point_frame=oracle_point_frame,
                           joint_pointmap=joint_pointmap)
    sampler = None
    rank = 0

    if distributed:
        rank = dist.get_rank()
        if shuffle:
            sampler = DistributedSampler(
                dataset,
                shuffle=True,
                seed=int(config.get("seed", 0)),
            )
        else:
            sampler = DistributedEvalSampler(dataset)

    loader_options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collate_touch_batch,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "generator": torch.Generator().manual_seed(int(config.get("seed", 0)) + rank),
    }
    if distributed and num_workers > 0:
        loader_options["multiprocessing_context"] = "spawn"

    return DataLoader(**loader_options)
