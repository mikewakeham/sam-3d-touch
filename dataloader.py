import json
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


def load_data_config(path):
    with open(path) as file:
        return yaml.safe_load(file)


class TouchDataset(Dataset):
    def __init__(self, config, include_touch=True):
        if isinstance(config, (str, Path)):
            config = load_data_config(config)

        dataset_config = config["dataset"]
        self.root = Path(dataset_config["root"])
        self.include_touch = include_touch
        if include_touch:
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

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(self.resolve_path(record["image_path"])) as image:
            image = np.array(image.convert("RGBA"), dtype=np.uint8)

        pointmap = np.load(
            self.resolve_path(record["pointmap_path"]), allow_pickle=False
        ).astype(np.float32, copy=False)
        target_shape = self.load_target(self.resolve_path(record["target_path"]))

        sample = {
            "image": torch.from_numpy(image.copy()),
            "pointmap": torch.from_numpy(np.ascontiguousarray(pointmap)),
            "target_shape": torch.from_numpy(target_shape),
            "sample_id": record["sample_id"],
        }
        if self.include_touch:
            touch_xyz = self.load_touch(self.resolve_path(record["touch_path"]))
            sample["touch_xyz"] = torch.from_numpy(touch_xyz)
        return sample


def collate_touch_batch(samples):
    if "touch_xyz" not in samples[0]:
        return default_collate(samples)

    touch_xyz = [sample["touch_xyz"] for sample in samples]
    lengths = torch.tensor([len(points) for points in touch_xyz])

    batch = default_collate(
        [
            {key: value for key, value in sample.items() if key != "touch_xyz"}
            for sample in samples
        ]
    )
    batch["touch_xyz"] = pad_sequence(touch_xyz, batch_first=True)
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
):
    if isinstance(config, (str, Path)):
        config = load_data_config(config)

    dataset = TouchDataset(config, include_touch=include_touch)
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
