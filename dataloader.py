import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def load_data_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class TouchDataset(Dataset):
    def __init__(self, config):
        if isinstance(config, (str, Path)):
            config = load_data_config(config)

        self.root = Path(config["dataset"]["root"])
        self.split = config["dataset"]["split"]
        self.records = []

        with open(self.path(config["dataset"]["split_file"])) as f:
            object_ids = set(json.load(f)[self.split])
        with open(self.path(config["dataset"]["manifest"])) as f:
            for line in f:
                record = json.loads(line)
                if record["object_id"] in object_ids:
                    self.records.append(record)

        touch = config["touch"]
        self.contact_count = touch["contacts"]["count"]
        self.shuffle_contacts = touch["contacts"]["shuffle_after_selection"]
        self.radius = touch["neighborhood"]["max_geodesic_distance"]
        self.points_per_contact = touch["point_sampling"]["points_per_contact"]

    def path(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def __len__(self):
        return len(self.records)

    def load_touch(self, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["format_version"]) != 4:
                raise ValueError(f"Unsupported touch format in {path}")

            contacts = []
            for contact in range(self.contact_count):
                start, end = data["offsets"][contact : contact + 2]
                point_ids = data["point_ids"][start:end]
                eligible = np.flatnonzero(data["geodesic_distance"][start:end] <= self.radius)
                center = np.flatnonzero(point_ids == data["center_point_ids"][contact])

                if len(center) != 1:
                    raise ValueError(f"Contact {contact} in {path} does not have exactly one center")

                center = center[0]
                others = eligible[eligible != center]
                if len(others) + 1 < self.points_per_contact:
                    raise ValueError(f"Contact {contact} in {path} has fewer than {self.points_per_contact} points")

                priorities = data["keep_priority"][start + others]
                others = others[np.argsort(priorities)[: self.points_per_contact - 1]]
                selected = start + np.concatenate(([center], others))

                points = data["points_local"][selected]
                rotation = data["R_camera_from_local"][contact]
                center_camera = data["centers_camera"][contact]
                contacts.append(points @ rotation.T + center_camera)

        touch_xyz = np.stack(contacts).astype(np.float32)
        if not np.isfinite(touch_xyz).all():
            raise ValueError(f"Non-finite touch points in {path}")
        if self.shuffle_contacts:
            touch_xyz = touch_xyz[np.random.permutation(len(touch_xyz))]
        return np.ascontiguousarray(touch_xyz)

    def load_target(self, path):
        with np.load(path, allow_pickle=False) as data:
            mean = data["mean"]
        if mean.shape != (8, 16, 16, 16) or mean.dtype != np.float32 or not np.isfinite(mean).all():
            raise ValueError(f"Expected a finite float32 [8,16,16,16] target in {path}")
        return np.ascontiguousarray(mean.transpose(1, 2, 3, 0).reshape(4096, 8))

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.path(record["image_path"])) as image:
            image = np.array(image.convert("RGBA"), dtype=np.uint8)
        pointmap = np.load(self.path(record["pointmap_path"]), allow_pickle=False).astype(np.float32)
        touch_xyz = self.load_touch(self.path(record["touch_path"]))
        target = self.load_target(self.path(record["target_path"]))

        return {
            "image": torch.from_numpy(image.copy()),
            "pointmap": torch.from_numpy(pointmap),
            "touch_xyz": torch.from_numpy(touch_xyz),
            "target_shape": torch.from_numpy(target),
            "sample_id": record["sample_id"],
        }


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(config, batch_size, num_workers, shuffle=True):
    if isinstance(config, (str, Path)):
        config = load_data_config(config)
    generator = torch.Generator().manual_seed(config.get("seed", 0))
    return DataLoader(
        TouchDataset(config), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker, generator=generator,
    )
