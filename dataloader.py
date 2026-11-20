import json
import hashlib
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
    RandomSampler,
    SequentialSampler,
    default_collate,
)


from data_generation.general.pointmaps import depth_to_pointmap
from data_generation.general.sample_full_surface import validate_surface, sam_camera_transform, transform_points, transform_normals
from data_generation.general.surface_pool import select_surface_pool
from data_generation.general.sample_touch_patches import select_patch_indices


def load_data_config(path):
    with open(path) as file:
        return yaml.safe_load(file)


def load_camera_target_transform(target_path, camera_path):
    with np.load(target_path, allow_pickle=False) as data:
        if str(data["coordinate_frame"]) != "normalized_sam_camera":
            raise ValueError(f"Pose training requires a camera-oriented target: {target_path}")
        camera_from_target = data["T_camera_from_target"].astype(np.float64)
        target_from_object = data["T_target_from_object"].astype(np.float64)
    for matrix in (camera_from_target, target_from_object):
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)):
            raise ValueError(f"Invalid target transform: {target_path}")
    scale = camera_from_target[0, 0]
    if scale <= 0 or not np.allclose(camera_from_target[:3, :3], np.eye(3) * scale,
                                    atol=1e-6, rtol=1e-5):
        raise ValueError(f"Camera-oriented target must have identity rotation and positive isotropic scale: {target_path}")
    if not np.allclose(camera_from_target @ target_from_object,
                       sam_camera_transform(camera_path), atol=1e-5, rtol=1e-5):
        raise ValueError(f"Target and camera transforms disagree: {target_path}")
    return np.ascontiguousarray(camera_from_target, dtype=np.float32)


class TouchDataset(Dataset):
    def __init__(self, config, include_touch=True, oracle_point_frame=False, joint_pointmap=False,
                 include_pose_targets=False, include_hand=False):
        if isinstance(config, (str, Path)):
            config = load_data_config(config)

        dataset_config = config["dataset"]
        self.root = Path(dataset_config["root"])
        self.include_touch = include_touch
        self.include_pose_targets = include_pose_targets
        self.target_frame = dataset_config.get("target_frame")
        if include_pose_targets and dataset_config.get("target_frame") != "camera":
            raise ValueError("Pose training requires dataset.target_frame: camera")
        self.joint_pointmap = joint_pointmap
        self.oracle_point_frame = oracle_point_frame
        self.point_source = config.get("touch", {}).get("source", "touch")
        self.include_normals = bool(config.get("touch", {}).get("include_normals", False))
        self.surface_pool_count = config.get("touch", {}).get("pool_points")
        self.surface_subset_seed = int(config.get("seed", 0))
        # Grasp views: views whose object mask covers less than this share of the image are replaced by the next view
        # (same measure as select_grasp_views.py --min-object-visible-share, on the full-res mask). 0 = only SAM 3D's
        # own minimum (a 2x2 px mask box).
        self.min_object_mask_share = float(dataset_config.get("min_object_mask_share", 0))
        if (self.include_normals or self.surface_pool_count is not None) and self.point_source != "full_surface":
            raise ValueError("Normals and surface pools require full-surface data")
        if self.surface_pool_count is not None:
            self.surface_pool_count = int(self.surface_pool_count)
            if self.surface_pool_count < 1:
                raise ValueError("touch.pool_points must be positive")
        if oracle_point_frame and (not include_touch or self.point_source != "full_surface"):
            raise ValueError("Oracle point frame requires full-surface conditioning")
        if self.point_source not in ("touch", "full_surface", "touch_patches", "grasp_contact"):
            raise ValueError(f"Unknown point source: {self.point_source}")
        self.include_hand = include_hand
        if include_hand and (self.point_source != "grasp_contact" or not joint_pointmap):
            raise ValueError("Hand points need grasp_contact data with joint pointmap inputs")
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
        if include_touch and self.point_source == "grasp_contact":
            # Saved in farthest-point order, so the first N points are spread over the whole contact.
            self.contact_points = int(config["touch"].get("points", 8192))
            if self.contact_points < 1:
                raise ValueError("touch.points must be positive")
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
        # Optional cap (used for validation): the first N views of each object, by view id.
        max_views = dataset_config.get("max_views_per_object")
        if max_views:
            kept = {}
            for record in sorted(self.records, key=lambda r: (r["object_id"], r["view_id"])):
                kept.setdefault(record["object_id"], [])
                if len(kept[record["object_id"]]) < max_views:
                    kept[record["object_id"]].append(record["sample_id"])
            keep = {sample_id for ids in kept.values() for sample_id in ids}
            self.records = [record for record in self.records if record["sample_id"] in keep]

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

    def load_grasp_contact(self, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data["data_kind"]) != "grasp_contact" or str(data["coordinate_frame"]) != "sam_camera":
                raise ValueError(f"Unsupported grasp-contact touch file: {path}")
            points = data["points_camera"][:self.contact_points]
        if len(points) == 0 or not np.isfinite(points).all():
            raise ValueError(f"Grasp-contact touch in {path} is empty or non-finite")
        return np.ascontiguousarray(points, dtype=np.float32)

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

    def load_rgba(self, record):
        """RGBA as SAM 3D takes it: alpha is the object mask. Grasp renders show the hand as well, so their alpha
        is replaced by object_mask.png (the hand stays visible in RGB, outside the mask)."""
        with Image.open(self.resolve_path(record["image_path"])) as image:
            image = np.array(image.convert("RGBA"), dtype=np.uint8)
        if record.get("object_mask_path"):
            with Image.open(self.resolve_path(record["object_mask_path"])) as mask:
                image[..., 3] = np.asarray(mask.convert("L"))
        return image

    def __getitem__(self, index):
        # A view whose object barely shows (e.g. a sliver between fingers) cannot be cropped by SAM 3D, or is under
        # the dataset's minimum visible share: it is replaced by the next view that passes. sample["skipped_views"]
        # is 1 when the requested view was replaced, so train.py can report the share of views skipped.
        skipped = 0
        requested = index
        while True:
            record = self.records[index]
            image = self.load_rgba(record)
            if mask_ok(image[..., 3], self.min_object_mask_share):
                break
            skipped += 1
            if skipped >= len(self):
                raise ValueError("No view passes the minimum object mask size")
            index = (index + 1) % len(self)

        pointmap = self.load_pointmap(record)
        target_shape = self.load_target(self.resolve_path(record["target_path"]))

        sample = {
            "image": torch.from_numpy(image.copy()),
            "pointmap": torch.from_numpy(np.ascontiguousarray(pointmap)),
            "target_shape": torch.from_numpy(target_shape),
            "sample_id": record["sample_id"],
            "skipped_views": int(index != requested),
        }
        if self.include_pose_targets:
            sample["camera_from_target"] = torch.from_numpy(load_camera_target_transform(
                self.resolve_path(record["target_path"]), self.resolve_path(record["camera_path"]),
            ))
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
            elif self.point_source == "grasp_contact":
                touch_xyz = self.load_grasp_contact(self.resolve_path(record["touch_path"]))
                if self.joint_pointmap:
                    # Saved by make_contact_touch.py --joint-pointmap: 1024 pointmap + 7168 touch points.
                    with np.load(self.resolve_path(record["touch_path"]), allow_pickle=False) as data:
                        if "joint_points_pre_encoder" not in data.files:
                            raise ValueError("Saved joint inputs are missing; rerun make_contact_touch.py with --joint-pointmap")
                        sample["joint_xyz"] = torch.from_numpy(np.ascontiguousarray(data["joint_points_pre_encoder"]))
                        for key in ("pointmap_scale", "pointmap_shift"):
                            sample["joint_" + key] = torch.from_numpy(data["joint_" + key].copy())
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
        if self.include_hand:
            # Saved by make_hand_points.py --joint-pointmap beside the view's touch file: 1024 pointmap + 7168 hand points.
            path = self.resolve_path(record["touch_path"]).with_name("hand_points.npz")
            with np.load(path, allow_pickle=False) as data:
                if "joint_points_pre_encoder" not in data.files:
                    raise ValueError(f"Saved joint hand inputs are missing in {path}; rerun make_hand_points.py with --joint-pointmap")
                sample["hand_xyz"] = torch.from_numpy(np.ascontiguousarray(data["joint_points_pre_encoder"]))
                for key in ("pointmap_scale", "pointmap_shift"):
                    sample["hand_joint_" + key] = torch.from_numpy(data["joint_" + key].copy())
        return sample


def mask_ok(alpha, min_share=0.0):
    """True when the object mask (nonzero alpha) covers at least min_share of the image and its box spans at least
    2 px each way, as SAM 3D's compute_mask_bbox requires."""
    ys, xs = np.nonzero(alpha)
    return (len(xs) > 0 and xs.max() - xs.min() >= 2 and ys.max() - ys.min() >= 2
            and len(xs) >= min_share * alpha.size)


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


class ResumableBatchSampler(BatchSampler):
    # Keep the full epoch length for checkpoint positions and validation timing.
    start_batch = 0

    def __iter__(self):
        return islice(super().__iter__(), self.start_batch, None)


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
    include_pose_targets=False,
    include_hand=False,
):
    if isinstance(config, (str, Path)):
        config = load_data_config(config)

    dataset = TouchDataset(config, include_touch=include_touch, oracle_point_frame=oracle_point_frame,
                           joint_pointmap=joint_pointmap, include_pose_targets=include_pose_targets,
                           include_hand=include_hand)
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

    generator = torch.Generator().manual_seed(int(config.get("seed", 0)) + rank)
    if sampler is None:
        sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)

    loader_options = {
        "dataset": dataset,
        "batch_sampler": ResumableBatchSampler(sampler, batch_size, drop_last=False),
        "num_workers": num_workers,
        "collate_fn": collate_touch_batch,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "generator": generator,
    }
    if distributed and num_workers > 0:
        loader_options["multiprocessing_context"] = "spawn"

    return DataLoader(**loader_options)
