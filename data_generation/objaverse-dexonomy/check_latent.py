import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

import os
os.environ["LIDRA_SKIP_INIT"] = "true"

from generate_target_latents import (
    DEFAULT_DATA_ROOT,
    DEFAULT_ENCODER_CHECKPOINT,
    load_encoder,
    load_normalized_mesh,
    validate_target,
    voxelize_mesh,
)

from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import (
    SparseStructureDecoderTdfyWrapper,
)


DEFAULT_DECODER_CHECKPOINT = Path(
    "/n/holylabs/qianqian_lab/Lab/mwakeham/"
    "visuotactile-objects/sam-3d-touch/checkpoints/hf/ss_decoder.ckpt"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_ENCODER_CHECKPOINT)
    parser.add_argument("--decoder-checkpoint", type=Path, default=DEFAULT_DECODER_CHECKPOINT)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--view-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-ply", action="store_true")

    return parser.parse_args()


def load_decoder(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    decoder = SparseStructureDecoderTdfyWrapper(
        out_channels=1,
        latent_channels=8,
        channels=[512, 128, 32],
        num_res_blocks=2,
        num_res_blocks_middle=2,
        reshape_input_to_cube=False,
        pretrained_ckpt_path=str(checkpoint_path),
    )

    return decoder.eval().to(device)


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def voxel_centers(occupancy):
    coords = np.argwhere(occupancy)
    return (coords.astype(np.float64) + 0.5) / 64 - 0.5


def occupancy_iou(first, second):
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()

    if union == 0:
        return 0

    return intersection / union


def alignment_distances(points_object, pointmap_points, camera_transform):
    points_camera = transform_points(points_object, camera_transform)
    tree = cKDTree(points_camera)
    distances, _ = tree.query(pointmap_points)

    return points_camera, distances


def print_alignment(name, distances):
    half_diagonal = np.sqrt(3) / (2 * 64)
    full_diagonal = np.sqrt(3) / 64

    print(f"{name}:")
    print(f"  median distance: {np.median(distances):.6f}")
    print(f"  95th percentile: {np.percentile(distances, 95):.6f}")
    print(f"  maximum distance: {distances.max():.6f}")
    print(f"  within half voxel diagonal: {(distances <= half_diagonal).mean():.4f}")
    print(f"  within one voxel diagonal: {(distances <= full_diagonal).mean():.4f}")


def save_alignment_ply(
    pointmap_points,
    target_points,
    decoded_points,
    output_path,
):
    points = np.concatenate(
        [
            pointmap_points,
            target_points,
            decoded_points,
        ],
        axis=0,
    )

    colors = np.concatenate(
        [
            np.tile([190, 190, 190, 255], (len(pointmap_points), 1)),
            np.tile([40, 120, 255, 255], (len(target_points), 1)),
            np.tile([255, 40, 120, 255], (len(decoded_points), 1)),
        ],
        axis=0,
    ).astype(np.uint8)

    trimesh.PointCloud(
        vertices=points,
        colors=colors,
    ).export(output_path)


def main():
    args = parse_args()

    view_id = f"{args.view_id:03d}"
    object_dir = args.data_root / "generated_data" / args.object_id
    view_dir = object_dir / "views" / view_id

    model_path = args.data_root / "objects" / args.object_id / "model.obj"
    object_transform_path = object_dir / "object_transform.npz"
    target_path = object_dir / "target_latent.npz"
    camera_path = view_dir / "camera.npz"
    pointmap_path = view_dir / "pointmap.npy"

    for path in [
        model_path,
        object_transform_path,
        target_path,
        camera_path,
        pointmap_path,
        args.encoder_checkpoint,
        args.decoder_checkpoint,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device(args.device)

    mesh = load_normalized_mesh(
        model_path=model_path,
        object_transform_path=object_transform_path,
    )

    bounds_min = mesh.vertices.min(axis=0)
    bounds_max = mesh.vertices.max(axis=0)
    center = (bounds_min + bounds_max) / 2
    extent = bounds_max - bounds_min

    occupancy = voxelize_mesh(mesh)
    input_occupancy = occupancy[0].numpy().astype(bool)

    saved_mean = validate_target(target_path)

    encoder = load_encoder(
        checkpoint_path=args.encoder_checkpoint,
        device=device,
    )

    with torch.inference_mode():
        output = encoder(occupancy.unsqueeze(0).to(device))
        recomputed_mean = output["mean"][0].float().cpu().numpy()

    latent_error = np.max(np.abs(saved_mean - recomputed_mean))

    if not np.allclose(
        saved_mean,
        recomputed_mean,
        rtol=1e-4,
        atol=1e-5,
    ):
        raise ValueError(
            f"Saved target does not match recomputed target: "
            f"max error {latent_error}"
        )

    del encoder

    if device.type == "cuda":
        torch.cuda.empty_cache()

    decoder = load_decoder(
        checkpoint_path=args.decoder_checkpoint,
        device=device,
    )

    mean_tensor = torch.from_numpy(saved_mean).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = decoder(mean_tensor)
        decoded_occupancy = (
            logits[0, 0].float().cpu().numpy() > 0
        )

    if not decoded_occupancy.any():
        raise ValueError("Decoder produced no occupied voxels")

    iou = occupancy_iou(
        input_occupancy,
        decoded_occupancy,
    )

    with np.load(camera_path, allow_pickle=False) as camera:
        T_camera_from_object = camera[
            "T_camera_from_object"
        ].astype(np.float64)

    if T_camera_from_object.shape != (4, 4):
        raise ValueError("Invalid T_camera_from_object")

    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    T_sam_from_object = flip @ T_camera_from_object

    pointmap = np.load(pointmap_path).astype(np.float64)
    valid_pointmap = np.isfinite(pointmap).all(axis=-1)
    pointmap_points = pointmap[valid_pointmap]

    if len(pointmap_points) == 0:
        raise ValueError("Pointmap contains no valid points")

    target_points_object = voxel_centers(input_occupancy)
    decoded_points_object = voxel_centers(decoded_occupancy)

    target_points_camera, target_distances = alignment_distances(
        points_object=target_points_object,
        pointmap_points=pointmap_points,
        camera_transform=T_sam_from_object,
    )

    decoded_points_camera, decoded_distances = alignment_distances(
        points_object=decoded_points_object,
        pointmap_points=pointmap_points,
        camera_transform=T_sam_from_object,
    )

    print()
    print(f"Object: {args.object_id}")
    print(f"View: {view_id}")
    print(f"Normalized center: {center}")
    print(f"Normalized extent: {extent}")
    print(f"Input occupied voxels: {input_occupancy.sum()}")
    print(f"Decoded occupied voxels: {decoded_occupancy.sum()}")
    print(f"Target mean shape: {saved_mean.shape}")
    print(f"Target mean range: {saved_mean.min():.6f} to {saved_mean.max():.6f}")
    print(f"Recomputed latent max error: {latent_error:.8f}")
    print(f"Encoder/decoder occupancy IoU: {iou:.6f}")
    print()

    print_alignment(
        "Pointmap -> input target occupancy",
        target_distances,
    )

    print()

    print_alignment(
        "Pointmap -> decoded target occupancy",
        decoded_distances,
    )

    if args.save_ply:
        output_path = view_dir / "target_latent_check.ply"

        save_alignment_ply(
            pointmap_points=pointmap_points,
            target_points=target_points_camera,
            decoded_points=decoded_points_camera,
            output_path=output_path,
        )

        print()
        print(f"Saved alignment PLY: {output_path}")
        print("PLY colors: pointmap gray, input target blue, decoded target pink")


if __name__ == "__main__":
    main()