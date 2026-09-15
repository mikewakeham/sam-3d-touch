"""Encode identical shapes at different rotations; measure target latent MSE."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("LIDRA_SKIP_INIT", "true")
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from omegaconf import OmegaConf
from dataloader import load_data_config
from experiments.coordinate_system.scripts.rotation_loss.rotation_utils import (
    select_groups, native_rotations, load_mesh, voxelize, load_encoder, encode,
    load_original, mse, rotate_grid, rotation,
)


def run(args):
    data = load_data_config(args.data_config)
    root = Path(data["dataset"]["root"])
    groups = select_groups(data, args.train_objects, args.val_objects, 2, args.seed)
    objects = {}
    for split, records in groups:
        for record in records:
            objects[record["object_id"]] = (split, record)
    examples = set([args.example_object] if args.example_object else list(objects)[:1])
    if args.example_object and args.example_object not in objects:
        objects[args.example_object] = ("example", {
            "object_id": args.example_object,
            "target_path": f"generated_data/{args.example_object}/target_latent.npz"})

    config = OmegaConf.load(args.pipeline_config)
    generator = OmegaConf.load(args.pipeline_config.parent/config.ss_generator_config_path)
    sigma = float(generator["module"]["generator"]["backbone"].get("sigma_min", 0.))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    encoder = load_encoder(args.encoder_checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"examples").mkdir(exist_ok=True)
    checks, references = [], {}
    start = time.monotonic()
    with (args.output_dir/"measurements.csv").open("w", newline="") as file:
        fields = ["object_id", "split", "part", "axis", "degrees", "scale",
                  "latent_mse", "occupied", "inverse_vertex_error", "inverse_grid_iou",
                  "mesh_vs_exact_grid_iou", "analytical_endpoint_velocity_mse_t_0.2",
                  "analytical_endpoint_velocity_mse_t_0.5", "analytical_endpoint_velocity_mse_t_0.8"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, (oid, (split, record)) in enumerate(objects.items()):
            mesh = load_mesh(root, oid)
            original = voxelize(mesh, stock=True)
            zero = encode(encoder, original, device)
            stored = load_original(root, record)
            np.testing.assert_allclose(zero, stored, rtol=1e-4, atol=1e-5,
                                       err_msg=f"Target regeneration differs: {oid}")
            repeat_mse = mse(encode(encoder, original, device), zero)
            checks.append({"object_id": oid, "split": split, "part": "native",
                           "zero_repeat_mse": repeat_mse,
                           "stored_target_max_abs_error": float(np.max(np.abs(zero-stored)))})
            scale = .45/np.linalg.norm(mesh.vertices, axis=1).max()
            padded = mesh.copy()
            padded.vertices = np.asarray(mesh.vertices)*scale
            padded_grid = voxelize(padded)
            padded_zero = encode(encoder, padded_grid, device)
            checks.append({"object_id": oid, "split": split, "part": "padded",
                           "zero_repeat_mse": mse(encode(encoder, padded_grid, device), padded_zero)})
            references[oid] = (split, zero, padded_zero)

            variants = [("native", axis or "none", angle, matrix, 1.)
                        for _, axis, angle, matrix in native_rotations()]
            variants += [("padded", "none", 0, np.eye(3), scale)]
            variants += [("padded", axis, angle, np.array(rotation(axis, angle)), scale)
                         for axis in "xyz" for angle in args.angles if angle != 0]
            z90_mse = None
            for part, axis, angle, matrix, factor in variants:
                vertex_error = inverse_iou = discretization_iou = None
                if part == "native":
                    grid = rotate_grid(original, matrix)
                    restored = rotate_grid(grid, matrix.T)
                    np.testing.assert_array_equal(restored, original)
                    inverse_iou = 1.
                    reference = zero
                else:
                    transformed = padded.copy()
                    transformed.vertices = np.asarray(padded.vertices)@matrix.T
                    vertex_error = float(np.max(np.abs(transformed.vertices@matrix-padded.vertices)))
                    grid = padded_grid if angle == 0 else voxelize(transformed)
                    reference = padded_zero
                    if angle in (90, 180):
                        exact = rotate_grid(padded_grid, np.rint(matrix).astype(int))
                        discretization_iou = float(np.count_nonzero(grid & exact)/np.count_nonzero(grid | exact))
                mean = reference if angle == 0 else encode(encoder, grid, device)
                distance = mse(mean, reference)
                row = dict(object_id=oid, split=split, part=part, axis=axis, degrees=angle,
                           scale=factor, latent_mse=distance, occupied=int(grid.sum()),
                           inverse_vertex_error=vertex_error, inverse_grid_iou=inverse_iou,
                           mesh_vs_exact_grid_iou=discretization_iou)
                for t in (.2, .5, .8):
                    row[f"analytical_endpoint_velocity_mse_t_{t}"] = distance/(1-(1-sigma)*t)**2
                writer.writerow(row)
                if part == "native" and axis == "z" and angle == 90:
                    z90_mse = distance
            file.flush()
            if oid in examples:
                rotated = rotate_grid(original, np.rint(rotation("z", 90)).astype(int))
                np.savez_compressed(args.output_dir/"examples"/f"{oid}.npz",
                    original=original, rotated=rotated,
                    restored=rotate_grid(rotated, np.rint(rotation("z", -90)).astype(int)),
                    latent_mse=z90_mse, restored_latent_mse=repeat_mse)
            elapsed = time.monotonic()-start
            print(f"Encoded {index+1}/{len(objects)} objects; {elapsed:.1f}s elapsed", flush=True)

    # Different-object reference uses the already computed zero means.
    cross_shape = []
    for split in ("train", "val"):
        ids = [oid for oid, value in references.items() if value[0] == split]
        for i, oid in enumerate(ids):
            other = ids[(i+1) % len(ids)]
            for part, column in (("native", 1), ("padded", 2)):
                cross_shape.append({"split": split, "object_id": oid, "other_object_id": other,
                                    "part": part, "latent_mse": mse(references[oid][column], references[other][column])})
    report = {"complete": True, "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "sigma_min": sigma, "selected_objects": [{"object_id": oid, "split": split} for oid, (split, _) in objects.items()],
              "checks": checks, "different_object_reference": cross_shape,
              "elapsed_seconds": time.monotonic()-start,
              "definitions": {"latent_mse": "MSE between deterministic SS means; not observed checkpoint velocity loss",
                              "native": "Original target scale; exact occupancy permutations",
                              "padded": "One fixed radius-0.45 scale per object for every angle"}}
    (args.output_dir/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Saved:", args.output_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=REPO/"configs/data_full_surface.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPO/"checkpoints/hf/pipeline.yaml")
    parser.add_argument("--encoder-checkpoint", type=Path, default=REPO/"checkpoints/hf/ss_encoder.ckpt")
    parser.add_argument("--train-objects", type=int, default=16)
    parser.add_argument("--val-objects", type=int, default=16)
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 5, 15, 30, 60, 90, 180])
    parser.add_argument("--example-object")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
