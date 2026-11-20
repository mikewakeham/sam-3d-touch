"""Settled-hand surface points per view (<view>/hand_points.npz), plus joint hand-pointmap inputs with --joint-pointmap."""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import fpsample
import numpy as np
import trimesh

from sample_full_surface import classify_visibility, sam_camera_transform, transform_points

NAME = "hand_points"
FORMAT_VERSION = 1
# Only the 1024 chosen hand pixels are kept (in joint_points_camera), not every hand pixel.
JOINT_KEYS = ("joint_points_camera", "joint_points_pre_encoder", "joint_pointmap_indices",
              "joint_pointmap_scale", "joint_pointmap_shift", "joint_source_sha256")


def fps(points, count, rng):
    """As make_contact_touch.fps (not imported: that module needs Warp). Farthest-point sampling with fpsample
    (bucket QuickFPS, h=7), from a seeded random start."""
    count = min(count, len(points))
    if count < 2:
        return np.arange(count)
    h = min(7, int(np.log2(len(points))))  # fpsample requires 2**h <= number of points
    return fpsample.bucket_fps_kdline_sampling(points, count, h=h, start_idx=int(rng.integers(len(points)))).astype(np.int64)


def posed_hand(tmpl, view):
    """The rendered hand in the view's settled pose, in the normalized object frame: x = a * (R p + t) + b per body,
    as make_contact_touch.view_contact_touch and view_contact_touch.py pose it. Returns vertices, faces and each
    face's index into tmpl['names']."""
    a, b = float(view["hand_a"]), np.asarray(view["hand_b"], float)
    xmat = np.asarray(view["body_xmat"], float).reshape(-1, 3, 3)
    xpos = np.asarray(view["body_xpos"], float)
    vertices, faces, part = [], [], []
    offset = 0
    for i in range(len(tmpl["names"])):
        bid = int(tmpl[f"{i}_body_id"])
        v = tmpl[f"{i}_vertices"].astype(np.float64) @ xmat[bid].T * a + a * xpos[bid] + b
        f = tmpl[f"{i}_faces"].astype(np.int64)
        vertices.append(v)
        faces.append(f + offset)
        part.append(np.full(len(f), i, np.int16))
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces), np.concatenate(part)


def view_hand_points(view, tmpl, args, rng):
    vertices, faces, face_part = posed_hand(tmpl, view)
    if not np.isfinite(vertices).all():
        raise ValueError("Non-finite posed hand")
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    points, face = trimesh.sample.sample_surface(mesh, args.hand_samples, seed=int(rng.integers(2**31)))
    chosen = fps(points, args.points, rng)
    return points[chosen], face_part[face[chosen]]


def has_joint(path):
    with np.load(path, allow_pickle=False) as saved:
        return all(key in saved.files for key in JOINT_KEYS)


def save_view(record, root, path, points, part, names, args):
    transform = sam_camera_transform(root / record["camera_path"])
    with np.load(root / record["camera_path"], allow_pickle=False) as camera:
        K, T = camera["K"], camera["T_camera_from_object"]
    with np.load(root / record["full_surface_path"], allow_pickle=False) as surface:
        vis_tol = float(surface["visibility_tolerance"])
    depth = np.load(root / record["depth_path"], allow_pickle=False)
    arrays = dict(
        format_version=np.int64(FORMAT_VERSION), data_kind="hand_points", coordinate_frame="sam_camera",
        points_camera=transform_points(points, transform).astype(np.float32),
        points_object=points.astype(np.float32), part_index=part.astype(np.int16), part_names=np.array(names),
        # Rendered depth includes the hand: 1 = the point is the visible hand surface, 0 = hidden (e.g. behind the
        # object), -1 = outside the image.
        point_visibility=classify_visibility(points, K, T, depth, vis_tol), visibility_tolerance=vis_tol,
        hand_samples=np.int64(args.hand_samples), seed=np.int64(args.seed), numpy_version=np.__version__)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


JOINT_HAND_POINTS = 7168  # as the touch joint: 1024 pointmap + 7168 points


def hand_joint_arrays(pointmap, hand_mask, mesh_camera, normalize, denormalize):
    """1024 farthest-point hand pixels (full-resolution pointmap under the hand mask) then the first 7168 hand mesh
    points, in the pointmap's normalized frame; normalize/denormalize map camera <-> that frame."""
    import torch
    from sam3d_objects.model.backbone.dit.embedder.surface_encoder_utils import farthest_point_indices
    pixels = pointmap[hand_mask & np.isfinite(pointmap).all(axis=-1)].astype(np.float32)
    if not len(pixels):
        raise ValueError("No valid hand pixels in the pointmap")
    mesh_camera = mesh_camera[np.arange(JOINT_HAND_POINTS) % len(mesh_camera)].astype(np.float32)
    with torch.no_grad():
        normalized = normalize(torch.from_numpy(pixels))
        chosen = farthest_point_indices(normalized[None], min(1024, len(pixels)))[0]
        chosen = chosen[torch.arange(1024) % len(chosen)]
        cloud = torch.cat((normalized[chosen], normalize(torch.from_numpy(mesh_camera))))
    restored = denormalize(cloud)
    if not np.allclose(restored, np.concatenate((pixels[chosen.numpy()], mesh_camera)), atol=1e-5, rtol=1e-5):
        raise ValueError("Joint points do not round-trip to the hand pixels and hand points")
    return dict(joint_points_camera=np.asarray(restored, dtype=np.float32),
                joint_points_pre_encoder=cloud.float().numpy(),
                joint_pointmap_indices=chosen.numpy().astype(np.int32))


def save_joint(record, root, path, args):
    """make_contact_touch.save_joint with the hand's pointmap: the image and pointmap are preprocessed as training
    does (object-mask normalization), but the 1024 pointmap points come from the hand mask."""
    import hashlib
    import sys
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from torchvision.transforms import InterpolationMode, functional as TF
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dataloader import TouchDataset
    from evaluation.input_visualizations import joint_camera_points
    from generate_target_latents import checkpoint_sha256
    from make_touch_data import EmptyPointmapMask, joint_preprocessor
    from train import normalize_touch_to_pointmap_frame, preprocess_pointmap_batch

    config_path = args.pipeline_config.resolve()
    config = OmegaConf.load(config_path)
    sources = [root / record[key] for key in ("image_path", "camera_path", "depth_path", "hand_mask_path")]
    for key in ("pointmap_path", "object_mask_path"):
        if record.get(key):
            sources.append(root / record[key])
    sources.append(config_path)
    if config.get("ss_preprocessor") is None:
        sources.append(config_path.parent / config.ss_generator_config_path)
    fingerprint = hashlib.sha256("".join(checkpoint_sha256(source) for source in sources).encode()).hexdigest()
    dataset = object.__new__(TouchDataset)
    dataset.root = root
    rgba = dataset.load_rgba(record)  # alpha = object mask, as training
    pointmap = dataset.load_pointmap(record)
    mask = torch.from_numpy((rgba[..., 3] / 255).astype(np.float32))[None]
    mask = TF.resize(mask, pointmap.shape[:2], interpolation=InterpolationMode.NEAREST)
    if not (mask > .5).any():
        raise EmptyPointmapMask(f"{record['sample_id']}: Empty foreground mask at pointmap resolution")
    preprocessor = joint_preprocessor(str(config_path))
    inputs = preprocess_pointmap_batch(preprocessor, torch.from_numpy(rgba)[None], torch.from_numpy(pointmap)[None], "cpu")
    hand = torch.from_numpy((np.asarray(Image.open(root / record["hand_mask_path"]).convert("L")) > 127).astype(np.float32))[None]
    hand = TF.resize(hand, pointmap.shape[:2], interpolation=InterpolationMode.NEAREST)[0].numpy() > .5

    def normalize(points):
        mask = torch.ones(points.shape[0], dtype=torch.bool)[None]
        return normalize_touch_to_pointmap_frame(points[None], mask, inputs, preprocessor)[0]

    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved)
    arrays.pop("joint_pointmap_candidates_camera", None)  # from earlier versions that kept every pixel
    arrays.update(hand_joint_arrays(pointmap, hand, arrays["points_camera"], normalize,
                                    lambda cloud: joint_camera_points(cloud, inputs, preprocessor)))
    arrays.update(joint_pointmap_scale=inputs["pointmap_scale"][0].float().cpu().numpy(),
                  joint_pointmap_shift=inputs["pointmap_shift"][0].float().cpu().numpy(),
                  joint_source_sha256=fingerprint)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def make_object(task):
    args, records = task
    root = args.data_root
    tmpl = WORKER["template"]
    names = [str(n) for n in tmpl["names"]]
    outputs, failures, timings = [], [], []
    views = None
    for record in records:
        path = (root / record["full_surface_path"]).with_name(f"{NAME}.npz")
        if path.exists() and not args.overwrite:
            if args.joint_pointmap and not has_joint(path):
                try:
                    save_joint(record, root, path, args)
                except Exception as error:
                    failures.append(dict(sample_id=record["sample_id"], reason=f"joint: {type(error).__name__}: {error}"))
                    continue
            outputs.append(record["sample_id"])
            continue
        started = time.perf_counter()
        if views is None:
            views = json.loads((root / record["grasp_views_path"]).read_text())["views"]
        view = views[int(record["grasp_view_index"])]
        # One seed per view, so a view's points do not depend on which other views were (re)made in this run.
        rng = np.random.default_rng([args.seed, int(record["sample_id"].encode().hex(), 16) % 2**32])
        try:
            points, part = view_hand_points(view, tmpl, args, rng)
            save_view(record, root, path, points, part, names, args)
            if args.joint_pointmap:
                save_joint(record, root, path, args)
        except Exception as error:  # one bad view must not stop the run; it is listed
            failures.append(dict(sample_id=record["sample_id"], reason=f"error: {type(error).__name__}: {error}"))
            continue
        outputs.append(record["sample_id"])
        timings.append(dict(sample_id=record["sample_id"], seconds=time.perf_counter() - started))
    return records[0]["object_id"], outputs, failures, timings


WORKER = {}


def format_time(seconds):
    """As make_data.format_time (not imported: make_data needs the render env)."""
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours:02d}:{rest // 60:02d}:{rest % 60:02d}"


def init_worker(template_path):
    with np.load(template_path, allow_pickle=False) as z:
        WORKER["template"] = dict(z)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", default="generated_data/samples.jsonl", help="Views to process")
    parser.add_argument("--points", type=int, default=8192, help="Points per view")
    parser.add_argument("--hand-samples", type=int, default=65536, help="Hand-surface samples before farthest-point sampling")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--object-id", action="append", help="Process only these objects (repeatable)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--joint-pointmap", action="store_true", help="Also save fixed 1024 hand-pointmap + 7168 "
                        "hand-mesh inputs; existing hand files only get these added")
    parser.add_argument("--pipeline-config", type=Path, default=Path("checkpoints/hf/pipeline.yaml"))
    args = parser.parse_args(argv)
    if args.points < 1 or args.hand_samples < args.points:
        parser.error("Need --points >= 1 and --hand-samples >= --points")
    args.data_root = args.data_root.resolve()
    generated = args.data_root / "generated_data"
    records = [json.loads(line) for line in (args.data_root / args.manifest).read_text().splitlines() if line.strip()]
    groups = {}
    for r in records:
        if args.object_id is None or r["object_id"] in args.object_id:
            groups.setdefault(r["object_id"], []).append(r)
    if not groups:
        raise ValueError("No views to process")
    template = args.data_root / "grasp_views" / "hand_template.npz"
    print(f"Processing {len(groups)} objects ({sum(map(len, groups.values()))} views) on {args.workers} workers", flush=True)
    outputs, failures, timings = [], [], []
    started = time.monotonic()
    tasks = [(args, rows) for rows in groups.values()]
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(template,)) as pool:
        for i, (object_id, out, fail, timing) in enumerate(pool.imap_unordered(make_object, tasks), 1):
            outputs += out
            failures += fail
            timings += timing
            elapsed = time.monotonic() - started
            eta = elapsed / i * (len(tasks) - i)
            print(f"[{i}/{len(tasks)}] elapsed={format_time(elapsed)} eta={format_time(eta)} "
                  f"{object_id}: {len(out)} views, {len(fail)} failed", flush=True)
    if timings:
        seconds = np.array([t["seconds"] for t in timings])
        print(f"Per view: median {np.median(seconds):.2f} s, max {seconds.max():.2f} s", flush=True)
    if args.object_id is not None:
        for f in failures:
            print("failed:", json.dumps(f), flush=True)
        return
    path = generated / f"{NAME}_failures.json"
    path.write_text(json.dumps(sorted(failures, key=lambda f: f["sample_id"]), indent=1) + "\n")
    print(f"{len(outputs)} views with hand points; {len(failures)} failed (reasons in {path})", flush=True)


if __name__ == "__main__":
    main()
