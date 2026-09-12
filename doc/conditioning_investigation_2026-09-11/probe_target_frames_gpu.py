"""Frozen pretrained SAM3D: does another target orientation reduce native loss?

No training or production data writes. Re-encode physically rotated occupancy;
do not assume that the VAE is equivariant. Original targets must reproduce.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from omegaconf import OmegaConf
from dataloader import build_dataloader, load_data_config
from experiments.noisy_target.evaluate import select_samples, tensor_digest
from train import amp, build_stage1_pipeline, prepare_batch
from frame_contract import rotations, rotate_grid
from rollout_gpu import occupancy, geometry_metrics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-config", type=Path, default=Path("checkpoints/hf/pipeline.yaml"))
    p.add_argument("--data-config", type=Path, default=Path("configs/data_full_surface.yaml"))
    p.add_argument("--encoder-checkpoint", type=Path, default=Path("checkpoints/hf/ss_encoder.ckpt"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--objects", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    a = p.parse_args()
    if a.objects < 2 or a.batch_size < 1:
        p.error("Require at least two objects and positive batch size")
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda")
    data = load_data_config(a.data_config); data["dataset"]["split"] = "train"
    loader = build_dataloader(data, a.batch_size, 0, shuffle=False, include_touch=False)
    ds = loader.dataset
    ids = select_samples(ds.records, a.objects, 2, a.seed)
    lookup = {r["sample_id"]: r for r in ds.records}
    ds.records = [lookup[s] for s in ids]
    object_ids = list(dict.fromkeys(r["object_id"] for r in ds.records))
    assert len(object_ids) == a.objects and len(ids) == 2 * a.objects
    assert all(sum(r["object_id"] == oid for r in ds.records) == 2 for oid in object_ids)
    view_index = {sid: [r["sample_id"] for r in ds.records if r["object_id"] == lookup[sid]["object_id"]].index(sid) for sid in ids}
    cfg = OmegaConf.load(a.pipeline_config)
    source = REPO / "data_generation/objaverse-dexonomy/generate_target_latents.py"
    spec = importlib.util.spec_from_file_location("target_generation", source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Preflight every mesh before allocating the model or creating outputs.
    for oid in object_ids:
        for path in [ds.root / "objects" / oid / "model.obj",
                     ds.root / "generated_data" / oid / "object_transform.npz"]:
            if not path.is_file():
                raise FileNotFoundError(path)
    encoder = module.load_encoder(a.encoder_checkpoint, device).requires_grad_(False)
    variants = rotations()
    report = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "sample_ids": ids, "object_ids": object_ids,
              "rotations_column_convention": {k: v.tolist() for k, v in variants.items()},
              "pipeline_yaml": a.pipeline_config.read_text(), "data_yaml": a.data_config.read_text(),
              "generator_yaml": (a.pipeline_config.parent / cfg.ss_generator_config_path).read_text(),
              "source_sha256": {str(q.relative_to(REPO)): sha(q) for q in [source, Path(__file__),
                  HERE / "frame_contract.py", REPO / "train.py", REPO / "dataloader.py"]},
              "encoder_checkpoint_sha256": sha(a.encoder_checkpoint),
              "generator_checkpoint_sha256": sha(a.pipeline_config.parent / cfg.ss_generator_ckpt_path),
              "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()},
              "target_checks": [], "target_reconstruction": [], "inputs": [], "loss_rows": [], "sampled_rows": [],
              "scope": "Frozen pretrained image+pointmap model. No VecSetX, adaptation or optimization. Shape-only loss with existing zero layout targets. Tests target orientation compatibility, not a proven training fix."}
    a.output_dir.mkdir(parents=True)
    def save():
        (a.output_dir / "results.partial.json").write_text(json.dumps(report, indent=2) + "\n")
    latent_bank = {}
    # fp32 exactly like target generation; no autocast when encoding.
    with torch.inference_mode():
        for oid in object_ids:
            record = next(r for r in ds.records if r["object_id"] == oid)
            mesh_path = ds.root / "objects" / oid / "model.obj"
            transform_path = ds.root / "generated_data" / oid / "object_transform.npz"
            mesh = module.load_normalized_mesh(mesh_path, transform_path)
            grid = module.voxelize_mesh(mesh)[0].numpy()
            stored = ds.load_target(ds.resolve_path(record["target_path"]))
            latent_bank[oid] = {}
            for name, rotation in variants.items():
                transformed = rotate_grid(grid, rotation)
                x = torch.from_numpy(transformed)[None, None].to(device)
                mean = encoder(x)["mean"][0].float().cpu().numpy()
                flat = np.ascontiguousarray(mean.transpose(1, 2, 3, 0).reshape(4096, 8))
                if not np.isfinite(flat).all():
                    raise FloatingPointError("Nonfinite encoded target")
                if name == "identity":
                    # Fail closed on source/encoder mismatch; do not silently
                    # compare a round-trip surrogate with original targets.
                    np.testing.assert_allclose(flat, stored, rtol=1e-4, atol=1e-5,
                                               err_msg="Original target regeneration mismatch: " + oid)
                    report["target_checks"].append({"object_id": oid,
                        "identity_max_abs_error": float(np.max(np.abs(flat - stored))),
                        "mesh_sha256": sha(mesh_path), "transform_sha256": sha(transform_path),
                        "stored_target_sha256": sha(ds.resolve_path(record["target_path"]))})
                    flat = stored  # Use original training target exactly.
                latent_bank[oid][name] = torch.from_numpy(flat)
                np.savez_compressed(a.output_dir / f"target_{oid}_{name}.npz", latent=flat,
                                    physical_occupancy=transformed)
            print("Encoded target orientations:", oid, flush=True)
            save()
    del encoder
    gc.collect(); torch.cuda.empty_cache()
    pipeline = build_stage1_pipeline(a.pipeline_config, device)
    gen = pipeline.ss_generator
    gen.requires_grad_(False)
    assert gen.self_consistency_prob == 0 and gen.fm_eps_max == 0
    assert gen.reverse_fn.p_unconditional == 0
    decoder = pipeline.init_ss_decoder(cfg.ss_decoder_config_path, cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    supports = {}
    with torch.no_grad(), amp(device, a.precision):
        for oid in object_ids:
            supports[oid] = {name: occupancy(decoder, latent[None].to(device))
                             for name, latent in latent_bank[oid].items()}
            for name, latent in latent_bank[oid].items():
                with np.load(a.output_dir / f"target_{oid}_{name}.npz", allow_pickle=False) as saved:
                    original = saved["physical_occupancy"].astype(bool)
                report["target_reconstruction"].append({"object_id": oid, "rotation": name,
                    "latent_mean_square": float(latent.square().mean()),
                    **geometry_metrics(supports[oid][name], original)})
    checked = False
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            targets, ca, kw, _, _ = prepare_batch(pipeline, batch, device, a.precision, False, False, False)
            sids = batch["sample_id"]
            report["inputs"].append({"sample_ids": sids, "image_sha256": tensor_digest(batch["image"]),
                                      "pointmap_sha256": tensor_digest(batch["pointmap"]),
                                      "target_sha256": tensor_digest(targets["shape"])})
            gen.reverse_fn.training = True
            for draw in range(2):
                torch.manual_seed(100000 + a.seed + 2 * bi + draw)
                noise = gen._generate_x0(targets)
                report["inputs"][-1].setdefault("noise_sha256", []).append({k: tensor_digest(v) for k, v in noise.items()})
                for name in variants:
                    alternative = dict(targets)
                    alternative["shape"] = torch.stack([latent_bank[lookup[s]["object_id"]][name] for s in sids]).to(device)
                    if name == "identity":
                        torch.testing.assert_close(alternative["shape"], targets["shape"], rtol=0, atol=0)
                    for time in [.2, .5, .8]:
                        t = targets["shape"].new_full((len(sids),), time)
                        xt = gen._generate_xt(noise, alternative, t)
                        velocity_target = gen._generate_target(noise, alternative)["shape"]
                        with amp(device, a.precision):
                            pred = gen.reverse_fn(xt, t * gen.time_scale, *ca, d=torch.zeros_like(t), **kw)["shape"]
                            errors = (pred.float() - velocity_target.float()).square().flatten(1).mean(1)
                            if not checked:
                                with patch.object(gen, "_generate_t", return_value=t), \
                                     patch.object(gen, "_generate_d", side_effect=lambda _: torch.zeros_like(t)), \
                                     patch.object(gen, "_generate_x0", return_value=noise):
                                    native, _ = gen.loss(alternative, *ca, **kw)
                                torch.testing.assert_close(native.float(), errors.mean(), rtol=.002, atol=2e-5)
                                report["native_loss_check"] = {"native": float(native), "explicit": float(errors.mean())}
                                checked = True
                        if not torch.isfinite(errors).all():
                            raise FloatingPointError("Nonfinite flow error")
                        for i, sid in enumerate(sids):
                            report["loss_rows"].append({"sample_id": sid, "object_id": lookup[sid]["object_id"],
                                "view": view_index[sid], "draw": draw, "time": time, "rotation": name,
                                "velocity_mse": float(errors[i])})
            # Two native rollouts per input, shared across every target candidate.
            # This avoids treating teacher-forced loss as sampled reconstruction.
            gen.no_shortcut = True; gen.inference_steps = 25
            gen.rescale_t = float(cfg.get("ss_rescale_t", 3))
            gen.reverse_fn.interval = list(cfg.get("ss_cfg_interval", [0, 500]))
            gen.reverse_fn.strength = float(cfg.get("ss_cfg_strength", 7))
            report["rollout_cfg"] = gen.reverse_fn.strength
            gen.reverse_fn.training = False
            for draw in range(2):
                torch.manual_seed(200000 + a.seed + 2 * bi + draw)
                with amp(device, a.precision):
                    pred = gen({k: tuple(v.shape) for k, v in targets.items()}, device, *ca, **kw)["shape"]
                    for i, sid in enumerate(sids):
                        support = occupancy(decoder, pred[i:i+1])
                        np.savez_compressed(a.output_dir / f"prediction_{sid}_{draw}.npz",
                                            latent=pred[i].float().cpu().numpy(), occupancy=support)
                        oid = lookup[sid]["object_id"]
                        for name in variants:
                            mse = float((pred[i].float().cpu() - latent_bank[oid][name]).square().mean())
                            report["sampled_rows"].append({"sample_id": sid, "object_id": oid,
                                "view": view_index[sid], "draw": draw, "rotation": name, "latent_mse": mse,
                                **geometry_metrics(support, supports[oid][name])})
            save()
            print("Finished batch", bi + 1, "/", len(loader), flush=True)
    assert checked
    (a.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
