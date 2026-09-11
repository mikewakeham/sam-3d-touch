"""Paired native Stage-1 rollouts; no fitting, Stage 2, or geometric alignment.

Uses the completed pilot's exact sample IDs and checkpoint paths. Only the
guidance strength changes within a checkpoint/noise pair. Save target and
prediction latents/support so geometry can be inspected after the job.
"""
import argparse
import gc
import hashlib
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from dataloader import build_dataloader
from evaluate import read_run, restore_run, stable_seed
from experiments.noisy_target.evaluate import tensor_digest
from train import build_stage1_pipeline, prepare_batch


def geometry_metrics(prediction, target):
    """Unregistered voxel-center distances; not source-mesh or Stage-2 CD."""
    assert prediction.shape == target.shape == (64, 64, 64)
    intersection = int(np.count_nonzero(prediction & target))
    union = int(np.count_nonzero(prediction | target))
    p = (np.argwhere(prediction) + .5) / 64 - .5
    q = (np.argwhere(target) + .5) / 64 - .5
    return {"voxel_iou": intersection / union if union else 1.,
            "predicted_occupied": len(p), "target_occupied": len(q),
            "unaligned_voxel_center_chamfer": float((cKDTree(p).query(q)[0].mean() +
                cKDTree(q).query(p)[0].mean()) / 2) if len(p) and len(q) else None}


def occupancy(decoder, latent):
    x = latent if decoder.reshape_input_to_cube else decoder.flat_to_cube(latent)
    logits = decoder(x)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite decoder output")
    return (logits[0, 0] > 0).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-json", type=Path, required=True)
    p.add_argument("--pipeline-config", type=Path, default=Path("checkpoints/hf/pipeline.yaml"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--strengths", type=float, nargs="+", default=[0., 1., 7.])
    p.add_argument("--seeds", type=int, nargs="+", default=[29, 30])
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--precision", choices=["pipeline", "bf16", "fp32"], default="pipeline")
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError("Choose a new output directory")
    if a.steps < 1 or not a.seeds or not a.strengths or any(s < 0 for s in a.strengths):
        p.error("Require positive steps, seeds, and nonnegative strengths")
    if len(set(a.seeds)) != len(a.seeds) or len(set(a.strengths)) != len(a.strengths):
        p.error("Duplicate seeds/strengths")
    pilot = json.loads(a.probe_json.read_text())
    if not pilot.get("paired_input_sha256"):
        raise ValueError("A completed pilot is required")
    selected = list(dict.fromkeys(r["sample_id"] for r in pilot["runs"][0]["rows"]))
    assert len(selected) == pilot["settings"]["objects"]
    cfg = OmegaConf.load(a.pipeline_config)
    if a.pipeline_config.read_text() != pilot["pipeline_yaml"]:
        raise ValueError("Pipeline config changed since pilot; resolve provenance first")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(a.precision)
    if dtype is None:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[cfg.get("shape_model_dtype") or cfg.get("dtype", "bfloat16")]
    def autocast():
        return torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
    device = torch.device("cuda")
    a.output_dir.mkdir(parents=True)
    result = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "sample_ids": selected, "effective_dtype": str(dtype), "runs": [],
              "pipeline_yaml": a.pipeline_config.read_text(),
              "generator_yaml": (a.pipeline_config.parent / cfg.ss_generator_config_path).read_text(),
              "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name()},
              "metric_definition": "Raw target-frame Stage-1 support; no pruning, Stage 2, scale normalization, rotation search, or ICP. CD is mean Euclidean NN distance between occupied voxel centers."}
    repo = Path(__file__).resolve().parents[2]
    result["source_sha256"] = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in pilot["source_sha256"]}
    if result["source_sha256"] != pilot["source_sha256"]:
        raise ValueError("Source changed since pilot; resolve provenance before comparison")
    paired = {}
    for pilot_run in pilot["runs"]:
        path = Path(pilot_run["checkpoint"])
        random.seed(29)
        np.random.seed(29)
        torch.manual_seed(29)
        checkpoint, _, data = read_run(path, split=pilot["settings"]["split"])
        if checkpoint["step"] != pilot_run["checkpoint_step"]:
            raise ValueError("Checkpoint step changed since pilot")
        conditioning = checkpoint.get("conditioning_config", {"no_pointmap": False, "oracle_point_frame": False})
        loader = build_dataloader(data, 1, 0, shuffle=False,
            include_touch=checkpoint["touch_config"] is not None,
            oracle_point_frame=conditioning.get("oracle_point_frame", False))
        lookup = {r["sample_id"]: r for r in loader.dataset.records}
        loader.dataset.records = [lookup[s] for s in selected]
        pipeline = build_stage1_pipeline(a.pipeline_config, device)
        model = restore_run(pipeline, checkpoint, device)
        model.requires_grad_(False)
        gen = pipeline.ss_generator
        # Reuse the inference pipeline's exact configuration override, without
        # allocating Stage 2. Unlike the first probe, this uses native CFG forward.
        pipeline.override_ss_generator_cfg_config(gen,
            cfg_strength=float(cfg.get("ss_cfg_strength", 7)), inference_steps=a.steps,
            rescale_t=float(cfg.get("ss_rescale_t", 3)),
            cfg_interval=list(cfg.get("ss_cfg_interval", [0, 500])),
            cfg_strength_pm=float(cfg.get("ss_cfg_strength_pm", 0)))
        gen.no_shortcut = True
        gen.eval()
        decoder = pipeline.init_ss_decoder(cfg.ss_decoder_config_path, cfg.ss_decoder_ckpt_path)
        decoder.eval().requires_grad_(False)
        run = {"checkpoint": str(path), "checkpoint_step": checkpoint["step"],
               "rescale_t": gen.rescale_t, "cfg_interval": list(gen.reverse_fn.interval),
               "no_shortcut": gen.no_shortcut, "rows": []}
        result["runs"].append(run)
        run_dir = a.output_dir / path.parent.name
        run_dir.mkdir()
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                sid = batch["sample_id"][0]
                targets, ca, kw, points, mask = prepare_batch(pipeline, batch, device,
                    "fp32" if dtype == torch.float32 else "bf16", model.touch_encoder is not None,
                    checkpoint["mode"] == "image_touch_joint", conditioning.get("oracle_point_frame", False))
                with autocast():
                    tokens = model.touch_encoder(points, mask) if model.touch_encoder is not None else None
                    target_voxels = occupancy(decoder, targets["shape"])
                kwargs = dict(kw)
                if tokens is not None:
                    kwargs["touch_tokens"] = tokens
                for seed in a.seeds:
                    torch.manual_seed(stable_seed(seed, sid))
                    shapes = {name: tuple(value.shape) for name, value in targets.items()}
                    noise = gen._generate_noise(shapes, device)
                    identity = (tensor_digest(batch["image"]), tensor_digest(batch["pointmap"]),
                        tuple((name, tensor_digest(targets[name]), tensor_digest(noise[name])) for name in sorted(noise)),
                        hashlib.sha256(target_voxels.tobytes()).hexdigest())
                    key = (sid, seed)
                    if key in paired and paired[key] != identity:
                        raise ValueError("Noise, target, native observations, or target decode were not paired")
                    paired[key] = identity
                    if bi == 0 and seed == a.seeds[0]:
                        # Validate native CFG and its numeric effect at a target-noise
                        # state using actual sampling precision, independently of bf16 pilot.
                        t = targets["shape"].new_tensor([.5])
                        xt = gen._generate_xt(noise, targets, t)
                        with autocast():
                            c = pipeline.backbone(xt, t * gen.time_scale, *ca, d=torch.zeros_like(t), **kwargs)["shape"]
                            u = pipeline.backbone(xt, t * gen.time_scale, *ca, d=torch.zeros_like(t), cfg=True, **kwargs)["shape"]
                            run["native_cfg_checks"] = []
                            for s in a.strengths:
                                gen.reverse_fn.strength = s
                                native = gen.reverse_fn(xt, t * gen.time_scale, *ca, d=torch.zeros_like(t), **kwargs)["shape"]
                                active = gen.reverse_fn.interval[0] <= .5 * gen.time_scale <= gen.reverse_fn.interval[1]
                                expected = (1 + s) * c - s * u if active else c
                                torch.testing.assert_close(native, expected, rtol=.002, atol=.00002)
                                velocity = gen._generate_target(noise, targets)["shape"]
                                run["native_cfg_checks"].append({"strength": s,
                                    "velocity_mse": (native.float() - velocity.float()).square().mean().item(),
                                    "formula_max_error": (native - expected).abs().max().item()})
                    for strength in a.strengths:
                        gen.reverse_fn.strength = strength
                        # Native generator may request fresh noise; supply independent
                        # clones of the same native draw for each counterfactual rollout.
                        with patch.object(gen, "_generate_noise", side_effect=lambda *args, **kw: {k: v.clone() for k, v in noise.items()}):
                            with autocast():
                                predicted = gen(shapes, device, *ca, **kwargs)["shape"]
                        row = {"sample_id": sid, "seed": seed, "strength": strength}
                        if not torch.isfinite(predicted).all():
                            row["failure"] = "Nonfinite generated latent"
                        else:
                            try:
                                with autocast():
                                    support = occupancy(decoder, predicted)
                                row.update(geometry_metrics(support, target_voxels))
                                row["latent_mse"] = (predicted.float() - targets["shape"].float()).square().mean().item()
                                filename = f"{sid}_seed{seed}_cfg{strength:g}.npz"
                                np.savez_compressed(run_dir / filename,
                                    prediction=predicted[0].float().cpu().numpy(), target=targets["shape"][0].float().cpu().numpy(),
                                    predicted_occupancy=support, target_occupancy=target_voxels)
                                row["artifact"] = str(Path(path.parent.name) / filename)
                            except FloatingPointError as e:
                                row["failure"] = str(e)
                        run["rows"].append(row)
                (a.output_dir / "partial.json").write_text(json.dumps(result, indent=2) + "\n")
                print(f"{path.parent.name}: {bi+1}/{len(loader)} samples", flush=True)
        del model, gen, pipeline, decoder, checkpoint, kwargs, tokens, ca, kw, predicted
        gc.collect()
        torch.cuda.empty_cache()
    result["paired_inputs_sha256"] = hashlib.sha256(repr(paired).encode()).hexdigest()
    (a.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
