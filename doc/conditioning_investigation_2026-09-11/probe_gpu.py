"""Read-only Stage-1 conditioning pilot; run from the repository root.

No training, Stage 2, or checkpoint writes. Measures matched conditional velocity
errors, shuffled-object token effects, slot-permutation invariance, and CFG effects.
The first forward is checked against the native generator.loss implementation.
"""
import argparse
import gc
import hashlib
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf

from dataloader import build_dataloader
from evaluate import read_run, restore_run
from experiments.noisy_target.evaluate import select_samples, tensor_digest
from train import amp, build_stage1_pipeline, prepare_batch


def mse(a, b):
    return (a.float() - b.float()).square().flatten(1).mean(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--pipeline-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--objects", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--times", type=float, nargs="+", default=[0., .2, .5, .8])
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=29)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.objects < 2 or a.batch_size < 2 or a.objects % a.batch_size or a.draws < 1:
        p.error("Use >=2 objects, >=2 batch size, divisible object count, and >=1 draws")
    if not all(0 <= t < 1 for t in a.times):
        p.error("Times must be in [0,1)")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    selected, identities = None, {}
    result = {"settings": {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(a).items() if k != "checkpoints"}, "runs": []}
    cfg = OmegaConf.load(a.pipeline_config)
    result["pipeline_yaml"] = a.pipeline_config.read_text()
    generator_yaml = a.pipeline_config.parent / cfg.ss_generator_config_path
    result["generator_yaml"] = generator_yaml.read_text()
    result["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(), "python": sys.version}
    repo = Path(__file__).resolve().parents[2]
    result["source_sha256"] = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in ["train.py", "evaluate.py", "dataloader.py",
                     "sam3d_objects/model/backbone/dit/embedder/touch.py",
                     "sam3d_objects/model/backbone/generator/classifier_free_guidance.py",
                     "sam3d_objects/model/backbone/generator/shortcut/model.py"]}
    for path in a.checkpoints:
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        ckpt, run_config, data = read_run(path, split=a.split)
        conditioning = ckpt.get("conditioning_config", {"no_pointmap": False, "oracle_point_frame": False})
        loader = build_dataloader(data, a.batch_size, 0, shuffle=False,
            include_touch=ckpt["touch_config"] is not None,
            oracle_point_frame=conditioning.get("oracle_point_frame", False))
        if selected is None:
            selected = select_samples(loader.dataset.records, a.objects, 1, a.seed)
            if len(selected) != a.objects:
                raise ValueError("Insufficient selected objects")
        lookup = {r["sample_id"]: r for r in loader.dataset.records}
        loader.dataset.records = [lookup[s] for s in selected]
        pipeline = build_stage1_pipeline(a.pipeline_config, device)
        model = restore_run(pipeline, ckpt, device)
        model.requires_grad_(False)
        gen, backbone = pipeline.ss_generator, pipeline.backbone
        if gen.fm_eps_max != 0 or gen.self_consistency_prob != 0:
            raise ValueError("Expected native d=0 training objective")
        if gen.reverse_fn.unconditional_handling != "add_flag" or not backbone.force_zeros_cond:
            raise ValueError("CFG probe requires inspected add_flag / force_zeros_cond contract")
        run = {"checkpoint": str(path), "checkpoint_step": ckpt["step"],
               "conditioning": conditioning, "touch_config": ckpt["touch_config"],
               "sigma_min": gen.sigma_min, "time_scale": gen.time_scale,
               "run_config": run_config, "rows": []}
        result["runs"].append(run)
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                targets, ca, kw, points, mask = prepare_batch(pipeline, batch, device, a.precision,
                    model.touch_encoder is not None, ckpt["mode"] == "image_touch_joint",
                    conditioning.get("oracle_point_frame", False))
                with amp(device, a.precision):
                    tokens = model.touch_encoder(points, mask) if model.touch_encoder is not None else None
                n = len(batch["sample_id"])
                assert len({lookup[s]["object_id"] for s in batch["sample_id"]}) == n
                variants = {"true": tokens}
                if tokens is not None:
                    variants["other_object"] = tokens.roll(1, 0)
                    variants["reverse_slots"] = tokens.flip(1)
                for draw in range(a.draws):
                    torch.manual_seed(a.seed + 10000 + bi * a.draws + draw)
                    noise = gen._generate_x0(targets)
                    identity = (tuple(batch["sample_id"]), gen.sigma_min, gen.time_scale,
                        tensor_digest(batch["image"]), tensor_digest(batch["pointmap"]),
                        tuple((k, tensor_digest(targets[k]), tensor_digest(noise[k])) for k in sorted(targets)))
                    key = (bi, draw)
                    if key in identities and identities[key] != identity:
                        raise ValueError("Unpaired native inputs, targets, noise, or flow settings")
                    identities[key] = identity
                    for time in a.times:
                        t = targets["shape"].new_full((n,), time)
                        xt, target = gen._generate_xt(noise, targets, t), gen._generate_target(noise, targets)
                        predictions = {}
                        with amp(device, a.precision):
                            for name, tok in variants.items():
                                kwargs = dict(kw)
                                if tok is not None:
                                    kwargs["touch_tokens"] = tok
                                predictions[name] = gen.reverse_fn(xt, t * gen.time_scale, *ca,
                                    d=torch.zeros_like(t), **kwargs)["shape"]
                            kwargs = dict(kw)
                            if tokens is not None:
                                kwargs["touch_tokens"] = tokens
                            if bi == 0 and draw == 0 and time == a.times[0]:
                                with patch.object(gen, "_generate_t", return_value=t), \
                                     patch.object(gen, "_generate_d", side_effect=lambda _: torch.zeros_like(t)), \
                                     patch.object(gen, "_generate_x0", return_value=noise):
                                    native, _ = gen.loss(targets, *ca, **kwargs)
                                explicit = mse(predictions["true"], target["shape"]).mean()
                                torch.testing.assert_close(native.float(), explicit, rtol=2e-3, atol=2e-5)
                                run["native_loss_check"] = {"native": native.item(), "explicit": explicit.item()}
                            uncond = backbone(xt, t * gen.time_scale, *ca,
                                d=torch.zeros_like(t), cfg=True, **kwargs)["shape"]
                        # Formula matches ClassifierFreeGuidanceWithExternalUnconditionalProbability.
                        # Same target/noise states; this is not a sampled-mesh CFG comparison.
                        for strength in (1., 7.):
                            active = float(cfg.get("ss_cfg_interval", [0, 500])[0]) <= time * gen.time_scale <= float(cfg.get("ss_cfg_interval", [0, 500])[1])
                            s = strength if active else 0.
                            predictions[f"cfg_{strength:g}"] = (1 + s) * predictions["true"].float() - s * uncond.float()
                        for name, pred in predictions.items():
                            error = mse(pred, target["shape"]).cpu().tolist()
                            delta = mse(pred, predictions["true"]).cpu().tolist()
                            for i, sid in enumerate(batch["sample_id"]):
                                run["rows"].append({"sample_id": sid, "time": time, "draw": draw,
                                    "variant": name, "velocity_mse": error[i], "prediction_delta_mse": delta[i],
                                    "donor_sample_id": batch["sample_id"][(i - 1) % n] if name == "other_object" else None})
                print(f"{path.parent.name}: {bi + 1}/{len(loader)} batches", flush=True)
        # Preserve completed checkpoints if a later checkpoint hits an environment issue.
        a.output.with_suffix(".partial.json").write_text(json.dumps(result, indent=2) + "\n")
        del model, gen, backbone, pipeline, ckpt, variants, tokens, predictions, pred, uncond, targets, xt, target, noise, ca, kw, kwargs, points, mask
        gc.collect()
        torch.cuda.empty_cache()
    result["paired_input_sha256"] = hashlib.sha256(repr(identities).encode()).hexdigest()
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
