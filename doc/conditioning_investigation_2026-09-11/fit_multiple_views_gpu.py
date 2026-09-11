"""Fit four views per object; reserve three other views of those same objects.

Same initialization, object exposure count, optimizer and trainable modules as
single-view tiny_fit_gpu.py. No held-out-object generalization claim.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tiny_fit_gpu import parameter_digest
from probe_tiny_fit_views import choose_views
from rollout_gpu import occupancy, geometry_metrics
import numpy as np
import torch
from omegaconf import OmegaConf
from dataloader import build_dataloader, load_data_config, collate_touch_batch
from train import build_stage1_pipeline, prepare_batch, build_optimizer, amp, TouchTrainingModel, trainable_state_dict, component_gradient_norms
from experiments.noisy_target.evaluate import tensor_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    reference_path = args.reference_fit_dir / "results.json"
    reference = json.loads(reference_path.read_text())
    settings = reference["settings"]
    assert settings["steps"] == 1000 and settings["objects"] == 4
    arm, seed, precision = settings["arm"], settings["seed"], settings["precision"]
    repo = Path(__file__).resolve().parents[2]
    for source, digest in reference["source_sha256"].items():
        assert hashlib.sha256((repo / source).read_bytes()).hexdigest() == digest, source
    pipeline_path = Path(settings["pipeline_config"])
    assert pipeline_path.read_text() == reference["pipeline_yaml"]
    cfg = OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent / cfg.ss_generator_config_path).read_text() == reference["generator_yaml"]
    data_path = Path(settings["data_config"])
    assert data_path.read_text() == reference["data_yaml"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda")
    pipeline = build_stage1_pipeline(pipeline_path, device)
    encoder = None
    if arm != "image":
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
        encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels,
                               trainable=False, use_position=False).to(device).eval()
    model = TouchTrainingModel(pipeline.ss_generator, encoder, False, arm == "oracle")
    optimizer, parameters = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope="full"))
    initial_hash = parameter_digest(model.named_parameters())
    assert initial_hash == reference["initial_all_parameters_sha256"]
    data = load_data_config(data_path); data["dataset"]["split"] = "train"
    loader = build_dataloader(data, 4, 0, shuffle=False, include_touch=encoder is not None,
                              oracle_point_frame=arm == "oracle")
    ds = loader.dataset
    record_groups = choose_views(ds.records, reference["sample_ids"], count=6)
    assert len(record_groups) == 7
    prepared, metadata = [], []
    for group, records in enumerate(record_groups):
        ds.records = records
        batch = collate_touch_batch([ds[i] for i in range(4)])
        targets, ca, kw, points, mask = prepare_batch(pipeline, batch, device, precision,
                                                   encoder is not None, False, arm == "oracle")
        entry = {"group": group, "split": "fit" if group < 4 else "reserved_view",
                 "sample_ids": [r["sample_id"] for r in records],
                 "image_sha256": tensor_digest(batch["image"]),
                 "pointmap_sha256": tensor_digest(batch["pointmap"]),
                 "target_sha256": tensor_digest(targets["shape"])}
        assert entry["target_sha256"] == reference["target_sha256"]
        features = None
        if encoder is not None:
            with torch.no_grad(), amp(device, precision):
                pp, mm, _, _ = encoder.prepare_points(points, mask)
                features = encoder.encoder.encode(pp, mm)["x"].detach()
                direct = encoder(points, mask)
                projected = encoder.output_projection(features) + encoder.touch_embedding
                torch.testing.assert_close(projected, direct, rtol=.002, atol=.00002)
            entry["features_sha256"] = tensor_digest(features)
        prepared.append((targets, ca, kw, features)); metadata.append(entry)
    for key in ["image_sha256", "pointmap_sha256", "target_sha256"]:
        assert metadata[0][key] == reference[key]
    if encoder is not None:
        assert metadata[0]["features_sha256"] == reference["point_features_sha256"]
    gen = pipeline.ss_generator
    assert gen.reverse_fn.p_unconditional == 0 and gen.self_consistency_prob == 0 and gen.fm_eps_max == 0
    def conditions(group, swap=False):
        targets, ca, kw, features = prepared[group]
        if encoder is None:
            return targets, ca, dict(kw)
        tokens = encoder.output_projection(features) + encoder.touch_embedding
        if swap:
            tokens = tokens.roll(1, 0)
        return targets, ca, {**kw, "touch_tokens": tokens}
    report = {"settings": {"arm": arm, "steps": 1000, "seed": seed, "precision": precision,
                           "objects": 4, "fit_views_per_object": 4, "reserved_views_per_object": 3},
              "initial_all_parameters_sha256": initial_hash,
              "initial_trainable_ca_sha256": reference["initial_trainable_ca_sha256"],
              "source_reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              "source_sha256": reference["source_sha256"],
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "view_selector_sha256": hashlib.sha256((Path(__file__).parent / "probe_tiny_fit_views.py").read_bytes()).hexdigest(),
              "input_batches": metadata, "training": [], "assessments": [], "sampled": [],
              "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()},
              "scope": "Fit on 16 records; 12 reserved views of the same four objects. No unseen-object evaluation. Each exact fit record receives 250 updates; each object receives 1000."}
    args.output_dir.mkdir(parents=True)
    def save_partial():
        (args.output_dir / "results.partial.json").write_text(json.dumps(report, indent=2) + "\n")
    def assess(step):
        py = random.getstate()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            gen.reverse_fn.training = True
            rows = []
            for group in range(7):
                values, wrong_values = [], []
                for draw in range(8):
                    for swap in ([False, True] if encoder is not None else [False]):
                        torch.manual_seed(100000 + seed + draw); random.seed(100000 + seed + draw)
                        with amp(device, precision):
                            targets, ca, kw = conditions(group, swap)
                            loss, _ = gen.loss(targets, *ca, **kw)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite assessment loss")
                        (wrong_values if swap else values).append(float(loss))
                rows.append({"group": group, "split": metadata[group]["split"],
                             "fresh_noise_native_losses": values, "swapped_surface_native_losses": wrong_values})
            if step == 0:
                np.testing.assert_allclose(rows[0]["fresh_noise_native_losses"], reference["assessments"][0]["fresh_noise_native_losses"], rtol=.01, atol=1e-5)
            report["assessments"].append({"step": step, "rows": rows})
        random.setstate(py); save_partial()
        print(arm, step, "fit loss", np.mean([r["fresh_noise_native_losses"] for r in rows[:4]]),
              "reserved-view loss", np.mean([r["fresh_noise_native_losses"] for r in rows[4:]]), flush=True)
    assess(0)
    for step in range(1, 1001):
        group = (step - 1) % 4
        torch.manual_seed(seed + step); random.seed(seed + step)
        optimizer.zero_grad(set_to_none=True)
        with amp(device, precision):
            targets, ca, kw = conditions(group)
            loss, _ = gen.loss(targets, *ca, **kw)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
        gradients = component_gradient_norms(model) if step == 1 or step % 20 == 0 else {}
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        report["training"].append({"step": step, "group": group, "loss": float(loss),
                                   "preclip_gradient_norm": float(norm), **gradients})
        if step % 20 == 0:
            print(arm, step, float(loss), flush=True)
        if step in [100, 300, 1000]:
            assess(step)
    final_hash = parameter_digest(model.named_parameters())
    assert final_hash != initial_hash
    report["final_all_parameters_sha256"] = final_hash
    torch.save({"model": trainable_state_dict(model), "step": 1000, "settings": report["settings"],
                "conditioning_config": model.conditioning_config}, args.output_dir / "fitted_parameters.pt")
    # Secondary check: decode final CFG-0 samples for every fit and reserved view.
    decoder = pipeline.init_ss_decoder(cfg.ss_decoder_config_path, cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    with torch.no_grad(), amp(device, precision):
        support_targets = [occupancy(decoder, prepared[0][0]["shape"][i:i+1]) for i in range(4)]
    gen.no_shortcut = True; gen.inference_steps = 25
    gen.rescale_t = float(cfg.get("ss_rescale_t", 3))
    gen.reverse_fn.interval = list(cfg.get("ss_cfg_interval", [0, 500]))
    gen.reverse_fn.strength = 0; gen.reverse_fn.training = False
    report["sample_noise_sha256"] = []
    with torch.no_grad():
        for draw in [0, 1]:
            torch.manual_seed(200000 + seed + draw)
            noise = gen._generate_x0(prepared[0][0])
            report["sample_noise_sha256"].append({k: tensor_digest(v) for k, v in noise.items()})
            for group in range(7):
                with amp(device, precision):
                    targets, ca, kw = conditions(group)
                    shapes = {k: tuple(v.shape) for k, v in targets.items()}
                    with patch.object(gen, "_generate_noise", side_effect=lambda *a, **k: {n: v.clone() for n, v in noise.items()}):
                        pred = gen(shapes, device, *ca, **kw)["shape"]
                for i, sid in enumerate(metadata[group]["sample_ids"]):
                    if not torch.isfinite(pred[i]).all():
                        raise FloatingPointError("Nonfinite rollout")
                    with amp(device, precision):
                        support = occupancy(decoder, pred[i:i+1])
                    filename = f"{sid}_draw{draw}.npz"
                    np.savez_compressed(args.output_dir / filename, prediction=pred[i].float().cpu().numpy(),
                                        target=targets["shape"][i].cpu().numpy(), predicted_occupancy=support,
                                        target_occupancy=support_targets[i])
                    report["sampled"].append({"group": group, "split": metadata[group]["split"],
                        "sample_id": sid, "noise_draw": draw, "cfg": 0, "artifact": filename,
                        "latent_mse": float((pred[i].float() - targets["shape"][i].float()).square().mean()),
                        **geometry_metrics(support, support_targets[i])})
                save_partial()
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
