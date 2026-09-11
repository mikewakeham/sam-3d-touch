"""Read-only loss probe of fitted weights on other views of the same objects.

No training or decoding. Anchor loss must reproduce the returned experiment.
Crossed modalities are diagnostic interventions, not natural-view benchmarks.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

from tiny_fit_gpu import parameter_digest
import numpy as np
import torch
from omegaconf import OmegaConf
from dataloader import build_dataloader, load_data_config, collate_touch_batch
from train import build_stage1_pipeline, prepare_batch, build_optimizer, amp, TouchTrainingModel
from experiments.noisy_target.evaluate import tensor_digest


def choose_views(records, ids, count=3):
    lookup = {r["sample_id"]: r for r in records}
    anchors = [lookup[sid] for sid in ids]
    groups = []
    for anchor in anchors:
        options = sorted((r for r in records if r["object_id"] == anchor["object_id"]
                          and r["sample_id"] != anchor["sample_id"]), key=lambda r: r["sample_id"])
        if len(options) < count:
            raise ValueError("Need three other views for " + anchor["sample_id"])
        random.Random(29).shuffle(options)
        groups.append(options[:count])
    return [anchors] + [[g[i] for g in groups] for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    previous = json.loads((a.fit_dir / "results.json").read_text())
    settings = previous["settings"]
    assert settings["objects"] == 4 and settings["steps"] == 1000
    arm = settings["arm"]
    seed = settings["seed"]
    precision = settings["precision"]
    device = torch.device("cuda")
    repo = Path(__file__).resolve().parents[2]
    for name, digest in previous["source_sha256"].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
    pipeline_path = Path(settings["pipeline_config"])
    assert pipeline_path.read_text() == previous["pipeline_yaml"]
    cfg = OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent / cfg.ss_generator_config_path).read_text() == previous["generator_yaml"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pipeline = build_stage1_pipeline(pipeline_path, device)
    encoder = None
    if arm != "image":
        from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
        encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels,
                               trainable=False, use_position=False).to(device).eval()
    model = TouchTrainingModel(pipeline.ss_generator, encoder, False, arm == "oracle")
    # Select exactly the parameters saved by the fitting job; no optimizer step occurs.
    optimizer, _ = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope="full"))
    del optimizer
    assert parameter_digest(model.named_parameters()) == previous["initial_all_parameters_sha256"]
    archive = torch.load(a.fit_dir / "fitted_parameters.pt", map_location="cpu", weights_only=True)
    assert archive["step"] == 1000 and archive["settings"] == settings
    parameters = dict(model.named_parameters())
    assert set(archive["model"]) == {n for n, p in parameters.items() if p.requires_grad}
    with torch.no_grad():
        for name, value in archive["model"].items():
            parameters[name].copy_(value)
    assert parameter_digest(model.named_parameters()) == previous["final_all_parameters_sha256"]
    model.requires_grad_(False)
    data_path = Path(settings["data_config"])
    assert data_path.read_text() == previous["data_yaml"]
    data = load_data_config(data_path); data["dataset"]["split"] = "train"
    loader = build_dataloader(data, 4, 0, shuffle=False, include_touch=encoder is not None,
                              oracle_point_frame=arm == "oracle")
    ds = loader.dataset
    batches = choose_views(ds.records, previous["sample_ids"])
    prepared = []
    provenance = []
    reference_features = None
    for records in batches:
        ds.records = records
        batch = collate_touch_batch([ds[i] for i in range(4)])
        targets, ca, kw, points, mask = prepare_batch(pipeline, batch, device, precision,
                                                   encoder is not None, False, arm == "oracle")
        tok = None
        entry = {"sample_ids": [r["sample_id"] for r in records],
                 "image_sha256": tensor_digest(batch["image"]),
                 "pointmap_sha256": tensor_digest(batch["pointmap"]),
                 "target_sha256": tensor_digest(targets["shape"])}
        assert entry["target_sha256"] == previous["target_sha256"]
        if encoder is not None:
            with torch.no_grad(), amp(device, precision):
                pp, mm, _, _ = encoder.prepare_points(points, mask)
                features = encoder.encoder.encode(pp, mm)["x"]
                tok = encoder.output_projection(features) + encoder.touch_embedding
            entry["features_sha256"] = tensor_digest(features)
            if reference_features is None:
                reference_features = features
                assert entry["features_sha256"] == previous["point_features_sha256"]
            delta = (features.float() - reference_features.float()).square().mean()
            entry["features_mse_vs_anchor"] = float(delta)
        prepared.append((targets, ca, kw, tok))
        provenance.append(entry)
    for key in ["image_sha256", "pointmap_sha256", "target_sha256"]:
        assert provenance[0][key] == previous[key], key
    gen = pipeline.ss_generator
    gen.reverse_fn.training = True
    assert gen.reverse_fn.p_unconditional == 0 and gen.self_consistency_prob == 0 and gen.fm_eps_max == 0
    def losses(visual_index, surface_index, swap=False):
        targets, ca, kw, _ = prepared[visual_index]
        tok = prepared[surface_index][3]
        if swap:
            tok = tok.roll(1, 0)
        cond = dict(kw) if tok is None else {**kw, "touch_tokens": tok}
        values = []
        with torch.no_grad():
            for draw in range(8):
                torch.manual_seed(100000 + seed + draw); random.seed(100000 + seed + draw)
                with amp(device, precision):
                    loss, _ = gen.loss(targets, *ca, **cond)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite probe loss")
                values.append(float(loss))
        return values
    rows = []
    anchor = losses(0, 0)
    expected = previous["assessments"][-1]["fresh_noise_native_losses"]
    np.testing.assert_allclose(anchor, expected, rtol=.01, atol=1e-5,
        err_msg="Anchor does not reproduce; stop before interpreting novel-view effects")
    rows.append({"view_group": 0, "condition": "anchor", "losses": anchor})
    for i in range(1, len(prepared)):
        cases = [("natural_other_view", i, i, False)]
        if encoder is not None:
            cases += [("anchor_visual_other_surface", 0, i, False),
                      ("other_visual_anchor_surface", i, 0, False),
                      ("natural_other_view_swapped_object_surface", i, i, True)]
        for label, vi, pi, swap in cases:
            values = losses(vi, pi, swap)
            rows.append({"view_group": i, "condition": label, "losses": values})
            print(arm, i, label, float(np.mean(values)), flush=True)
    result = {"settings": vars(a) | {"arm": arm}, "input_batches": provenance,
              "source_fit_results_sha256": hashlib.sha256((a.fit_dir / "results.json").read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()},
              "rows": rows, "scope": "Previously unseen views of four fitted objects. No training. Crossed camera-frame modalities may contradict each other; not natural benchmark inputs."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
