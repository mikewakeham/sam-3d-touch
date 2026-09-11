"""Summarize the returned paired tiny-fit experiment; standard library only."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from check_tiny_fit import check

HERE = Path(__file__).resolve().parent
ROOT = HERE / "tiny_fit_returned_46083371"
REPO = HERE.parents[1]


def main():
    check(ROOT)
    out = {"arms": {}, "source_results_sha256": {}}
    for arm in ["image", "camera", "oracle"]:
        path = ROOT / arm / "results.json"
        report = json.loads(path.read_text())
        out["source_results_sha256"][arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        for source, digest in report["source_sha256"].items():
            assert hashlib.sha256((REPO / source).read_bytes()).hexdigest() == digest, source
        training = report["training"]
        assert [r["step"] for r in training] == list(range(1, 1001))
        assert all(math.isfinite(r["loss"]) and math.isfinite(r["preclip_gradient_norm"]) for r in training)
        if arm != "image":
            assert all(r.get("gradients/vecsetx_encoder", 0) == 0 for r in training)
            assert any(r.get("gradients/touch_output_projection", 0) > 0 for r in training)
        assert any(r.get("gradients/shape_cross_attention", 0) > 0 for r in training)
        result = {"assessments": [], "per_object_final": [],
                  "train_loss_last100": mean(r["loss"] for r in training[-100:]),
                  "train_loss_previous100": mean(r["loss"] for r in training[-200:-100])}
        target_counts = {}
        for assessment in report["assessments"]:
            expected = {(sid, draw, cfg) for sid in report["sample_ids"] for draw in [0, 1] for cfg in [0, 7]}
            rows = assessment["sampled"]
            assert len(rows) == len(expected)
            assert {(r["sample_id"], r["noise_draw"], r["cfg"]) for r in rows} == expected
            for row in rows:
                assert not row.get("failure")
                for key in ["latent_mse", "voxel_iou", "unaligned_voxel_center_chamfer"]:
                    assert row[key] is not None and math.isfinite(row[key]), key
                sid = row["sample_id"]
                assert target_counts.setdefault(sid, row["target_occupied"]) == row["target_occupied"]
            losses = assessment["fresh_noise_native_losses"]
            swaps = assessment["swapped_surface_native_losses"]
            assert len(losses) == 8 and all(math.isfinite(x) for x in losses + swaps)
            item = {"step": assessment["step"], "fresh_noise_loss": mean(losses),
                    "swapped_loss": mean(swaps) if swaps else None,
                    "swap_worse_draws": sum(b > a for a, b in zip(losses, swaps)) if swaps else None,
                    "geometry": {}}
            if swaps:
                assert len(swaps) == 8
                item["swap_loss_ratio"] = mean(swaps) / mean(losses)
            for cfg in [0, 7]:
                selected = [r for r in rows if r["cfg"] == cfg]
                item["geometry"][cfg] = {k: mean(r[k] for r in selected) for k in
                    ["latent_mse", "voxel_iou", "unaligned_voxel_center_chamfer"]}
                item["geometry"][cfg]["minimum_iou"] = min(r["voxel_iou"] for r in selected)
                if assessment["step"] == 1000:
                    for sid in report["sample_ids"]:
                        pair = [r for r in selected if r["sample_id"] == sid]
                        result["per_object_final"].append({"sample_id": sid, "cfg": cfg,
                            **{k: mean(r[k] for r in pair) for k in ["latent_mse", "voxel_iou", "unaligned_voxel_center_chamfer"]}})
            result["assessments"].append(item)
        out["arms"][arm] = result
    base = out["arms"]["image"]["assessments"][-1]["fresh_noise_loss"]
    for arm in ["camera", "oracle"]:
        out["arms"][arm]["final_loss_reduction_vs_image"] = 1 - out["arms"][arm]["assessments"][-1]["fresh_noise_loss"] / base
    out["scope"] = "Four training objects, one fixed view each, one training seed; fresh noise is not unseen-object validation. Raw Stage-1 support metrics are not aligned Stage-2 mesh CD."
    (HERE / "tiny_fit_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print("Validated complete measurements; wrote tiny_fit_summary.json")


if __name__ == "__main__":
    main()
