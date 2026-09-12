"""Validate returned multi-view fitting reports; no GPU dependencies."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE / "multiple_view_returned_46083371"


def main():
    reports = {a: json.loads((ROOT / (a + ".json")).read_text()) for a in ["image", "camera", "oracle"]}
    output = {"arms": {}, "provenance": {}}
    for arm, report in reports.items():
        assert report["driver_sha256"] == hashlib.sha256((HERE / "fit_multiple_views_gpu.py").read_bytes()).hexdigest()
        assert report["view_selector_sha256"] == hashlib.sha256((HERE / "probe_tiny_fit_views.py").read_bytes()).hexdigest()
        refpath = HERE / "tiny_fit_returned_46083371" / arm / "results.json"
        assert report["source_reference_sha256"] == hashlib.sha256(refpath.read_bytes()).hexdigest()
        ref = json.loads(refpath.read_text())
        for key in ["initial_all_parameters_sha256", "initial_trainable_ca_sha256", "source_sha256"]:
            assert report[key] == ref[key], (arm, key)
        assert report["final_all_parameters_sha256"] != report["initial_all_parameters_sha256"]
        assert report["sample_noise_sha256"] == reports["image"]["sample_noise_sha256"]
        for src, digest in report["source_sha256"].items():
            assert hashlib.sha256((HERE.parents[1] / src).read_bytes()).hexdigest() == digest, src
        for key in ["steps", "seed", "precision", "objects", "fit_views_per_object", "reserved_views_per_object"]:
            assert report["settings"][key] == reports["image"]["settings"][key]
        batches = report["input_batches"]
        assert [b["group"] for b in batches] == list(range(7))
        fit_ids, reserved_ids = set(), set()
        for i, batch in enumerate(batches):
            assert batch["split"] == ("fit" if i < 4 else "reserved_view")
            for key in ["sample_ids", "image_sha256", "pointmap_sha256", "target_sha256"]:
                assert batch[key] == reports["image"]["input_batches"][i][key], (arm, i, key)
            assert batch["target_sha256"] == ref["target_sha256"]
            assert [s.rsplit("_", 1)[0] for s in batch["sample_ids"]] == [s.rsplit("_", 1)[0] for s in ref["sample_ids"]]
            (fit_ids if i < 4 else reserved_ids).update(batch["sample_ids"])
        assert len(fit_ids) == 16 and len(reserved_ids) == 12 and not fit_ids & reserved_ids
        previous_probe = json.loads((HERE / "view_probe_returned_46083371" / (arm + ".json")).read_text())
        for i in range(4):
            assert batches[i]["sample_ids"] == previous_probe["input_batches"][i]["sample_ids"]
        training = report["training"]
        assert [r["step"] for r in training] == list(range(1, 1001))
        assert Counter(r["group"] for r in training) == Counter({i: 250 for i in range(4)})
        assert all(r["group"] == (r["step"] - 1) % 4 for r in training)
        assert all(math.isfinite(r["loss"]) and math.isfinite(r["preclip_gradient_norm"]) for r in training)
        if arm != "image":
            assert all(r.get("gradients/vecsetx_encoder", 0) == 0 for r in training)
            assert any(r.get("gradients/touch_output_projection", 0) > 0 for r in training)
        result = {"assessments": [], "geometry": {}, "per_object": [],
                  "last100_train_loss": mean(r["loss"] for r in training[-100:]),
                  "previous100_train_loss": mean(r["loss"] for r in training[-200:-100])}
        assert [a["step"] for a in report["assessments"]] == [0, 100, 300, 1000]
        for assessment in report["assessments"]:
            assert [r["group"] for r in assessment["rows"]] == list(range(7))
            for row in assessment["rows"]:
                assert row["split"] == batches[row["group"]]["split"]
                assert len(row["fresh_noise_native_losses"]) == 8
                assert len(row["swapped_surface_native_losses"]) == (0 if arm == "image" else 8)
                assert all(math.isfinite(x) for x in row["fresh_noise_native_losses"] + row["swapped_surface_native_losses"])
            if assessment["step"] == 0:
                assert assessment["rows"][0]["fresh_noise_native_losses"] == ref["assessments"][0]["fresh_noise_native_losses"]
            entry = {"step": assessment["step"]}
            for split in ["fit", "reserved_view"]:
                rows = [r for r in assessment["rows"] if r["split"] == split]
                losses = [x for r in rows for x in r["fresh_noise_native_losses"]]
                wrong = [x for r in rows for x in r["swapped_surface_native_losses"]]
                entry[split] = {"loss": mean(losses), "swapped_loss": mean(wrong) if wrong else None,
                    "swapped_worse_draws": sum(b > a for a, b in zip(losses, wrong)) if wrong else None,
                    "per_group_loss": {r["group"]: mean(r["fresh_noise_native_losses"]) for r in rows}}
            result["assessments"].append(entry)
        rows = report["sampled"]
        assert len(rows) == 56
        expected = {(b["group"], sid, d) for b in batches for sid in b["sample_ids"] for d in [0, 1]}
        assert {(r["group"], r["sample_id"], r["noise_draw"]) for r in rows} == expected
        for r in rows:
            assert r["split"] == batches[r["group"]]["split"] and r["cfg"] == 0
            assert all(r[k] is not None and math.isfinite(r[k]) for k in ["latent_mse", "voxel_iou", "unaligned_voxel_center_chamfer"])
        for split in ["fit", "reserved_view"]:
            selected = [r for r in rows if r["split"] == split]
            result["geometry"][split] = {k: mean(r[k] for r in selected) for k in ["latent_mse", "voxel_iou", "unaligned_voxel_center_chamfer"]}
            result["geometry"][split]["minimum_iou"] = min(r["voxel_iou"] for r in selected)
            for sid in ref["sample_ids"]:
                oid = sid.rsplit("_", 1)[0]
                own = [r for r in selected if r["sample_id"].rsplit("_", 1)[0] == oid]
                result["per_object"].append({"object_id": oid, "split": split,
                    **{k: mean(r[k] for r in own) for k in ["latent_mse", "voxel_iou", "unaligned_voxel_center_chamfer"]}})
        output["arms"][arm] = result
        output["provenance"][arm] = hashlib.sha256((ROOT / (arm + ".json")).read_bytes()).hexdigest()
    output["limits"] = "Four objects, one training seed. Reserved views are not unseen objects. The prior one-view probe used different reserved views, so its mean transfer loss is not a matched baseline for the new reserved split."
    (HERE / "multiple_view_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print("Pairing, exact initial anchor reproduction, source hashes, 250 exposures per record, disjoint reserved views and all measurements verified.")
    for arm, r in output["arms"].items():
        print(arm, r["assessments"][-1], r["geometry"])
        for o in r["per_object"]:
            if o["split"] == "reserved_view":
                print(o["object_id"][:8], o["voxel_iou"], o["unaligned_voxel_center_chamfer"])


if __name__ == "__main__":
    main()
