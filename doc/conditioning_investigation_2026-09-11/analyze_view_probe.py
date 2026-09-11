"""Validate and summarize view-transfer loss interventions; stdlib only."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE / "view_probe_returned_46083371"


def main():
    reports = {a: json.loads((ROOT / (a + ".json")).read_text()) for a in ["image", "camera", "oracle"]}
    summary = {"arms": {}, "provenance": {}}
    for arm, report in reports.items():
        assert report["probe_sha256"] == hashlib.sha256((HERE / "probe_tiny_fit_views.py").read_bytes()).hexdigest()
        fit_path = HERE / "tiny_fit_returned_46083371" / arm / "results.json"
        assert report["source_fit_results_sha256"] == hashlib.sha256(fit_path.read_bytes()).hexdigest()
        fit = json.loads(fit_path.read_text())
        for i, batch in enumerate(report["input_batches"]):
            for key in ["sample_ids", "image_sha256", "pointmap_sha256", "target_sha256"]:
                assert batch[key] == reports["image"]["input_batches"][i][key], (arm, i, key)
            assert batch["target_sha256"] == fit["target_sha256"]
        assert report["input_batches"][0]["sample_ids"] == fit["sample_ids"]
        for key in ["image_sha256", "pointmap_sha256", "target_sha256"]:
            assert report["input_batches"][0][key] == fit[key]
        if arm != "image":
            assert report["input_batches"][0]["features_sha256"] == fit["point_features_sha256"]
        rows = {(r["view_group"], r["condition"]): r["losses"] for r in report["rows"]}
        conditions = ["natural_other_view"]
        if arm != "image":
            conditions += ["anchor_visual_other_surface", "other_visual_anchor_surface", "natural_other_view_swapped_object_surface"]
        expected = {(0, "anchor")} | {(i, c) for i in [1, 2, 3] for c in conditions}
        assert set(rows) == expected and len(report["rows"]) == len(expected)
        assert all(len(v) == 8 and all(math.isfinite(x) for x in v) for v in rows.values())
        assert rows[0, "anchor"] == fit["assessments"][-1]["fresh_noise_native_losses"]
        result = {"anchor_loss": mean(rows[0, "anchor"]), "mean_by_condition": {}, "per_view_group": []}
        for condition in conditions:
            result["mean_by_condition"][condition] = mean(x for i in [1, 2, 3] for x in rows[i, condition])
        for i in [1, 2, 3]:
            result["per_view_group"].append({"view_group": i, **{c: mean(rows[i, c]) for c in conditions}})
        result["natural_other_view_to_anchor_ratio"] = result["mean_by_condition"]["natural_other_view"] / result["anchor_loss"]
        if arm != "image":
            result["mean_features_mse_vs_anchor"] = mean(b["features_mse_vs_anchor"] for b in report["input_batches"][1:])
            result["swapped_worse_paired_draws_of_24"] = sum(b > a for i in [1, 2, 3] for a, b in zip(rows[i, "natural_other_view"], rows[i, "natural_other_view_swapped_object_surface"]))
        summary["arms"][arm] = result
        summary["provenance"][arm] = hashlib.sha256((ROOT / (arm + ".json")).read_bytes()).hexdigest()
        print(arm, json.dumps(result, indent=2))
    summary["limits"] = "Four objects, three new views each, one training seed. Loss draws are paired batch averages, not independent object-level replicas. No decoded new-view geometry. Image/pointmap are changed together."
    (HERE / "view_probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("All pairing, driver hash, source-fit hashes and exact anchor reproduction checks passed.")


if __name__ == "__main__":
    main()
