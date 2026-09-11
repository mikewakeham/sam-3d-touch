"""Validate completed rollout coverage and report object-paired descriptive results."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

METRICS = ["latent_mse", "voxel_iou", "unaligned_voxel_center_chamfer", "predicted_occupied"]


def summarize(data):
    assert data.get("paired_inputs_sha256"), "Use completed results.json"
    ids = data["sample_ids"]
    seeds, strengths = data["settings"]["seeds"], data["settings"]["strengths"]
    assert len(ids) == len(set(ids)) and len(seeds) == len(set(seeds))
    assert len(strengths) == len(set(strengths))
    target_counts = {}
    runs = []
    for run in data["runs"]:
        rows = run["rows"]
        lookup = {(r["sample_id"], r["seed"], r["strength"]): r for r in rows}
        expected = {(i, s, g) for i in ids for s in seeds for g in strengths}
        assert len(lookup) == len(rows) and set(lookup) == expected
        assert run["no_shortcut"] is True
        assert {r["strength"] for r in run["native_cfg_checks"]} == set(strengths)
        assert all(r["formula_max_error"] == 0 for r in run["native_cfg_checks"])
        for row in rows:
            if row.get("failure"):
                continue
            for metric in METRICS:
                v = row[metric]
                assert v is None or (math.isfinite(v) and v >= 0)
            assert row["voxel_iou"] <= 1
            assert 0 <= row["predicted_occupied"] <= 64**3
            assert 0 <= row["target_occupied"] <= 64**3
            sid = row["sample_id"]
            if sid in target_counts:
                assert row["target_occupied"] == target_counts[sid]
            target_counts[sid] = row["target_occupied"]
        settings_summary = []
        for g in strengths:
            subset = [r for r in rows if r["strength"] == g]
            valid = [r for r in subset if not r.get("failure")]
            stats = {}
            for metric in METRICS:
                vals = [r[metric] for r in valid if r[metric] is not None]
                object_values = {sid: [lookup[sid, s, g].get(metric) for s in seeds] for sid in ids}
                # Only complete objects enter paired descriptive comparisons.
                object_means = {sid: mean(v) for sid, v in object_values.items() if all(x is not None for x in v)}
                stats[metric] = {"mean_valid": mean(vals) if vals else None,
                                 "valid_measurements": len(vals),
                                 "median_object_mean": median(object_means.values()) if object_means else None,
                                 "object_means": object_means}
            settings_summary.append({"strength": g, "expected_measurements": len(subset),
                "failed_measurements": len(subset) - len(valid), "metrics": stats})
        comparisons = []
        if 0 in strengths:
            for g in strengths:
                if g == 0:
                    continue
                for metric in METRICS[:3]:
                    diffs = {}
                    for sid in ids:
                        pairs = [(lookup[sid, s, g].get(metric), lookup[sid, s, 0].get(metric)) for s in seeds]
                        if all(x is not None and y is not None for x, y in pairs):
                            diffs[sid] = mean(x - y for x, y in pairs)
                    comparisons.append({"strength": g, "reference_strength": 0, "metric": metric,
                        "object_differences": diffs, "mean_difference": mean(diffs.values()) if diffs else None,
                        "objects_improved": sum(d > 0 if metric == "voxel_iou" else d < 0 for d in diffs.values())})
        runs.append({"checkpoint": run["checkpoint"], "checkpoint_step": run["checkpoint_step"],
                     "settings": settings_summary, "paired_comparisons": comparisons})
    return {"checks_passed": True, "sample_ids": ids, "runs": runs,
            "limits": "Descriptive pilot: 8 training objects, 2 seeds, 2 different validation-selected checkpoints. Seeds are repeated measurements; no population significance claim. Metrics have not yet been independently recomputed from NPZ artifacts."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = summarize(json.loads(a.input.read_text()))
    result["input_sha256"] = hashlib.sha256(a.input.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    for run in result["runs"]:
        print(run["checkpoint"])
        for s in run["settings"]:
            print(s["strength"], {k: round(v["mean_valid"], 7) if v["mean_valid"] is not None else None
                                   for k, v in s["metrics"].items()}, "failures", s["failed_measurements"])


if __name__ == "__main__":
    main()
