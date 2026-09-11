"""Validate and summarize a completed GPU probe. Standard library only."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def analyze(data):
    settings = data["settings"]
    assert data.get("paired_input_sha256"), "Use the completed output, not the partial copy"
    expected_ids = None
    summaries = []
    for run in data["runs"]:
        rows = run["rows"]
        ids = sorted({r["sample_id"] for r in rows})
        assert len(ids) == settings["objects"]
        if expected_ids is None:
            expected_ids = ids
        assert ids == expected_ids
        variants = {"true", "cfg_1", "cfg_7"}
        if run["touch_config"] is not None:
            variants |= {"other_object", "reverse_slots"}
        keys = {(r["sample_id"], r["time"], r["draw"], r["variant"]) for r in rows}
        expected = {(sid, t, d, v) for sid in ids for t in settings["times"]
                    for d in range(settings["draws"]) for v in variants}
        assert len(keys) == len(rows) and keys == expected, "Incomplete or duplicate measurements"
        assert all(math.isfinite(r[k]) and r[k] >= 0 for r in rows
                   for k in ["velocity_mse", "prediction_delta_mse"])
        assert all(r["donor_sample_id"] in ids and r["donor_sample_id"] != r["sample_id"]
                   for r in rows if r["variant"] == "other_object")
        check = run["native_loss_check"]
        assert math.isclose(check["native"], check["explicit"], rel_tol=.002, abs_tol=.00002)
        lookup = {(r["sample_id"], r["time"], r["draw"], r["variant"]): r for r in rows}
        times = []
        for t in settings["times"]:
            stats = {}
            e0 = mean(lookup[sid, t, d, "true"]["velocity_mse"]
                      for sid in ids for d in range(settings["draws"]))
            for v in sorted(variants):
                selected = [r for r in rows if r["time"] == t and r["variant"] == v]
                object_diffs = {}
                for sid in ids:
                    object_diffs[sid] = mean(
                        lookup[sid, t, d, v]["velocity_mse"] - lookup[sid, t, d, "true"]["velocity_mse"]
                        for d in range(settings["draws"]))
                stats[v] = {"mean_velocity_mse": mean(r["velocity_mse"] for r in selected),
                            "mean_prediction_delta_mse": mean(r["prediction_delta_mse"] for r in selected),
                            "error_ratio_to_true": mean(r["velocity_mse"] for r in selected) / e0,
                            "mean_error_difference": mean(object_diffs.values()),
                            "object_mean_error_differences": object_diffs,
                            "objects_with_higher_error": sum(x > 0 for x in object_diffs.values())}
            times.append({"time": t, "variants": stats})
        summaries.append({"checkpoint": run["checkpoint"], "checkpoint_step": run["checkpoint_step"],
                          "rows": len(rows), "native_loss_check": check, "times": times})
    return {"settings": settings, "sample_ids": expected_ids, "checks_passed": True,
            "runs": summaries,
            "limits": "Eight training objects, one view each, one checkpoint pair; draws are repeated measures. Donor assignment creates dependencies; no population p-values or bootstrap CIs are claimed. CFG errors are at noised-target states, not sampled trajectories."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.input.read_text())
    result = analyze(data)
    result["input_sha256"] = hashlib.sha256(a.input.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    for r in result["runs"]:
        print(r["checkpoint"])
        for t in r["times"]:
            print(t["time"], {v: round(x["mean_velocity_mse"], 8) for v, x in t["variants"].items()})


if __name__ == "__main__":
    main()
