"""Choose yaw on view 0 / noise 0; assess on view 1 / noise 1 only.

Analysis is specified before seeing GPU results. Per-object yaw selection is a
target-assisted diagnostic, never an inference algorithm or benchmark score.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

YAW = ["identity", "z90", "z180", "z270"]


def analyze(report):
    objects = report["object_ids"]
    losses, sampled = report["loss_rows"], report["sampled_rows"]
    assert len(losses) == len(objects) * 2 * 2 * 3 * 6
    assert len(sampled) == len(objects) * 2 * 2 * 6
    for rows, fields in [(losses, ["object_id", "view", "draw", "time", "rotation"]),
                         (sampled, ["object_id", "view", "draw", "rotation"])]:
        assert len({tuple(r[k] for k in fields) for r in rows}) == len(rows)
    assert all(math.isfinite(r["velocity_mse"]) for r in losses)
    assert all(math.isfinite(r["latent_mse"]) and math.isfinite(r["voxel_iou"]) for r in sampled)
    def loss(oid, rotation, view, draw):
        selected = [r["velocity_mse"] for r in losses if r["object_id"] == oid
                    and r["rotation"] == rotation and r["view"] == view and r["draw"] == draw]
        assert len(selected) == 3
        return mean(selected)
    def sample(oid, rotation):
        selected = [r for r in sampled if r["object_id"] == oid and r["rotation"] == rotation
                    and r["view"] == 1 and r["draw"] == 1]
        assert len(selected) == 1
        return {k: selected[0][k] for k in ["latent_mse", "voxel_iou", "unaligned_voxel_center_chamfer"]}
    rows = []
    for oid in objects:
        selected = min(YAW, key=lambda r: loss(oid, r, 0, 0))
        baseline, changed = loss(oid, "identity", 1, 1), loss(oid, selected, 1, 1)
        rows.append({"object_id": oid, "selected_yaw": selected,
                     "selection_losses": {r: loss(oid, r, 0, 0) for r in YAW},
                     "held_view_fresh_noise_identity_loss": baseline,
                     "held_view_fresh_noise_selected_loss": changed,
                     "held_view_loss_relative_change": changed / baseline - 1,
                     "held_view_identity_sample": sample(oid, "identity"),
                     "held_view_selected_sample": sample(oid, selected)})
    selection_objects = objects[:len(objects)//2]
    test_objects = objects[len(objects)//2:]
    global_yaw = min(YAW, key=lambda r: mean(loss(oid, r, 0, 0) for oid in selection_objects))
    return {"objects": rows, "selected_yaw_counts": dict(Counter(r["selected_yaw"] for r in rows)),
            "global_yaw": {"selected": global_yaw, "selection_objects": selection_objects,
                           "test_objects": test_objects,
                           "test_identity_loss": mean(loss(oid, "identity", 1, 1) for oid in test_objects),
                           "test_selected_loss": mean(loss(oid, global_yaw, 1, 1) for oid in test_objects)},
            "axis_controls_held_view_mean_loss": {rotation: mean(loss(oid, rotation, 1, 1) for oid in objects)
                for rotation in ["identity", "x90_axis_control", "x270_axis_control"]},
            "target_reconstruction": report["target_reconstruction"],
            "limits": "Eight objects / two views is exploratory. Report paired objects, not time draws as independent samples. Per-object rotations use targets during selection; benefits are diagnostic, not deployable performance. No optimization occurred. No rotation benefit does not prove architectural impossibility."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = analyze(json.loads(a.results.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
