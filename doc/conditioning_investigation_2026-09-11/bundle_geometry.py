"""Bundle selected saved geometry and inputs for CPU inspection; no GPU needed.

Run in the existing cluster repository. Reads existing files, creates one new
ZIP, and fails before writing if required files are absent. Does not launch jobs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

CASES = {
    "479bbf9062de4b3196f08e52d1c12821_000": "Persistent high error in both checkpoints and seeds",
    "e846574df9334973955c80286af47ba7_007": "Second persistent high-error example",
    "fe200ce00a5d4c298eee89de0fc15f01_011": "Full-surface checkpoint seed-sensitive failure",
    "c5867189a04446d7b20b79fcc24c6afa_004": "Image checkpoint seed-sensitive failure",
    "146b6c1f10bd445da977424c64dbb051_002": "Low-CD control in both checkpoints with substantial target support",
}


def under(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def collect(rollout_dir, probe_path, data_root=None):
    results_path = rollout_dir / "results.json"
    results = json.loads(results_path.read_text())
    if not results.get("paired_inputs_sha256"):
        raise ValueError("A completed results.json is required")
    probe = json.loads(probe_path.read_text())
    surface_run = next(r for r in probe["runs"] if r["touch_config"] is not None)
    dataset = surface_run["run_config"]["data"]["dataset"]
    data_root = Path(data_root or dataset["root"])
    manifest = under(data_root, dataset["manifest"])
    records = {r["sample_id"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    files = {"results.json": results_path, "probe.json": probe_path}
    selected_records = {}
    for sid in CASES:
        if sid not in results["sample_ids"]:
            raise ValueError(f"Case not present in rollout: {sid}")
        selected_records[sid] = records[sid]
        for field in ["image_path", "pointmap_path", "camera_path", "full_surface_path", "object_transform_path", "target_path"]:
            relative = records[sid][field]
            path = under(data_root, relative)
            files["data/" + str(path.relative_to(data_root.resolve()))] = path
    expected = len(CASES) * len(results["runs"]) * len(results["settings"]["seeds"])
    artifacts = 0
    for run in results["runs"]:
        for row in run["rows"]:
            if row["sample_id"] in CASES and row["strength"] == 7:
                if row.get("failure") or not row.get("artifact"):
                    raise ValueError(f"Missing successful artifact for {row['sample_id']}")
                relative = row["artifact"]
                path = under(rollout_dir, relative)
                files["rollouts/" + str(path.relative_to(rollout_dir.resolve()))] = path
                artifacts += 1
    if artifacts != expected:
        raise ValueError(f"Expected {expected} CFG-7 artifacts; found {artifacts}")
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required inputs missing:\n" + "\n".join(missing))
    return files, {"cases": CASES, "records": selected_records,
                   "artifact_prefix": "rollouts", "data_prefix": "data",
                   "selection": "Diagnostic examples selected after observing outcomes; not an unbiased evaluation subset."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-dir", type=Path, required=True)
    p.add_argument("--probe-json", type=Path, default=Path(__file__).with_name("gpu_probe_train_46083371.json"))
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    files, metadata = collect(a.rollout_dir, a.probe_json, a.data_root)
    metadata["files"] = [{"archive_path": name, "source_path": str(path), "bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in files.items()]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("bundle_manifest.json", json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {a.output}: {len(files)} source files, {a.output.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
