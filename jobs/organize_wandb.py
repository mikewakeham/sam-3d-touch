"""Preview W&B names/groups for the dataset organization; --apply saves changes."""

import argparse
from pathlib import Path

from organize_runs import run_destination


def organize(runs, apply=False):
    updates = []
    for run in runs:
        source = Path(run.config.get("output_dir") or run.name or "unknown")
        destination = run_destination(source, {"arguments": run.config}, Path("outputs"))
        if destination is None:
            print(f"REVIEW (unchanged): {run.id} {run.name}")
            continue
        group = destination.parent.name
        name = destination.name if group == "zeroverse" else run.name
        if (run.name, run.group) == (name, group):
            continue
        if run.state in ("running", "pending", "preempting"):
            print(f"SKIP active run: {run.id} {run.name}; rerun after it finishes")
            continue
        print(f"UPDATE {run.id}: name {run.name!r} -> {name!r}; group {run.group!r} -> {group!r}")
        updates.append((run, name, group))
    if apply:
        for index, (run, name, group) in enumerate(updates, 1):
            # Official Public API: edit attributes, then persist with update().
            # https://docs.wandb.ai/models/ref/python/public-api/run
            run.name = name
            run.group = group
            run.update()
            print(f"[{index}/{len(updates)}] Updated {run.id}", flush=True)
    print(f"{'Applied' if apply else 'Preview'}: {len(updates)} W&B updates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="mwakeham29/sam-3d-touch", help="entity/project")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    import wandb
    organize(wandb.Api().runs(args.project), apply=args.apply)


if __name__ == "__main__":
    main()
