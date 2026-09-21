"""Preview historical output/log moves; use --apply only after their jobs stop."""

import argparse
import re
from pathlib import Path

import yaml


OBJAVERSE_CONFIGS = {
    "data1.yaml", "data_full_surface.yaml", "data_oa_touch.yaml", "data_oa_full_surface.yaml",
}


def run_destination(source, config, outputs):
    arguments = config.get("arguments", {})
    dataset = config.get("data", {}).get("dataset", {})
    data_config = Path(arguments.get("data_config") or
                       arguments.get("selection_data_config") or "").name
    if data_config == "data_zeroverse_5000_8views_full_surface.yaml" or "zeroverse_5000" in str(dataset.get("root", "")):
        # The old "image" runs actually included pointmaps. Trust saved flags.
        no_pointmap = arguments.get("no_pointmap")
        if not isinstance(no_pointmap, bool) or arguments.get("no_visual", False):
            return None
        name = source.name.removeprefix("zeroverse_").removeprefix("8views_")
        condition = "image" if no_pointmap else "pointmap"
        if name.startswith(("image", "pointmap")):
            name = condition + name.removeprefix("image").removeprefix("pointmap")
        elif name.startswith(("craftsman_", "triposg_")):
            name = condition + "_" + name
        else:
            return None
        if name.endswith("_lr1e5") and arguments.get("cross_attention_learning_rate") == 1e-5:
            name = name.removesuffix("_lr1e5")
        return outputs / "zeroverse" / ("zeroverse_" + name)
    if data_config in OBJAVERSE_CONFIGS or "objaverse" in str(dataset.get("root", "")).lower():
        return outputs / "objaverse" / source.name
    return None


def plan_moves(repo):
    outputs, logs = repo / "outputs", repo / "logs"
    moves, review = [], []
    grouped = {"zeroverse", "zeroverse_8views", "objaverse", "objaverse_1k"}
    containers = {"conditioning_investigation", "evaluation", "evaluation_coordinate_system"}
    roots = [outputs] + [outputs / name for name in sorted(grouped)]
    for root in roots:
        if not root.is_dir():
            continue
        for source in sorted(root.iterdir()):
            if not source.is_dir() or (root == outputs and source.name in grouped):
                continue
            config_path = source / "config.yaml"
            if config_path.is_file():
                destination = run_destination(source, yaml.safe_load(config_path.read_text()), outputs)
            elif source.name in containers:
                # Historical Objaverse evaluation/conditioning directories.
                destination = outputs / "objaverse" / source.name
            else:
                destination = None
            if destination is None:
                review.append(source)
            elif source != destination:
                moves.append((source, destination))

    # Keep historical log filenames, including old experiment labels.
    log_roots = [(logs, None), (logs / "zeroverse_8views", "zeroverse"),
                 (logs / "objaverse_1k", "objaverse")]
    for root, group in log_roots:
        if not root.is_dir():
            continue
        for source in sorted(root.iterdir()):
            if not source.is_file():
                continue
            destination_group = group
            if group is None:
                if re.match(r"zv_.+-\d+\.(out|err)$", source.name):
                    destination_group = "zeroverse"
                elif re.match(r"(?:s1_.+|evaluate|eval_orbits_geometry)-\d+\.(out|err)$", source.name):
                    destination_group = "objaverse"
            if destination_group:
                moves.append((source, logs / destination_group / source.name))
            else:
                review.append(source)
    return moves, review


def check_moves(moves):
    destinations = set()
    for source, destination in moves:
        if source.is_symlink():
            raise ValueError(f"Review symlink manually: {source}")
        if destination.exists() or destination.is_symlink() or destination in destinations:
            raise FileExistsError(f"Refusing to overwrite or merge: {destination}")
        destinations.add(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo = Path.cwd()
    if not (repo / "train.py").is_file():
        parser.error("Run from the sam-3d-touch repository root")
    moves, review = plan_moves(repo)
    for source, destination in moves:
        print(f"MOVE {source.relative_to(repo)} -> {destination.relative_to(repo)}")
    for source in review:
        print(f"REVIEW (left unchanged): {source.relative_to(repo)}")
    check_moves(moves)
    if not args.apply:
        print(f"Preview: {len(moves)} moves. Stop affected jobs, then rerun with --apply.")
        return
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    for group in ("zeroverse", "objaverse"):
        (repo / "logs" / group).mkdir(parents=True, exist_ok=True)
    print(f"Moved {len(moves)} paths. Checkpoints/configs and W&B online records were not edited.")


if __name__ == "__main__":
    main()
