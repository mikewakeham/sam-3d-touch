"""Freeze a category-balanced Toys4K selection for make_data --objects-manifest."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(metadata, count=100, seed=29, min_score=4.5):
    groups = defaultdict(list)
    seen_hashes, seen_paths = set(), set()
    with metadata.open(newline="") as source:
        rows = sorted(csv.DictReader(source), key=lambda row: row["file_identifier"])
    for row in rows:
        score = float(row["aesthetic_score"])
        if not math.isfinite(score) or score < min_score:
            continue
        relative = Path(row["file_identifier"])
        digest = row["sha256"]
        if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2
                or relative.suffix.lower() != ".blend"):
            raise ValueError(f"Invalid asset path: {relative}")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid SHA256: {relative}")
        if relative.as_posix() in seen_paths:
            raise ValueError(f"Duplicate metadata path: {relative}")
        seen_paths.add(relative.as_posix())
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        category = relative.parts[0]
        groups[category].append(dict(source_relative_path=relative.as_posix(),
                                     sha256=digest, category=category, aesthetic_score=score))
    if sum(map(len, groups.values())) < count:
        raise ValueError(f"Fewer than {count} eligible unique assets")
    rng = random.Random(seed)
    categories = sorted(groups)
    rng.shuffle(categories)
    for category in categories:
        rng.shuffle(groups[category])
    # One object per category per round, before revisiting a category.
    ordered = [groups[category][index]
               for index in range(max(map(len, groups.values())))
               for category in categories if index < len(groups[category])]
    return dict(format_version=1, selection_method="category_balanced_round_robin_v1",
                metadata_sha256=sha256(metadata), seed=seed, min_aesthetic_score=min_score,
                count=count, objects=ordered[:count], reserves=ordered[count:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parent
    parser.add_argument("--metadata", type=Path, default=directory / "Toys4k.csv")
    parser.add_argument("--objects-root", type=Path, required=True,
                        help="Directory containing category folders, or its extracted parent")
    parser.add_argument("--output", type=Path, default=directory / "selection.json")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--min-score", type=float, default=4.5)
    args = parser.parse_args()
    if args.count < 1 or not math.isfinite(args.min_score):
        parser.error("count must be positive and min-score must be finite")
    selection = select(args.metadata, args.count, args.seed, args.min_score)
    if args.output.exists() and json.loads(args.output.read_text()) != selection:
        raise ValueError("Existing selection differs; use a new output path, not silent reselection")
    root = args.objects_root.expanduser().resolve()
    if (root / "toys4k_blend_files").is_dir():
        root = root / "toys4k_blend_files"
    for asset in selection["objects"]:
        path = root / asset["source_relative_path"]
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Asset escapes objects root: {path}")
        if sha256(path) != asset["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")
    # No automatic replacements: missing/corrupt assets must be investigated.
    if not args.output.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(json.dumps(selection, indent=2) + "\n")
    print(f"Verified {len(selection['objects'])} selected assets")
    print(f"Categories: {dict(sorted(Counter(a['category'] for a in selection['objects']).items()))}")
    print(f"Frozen selection: {args.output}")
    print(f"Objects root for make_data: {root}")
    print("Reserves are ordered; use the first unused same-category reserve if a replacement is approved.")


if __name__ == "__main__":
    main()
