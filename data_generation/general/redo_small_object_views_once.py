"""ONE-TIME script for the sam3d_32views_grasp targeted redo of views whose object barely shows. Delete it afterwards.

The old select_grasp_views.py let through views where the object is a sliver (its low-res occlusion check counted an
unseen object as 0% hidden; no minimum visible size), which crashes SAM 3D's crop. select_grasp_views.py now has the
fix. This script prepares the redo of only the affected objects:

  report (default, writes nothing): reads DATA_ROOT/grasp_views/*/views.json, prints how many views/objects each
      cutoff on the saved low-res visible object share would flag, and cross-checks the full-res mask scan
      (--mask-sizes: JSON list of [sample_id, pixels, bbox_w, bbox_h]).
  --apply: also writes redo_object_ids.json / redo_uids.txt to --out-dir (inputs for the next jobs), backs up grasp_views/summary.json and generated_data/failed_objects.json (both overwritten by the
      redo), deletes each flagged object's views.json + candidates.npz and its generated_data/<object_id> folder,
      and prints the job commands to run next.
"""
import argparse
import json
import shutil
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--min-object-visible-share", type=float, default=0.002,
                   help="must equal the select_grasp_views.py --min-object-visible-share used for the redo")
    p.add_argument("--mask-sizes", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path.cwd())
    p.add_argument("--apply", action="store_true", help="back up, delete the flagged objects' views and outputs")
    args = p.parse_args()
    root, out = args.data_root.resolve(), args.out_dir.resolve()

    views = []
    paths = sorted((root / "grasp_views").glob("*/views.json"))
    print(f"Reading {len(paths)} views.json files", flush=True)
    started = time.monotonic()
    for i, path in enumerate(paths, 1):
        if i % 500 == 0 or i == len(paths):
            elapsed = time.monotonic() - started
            print(f"  [{i}/{len(paths)}] elapsed={elapsed / 60:.1f}m eta={elapsed / i * (len(paths) - i) / 60:.1f}m",
                  flush=True)
        doc = json.loads(path.read_text())
        for v in doc["views"]:
            views.append(dict(uid=doc["uid"], object_id=doc["object_id"], sample_id=f"{doc['object_id']}_{v['view']:03d}",
                              fallback=v["fallback"], visible=v["stats"]["object_visible_share"]))
    n_objects = len({v["object_id"] for v in views})
    print(f"{len(views)} views in {n_objects} objects")
    vis = sorted(v["visible"] for v in views)
    worst = {}
    for v in views:
        worst[v["object_id"]] = min(worst.get(v["object_id"], 1.0), v["visible"])
    worst = sorted(worst.values())
    pct = (0.1, 0.5, 1, 2, 5, 10, 25, 50)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p / 100 * len(xs)))]
    print("Low-res visible object share (fraction of the image the object covers where the hand does not hide it)")
    print("  per view:            " + "  ".join(f"p{p:g}={q(vis, p):.4f}" for p in pct))
    print("  per object (worst):  " + "  ".join(f"p{p:g}={q(worst, p):.4f}" for p in pct))
    print(f"  {'cutoff':>8}  {'views below':>18}  {'objects to redo':>18}  {'fallback views':>14}")
    for cut in sorted({0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03,
                       args.min_object_visible_share}):
        f = [v for v in views if v["visible"] < cut]
        n_obj = len({v["object_id"] for v in f})
        print(f"  {cut:>8g}  {len(f):>8} ({100 * len(f) / len(views):5.2f}%)  {n_obj:>8} ({100 * n_obj / n_objects:5.1f}%)"
              f"  {sum(v['fallback'] for v in f):>14}")

    flagged = [v for v in views if v["visible"] < args.min_object_visible_share]
    if args.mask_sizes:
        sizes = json.loads(args.mask_sizes.read_text())
        ids = {v["sample_id"] for v in flagged}
        crash = [s for s in sizes if min(s[2], s[3]) < 2]
        missed = [s for s in crash if s[0] not in ids]
        print(f"full-res masks under 2 px (crash SAM 3D): {len(crash)}; not caught by {args.min_object_visible_share:g}: "
              f"{len(missed)}")
        for s in missed[:20]:
            print("  missed:", s)
    objects = sorted({(v["uid"], v["object_id"]) for v in flagged})
    print(f"cutoff {args.min_object_visible_share:g}: {len(flagged)} views, {len(objects)} objects to redo")
    if not args.apply:
        print("Report only, nothing written; rerun with --apply to back up and delete the flagged objects' views "
              "and outputs.")
        return
    # Inputs for the next jobs: select_grasp_views.py --uids and make_data.py --object-ids
    (out / "redo_object_ids.json").write_text(json.dumps([o for _, o in objects], indent=1))
    (out / "redo_uids.txt").write_text(" ".join(u for u, _ in objects) + "\n")
    print(f"wrote {out / 'redo_object_ids.json'} and {out / 'redo_uids.txt'}")

    for src in (root / "grasp_views" / "summary.json", root / "generated_data" / "failed_objects.json"):
        backup = src.with_name(src.stem + "_before_redo.json")
        if src.is_file() and not backup.exists():
            shutil.copy2(src, backup)
            print(f"backed up {src} -> {backup}")
    started = time.monotonic()
    for i, (_, oid) in enumerate(objects, 1):
        for name in ("views.json", "candidates.npz"):
            (root / "grasp_views" / oid / name).unlink(missing_ok=True)
        shutil.rmtree(root / "generated_data" / oid, ignore_errors=True)
        if i % 100 == 0 or i == len(objects):
            elapsed = time.monotonic() - started
            print(f"  deleted [{i}/{len(objects)}] elapsed={elapsed / 60:.1f}m "
                  f"eta={elapsed / i * (len(objects) - i) / 60:.1f}m", flush=True)
    print(f"deleted views and outputs of {len(objects)} objects. Next, from sam-3d-touch:")
    share = "" if args.min_object_visible_share == 0.002 else f" --min-object-visible-share {args.min_object_visible_share:g}"
    print(f"  1. sbatch data_generation/general/jobs/select_grasp_views_objaverse_oa.sbatch --uids $(cat {out / 'redo_uids.txt'}){share}")
    print("  2. git show 708ffd8:data_generation/general/generate_target_latents.py > data_generation/general/generate_target_latents.py")
    print(f"     sbatch --array=0 data_generation/general/jobs/objaverse_oa_32views_grasp.sbatch --object-ids {out / 'redo_object_ids.json'}")
    print("     after it finishes: git checkout -- data_generation/general/generate_target_latents.py")
    print(f"  3. rm {root / 'generated_data' / 'target_latents_camframe.json'} && "
          "sbatch data_generation/general/jobs/make_latents_objaverse_oa_32views_grasp_camframe.sbatch")
    print("  4. the make_contact_touch full-run command (finished views are skipped)")


if __name__ == "__main__":
    main()
