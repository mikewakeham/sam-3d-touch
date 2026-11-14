"""ONE-TIME script: replace the sam3d_32views_grasp views whose object barely shows. Delete it afterwards.

The old select_grasp_views.py let through views where the object is a sliver (its low-res occlusion check counted an
unseen object as 0% hidden; no minimum visible size), which crashes SAM 3D's crop. select_grasp_views.py now has the
fix (--min-object-visible-share). This script replaces only the failing views; every other view keeps its files.

  report (default, writes nothing): distribution of the saved low-res visible object share per view and per object
      (worst view), and how many views/objects each cutoff flags. --mask-sizes (JSON list of
      [sample_id, pixels, bbox_w, bbox_h] from the full-res mask scan) checks that the cutoff catches every mask that
      crashes SAM 3D.
  --apply: writes redo_views.json (the failing views) and redo_uids.txt (select_grasp_views.py --uids) to --out-dir,
      backs up grasp_views/summary.json, and moves each affected object's views.json to views_before_redo.json so
      select_grasp_views.py redoes that object. Nothing else is deleted.
  then: select_grasp_views.py --uids $(cat redo_uids.txt) with the same --min-object-visible-share. Selection is
      deterministic, so each view gets the same grasp and settled hand and the same camera candidates as before.
  --finish (GPU job, sam3d-data env): per affected object, views.json = the old views with only the failing ones
      taken from the new selection; renders just those views (into a temporary folder, then copies their image,
      depth, camera and masks in); repackages the object (make_data.package_object: full_surface, samples.json);
      deletes the failing views' camera-frame latents and touch files so those jobs regenerate them; then rewrites
      samples_rendered.jsonl / samples.jsonl. Rerunnable: finished objects are listed in redo_finished.txt.
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

VIEW_FILES = ("image.png", "depth.npy", "camera.npz", "object_mask.png", "hand_mask.png")
STALE_VIEW_FILES = ("target_latent_camframe.npz", "touch_grasp_contact.npz", "simulated_touches.npz")


def progress(label, i, n, started, every):
    if i % every == 0 or i == n:
        elapsed = time.monotonic() - started
        print(f"  {label} [{i}/{n}] elapsed={elapsed / 60:.1f}m eta={elapsed / i * (n - i) / 60:.1f}m", flush=True)


def report(root, args):
    views = []
    paths = sorted((root / "grasp_views").glob("*/views.json"))
    print(f"Reading {len(paths)} views.json files", flush=True)
    started = time.monotonic()
    for i, path in enumerate(paths, 1):
        progress("read", i, len(paths), started, 500)
        doc = json.loads(path.read_text())
        for v in doc["views"]:
            views.append(dict(uid=doc["uid"], object_id=doc["object_id"], view=v["view"],
                              sample_id=f"{doc['object_id']}_{v['view']:03d}", fallback=v["fallback"],
                              visible=v["stats"]["object_visible_share"]))
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
    print(f"  {'cutoff':>8}  {'views to replace':>18}  {'objects touched':>18}  {'fallback views':>14}")
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
    return flagged


def apply(root, out, args, flagged):
    in_progress = sorted(root.glob("grasp_views/*/views_before_redo.json"))
    if (out / "redo_views.json").exists() or in_progress:
        raise SystemExit(f"A redo is already in progress ({out / 'redo_views.json'} or {len(in_progress)} "
                         "views_before_redo.json files exist); continue with the selection rerun / --finish instead")
    # Objects that never rendered (in generated_data/failed_objects.json) are not in the dataset: leave them alone.
    unrendered = {v["object_id"] for v in flagged if not (root / "generated_data" / v["object_id"] / "samples.json").is_file()}
    if unrendered:
        print(f"skipping {len(unrendered)} flagged objects that have no renders: {sorted(unrendered)}")
        flagged = [v for v in flagged if v["object_id"] not in unrendered]
    objects = sorted({(v["uid"], v["object_id"]) for v in flagged})
    (out / "redo_views.json").write_text(json.dumps(dict(min_object_visible_share=args.min_object_visible_share,
                                                         views=flagged), indent=1))
    (out / "redo_uids.txt").write_text(" ".join(u for u, _ in objects) + "\n")
    print(f"wrote {out / 'redo_views.json'} and {out / 'redo_uids.txt'}")
    summary = root / "grasp_views" / "summary.json"
    if summary.is_file() and not summary.with_name("summary_before_redo.json").exists():
        shutil.copy2(summary, summary.with_name("summary_before_redo.json"))
        print(f"backed up {summary}")
    started = time.monotonic()
    for i, (_, oid) in enumerate(objects, 1):
        d = root / "grasp_views" / oid
        if (d / "views.json").exists() and not (d / "views_before_redo.json").exists():
            (d / "views.json").rename(d / "views_before_redo.json")
        progress("set aside", i, len(objects), started, 200)
    share = f" --min-object-visible-share {args.min_object_visible_share:g}"
    print(f"Moved {len(objects)} views.json aside. Next, from sam-3d-touch:")
    print(f"  1. sbatch data_generation/general/jobs/select_grasp_views_objaverse_oa.sbatch --uids $(cat {out / 'redo_uids.txt'}){share}")
    print("  2. sbatch --gres=gpu:1 --cpus-per-task=32 --mem=64G data_generation/general/jobs/redo_small_object_views_once.sbatch "
          f"--out-dir {out} --render-workers-per-gpu 8 --blender-threads 4")
    print(f"  3. rm {root / 'generated_data' / 'target_latents_camframe.json'} && "
          "sbatch --mem=128G data_generation/general/jobs/make_latents_objaverse_oa_32views_grasp_camframe.sbatch")
    print("  4. the make_contact_touch full-run command (only the replaced views are redone)")


def finish_object(task):
    """Merge, render the failing views, copy their files in, repackage. Runs in a worker process."""
    import make_data
    oid, view_ids, ctx = task
    root, out = Path(ctx["root"]), Path(ctx["out"])
    gv = root / "grasp_views" / oid
    if not (gv / "views_before_redo.json").exists() and (gv / "views.json").exists():
        return oid, 0, 0, []  # finished before a crash, but not yet recorded in redo_finished.txt
    if not (gv / "views.json").exists():
        raise FileNotFoundError(f"{oid}: no new views.json; the selection rerun did not produce this object")
    old = json.loads((gv / "views_before_redo.json").read_text())
    new = json.loads((gv / "views.json").read_text())
    if [v["view"] for v in old["views"]] != [v["view"] for v in new["views"]]:
        raise ValueError(f"{oid}: view lists differ between the old and new selection")
    merged = dict(new, views=[new["views"][i] if i in view_ids else old["views"][i] for i in range(len(old["views"]))])
    changed_grasp = sum(merged["views"][i]["grasp_row"] != old["views"][i]["grasp_row"] for i in view_ids)
    still_small = [i for i in view_ids
                   if merged["views"][i]["stats"]["object_visible_share"] < ctx["min_visible"]]
    tmp = out / "redo_render_tmp" / oid
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / "views.json").write_text(json.dumps(dict(merged, views=[merged["views"][i] for i in sorted(view_ids)])))
    s = ctx["settings"]
    obj = ctx["objects"][oid]
    digest = make_data.hashlib.sha256(f"render-v1:{s['seed']}:{oid}".encode()).digest()
    command = [ctx["blender"], "--background", "--factory-startup", "--disable-autoexec",
               "--threads", str(ctx["blender_threads"]), "--python-exit-code", "1",
               "--python", str(Path(make_data.__file__).with_name("render_grasp_views.py")), "--",
               "--object", obj["model_path"], "--output", str(tmp / "render"), "--views", str(tmp / "views.json"),
               "--hand", str(gv.parent / "hand_template.npz"), "--num-views", str(len(view_ids)),
               "--resolution", str(s["resolution"]), "--seed", str(int.from_bytes(digest[:8], "big")),
               "--samples", str(s["samples"]), "--device", s["render_device"], "--frame", str(s["frame"]),
               "--lighting", s["lighting"]]
    gpu = ctx["slots"].get()
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu) if gpu is not None else None
        with (tmp / "blender.log").open("w") as log:
            if subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode:
                raise RuntimeError(f"{oid}: Blender failed; see {tmp / 'blender.log'}")
    finally:
        ctx["slots"].put(gpu)
    object_dir = root / "generated_data" / oid
    for i in view_ids:
        src, dst = tmp / "render" / "views" / f"{i:03d}", object_dir / "views" / f"{i:03d}"
        for name in VIEW_FILES:
            shutil.copy2(src / name, dst / name)
        for name in STALE_VIEW_FILES:
            (dst / name).unlink(missing_ok=True)
    tmp_views = gv / "views.json.tmp"
    tmp_views.write_text(json.dumps(merged))
    tmp_views.replace(gv / "views.json")
    pack = make_data.argparse.Namespace(data_root=root, grasp_views=gv.parent, **{
        k: s[k] for k in ("num_views", "num_points", "resolution", "seed", "surface_pool_points", "visibility_tolerance")})
    make_data.package_object(obj, ctx["splits"][oid], pack)
    (gv / "views_before_redo.json").unlink()
    shutil.rmtree(tmp)
    return oid, len(view_ids), changed_grasp, still_small


def finish(root, out, args):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import make_data
    redo = json.loads((out / "redo_views.json").read_text())
    by_object = {}
    for v in redo["views"]:
        by_object.setdefault(v["object_id"], set()).add(v["view"])
    done_path = out / "redo_finished.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    pending = [(o, ids) for o, ids in sorted(by_object.items()) if o not in done]
    generated = root / "generated_data"
    settings = json.loads((generated / "settings.json").read_text())
    objects = {o["object_id"]: o for o in json.loads((generated / "objects.json").read_text())}
    splits = {o: split for split, ids in json.loads((generated / "splits.json").read_text()).items() for o in ids}
    gpus = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip() and g.strip() != "-1"]
    print(f"Replacing {sum(len(ids) for _, ids in pending)} views in {len(pending)} objects "
          f"({len(done)} objects already finished) on GPUs {gpus or ['none']}", flush=True)
    failures = []
    if pending:
        with mp.Manager() as manager:
            slots = manager.Queue()
            for slot in (gpus * args.render_workers_per_gpu) or [None]:
                slots.put(slot)
            ctx = dict(root=str(root), out=str(out), settings=settings, splits=splits, slots=slots,
                       objects={o: objects[o] for o, _ in pending}, blender=str(args.blender),
                       blender_threads=args.blender_threads, min_visible=redo["min_object_visible_share"])
            workers = max(1, len(gpus) * args.render_workers_per_gpu)
            started = time.monotonic()
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
                futures = {pool.submit(finish_object, (o, ids, ctx)): o for o, ids in pending}
                for i, future in enumerate(as_completed(futures), 1):
                    oid = futures[future]
                    try:
                        oid, n, changed_grasp, still_small = future.result()
                        with done_path.open("a") as f:
                            f.write(oid + "\n")
                        note = f"{n} views replaced" + (f", {changed_grasp} with a different grasp" if changed_grasp else "") \
                            + (f", still under the cutoff (fallback) {still_small}" if still_small else "")
                    except Exception as error:
                        failures.append(dict(object_id=oid, error=f"{type(error).__name__}: {error}"))
                        note = f"FAILED {type(error).__name__}: {error}"
                    elapsed = time.monotonic() - started
                    print(f"  [{i}/{len(pending)}] elapsed={elapsed / 60:.1f}m eta={elapsed / i * (len(pending) - i) / 60:.1f}m "
                          f"{oid}: {note}", flush=True)
    if failures:
        (out / "redo_failures.json").write_text(json.dumps(failures, indent=1))
        raise SystemExit(f"{len(failures)} objects failed (see {out / 'redo_failures.json'}); rerun --finish to retry them")
    rendered, ready = make_data.write_manifests(argparse.Namespace(data_root=root), list(objects.values()))
    print(f"Rewrote manifests: {len(rendered)} rendered samples, {len(ready)} training-ready samples. "
          f"Next: camera-frame latents (delete generated_data/target_latents_camframe.json first), then contact touch.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--min-object-visible-share", type=float, default=0.008,
                   help="must equal the select_grasp_views.py --min-object-visible-share used for the redo")
    p.add_argument("--mask-sizes", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path.cwd())
    p.add_argument("--apply", action="store_true", help="list the failing views and set their objects' views.json aside")
    p.add_argument("--finish", action="store_true", help="after the selection rerun: merge, render, repackage")
    p.add_argument("--blender", type=Path, default=Path("/n/holylabs/qianqian_lab/Lab/mwakeham/blender/"
                                                        "blender-4.5.9-linux-x64/blender"))
    p.add_argument("--blender-threads", type=int, default=6)
    p.add_argument("--render-workers-per-gpu", type=int, default=2)
    args = p.parse_args()
    root, out = args.data_root.resolve(), args.out_dir.resolve()
    if args.finish:
        finish(root, out, args)
        return
    flagged = report(root, args)
    print(f"cutoff {args.min_object_visible_share:g}: {len(flagged)} views to replace in "
          f"{len({v['object_id'] for v in flagged})} objects")
    if not args.apply:
        print("Report only, nothing written.")
        return
    apply(root, out, args, flagged)


if __name__ == "__main__":
    main()
