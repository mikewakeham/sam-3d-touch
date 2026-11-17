"""Write index.html next to make_orbit.py's orbits: one row per sample with the input image and every condition's
orbit side by side (metrics underneath), the input-point orbits in a dropdown, and a download link per GIF.

    python -m evaluation.make_orbit_gallery --evaluation-dir outputs/objaverse_oa/evaluation_grasp_camframe

Open orbits/index.html, or serve the folder (python -m http.server) so download links save with readable names.
Rerun after rendering more orbits; it only reads files that exist. The grid shows small preview GIFs (saved in each
sample's preview/ folder, remade only when the orbit changes); enlarging or downloading uses the full GIF.
"""
import argparse
import csv
import html
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageSequence

FIXED_LABELS = {"official": "SAM 3D (official)", "decoded_gt": "decoded GT"}
# input_visualizations.input_groups names; longer prefixes first.
INPUT_LABELS = [("input_mesh_textured", "mesh (textured)"), ("input_mesh_pointmap_surface", "mesh + pointmap + points"),
                ("input_mesh_pointmap", "mesh + pointmap"), ("input_mesh_surface", "mesh + points"),
                ("input_mesh", "mesh"), ("input_pointmap_surface", "pointmap + points"),
                ("input_pointmap", "pointmap"), ("input_surface", "points")]
METRICS = [("fscore_0.01", "F@1%", 3), ("chamfer", "CD", 4)]

CSS = """
:root { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color: #222; background: #fff; }
body { margin: 0; font-size: 15px; }
.page { width: min(100% - 32px, 1600px); margin: 24px auto 64px; }
h1 { font-size: 22px; font-weight: 500; margin: 0 0 16px; }
.controls { display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: center; padding: 10px 0;
            border-bottom: 1px solid #ddd; position: sticky; top: 0; background: #fff; z-index: 2; }
.controls label { white-space: nowrap; }
button { font: inherit; background: #fff; border: 1px solid #bbb; padding: 3px 10px; cursor: pointer; }
.sample { border-bottom: 1px solid #ddd; padding: 14px 0; }
.sample-id { color: #555; margin-bottom: 8px; font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(var(--size, 240px), 100%), 1fr));
        gap: 10px; }
.cell img { cursor: zoom-in; }
#zoom { position: fixed; inset: 0; background: rgba(255, 255, 255, 0.97); z-index: 10; display: flex;
        flex-direction: column; align-items: center; justify-content: center; gap: 8px; }
#zoom img { width: min(94vw, 84vh); height: min(94vw, 84vh); object-fit: contain; border: 1px solid #eee;
            cursor: zoom-out; }
#zoom .name, #zoom .meta { font-size: 14px; text-align: center; }
.cell { min-width: 0; }
.cell img { width: 100%; aspect-ratio: 1; object-fit: contain; display: block; border: 1px solid #eee;
            background: #fff; }
.name { font-size: 13px; margin-top: 4px; overflow-wrap: anywhere; }
.meta { font-size: 12px; color: #555; }
.meta a { color: #2859a5; text-decoration: none; margin-right: 8px; }
.meta a:hover { text-decoration: underline; }
details { margin-top: 10px; }
summary { cursor: pointer; color: #555; font-size: 13px; }
details .grid { margin-top: 8px; }
.hidden { display: none !important; }
"""

JS = """
const boxes = [...document.querySelectorAll('input[data-cond]')];
function apply() {
  const off = new Set(boxes.filter(b => !b.checked).map(b => b.dataset.cond));
  const mode = document.querySelector('input[name=mode]:checked');
  document.querySelectorAll('.cell[data-cond]').forEach(c => {
    c.classList.toggle('hidden', off.has(c.dataset.cond) || (mode && c.dataset.mode && c.dataset.mode !== mode.value));
  });
}
const size = document.getElementById('size');
size.addEventListener('input', () => document.documentElement.style.setProperty('--size', size.value + 'px'));
// Click a GIF to see it large; arrow keys step through the visible ones, Esc or a click closes.
const zoom = document.getElementById('zoom');
let current = null;
function visible() { return [...document.querySelectorAll('.cell img')].filter(i => i.offsetParent !== null); }
function show(img) {
  current = img;
  const cell = img.closest('.cell');
  zoom.innerHTML = '<img src="' + (img.dataset.full || img.dataset.src) + '" alt="">' + cell.querySelector('.name').outerHTML
    + cell.querySelector('.meta').outerHTML + '<div class="meta">' + cell.closest('.sample').querySelector('.sample-id').textContent + '</div>';
  zoom.classList.remove('hidden');
}
function close() { zoom.classList.add('hidden'); current = null; }
document.querySelectorAll('.cell img').forEach(img => img.addEventListener('click', () => show(img)));
// Load an image only while it is on screen (plus a small margin) and unload it once it scrolls away, so only the
// visible GIFs download and animate.
const onScreen = new IntersectionObserver(entries => entries.forEach(e => {
  const img = e.target;
  if (e.isIntersecting) { if (!img.getAttribute('src')) img.src = img.dataset.src; }
  else if (img.getAttribute('src')) { img.removeAttribute('src'); img.dataset.tries = 0; }
}), { rootMargin: '300px 0px' });
document.querySelectorAll('.cell img').forEach(img => onScreen.observe(img));
// Retry images that fail to load (dropped requests through a port forward), up to three times.
document.querySelectorAll('.cell img').forEach(img => img.addEventListener('error', () => {
  const tries = Number(img.dataset.tries || 0);
  if (tries >= 3 || !img.getAttribute('src')) return;
  img.dataset.tries = tries + 1;
  setTimeout(() => { if (img.getAttribute('src')) img.src = img.dataset.src + '?retry=' + (tries + 1); },
             1000 * (tries + 1));
}));
zoom.addEventListener('click', e => { if (!e.target.closest('a')) close(); });
document.addEventListener('keydown', e => {
  if (!current) return;
  if (e.key === 'Escape') close();
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    const list = visible(), i = list.indexOf(current);
    if (i >= 0) show(list[(i + (e.key === 'ArrowRight' ? 1 : list.length - 1)) % list.length]);
  }
});
document.querySelectorAll('.controls input').forEach(i => i.addEventListener('change', apply));
document.getElementById('restart').addEventListener('click', () => {
  document.querySelectorAll('.cell img[src]').forEach(img => { img.removeAttribute('src'); img.src = img.dataset.src; });
});
apply();
"""


def make_preview(task):
    """Downscaled copy of one orbit GIF, keeping every `step`-th frame at the same playback speed."""
    source, target, size, step = task
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return None
    try:
        write_preview(source, target, size, step)
    except Exception as error:  # e.g. an orbit still being written; the grid then uses the full GIF
        return f"{source}: {type(error).__name__}: {error}"
    return None


def write_preview(source, target, size, step):
    with Image.open(source) as gif:
        duration = gif.info.get("duration", 100)
        frames = []
        for index, frame in enumerate(ImageSequence.Iterator(gif)):
            if index % step == 0:
                image = frame.convert("RGB")
                image.thumbnail((size, size), Image.Resampling.LANCZOS)
                frames.append(image)
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.gif")
    frames[0].save(temporary, save_all=True, append_images=frames[1:], duration=duration * step, loop=0,
                   disposal=2, optimize=True)
    temporary.replace(target)


def safe(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)  # make_orbit.orbit_names


# Display order, as input_visualizations.input_groups: the shared inputs, then each touch run's four views.
INPUT_ORDER = ["input_mesh_textured", "input_mesh", "input_mesh_pointmap", "input_pointmap", "input_mesh_surface",
               "input_mesh_pointmap_surface", "input_surface", "input_pointmap_surface"]


def split_input(stem):
    for prefix, label in INPUT_LABELS:
        if stem.startswith(prefix):
            return prefix, label, stem[len(prefix):].lstrip("_")
    return stem, stem, ""


def input_label(stem, labels):
    _, label, rest = split_input(stem)
    return label + (f" · {labels.get(rest, rest)}" if rest else "")


def input_key(stem):
    prefix, _, rest = split_input(stem)
    return bool(rest), rest, INPUT_ORDER.index(prefix) if prefix in INPUT_ORDER else len(INPUT_ORDER)


def metric_text(row):
    parts = []
    for key, name, digits in METRICS:
        try:
            value = float(row.get(key, "nan"))
        except ValueError:
            continue
        if math.isfinite(value):
            parts.append(f"{name} {value:.{digits}f}")
    return " · ".join(parts)


def links(sample, stem, path):
    out = []
    for ext in (".gif", ".png"):
        if path.with_suffix(ext).is_file():
            href = html.escape(f"{sample}/{stem}{ext}")
            out.append(f'<a href="{href}" download="{html.escape(f"{sample}_{stem}{ext}")}">{ext[1:]}</a>')
    return "".join(out)


def cell(sample, stem, path, label, extra="", cond=None, mode=None):
    attrs = (f' data-cond="{html.escape(cond)}"' if cond else "") + (f' data-mode="{mode}"' if mode else "")
    full = f"{sample}/{stem}{path.suffix}"
    preview = path.parent / "preview" / path.name
    src = f"{sample}/preview/{path.name}" if path.suffix == ".gif" and preview.is_file() else full
    return (f'<div class="cell"{attrs}><img data-src="{html.escape(src)}" '
            f'data-full="{html.escape(full)}" alt="">'
            f'<div class="name">{html.escape(label)}</div>'
            f'<div class="meta">{html.escape(extra)}{" · " if extra else ""}{links(sample, stem, path)}</div></div>')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--orbits-dir", type=Path, help="Default: EVALUATION_DIR/orbits")
    parser.add_argument("--label", action="append", default=[], metavar="CONDITION=TEXT",
                        help="Display name for a condition (repeatable)")
    parser.add_argument("--title", help="Default: the evaluation folder name")
    parser.add_argument("--preview-size", type=int, default=256, help="Largest side of the grid's preview GIFs")
    parser.add_argument("--preview-step", type=int, default=2, help="Keep every Nth frame in the previews")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    orbits = args.orbits_dir or args.evaluation_dir / "orbits"
    labels = dict(FIXED_LABELS)
    for item in args.label:
        key, _, text = item.partition("=")
        labels[key] = text

    metrics, conditions = {}, []
    metrics_path = args.evaluation_dir / "metrics.csv"
    if metrics_path.is_file():
        with open(metrics_path, newline="") as file:
            for row in csv.DictReader(file):
                metrics[(row["condition"], row["sample_id"])] = row
                if row["condition"] not in conditions:
                    conditions.append(row["condition"])
    order = [c for c in conditions if c == "official"] + [c for c in conditions if c not in FIXED_LABELS] \
        + [c for c in conditions if c == "decoded_gt"]
    labels.update({safe(c): labels.get(c, c) for c in order})

    selected = orbits / "selected_samples.json"
    sample_ids = json.loads(selected.read_text())["sample_ids"] if selected.is_file() else []
    sample_ids += sorted(p.name for p in orbits.iterdir() if p.is_dir() and p.name not in sample_ids)
    sample_ids = [s for s in sample_ids if (orbits / s).is_dir() and any((orbits / s).glob("*.gif"))]
    tasks = [(gif, gif.parent / "preview" / gif.name, args.preview_size, args.preview_step)
             for s in sample_ids for gif in (orbits / s).glob("*.gif")]
    with ProcessPoolExecutor(args.workers) as pool:
        for done, error in enumerate(pool.map(make_preview, tasks), 1):
            if error:
                print(f"preview skipped: {error}", flush=True)
            if done % 50 == 0 or done == len(tasks):
                print(f"previews {done}/{len(tasks)}", flush=True)
    if not sample_ids:
        raise SystemExit(f"No orbit GIFs under {orbits}")

    modes = [m for m in ("mesh", "voxel") if any(orbits.glob(f"*/{m}_*.gif"))]
    shown = [c for c in order if any(orbits.glob(f"*/{m}_{safe(c)}.gif") for m in modes)]
    body = []
    for sample in sample_ids:
        folder = orbits / sample
        cells = []
        if (folder / "input_view.png").is_file():
            cells.append(cell(sample, "input_view", folder / "input_view.png", "input image"))
        if (folder / "mesh_ground_truth.gif").is_file():
            cells.append(cell(sample, "mesh_ground_truth", folder / "mesh_ground_truth.gif", "ground truth"))
        for condition in shown:
            for mode in modes:
                path = folder / f"{mode}_{safe(condition)}.gif"
                if path.is_file():
                    cells.append(cell(sample, path.stem, path, labels.get(condition, condition),
                                      metric_text(metrics.get((condition, sample), {})), condition, mode))
        inputs = sorted(folder.glob("input_*.gif"), key=lambda p: input_key(p.stem))
        input_cells = "".join(cell(sample, p.stem, p, input_label(p.stem, labels)) for p in inputs)
        dropdown = (f'<details><summary>inputs ({len(inputs)})</summary>'
                    f'<div class="grid">{input_cells}</div></details>'
                    if inputs else "")
        body.append(f'<div class="sample"><div class="sample-id">{html.escape(sample)}</div>'
                    f'<div class="grid main">{"".join(cells)}</div>{dropdown}</div>')

    controls = "".join(f'<label><input type="checkbox" data-cond="{html.escape(c)}" checked> '
                       f'{html.escape(labels.get(c, c))}</label>' for c in shown)
    if len(modes) > 1:
        controls += "".join(f'<label><input type="radio" name="mode" value="{m}"{" checked" if m == "mesh" else ""}>'
                            f' {m}</label>' for m in modes)
    controls += ('<label>size <input id="size" type="range" min="140" max="900" step="20" value="240"></label>'
                 '<button id="restart" type="button">restart GIFs</button>')
    title = args.title or args.evaluation_dir.resolve().name
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
            f'<style>{CSS}</style></head><body><div class="page"><h1>{html.escape(title)}</h1>'
            f'<div class="controls">{controls}</div>{"".join(body)}</div><div id="zoom" class="hidden"></div>'
            f'<script>{JS}</script></body></html>\n')
    (orbits / "index.html").write_text(page)
    print(f"Wrote {orbits / 'index.html'} ({len(sample_ids)} samples, {len(shown)} conditions)")


if __name__ == "__main__":
    main()
