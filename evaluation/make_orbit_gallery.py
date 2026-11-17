"""Write index.html next to make_orbit.py's orbits: one row per sample with the input image and every condition's
orbit side by side (metrics underneath), the input-point orbits in a dropdown, and download links per orbit.

    python -m evaluation.make_orbit_gallery --evaluation-dir outputs/objaverse_oa/evaluation_grasp_camframe

Add --serve 8000 to also serve the page and open http://localhost:8000. Use this rather than python -m http.server:
it answers partial-file (Range) requests, which Safari needs to play MP4s at all and every browser needs to seek.
Each orbit is shown as an MP4 with a small poster image, as the persistent-dynamic-state viewers do: the page only
downloads posters up front, and a video downloads and plays only while it is on screen. Orbits rendered without
make_orbit --mp4 are converted once from their GIF (ffmpeg); posters go in each sample's preview/ folder. Both are
remade only when the orbit changes. Rerun after rendering more orbits; it only reads files that exist.
"""
import argparse
import csv
import functools
import html
import http.server
import json
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

FIXED_LABELS = {"official": "SAM 3D (official)", "decoded_gt": "decoded GT"}
# input_visualizations.input_groups names; longer prefixes first.
INPUT_LABELS = [("input_mesh_textured", "mesh (textured)"), ("input_mesh_pointmap_surface", "mesh + pointmap + points"),
                ("input_mesh_pointmap", "mesh + pointmap"), ("input_mesh_surface", "mesh + points"),
                ("input_mesh", "mesh"), ("input_pointmap_surface", "pointmap + points"),
                ("input_pointmap", "pointmap"), ("input_surface", "points")]
# Display order, as input_visualizations.input_groups: the shared inputs, then each touch run's four views.
INPUT_ORDER = ["input_mesh_textured", "input_mesh", "input_mesh_pointmap", "input_pointmap", "input_mesh_surface",
               "input_mesh_pointmap_surface", "input_surface", "input_pointmap_surface"]
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
.cell { min-width: 0; }
.media { width: 100%; aspect-ratio: 1; object-fit: contain; display: block; border: 1px solid #eee;
         background: #fff; cursor: zoom-in; }
.name { font-size: 13px; margin-top: 4px; overflow-wrap: anywhere; }
.meta { font-size: 12px; color: #555; }
.meta a { color: #2859a5; text-decoration: none; margin-right: 8px; }
.meta a:hover { text-decoration: underline; }
details { margin-top: 10px; }
summary { cursor: pointer; color: #555; font-size: 13px; }
details .grid { margin-top: 8px; }
#zoom { position: fixed; inset: 0; background: rgba(255, 255, 255, 0.97); z-index: 10; display: flex;
        flex-direction: column; align-items: center; justify-content: center; gap: 8px; }
#zoom .big { width: min(94vw, 84vh); height: min(94vw, 84vh); object-fit: contain; border: 1px solid #eee;
             cursor: zoom-out; }
#zoom .name, #zoom .meta { font-size: 14px; text-align: center; }
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

// Videos have preload="none" and a poster, so only posters download up front. A video plays (and so downloads)
// only while it is on screen and pauses when it scrolls away. Still images load the same way.
const media = [...document.querySelectorAll('.media')];
const onScreen = new IntersectionObserver(entries => entries.forEach(e => {
  const m = e.target;
  if (m.tagName === 'VIDEO') {
    if (e.isIntersecting) m.play().catch(() => {}); else m.pause();
  } else if (e.isIntersecting && !m.getAttribute('src')) {
    m.src = m.dataset.src;
  }
}), { rootMargin: '200px 0px' });
media.forEach(m => onScreen.observe(m));
// Retry media that fail to load (dropped requests through a port forward), up to three times.
media.forEach(m => m.addEventListener('error', () => {
  const tries = Number(m.dataset.tries || 0);
  if (tries >= 3) return;
  m.dataset.tries = tries + 1;
  setTimeout(() => {
    if (m.tagName === 'VIDEO') { m.load(); m.play().catch(() => {}); } else { m.src = m.dataset.src + '?retry=' + (tries + 1); }
  }, 1000 * (tries + 1));
}, true));

// Click any orbit to see it large; arrow keys step through the visible ones, Esc or a click closes.
const zoom = document.getElementById('zoom');
let current = null;
function visible() { return media.filter(m => m.offsetParent !== null); }
function show(m) {
  current = m;
  const cell = m.closest('.cell');
  let big;
  if (m.tagName === 'VIDEO') {
    // Built as an element (not an HTML string) so muted is set before play(); the poster shows right away and
    // controls are there if the browser still refuses to start it.
    big = document.createElement('video');
    big.muted = true; big.loop = true; big.playsInline = true; big.controls = true; big.preload = 'auto';
    if (m.poster) big.poster = m.poster;
    big.src = m.dataset.full;
    const start = m.currentTime;
    big.addEventListener('loadedmetadata', () => { if (start > 0) big.currentTime = start; }, { once: true });
    big.addEventListener('error', () => setTimeout(() => { big.load(); big.play().catch(() => {}); }, 1000), { once: true });
  } else {
    big = document.createElement('img');
    big.src = m.dataset.full;
  }
  big.className = 'big';
  zoom.replaceChildren(big);
  zoom.insertAdjacentHTML('beforeend', cell.querySelector('.name').outerHTML + cell.querySelector('.meta').outerHTML
    + '<div class="meta">' + cell.closest('.sample').querySelector('.sample-id').textContent + '</div>');
  zoom.classList.remove('hidden');
  if (big.tagName === 'VIDEO') big.play().catch(() => {});
}
function close() { zoom.classList.add('hidden'); zoom.innerHTML = ''; current = null; }
media.forEach(m => m.addEventListener('click', () => show(m)));
zoom.addEventListener('click', e => { if (!e.target.closest('a') && !e.target.closest('video')) close(); });
document.addEventListener('keydown', e => {
  if (!current) return;
  if (e.key === 'Escape') close();
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    const list = visible(), i = list.indexOf(current);
    if (i >= 0) show(list[(i + (e.key === 'ArrowRight' ? 1 : list.length - 1)) % list.length]);
  }
});
document.querySelectorAll('.controls input').forEach(i => i.addEventListener('change', apply));
// Restart every playing video at frame 0 so the visible orbits line up.
document.getElementById('restart').addEventListener('click', () => {
  media.filter(m => m.tagName === 'VIDEO' && !m.paused).forEach(v => { v.currentTime = 0; });
});
apply();
"""


def ffmpeg_exe():
    try:
        import imageio_ffmpeg  # ships with moviepy, which make_orbit --mp4 uses
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def stale(target, source):
    return not target.is_file() or target.stat().st_mtime < source.stat().st_mtime


def prepare(task):
    """MP4 (from the GIF, unless make_orbit --mp4 already wrote one) and a small JPG poster for one orbit."""
    gif, poster_size, ffmpeg = task
    try:
        poster = gif.parent / "preview" / (gif.stem + ".jpg")
        if stale(poster, gif):
            write_poster(gif.with_suffix(".png") if gif.with_suffix(".png").is_file() else gif, poster, poster_size)
        mp4 = gif.with_suffix(".mp4")
        # make_orbit --mp4 writes its MP4 just before the GIF, so only an MP4 clearly older than its GIF is stale.
        if not mp4.is_file() or mp4.stat().st_size == 0 or mp4.stat().st_mtime < gif.stat().st_mtime - 60:
            if ffmpeg is None:
                raise RuntimeError("no ffmpeg (pip install imageio-ffmpeg)")
            temporary = mp4.with_name(f".{mp4.stem}.tmp.mp4")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(gif), "-movflags", "+faststart",
                            "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                            "-crf", "20", str(temporary)], check=True, capture_output=True, text=True)
            temporary.replace(mp4)
    except subprocess.CalledProcessError as error:
        lines = error.stderr.strip().splitlines()
        return f"{gif.name}: ffmpeg failed: {lines[-1][-200:] if lines else error}"
    except Exception as error:  # e.g. an orbit still being written; the page then shows the GIF itself
        return f"{gif}: {type(error).__name__}: {error}"
    return None


def write_poster(source, target, size):
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_name(f".{target.stem}.tmp.jpg")
        image.save(temporary, quality=85)
        temporary.replace(target)


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """http.server's file handler plus single byte-range requests (206 Partial Content) for video playback."""

    remaining = None

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not match or os.path.isdir(path) or match.groups() == ("", ""):
            return super().send_head()
        try:
            file = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = os.fstat(file.fileno()).st_size
        first, last = match.groups()
        if first == "":  # "bytes=-N": the last N bytes
            start, end = max(0, size - int(last)), size - 1
        else:
            start, end = int(first), min(int(last), size - 1) if last else size - 1
        if start >= size or start > end:
            file.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        file.seek(start)
        self.remaining = end - start + 1
        return file

    def copyfile(self, source, outputfile):
        remaining, self.remaining = self.remaining, None
        try:
            if remaining is None:
                return super().copyfile(source, outputfile)
            while remaining > 0:
                chunk = source.read(min(1 << 16, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):  # the browser dropped a request it no longer needs
            pass


def serve(directory, port):
    handler = functools.partial(RangeHandler, directory=str(directory))
    with http.server.ThreadingHTTPServer(("", port), handler) as server:
        print(f"Serving {directory} at http://localhost:{port} (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def safe(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)  # make_orbit.orbit_names


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


def links(sample, stem, folder):
    out = []
    for ext in (".mp4", ".gif", ".png"):
        if (folder / f"{stem}{ext}").is_file():
            href = html.escape(f"{sample}/{stem}{ext}")
            out.append(f'<a href="{href}" download="{html.escape(f"{sample}_{stem}{ext}")}">{ext[1:]}</a>')
    return "".join(out)


def cell(sample, folder, stem, label, extra="", cond=None, mode=None):
    attrs = (f' data-cond="{html.escape(cond)}"' if cond else "") + (f' data-mode="{mode}"' if mode else "")
    mp4, gif, png = (folder / f"{stem}{ext}" for ext in (".mp4", ".gif", ".png"))
    poster = folder / "preview" / f"{stem}.jpg"
    if mp4.is_file() and mp4.stat().st_size > 0:
        poster_attr = f' poster="{html.escape(f"{sample}/preview/{stem}.jpg")}"' if poster.is_file() else ""
        src = html.escape(f"{sample}/{stem}.mp4")
        tag = (f'<video class="media" src="{src}" data-full="{src}" preload="none" muted loop playsinline'
               f'{poster_attr}></video>')
    else:  # still image (input view), or an orbit whose MP4 could not be made
        full = gif if gif.is_file() else png
        src = html.escape(f"{sample}/{full.name}")
        tag = f'<img class="media" data-src="{src}" data-full="{src}" alt="">'
    return (f'<div class="cell"{attrs}>{tag}<div class="name">{html.escape(label)}</div>'
            f'<div class="meta">{html.escape(extra)}{" · " if extra else ""}{links(sample, stem, folder)}</div></div>')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--orbits-dir", type=Path, help="Default: EVALUATION_DIR/orbits")
    parser.add_argument("--label", action="append", default=[], metavar="CONDITION=TEXT",
                        help="Display name for a condition (repeatable)")
    parser.add_argument("--title", help="Default: the evaluation folder name")
    parser.add_argument("--poster-size", type=int, default=384, help="Largest side of the poster images")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--serve", type=int, metavar="PORT", help="After writing the page, serve the orbits folder")
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
    if not sample_ids:
        raise SystemExit(f"No orbit GIFs under {orbits}")
    ffmpeg = ffmpeg_exe()
    tasks = [(gif, args.poster_size, ffmpeg) for s in sample_ids for gif in sorted((orbits / s).glob("*.gif"))]
    with ProcessPoolExecutor(args.workers) as pool:
        for done, error in enumerate(pool.map(prepare, tasks), 1):
            if error:
                print(f"skipped: {error}", flush=True)
            if done % 50 == 0 or done == len(tasks):
                print(f"videos and posters {done}/{len(tasks)}", flush=True)
    for s in sample_ids:  # the input image as a poster too
        view = orbits / s / "input_view.png"
        if view.is_file() and stale(view.parent / "preview" / "input_view.jpg", view):
            write_poster(view, view.parent / "preview" / "input_view.jpg", args.poster_size)

    modes = [m for m in ("mesh", "voxel") if any(orbits.glob(f"*/{m}_*.gif"))]
    shown = [c for c in order if any(orbits.glob(f"*/{m}_{safe(c)}.gif") for m in modes)]
    body = []
    for sample in sample_ids:
        folder = orbits / sample
        cells = []
        if (folder / "input_view.png").is_file():
            view = f"{sample}/preview/input_view.jpg" if (folder / "preview" / "input_view.jpg").is_file() \
                else f"{sample}/input_view.png"
            cells.append(f'<div class="cell"><img class="media" data-src="{html.escape(view)}" '
                         f'data-full="{html.escape(f"{sample}/input_view.png")}" alt="">'
                         f'<div class="name">input image</div>'
                         f'<div class="meta">{links(sample, "input_view", folder)}</div></div>')
        if (folder / "mesh_ground_truth.gif").is_file():
            cells.append(cell(sample, folder, "mesh_ground_truth", "ground truth"))
        for condition in shown:
            for mode in modes:
                stem = f"{mode}_{safe(condition)}"
                if (folder / f"{stem}.gif").is_file():
                    cells.append(cell(sample, folder, stem, labels.get(condition, condition),
                                      metric_text(metrics.get((condition, sample), {})), condition, mode))
        inputs = sorted(folder.glob("input_*.gif"), key=lambda p: input_key(p.stem))
        input_cells = "".join(cell(sample, folder, p.stem, input_label(p.stem, labels)) for p in inputs)
        dropdown = (f'<details><summary>inputs ({len(inputs)})</summary><div class="grid">{input_cells}</div></details>'
                    if inputs else "")
        body.append(f'<div class="sample"><div class="sample-id">{html.escape(sample)}</div>'
                    f'<div class="grid main">{"".join(cells)}</div>{dropdown}</div>')

    controls = "".join(f'<label><input type="checkbox" data-cond="{html.escape(c)}" checked> '
                       f'{html.escape(labels.get(c, c))}</label>' for c in shown)
    if len(modes) > 1:
        controls += "".join(f'<label><input type="radio" name="mode" value="{m}"{" checked" if m == "mesh" else ""}>'
                            f' {m}</label>' for m in modes)
    controls += ('<label>size <input id="size" type="range" min="140" max="900" step="20" value="240"></label>'
                 '<button id="restart" type="button">sync orbits</button>')
    title = args.title or args.evaluation_dir.resolve().name
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
            f'<style>{CSS}</style></head><body><div class="page"><h1>{html.escape(title)}</h1>'
            f'<div class="controls">{controls}</div>{"".join(body)}</div><div id="zoom" class="hidden"></div>'
            f'<script>{JS}</script></body></html>\n')
    (orbits / "index.html").write_text(page)
    print(f"Wrote {orbits / 'index.html'} ({len(sample_ids)} samples, {len(shown)} conditions)")
    if args.serve:
        serve(orbits.resolve(), args.serve)


if __name__ == "__main__":
    main()
