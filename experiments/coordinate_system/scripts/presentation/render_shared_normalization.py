"""Plot both clouds after the integrated shared surface-derived normalization."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from normalization_coordinates import method
from render_conditioning import plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object_dir", type=Path)
    parser.add_argument("normalization_dir", type=Path)
    args = parser.parse_args()
    repo = next(p for p in Path(__file__).resolve().parents if (p/"train.py").exists())
    init = method(repo/"train.py", "SurfacePointmapNormalizer", "__init__", dict(torch=torch))
    normalize = method(repo/"train.py", "SurfacePointmapNormalizer", "normalize", dict(torch=torch))
    vec = method(repo/"sam3d_objects/model/backbone/dit/embedder/touch.py",
                 "TouchEncoder", "normalize_points_for_vecsetx", dict(torch=torch))
    display = json.loads((args.normalization_dir/"display.json").read_text())
    report = {}
    for view in ["004", "005", "006"]:
        source = args.object_dir/"views"/view
        surface = torch.from_numpy(np.load(source/"full_surface.npz")["points_camera"])
        pointmap = torch.from_numpy(np.load(source/"pointmap.npy")).permute(2, 0, 1)
        rgba = np.asarray(Image.open(source/"image.png").convert("RGBA"))
        valid = np.isfinite(pointmap.numpy()).all(0)&(rgba[..., 3] > 0)&(pointmap[2].numpy() > 0)
        normalizer = SimpleNamespace()
        init(normalizer, surface)
        normalized_pm, _, _ = normalize(normalizer, pointmap, torch.from_numpy(valid)[None])
        normalized_surface, _, _ = vec(None, surface[None], torch.ones((1, len(surface)), dtype=torch.bool))
        torch.testing.assert_close(normalized_surface[0],
                                   (surface-normalizer.center)/normalizer.radius)
        indices = np.flatnonzero(valid)
        indices = indices[np.linspace(0, len(indices)-1, min(5000, len(indices)), dtype=int)]
        points = normalized_pm.permute(1, 2, 0).reshape(-1, 3).numpy()[indices]
        dest = args.normalization_dir/"shared"/view
        dest.mkdir(parents=True, exist_ok=True)
        plot(dest/"surface_and_pointmap.png",
             [(normalized_surface[0].numpy(), "#337bc4", .9), (points, "#f18b32", .9)],
             np.asarray(display["center"]), display["limit"], grid_spacing=2.,
             coordinate_order=(2, 0, 1), object_axes=None, marker_size=.25)
        report[view] = dict(center=normalizer.center.tolist(), radius=float(normalizer.radius))
    (args.normalization_dir/"shared"/"normalization.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
