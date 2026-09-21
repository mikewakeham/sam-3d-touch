"""Run with Blender: export the same baked, normalized mesh with its materials."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_generation.general.render_blender import clear_scene, load_object, freeze_geometry, normalize_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    metadata = json.loads((args.object_dir / 'render_metadata.json').read_text())
    source = Path(metadata['source'])
    if not source.is_file():
        raise FileNotFoundError(f'Original textured asset is missing: {source}')
    for asset in metadata.get('external_assets', []):
        path = Path(asset['path'])
        if not path.is_file() or path.stat().st_size != asset['size'] or path.stat().st_mtime_ns != asset['mtime_ns']:
            raise ValueError(f'Texture changed since dataset generation: {path}')
    clear_scene()
    load_object(source)
    objects = freeze_geometry(metadata['frame'], metadata['rotation_degrees'])
    # Never write to the dataset: normalize_scene writes its geometry into a temporary directory.
    with tempfile.TemporaryDirectory() as temporary:
        normalize_scene(objects, Path(temporary))
        with np.load(Path(temporary) / 'mesh.npz', allow_pickle=False) as actual, np.load(
            args.object_dir / 'mesh.npz', allow_pickle=False
        ) as expected:
            if (actual['vertices'].shape != expected['vertices'].shape
                    or not np.allclose(actual['vertices'], expected['vertices'], atol=1e-6, rtol=0)
                    or not np.array_equal(actual['faces'], expected['faces'])):
                raise ValueError('Textured geometry differs from saved mesh.npz; check the source asset and Blender version')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + '.tmp.glb')
    # Keep Blender's axes: the renderer applies the saved object-to-evaluation transform.
    bpy.ops.export_scene.gltf(filepath=str(temporary), export_format='GLB', export_yup=False)
    temporary.replace(args.output)
    print(f'Saved verified textured mesh: {args.output}', flush=True)


if __name__ == '__main__':
    main()
