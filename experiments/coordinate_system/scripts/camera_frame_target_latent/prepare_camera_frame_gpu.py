"""Prepare per-view camera targets and transformation figures for a quick held-view rerun.

Writes only into a fresh experiment output directory. No training or Stage 2.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p/'train.py').is_file())
sys.path.insert(0, str(REPO))
from experiments.coordinate_system.scripts.camera_frame_target_latent.camera_frame_geometry import (
    affine, digest, make_frames, select_records)
from experiments.coordinate_system.scripts.camera_frame_target_latent.camera_frame_plot import plot_transformations


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', type=Path, default=Path('configs/data_full_surface.yaml'))
    parser.add_argument('--pipeline-config', type=Path, default=Path('checkpoints/hf/pipeline.yaml'))
    parser.add_argument('--encoder-checkpoint', type=Path, default=Path('checkpoints/hf/ss_encoder.ckpt'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-objects', type=int, default=16)
    parser.add_argument('--val-objects', type=int, default=0)
    parser.add_argument('--held-views', type=int, default=4)
    parser.add_argument('--train-views', type=int, default=8, help='Training views per object; 0 uses all remaining')
    parser.add_argument('--seed', type=int, default=29)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError('Use a fresh experiment directory')
    import torch
    import yaml
    import trimesh
    from PIL import Image
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
    if not torch.cuda.is_available(): raise RuntimeError('Target encoding needs the cluster CUDA environment')
    device = torch.device('cuda', 0)
    config = yaml.safe_load(args.data_config.read_text())
    if config['touch']['source'] != 'full_surface': raise ValueError('This experiment requires full surfaces')
    root = Path(config['dataset']['root'])
    def resolve(path): return Path(path) if Path(path).is_absolute() else root/path
    manifest = resolve(config['dataset']['manifest'])
    split_file = resolve(config['dataset']['split_file'])
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    groups, ids = select_records(records, json.loads(split_file.read_text()), args.seed,
                                args.train_objects, args.val_objects, args.held_views, args.train_views)
    all_rows = {r['sample_id']: r for rows in groups.values() for r in rows}
    grouped = {}
    for row in all_rows.values(): grouped.setdefault(row['object_id'], []).append(row)
    # Figures are chosen before measuring any geometry or reconstruction score.
    figure_ids = set()
    for split in ('train', 'held_view', 'val'):
        first_ids = list(dict.fromkeys(r['object_id'] for r in groups[split]))[:4]
        for oid in first_ids:
            figure_ids.add(next(r['sample_id'] for r in groups[split] if r['object_id'] == oid))
    out = args.output_dir.resolve(); (out/'targets').mkdir(parents=True); (out/'figures').mkdir()
    report = dict(complete=False, settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        selection={split: [r['sample_id'] for r in rows] for split, rows in groups.items()},
        object_ids=ids, counts={s: len(v) for s, v in groups.items()}, figure_ids=sorted(figure_ids),
        input_sha256={str(p): sha(p) for p in (manifest, split_file, args.data_config, args.pipeline_config, args.encoder_checkpoint)},
        source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        objects=[], observations=[], figures=[],
        scope='Preparation only. Camera-oriented targets use mesh unit-box normalization. '
              'Proposed shared pointmap uses observed full-surface center/radius; no GT geometry enters that normalization. '
              'Target units remain distinct from VecSetX units. No training or Stage2.')
    write_json(out/'preparation.partial.json', report)
    target_source = REPO/'data_generation/objaverse-dexonomy/generate_target_latents.py'
    spec = importlib.util.spec_from_file_location('formal_target_builder', target_source)
    target_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(target_module)
    vae = target_module.load_encoder(args.encoder_checkpoint, device).requires_grad_(False)
    pipeline_cfg = OmegaConf.load(args.pipeline_config)
    shell = InferencePipeline.__new__(InferencePipeline)
    shell.workspace_dir = str(args.pipeline_config.resolve().parent); shell.device = device
    decoder = shell.init_ss_decoder(pipeline_cfg.ss_decoder_config_path, pipeline_cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    pre_cfg = pipeline_cfg.get('ss_preprocessor')
    preprocessor = shell.init_ss_preprocessor(instantiate(pre_cfg) if pre_cfg is not None else None,
                                             pipeline_cfg.ss_generator_config_path)
    if not preprocessor.normalize_pointmap:
        raise ValueError('Expected the existing normalized pointmap pipeline; inspect config before changing this protocol')
    report['runtime'] = dict(torch=torch.__version__, trimesh=trimesh.__version__, gpu=torch.cuda.get_device_name())
    report['source_sha256']['target_builder'] = sha(target_source)
    for key in ('ss_decoder_config_path', 'ss_decoder_ckpt_path'):
        path = Path(shell.workspace_dir)/pipeline_cfg[key]
        report['input_sha256'][str(path)] = sha(path)

    def decode(mean):
        flat = mean.permute(0, 2, 3, 4, 1).reshape(len(mean), 4096, 8)
        value = flat if decoder.reshape_input_to_cube else decoder.flat_to_cube(flat)
        prediction = decoder(value)
        if not torch.isfinite(prediction).all(): raise ValueError('Nonfinite target decoder')
        return (prediction[0, 0] > 0).cpu().numpy()

    def iou(a, b):
        union = np.count_nonzero(a | b)
        return float(np.count_nonzero(a & b)/union) if union else 1.

    with torch.inference_mode():
        for index, (oid, rows) in enumerate(grouped.items()):
            record = rows[0]
            mesh_path = root/'objects'/oid/'model.obj'
            transform_path = resolve(record['object_transform_path'])
            mesh = target_module.load_normalized_mesh(mesh_path, transform_path)
            with np.load(resolve(record['touch_path']), allow_pickle=False) as stored:
                seed_parts = stored['surface_seed_parts'].astype(np.int64)
            source_seed = int(np.random.default_rng(seed_parts).integers(2**31))
            with np.load(resolve(record['full_surface_path']), allow_pickle=False) as stored:
                count = len(stored['points_camera'])
            source_points, _ = trimesh.sample.sample_surface(mesh, count, seed=source_seed)
            grid_original = target_module.voxelize_mesh(mesh)[0].numpy().astype(bool)
            with np.load(resolve(record['target_path']), allow_pickle=False) as stored:
                old_mean = stored['mean'].astype(np.float32)
            old_support = decode(torch.from_numpy(old_mean[None]).to(device))
            report['objects'].append(dict(object_id=oid, mesh_sha256=sha(mesh_path),
                transform_sha256=sha(transform_path), source_sample_seed=source_seed,
                original_target_decode_iou=iou(old_support, grid_original)))
            for row in rows:
                sid = row['sample_id']
                if row['object_transform_path'] != record['object_transform_path']:
                    raise ValueError(f'{sid}: object transform differs across views')
                with np.load(resolve(row['camera_path']), allow_pickle=False) as stored:
                    camera_from_object = np.diag([-1., -1., 1., 1.])@stored['T_camera_from_object']
                with np.load(resolve(row['full_surface_path']), allow_pickle=False) as stored:
                    if stored['coordinate_frame'].item() != 'sam_camera': raise ValueError('Unexpected surface frame')
                    surface = stored['points_camera'].astype(np.float64)
                if surface.shape != source_points.shape: raise ValueError('Surface sample count differs')
                expected = affine(source_points, camera_from_object)
                source_error = float(np.max(np.abs(expected-surface)))
                if source_error > 2e-5:
                    raise ValueError(f'{sid}: source-sample replay error {source_error:g}; inspect source/version before interpreting alignment')
                frames = make_frames(mesh.vertices, surface, camera_from_object)
                new_mesh = mesh.copy(); new_mesh.vertices = affine(mesh.vertices, frames['target_from_object'])
                grid = target_module.voxelize_mesh(new_mesh)
                mean = vae(grid[None].to(device))['mean'].float()
                if mean.shape != (1, 8, 16, 16, 16) or not torch.isfinite(mean).all():
                    raise ValueError('Invalid encoded target')
                support = decode(mean)
                if not support.any(): raise ValueError(f'{sid}: camera target decoded empty')
                target_file = out/'targets'/f'{sid}.npz'
                np.savez_compressed(target_file, mean=mean[0].cpu().numpy())
                target_points = affine(surface, frames['target_from_camera'])
                vec_points = affine(surface, frames['vec_from_camera'])
                roundtrip_error = float(np.max(np.abs(affine(vec_points, frames['target_from_vec'])-target_points)))
                image = np.asarray(Image.open(resolve(row['image_path'])).convert('RGBA'))
                pm = np.load(resolve(row['pointmap_path']), allow_pickle=False).astype(np.float32)
                valid = (image[..., 3] > 0) & np.isfinite(pm).all(-1)
                if not valid.any(): raise ValueError(f'{sid}: empty visible pointmap')
                pm_tensor = torch.from_numpy(pm).permute(2, 0, 1)
                mask_tensor = torch.from_numpy((image[..., 3]/255.).astype(np.float32))[None]
                stock_pm, stock_scale, stock_shift = preprocessor.pointmap_normalizer.normalize(pm_tensor, mask_tensor)
                stock_points = stock_pm.permute(1, 2, 0).numpy()[valid]
                if not np.isfinite(stock_points).all(): raise ValueError('Nonfinite object pointmap after stock normalization')
                new_iou = iou(support, grid[0].numpy().astype(bool))
                observation = dict(sample_id=sid, object_id=oid, frames=frames,
                    source_replay_max_error=source_error, vec_to_target_roundtrip_max_error=roundtrip_error,
                    physical_vs_decoded_target_iou=new_iou, target_sha256=sha(target_file),
                    full_surface_sha256=sha(resolve(row['full_surface_path'])),
                    camera_sha256=sha(resolve(row['camera_path'])), pointmap_sha256=sha(resolve(row['pointmap_path'])),
                    stock_pointmap_scale=stock_scale.tolist(), stock_pointmap_shift=stock_shift.tolist(),
                    masked_pointmap_points=int(valid.sum()))
                report['observations'].append(observation)
                if sid in figure_ids:
                    path = out/'figures'/f'{sid}.png'
                    decoded_points = (np.argwhere(support)+.5)/64-.5
                    plot_transformations(path, sid, source_points, surface, pm[valid], stock_points, frames, image, decoded_points)
                    report['figures'].append(str(path.relative_to(out)))
            write_json(out/'preparation.partial.json', report)
            print(f'Prepared object {index+1}/{len(grouped)}: {oid} ({len(rows)} views)', flush=True)
    # Both target conventions use identical input records and the same selected views.
    for convention in ('object', 'camera'):
        for split, rows in groups.items():
            if not rows: continue
            filename = out/f'{convention}_{split}.jsonl'
            with filename.open('w') as stream:
                for row in rows:
                    value = copy.deepcopy(row)
                    if convention == 'camera': value['target_path'] = str(out/'targets'/f"{row['sample_id']}.npz")
                    stream.write(json.dumps(value)+'\n')
        with (out/f'{convention}.jsonl').open('w') as stream:
            for split in ('train', 'val'):
                if not groups[split]: continue
                stream.write((out/f'{convention}_{split}.jsonl').read_text())
    # A manifest per split avoids assigning the same identity to train and val in a split file.
    write_json(out/'selected_splits.json', ids)
    report['complete'] = True
    report['target_bytes'] = sum(p.stat().st_size for p in (out/'targets').glob('*.npz'))
    report['selection_sha256'] = digest(report['selection'])
    report['manifest_sha256'] = {p.name: sha(p) for p in out.glob('*.jsonl')}
    report['counts']['training_objects'] = len(ids['train']); report['counts']['validation_objects'] = len(ids['val'])
    errors = [v['source_replay_max_error'] for v in report['observations']]
    report['max_source_replay_error'] = max(errors)
    quality = [v['physical_vs_decoded_target_iou'] for v in report['observations']]
    report['target_decode_iou'] = dict(mean=float(np.mean(quality)), p10=float(np.quantile(quality,.1)), minimum=min(quality))
    write_json(out/'preparation.json', report); (out/'preparation.partial.json').unlink()
    print(f'Prepared {len(all_rows)} camera targets ({report["target_bytes"]/2**20:.1f} MiB). Figures: {out/"figures"}', flush=True)
    print('No training has run. Training uses fit_camera_frame_gpu.py with this preparation directory.', flush=True)


if __name__ == '__main__': main()
