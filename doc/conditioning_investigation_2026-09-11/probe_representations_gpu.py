"""No-training surface representation probe. No DiT, RGB model or Stage 2.

Point inputs are built before loading source meshes/targets for scoring. Both
SAM VAE and native VecSetX are frozen. No target-assisted input repair.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from dataloader import TouchDataset, load_data_config
from representation_geometry import point_grid, overlap, sample_mesh, surface_agreement, zero_surface

DEFAULT_VECSET = Path('/n/home12/mwakeham/.cache/huggingface/hub/models--Zbalpha--VecSetX/snapshots/5fb84917189d2bee8392404f833f42ca5c067e0b/learnable_vec1024x32_dim1024_depth24_sdf_nb/checkpoint-125.pth')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', type=Path, default=Path('configs/data_full_surface.yaml'))
    parser.add_argument('--encoder-checkpoint', type=Path, default=Path('checkpoints/hf/ss_encoder.ckpt'))
    parser.add_argument('--decoder-checkpoint', type=Path, default=Path('checkpoints/hf/ss_decoder.ckpt'))
    parser.add_argument('--vecset-checkpoint', type=Path, default=DEFAULT_VECSET)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--objects', type=int, default=32)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--resolution', type=int, default=64, help='VecSetX field cells per axis; SAM remains 64')
    parser.add_argument('--query-chunk', type=int, default=4096)
    args = parser.parse_args()
    if args.objects < 2 or args.resolution < 32 or args.query_chunk < 1:
        parser.error('Require >=2 objects, resolution >=32, positive chunk size')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for p in [args.encoder_checkpoint, args.decoder_checkpoint, args.vecset_checkpoint]:
        if not p.is_file():
            raise FileNotFoundError(p)
    from skimage.measure import marching_cubes  # preflight dependency, before allocation
    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder, ENCODERS
    from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import SparseStructureDecoderTdfyWrapper
    source_path = REPO / 'data_generation/objaverse-dexonomy/generate_target_latents.py'
    source = module_from(source_path, 'representation_target_source')
    config = load_data_config(args.data_config); config['dataset']['split'] = 'train'
    ds = TouchDataset(config, include_touch=True, oracle_point_frame=True)
    grouped = {}
    for record in ds.records:
        grouped.setdefault(record['object_id'], []).append(record)
    rng = random.Random(args.seed)
    object_ids = sorted(grouped); rng.shuffle(object_ids)
    object_ids = object_ids[:args.objects]
    if len(object_ids) != args.objects:
        raise ValueError('Not enough training objects')
    records = []
    for oid in object_ids:
        options = sorted(grouped[oid], key=lambda r: r['sample_id'])
        random.Random(args.seed).shuffle(options)
        records.append(options[0])
    for record in records:
        paths = [ds.resolve_path(record[k]) for k in ['camera_path', 'full_surface_path', 'target_path', 'object_transform_path']]
        paths += [ds.root / 'objects' / record['object_id'] / 'model.obj']
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device('cuda')
    # Keep numerical behavior close to original fp32 target encoding. Decoder
    # precision is not changed between reference and surface-derived latents.
    encoder = source.load_encoder(args.encoder_checkpoint, device).requires_grad_(False)
    decoder = SparseStructureDecoderTdfyWrapper(out_channels=1, latent_channels=8,
        channels=[512, 128, 32], num_res_blocks=2, num_res_blocks_middle=2,
        reshape_input_to_cube=False, pretrained_ckpt_path=str(args.decoder_checkpoint)).to(device).eval().requires_grad_(False)
    report = {'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'data_yaml': args.data_config.read_text(), 'records': records,
        'source_sha256': {str(p.relative_to(REPO)): sha(p) for p in [Path(__file__), HERE / 'representation_geometry.py',
            source_path, REPO / 'dataloader.py', REPO / 'sam3d_objects/model/backbone/dit/embedder/touch.py',
            REPO / 'sam3d_objects/model/backbone/dit/embedder/vecsetx/autoencoder.py',
            REPO / 'sam3d_objects/model/backbone/dit/embedder/vecsetx/utils.py',
            REPO / 'sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_vae.py']},
        'checkpoint_sha256': {'sam_encoder': sha(args.encoder_checkpoint), 'sam_decoder': sha(args.decoder_checkpoint),
                             'vecsetx': sha(args.vecset_checkpoint)},
        'runtime': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'precision': 'fp32 weights and outer computation; VecSetX attention internally casts Q/K/V to bf16',
                    'cuda': torch.version.cuda, 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                    'tf32_cudnn': torch.backends.cudnn.allow_tf32},
        'sam_rows': [], 'vecset_rows': [],
        'scope': 'Frozen representation reconstruction of recorded surfaces. Oracle inversion is privileged. Camera VecSetX skips isotropic SSI because it cancels under bbox/radius normalization; no image/pointmap conditioning. No training or Stage 2.'}
    args.output_dir.mkdir(parents=True)
    def save():
        (args.output_dir / 'results.partial.json').write_text(json.dumps(report, indent=2) + '\n')
    inputs = {}
    surface_latents, target_latents = [], []
    with torch.inference_mode():
        for record in records:
            started = time.monotonic(); sid = record['sample_id']; oid = record['object_id']
            pc = ds.load_full_surface(ds.resolve_path(record['full_surface_path'])).astype(np.float64)
            with np.load(ds.resolve_path(record['camera_path']), allow_pickle=False) as data:
                transform = np.diag([-1., -1., 1., 1.]) @ data['T_camera_from_object'].astype(np.float64)
            assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            oracle = (pc - transform[:3, 3]) @ np.linalg.inv(transform[:3, :3]).T
            grid = point_grid(oracle)  # depends only on observed points and camera
            surface_latent = encoder(torch.from_numpy(grid.astype(np.float32))[None, None].to(device))['mean']
            surface_logits = decoder(surface_latent)
            # Targets and original mesh are read only after constructing the
            # proposed conditioning representation, for scoring/calibration.
            with np.load(ds.resolve_path(record['target_path']), allow_pickle=False) as data:
                target_np = data['mean'].copy()
            target = torch.from_numpy(target_np)[None].to(device)
            target_logits = decoder(target)
            mesh_path = ds.root / 'objects' / oid / 'model.obj'
            mesh = source.load_normalized_mesh(mesh_path, ds.resolve_path(record['object_transform_path']))
            mesh_grid_tensor = source.voxelize_mesh(mesh)
            regenerated = encoder(mesh_grid_tensor[None].to(device))['mean']
            torch.testing.assert_close(regenerated, target, rtol=1e-4, atol=1e-5,
                msg='Original target regeneration failed: ' + oid)
            for tensor in [surface_latent, surface_logits, target_logits, regenerated]:
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError('Nonfinite SAM VAE result')
            pred_grid = (surface_logits[0, 0] > 0).cpu().numpy()
            target_grid = (target_logits[0, 0] > 0).cpu().numpy()
            mesh_grid = mesh_grid_tensor[0].numpy().astype(bool)
            reference = sample_mesh(mesh.vertices, mesh.faces, seed=args.seed)
            calibration = sample_mesh(mesh.vertices, mesh.faces, seed=args.seed + 1)
            np.savez_compressed(args.output_dir / f'{sid}_sam.npz', input_point_grid=grid,
                surface_latent=surface_latent[0].cpu().numpy(), target_latent=target_np,
                surface_decoded_occupancy=pred_grid, target_decoded_occupancy=target_grid, mesh_occupancy=mesh_grid)
            inputs[sid] = (pc, oracle, transform, reference, target_grid)
            surface_latents.append(surface_latent[0].cpu().numpy().copy())
            target_latents.append(target_np)
            report['sam_rows'].append({'sample_id': sid, 'point_count': len(pc),
                'inputs_sha256': {key: sha(ds.resolve_path(record[key])) for key in ['camera_path', 'full_surface_path', 'target_path', 'object_transform_path']},
                'mesh_sha256': sha(mesh_path), 'point_grid_sha256': digest(grid),
                'target_regeneration_max_error': float((regenerated - target).abs().max()),
                'surface_latent_mse': float((surface_latent - target).square().mean()),
                'target_latent_mean_square': float(target.square().mean()),
                'point_grid_vs_target': overlap(grid, target_grid),
                'decoded_surface_vs_target': overlap(pred_grid, target_grid),
                'decoded_surface_vs_mesh': overlap(pred_grid, mesh_grid),
                'target_decode_vs_mesh': overlap(target_grid, mesh_grid),
                'input_points_vs_reference_surface': surface_agreement(oracle, reference),
                'mesh_sampling_calibration': surface_agreement(calibration, reference),
                'seconds': time.monotonic() - started})
            save(); print('SAM representation', sid, report['sam_rows'][-1]['surface_latent_mse'], flush=True)
    # Scoring-only controls: is low latent error object-specific, or would a
    # generic target-code average look similarly good? Never used as input.
    bank = np.stack(target_latents).astype(np.float64)
    pairwise = [[float(np.mean((z.astype(np.float64) - target_code) ** 2))
                 for target_code in bank] for z in surface_latents]
    report['sam_latent_controls'] = {
        'sample_ids': [r['sample_id'] for r in records],
        'surface_vs_each_target_mse': pairwise,
        'leave_one_object_out_target_mean_mse': [float(np.mean((
            (bank.sum(0) - bank[i]) / (len(bank) - 1) - bank[i]) ** 2)) for i in range(len(bank))],
        'scope': 'Scoring controls, not additional predictors or independent test objects.'}
    save()
    del encoder, decoder, surface_latent, surface_logits, target_logits, regenerated, target
    gc.collect(); torch.cuda.empty_cache()

    class NativeInputAdapter(TouchEncoder):
        def __init__(self):
            # Reuse production prepare_points/normalization, excluding the
            # trainable projector because this probes native reconstruction.
            torch.nn.Module.__init__(self)
            self.encoder = ENCODERS['vecsetx']['constructor']()
            self.num_points = self.encoder.num_inputs
            checkpoint = torch.load(args.vecset_checkpoint, map_location='cpu', weights_only=False)
            self.encoder.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
    adapter = NativeInputAdapter().to(device).eval().requires_grad_(False)
    lower, upper = -1.05, 1.05
    axis = np.linspace(lower, upper, args.resolution + 1, dtype=np.float32)
    queries_np = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    queries = torch.from_numpy(queries_np).to(device)[None]
    report['vecset_field_grid'] = {'resolution_cells': args.resolution, 'lower': lower, 'upper': upper,
                                  'axis_order': 'ij xyz', 'queries_sha256': digest(queries_np)}
    with torch.inference_mode():
        for record in records:
            sid = record['sample_id']; pc, oracle, transform, reference, target_grid = inputs[sid]
            for frame, points in [('oracle', oracle), ('camera', pc)]:
                started = time.monotonic()
                p = torch.from_numpy(points.astype(np.float32))[None].to(device)
                pp, mask, shifts, scales = adapter.prepare_points(p)
                restored = pp / scales[:, None] + shifts[:, None]
                # Existing data have exactly num_inputs points, so no FPS/reorder.
                if len(points) == adapter.num_points:
                    torch.testing.assert_close(restored, p, rtol=1e-5, atol=2e-6)
                encoded = adapter.encoder.encode(pp, mask)['x']
                learned = adapter.encoder.learn(encoded)
                # Validate native encode->learn->decode composition against forward.
                # Native forward omits a mask; production encode receives one.
                # Supply that same mask so the check uses the same attention kernel.
                encode_fn = adapter.encoder.encode
                with patch.object(adapter.encoder, 'encode', side_effect=lambda x: encode_fn(x, mask)):
                    direct = adapter.encoder(pp, queries[:, :64])['o']
                cached = adapter.encoder.decode(learned, queries[:, :64]).squeeze(-1)
                torch.testing.assert_close(cached, direct, rtol=1e-4, atol=1e-5)
                if not torch.isfinite(encoded).all():
                    raise FloatingPointError('Nonfinite VecSetX features')
                outputs = []
                for start in range(0, queries.shape[1], args.query_chunk):
                    outputs.append(adapter.encoder.decode(learned, queries[:, start:start + args.query_chunk]).float().cpu().numpy().reshape(-1))
                field = np.concatenate(outputs).reshape((args.resolution + 1,) * 3)
                surface = zero_surface(field, lower, upper)
                shift = shifts[0].cpu().numpy(); scale = float(scales[0, 0])
                row = {'sample_id': sid, 'frame': frame, 'shift': shift.tolist(), 'scale': scale,
                    'point_count_after_preparation': pp.shape[1], 'feature_sha256': digest(encoded.cpu().numpy()),
                    'field_min': float(field.min()), 'field_max': float(field.max()),
                    'native_forward_max_error': float((cached - direct).abs().max())}
                saved = {'field': field, 'shift': shift, 'scale': np.array(scale), 'features': encoded[0].cpu().numpy()}
                if surface is None:
                    row['status'] = 'no_zero_crossing'
                    row['surface_agreement'] = surface_agreement(np.empty((0, 3)), reference)
                else:
                    vertices, faces = surface
                    # Transform decoded native coordinates to the object frame
                    # using input-only scale/shift and the supplied camera pose.
                    vertices = vertices / scale + shift
                    if frame == 'camera':
                        vertices = (vertices - transform[:3, 3]) @ np.linalg.inv(transform[:3, :3]).T
                    sample = sample_mesh(vertices, faces, seed=args.seed + 2)
                    row.update(status='ok', vertices=len(vertices), faces=len(faces),
                        surface_agreement=surface_agreement(sample, reference),
                        object_bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()])
                    # Boundary clipping can invalidate a low apparent coverage.
                    native_vertices = surface[0]
                    margin = (upper - lower) / args.resolution
                    row['near_query_boundary_vertex_fraction'] = float(np.mean(np.any((native_vertices <= lower + margin) | (native_vertices >= upper - margin), axis=1)))
                    saved.update(vertices_object=vertices, faces=faces)
                row['seconds'] = time.monotonic() - started
                row['artifact'] = f'{sid}_vecset_{frame}.npz'
                np.savez_compressed(args.output_dir / row['artifact'], **saved)
                report['vecset_rows'].append(row)
                save(); print('VecSetX', sid, frame, row['status'], row['surface_agreement'], flush=True)
    report['complete'] = True
    (args.output_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
