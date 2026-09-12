"""Frozen VecSetX geometry -> target voxelizer -> frozen Stage-1 VAE.

Uses saved native oracle meshes, without learning a token-to-latent translation.
No DiT, images, Stage 2, mesh fitting, or target-based input normalization.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import trimesh
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from representation_geometry import overlap, zero_surface


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--representation-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--encoder-checkpoint', type=Path, default=Path('checkpoints/hf/ss_encoder.ckpt'))
    p.add_argument('--decoder-checkpoint', type=Path, default=Path('checkpoints/hf/ss_decoder.ckpt'))
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    src = json.loads((a.representation_dir / 'results.json').read_text())
    assert src['complete'] and len(src['records']) == 32
    for key, path in [('sam_encoder', a.encoder_checkpoint), ('sam_decoder', a.decoder_checkpoint)]:
        assert sha(path) == src['checkpoint_sha256'][key], key
    source_path = REPO / 'data_generation/objaverse-dexonomy/generate_target_latents.py'
    for path in [source_path, HERE / 'representation_geometry.py',
                 REPO / 'sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_vae.py']:
        assert sha(path) == src['source_sha256'][str(path.relative_to(REPO))], path
    spec = importlib.util.spec_from_file_location('spatial_target_source', source_path)
    source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
    ids = [r['sample_id'] for r in src['records']]
    assert len(set(ids)) == len({r['object_id'] for r in src['records']}) == 32
    metadata = {r['sample_id']: r for r in src['vecset_rows'] if r['frame'] == 'oracle'}
    sam_meta = {r['sample_id']: r for r in src['sam_rows']}
    inputs, input_checks, artifact_hashes = [], [], {}
    # Construct every candidate input before reading target arrays for scoring.
    # Saved vertices are already in object coordinates. No additional B rotation,
    # recentering, rescaling, filling, dilation, or target-based repair is applied.
    for sid in ids:
        meta = metadata[sid]
        assert meta['status'] == 'ok'
        artifact = a.representation_dir / meta['artifact']
        artifact_hashes[artifact.name] = sha(artifact)
        with np.load(artifact, allow_pickle=False) as z:
            vertices = z['vertices_object'].copy(); faces = z['faces'].copy()
            field = z['field'].copy(); shift = z['shift'].copy(); scale = float(z['scale'])
            assert digest(z['features']) == meta['feature_sha256']
        assert np.isfinite(vertices).all() and scale > 0 and np.isfinite(scale)
        np.testing.assert_allclose(shift, meta['shift'], rtol=0, atol=0)
        assert scale == meta['scale'] and len(vertices) == meta['vertices'] and len(faces) == meta['faces']
        np.testing.assert_allclose([vertices.min(0), vertices.max(0)], meta['object_bounds'], atol=1e-7, rtol=0)
        geometry = zero_surface(field, src['vecset_field_grid']['lower'], src['vecset_field_grid']['upper'])
        assert geometry is not None
        restored = geometry[0] / scale + shift
        np.testing.assert_allclose(restored, vertices, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(geometry[1], faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        grid = source.voxelize_mesh(mesh)[0].numpy().astype(bool)
        inputs.append(grid)
        input_checks.append({'sample_id': sid, 'native_surface_sha256': digest(vertices),
            'grid_sha256': digest(grid), 'field_to_saved_vertices_max_error': float(np.max(np.abs(restored - vertices))),
            'outside_cube_vertex_fraction': float(np.mean(np.any(np.abs(vertices) > .5, axis=1))),
            'max_outside_cube_distance': float(max(0, np.max(np.abs(vertices)) - .5)),
            'clipping_policy': 'Original target voxelizer clips vertices to +/- (0.5 - 1e-6); no target-dependent adjustment.'})

    targets, point_grids, mesh_grids, decoded_targets, previous_point_latents = [], [], [], [], []
    data_root = Path(yaml.safe_load(src['data_yaml'])['dataset']['root'])
    for record in src['records']:
        sid = record['sample_id']
        artifact = a.representation_dir / f'{sid}_sam.npz'
        artifact_hashes[artifact.name] = sha(artifact)
        with np.load(artifact, allow_pickle=False) as z:
            targets.append(z['target_latent'].copy())
            point_grids.append(z['input_point_grid'].copy().astype(bool))
            mesh_grids.append(z['mesh_occupancy'].copy().astype(bool))
            decoded_targets.append(z['target_decoded_occupancy'].copy().astype(bool))
            previous_point_latents.append(z['surface_latent'].copy())
        assert digest(point_grids[-1]) == sam_meta[sid]['point_grid_sha256']
        original_path = Path(record['target_path'])
        if not original_path.is_absolute():
            original_path = data_root / original_path
        assert sha(original_path) == sam_meta[sid]['inputs_sha256']['target_path']
        with np.load(original_path, allow_pickle=False) as original:
            np.testing.assert_array_equal(targets[-1], original['mean'])
    bank = np.stack(targets)
    assert bank.shape == (32, 8, 16, 16, 16) and np.isfinite(bank).all()
    fit_mean = bank[:24].mean(0)
    controls = ((bank - fit_mean) ** 2).mean((1, 2, 3, 4))
    device = torch.device('cuda')
    encoder = source.load_encoder(a.encoder_checkpoint, device).requires_grad_(False)
    from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import SparseStructureDecoderTdfyWrapper
    decoder = SparseStructureDecoderTdfyWrapper(out_channels=1, latent_channels=8,
        channels=[512, 128, 32], num_res_blocks=2, num_res_blocks_middle=2,
        reshape_input_to_cube=False, pretrained_ckpt_path=str(a.decoder_checkpoint)).to(device).eval().requires_grad_(False)
    report = {'settings': {k: str(v) for k, v in vars(a).items()}, 'sample_ids': ids,
        'source_report_sha256': sha(a.representation_dir / 'results.json'),
        'driver_sha256': sha(Path(__file__)), 'artifact_sha256': artifact_hashes,
        'checkpoint_sha256': src['checkpoint_sha256'], 'input_checks': input_checks,
        'target_sha256_channels_first': digest(bank), 'fit_mean_target_control_mse': controls.tolist(),
        'runtime': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                    'precision': 'fp32 weights and computation',
                    'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                    'tf32_cudnn': torch.backends.cudnn.allow_tf32},
        'rows': [], 'scope': 'Frozen spatial bridge; all 32 objects have no adaptation here. '
            'The 24/8 labels only match prior readout scoring. Oracle frame is privileged. '
            'Saved native mesh field resolution may limit accuracy. Source mesh grids and targets '
            'enter scoring controls only. No Stage 2 and no sparse-touch feasibility claim.'}
    a.output_dir.mkdir(parents=True)

    def encode(grid):
        z = encoder(torch.from_numpy(grid.astype(np.float32))[None, None].to(device))['mean']
        assert torch.isfinite(z).all()
        return z

    def decode(z):
        logits = decoder(z)
        assert torch.isfinite(logits).all()
        return (logits[0, 0] > 0).cpu().numpy()

    with torch.inference_mode():
        for i, sid in enumerate(ids):
            target = torch.from_numpy(bank[i:i+1]).to(device)
            regenerated = encode(mesh_grids[i])
            torch.testing.assert_close(regenerated, target, rtol=1e-4, atol=1e-5)
            target_grid = decode(target)
            np.testing.assert_array_equal(target_grid, decoded_targets[i])
            encoded = {name: encode(grid) for name, grid in [('native_mesh', inputs[i]), ('point_hits', point_grids[i])]}
            torch.testing.assert_close(encoded['point_hits'], torch.from_numpy(previous_point_latents[i])[None].to(device), rtol=1e-4, atol=1e-5)
            for name, grid in [('native_mesh', inputs[i]), ('point_hits', point_grids[i])]:
                z = encoded[name]
                pred = decode(z)
                row = {'sample_id': sid, 'split': 'fit' if i < 24 else 'reserved_object', 'condition': name,
                    'latent_mse': float((z - target).square().mean()),
                    'input_vs_target': overlap(grid, target_grid), 'decoded_vs_target': overlap(pred, target_grid),
                    'target_regeneration_max_error': float((regenerated - target).abs().max())}
                report['rows'].append(row)
                np.savez_compressed(a.output_dir / f'{sid}_{name}.npz', latent=z[0].cpu().numpy(), input_grid=grid)
            # Prespecified sensitivity control, not an orientation/translation search:
            # translate the actual target shell one voxel along +x, cropping, never wrapping.
            shifted = np.zeros_like(mesh_grids[i]); shifted[1:] = mesh_grids[i][:-1]
            z_shift = encode(shifted)
            report['rows'].append({'sample_id': sid, 'split': 'fit' if i < 24 else 'reserved_object',
                'condition': 'target_shell_plus_one_x_voxel_control',
                'latent_mse': float((z_shift - target).square().mean()),
                'input_vs_target': overlap(shifted, target_grid),
                'cropped_occupied_voxels': int(mesh_grids[i][-1].sum()),
                'scope': 'Target-derived sensitivity control, never a conditioner or candidate correction.'})
            (a.output_dir / 'results.partial.json').write_text(json.dumps(report, indent=2) + '\n')
            print(sid, {r['condition']: r['latent_mse'] for r in report['rows'][-3:]}, flush=True)
    report['summary'] = {}
    for condition in ['native_mesh', 'point_hits', 'target_shell_plus_one_x_voxel_control']:
        report['summary'][condition] = {}
        for split in ['fit', 'reserved_object']:
            rows = [r for r in report['rows'] if r['condition'] == condition and r['split'] == split]
            summary = {'latent_mse': mean(r['latent_mse'] for r in rows),
                       'input_iou': mean(r['input_vs_target']['iou'] for r in rows)}
            if condition != 'target_shell_plus_one_x_voxel_control':
                summary['decoded_iou'] = mean(r['decoded_vs_target']['iou'] for r in rows)
            report['summary'][condition][split] = summary
    report['complete'] = True
    (a.output_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
