"""Diagnostic spatial readout from frozen VecSetX codes to Stage-1 latents.

Uses saved representation-probe artifacts. No SAM3D flow model or Stage 2.
Direct regression is a full-surface accessibility test, not a sparse completion
objective. Targets enter supervision only, never the readout input.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from representation_geometry import overlap


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class SpatialReadout(nn.Module):
    def __init__(self, input_dim, slot_identity=False, width=256):
        super().__init__()
        self.slot_identity = slot_identity
        # Seed the shared trunk separately from input projections so raw/processed
        # input width does not perturb initialization of the comparison head.
        with torch.random.fork_rng():
            torch.manual_seed(1929)
            self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        axis = (torch.arange(16, dtype=torch.float32) + .5) / 16 - .5
        xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(4096, 3)
        phases = xyz[..., None] * (2 ** torch.arange(6)) * (2 * math.pi)
        self.register_buffer('xyz_features', torch.cat([xyz, phases.sin().flatten(1), phases.cos().flatten(1)], 1))
        slots = torch.arange(1024, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.) / width))
        phase = slots * frequencies
        self.register_buffer('slot_features', torch.stack([phase.sin(), phase.cos()], -1).flatten(1))
        self.queries = nn.Linear(39, width)
        self.query_norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.q = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.kv = nn.ModuleList([nn.Linear(width, width * 2) for _ in range(2)])
        self.proj = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 4),
                                nn.GELU(), nn.Linear(width * 4, width)) for _ in range(2)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 8))

    def forward(self, features):
        context = self.input(features)
        if self.slot_identity:
            context = context + self.slot_features.to(context)
        x = self.queries(self.xyz_features).unsqueeze(0).expand(len(features), -1, -1)
        for i in range(2):
            q = self.q[i](self.query_norm[i](x)).reshape(len(x), 4096, 8, -1).transpose(1, 2)
            k, v = self.kv[i](context).chunk(2, dim=-1)
            k = k.reshape(len(x), 1024, 8, -1).transpose(1, 2)
            v = v.reshape(len(x), 1024, 8, -1).transpose(1, 2)
            attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape_as(x)
            x = x + self.proj[i](attended)
            x = x + self.ff[i](x)
        return self.out(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--representation-dir', type=Path, required=True)
    p.add_argument('--arm', choices=['raw', 'raw_slot', 'processed'], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--decoder-checkpoint', type=Path, default=Path('checkpoints/hf/ss_decoder.ckpt'))
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    if a.steps < 1 or a.learning_rate <= 0:
        p.error('Positive steps and learning rate required')
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    source_file = a.representation_dir / 'results.json'
    source = json.loads(source_file.read_text())
    assert source['complete'] and source['settings']['objects'] == 32
    assert file_digest(a.decoder_checkpoint) == source['checkpoint_sha256']['sam_decoder']
    for path, expected in source['source_sha256'].items():
        assert file_digest(REPO / path) == expected, 'Source changed: ' + path
    rows = source['records']; ids = [r['sample_id'] for r in rows]
    data_root = Path(yaml.safe_load(source['data_yaml'])['dataset']['root'])
    assert len(ids) == len(set(ids)) == len({r['object_id'] for r in rows}) == 32
    meta = {r['sample_id']: r for r in source['vecset_rows'] if r['frame'] == 'oracle'}
    raw, targets, target_grids = [], [], []
    for record in rows:
        sid = record['sample_id']
        with np.load(a.representation_dir / meta[sid]['artifact'], allow_pickle=False) as z:
            features = z['features'].copy()
        assert features.shape == (1024, 32) and digest(features) == meta[sid]['feature_sha256']
        with np.load(a.representation_dir / f'{sid}_sam.npz', allow_pickle=False) as z:
            target = z['target_latent'].copy(); support = z['target_decoded_occupancy'].copy()
        assert target.shape == (8, 16, 16, 16) and np.isfinite(target).all()
        original_path = Path(record['target_path'])
        if not original_path.is_absolute():
            original_path = data_root / original_path
        sam_meta = next(r for r in source['sam_rows'] if r['sample_id'] == sid)
        assert file_digest(original_path) == sam_meta['inputs_sha256']['target_path']
        with np.load(original_path, allow_pickle=False) as original:
            np.testing.assert_array_equal(target, original['mean'])
        raw.append(features); targets.append(target.transpose(1, 2, 3, 0).reshape(4096, 8)); target_grids.append(support)
    features = torch.from_numpy(np.stack(raw)).cuda()
    target = torch.from_numpy(np.stack(targets)).cuda()
    report = {'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'source_report_sha256': file_digest(source_file), 'driver_sha256': file_digest(Path(__file__)),
        'sample_ids': ids, 'fit_sample_ids': ids[:24], 'reserved_object_ids': ids[24:],
        'target_sha256': digest(np.stack(targets)), 'raw_features_sha256': digest(np.stack(raw)),
        'training': [], 'assessments': [], 'runtime': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()},
        'scope': '24 fitted objects / 8 separate objects from the earlier 32-object representation screen. Oracle frame, no visual input. Fixed frozen features; no geometric augmentation. Reserved results are exploratory, not an untouched dataset benchmark.'}
    if a.arm == 'processed':
        from sam3d_objects.model.backbone.dit.embedder.touch import ENCODERS
        checkpoint = Path(source['settings']['vecset_checkpoint'])
        assert file_digest(checkpoint) == source['checkpoint_sha256']['vecsetx']
        native = ENCODERS['vecsetx']['constructor']()
        weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
        native.load_state_dict(weights.get('model', weights), strict=True)
        del weights
        native = native.cuda().eval().requires_grad_(False)
        with torch.no_grad():
            features = torch.cat([native.learn(features[i:i+1]) for i in range(32)])
        del native
        gc.collect(); torch.cuda.empty_cache()
    assert torch.isfinite(features).all()
    report['readout_features_sha256'] = digest(features.cpu().numpy())
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    model = SpatialReadout(features.shape[-1], a.arm == 'raw_slot').cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=0)
    report['trainable_parameters'] = sum(p.numel() for p in model.parameters())
    common = {n: v.detach().cpu().numpy() for n, v in model.named_parameters() if not n.startswith('input.')}
    report['initial_common_parameters_sha256'] = hashlib.sha256(b''.join(n.encode() + v.tobytes() for n, v in common.items())).hexdigest()
    start_step = 0
    if a.resume:
        state = torch.load(a.resume, map_location='cpu', weights_only=False)
        assert state['arm'] == a.arm and state['source_report_sha256'] == report['source_report_sha256']
        assert state['seed'] == a.seed and state['learning_rate'] == a.learning_rate
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start_step = state['step']
        if a.steps <= start_step:
            raise ValueError('steps must exceed resumed step')
    report['start_step'] = start_step
    report['numeric_precision'] = 'fp32 parameters; bf16 autocast for readout, fp32 VAE decoding'
    fit_mean = target[:24].mean(0)
    report['fit_mean_target_control_mse'] = (target - fit_mean).square().mean((1, 2)).cpu().tolist()
    a.output_dir.mkdir(parents=True)
    from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import SparseStructureDecoderTdfyWrapper
    decoder = SparseStructureDecoderTdfyWrapper(out_channels=1, latent_channels=8,
        channels=[512, 128, 32], num_res_blocks=2, num_res_blocks_middle=2,
        reshape_input_to_cube=False, pretrained_ckpt_path=str(a.decoder_checkpoint)).cuda().eval().requires_grad_(False)
    def save():
        (a.output_dir / 'results.partial.json').write_text(json.dumps(report, indent=2) + '\n')
    def assess(step, decode=False):
        model.eval()
        result = {'step': step, 'objects': []}
        with torch.no_grad():
            for i, sid in enumerate(ids):
                group_start = 0 if i < 24 else 24; group_size = 24 if i < 24 else 8
                donor = group_start + ((i - group_start + 1) % group_size)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(features[i:i+1]).float()
                    swapped = model(features[donor:donor+1]).float()
                if not torch.isfinite(pred).all() or not torch.isfinite(swapped).all():
                    raise FloatingPointError('Nonfinite readout')
                row = {'sample_id': sid, 'split': 'fit' if i < 24 else 'reserved_object',
                       'latent_mse': float((pred - target[i:i+1]).square().mean()),
                       'wrong_object_latent_mse': float((swapped - target[i:i+1]).square().mean())}
                if decode:
                    logits = decoder(pred.transpose(1, 2).reshape(1, 8, 16, 16, 16))
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError('Nonfinite decode')
                    grid = (logits[0, 0] > 0).cpu().numpy()
                    row['decoded_vs_target'] = overlap(grid, target_grids[i].astype(bool))
                    np.savez_compressed(a.output_dir / f'step{step}_{sid}.npz', prediction=pred[0].cpu().numpy(),
                                        target=target[i].cpu().numpy(), predicted_occupancy=grid, target_occupancy=target_grids[i])
                result['objects'].append(row)
        report['assessments'].append(result); save()
        for split in ['fit', 'reserved_object']:
            errors = [r['latent_mse'] for r in result['objects'] if r['split'] == split]
            print(a.arm, step, split, sum(errors) / len(errors), flush=True)
        model.train()
    assess(start_step)
    for step in range(start_step + 1, a.steps + 1):
        started = time.monotonic()
        # Six balanced batches of four per epoch, deterministic across arms and resumes.
        epoch, batch = divmod(step - 1, 6)
        order = list(range(24)); random.Random(a.seed + epoch).shuffle(order)
        indices = order[batch * 4:(batch + 1) * 4]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(features[indices])
            loss = F.mse_loss(pred.float(), target[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == start_step + 1 or step % 20 == 0:
            report['training'].append({'step': step, 'loss': float(loss), 'gradient_norm': float(norm),
                                       'seconds_this_step': time.monotonic() - started})
        if step % 500 == 0 or step == a.steps:
            assess(step, decode=step == a.steps)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                        'arm': a.arm, 'seed': a.seed, 'learning_rate': a.learning_rate,
                        'source_report_sha256': report['source_report_sha256']}, a.output_dir / f'checkpoint_{step}.pt')
    report['complete'] = True
    (a.output_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
