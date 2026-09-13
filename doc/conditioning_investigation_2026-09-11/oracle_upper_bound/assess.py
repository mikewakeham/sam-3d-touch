"""Paired native loss and sampled Stage-1 supports at fixed epoch endpoints."""
import copy
import json
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from dataloader import collate_touch_batch
from experiments.noisy_target.evaluate import tensor_digest
from train import amp, prepare_batch
from rollout_gpu import occupancy


def support_metrics(pred, target):
    intersection = np.count_nonzero(pred & target)
    union = np.count_nonzero(pred | target)
    p, q = np.argwhere(pred), np.argwhere(target)
    if not len(q):
        raise ValueError('Empty decoded target is not an admissible fidelity reference')
    precision = float(np.mean(cKDTree(q).query(p)[0] <= 2.00000001)) if len(p) else 0.
    recall = float(np.mean(cKDTree(p).query(q)[0] <= 2.00000001)) if len(p) else 0.
    return dict(iou=float(intersection / union), precision_2v=precision, recall_2v=recall,
                fscore_2v=2 * precision * recall / (precision + recall) if precision + recall else 0.,
                predicted_count=len(p), target_count=len(q))


def assess(output, pipeline, model, datasets, selections, config, device, seed, epoch, step, pipeline_path):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    cfg = OmegaConf.load(pipeline_path)
    decoder = pipeline.init_ss_decoder(cfg.ss_decoder_config_path, cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    gen = model.generator
    report = dict(epoch=epoch, step=step, conditioning=config, inputs=[], native=[], samples=[], complete=False,
                  scope='Two selected views each of up to 32 training and 32 identity-disjoint validation objects. '
                  'All visuals present. Fixed-target Stage-1 reference; raw frame metrics, saved supports allow '
                  'subsequent proper rigid shape analysis. Two draws, one trained checkpoint, no Stage 2.')
    def save():
        (output / 'results.partial.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device.index]), torch.no_grad():
            for split, groups in selections.items():
                ds = copy.copy(datasets[split])
                for gi, records in enumerate(groups):
                    ds.records = records
                    batch = collate_touch_batch([ds[i] for i in range(len(records))])
                    targets, ca, kw, points, mask = prepare_batch(pipeline, batch, device, 'bf16', config['touch'], False, config['oracle'])
                    assert len(ca) == 1 and not kw
                    with amp(device, 'bf16'):
                        tokens = model.surface_tokens(points, mask)
                        target_support = np.stack([occupancy(decoder, t[None]) for t in targets['shape']])
                    ids = [r['object_id'] for r in records]
                    assert len(set(ids)) == 4
                    report['inputs'].append(dict(split=split, group=gi, sample_ids=batch['sample_id'],
                        object_ids=ids, image_sha256=tensor_digest(batch['image']),
                        pointmap_sha256=tensor_digest(batch['pointmap']), visual_sha256=tensor_digest(ca[0]),
                        targets_sha256=tensor_digest(targets['shape']),
                        tokens_sha256=tensor_digest(tokens) if tokens is not None else None))
                    conditions = ['correct', 'wrong_1', 'wrong_2', 'wrong_3'] if tokens is not None and not config['constant'] else ['correct']
                    for condition in conditions:
                        conditioned = tokens.roll(int(condition[-1]), 0) if condition.startswith('wrong_') else tokens
                        kwargs = {} if conditioned is None else {'touch_tokens': conditioned}
                        expected_context = ca[0] if conditioned is None else torch.cat((ca[0], conditioned.to(ca[0])), 1)
                        seen = []
                        def check_context(module, inputs):
                            torch.testing.assert_close(inputs[0], expected_context.to(inputs[0]), rtol=0, atol=0)
                            seen.append(True)
                        hook = pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check_context)
                        try:
                            for draw in range(2):
                                draw_seed = 1500000 + seed + (0 if split == 'train' else 10000) + gi * 10 + draw
                                torch.manual_seed(draw_seed); random.seed(draw_seed)
                                gen.reverse_fn.training = True
                                original_loss = gen.loss_fn
                                functions = dict(original_loss) if isinstance(original_loss, dict) else {k: original_loss for k in gen.loss_weights}
                                shape_loss = functions['shape']; per_object = []
                                def capture(pred, target):
                                    native = shape_loss(pred, target)
                                    values = F.mse_loss(pred.float(), target.float(), reduction='none').flatten(1).mean(1)
                                    torch.testing.assert_close(values.mean(), native.float(), rtol=1e-5, atol=1e-7)
                                    per_object.append(values.cpu().tolist())
                                    return native
                                functions['shape'] = capture
                                with amp(device, 'bf16'), patch.object(gen, 'loss_fn', functions):
                                    loss, _ = gen.loss(targets, ca[0], **kwargs)
                                assert len(per_object) == 1 and torch.isfinite(loss)
                                report['native'].append(dict(split=split, group=gi, condition=condition,
                                    draw=draw, object_ids=ids, sample_ids=batch['sample_id'], losses=per_object[0]))
                                if condition not in ('correct', 'wrong_1'):
                                    continue
                                gen.no_shortcut = True; gen.inference_steps = 25
                                gen.rescale_t = float(cfg.get('ss_rescale_t', 3))
                                gen.reverse_fn.interval = list(cfg.get('ss_cfg_interval', [0, 500]))
                                gen.reverse_fn.strength = 0; gen.reverse_fn.training = False
                                torch.manual_seed(draw_seed + 100000)
                                noise = gen._generate_x0(targets)
                                with amp(device, 'bf16'), patch.object(gen, '_generate_noise', side_effect=lambda *a, **k: {n: v.clone() for n, v in noise.items()}):
                                    pred = gen({n: tuple(v.shape) for n, v in targets.items()}, device, ca[0], **kwargs)['shape']
                                if not torch.isfinite(pred).all():
                                    raise FloatingPointError('Nonfinite sampled shape')
                                supports = []
                                for i, record in enumerate(records):
                                    with amp(device, 'bf16'):
                                        support = occupancy(decoder, pred[i:i + 1])
                                    supports.append(support)
                                    report['samples'].append(dict(split=split, group=gi, condition=condition,
                                        draw=draw, sample_id=record['sample_id'], object_id=record['object_id'],
                                        noise_sha256={n: tensor_digest(v[i]) for n, v in noise.items()},
                                        **support_metrics(support, target_support[i])))
                                np.savez_compressed(output / f'{split}_g{gi}_{condition}_d{draw}.npz',
                                    predicted_occupancy=np.stack(supports), target_occupancy=target_support,
                                    sample_ids=np.array(batch['sample_id']))
                            assert seen
                        finally:
                            hook.remove()
                    save(); print('Assessment', epoch, split, gi + 1, '/', len(groups), flush=True)
    finally:
        gen.reverse_fn.training = True
        random.setstate(python_state); np.random.set_state(numpy_state)
        del decoder
    report['complete'] = True
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
