"""Frozen multi-view oracle checkpoint: separate RGB, silhouette and pointmap effects.

Capture real post-projection fuser tokens, then cross complete streams between
normally preprocessed views. No raw pixel crossing, optimization or decoding.
"""
import argparse
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from omegaconf import OmegaConf

from dataloader import build_dataloader, collate_touch_batch, load_data_config
from experiments.noisy_target.evaluate import tensor_digest
from probe_tiny_fit_views import choose_views
from tiny_fit_gpu import parameter_digest
from train import TouchTrainingModel, amp, build_optimizer, build_stage1_pipeline, prepare_batch

STREAMS = {'rgb': ('image', 'rgb_image'),
           'silhouette': ('mask', 'rgb_image_mask'),
           'pointmap': ('pointmap', 'rgb_pointmap')}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_fit(explicit, reference):
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = []
        for root in (REPO / 'outputs/conditioning_investigation/multiple_view_fit',
                     REPO / 'outputs/multiple_view_fit'):
            if root.exists():
                candidates.extend(p.parent for p in root.rglob('results.json'))
    matches = []
    for candidate in candidates:
        path = candidate / 'results.json'
        if path.exists() and (candidate / 'fitted_parameters.pt').exists():
            if json.loads(path.read_text()) == reference:
                matches.append(candidate.resolve())
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise ValueError('Need exactly one matching completed oracle fit; use --fit-dir. '
                         f'Matches: {matches}. Expected the original multi-view oracle checkpoint.')
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path)
    parser.add_argument('--reference-root', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.partial.json').exists():
        raise FileExistsError(args.output)
    single_path = args.reference_root / 'tiny_fit_returned_46083371/oracle/results.json'
    multi_path = args.reference_root / 'multiple_view_returned_46083371/oracle.json'
    single, multi = (json.loads(p.read_text()) for p in (single_path, multi_path))
    assert multi['source_reference_sha256'] == digest(single_path)
    assert multi['driver_sha256'] == digest(HERE / 'fit_multiple_views_gpu.py')
    assert multi['view_selector_sha256'] == digest(HERE / 'probe_tiny_fit_views.py')
    assert multi['settings'] == dict(arm='oracle', steps=1000, seed=29, precision='bf16',
                                   objects=4, fit_views_per_object=4, reserved_views_per_object=3)
    fit_dir = locate_fit(args.fit_dir, multi)
    settings = single['settings']
    seed, precision = settings['seed'], settings['precision']
    for source, expected in single['source_sha256'].items():
        assert digest(REPO / source) == expected, source
    pipeline_path, data_path = Path(settings['pipeline_config']), Path(settings['data_config'])
    assert pipeline_path.read_text() == single['pipeline_yaml']
    assert data_path.read_text() == single['data_yaml']
    cfg = OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent / cfg.ss_generator_config_path).read_text() == single['generator_yaml']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda')
    pipeline = build_stage1_pipeline(pipeline_path, device)
    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
    encoder = TouchEncoder(output_dim=pipeline.backbone.cond_channels,
                           trainable=False, use_position=False).to(device).eval()
    model = TouchTrainingModel(pipeline.ss_generator, encoder, False, True)
    optimizer, _ = build_optimizer(encoder, pipeline.backbone, SimpleNamespace(
        learning_rate=1e-4, cross_attention_learning_rate=1e-5, cross_attention_scope='full'))
    del optimizer
    assert parameter_digest(model.named_parameters()) == multi['initial_all_parameters_sha256']
    archive = torch.load(fit_dir / 'fitted_parameters.pt', map_location='cpu', weights_only=True)
    assert archive['step'] == 1000 and archive['settings'] == multi['settings']
    assert archive['conditioning_config'] == model.conditioning_config
    parameters = dict(model.named_parameters())
    assert set(archive['model']) == {n for n, p in parameters.items() if p.requires_grad}
    with torch.no_grad():
        for name, value in archive['model'].items():
            parameters[name].copy_(value)
    del archive
    final_digest = parameter_digest(model.named_parameters())
    assert final_digest == multi['final_all_parameters_sha256']
    model.requires_grad_(False)
    fuser = pipeline.ss_condition_embedder
    assert fuser is not None and not fuser.training
    assert fuser.compression_projection_multiplier == 0 and not fuser.force_drop_modalities
    data = load_data_config(data_path)
    data['dataset']['split'] = 'train'
    ds = build_dataloader(data, 4, 0, shuffle=False, include_touch=True,
                          oracle_point_frame=True).dataset
    groups = choose_views(ds.records, single['sample_ids'], count=6)
    prepared, metadata = [], []
    for group, records in enumerate(groups):
        ds.records = records
        batch = collate_touch_batch([ds[i] for i in range(4)])
        captured = []
        original_dropout = fuser._dropout_modalities

        def capture(names, tokens):
            result = original_dropout(names, tokens)
            captured.append((list(names), [token.detach().clone() for token in result]))
            return result

        with patch.object(fuser, '_dropout_modalities', side_effect=capture):
            targets, ca, kw, points, mask = prepare_batch(
                pipeline, batch, device, precision, True, False, True)
        assert len(captured) == 1 and len(ca) == 1 and not kw
        names, tokens = captured[0]
        assert len(names) == len(set(names)) == 6
        assert set(names) == {n for values in STREAMS.values() for n in values}
        assert torch.equal(torch.cat(tokens, dim=1), ca[0]), 'Fuser capture changed tokens'
        with torch.no_grad(), amp(device, precision):
            pp, mm, _, _ = encoder.prepare_points(points, mask)
            features = encoder.encoder.encode(pp, mm)['x']
            touch = encoder.output_projection(features) + encoder.touch_embedding
        entry = dict(group=group, split='fit' if group < 4 else 'reserved_view',
                     sample_ids=[r['sample_id'] for r in records],
                     image_sha256=tensor_digest(batch['image']),
                     pointmap_sha256=tensor_digest(batch['pointmap']),
                     target_sha256=tensor_digest(targets['shape']),
                     features_sha256=tensor_digest(features))
        assert entry == multi['input_batches'][group], f'Input mismatch at group {group}'
        entry = {**entry, 'token_order': names,
                 'token_shapes': {n: list(t.shape) for n, t in zip(names, tokens)},
                 'token_sha256': {n: tensor_digest(t) for n, t in zip(names, tokens)}}
        metadata.append(entry)
        prepared.append((targets, ca[0], dict(zip(names, tokens)), touch))
    gen = pipeline.ss_generator
    gen.reverse_fn.training = True
    assert gen.reverse_fn.p_unconditional == 0 and gen.self_consistency_prob == 0 and gen.fm_eps_max == 0

    def losses(visual, touch, bank):
        values = []
        with torch.no_grad():
            for draw in range(8):
                torch.manual_seed(bank + seed + draw)
                random.seed(bank + seed + draw)
                with amp(device, precision):
                    loss, _ = gen.loss(prepared[0][0], visual, touch_tokens=touch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                values.append(float(loss))
        return values

    report = dict(settings={'fit_dir': str(fit_dir), 'arm': 'oracle', 'draws': 8,
                            'probe_bank_base': 300000, 'replay_bank_base': 100000,
                            'seed': seed, 'precision': precision},
                  source_single_sha256=digest(single_path), source_multi_sha256=digest(multi_path),
                  final_parameters_sha256=final_digest,
                  source_sha256={str(p.relative_to(REPO)): digest(p) for p in
                                 (Path(__file__), HERE / 'probe_tiny_fit_views.py',
                                  REPO / 'sam3d_objects/model/backbone/dit/embedder/embedder_fuser.py')},
                  input_batches=metadata, preflight=[], rows=[], complete=False,
                  runtime={'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()},
                  scope='Frozen four-object multi-view oracle model. Surface and target fixed to '
                        'anchor group 0. Mixed streams are inconsistent diagnostic conditions; '
                        'they do not establish a coordinate bug or a benefit from removing a modality.')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.with_suffix('.partial.json').write_text(json.dumps(report, indent=2) + '\n')

    expected_rows = multi['assessments'][-1]['rows']
    assert multi['assessments'][-1]['step'] == 1000
    save()
    for group, (_, visual, _, touch) in enumerate(prepared):
        for swap in (False, True):
            values = losses(visual, touch.roll(1, 0) if swap else touch, 100000)
            key = 'swapped_surface_native_losses' if swap else 'fresh_noise_native_losses'
            expected = expected_rows[group][key]
            np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-7,
                                       err_msg=f'Historical endpoint mismatch: group {group}, swap {swap}')
            report['preflight'].append(dict(group=group, swap=swap, losses=values,
                                           max_abs_error=float(np.max(np.abs(np.array(values)-expected)))))
        print('Replayed group', group, flush=True)
        save()
    report['preflight_passed'] = True
    save()
    # One independent common noise/time bank. Reuse the identical all-anchor cell.
    anchor = prepared[0]
    anchor_losses = {swap: losses(anchor[1], anchor[3].roll(1, 0) if swap else anchor[3], 300000)
                     for swap in (False, True)}
    for group in range(1, 7):
        for bits in itertools.product((0, 1), repeat=3):
            selected = dict(zip(STREAMS, bits))
            token_source = {n: group if selected[s] else 0 for s, ns in STREAMS.items() for n in ns}
            visual = torch.cat([prepared[token_source[n]][2][n] for n in metadata[0]['token_order']], dim=1)
            if bits == (0, 0, 0):
                assert torch.equal(visual, anchor[1])
            if bits == (1, 1, 1):
                assert torch.equal(visual, prepared[group][1])
            for swap in (False, True):
                values = anchor_losses[swap] if bits == (0, 0, 0) else losses(
                    visual, anchor[3].roll(1, 0) if swap else anchor[3], 300000)
                report['rows'].append(dict(group=group, split=metadata[group]['split'],
                                          other_view_streams=selected, swapped_surface=swap, losses=values))
        save()
        print('Completed RGB/silhouette/pointmap factorial for group', group, flush=True)
    assert parameter_digest(model.named_parameters()) == final_digest
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('Wrote', args.output, flush=True)


if __name__ == '__main__':
    main()
