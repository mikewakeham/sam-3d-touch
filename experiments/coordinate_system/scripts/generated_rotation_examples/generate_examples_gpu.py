"""Fixed validation views: actual Stage-1 predictions and their input camera transforms."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image


def save_inputs(dataset, record, output):
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dataset.resolve_path(record['image_path']), output/'input_view.png')
    rgba = np.asarray(Image.open(output/'input_view.png').convert('RGBA'))
    pm = np.load(dataset.resolve_path(record['pointmap_path']))
    valid = np.isfinite(pm).all(-1) & (rgba[..., 3] > 0) & (pm[..., 2] > 0)
    ids = np.flatnonzero(valid)
    ids = ids[np.linspace(0, len(ids)-1, min(5000, len(ids)), dtype=int)] if len(ids) else ids
    surface = dataset.load_full_surface(dataset.resolve_path(record['full_surface_path']))
    np.savez_compressed(output/'input_clouds.npz', surface=surface,
        pointmap=pm.reshape(-1, 3)[ids], pointmap_rgb=rgba.reshape(-1, 4)[ids, :3]/255.)
    with np.load(dataset.resolve_path(record['camera_path'])) as camera:
        transform = np.diag([-1., -1., 1., 1.]) @ camera['T_camera_from_object']
    return transform


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--checkpoint', default='best.pt')
    p.add_argument('--pipeline-config', type=Path, default=REPO/'checkpoints/hf/pipeline.yaml')
    p.add_argument('--val-objects', type=int, default=8)
    p.add_argument('--views', type=int, default=3)
    p.add_argument('--draws', type=int, default=1)
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--inference-steps', type=int, default=25)
    p.add_argument('--cfg-strength', type=float, default=0.)
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()

    import torch
    from omegaconf import OmegaConf
    from dataloader import TouchDataset, collate_touch_batch
    from evaluate import read_run, restore_run, stable_seed
    from train import amp, build_stage1_pipeline, prepare_batch
    from experiments.coordinate_system.scripts.full_checkpoints.protocol import evaluation_groups
    from experiments.coordinate_system.scripts.full_checkpoints.checkpoint_rollout_core import sample_from_noise

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    checkpoint, _, data = read_run(args.run_dir/args.checkpoint)
    conditioning = checkpoint.get('conditioning_config', {})
    if conditioning.get('oracle_point_frame', False):
        p.error('Use a checkpoint trained without oracle for this camera-orientation test.')
    pipeline = build_stage1_pipeline(args.pipeline_config, device)
    model = restore_run(pipeline, checkpoint, device).requires_grad_(False).eval()
    step = checkpoint['step']
    mode = checkpoint['mode']
    del checkpoint
    data = copy.deepcopy(data)
    data['dataset']['split'] = 'val'
    dataset = TouchDataset(data, include_touch=True, oracle_point_frame=False)
    groups = evaluation_groups(dataset.records, args.seed, args.val_objects, args.views)
    lookup = {r['sample_id']: i for i, r in enumerate(dataset.records)}
    cfg = OmegaConf.load(args.pipeline_config)
    gen = pipeline.ss_generator
    gen.fm_eps_max = 0.
    gen.self_consistency_prob = 0.
    gen.reverse_fn.p_unconditional = 0.
    pipeline.override_ss_generator_cfg_config(gen, cfg_strength=args.cfg_strength,
        inference_steps=args.inference_steps, rescale_t=float(cfg.get('ss_rescale_t', 3)),
        cfg_interval=list(cfg.get('ss_cfg_interval', [0, 500])), cfg_strength_pm=0.)
    gen.no_shortcut = True
    gen.reverse_fn.training = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoint_step=step, conditioning=conditioning,
        selection='Fixed validation identities/views chosen from metadata before generation; all selected predictions retained.',
        examples=[])
    with torch.no_grad():
        for gi, records in enumerate(groups):
            batch = collate_touch_batch([dataset[lookup[r['sample_id']]] for r in records])
            use_touch = model.touch_encoder is not None
            targets, ca, kw, xyz, mask = prepare_batch(pipeline, batch, device, args.precision,
                use_touch, joint_pointmap=mode == 'image_touch_joint',
                oracle_point_frame=False,
                shared_pointmap_normalization=conditioning.get('shared_pointmap_normalization', False))
            with amp(device, args.precision):
                tokens = model.get_touch_tokens(xyz, mask)
            seed = stable_seed(args.seed, '|'.join(batch['sample_id']))
            transforms = {}
            for record in records:
                transforms[record['sample_id']] = save_inputs(dataset, record,
                    args.output_dir/'inputs'/record['sample_id'])
            for draw in range(args.draws):
                sample_seed = seed+500000+draw
                torch.manual_seed(sample_seed)
                noise = gen._generate_noise({k: tuple(v.shape) for k, v in targets.items()}, device)
                with amp(device, args.precision):
                    prediction = sample_from_noise(gen, noise, ca[0], tokens)
                for i, record in enumerate(records):
                    predicted = prediction[i].float().cpu().numpy()
                    target = targets['shape'][i].float().cpu().numpy()
                    filename = f"{record['sample_id']}_draw{draw}.npz"
                    np.savez_compressed(args.output_dir/filename, prediction=predicted, target=target,
                        camera_from_object=transforms[record['sample_id']])
                    report['examples'].append(dict(object_id=record['object_id'], sample_id=record['sample_id'],
                        draw=draw, seed=sample_seed, latent_file=filename,
                        input_dir=f"inputs/{record['sample_id']}",
                        latent_mse=float(np.mean((predicted.astype(np.float64)-target)**2))))
            (args.output_dir/'examples.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'Generated group {gi+1}/{len(groups)}', flush=True)
    print(f"Saved all {len(report['examples'])} predictions and input views. No decoding or target re-encoding.")


if __name__ == '__main__':
    main()
