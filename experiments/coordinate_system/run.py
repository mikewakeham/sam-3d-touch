import gc
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from omegaconf import OmegaConf

from dataloader import TouchDataset, collate_touch_batch, load_data_config
from train import prepare_batch
from evaluation.evaluate import (
    build_pipeline, read_run, restore_run, select_records, stable_seed,
    sample_stage1, decode_voxels, save_stage1_artifact, safe_name, write_metrics,
)
from evaluation.geometry import load_dataset_mesh
from evaluation.input_visualizations import joint_camera_points
from evaluation.pose_metrics import save_pose
from experiments.coordinate_system.analyze import analyze, rotation_matrix, rotation_metrics

# Edit these, then run: python experiments/coordinate_system/run.py
N_OBJECTS = 50
SEED = 29
STEPS = 25
DEVICE = 'cuda'
OUTPUT = Path('experiments/coordinate_system/results')
PIPELINE = Path('checkpoints/hf/pipeline.yaml')
REFERENCE = Path('outputs/zeroverse/evaluation_lr1e4_sampled/selected_samples.json')
DATA = {
    'old': 'configs/data_zeroverse_5000_8views_touch_32x256.yaml',
    'sampled': 'configs/data_zeroverse_5000_8views_sampled_touch_32x256.yaml',
}
CHECKPOINTS = {
    'base': None,
    'old': 'outputs/zeroverse/zeroverse_pointmap_touch_32x256_joint/best.pt',
    'new': 'outputs/zeroverse/zeroverse_pointmap_touch_32x256_joint_sampled_camframe/best.pt',
}


@torch.inference_mode()
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configs, datasets = {}, {}
    for name, path in DATA.items():
        configs[name] = load_data_config(path)
        configs[name]['dataset']['split'] = 'val'
        datasets[name] = TouchDataset(configs[name], joint_pointmap=True)

    # Same object selection as normal evaluation; keep all eight views this time.
    selected, _ = select_records(datasets['sampled'], configs['sampled'], N_OBJECTS, SEED, 'random')
    objects = {r['object_id'] for r in selected}
    normal_objects = {r['object_id'] for r in json.loads(REFERENCE.read_text())}
    assert len(objects) == N_OBJECTS and objects <= normal_objects
    for dataset in datasets.values():
        dataset.records = sorted([r for r in dataset.records if r['object_id'] in objects],
                                 key=lambda r: (r['object_id'], int(r['view_id'])))
        for obj in objects:
            assert len({r['view_id'] for r in dataset.records if r['object_id'] == obj}) == 8
        assert len(dataset.records) == N_OBJECTS * 8

    selection, rotations, meshes = [], {}, {}
    for name, dataset in datasets.items():
        rotations[name] = []
        for record in dataset.records:
            obj, sample = record['object_id'], record['sample_id']
            if (name, obj) not in meshes:
                meshes[name, obj], _ = load_dataset_mesh(record, dataset)
                if name == 'sampled':
                    np.testing.assert_allclose(meshes[name, obj].vertices, meshes['old', obj].vertices, atol=1e-6)
            folder = OUTPUT / 'inputs' / name / safe_name(sample)
            folder.mkdir(parents=True, exist_ok=True)
            with np.load(dataset.resolve_path(record['camera_path'])) as archive:
                camera = dict(archive)
            rotations[name].append(camera['rotation'])
            # Save camera-frame target normalization too, for later shape visualization.
            vertices = meshes[name, obj].vertices @ camera['rotation'].T
            center = (vertices.min(0) + vertices.max(0)) / 2
            extent = np.ptp(vertices, axis=0).max()
            target_from_object = np.eye(4)
            target_from_object[:3, :3] = camera['rotation'] / extent
            target_from_object[:3, 3] = -center / extent
            camera_from_target = np.eye(4)
            camera_from_target[:3, :3] *= extent
            camera_from_target[:3, 3] = center + camera['translation']
            np.savez_compressed(folder / 'camera.npz', **camera,
                                T_target_from_object=target_from_object, T_camera_from_target=camera_from_target)
            shutil.copy2(dataset.resolve_path(record['image_path']), folder / 'image.png')
            paths = {k: str(dataset.resolve_path(v).resolve()) for k, v in record.items() if k.endswith('_path')}
            selection.append(dict(record, **paths, camera_set=name))

    references = np.array(rotations['old'][:8])
    np.testing.assert_allclose(np.array(rotations['old']).reshape(N_OBJECTS, 8, 3, 3),
                               np.broadcast_to(references, (N_OBJECTS, 8, 3, 3)), atol=1e-5)
    np.savez(OUTPUT / 'reference_cameras.npz', rotations=references,
             view_ids=[r['view_id'] for r in datasets['old'].records[:8]])
    (OUTPUT / 'selected_samples.json').write_text(json.dumps(selection, indent=2))
    settings = dict(objects=N_OBJECTS, seed=SEED, steps=STEPS, checkpoints=CHECKPOINTS,
                    data=configs, pipeline=str(PIPELINE), reference=str(REFERENCE))
    (OUTPUT / 'settings.json').write_text(json.dumps(settings, indent=2))

    rows = []  # Rerunning starts over and overwrites the result files.
    for name, checkpoint_path in CHECKPOINTS.items():
        pipeline, pipeline_config = build_pipeline(PIPELINE, DEVICE)
        pipeline.ss_generator.no_shortcut = True
        model, checkpoint = None, None
        if checkpoint_path is not None:
            checkpoint, config, _ = read_run(Path(checkpoint_path))
            assert checkpoint['conditioning_config']['joint_pointmap_points'] == 1024
            model = restore_run(pipeline, checkpoint, DEVICE)
            (OUTPUT / f'{name}_config.json').write_text(json.dumps(config, indent=2))
        (OUTPUT / 'pipeline.json').write_text(json.dumps(OmegaConf.to_container(pipeline_config, resolve=True), indent=2))
        for camera_set, dataset in datasets.items():
            dataset.include_touch = model is not None
            for index, record in enumerate(dataset.records):
                batch = collate_touch_batch([dataset[index]])
                _, condition_args, condition_kwargs, points, mask, inputs = prepare_batch(
                    pipeline, batch, torch.device(DEVICE), 'bf16', model is not None,
                    joint_pointmap=model is not None, fixed_joint_patches=True, return_inputs=True)
                seed = stable_seed(SEED, record['sample_id'])
                torch.manual_seed(seed)
                with torch.autocast(device_type=torch.device(DEVICE).type,
                                    dtype=pipeline.shape_model_dtype, enabled=DEVICE.startswith('cuda')):
                    tokens = model.get_touch_tokens(points, mask) if model is not None else None
                    outputs = sample_stage1(pipeline, condition_args, condition_kwargs, tokens, STEPS, DEVICE)
                    voxels = decode_voxels(pipeline.models['ss_decoder'], outputs['shape'])
                    # This is only a saved canonical reference, never inference conditioning.
                    target = decode_voxels(pipeline.models['ss_decoder'], batch['target_shape'].to(DEVICE))
                coords = torch.argwhere(voxels).int()
                cloud = points[0, mask[0]].cpu().numpy() if model is not None else np.empty((0, 3))
                camera_cloud = (joint_camera_points(points[0, mask[0]], inputs, pipeline.ss_preprocessor)
                                if model is not None else np.empty((0, 3)))
                stage1_path = save_stage1_artifact(
                    OUTPUT, f'{name}_{camera_set}', record['sample_id'], outputs['shape'], voxels[0],
                    target[0], coords, coords, 1, cloud,
                    dict(encoder_input_camera=camera_cloud, target_coordinate_frame='canonical_reference_only'))
                pose_path = stage1_path.with_name('pose.npz')
                save_pose(pose_path, pipeline, outputs, inputs, 1, dataset.resolve_path(record['camera_path']))
                with np.load(pose_path) as pose:
                    metrics = rotation_metrics(rotation_matrix(pose['prediction_transform']),
                                               rotation_matrix(pose['target_transform']), references, name == 'new')
                rows.append(dict(model=name, camera_set=camera_set, object_id=record['object_id'],
                                 sample_id=record['sample_id'], view_id=record['view_id'], seed=seed,
                                 pose_path=str(pose_path.relative_to(OUTPUT)),
                                 stage1_path=str(stage1_path.relative_to(OUTPUT)), **metrics))
                write_metrics(OUTPUT / 'metrics.csv', rows)
                print(name, camera_set, index + 1, '/', len(dataset), flush=True)
        del pipeline, model, checkpoint, outputs, tokens, voxels, target, inputs, condition_args, condition_kwargs
        gc.collect()
        torch.cuda.empty_cache()
    analyze(OUTPUT)


if __name__ == '__main__':
    main()
