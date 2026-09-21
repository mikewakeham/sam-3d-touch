# Mesh protocol copied from the original SAM3D evaluate.py.
import itertools

import numpy as np
from scipy.spatial import cKDTree


def normalize_mesh(mesh):
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    scale = 2.0 / (bounds[1] - bounds[0]).max()
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -scale * center
    mesh.apply_transform(transform)
    return mesh, transform


def sample_surface(mesh, count, seed):
    import trimesh

    points, faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = np.asarray(mesh.face_normals[faces], dtype=np.float32)
    return points.astype(np.float32), normals


def proper_axis_rotations():
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3))
            rotation[range(3), permutation] = signs
            if np.linalg.det(rotation) > 0:
                rotations.append(rotation)
    return rotations


def principal_basis(points):
    covariance = np.cov(points, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def symmetric_distance(source, target, workers):
    source_to_target = cKDTree(target).query(source, workers=workers)[0].mean()
    target_to_source = cKDTree(source).query(target, workers=workers)[0].mean()
    return float((source_to_target + target_to_source) / 2)


def align_mesh(prediction_mesh, target_mesh, prediction_points, target_points, workers):
    import open3d as o3d
    import trimesh

    source_basis = principal_basis(prediction_points)
    target_basis = principal_basis(target_points)
    source_center = prediction_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_radius = np.sqrt(np.mean(np.sum((prediction_points - source_center) ** 2, axis=1)))
    target_radius = np.sqrt(np.mean(np.sum((target_points - target_center) ** 2, axis=1)))
    scale = target_radius / source_radius
    candidates = []
    coarse_source = prediction_points[::5]
    coarse_target = target_points[::5]
    for axis_rotation in proper_axis_rotations():
        transform = np.eye(4)
        transform[:3, :3] = scale * target_basis @ axis_rotation @ source_basis.T
        transform[:3, 3] = target_center - transform[:3, :3] @ source_center
        transformed = trimesh.transform_points(coarse_source, transform)
        candidates.append((symmetric_distance(transformed, coarse_target, workers), transform))

    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prediction_points))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
    refined = []
    for _, transform in sorted(candidates, key=lambda item: item[0])[:4]:
        registration = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            0.2,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        transformed = trimesh.transform_points(prediction_points, registration.transformation)
        refined.append((
            symmetric_distance(transformed, target_points, workers),
            registration.transformation,
            float(registration.fitness),
            float(registration.inlier_rmse),
        ))

    error, transform, fitness, inlier_rmse = min(refined, key=lambda item: item[0])
    aligned = prediction_mesh.copy()
    aligned.apply_transform(transform)
    return aligned, transform, error, fitness, inlier_rmse


def voxelize_points(points, resolution=64):
    occupancy = np.zeros((resolution, resolution, resolution), dtype=bool)
    coords = np.floor((points + 1) * resolution / 2).astype(np.int64)
    if len(coords):
        coords = coords[np.all((coords >= 0) & (coords < resolution), axis=1)]
        occupancy[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return occupancy


def sinkhorn_emd(prediction, target, device, epsilon=0.01, iterations=100):
    import torch

    prediction = torch.from_numpy(prediction).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(target).to(device=device, dtype=torch.float32)
    cost = torch.cdist(prediction, target).square()
    log_kernel = -cost / epsilon
    log_mass_prediction = -np.log(len(prediction))
    log_mass_target = -np.log(len(target))
    log_u = torch.zeros(len(prediction), device=device)
    log_v = torch.zeros(len(target), device=device)
    for _ in range(iterations):
        log_u = log_mass_prediction - torch.logsumexp(log_kernel + log_v[None], dim=1)
        log_v = log_mass_target - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    transport = torch.exp(log_kernel + log_u[:, None] + log_v[None])
    return float(torch.sqrt((transport * cost).sum()).cpu())


def mesh_metrics(prediction_points, prediction_normals, target_points, target_normals,
                 emd_points, device, workers):
    target_tree = cKDTree(target_points)
    prediction_tree = cKDTree(prediction_points)
    prediction_distances, prediction_neighbors = target_tree.query(
        prediction_points, workers=workers
    )
    target_distances, target_neighbors = prediction_tree.query(
        target_points, workers=workers
    )

    threshold = 0.01
    f_precision = float(np.mean(prediction_distances < threshold))
    f_recall = float(np.mean(target_distances < threshold))
    fscore = 2 * f_precision * f_recall / max(f_precision + f_recall, 1e-12)
    normal_forward = np.abs(np.sum(prediction_normals * target_normals[prediction_neighbors], axis=1)).mean()
    normal_backward = np.abs(np.sum(target_normals * prediction_normals[target_neighbors], axis=1)).mean()

    prediction_voxels = voxelize_points(prediction_points)
    target_voxels = voxelize_points(target_points)
    intersection = np.logical_and(prediction_voxels, target_voxels).sum()
    union = np.logical_or(prediction_voxels, target_voxels).sum()
    emd_count = min(emd_points, len(prediction_points), len(target_points))
    return {
        "fscore_0.01": fscore,
        "f_precision_0.01": f_precision,
        "f_recall_0.01": f_recall,
        "voxel_iou_64": float(intersection / max(union, 1)),
        "chamfer": float((prediction_distances.mean() + target_distances.mean()) / 2),
        "normal_consistency": float((normal_forward + normal_backward) / 2),
        "emd": sinkhorn_emd(
            prediction_points[:emd_count], target_points[:emd_count], device
        ) if emd_count > 0 else float("nan"),
    }


def main():
    """Re-score saved SAM3D meshes without loading a checkpoint or generating again."""
    import argparse
    import csv
    import json
    from pathlib import Path
    from evaluation.geometry import load_mesh, resolve

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Separate CSV; never replaces the inference metrics')
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--sample-id')
    parser.add_argument('--surface-points', type=int, default=1_000_000)
    parser.add_argument('--emd-points', type=int, default=0, help='0 disables optional Sinkhorn distance')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.surface_points < 1 or args.emd_points < 0 or args.workers == 0 or args.workers < -1:
        raise ValueError('Invalid sample counts or worker count')
    source = args.evaluation_dir / 'metrics.csv'
    output = args.output or args.evaluation_dir / 'metrics_rescored.csv'
    if output.resolve() == source.resolve():
        raise ValueError('Use a separate output CSV to preserve inference/resume records')
    with source.open(newline='') as file:
        rows = [row for row in csv.DictReader(file)
                if (not args.conditions or row['condition'] in args.conditions)
                and (not args.sample_id or row['sample_id'] == args.sample_id)]
    if not rows:
        raise ValueError('No matching evaluation rows')
    results = []
    target_cache = {}
    for index, row in enumerate(rows, 1):
        result = {key: row[key] for key in ('condition', 'sample_id', 'object_id')}
        result.update({key: float('nan') for key in (
            'fscore_0.01', 'f_precision_0.01', 'f_recall_0.01', 'voxel_iou_64',
            'chamfer', 'normal_consistency', 'emd')})
        result['error'] = row['error']
        try:
            if row['error']:
                raise ValueError(f"Inference failed: {row['error']}")
            target_path = resolve(args.evaluation_dir, row['target_mesh_path'])
            if target_path not in target_cache:
                with np.load(resolve(args.evaluation_dir, row['target_points_path']), allow_pickle=False) as data:
                    seed = int(data['surface_seed'])
                target_cache[target_path] = sample_surface(load_mesh(target_path), args.surface_points, seed)
            with np.load(resolve(args.evaluation_dir, row['alignment_path']), allow_pickle=False) as data:
                seed = int(data['surface_seed'])
            points, normals = sample_surface(
                load_mesh(resolve(args.evaluation_dir, row['mesh_aligned_path'])), args.surface_points, seed)
            target_points, target_normals = target_cache[target_path]
            result.update(mesh_metrics(points, normals, target_points, target_normals,
                                       args.emd_points, args.device, args.workers))
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        results.append(result)
        print(f"[{index}/{len(rows)}] {row['condition']} / {row['sample_id']} {result['error']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {}
    for name in sorted({row['condition'] for row in results}):
        condition = [row for row in results if row['condition'] == name]
        summary[name] = {'completed': sum(not row['error'] for row in condition),
                         'failed': sum(bool(row['error']) for row in condition)}
        for key in ('chamfer', 'fscore_0.01', 'normal_consistency', 'voxel_iou_64', 'emd'):
            values = [row[key] for row in condition if not row['error'] and np.isfinite(row[key])]
            summary[name][key] = {'mean': float(np.mean(values)), 'median': float(np.median(values))} if values else None
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(alignment='saved mesh_aligned, no new registration',
                    chamfer='symmetric mean unsquared Euclidean distance',
                    fscore_threshold=.01, voxel_iou_64='sampled surface occupancy, not filled volume',
                    summary=summary)
    output.with_suffix('.json').write_text(json.dumps(settings, indent=2) + '\n')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
