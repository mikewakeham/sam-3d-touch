import math

import numpy as np
import torch
from point_cloud_utils import k_nearest_neighbors
from scipy.spatial import ConvexHull, cKDTree
from trimesh import util
from trimesh.transformations import transform_points
from trimesh.typed import ArrayLike, Optional

from evaluation.metrics import sample_surface


def save_pose(path, pipeline, outputs, inputs, downsample_factor, camera_path, target_path=None):
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    saved_pose = {
        "raw_" + key: value.detach().float().cpu().numpy()
        for key, value in outputs.items() if key != "shape"
    }
    saved_pose["pointmap_scale"] = inputs["pointmap_scale"].detach().float().cpu().numpy()
    saved_pose["pointmap_shift"] = inputs["pointmap_shift"].detach().float().cpu().numpy()
    saved_pose["downsample_factor"] = float(downsample_factor)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **saved_pose)
    decoded = pipeline.pose_decoder(
        outputs, scene_scale=inputs["pointmap_scale"], scene_shift=inputs["pointmap_shift"],
    )
    scale = decoded["scale"] * downsample_factor
    transform = compose_transform(
        scale=scale, rotation=quaternion_to_matrix(decoded["rotation"]),
        translation=decoded["translation"],
    ).get_matrix()[0].T
    saved_pose.update(
        rotation=decoded["rotation"].detach().float().cpu().numpy(),
        translation=decoded["translation"].detach().float().cpu().numpy(),
        scale=scale.detach().float().cpu().numpy(),
        prediction_transform=transform.detach().float().cpu().numpy(),
        coordinate_frame="sam_camera",
        transform_convention="column_vectors",
        rotation_representation="quaternion_wxyz_row_vectors",
    )
    with np.load(camera_path, allow_pickle=False) as camera:
        expected = {"pose_source_frame": "normalized_object", "pose_coordinate_frame": "sam_camera",
                    "pose_rotation_convention": "column_vectors"}
        if any(str(camera[key]) != value for key, value in expected.items()):
            raise ValueError(f"Unsupported ground-truth pose convention: {camera_path}")
        target = np.eye(4)
        target[:3, :3] = camera["rotation"] @ np.diag(camera["scale"])
        target[:3, 3] = camera["translation"]
        if not np.allclose(target, np.diag([-1., -1., 1., 1.]) @ camera["T_camera_from_object"],
                           atol=1e-5, rtol=1e-5):
            raise ValueError(f"Ground-truth pose disagrees with camera transform: {camera_path}")
    if not np.isfinite(saved_pose["prediction_transform"]).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite object pose")
    if not np.all(saved_pose["scale"] > 0):
        raise ValueError("Predicted scale must be positive")
    saved_pose["target_transform"] = target
    if target_path is not None:
        from dataloader import load_camera_target_transform
        saved_pose["camera_target_transform"] = load_camera_target_transform(target_path, camera_path)
    np.savez_compressed(path, **saved_pose)


def pose_head_errors(prediction_transform, camera_target_transform, downsample_factor):
    """Compare raw head placement to the normalized camera-shape target, before geometry downsampling."""
    prediction_scale = np.linalg.norm(prediction_transform[:3, :3], axis=0)
    target_scale = np.linalg.norm(camera_target_transform[:3, :3], axis=0)
    if (not np.isfinite(downsample_factor) or downsample_factor <= 0
            or np.any(prediction_scale <= 0) or np.any(target_scale <= 0)):
        raise ValueError("Pose head errors require positive scales and downsample factor")
    prediction_rotation = prediction_transform[:3, :3] / prediction_scale
    target_rotation = camera_target_transform[:3, :3] / target_scale
    return {
        "pose_head_rotation_deg": float(re(prediction_rotation, target_rotation)),
        "pose_head_translation_error": float(np.linalg.norm(prediction_transform[:3, 3] - camera_target_transform[:3, 3])),
        "pose_head_scale_rel_error": float(np.mean(np.abs(prediction_scale / downsample_factor / target_scale - 1))),
    }


def score_pose(raw_mesh, object_mesh, pose_path, surface_points, icp_points, seed):
    with np.load(pose_path, allow_pickle=False) as pose:
        prediction = raw_mesh.copy().apply_transform(pose["prediction_transform"])
        target = object_mesh.copy().apply_transform(pose["target_transform"])
        head_errors = (pose_head_errors(pose["prediction_transform"], pose["camera_target_transform"],
                                       float(pose["downsample_factor"]))
                       if "camera_target_transform" in pose else {})
    if not np.isfinite(prediction.vertices).all() or not np.isfinite(target.vertices).all():
        raise ValueError("Non-finite posed mesh")

    vertices = np.asarray(target.vertices)
    hull = ConvexHull(vertices)
    diameter = calc_pts_diameter(vertices[hull.vertices])
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Ground-truth diameter must be positive")
    bounds = [torch.tensor(mesh.bounds.reshape(1, 6), dtype=torch.float64)
              for mesh in (prediction, target)]
    box_iou = float(axis_aligned_bbox_overlaps_3d(*bounds, eps=1e-15).item())
    prediction_points, _ = sample_surface(prediction, surface_points, seed + 3)
    target_points, _ = sample_surface(target, surface_points, seed + 4)
    add_s = float(chamfer_distance(prediction_points, target_points) / (2 * diameter))
    source, _ = sample_surface(prediction, icp_points, seed + 5)
    destination, _ = sample_surface(target, icp_points, seed + 6)
    center = target.bounds.mean(axis=0)
    correction, _, _ = icp(
        (source - center) / diameter, (destination - center) / diameter,
        initial=np.eye(4), threshold=1e-7,
        max_iterations=100, scale=False, reflection=False,
    )
    return {
        **head_errors,
        "pose_box_iou_3d": box_iou,
        "pose_add_s": add_s,
        "pose_icp_rot_deg": float(re(correction[:3, :3], np.eye(3))),
    }


# https://github.com/open-mmlab/mmdetection3d
def axis_aligned_bbox_overlaps_3d(bboxes1,
                                  bboxes2,
                                  mode='iou',
                                  is_aligned=False,
                                  eps=1e-6):
    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'

    assert (bboxes1.size(-1) == 6 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 6 or bboxes2.size(0) == 0)


    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 3] -
             bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1]) * (
                 bboxes1[..., 5] - bboxes1[..., 2])
    area2 = (bboxes2[..., 3] -
             bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1]) * (
                 bboxes2[..., 5] - bboxes2[..., 2])

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])

        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(bboxes1[..., :, None, :3],
                       bboxes2[..., None, :, :3])
        rb = torch.min(bboxes1[..., :, None, 3:],
                       bboxes2[..., None, :, 3:])

        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :3],
                                    bboxes2[..., None, :, :3])
            enclosed_rb = torch.max(bboxes1[..., :, None, 3:],
                                    bboxes2[..., None, :, 3:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou']:
        return ious

    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


# https://github.com/fwilliams/point-cloud-utils
def chamfer_distance(x, y, return_index=False, p_norm=2, max_points_per_leaf=10):
    dists_x_to_y, corrs_x_to_y = k_nearest_neighbors(x, y, k=1,
                                                     squared_distances=False,
                                                     max_points_per_leaf=max_points_per_leaf)
    dists_y_to_x, corrs_y_to_x = k_nearest_neighbors(y, x, k=1,
                                                     squared_distances=False,
                                                     max_points_per_leaf=max_points_per_leaf)

    dists_x_to_y = np.linalg.norm(x[corrs_y_to_x] - y, axis=-1, ord=p_norm).mean()
    dists_y_to_x = np.linalg.norm(y[corrs_x_to_y] - x, axis=-1, ord=p_norm).mean()

    cham_dist = np.mean(dists_x_to_y) + np.mean(dists_y_to_x)

    if return_index:
        return cham_dist, corrs_x_to_y, corrs_y_to_x

    return cham_dist


# https://github.com/thodan/bop_toolkit
def re(R_est, R_gt):
    assert R_est.shape == R_gt.shape == (3, 3)
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))


    error_cos = min(1.0, max(-1.0, error_cos))

    error = math.acos(error_cos)
    error = 180.0 * error / np.pi
    return error


# https://github.com/thodan/bop_toolkit
def calc_pts_diameter(pts):
    diameter = -1.0
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter


# https://github.com/mikedh/trimesh
def procrustes(
    a: ArrayLike,
    b: ArrayLike,
    weights: Optional[ArrayLike] = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
    return_cost: bool = True,
):
    a_original = np.asanyarray(a, dtype=np.float64)
    b_original = np.asanyarray(b, dtype=np.float64)
    if not util.is_shape(a_original, (-1, 3)) or not util.is_shape(b_original, (-1, 3)):
        raise ValueError("points must be (n,3)!")
    if len(a_original) != len(b_original):
        raise ValueError("a and b must contain same number of points!")

    if weights is None:
        weights = np.ones(len(a_original))
    w = np.maximum(np.asanyarray(weights, dtype=np.float64), 0)
    if len(w) != len(a):
        raise ValueError("weights must have same length as a and b!")
    w_norm = (w / w.sum()).reshape((-1, 1))


    nonzero_weights = w_norm[:, 0] > 0.0
    a_nonzero = a_original[nonzero_weights]
    b_nonzero = b_original[nonzero_weights]
    w_norm = w_norm[nonzero_weights]


    if translation:

        acenter = (a_nonzero * w_norm).sum(axis=0)
        bcenter = (b_nonzero * w_norm).sum(axis=0)
    else:
        acenter = np.zeros(a_nonzero.shape[1])
        bcenter = np.zeros(b_nonzero.shape[1])


    if scale:


        ascale = np.sqrt((((a_nonzero - acenter) ** 2) * w_norm).sum())
        bscale = np.sqrt((((b_nonzero - bcenter) ** 2) * w_norm).sum())
    else:
        ascale = 1
        bscale = 1


    target = np.dot(((b_nonzero - bcenter) / bscale).T, ((a_nonzero - acenter) / ascale))

    u, _s, vh = np.linalg.svd(target)

    if reflection:
        R = np.dot(u, vh)
    else:

        R = np.dot(np.dot(u, np.diag([1, 1, np.linalg.det(np.dot(u, vh))])), vh)


    translation = bcenter - (bscale / ascale) * np.dot(R, acenter)
    matrix = np.hstack((bscale / ascale * R, translation.reshape(-1, 1)))
    matrix = np.vstack((matrix, np.array([0.0] * (a.shape[1]) + [1.0]).reshape(1, -1)))

    if return_cost:

        transformed = transform_points(a_original, matrix)


        cost = (((b_nonzero - transformed[nonzero_weights]) ** 2) * w_norm).sum()
        return matrix, transformed, cost

    return matrix


def icp(a, b, initial=None, threshold=1e-5, max_iterations=20, **kwargs):
    a = np.asanyarray(a, dtype=np.float64)
    if not util.is_shape(a, (-1, 3)):
        raise ValueError("points must be (n,3)!")

    if initial is None:
        initial = np.eye(4)

    is_mesh = util.is_instance_named(b, "Trimesh")
    if not is_mesh:
        b = np.asanyarray(b, dtype=np.float64)
        if not util.is_shape(b, (-1, 3)):
            raise ValueError("points must be (n,3)!")
        btree = cKDTree(b)


    a = transform_points(a, initial)
    total_matrix = initial


    old_cost = np.inf


    for _ in range(max_iterations):

        if is_mesh:
            closest, _distance, _faces = b.nearest.on_surface(a)
        else:
            _distances, ix = btree.query(a, 1)
            closest = b[ix]


        matrix, transformed, cost = procrustes(a=a, b=closest, **kwargs)


        a = transformed
        total_matrix = np.dot(matrix, total_matrix)

        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost

    return total_matrix, transformed, cost
