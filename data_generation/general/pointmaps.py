import numpy as np


def depth_to_pointmap(depth, K):
    """
    Back-project camera-axis depth into the camera convention expected by
    SAM 3D's external pointmap path.

    Blender/OpenCV camera:
        +x right, +y down, +z forward

    SAM 3D/PyTorch3D pointmap:
        +x left, +y up, +z forward
    """
    height, width = depth.shape

    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    pointmap = np.stack([-x, -y, z], axis=-1).astype(np.float32)

    valid = np.isfinite(depth) & (depth > 0)
    pointmap[~valid] = np.nan

    return pointmap


