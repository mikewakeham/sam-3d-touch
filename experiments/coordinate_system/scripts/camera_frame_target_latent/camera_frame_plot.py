"""Static, labeled 3D panels of measured geometry; no independent rescaling of overlays."""
import numpy as np


def plot_transformations(path, sample_id, object_points, camera_surface, camera_pointmap,
                         stock_pointmap, frames, image=None, decoded_points=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .camera_frame_geometry import affine

    camera_mesh = affine(object_points, frames['camera_from_object'])
    target_mesh = affine(camera_mesh, frames['target_from_camera'])
    target_surface = affine(camera_surface, frames['target_from_camera'])
    vec_surface = affine(camera_surface, frames['vec_from_camera'])
    shared_pointmap = affine(camera_pointmap, frames['vec_from_camera'])
    # Identical sampling across panels of each point set; display only, no training subsampling.
    def thin(points):
        return points[np.linspace(0, len(points)-1, min(len(points), 1800), dtype=int)]
    panels = [
        ('1. Original object frame', [('mesh samples', object_points, '#2864b7')], None),
        ('2. Recorded camera transform', [('mesh in camera axes', camera_mesh, '#2864b7'),
                                         ('stored full surface', camera_surface, '#ec8c28'),
                                         ('raw visible pointmap', camera_pointmap, '#128b68')], None),
        ('3. Camera-oriented target / unit box', [('target mesh', target_mesh, '#2864b7'),
                                              ('surface in target units (display only)', target_surface, '#ec8c28')], .6),
        ('4. Existing conditioning units', [('VecSetX input', vec_surface, '#ec8c28'),
                                            ('stock-normalized pointmap', stock_pointmap, '#128b68')], 'condition'),
        ('5. Shared conditioning units', [('same VecSetX input', vec_surface, '#ec8c28'),
                                          ('pointmap using surface center/radius', shared_pointmap, '#128b68')], 'condition'),
        ('6. Remaining target/VecSetX unit difference', [('target mesh', target_mesh, '#2864b7'),
                                                       ('actual VecSetX input', vec_surface, '#ec8c28')], 1.1),
    ]
    if decoded_points is not None:
        panels.append(('7. Frozen VAE target reconstruction', [('target mesh', target_mesh, '#2864b7'),
                                                               ('decoded occupied voxels', decoded_points, '#9e42b4')], .6))
    figure = plt.figure(figsize=(16, 5.1*((len(panels)+2)//3)))
    condition_limit = max(1.1, float(np.abs(stock_pointmap).max())*1.05,
                          float(np.abs(shared_pointmap).max())*1.05)
    for index, (title, sets, limit) in enumerate(panels):
        ax = figure.add_subplot((len(panels)+2)//3, 3, index+1, projection='3d')
        for label, points, color in sets:
            points = thin(points)
            ax.scatter(*points.T, s=2, alpha=.55, color=color, label=label, rasterized=True)
        if limit == 'condition': limit = condition_limit
        if limit is None:
            all_points = np.concatenate([v for _, v, _ in sets])
            center = (all_points.min(0)+all_points.max(0))/2
            radius = max(float(np.ptp(all_points, axis=0).max())*.55, 1e-6)
        else:
            center = np.zeros(3); radius = limit
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-55)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(title, fontsize=10); ax.legend(fontsize=7, loc='upper left')
    if image is not None and len(panels) < ((len(panels)+2)//3)*3:
        ax = figure.add_subplot((len(panels)+2)//3, 3, len(panels)+1)
        ax.imshow(image); ax.set_title('Conditioning image'); ax.axis('off')
    figure.suptitle(f'{sample_id}\nBlue = target/source geometry; orange = full surface; green = observed pointmap. '
                     'Panels 4–5 share identical axes. Panel 6 is deliberately NOT aligned.', fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .95)); figure.savefig(path, dpi=160); plt.close(figure)
