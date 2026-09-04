import argparse
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.io_utils import axis_conversion
import numpy as np
from mathutils import Vector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-views", type=int, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--camera-radius", type=float, default=2.0)
    parser.add_argument("--fov-degrees", type=float, default=40.0)

    argv = sys.argv[sys.argv.index("--") + 1:]
    return parser.parse_args(argv)


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)

    for texture in list(bpy.data.textures):
        bpy.data.textures.remove(texture, do_unlink=True)

    for image in list(bpy.data.images):
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path):
    bpy.ops.import_scene.obj(filepath=str(object_path))

    mesh_objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
    ]

    if not mesh_objects:
        raise RuntimeError(f"No mesh found in {object_path}")

    return mesh_objects


def recalculate_normals(mesh_objects):
    for obj in mesh_objects:
        bpy.ops.object.select_all(action="DESELECT")

        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")

        obj.select_set(False)


def scene_bounds():
    bounds_min = Vector((math.inf, math.inf, math.inf))
    bounds_max = Vector((-math.inf, -math.inf, -math.inf))

    mesh_objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
    ]

    for obj in mesh_objects:
        for corner in obj.bound_box:
            corner = obj.matrix_world @ Vector(corner)

            bounds_min.x = min(bounds_min.x, corner.x)
            bounds_min.y = min(bounds_min.y, corner.y)
            bounds_min.z = min(bounds_min.z, corner.z)

            bounds_max.x = max(bounds_max.x, corner.x)
            bounds_max.y = max(bounds_max.y, corner.y)
            bounds_max.z = max(bounds_max.z, corner.z)

    return bounds_min, bounds_max


def normalize_scene():
    root_objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj.parent is None
    ]

    if len(root_objects) == 1:
        scene_root = root_objects[0]
    else:
        scene_root = bpy.data.objects.new("ObjectRoot", None)
        bpy.context.scene.collection.objects.link(scene_root)

        for obj in root_objects:
            obj.parent = scene_root

    bounds_min, bounds_max = scene_bounds()

    extent = bounds_max - bounds_min
    scale = 1.0 / max(extent)

    scene_root.scale *= scale
    bpy.context.view_layer.update()

    bounds_min, bounds_max = scene_bounds()
    offset = -(bounds_min + bounds_max) / 2.0

    scene_root.matrix_world.translation += offset
    bpy.context.view_layer.update()

    T_normalized_from_source = np.eye(4, dtype=np.float32)
    T_normalized_from_source[:3, :3] *= scale
    T_normalized_from_source[:3, 3] = np.array(offset)

    T_blender_from_source = np.array(
        axis_conversion(from_forward="-Z", from_up="Y").to_4x4(),
        dtype=np.float32,
    )

    return T_normalized_from_source @ T_blender_from_source


def initialize_render(resolution):
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True

    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.filter_type = "BOX"
    scene.cycles.filter_width = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.use_denoising = True

    cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
    cycles_preferences.compute_device_type = "CUDA"
    cycles_preferences.get_devices()

    for device in cycles_preferences.devices:
        device.use = device.type == "CUDA"

    if not any(device.type == "CUDA" for device in cycles_preferences.devices):
        raise RuntimeError("No CUDA rendering device found")

    scene.cycles.device = "GPU"


def initialize_depth_output():
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    scene.use_nodes = True
    view_layer.use_pass_z = True

    nodes = scene.node_tree.nodes
    links = scene.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    render_layers = nodes.new("CompositorNodeRLayers")
    render_layers.layer = view_layer.name

    depth_output = nodes.new("CompositorNodeOutputFile")
    depth_output.file_slots[0].use_node_format = True
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_mode = "RGB"
    depth_output.format.color_depth = "32"

    links.new(render_layers.outputs["Depth"], depth_output.inputs[0])

    return depth_output


def initialize_camera():
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)

    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    camera_data.sensor_width = 32.0
    camera_data.sensor_height = 32.0
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.clip_start = 0.1
    camera_data.clip_end = 100.0

    target = bpy.data.objects.new("CameraTarget", None)
    target.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(target)

    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    return camera


def initialize_lighting():
    key_data = bpy.data.lights.new("KeyLight", type="POINT")
    key_data.energy = 1000

    key_light = bpy.data.objects.new("KeyLight", key_data)
    key_light.location = (4, 1, 6)
    bpy.context.collection.objects.link(key_light)

    top_data = bpy.data.lights.new("TopLight", type="AREA")
    top_data.energy = 10000

    top_light = bpy.data.objects.new("TopLight", top_data)
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    bpy.context.collection.objects.link(top_light)

    bottom_data = bpy.data.lights.new("BottomLight", type="AREA")
    bottom_data.energy = 1000

    bottom_light = bpy.data.objects.new("BottomLight", bottom_data)
    bottom_light.location = (0, 0, -10)
    bpy.context.collection.objects.link(bottom_light)


def radical_inverse(base, number):
    value = 0.0
    inverse_base = 1.0 / base
    inverse_digit = inverse_base

    while number > 0:
        digit = number % base
        value += digit * inverse_digit
        number //= base
        inverse_digit *= inverse_base

    return value


def camera_angles(index, num_views, offset):
    u = index / num_views
    v = radical_inverse(2, index)

    u += offset[0] / num_views
    v += offset[1]

    if u < 0.25:
        u = 2.0 * u
    else:
        u = (2.0 / 3.0) * u + (1.0 / 3.0)

    pitch = np.arccos(1.0 - 2.0 * u) - np.pi / 2.0
    yaw = (v % 1.0) * 2.0 * np.pi

    return yaw, pitch


def set_camera_pose(camera, yaw, pitch, radius, fov_degrees):
    camera.location = (
        radius * np.cos(yaw) * np.cos(pitch),
        radius * np.sin(yaw) * np.cos(pitch),
        radius * np.sin(pitch),
    )

    fov = np.deg2rad(fov_degrees)
    camera.data.lens = (
        camera.data.sensor_width / 2.0
    ) / np.tan(fov / 2.0)

    bpy.context.view_layer.update()


def camera_intrinsics(resolution, fov_degrees):
    fov = np.deg2rad(fov_degrees)
    focal_length = 0.5 * resolution / np.tan(fov / 2.0)
    center = (resolution - 1.0) / 2.0

    return np.array(
        [
            [focal_length, 0.0, center],
            [0.0, focal_length, center],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def camera_extrinsics(camera):
    T_blender_camera_from_object = np.array(
        camera.matrix_world.inverted(),
        dtype=np.float32,
    )

    T_opencv_from_blender_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return (
        T_opencv_from_blender_camera
        @ T_blender_camera_from_object
    )


def load_depth_exr(path):
    image = bpy.data.images.load(
        str(path),
        check_existing=False,
    )

    width, height = image.size
    channels = image.channels

    pixels = np.array(
        image.pixels[:],
        dtype=np.float32,
    )

    pixels = pixels.reshape(height, width, channels)
    depth = np.flipud(pixels[:, :, 0]).copy()

    bpy.data.images.remove(image)

    return depth



def render_view(view_index, camera, K, output_dir, depth_output):
    view_id = f"{view_index:03d}"
    view_dir = output_dir / "views" / view_id
    view_dir.mkdir(parents=True, exist_ok=True)

    image_path = view_dir / "image.png"

    scene = bpy.context.scene
    scene.frame_set(view_index + 1)
    scene.render.filepath = str(image_path)

    depth_output.base_path = str(view_dir)
    depth_output.file_slots[0].path = "depth_"

    bpy.ops.render.render(write_still=True)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    depth_files = list(view_dir.glob("depth_*.exr"))

    if len(depth_files) != 1:
        raise RuntimeError(
            f"Expected one depth EXR in {view_dir}, "
            f"found {len(depth_files)}"
        )

    depth_exr_path = depth_files[0]
    # Cycles 3.0.1 already writes camera-axis Z depth.
    z_depth = load_depth_exr(depth_exr_path)
    valid = np.isfinite(z_depth) & (z_depth > 0) & (z_depth < camera.data.clip_end)
    z_depth[~valid] = np.nan

    np.save(view_dir / "depth.npy", z_depth)

    np.savez(
        view_dir / "camera.npz",
        K=K,
        T_camera_from_object=camera_extrinsics(camera),
    )

    depth_exr_path.unlink()


def main():
    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    clear_scene()

    mesh_objects = load_object(args.object)
    recalculate_normals(mesh_objects)

    T_normalized_from_source = normalize_scene()

    np.savez(
        args.output / "object_transform.npz",
        T_normalized_from_source=T_normalized_from_source,
    )

    initialize_render(args.resolution)
    depth_output = initialize_depth_output()

    camera = initialize_camera()
    initialize_lighting()

    K = camera_intrinsics(
        args.resolution,
        args.fov_degrees,
    )

    rng = np.random.RandomState(args.seed)
    offset = (rng.rand(), rng.rand())

    for view_index in range(args.num_views):
        yaw, pitch = camera_angles(
            view_index,
            args.num_views,
            offset,
        )

        set_camera_pose(
            camera,
            yaw,
            pitch,
            args.camera_radius,
            args.fov_degrees,
        )

        render_view(
            view_index,
            camera,
            K,
            args.output,
            depth_output,
        )

        print(
            f"Rendered view {view_index + 1}/{args.num_views}",
            flush=True,
        )


if __name__ == "__main__":
    main()