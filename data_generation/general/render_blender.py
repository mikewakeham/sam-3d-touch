import argparse
import math
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-views", type=int, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--camera-radius", type=float, default=2.7)
    parser.add_argument("--fov-degrees", type=float, default=40.0)

    parser.add_argument("--device", choices=["CUDA", "OPTIX", "CPU"], default="CUDA")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--camera-mode", choices=["trellis2", "fixed"], default="trellis2")
    parser.add_argument("--lighting", choices=["trellis2", "fixed"], default="trellis2")
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--rotation", type=float, nargs=3, default=[0, 0, 0],
                        help="XYZ Euler rotation in degrees after Blender import")

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
    # Import operators follow TRELLIS and zeroverse/2gltf2; 4.x renamed OBJ/PLY/STL.
    extension = object_path.suffix.lower()
    path = str(object_path)
    if extension == ".blend":
        with bpy.data.libraries.load(path, link=False) as (source, target):
            target.objects = source.objects
        for obj in target.objects:
            if obj is not None:
                bpy.context.scene.collection.objects.link(obj)
    elif extension in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif extension == ".obj":
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.obj_import(filepath=path, forward_axis="NEGATIVE_Z", up_axis="Y")
        else:
            bpy.ops.import_scene.obj(filepath=path, axis_forward="-Z", axis_up="Y")
    elif extension == ".ply":
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.ply_import(filepath=path)
        else:
            bpy.ops.import_mesh.ply(filepath=path)
    elif extension == ".stl":
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.stl_import(filepath=path)
        else:
            bpy.ops.import_mesh.stl(filepath=path)
    elif extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif extension in {".usd", ".usda", ".usdc", ".usdz"}:
        bpy.ops.wm.usd_import(filepath=path)
    elif extension == ".dae":
        bpy.ops.wm.collada_import(filepath=path)
    elif extension == ".abc":
        bpy.ops.wm.alembic_import(filepath=path)
    else:
        raise ValueError(f"Unsupported object format: {extension}")


def freeze_geometry(frame, rotation):
    # Bake evaluated geometry once. Renders, surface samples and voxels then use
    # precisely the same triangles, including object instances and modifiers.
    from mathutils import Euler
    bpy.context.scene.frame_set(frame)
    for obj in bpy.context.scene.objects:
        for modifier in obj.modifiers:
            modifier.show_viewport = modifier.show_render
            if modifier.type == "SUBSURF":
                modifier.levels = modifier.render_levels
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    original_objects = list(bpy.context.scene.objects)
    meshes = []
    rotation_matrix = Euler(np.deg2rad(rotation), "XYZ").to_matrix().to_4x4()
    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"} or obj.hide_render:
            continue
        if not instance.show_self:
            continue
        mesh = bpy.data.meshes.new_from_object(obj, depsgraph=depsgraph)
        if not mesh.polygons:
            bpy.data.meshes.remove(mesh)
            continue
        mesh.transform(rotation_matrix @ instance.matrix_world)
        mesh.calc_loop_triangles()
        # Triangulate the baked mesh used for both rendering and export.
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        meshes.append(mesh)
    for obj in original_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    if not meshes:
        raise ValueError("No renderable mesh faces found")
    objects = []
    for index, mesh in enumerate(meshes):
        obj = bpy.data.objects.new(f"Mesh_{index}", mesh)
        bpy.context.scene.collection.objects.link(obj)
        objects.append(obj)
    return objects


def normalize_scene(mesh_objects, output_dir):
    vertices = np.concatenate([
        np.array([vertex.co[:] for vertex in obj.data.vertices], dtype=np.float64)
        for obj in mesh_objects
    ])
    bounds_min, bounds_max = vertices.min(axis=0), vertices.max(axis=0)
    extent = (bounds_max - bounds_min).max()
    if not np.isfinite(vertices).all() or not np.isfinite(extent) or extent <= 0:
        raise ValueError("Object must have finite vertices and nonzero extent")
    scale = 1.0 / extent
    offset = -(bounds_min + bounds_max) * (0.5 * scale)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = offset
    from mathutils import Matrix
    for obj in mesh_objects:
        obj.data.transform(Matrix(transform))
        obj.data.update()
    # This transform starts in the imported, evaluated Blender world frame.
    # Import-specific axes and hierarchy are already baked into mesh.npz.
    np.savez(output_dir / "object_transform.npz",
             T_normalized_from_source=transform.astype(np.float32),
             source_coordinate_frame="evaluated_blender_world_after_rotation",
             coordinate_frame="normalized_object")
    vertices, faces = [], []
    vertex_offset = 0
    for obj in mesh_objects:
        vertices.append(np.array([v.co[:] for v in obj.data.vertices], dtype=np.float32))
        faces.append(np.array([f.vertices[:] for f in obj.data.polygons], dtype=np.int32) + vertex_offset)
        vertex_offset += len(obj.data.vertices)
    np.savez_compressed(output_dir / "mesh.npz", vertices=np.concatenate(vertices),
                        faces=np.concatenate(faces), coordinate_frame="normalized_object")



def initialize_render(resolution, device="CUDA", samples=32):
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_persistent_data = True

    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_animated_seed = False
    scene.cycles.filter_type = "BOX"
    scene.cycles.filter_width = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.use_denoising = True

    # Blender 3.x/TRELLIS used Filmic; explicitly keep it under Blender 4.x too.
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.use_motion_blur = False
    scene.render.use_border = False
    if device == "CPU":
        scene.cycles.device = "CPU"
        return
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = device
    preferences.get_devices()
    for item in preferences.devices:
        item.use = item.type == device
    if not any(item.type == device for item in preferences.devices):
        raise RuntimeError(f"No {device} rendering device found")
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
    world = bpy.data.worlds.new("DatasetWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.05, 0.05, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world
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
    bottom_light.rotation_euler.x = math.pi
    bpy.context.collection.objects.link(bottom_light)


def init_random_lighting(camera_dir, rng) -> None:
    # Adapted from https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/blender_script/render_cond.py
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create environment light
    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world = bpy.context.scene.world

    # Enabling nodes
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # Remove default nodes
    for node in nodes:
        nodes.remove(node)

    # Random place lights
    num_lights = rng.randint(1, 4)
    total_strength = 1.5
    for i in range(num_lights):
        new_light = bpy.data.objects.new(f"Light_{i}", bpy.data.lights.new(f"Light_{i}", type="POINT"))
        bpy.context.collection.objects.link(new_light)

        new_light_distance = 1 / rng.uniform(1/100, 1/10)
        new_light_dir = rng.randn(3)
        new_light_dir[2] += 0.6
        new_light_dir = new_light_dir / np.linalg.norm(new_light_dir)
        new_light_location = new_light_dir * new_light_distance
        new_light_camera_strength_ratio = max(np.sum(camera_dir * new_light_dir) * 0.5 + 0.5, 0)
        new_light_max_energy = total_strength / (np.sum(camera_dir * new_light_dir) * 0.45 + 0.55)
        new_light_strength = np.sqrt(rng.uniform(0.01, 1)) * new_light_max_energy
        new_light_camera_strength = new_light_camera_strength_ratio * new_light_strength
        total_strength -= new_light_camera_strength

        new_light.location = (new_light_location[0], new_light_location[1], new_light_location[2])
        new_light.data.color = (1.0, 1.0, 1.0)
        new_light.data.energy = new_light_strength * new_light_distance**2 * 31.4
        new_light.data.shadow_soft_size = rng.uniform(0.1, 0.1 * new_light_distance)

    # Create background node
    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = total_strength
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


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
    scene.render.filepath = str(image_path)

    depth_output.base_path = str(view_dir)
    depth_output.file_slots[0].path = "depth_"

    for stale in view_dir.glob("depth_*.exr"):
        stale.unlink()
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
    # Cycles camera_z_depth writes camera-axis Z, not Euclidean ray length.
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

    bpy.context.scene.render.engine = "CYCLES"
    clear_scene()

    load_object(args.object)
    assets = []
    for image in bpy.data.images:
        if image.source == "FILE" and not image.packed_file and image.filepath:
            path = Path(bpy.path.abspath(image.filepath, library=image.library)).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing texture: {path}")
            assets.append({"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
    mesh_objects = freeze_geometry(args.frame, args.rotation)
    normalize_scene(mesh_objects, args.output)
    initialize_render(args.resolution, args.device, args.samples)
    bpy.context.scene.cycles.seed = args.seed
    depth_output = initialize_depth_output()

    camera = initialize_camera()
    initialize_lighting()

    with (args.output / "render_metadata.json").open("w") as file:
        json.dump({"blender_version": bpy.app.version_string, "source": str(args.object),
                   "frame": args.frame, "rotation_degrees": args.rotation,
                   "camera_mode": args.camera_mode, "lighting": args.lighting,
                   "samples": args.samples, "persistent_data": True,
                   "external_assets": assets,
                   "depth_convention": "camera_axis_z", "color_transform": "Filmic",
                   "pointmap_convention": "x_left_y_up_z_forward"}, file, indent=2)

    rng = np.random.RandomState(args.seed)
    offset = (rng.rand(), rng.rand())
    # TRELLIS.2 data_toolkit/render_cond.py samples inverse squared distance.
    # Use the same schedule for every object; add 5% framing margin.
    camera_rng = np.random.RandomState(args.seed)
    light_rng = np.random.RandomState(args.seed)
    radius_min = np.sqrt(3) / 2 / np.sin(np.deg2rad(70 / 2))
    radius_max = np.sqrt(3) / 2 / np.sin(np.deg2rad(10 / 2))
    radii = 1 / np.sqrt(camera_rng.uniform(1 / radius_max**2, 1 / radius_min**2, args.num_views))
    fovs = np.rad2deg(2 * np.arcsin(np.sqrt(3) / 2 / radii))

    for view_index in range(args.num_views):
        yaw, pitch = camera_angles(
            view_index,
            args.num_views,
            offset,
        )

        radius = radii[view_index] * 1.05 if args.camera_mode == "trellis2" else args.camera_radius
        fov = fovs[view_index] if args.camera_mode == "trellis2" else args.fov_degrees
        K = camera_intrinsics(args.resolution, fov)
        if args.lighting == "trellis2":
            camera_dir = np.array([np.cos(yaw) * np.cos(pitch),
                                   np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
            init_random_lighting(camera_dir, light_rng)

        set_camera_pose(
            camera,
            yaw,
            pitch,
            radius,
            fov,
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
