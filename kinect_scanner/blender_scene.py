"""Blender-only scene writer, invoked by blender_export in a fresh process.

One camera and one background-image slot, using an external relative PNG
sequence. Standard keyframes work after reopening without Python handlers.
"""

import json
from pathlib import Path
import sys


def constant_keys(block):
    """Handle both legacy actions and Blender 4.4+ slotted actions."""
    animation = block.animation_data
    if not animation or not animation.action:
        return
    action = animation.action
    if getattr(action, "is_action_layered", False):
        curves = [curve for layer in action.layers for strip in layer.strips
                  for bag in strip.channelbags for curve in bag.fcurves]
    else:
        curves = action.fcurves
    for curve in curves:
        for key in curve.keyframe_points:
            key.interpolation = "CONSTANT"


def visible_at(obj, frame):
    """A snapshot is visible at exactly one animation frame, including renders."""
    for time, hidden in ((1, True), (frame, False), (frame + 1, True)):
        obj.hide_viewport = obj.hide_render = hidden
        obj.keyframe_insert(data_path="hide_viewport", frame=time)
        obj.keyframe_insert(data_path="hide_render", frame=time)
    constant_keys(obj)


def write_scene(configuration, staged, final):
    import bpy
    import numpy as np
    from mathutils import Matrix

    if bpy.app.version < (4, 2, 0):
        raise RuntimeError("Blender 4.2 or newer is required")
    config = json.loads(Path(configuration).read_text(encoding="utf-8"))
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    # Scanner axes: right, down, forward. Blender world: right, forward, up.
    world = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
    optical = Matrix.Diagonal((1, -1, -1, 1))  # Blender camera looks along -Z.
    root = bpy.data.objects.new("Scan root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.25
    scene.collection.objects.link(root)
    bpy.ops.wm.obj_import(filepath=config["model"], forward_axis="Y", up_axis="Z")
    fused = [obj for obj in scene.objects if obj.type == "MESH"]
    if len(fused) != 1 or not fused[0].data.materials:
        raise RuntimeError("Expected one textured fused mesh")
    fused = fused[0]
    fused.name = "Fused mesh"
    fused.parent = root
    fused.matrix_world = world
    for collection in list(fused.users_collection):
        collection.objects.unlink(fused)
    scene.collection.objects.link(fused)
    fused.hide_viewport = fused.hide_render = False
    snapshots = bpy.data.collections.new("Depth snapshots")
    scene.collection.children.link(snapshots)
    data = bpy.data.cameras.new("Snapshot camera")
    camera = bpy.data.objects.new("Snapshot camera", data)
    scene.collection.objects.link(camera)
    camera.parent = root
    camera.rotation_mode = "QUATERNION"
    scene.camera = camera
    c = config["camera"]
    scene.render.resolution_x, scene.render.resolution_y = c["width"], c["height"]
    scene.render.resolution_percentage = 100
    # Blender clamps both pixel aspect components to >= 1. Express the ratio
    # on the other axis when fy > fx, rather than silently losing calibration.
    aspect = c["fx"] / c["fy"]
    scene.render.pixel_aspect_x = max(1, 1 / aspect)
    scene.render.pixel_aspect_y = max(1, aspect)
    data.sensor_fit, data.sensor_width = "HORIZONTAL", 36
    data.lens = c["fx"] * data.sensor_width / c["width"]
    data.shift_x = ((c["width"] - 1) / 2 - c["cx"]) / c["width"]
    data.shift_y = (c["cy"] - (c["height"] - 1) / 2) * aspect / c["width"]
    data.clip_start, data.clip_end = 0.01, 1000
    data.show_background_images = True
    background = data.background_images.new()
    sequence = bpy.data.images.load(str(Path(final).parent / config["images"] / "000001.png"), check_existing=False)
    sequence.source = "SEQUENCE"
    background.image = sequence
    background.frame_method = "STRETCH"
    background.display_depth = "BACK"
    background.image_user.frame_start = 2
    background.image_user.frame_duration = len(config["snapshots"])
    background.image_user.use_auto_refresh = True
    background.image_user.use_cyclic = False
    for frame, alpha in ((1, 0.0), (2, 0.5)):
        background.alpha = alpha
        background.keyframe_insert(data_path="alpha", frame=frame)
    constant_keys(data)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1
    scene.render.fps = 10
    scene.frame_start, scene.frame_end = 1, len(config["snapshots"]) + 1
    scene.timeline_markers.new("Fused mesh only", frame=1)
    for number, snapshot in enumerate(config["snapshots"], 1):
        frame = snapshot["animation_frame"]
        transform = world @ Matrix(snapshot["rgb_to_world"]) @ optical
        camera.matrix_world = transform
        if number == 1:
            camera.keyframe_insert(data_path="location", frame=1)
            camera.keyframe_insert(data_path="rotation_quaternion", frame=1)
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        scene.timeline_markers.new(f"Snapshot {snapshot['index']:06d}", frame=frame)
        with np.load(snapshot["geometry"], allow_pickle=False) as geometry:
            vertices, faces = geometry["vertices"], geometry["faces"]
            mesh = bpy.data.meshes.new(f"Depth {snapshot['index']:06d}")
            mesh.from_pydata(vertices.tolist(), [], faces.tolist())
            mesh.update()
            uv = mesh.uv_layers.new(name="RGB")
            uv.data.foreach_set("uv", geometry["uv"][faces].reshape(-1))
            observed = geometry["observed"]
        obj = bpy.data.objects.new(mesh.name, mesh)
        snapshots.objects.link(obj)
        obj.parent = root
        obj.matrix_world = world @ Matrix(snapshot["camera_to_world"])
        obj["snapshot_index"], obj["animation_frame"] = snapshot["index"], frame
        obj["observation"] = json.dumps({k: v for k, v in snapshot.items() if k not in ("geometry", "image")})
        material = bpy.data.materials.new(f"RGB {snapshot['index']:06d}")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        image = bpy.data.images.load(str(Path(final).parent / config["images"] / snapshot["image"]), check_existing=False)
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        material.node_tree.links.new(texture.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
        mesh.materials.append(material)
        fallback = bpy.data.materials.get("Unobserved RGB") or bpy.data.materials.new("Unobserved RGB")
        fallback.diffuse_color = (0.35, 0.35, 0.35, 1)
        mesh.materials.append(fallback)
        mesh.polygons.foreach_set("material_index", (~observed).astype(np.int32))
        visible_at(obj, frame)
    constant_keys(camera)
    camera.hide_viewport = camera.hide_render = False
    scene["scanner_export"] = json.dumps({k: v for k, v in config.items() if k not in ("model", "snapshots")})
    text = bpy.data.texts.new("Scanner export notes")
    text.write("Frame 1: camera visible; fused mesh only; no photo background.\n"
               "Move, rotate or scale Scan root to transform the entire scan, including the animated camera.\n"
               "Frames 2 onward: one camera pose, one sequence background and one depth snapshot.\n"
               "Fused mesh visibility is not animated. Background opacity is 0.5.\n"
               "Depth meshes use only their own prepared depth and RGB observation.\n"
               "Keep the relative .images folder beside this .blend; background sequences cannot be packed.\n"
               "Only accepted final poses are placed in the shared scene. See scanner_export for skipped indices.\n")
    grid = config["snapshot_grid"]
    text.write(f"Depth sampling: every {config['snapshot_stride']} pixel(s), {grid['width']} x {grid['height']} grid.\n"
               "Sampling controls depth mesh detail; fusion voxel size and the fused triangle budget do not.\n"
               "Missing depth or sharp depth jumps omit whole sampled cells. Finer sampling makes these holes smaller.\n")
    # Pack ordinary material images; retain just one relative external sequence.
    for image in bpy.data.images:
        if image.source == "FILE" and image.filepath:
            image.pack()
    sequence.filepath = f"//{config['images']}/000001.png"
    scene.frame_set(1)
    bpy.ops.object.select_all(action="DESELECT")
    fused.select_set(True)
    bpy.context.view_layer.objects.active = fused
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.type = "MATERIAL"
                area.spaces.active.region_3d.view_perspective = "CAMERA"
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(staged), check_existing=False, relative_remap=False)


if __name__ == "__main__":
    write_scene(*sys.argv[sys.argv.index("--") + 1:])
