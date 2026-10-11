"""Single-observation geometry and saved Blender animation round trips."""

import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from kinect_scanner.blender_export import convert_session, find_blender, snapshot_geometry, _snapshot
from shared.sensor_calibration import load_calibration
from shared.settings import CameraCalibration, ScanSettings


def png(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def fixture(root):
    settings = ScanSettings(camera=CameraCalibration(fx=531, fy=519, cx=303.2, cy=231.7), filter_depth=False)
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[:, :320, 0], rgb[:, 320:, 2] = 255, 255
    depth = np.zeros((480, 640), np.uint16)
    depth[220:231, 300:311] = 1000
    poses = []
    for index in (0, 2):
        pose = np.eye(4)
        pose[0, 3] = index * .05
        angle = index * .12
        pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        poses.append({"index": index, "camera_to_world": pose.tolist()})
    report = {"session_id": "test-session", "pose_convention": "camera_to_world", "length_unit": "metres", "poses": poses}
    manifest = {"settings": settings.to_dict(), "depth_unit": "millimetres", "reconstruction": "reconstruction.json", "mesh": "mesh.ply",
                "frames": [{"rgb": f"rgb/{i}.png", "depth": f"depth/{i}.png", "timestamp_s": i * .1,
                            "metadata": {"server_index": i}} for i in range(3)]}
    session, model = root / "session.zip", root / "model.zip"
    with zipfile.ZipFile(session, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("reconstruction.json", json.dumps(report))
        archive.writestr("mesh.ply", "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n-0.1 -0.1 1 255 0 0\n0.1 -0.1 1 255 0 0\n0 0.1 1 255 0 0\n3 0 2 1\n")
        for index in range(3):
            archive.writestr(f"rgb/{index}.png", png(rgb if index != 2 else rgb[:, ::-1]))
            archive.writestr(f"depth/{index}.png", png(depth))
    with zipfile.ZipFile(model, "w") as archive:
        archive.writestr("scan.obj", "mtllib scan.mtl\no scan\nv -0.1 -0.1 1\nv 0.1 -0.1 1\nv 0 0.1 1\nvt 0 0\nvt 1 0\nvt 0.5 1\nusemtl RGB\nf 1/1 3/3 2/2\n")
        archive.writestr("scan.mtl", "newmtl RGB\nKd 1 1 1\nmap_Kd texture.png\n")
        archive.writestr("texture.png", png(rgb))
    return session, model, report


class BlenderPreparationTests(unittest.TestCase):
    def test_sampling_does_not_bridge_missing_pixels_or_depth_discontinuities(self):
        settings = ScanSettings(filter_depth=False)
        depth = np.zeros((480, 640), np.uint16)
        depth[100:105, 100:105] = 1000
        _, complete = snapshot_geometry(depth, settings, stride=2)
        depth[101, 101] = 0  # This pixel is not itself sampled.
        vertices, hole = snapshot_geometry(depth, settings, stride=2)
        self.assertEqual(8, len(complete))
        self.assertEqual(6, len(hole))
        self.assertTrue(np.isfinite(vertices).all())
        depth[101, 101] = 2000
        _, jump = snapshot_geometry(depth, settings, stride=2)
        self.assertEqual(len(hole), len(jump))

    def test_native_rgb_camera_uses_fixed_offset_and_rectified_image_projection(self):
        settings = ScanSettings(sensor_calibration=load_calibration(), filter_depth=False)
        rgb = np.zeros((1024, 1280, 3), np.uint8)
        raw = np.full((480, 640), 750, np.uint16)
        image, vertices, faces, uv, observed, extrinsic = _snapshot(rgb, raw, settings, 8)
        self.assertEqual(rgb.shape, image.shape)
        np.testing.assert_allclose(extrinsic[:3, :3], settings.sensor_calibration.rotation)
        np.testing.assert_allclose(extrinsic[:3, 3], np.array(settings.sensor_calibration.translation_mm) / 1000)
        points = vertices @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        c = settings.rgb_camera
        expected = np.column_stack(((points[:, 0] / points[:, 2] * c.fx + c.cx + .5) / c.width,
                                    1 - (points[:, 1] / points[:, 2] * c.fy + c.cy + .5) / c.height))
        np.testing.assert_allclose(expected, uv)
        self.assertTrue(observed.any())
        self.assertGreater(len(faces), 100)

    def test_failed_blender_preserves_destination_and_removes_new_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session, model, _ = fixture(root)
            destination = root / "scan.blend"
            destination.write_bytes(b"existing scene")
            fake = root / "blender.exe"
            fake.write_bytes(b"test")
            with patch("kinect_scanner.blender_export.subprocess.run", return_value=subprocess.CompletedProcess([], 1)):
                with self.assertRaisesRegex(RuntimeError, "could not create"):
                    convert_session(session, destination, blender=str(fake), textured_model=model, stride=8)
            self.assertEqual(b"existing scene", destination.read_bytes())
            self.assertEqual([], list(root.glob("scan.images-*")))

    def test_selected_blender_path_never_silently_uses_another_installation(self):
        with self.assertRaises(ValueError):
            find_blender("missing-blender-installation.exe")

    def test_frozen_client_uses_its_bundled_plain_blender_script(self):
        from kinect_scanner.blender_export import _scene_script
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "kinect_scanner/blender_scene.py"
            script.parent.mkdir()
            script.write_text("# bundled Blender entrypoint", encoding="utf-8")
            with patch.object(os.sys, "_MEIPASS", str(root), create=True):
                self.assertEqual(script, _scene_script())

    def test_help_needs_no_optional_dependencies(self):
        result = subprocess.run([os.sys.executable, "-S", "scripts/export_blender_session.py", "--help"],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--snapshot-stride", result.stdout)


class BlenderRoundTripTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("BLENDER_EXECUTABLE"), "Set BLENDER_EXECUTABLE for the saved-scene regression")
    def test_standalone_converter_bakes_saved_mesh_without_a_server_or_refusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session, _, _ = fixture(root)
            result = convert_session(session, root / "offline.blend", stride=8,
                                     texture_options={"size": 256, "max_triangles": 100})
            self.assertEqual(2, result["snapshots"])
            self.assertTrue((root / "offline.blend").is_file())

    @unittest.skipUnless(os.environ.get("BLENDER_EXECUTABLE"), "Set BLENDER_EXECUTABLE for the saved-scene regression")
    def test_reopened_scene_has_one_camera_one_background_and_matching_animation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session, model, report = fixture(root)
            output = root / "scan.blend"
            result = convert_session(session, output, textured_model=model, stride=2)
            self.assertEqual({"snapshots": 2, "unregistered": 1}, {k: result[k] for k in ("snapshots", "unregistered")})
            # Move both artifacts to prove all background dependencies are relative.
            moved = root / "moved"
            moved.mkdir()
            output = output.replace(moved / output.name)
            Path(result["images"]).rename(moved / Path(result["images"]).name)
            check = root / "verify.py"
            check.write_text('''import bpy, json, os
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view
s = bpy.context.scene
c = s.camera
assert len([o for o in s.objects if o.type == 'CAMERA']) == 1
assert len(c.data.background_images) == 1
b = c.data.background_images[0]
assert b.image.source == 'SEQUENCE'
assert len(s.collection.children) == 3
assert {v.name for v in s.collection.children} == {'Fused mesh', 'Depth snapshots', 'Camera'}
fused = bpy.data.objects['Fused mesh']
assert fused.animation_data is None
assert len(fused.data.polygons) == 1
assert all(n.image.packed_file for m in fused.data.materials for n in m.node_tree.nodes if n.type == 'TEX_IMAGE')
meshes = list(bpy.data.collections['Depth snapshots'].objects)
assert len(meshes) == 2
assert len(bpy.app.handlers.frame_change_post) == 0
assert not any(t.use_module for t in bpy.data.texts)
world = Matrix(((1,0,0,0),(0,0,1,0),(0,-1,0,0),(0,0,0,1)))
optical = Matrix.Diagonal((1,-1,-1,1))
for frame in (1, 2, 3, 1, 3, 2):
    s.frame_set(frame)
    assert not c.hide_viewport and not c.hide_render
    assert not fused.hide_viewport and not fused.hide_render
    visible = [o for o in meshes if not o.hide_viewport]
    assert len(visible) == (0 if frame == 1 else 1), (frame, visible)
    assert b.alpha == (0 if frame == 1 else .5), (frame, b.alpha)
    if frame > 1:
        o = visible[0]
        assert not o.hide_render
        observation = json.loads(o['observation'])
        assert o['animation_frame'] == frame
        expected = world @ Matrix(observation['rgb_to_world']) @ optical
        assert max(abs(c.matrix_world[i][j]-expected[i][j]) for i in range(4) for j in range(4)) < 1e-6
        calibration = json.loads(s['scanner_export'])['camera']
        # Pixel-center projection must agree with the exported calibrated camera.
        local = Vector((.05, -.03, 1))
        point = world @ Matrix(observation['rgb_to_world']) @ local
        projected = world_to_camera_view(s, c, point)
        u = (.05 * calibration['fx'] + calibration['cx'] + .5) / calibration['width']
        v = 1 - (-.03 * calibration['fy'] + calibration['cy'] + .5) / calibration['height']
        assert abs(projected.x-u) < 1e-5 and abs(projected.y-v) < 1e-5, (projected, u, v)
        path = b.image.filepath_from_user(image_user=b.image_user)
        assert path.endswith('%06d.png' % (frame-1)), (frame, path, b.image_user.frame_current)
        assert os.path.isfile(bpy.path.abspath(path)), path
fused.hide_viewport = True
s.frame_set(3)
assert fused.hide_viewport
s.frame_set(2)
pose = c.matrix_world.copy()
s.frame_set(2, subframe=.5)
assert c.matrix_world == pose
print('REOPENED_BLENDER_SCENE_OK')
''', encoding="utf-8")
            process = subprocess.run([find_blender(), "--background", "--factory-startup", str(output),
                                      "--threads", "2", "--python-exit-code", "1", "--python", str(check)],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
            self.assertEqual(0, process.returncode, process.stdout)
            self.assertIn("REOPENED_BLENDER_SCENE_OK", process.stdout)


class BlenderTaskTests(unittest.TestCase):
    def test_worker_downloads_session_and_textures_then_reports_only_completed_scene(self):
        from kinect_scanner.server_client import ServerClient
        from kinect_scanner.server_task_worker import ServerTask, ServerTaskType, ServerTaskWorker

        client = ServerClient()
        worker = ServerTaskWorker(client)
        downloads, completed = [], []
        def download(fmt, path, options=None):
            downloads.append((fmt, path, options))
            Path(path).write_bytes(b"fixture")
            return True
        client.request_export = download
        client.export_done.connect(lambda *args: completed.append(args))
        with patch("kinect_scanner.blender_export.convert_session", return_value={"snapshots": 2}) as convert:
            worker._dispatch(ServerTask(ServerTaskType.EXPORT_BLENDER, {
                "path": "scene.blend", "blender": "blender", "stride": 4, "options": {"max_triangles": 100},
            }))
            self.assertEqual(["session", "obj.zip"], [call[0] for call in downloads])
            self.assertEqual({"max_triangles": 100}, downloads[1][2])
            self.assertEqual(4, convert.call_args.kwargs["stride"])
            self.assertEqual([(True, "scene.blend")], completed)
        self.assertFalse(Path(downloads[0][1]).exists())

    def test_failed_download_does_not_launch_blender_or_emit_success(self):
        from kinect_scanner.server_client import ServerClient
        from kinect_scanner.server_task_worker import ServerTask, ServerTaskType, ServerTaskWorker

        client = ServerClient()
        client.request_export = lambda *args, **kwargs: False
        worker = ServerTaskWorker(client)
        completed = []
        client.export_done.connect(lambda *args: completed.append(args))
        with patch("kinect_scanner.blender_export.convert_session") as convert:
            with self.assertRaisesRegex(RuntimeError, "Could not download session"):
                worker._dispatch(ServerTask(ServerTaskType.EXPORT_BLENDER, {"path": "scene.blend"}))
            convert.assert_not_called()
        self.assertEqual([], completed)


if __name__ == "__main__":
    unittest.main()
