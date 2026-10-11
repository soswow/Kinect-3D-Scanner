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

from kinect_scanner.blender_export import convert_session, find_blender, snapshot_geometry, _snapshot, _surface_depth
from shared.sensor_calibration import load_calibration
from shared.settings import CameraCalibration, ScanSettings


def png(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def fixture(root, *, fx=531, fy=519):
    settings = ScanSettings(camera=CameraCalibration(fx=fx, fy=fy, cx=303.2, cy=231.7), filter_depth=False)
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
    def test_surface_fit_preserves_a_perspective_plane_and_reduces_quantization(self):
        from shared.calibration import raw_depth_lut, pinhole_rays
        settings = ScanSettings(sensor_calibration=load_calibration(), filter_depth=False)
        rays = pinhole_rays(settings.camera)
        plane = 1 / (.45 + .12 * rays[..., 0] - .07 * rays[..., 1])
        fitted = _surface_depth(plane)
        np.testing.assert_allclose(fitted[5:-5, 5:-5], plane[5:-5, 5:-5], atol=1e-11)
        calibration = settings.sensor_calibration
        codes = np.rint((calibration.scale / plane - calibration.b) / calibration.a_per_code).astype(int)
        quantized = np.rint(raw_depth_lut(calibration)[codes]) / 1000
        fitted = _surface_depth(quantized)
        interior = np.s_[10:-10, 10:-10]
        self.assertLess(np.mean((fitted[interior] - plane[interior]) ** 2),
                        .4 * np.mean((quantized[interior] - plane[interior]) ** 2))

    def test_surface_modes_preserve_measured_topology_holes_edges_and_inputs(self):
        from shared.calibration import prepare_metric_depth
        settings = ScanSettings(filter_depth=False)
        depth = np.full((480, 640), 1800, np.uint16)
        depth[:, 320:] = 2400
        depth[40:50, 40:50] = 0
        original = depth.copy()
        metric = prepare_metric_depth(depth, settings).astype(float) / 1000
        fitted = _surface_depth(metric)
        np.testing.assert_array_equal(fitted[metric == 0], 0)
        np.testing.assert_array_equal(fitted[:, 319:321], metric[:, 319:321])
        self.assertTrue(np.all(np.abs(fitted - metric) <= np.maximum(.01, .01 * metric)))
        for stride in (1, 4, 8):
            raw_vertices, raw_faces = snapshot_geometry(depth, settings, stride, "measured")
            vertices, faces = snapshot_geometry(depth, settings, stride, "smooth")
            np.testing.assert_array_equal(faces, raw_faces)
            np.testing.assert_allclose(vertices, raw_vertices)
        np.testing.assert_array_equal(depth, original)
        with self.assertRaisesRegex(ValueError, "surface"):
            snapshot_geometry(depth, settings, surface="unknown")

    def test_snapshot_grid_uses_depth_pixels_independently_of_fusion_voxels(self):
        depth = np.full((480, 640), 1000, np.uint16)
        fine = ScanSettings(filter_depth=False, final_voxel_m=.002)
        coarse = ScanSettings(filter_depth=False, voxel_m=.01, final_voxel_m=.01)
        for stride in (1, 2, 4, 8):
            vertices, faces = snapshot_geometry(depth, fine, stride=stride)
            height, width = 480 // stride, 640 // stride
            self.assertEqual(height * width, len(vertices))
            self.assertEqual(2 * (height - 1) * (width - 1), len(faces))
            other_vertices, other_faces = snapshot_geometry(depth, coarse, stride=stride)
            np.testing.assert_array_equal(vertices, other_vertices)
            np.testing.assert_array_equal(faces, other_faces)

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

    def test_low_space_stops_before_preparation_or_blender_and_preserves_outputs(self):
        from types import SimpleNamespace
        for low_at in (0, 1, 3, 4):
            with self.subTest(low_at=low_at), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                session, model, _ = fixture(root)
                destination = root / "scan.blend"
                destination.write_bytes(b"previous scene")
                fake = root / "blender.exe"
                fake.write_bytes(b"test")
                available = [SimpleNamespace(free=10 * 1024**3)] * low_at + [SimpleNamespace(free=249 * 1024**2)]
                with patch("kinect_scanner.blender_export.shutil.disk_usage", side_effect=available), patch(
                        "kinect_scanner.blender_export.subprocess.run", return_value=subprocess.CompletedProcess([], 1)) as run:
                    with self.assertRaisesRegex(RuntimeError, "Not enough disk space.*249 MiB"):
                        convert_session(session, destination, blender=str(fake), textured_model=model, stride=8)
                    self.assertEqual(1 if low_at == 4 else 0, run.call_count)
                self.assertEqual(b"previous scene", destination.read_bytes())
                self.assertEqual([], list(root.glob("scan.images-*")))
                self.assertEqual([], list(root.glob(".scanner-blend-*")))
                self.assertTrue(session.is_file())
                self.assertTrue(model.is_file())

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
        self.assertIn("--snapshot-surface", result.stdout)


class BlenderRoundTripTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("BLENDER_EXECUTABLE"), "Set BLENDER_EXECUTABLE for the saved-scene regression")
    def test_camera_projection_preserves_focal_ratio_when_fy_exceeds_fx(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session, model, _ = fixture(root, fx=519, fy=531)
            output = root / "scan.blend"
            run = subprocess.run
            def create(command, **kwargs):
                configuration = Path(command[-3])
                geometries = [Path(row["geometry"]) for row in json.loads(configuration.read_text())["snapshots"]]
                for path in geometries:
                    with zipfile.ZipFile(path) as archive:
                        self.assertTrue(all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist()))
                self.assertFalse((configuration.parent / "textured-model.zip").exists())
                result = run(command, **kwargs)
                self.assertTrue(all(not path.exists() for path in geometries))
                return result
            original_model = model.read_bytes()
            with patch("kinect_scanner.blender_export.subprocess.run", side_effect=create):
                convert_session(session, output, textured_model=model, stride=8, surface="measured")
            self.assertEqual(original_model, model.read_bytes())
            self.assertNotEqual(b"BLENDER", output.read_bytes()[:7])  # Compressed scene; reopened below.
            check = root / "verify.py"
            check.write_text('''import bpy, json
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view
s = bpy.context.scene
c = json.loads(s['scanner_export'])['camera']
assert json.loads(s['scanner_export'])['snapshot_surface'] == 'measured'
assert not any(p.use_smooth for o in bpy.data.collections['Depth snapshots'].objects for p in o.data.polygons)
assert abs(s.render.pixel_aspect_y/s.render.pixel_aspect_x-c['fx']/c['fy']) < 1e-6
world = Matrix(((1,0,0,0),(0,0,1,0),(0,-1,0,0),(0,0,0,1)))
for o in bpy.data.collections['Depth snapshots'].objects:
    s.frame_set(o['animation_frame'])
    observation = json.loads(o['observation'])
    point = world @ Matrix(observation['rgb_to_world']) @ Vector((.05, -.03, 1))
    p = world_to_camera_view(s, s.camera, point)
    assert abs(p.x-(.05*c['fx']+c['cx']+.5)/c['width']) < 1e-5
    assert abs(p.y-(1-(-.03*c['fy']+c['cy']+.5)/c['height'])) < 1e-5
print('FOCAL_RATIO_OK')
''', encoding="utf-8")
            process = subprocess.run([find_blender(), "--background", "--factory-startup", str(output),
                                      "--threads", "2", "--python-exit-code", "1", "--python", str(check)],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
            self.assertEqual(0, process.returncode, process.stdout)
            self.assertIn("FOCAL_RATIO_OK", process.stdout)

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
assert {v.name for v in s.collection.children} == {'Depth snapshots'}
root = bpy.data.objects['Scan root']
assert root.type == 'EMPTY' and root.parent is None
assert root.matrix_world == Matrix.Identity(4)
fused = bpy.data.objects['Fused mesh']
assert set(s.collection.objects) == {root, fused, c}
assert all(o.parent == root for o in s.objects if o != root)
assert json.loads(s['scanner_export'])['snapshot_grid'] == {'width': 320, 'height': 240}
assert json.loads(s['scanner_export'])['snapshot_surface'] == 'smooth'
assert fused.animation_data is None
assert len(fused.data.polygons) == 1
assert all(n.image.packed_file for m in fused.data.materials for n in m.node_tree.nodes if n.type == 'TEX_IMAGE')
meshes = list(bpy.data.collections['Depth snapshots'].objects)
assert len(meshes) == 2
assert all(list(o.users_collection) == [bpy.data.collections['Depth snapshots']] for o in meshes)
assert all(p.use_smooth for o in meshes for p in o.data.polygons)
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
        expected_depth = world @ Matrix(observation['camera_to_world'])
        assert max(abs(o.matrix_world[i][j]-expected_depth[i][j]) for i in range(4) for j in range(4)) < 1e-6
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
# Repositioning the root must move every snapshot and the animated camera,
# preserving calibrated projection and alignment throughout the timeline.
fused.hide_viewport = False
root.location = (1.2, -2.3, .7)
root.rotation_euler = (.2, -.3, .4)
root.scale = (1.5, 1.5, 1.5)
for frame in (2, 3, 1, 3, 2):
    s.frame_set(frame)
    assert max(abs(fused.matrix_world[i][j]-(root.matrix_world @ world)[i][j]) for i in range(4) for j in range(4)) < 1e-6
    for o in meshes:
        observation = json.loads(o['observation'])
        expected_local = world @ Matrix(observation['camera_to_world'])
        assert max(abs(o.matrix_basis[i][j]-expected_local[i][j]) for i in range(4) for j in range(4)) < 1e-6
        if frame == o['animation_frame']:
            expected = root.matrix_world @ expected_local
            assert max(abs(o.matrix_world[i][j]-expected[i][j]) for i in range(4) for j in range(4)) < 1e-6
            expected = root.matrix_world @ world @ Matrix(observation['rgb_to_world']) @ optical
            assert max(abs(c.matrix_world[i][j]-expected[i][j]) for i in range(4) for j in range(4)) < 1e-6
            point = root.matrix_world @ world @ Matrix(observation['rgb_to_world']) @ local
            projected = world_to_camera_view(s, c, point)
            assert abs(projected.x-u) < 1e-5 and abs(projected.y-v) < 1e-5
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
                "path": "scene.blend", "blender": "blender", "stride": 4, "surface": "measured", "options": {"max_triangles": 100},
            }))
            self.assertEqual(["session", "obj.zip"], [call[0] for call in downloads])
            self.assertEqual({"max_triangles": 100}, downloads[1][2])
            self.assertEqual(4, convert.call_args.kwargs["stride"])
            self.assertEqual("measured", convert.call_args.kwargs["surface"])
            self.assertEqual([(True, "scene.blend")], completed)
        self.assertFalse(Path(downloads[0][1]).exists())

    def test_blender_assembles_verified_local_and_server_images_before_conversion(self):
        import httpx
        from shared.project_archive import pixel_digest
        from shared.recording import RecordingWriter
        from kinect_scanner.server_client import ServerClient
        from kinect_scanner.server_task_worker import ServerTask, ServerTaskType, ServerTaskWorker

        rgb = np.full((3, 4, 3), [30, 90, 180], np.uint8)
        depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
        proof = {"index": 0, "rgb_sha256": pixel_digest(rgb), "depth_sha256": pixel_digest(depth)}
        for changed_during_transfer in (False, True):
            with self.subTest(changed=changed_during_transfer), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                recording = RecordingWriter(root / "local", {}, session_id="same")
                for index in range(3):
                    recording.append(rgb, depth, {"server_index": index})
                # One mismatched pair and one unavailable pair must come from the server.
                Image.fromarray(np.zeros_like(rgb)).save(recording.path / "rgb/000001.png")
                (recording.path / "depth/000002.png").unlink()
                requests = []
                def transport(request):
                    requests.append(request)
                    if request.url.path.endswith("obj.zip"):
                        return httpx.Response(200, content=b"textured model", headers={"content-type": "application/octet-stream"})
                    body = json.loads(request.content)
                    self.assertEqual("POST", request.method)
                    self.assertNotIn("rgb_format", body)  # Blender retains lossless default.
                    offered = body["reuse"]["captures"]
                    self.assertEqual([0, 1], [row["index"] for row in offered])
                    self.assertEqual(proof, offered[0])
                    self.assertNotEqual(proof["rgb_sha256"], offered[1]["rgb_sha256"])
                    manifest = {"frames": [{"rgb": f"rgb/{i:06d}.png", "depth": f"depth/{i:06d}.png"} for i in range(3)],
                                "local_captures": {"session_id": "same", "captures": [proof]}}
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w") as archive:
                        archive.writestr("manifest.json", json.dumps(manifest))
                        archive.writestr("reconstruction.json", b"final server poses")
                        for index in (1, 2):
                            archive.writestr(f"rgb/{index:06d}.png", png(rgb))
                            archive.writestr(f"depth/{index:06d}.png", png(depth))
                    if changed_during_transfer:
                        Image.fromarray(np.zeros_like(rgb)).save(recording.path / "rgb/000000.png")
                    return httpx.Response(200, content=buffer.getvalue(), headers={"content-type": "application/octet-stream"})
                client = ServerClient()
                client.session_id = "same"
                client._project_export_supported = True
                client._http = httpx.Client(base_url="http://test-server", transport=httpx.MockTransport(transport))
                worker = ServerTaskWorker(client)
                worker._recording, worker._recording_session_id = recording, "same"
                output = root / "scene.blend"
                output.write_bytes(b"previous scene")
                completed = []
                client.export_done.connect(lambda *args: completed.append(args))
                def convert(session, path, **kwargs):
                    with zipfile.ZipFile(session) as archive:
                        self.assertNotIn("local_captures", json.loads(archive.read("manifest.json")))
                        self.assertEqual(b"final server poses", archive.read("reconstruction.json"))
                        for frame in range(3):
                            for key, pixels in (("rgb", rgb), ("depth", depth)):
                                np.testing.assert_array_equal(pixels, np.array(Image.open(io.BytesIO(archive.read(f"{key}/{frame:06d}.png")))))
                    self.assertEqual(b"textured model", kwargs["textured_model"].read_bytes())
                    output.write_bytes(b"new scene")
                    return {"snapshots": 3}
                with patch("kinect_scanner.blender_export.convert_session", side_effect=convert) as conversion:
                    task = ServerTask(ServerTaskType.EXPORT_BLENDER, {"path": str(output), "blender": "blender", "stride": 1,
                                                                     "options": {"max_triangles": 100}})
                    if changed_during_transfer:
                        with self.assertRaisesRegex(ValueError, "changed during save"):
                            worker._dispatch(task)
                        conversion.assert_not_called()
                        self.assertEqual(b"previous scene", output.read_bytes())
                        self.assertEqual([], completed)
                    else:
                        worker._dispatch(task)
                        self.assertEqual(b"new scene", output.read_bytes())
                        self.assertEqual([(True, str(output))], completed)
                        self.assertEqual("GET", requests[1].method)
                        self.assertEqual("100", requests[1].url.params["max_triangles"])
                client._http.close()

    def test_blender_full_download_for_other_session_older_server_or_absent_recording(self):
        from unittest.mock import Mock
        from kinect_scanner.server_client import ServerClient
        from kinect_scanner.server_task_worker import ServerTask, ServerTaskType, ServerTaskWorker

        for recording, session, supported in ((Mock(), "old", True), (Mock(), "same", False), (None, "same", True)):
            with self.subTest(session=session, supported=supported, recording=recording):
                client = ServerClient()
                client.session_id, client._project_export_supported = "same", supported
                client.request_export = Mock(return_value=True)
                worker = ServerTaskWorker(client)
                worker._recording, worker._recording_session_id = recording, session
                with patch("shared.project_archive.recording_captures") as captures, patch("kinect_scanner.blender_export.convert_session"):
                    worker._dispatch(ServerTask(ServerTaskType.EXPORT_BLENDER, {"path": "scene.blend", "blender": "blender", "stride": 1}))
                    captures.assert_not_called()
                    for call in client.request_export.call_args_list:
                        self.assertNotIn("local_captures", call.kwargs)

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
