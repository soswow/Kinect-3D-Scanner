"""Portable projects restore observations, geometry and tracking transactionally."""

import os
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d

from scanner_server.engine import ScanEngine
from scanner_server.session import export_session, load_session
from shared.settings import ScanSettings


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "project.zip"
        self.source = ScanEngine(device="cpu")
        self.addCleanup(self.source.shutdown)
        self.source.reset(settings=ScanSettings(voxel_m=.01, final_weight=.5))
        rgb = np.full((480, 640, 3), [50, 100, 150], np.uint8)
        depth = np.full((480, 640), 1000, np.uint16)
        for i in range(2):
            self.source.store_frame(rgb, depth, {"frame_id": f"capture-{i}", "timestamp_s": i,
                                               "custom": {"retained": True}})
        self.source.poses = [(0, np.eye(4))]
        self.source.diagnostics = [{"index": 0, "success": True, "pose": np.eye(4).tolist()}]
        self.source._processed_count = self.source.frame_count = 1
        self.source.mesh = o3d.geometry.TriangleMesh.create_box(.1, .2, .3)
        self.source.mesh.paint_uniform_color([.1, .4, .7])
        export_session(self.source, self.path)

    def candidate(self):
        candidate = ScanEngine(device="cpu")
        self.addCleanup(candidate.shutdown)
        return candidate

    def rewrite(self, modify):
        with zipfile.ZipFile(self.path) as source:
            members = {i.filename: source.read(i) for i in source.infolist()}
        modify(members)
        with zipfile.ZipFile(self.path, "w") as target:
            for name, data in members.items():
                target.writestr(name, data)

    def test_roundtrip_keeps_native_pixels_settings_poses_and_final_mesh(self):
        candidate = self.candidate()
        manifest = load_session(candidate, self.path)
        self.assertEqual(3, manifest["version"])
        self.assertEqual(self.source.settings, candidate.settings)
        self.assertEqual(2, candidate.stored_count)
        self.assertEqual(1, candidate.unprocessed_count)
        self.assertEqual(1, candidate.frame_count)
        for original, restored in zip(self.source.raw_frames, candidate.raw_frames):
            for a, b in zip(original, restored):
                np.testing.assert_array_equal(a, b)
        self.assertEqual([{**m, "server_index": i} for i, m in enumerate(self.source.frame_metadata)],
                         candidate.frame_metadata)
        np.testing.assert_array_equal(self.source.poses[0][1], candidate.poses[0][1])
        np.testing.assert_allclose(self.source.mesh.vertices, candidate.mesh.vertices)
        np.testing.assert_allclose(self.source.mesh.vertex_colors, candidate.mesh.vertex_colors, atol=1 / 255)
        self.assertGreater(candidate.vbg.hashmap().size(), 0)
        candidate.process_frames()
        self.assertEqual(0, candidate.unprocessed_count)
        self.assertEqual(2, len(candidate.diagnostics))

    def test_old_session_opens_without_mesh_and_preserves_sensor_members_on_resave(self):
        def legacy(members):
            manifest = json.loads(members["manifest.json"])
            manifest.update(version=2, sensor_archive={"segments": [], "complete": True})
            manifest.pop("mesh")
            members.pop("mesh.ply")
            members["manifest.json"] = json.dumps(manifest)
            members["sensors/test/accelerometer.jsonl"] = b'{"valid":false}\n'
        self.rewrite(legacy)
        candidate = self.candidate()
        load_session(candidate, self.path)
        self.assertIsNone(candidate.mesh)
        candidate._project_archive_path = str(self.path)
        saved = self.path.with_name("saved.zip")
        export_session(candidate, saved)
        with zipfile.ZipFile(saved) as archive:
            self.assertEqual(b'{"valid":false}\n', archive.read("sensors/test/accelerometer.jsonl"))
            self.assertIn("sensor_archive", json.loads(archive.read("manifest.json")))
            with zipfile.ZipFile(self.path) as original:
                self.assertEqual(original.read("rgb/000000.png"), archive.read("rgb/000000.png"))
        with self.assertRaises(ValueError):
            export_session(candidate, self.path)

    def test_invalid_poses_and_member_paths_are_rejected(self):
        for field in ("pose", "path"):
            with self.subTest(field=field):
                export_session(self.source, self.path)
                def damage(members):
                    if field == "pose":
                        report = json.loads(members["reconstruction.json"])
                        report["poses"][0]["camera_to_world"][0][0] = 2
                        members["reconstruction.json"] = json.dumps(report)
                    else:
                        manifest = json.loads(members["manifest.json"])
                        manifest["frames"][0]["rgb"] = "../outside.png"
                        members["manifest.json"] = json.dumps(manifest)
                self.rewrite(damage)
                with self.assertRaises(ValueError):
                    load_session(self.candidate(), self.path)


if __name__ == "__main__":
    unittest.main()
