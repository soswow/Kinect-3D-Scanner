"""Portable projects restore observations, geometry and tracking transactionally."""

import os

os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from scanner_server.engine import ScanEngine
from scanner_server.session import export_session, load_session
from shared.project_archive import (
    complete_local_captures,
    pixel_digest,
    recording_captures,
)
from shared.recording import RecordingWriter
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

    def test_compact_display_copy_opens_and_resaves_with_original_observations(self):
        from scripts.analyze_session_contribution import compact_display_copies

        self.source.frame_metadata[0]["orientation"] = {"rotation_cw_degrees": 90}
        export_session(self.source, self.path)
        compact = self.path.with_name("compact.zip")
        compact_display_copies(self.path, compact)
        candidate = self.candidate()
        manifest = load_session(candidate, compact)
        self.assertNotIn("display_rgb", manifest["frames"][0])
        self.assertNotIn("display_depth", manifest["frames"][0])
        self.assertEqual(90, candidate.frame_metadata[0]["orientation"]["rotation_cw_degrees"])
        self.assertEqual(self.source.settings, candidate.settings)
        for original, restored in zip(self.source.raw_frames, candidate.raw_frames):
            for a, b in zip(original, restored):
                np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(self.source.poses[0][1], candidate.poses[0][1])
        np.testing.assert_allclose(self.source.mesh.vertices, candidate.mesh.vertices)
        candidate._project_archive_path = str(compact)
        saved = self.path.with_name("resaved.zip")
        export_session(candidate, saved)
        with zipfile.ZipFile(compact) as source, zipfile.ZipFile(saved) as target:
            for frame in manifest["frames"]:
                for key in ("rgb", "depth"):
                    self.assertEqual(source.read(frame[key]), target.read(frame[key]))

    def finished_with_skips(self):
        rgb, depth = self.source.raw_frames[1]
        rgb = rgb.copy()
        rgb[:, :, 0] = 180
        self.source.store_frame(rgb, depth, {"frame_id": "capture-2", "timestamp_s": 2,
                                             "orientation": {"rotation_cw_degrees": 90}})
        moved = np.eye(4)
        moved[0, 3] = .02
        self.source.poses = [(0, np.eye(4)), (2, moved)]
        self.source.diagnostics = [{"index": i, "success": i != 1, "metadata": self.source.frame_metadata[i]}
                                   for i in range(3)]
        self.source.frame_count, self.source._processed_count = 2, 3
        self.source.original_poses = [(0, np.eye(4)), (2, moved.copy())]
        self.source.tracking_edges = [{"source_index": 2, "target_index": 0}, {"source_index": 1, "target_index": 0}]
        self.source.mesh = o3d.geometry.TriangleMesh.create_box(.1, .2, .3)
        self.source.mesh.paint_uniform_color([.1, .4, .7])

    def test_integrated_compaction_remaps_only_final_views_and_preserves_texture(self):
        from scanner_server.session import compact_project_info
        from scanner_server.texturing import _views

        self.finished_with_skips()
        plan = compact_project_info(self.source)
        self.assertTrue(plan["available"])
        self.assertEqual([0, 2], plan["retained_indices"])
        self.assertEqual(1, plan["excluded_count"])
        expected_views = [i for i, _ in _views(self.source, 64)]
        export_session(self.source, self.path, compact=True, compact_revision=plan["revision"])
        candidate = self.candidate()
        manifest = load_session(candidate, self.path)
        self.assertEqual(2, candidate.stored_count)
        self.assertEqual([0, 1], [i for i, _ in candidate.poses])
        self.assertEqual(plan["retained_indices"], manifest["project_compaction"]["retained_source_indices"])
        self.assertEqual(2, candidate.frame_metadata[1]["source_server_index"])
        self.assertEqual(1, candidate.frame_metadata[1]["server_index"])
        self.assertEqual(expected_views, [plan["retained_indices"][i] for i, _ in _views(candidate, 64)])
        for (old, original_pose), (new, restored_pose) in zip(self.source.poses, candidate.poses):
            np.testing.assert_array_equal(original_pose, restored_pose)
            for a, b in zip(self.source.raw_frames[old], candidate.raw_frames[new]):
                np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(self.source.mesh.vertices, candidate.mesh.vertices)
        self.assertEqual(3, self.source.stored_count)
        self.assertEqual([], candidate.tracking_edges)
        self.assertFalse(candidate.fragment_reconnection["applied"])
        self.assertEqual(2, candidate._preserved_project_pose_count)
        self.assertIsNone(candidate._reconnection_count)
        with zipfile.ZipFile(self.path) as archive:
            self.assertFalse(any(name.startswith("display/") for name in archive.namelist()))
            history = json.loads(archive.read("provenance/source-reconstruction.json"))
            self.assertEqual([0, 2], [p["index"] for p in history["poses"]])
            self.assertEqual(self.source.tracking_edges, history["tracking_edges"])
        candidate._project_archive_path = str(self.path)
        saved = self.path.with_name("resaved.zip")
        export_session(candidate, saved)
        with zipfile.ZipFile(saved) as archive:
            self.assertEqual(manifest["project_compaction"], json.loads(archive.read("manifest.json"))["project_compaction"])
            self.assertEqual(history, json.loads(archive.read("provenance/source-reconstruction.json")))
        twice = self.path.with_name("compact-twice.zip")
        second_plan = compact_project_info(candidate)
        export_session(candidate, twice, compact=True, compact_revision=second_plan["revision"])
        with zipfile.ZipFile(twice) as archive:
            self.assertEqual(history, json.loads(archive.read("provenance/previous/source-reconstruction.json")))
            self.assertEqual(manifest["project_compaction"], json.loads(archive.read("provenance/previous/compaction.json")))

    def test_compact_review_requires_finished_scan_and_rejects_stale_save(self):
        from scanner_server.session import compact_project_info

        self.assertFalse(compact_project_info(self.source)["available"])
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "unavailable or stale"):
            export_session(self.source, self.path, compact=True, compact_revision="unreviewed")
        self.assertEqual(before, self.path.read_bytes())
        self.finished_with_skips()
        plan = compact_project_info(self.source)
        self.source.poses[-1][1][0, 3] += .01
        with self.assertRaisesRegex(ValueError, "unavailable or stale"):
            export_session(self.source, self.path, compact=True, compact_revision=plan["revision"])
        self.assertEqual(before, self.path.read_bytes())

    def test_compact_depth_project_preserves_saved_poses_until_a_new_capture(self):
        from dataclasses import replace
        from unittest.mock import patch
        from scanner_server.session import compact_project_info

        self.finished_with_skips()
        self.source.settings = replace(self.source.settings, offline_registration="depth")
        plan = compact_project_info(self.source)
        export_session(self.source, self.path, compact=True, compact_revision=plan["revision"])
        candidate = self.candidate()
        load_session(candidate, self.path)
        with patch("scanner_server.depth_graph.propose_depth_poses", side_effect=AssertionError("Saved poses must stay fixed")):
            self.assertTrue(candidate._reconnect_volume())
        self.assertFalse(candidate.fragment_reconnection["applied"])
        self.assertIsNone(candidate._reconnection_count)
        rgb, depth = candidate.raw_frames[-1]
        self.assertTrue(candidate.store_frame(rgb, depth, {"frame_id": "new-capture"})["success"])
        self.assertIsNone(candidate._preserved_project_pose_count)
        with patch("scanner_server.depth_graph.propose_depth_poses", return_value=(None, {"applied": False, "reason": "test"})) as register:
            self.assertTrue(candidate._reconnect_volume())
        register.assert_called_once()

    def test_compact_opened_project_copies_original_members_by_source_index(self):
        from scanner_server.session import compact_project_info

        self.finished_with_skips()
        export_session(self.source, self.path, rgb_format="jpeg")
        candidate = self.candidate()
        load_session(candidate, self.path)
        candidate._project_archive_path = str(self.path)
        compact = self.path.with_name("compact-jpeg.zip")
        plan = compact_project_info(candidate)
        export_session(candidate, compact, rgb_format="jpeg", compact=True, compact_revision=plan["revision"])
        with zipfile.ZipFile(self.path) as original, zipfile.ZipFile(compact) as saved:
            self.assertEqual(original.read("rgb/000002.jpg"), saved.read("rgb/000001.jpg"))
            self.assertEqual(original.read("depth/000002.png"), saved.read("depth/000001.png"))
            self.assertFalse(any(n.startswith("display/") for n in saved.namelist()))

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

    def test_default_compresses_mesh_and_metadata_without_outer_image_compression(self):
        with zipfile.ZipFile(self.path) as archive:
            for name in ("mesh.ply", "manifest.json", "reconstruction.json"):
                self.assertEqual(zipfile.ZIP_DEFLATED, archive.getinfo(name).compress_type)
            for name in ("rgb/000000.png", "depth/000000.png"):
                self.assertEqual(zipfile.ZIP_STORED, archive.getinfo(name).compress_type)
            self.assertNotIn(b'\n  ', archive.read("reconstruction.json"))

    def test_jpeg_roundtrip_keeps_depth_and_mesh_and_does_not_recompress_on_resave(self):
        rgb = np.random.default_rng(31).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        depth = self.source.raw_frames[0][1]
        self.source.raw_frames[0] = (rgb, depth)
        self.source.frame_metadata[0]["orientation"] = {"rotation_cw_degrees": 90}
        export_session(self.source, self.path, rgb_format="jpeg")
        with zipfile.ZipFile(self.path) as archive:
            original = archive.read("rgb/000000.jpg")
            self.assertEqual(b'\xff\xd8', original[:2])
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(95, manifest["jpeg_quality"])
            self.assertEqual(4, manifest["version"])
            self.assertEqual("JPEG", Image.open(io.BytesIO(archive.read("display/rgb/000000.jpg"))).format)
        candidate = self.candidate()
        load_session(candidate, self.path)
        np.testing.assert_array_equal(depth, candidate.raw_frames[0][1])
        self.assertFalse(np.array_equal(rgb, candidate.raw_frames[0][0]))
        np.testing.assert_array_equal(rgb, self.source.raw_frames[0][0])
        np.testing.assert_allclose(self.source.mesh.vertices, candidate.mesh.vertices)
        candidate._project_archive_path = str(self.path)
        saved = self.path.with_name("resaved.zip")
        export_session(candidate, saved, rgb_format="jpeg")
        with zipfile.ZipFile(saved) as archive:
            self.assertEqual(original, archive.read("rgb/000000.jpg"))

    def test_local_capture_reuse_is_verified_and_falls_back_for_mismatched_pairs(self):
        recording = RecordingWriter(self.path.with_name("local"), self.source.settings.to_dict())
        for i, ((rgb, depth), metadata) in enumerate(zip(self.source.raw_frames, self.source.frame_metadata)):
            recording.append(rgb, depth, {**metadata, "server_index": i})
        local = recording_captures(recording)
        proofs = [{"index": i, **{key: row[key] for key in ("rgb_sha256", "depth_sha256")}}
                  for i, row in local.items()]
        proofs[1]["rgb_sha256"] = "0" * 64
        for rgb_format in ("png", "jpeg"):
            export_session(self.source, self.path, rgb_format=rgb_format,
                           reuse={"session_id": self.source.session_id, "captures": proofs})
            with zipfile.ZipFile(self.path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual([0], [p["index"] for p in manifest["local_captures"]["captures"]])
                self.assertNotIn(manifest["frames"][0]["rgb"], archive.namelist())
                self.assertIn(manifest["frames"][1]["rgb"], archive.namelist())
            with self.assertRaisesRegex(ValueError, "Incomplete project"):
                load_session(self.candidate(), self.path)
            self.assertEqual(1, complete_local_captures(self.path, local, self.source.session_id))
            candidate = self.candidate()
            manifest = load_session(candidate, self.path)
            self.assertNotIn("local_captures", manifest)
            np.testing.assert_array_equal(self.source.raw_frames[0][1], candidate.raw_frames[0][1])
            if rgb_format == "png":
                np.testing.assert_array_equal(self.source.raw_frames[0][0], candidate.raw_frames[0][0])

    def test_reuse_rejects_other_sessions_and_changed_local_pixels_transactionally(self):
        with self.assertRaisesRegex(ValueError, "current server session"):
            export_session(self.source, self.path, reuse={"session_id": "stale", "captures": []})
        rgb, depth = self.source.raw_frames[0]
        proof = {"index": 0, "rgb_sha256": pixel_digest(rgb), "depth_sha256": pixel_digest(depth)}
        export_session(self.source, self.path, reuse={"session_id": self.source.session_id, "captures": [proof]})
        previous = self.path.read_bytes()
        local = {0: {"rgb": self.path.with_name("rgb.png"), "depth": self.path.with_name("depth.png")}}
        Image.fromarray(np.zeros_like(rgb)).save(local[0]["rgb"])
        Image.fromarray(depth).save(local[0]["depth"])
        with self.assertRaisesRegex(ValueError, "changed during save"):
            complete_local_captures(self.path, local, self.source.session_id)
        self.assertEqual(previous, self.path.read_bytes())
        self.assertFalse(self.path.with_name(self.path.name + ".captures.tmp").exists())

    def test_recordings_survive_client_restart_and_missing_images_use_server(self):
        root = self.path.with_name("recordings")
        recording = RecordingWriter(root / "scan-current", self.source.settings.to_dict(), session_id=self.source.session_id)
        rgb, depth = self.source.raw_frames[0]
        recording.append(rgb, depth, {"server_index": 0})
        recovered = RecordingWriter.for_session(root, self.source.session_id)
        self.assertEqual(recording.path, recovered.path)
        self.assertEqual([0], list(recording_captures(recovered)))
        self.assertIsNone(RecordingWriter.for_session(root, "another-session"))
        (recording.path / "depth/000000.png").unlink()
        self.assertEqual({}, recording_captures(recovered))
        legacy = RecordingWriter(root / "scan-legacy", self.source.settings.to_dict())
        legacy.save_reconstruction({"session_id": "legacy-session"})
        self.assertEqual(legacy.path, RecordingWriter.for_session(root, "legacy-session").path)


if __name__ == "__main__":
    unittest.main()
