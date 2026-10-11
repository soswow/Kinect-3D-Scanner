"""Pending-only recovery provenance survives opening, continuation and saving."""

import asyncio
import copy
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scanner_server.engine import ScanEngine
from scanner_server.session import export_session, load_session
from shared.settings import ScanSettings


class RecoveryProjectSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "recovery.zip"
        with patch.dict(os.environ, {"KINECT_NATIVE": "off"}), patch.object(ScanEngine, "BLOCK_COUNT", 100):
            self.engine = ScanEngine(device="cpu")
        self.engine.BLOCK_COUNT = 100
        self.engine.reset(settings=ScanSettings())
        self.addCleanup(self.engine.shutdown)
        self.rgb = np.zeros((480, 640, 3), np.uint8)
        self.depth = np.full((480, 640), 1000, np.uint16)
        self.engine.store_frame(self.rgb, self.depth, {"frame_id": "retained"})
        self.provenance = {"version": 1, "partial": True, "scope": "pending captures only",
                           "source_session_id": "old", "source_stream_id": None, "capture_count": 1,
                           "ordering": "durable sequence"}
        self.engine.project_recovery = copy.deepcopy(self.provenance)
        export_session(self.engine, self.path)

    def test_reopen_continue_resave_and_reset_preserve_historical_scope(self):
        self.engine.reset(settings=ScanSettings())
        self.assertIsNone(self.engine.project_recovery)
        load_session(self.engine, self.path)
        self.assertEqual(self.provenance, self.engine.project_recovery)
        self.assertEqual(self.provenance, self.engine.reconstruction_report()["project_recovery"])
        self.engine.store_frame(self.rgb, self.depth, {"frame_id": "continued"})
        destination = self.path.with_name("continued.zip")
        export_session(self.engine, destination)
        with zipfile.ZipFile(destination) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(2, len(manifest["frames"]))
        self.assertEqual(self.provenance, manifest["recovery"])
        load_session(self.engine, destination)
        self.assertEqual(self.provenance, self.engine.project_recovery)
        self.engine.reset(settings=ScanSettings())
        self.assertIsNone(self.engine.project_recovery)

    def test_malformed_marker_rejected_before_candidate_reset(self):
        with zipfile.ZipFile(self.path) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        for change in ({"version": True}, {"version": 2}, {"partial": False}, {"scope": "all captures"},
                       {"capture_count": True}, {"capture_count": -1}, {"source_session_id": 3},
                       {"source_stream_id": "x" * 129}):
            manifest = json.loads(members["manifest.json"])
            manifest["recovery"].update(change)
            invalid = self.path.with_name("invalid.zip")
            with zipfile.ZipFile(invalid, "w") as archive:
                for name, data in members.items():
                    archive.writestr(name, json.dumps(manifest) if name == "manifest.json" else data)
            previous = self.engine.session_id
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_session(self.engine, invalid)
            self.assertEqual(previous, self.engine.session_id)
            self.assertEqual(self.provenance, self.engine.project_recovery)

    def test_compact_copy_preserves_recovery_origin_without_relabeling_coverage(self):
        import open3d as o3d
        from scanner_server.session import compact_project_info
        self.engine.store_frame(self.rgb, self.depth, {"frame_id": "unused"})
        self.engine.poses = [(0, np.eye(4))]
        self.engine.diagnostics = [{"index": 0, "success": True}, {"index": 1, "success": False}]
        self.engine._processed_count = 2
        self.engine.frame_count = 1
        self.engine.mesh = o3d.geometry.TriangleMesh.create_box(.1, .1, .1)
        self.engine.project_recovery["capture_count"] = 5  # Historical origin, not current retained count.
        plan = compact_project_info(self.engine)
        path = self.path.with_name("compact.zip")
        export_session(self.engine, path, compact=True, compact_revision=plan["revision"])
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            report = json.loads(archive.read("reconstruction.json"))
        self.assertEqual(1, len(manifest["frames"]))
        self.assertEqual(5, manifest["recovery"]["capture_count"])
        self.assertEqual(manifest["recovery"], report["project_recovery"])

    def test_status_exposes_partial_provenance(self):
        from scanner_server import app as server
        original = server.engine
        try:
            server.engine = self.engine
            result = asyncio.run(server.scan_status())
            self.assertEqual(self.provenance, result["project_recovery"])
        finally:
            server.engine = original


if __name__ == "__main__":
    unittest.main()
