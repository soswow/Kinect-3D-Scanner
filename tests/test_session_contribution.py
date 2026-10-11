"""Contribution proposals and lossless project-copy safety; no live server."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.analyze_session_contribution import (
    analyze, compact_display_copies, select_redundant, storage_audit, surface_weights,
)


ROOT = Path(__file__).resolve().parents[1]


class ContributionTests(unittest.TestCase):
    def test_duplicate_observations_keep_noise_support_and_texture_photo(self):
        weights = np.ones((5, 10), np.float32)
        kept = select_redundant(weights, {0}, min_observations=3)
        self.assertTrue(kept[0])
        self.assertEqual(3, kept.sum())
        self.assertTrue(np.all((weights[kept] > 0).sum(axis=0) == 3))

    def test_small_unique_surface_and_best_incidence_survive(self):
        weights = np.array([[1, 0, 0], [.8, 1, 0], [.8, 0, 1], [.8, 0, 0]], np.float32)
        kept = select_redundant(weights, set(), min_observations=1, quality_ratio=.9)
        np.testing.assert_array_equal([True, True, True, False], kept)

    def test_simultaneous_removals_do_not_erase_shared_coverage(self):
        weights = np.array([[1, 0], [1, 1], [0, 1]], np.float32)
        kept = select_redundant(weights, set(), min_observations=1)
        self.assertEqual(1, kept.sum())
        np.testing.assert_array_equal([True, True], (weights[kept] > 0).any(axis=0))

    def test_depth_occlusion_and_camera_transform_are_respected(self):
        settings = SimpleNamespace(camera=SimpleNamespace(width=5, height=5, fx=2, fy=2, cx=2, cy=2),
                                   near_m=.2, far_m=3)
        pose = np.eye(4)
        pose[:3, 3] = [10, 0, 0]
        points = np.array([[10, 0, 1], [10, 0, 2], [12, 0, 1], [10, 0, -1]])
        normals = np.tile([0, 0, 1], (4, 1))
        weights = surface_weights(points, normals, np.ones((5, 5)), pose, settings)
        self.assertGreater(weights[0], 0)
        np.testing.assert_array_equal(0, weights[1:])

    def test_help_works_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/analyze_session_contribution.py"), "--help"],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--compact-output", result.stdout)


class ProjectCopyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "original.zip"
        self.manifest = {"version": 3, "frames": [
            {"rgb": "rgb/0.png", "depth": "depth/0.png", "display_rgb": "display/rgb/0.png",
             "metadata": {"orientation": {"rotation_cw_degrees": 90}}},
            {"rgb": "rgb/1.png", "depth": "depth/1.png"},
        ], "reconstruction": "reconstruction.json", "mesh": "mesh.ply"}
        self.report = {"poses": [{"index": 0}], "frames": [{"index": 0, "success": True}, {"index": 1, "success": False}]}
        with zipfile.ZipFile(self.source, "w") as archive:
            archive.writestr("manifest.json", json.dumps(self.manifest))
            archive.writestr("reconstruction.json", json.dumps(self.report))
            for name in ("rgb/0.png", "depth/0.png", "rgb/1.png", "depth/1.png", "display/rgb/0.png", "mesh.ply", "sensors/history.jsonl"):
                archive.writestr(name, name.encode())

    def test_compact_project_preserves_every_capture_and_other_member(self):
        original_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        target = self.root / "compact.zip"
        result = compact_display_copies(self.source, target)
        self.assertEqual(1, result["removed_display_members"])
        with zipfile.ZipFile(self.source) as source, zipfile.ZipFile(target) as compact:
            m = json.loads(compact.read("manifest.json"))
            self.assertEqual(2, len(m["frames"]))
            self.assertNotIn("display_rgb", m["frames"][0])
            self.assertEqual(self.manifest["frames"][0]["metadata"], m["frames"][0]["metadata"])
            for name in compact.namelist():
                if name != "manifest.json":
                    self.assertEqual(source.read(name), compact.read(name), name)
        self.assertEqual(original_hash, hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_existing_target_and_input_hardlink_are_refused(self):
        target = self.root / "compact.zip"
        target.write_bytes(b"existing project")
        with self.assertRaises(ValueError):
            compact_display_copies(self.source, target)
        self.assertEqual(b"existing project", target.read_bytes())
        alias = self.root / "alias.zip"
        alias.hardlink_to(self.source)
        with self.assertRaises(ValueError):
            compact_display_copies(self.source, alias)

    def test_failed_publication_leaves_no_partial_project(self):
        target = self.root / "compact.zip"
        before = set(self.root.iterdir())
        with patch("scripts.analyze_session_contribution.os.link", side_effect=OSError("no link support")):
            with self.assertRaises(OSError):
                compact_display_copies(self.source, target)
        self.assertEqual(before, set(self.root.iterdir()))

    def test_shared_native_display_member_is_preserved(self):
        self.manifest["frames"][0]["display_rgb"] = "rgb/1.png"
        with zipfile.ZipFile(self.source, "r") as archive:
            members = {n: archive.read(n) for n in archive.namelist()}
        members["manifest.json"] = json.dumps(self.manifest).encode()
        with zipfile.ZipFile(self.source, "w") as archive:
            for name, value in members.items():
                archive.writestr(name, value)
        target = self.root / "compact.zip"
        compact_display_copies(self.source, target)
        with zipfile.ZipFile(target) as archive:
            self.assertEqual(b"rgb/1.png", archive.read("rgb/1.png"))

    def test_storage_audit_counts_payload_and_unused_shots(self):
        with zipfile.ZipFile(self.source) as archive:
            rows, sizes = storage_audit(archive, self.manifest, {0})
        self.assertTrue(rows[0]["in_final_poses"])
        self.assertFalse(rows[1]["in_final_poses"])
        self.assertEqual(len(b"rgb/1.png") + len(b"depth/1.png"), sizes["outside_final_pose_image_payload_bytes"])

    def test_storage_only_requires_no_mesh_or_numerical_analysis(self):
        summary = analyze(self.source, self.root / "audit", storage_only=True)
        self.assertEqual(2, summary["captures"])
        self.assertEqual([1], summary["outside_final_pose_indices"])
        self.assertFalse(summary["surface_analysis"]["executed"])


if __name__ == "__main__":
    unittest.main()
