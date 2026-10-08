import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d
from PIL import Image

from scripts.evaluate_session_confidence import (
    SessionReader, compact_report, evaluate, file_hash, require_distinct_paths,
    same_file, score_mesh, split_accepted, stable_depth_samples,
)
from shared.settings import ScanSettings


def plane_mesh(width=1.0, offset_x=0.0, z=1.0):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        [-width / 2 + offset_x, -0.5, z], [width / 2 + offset_x, -0.5, z],
        [width / 2 + offset_x, 0.5, z], [-width / 2 + offset_x, 0.5, z],
    ])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    return mesh


def point_samples():
    directions = np.array([[-0.3, 0, 1], [0.3, 0, 1]], np.float32)
    depth = np.array([1.01, 1.01], np.float32)
    return [{
        "index": 4, "timestamp_s": 4.0, "points": directions * depth[:, None],
        "rays": np.concatenate((np.zeros_like(directions), directions), axis=-1),
        "depth_m": depth,
        "counts": {"valid_pixels": 2, "stable_pixels": 2, "sampled_pixels": 2},
    }]


def write_session(path, depths=(1000, 1000, 1000, 1500)):
    settings = ScanSettings(filter_depth=False, min_component_triangles=0,
                            final_block_count=100, final_weight=2.0)
    manifest = {
        "version": 1, "depth_unit": "millimetres", "settings": settings.to_dict(),
        "frames": [], "reconstruction": "reconstruction.json",
    }
    report = {"pose_convention": "camera_to_world", "length_unit": "metres", "poses": []}
    rgb = np.full((480, 640, 3), 100, np.uint8)
    with zipfile.ZipFile(path, "w") as archive:
        for index, value in enumerate(depths):
            paths = {"rgb": f"rgb/{index}.png", "depth": f"depth/{index}.png"}
            depth = np.zeros((480, 640), np.uint16)
            depth[220:260, 300:340] = value
            for name, array in (("rgb", rgb), ("depth", depth)):
                data = io.BytesIO()
                Image.fromarray(array).save(data, format="PNG")
                archive.writestr(paths[name], data.getvalue())
            manifest["frames"].append({**paths, "timestamp_s": float(index)})
            report["poses"].append({"index": index, "camera_to_world": np.eye(4).tolist()})
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("reconstruction.json", json.dumps(report))


class SessionConfidenceHelperTests(unittest.TestCase):
    def test_output_alias_guards_cover_resolved_paths_symlinks_and_hardlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "session.zip"
            source.write_bytes(b"protected raw session")
            before = file_hash(source)
            symlink = directory / "symlink.json"
            aliases = [source, directory / "unused" / ".." / "session.zip"]
            try:
                symlink.symlink_to(source)
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                # Windows can disallow symlinks without Developer Mode. Still
                # exercise resolved-path and hardlink protection on that host.
            else:
                aliases.append(symlink)
            hardlink = directory / "hardlink.json"
            os.link(source, hardlink)
            for alias in aliases + [hardlink]:
                with self.subTest(alias=alias):
                    self.assertTrue(same_file(source, alias))
                    with self.assertRaisesRegex(ValueError, "aliases source ZIP"):
                        require_distinct_paths(source, detailed=alias)
                    with self.assertRaisesRegex(ValueError, "aliases source ZIP"):
                        require_distinct_paths(source, summary=alias)
                    with self.assertRaisesRegex(ValueError, "aliases source ZIP"):
                        require_distinct_paths(source, mesh=alias)
            self.assertEqual(before, file_hash(source))

    def test_detail_and_summary_must_not_alias_each_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, detail = directory / "session.zip", directory / "detail.json"
            source.write_bytes(b"protected raw session")
            detail.write_text("{}")
            alias = directory / "summary.json"
            os.link(detail, alias)
            with self.assertRaisesRegex(ValueError, "Summary report aliases detailed report"):
                require_distinct_paths(source, detail, alias)
            unrelated = directory / "separate.json"
            unrelated.write_text("{}")
            require_distinct_paths(source, detail, unrelated)
            self.assertFalse(same_file(detail, unrelated))

    def test_evaluator_refuses_mesh_hardlink_before_reading_or_fusing_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "session.zip"
            source.write_bytes(b"protected raw session")
            os.link(source, directory / "session-legacy.ply")
            with patch("scripts.evaluate_session_confidence.SessionReader",
                       side_effect=AssertionError("Frame ingestion must not run")):
                report = evaluate(source, mesh_dir=directory)
            self.assertEqual("refused", report["status"])
            self.assertIn("Mesh output aliases source ZIP", report["reason"])
            self.assertTrue(report["input_zip_unchanged"])
            self.assertEqual(b"protected raw session", source.read_bytes())

    def test_split_is_by_accepted_position_not_original_index(self):
        poses = [(index, np.eye(4)) for index in (0, 2, 7, 9, 10, 12, 20, 21, 24, 31, 33)]
        training, withheld = split_accepted(poses)
        self.assertEqual([10, 31], [index for index, _ in withheld])
        self.assertEqual([0, 2, 7, 9, 12, 20, 21, 24, 33], [index for index, _ in training])
        short_train, short_test = split_accepted(poses[:4])
        self.assertEqual([9], [index for index, _ in short_test])
        self.assertEqual(3, len(short_train))
        with self.assertRaises(ValueError):
            split_accepted(poses[:1])

    def test_stable_sample_mask_preserves_holes_steps_and_budget(self):
        camera = ScanSettings().camera
        depth = np.full((480, 640), 1000, np.uint16)
        depth[100:120, 100:120] = 0
        depth[:, 400:] = 1500
        before = depth.copy()
        directions, measured, counts = stable_depth_samples(depth, camera, 0.5, 2.0, 1, 500)
        again = stable_depth_samples(depth, camera, 0.5, 2.0, 1, 500)
        np.testing.assert_array_equal(depth, before)
        np.testing.assert_array_equal(directions, again[0])
        np.testing.assert_array_equal(measured, again[1])
        self.assertEqual(500, counts["sampled_pixels"])
        self.assertTrue(np.all(np.isin(measured, (1.0, 1.5))))
        x = np.rint(directions[:, 0] * camera.fx + camera.cx).astype(int)
        y = np.rint(directions[:, 1] * camera.fy + camera.cy).astype(int)
        self.assertFalse(np.any(np.isin(x, (399, 400))))
        self.assertFalse(np.any((x >= 99) & (x <= 120) & (y >= 99) & (y <= 120)))

    def test_reader_rejects_duplicate_archived_pose_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.zip"
            write_session(path)
            with zipfile.ZipFile(path) as archive:
                report = json.loads(archive.read("reconstruction.json"))
                contents = {name: archive.read(name) for name in archive.namelist()}
            report["poses"].append(report["poses"][0])
            contents["reconstruction.json"] = json.dumps(report).encode()
            with zipfile.ZipFile(path, "w") as archive:
                for name, data in contents.items():
                    archive.writestr(name, data)
            with zipfile.ZipFile(path) as archive:
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    SessionReader(archive)

    def test_compact_report_preserves_identity_without_full_sensor_profiles(self):
        settings = ScanSettings().to_dict()
        settings["sensor_calibration"] = {"large_profile": [1, 2, 3]}
        report = {
            "input_zip_sha256": "original", "archived_settings": settings,
            "fusion_settings": settings, "training_indices": [0, 1],
            "heldout_indices": [2], "variants": {
                "legacy": {"training_indices": [0, 1], "heldout": {
                    "surface_completeness_within_threshold": 0.5,
                    "per_view": [{"index": 2}],
                }},
            },
        }
        summary = compact_report(report)
        self.assertEqual("original", summary["input_zip_sha256"])
        self.assertEqual([0, 1], summary["training_indices"])
        self.assertEqual(0.5, summary["variants"]["legacy"]["heldout"]["surface_completeness_within_threshold"])
        self.assertNotIn("per_view", summary["variants"]["legacy"]["heldout"])
        self.assertNotIn("sensor_calibration", summary["archived_settings"])
        self.assertIn("sensor_calibration_sha256", summary["archived_settings"])
        self.assertEqual(64, len(summary["fusion_settings_sha256"]))


class SessionConfidenceGeometryTests(unittest.TestCase):
    def test_triangle_distances_do_not_depend_on_vertex_density(self):
        samples = point_samples()
        coarse = plane_mesh()
        fine = coarse.subdivide_midpoint(number_of_iterations=2)
        left = score_mesh(coarse, samples, threshold_m=0.02)
        right = score_mesh(fine, samples, threshold_m=0.02)
        self.assertAlmostEqual(0.01, left["surface_distance_rmse_m"], places=5)
        self.assertEqual(1.0, left["surface_completeness_within_threshold"])
        self.assertEqual(1.0, left["ray_completeness_within_threshold"])
        for key in ("surface_distance_rmse_m", "surface_capped_rmse_including_missing_m",
                    "ray_depth_rmse_over_hits_m", "ray_completeness_within_threshold"):
            self.assertAlmostEqual(left[key], right[key], places=6)

    def test_missing_surface_is_penalized_despite_small_error_over_hits(self):
        partial = score_mesh(plane_mesh(width=0.5, offset_x=0.25), point_samples(), threshold_m=0.02)
        self.assertAlmostEqual(0.01, partial["ray_depth_rmse_over_hits_m"], places=5)
        self.assertEqual(0.5, partial["ray_completeness_within_threshold"])
        self.assertGreater(partial["ray_capped_rmse_including_missing_m"], 0.07)
        self.assertEqual(0.5, partial["surface_completeness_within_threshold"])
        empty = score_mesh(None, point_samples(), threshold_m=0.02)
        self.assertEqual(0, empty["ray_completeness_within_threshold"])
        self.assertEqual(0, empty["surface_completeness_within_threshold"])
        self.assertAlmostEqual(0.1, empty["surface_capped_rmse_including_missing_m"])
        self.assertAlmostEqual(0.1, empty["ray_capped_rmse_including_missing_m"])
        self.assertIsNone(empty["surface_distance_rmse_m"])

    def test_session_fuses_only_training_frames_and_leaves_zip_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.zip"
            write_session(path)
            before = file_hash(path)
            seen = []
            from scripts.evaluate_session_confidence import depth_confidence
            def estimator(depth, camera):
                seen.append(int(depth.max()))
                return depth_confidence(depth, camera)
            with patch("scripts.evaluate_session_confidence.depth_confidence", estimator):
                report = evaluate(path, pixel_stride=4, samples_per_view=80)
            self.assertEqual("evaluated", report["status"], report.get("reason"))
            self.assertEqual([0, 1, 2], report["training_indices"])
            self.assertEqual([3], report["heldout_indices"])
            self.assertEqual([1000] * 3, seen)
            self.assertTrue(report["input_zip_unchanged"])
            self.assertEqual(before, file_hash(path))
            for variant in report["variants"].values():
                self.assertEqual([0, 1, 2], variant["training_indices"])
                self.assertGreater(variant["triangles"], 0)
                self.assertEqual(0, variant["heldout"]["surface_completeness_within_threshold"])
                self.assertAlmostEqual(0.1, variant["heldout"]["surface_capped_rmse_including_missing_m"])

    def test_block_budget_refuses_before_allocating_candidate_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.zip"
            write_session(path)
            with patch("scripts.evaluate_session_confidence.fuse_variant",
                       side_effect=AssertionError("Candidate fusion must not run")):
                report = evaluate(path, block_budget=1)
            self.assertEqual("refused", report["status"])
            self.assertIn("hard block budget", report["reason"])
            self.assertFalse(report["planning"]["within_budget"])
            self.assertEqual({}, report["variants"])
            self.assertTrue(report["input_zip_unchanged"])


if __name__ == "__main__":
    unittest.main()
