"""Recorded proposal helpers preserve archived pose and timing provenance."""

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from scripts.benchmark_bundle_adjustment import file_checksum, recording, score, validate_output
from shared.settings import ScanSettings


class BundleBenchmarkTests(unittest.TestCase):
    def test_report_destination_protects_archive_and_external_pose_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "input.zip"
            poses = Path(folder) / "poses.json"
            archive.write_bytes(b"original archive")
            poses.write_bytes(b"original poses")
            alias = Path(folder) / "alias.json"
            os.link(archive, alias)
            for destination in (archive, poses, alias):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    validate_output(destination, archive, poses)
            validate_output(Path(folder) / "report.json", archive, poses)
            self.assertEqual(b"original archive", archive.read_bytes())
            self.assertEqual(b"original poses", poses.read_bytes())

    def archive(self, *, indices=(0, 1, 2), delta=31):
        manifest = {"settings": ScanSettings().to_dict(), "reconstruction": "saved/reconstruction.json",
                    "frames": [{"rgb": f"rgb/{i}.png", "depth": f"depth/{i}.png",
                                "metadata": {"rgb_depth_delta_ms": delta, "frame_id": f"raw-{i}"}}
                               for i in range(3)]}
        reconstruction = {"pose_convention": "camera_to_world", "length_unit": "metres",
                          "session_id": "archived-test", "poses": [
                              {"index": i, "camera_to_world": np.eye(4).tolist()} for i in indices]}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("saved/reconstruction.json", json.dumps(reconstruction))
            for entry in manifest["frames"]:
                for member, array in [(entry["rgb"], np.zeros((4, 4, 3), np.uint8)),
                                      (entry["depth"], np.full((4, 4), 1200, np.uint16))]:
                    image = io.BytesIO()
                    Image.fromarray(array).save(image, format="PNG")
                    archive.writestr(member, image.getvalue())
        stream.seek(0)
        return zipfile.ZipFile(stream), manifest, reconstruction

    def test_archived_poses_keep_original_indices_images_and_timing(self):
        archive, manifest, _ = self.archive(indices=(0, 2))
        with archive:
            engine = recording(archive, archived_poses=True)
            self.assertEqual([0, 2], [i for i, _ in engine.poses])
            self.assertEqual([entry["metadata"] for entry in manifest["frames"]], engine.frame_metadata)
            self.assertEqual("embedded_archived_camera_estimates", engine.pose_provenance["kind"])
            self.assertEqual("saved/reconstruction.json", engine.pose_provenance["member"])
            self.assertEqual([0, 2], engine.pose_provenance["accepted_indices"])
            self.assertEqual((4, 4, 3), engine.raw_frames[2][0].shape)
            np.testing.assert_array_equal(engine.raw_frames[2][1], np.full((4, 4), 1200, np.uint16))

    def test_original_timing_gate_refuses_without_modifying_input(self):
        archive, manifest, _ = self.archive()
        with archive:
            engine = recording(archive, archived_poses=True)
            result = score(engine, None)
            self.assertFalse(result["proposal_accepted"])
            self.assertIn("not synchronized", result["proposal"]["reason"])
            self.assertEqual([entry["metadata"] for entry in manifest["frames"]], engine.frame_metadata)
            self.assertEqual(engine.pose_provenance, result["pose_provenance"])

    def test_external_poses_remain_supported_and_sources_cannot_be_mixed(self):
        archive, _, reconstruction = self.archive(indices=(0, 2))
        with archive, tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "poses.json"
            path.write_text(json.dumps(reconstruction))
            engine = recording(archive, path)
            self.assertEqual("external_connected_camera_estimates", engine.pose_provenance["kind"])
            self.assertEqual(file_checksum(path), engine.pose_provenance["reconstruction_sha256"])
            with self.assertRaises(ValueError):
                recording(archive, path, archived_poses=True)
            reconstruction.pop("pose_convention")
            reconstruction.pop("length_unit")
            path.write_text(json.dumps(reconstruction))
            legacy = recording(archive, path)
            self.assertEqual([0, 2], legacy.pose_provenance["accepted_indices"])
            self.assertEqual({"pose_convention": False, "length_unit": False},
                             legacy.pose_provenance["coordinate_labels_present"])

    def test_invalid_archived_indices_are_rejected(self):
        for indices in ((0, 0, 2), (0, 1, 5)):
            with self.subTest(indices=indices):
                archive, _, _ = self.archive(indices=indices)
                with archive, self.assertRaises(ValueError):
                    recording(archive, archived_poses=True)
