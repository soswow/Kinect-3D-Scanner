"""Stdlib-only current field fixture provenance contracts; no native imports."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/research/gpu_icp_experiment_capture.py"
spec = importlib.util.spec_from_file_location("field_pair_capture_contract", PATH)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CaptureContract(unittest.TestCase):
    def valid(self):
        return {"source_sha256": capture.CURRENT, "input_sha256": "a" * 64,
            "session": "chest-6.zip", "source_changed_during_profile": False,
            "input_changed_during_profile": False, "pose_seeds_used": False,
            "finish_requested": True, "mesh_built": True, "experimental_visual_fallback": False,
            "native_mode": "on", "omp_threads": "8", "frames": 3, "selected_indices": [0, 1, 2],
            "thread_policy": {"opencv_threads": 20, "open3d_threads": 20},
            "versions": {"open3d": "0.20.0"},
            "native_extension": {"path": "original.pyd", "sha256": "b" * 64, "changed_during_profile": False},
            "fragment_reconnection": {"fragments": [{"id": 0}]}}

    def check(self, report):
        return capture.profile_contract(report, Path("chest-6.zip"), capture.CURRENT, "a" * 64)

    def test_valid_identity_preserved(self):
        p = self.valid()
        before = copy.deepcopy(p)
        self.assertIs(self.check(p), p["native_extension"])
        self.assertEqual(p, before)

    def test_source_transition_refused(self):
        p = self.valid(); p["source_sha256"] = "9" * 64
        with self.assertRaises(ValueError): self.check(p)

    def test_raw_identity_refused(self):
        p = self.valid(); p["input_sha256"] = "c" * 64
        with self.assertRaises(ValueError): self.check(p)

    def test_archived_pose_seed_refused(self):
        p = self.valid(); p["pose_seeds_used"] = True
        with self.assertRaises(ValueError): self.check(p)

    def test_failed_or_partial_finish_refused(self):
        for key in ("finish_requested", "mesh_built"):
            p = self.valid(); p[key] = False
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(p)

    def test_input_source_mutation_refused(self):
        for key in ("source_changed_during_profile", "input_changed_during_profile"):
            p = self.valid(); p[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(p)

    def test_selected_views_are_all_original(self):
        for indices in ([0, 2], [0, 1, 1], [False, 1, 2]):
            p = self.valid(); p["selected_indices"] = indices
            with self.subTest(indices=indices), self.assertRaises(ValueError): self.check(p)

    def test_thread_confound_refused(self):
        for key in ("opencv_threads", "open3d_threads"):
            p = self.valid(); p["thread_policy"][key] = 8
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(p)

    def test_native_mutation_refused(self):
        p = self.valid(); p["native_extension"]["changed_during_profile"] = True
        with self.assertRaises(ValueError): self.check(p)

    def test_versions_refused(self):
        p = self.valid(); p["versions"]["open3d"] = "0.19.0"
        with self.assertRaises(ValueError): self.check(p)

    def test_no_fragment_inputs_refused(self):
        p = self.valid(); p["fragment_reconnection"]["fragments"] = []
        with self.assertRaises(ValueError): self.check(p)

    def test_same_basename_required(self):
        p = self.valid(); p["session"] = "other.zip"
        with self.assertRaises(ValueError): self.check(p)

    def test_original_column_major_seed_values_are_not_rejected(self):
        # Model the genuine Eigen/global-proposal column-major representation
        # without importing numerical libraries during the source contract test.
        dtype = SimpleNamespace(str="<f8")
        payload = bytes(range(128))
        original = SimpleNamespace(dtype=dtype, shape=(4, 4), nbytes=128,
            strides=(8, 32), flags=SimpleNamespace(c_contiguous=False))
        original.tobytes = lambda order: payload if order == "C" else self.fail("Wrong value order")
        fake_np = SimpleNamespace(asarray=lambda array: array, dtype=lambda name: dtype,
            isfinite=lambda array: SimpleNamespace(all=lambda: True))
        with patch.dict(sys.modules, {"numpy": fake_np}):
            row = capture.descriptor(original)
        self.assertEqual(row["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(row["original_strides"], [8, 32])
        self.assertFalse(row["original_c_contiguous"])
        self.assertEqual(original.strides, (8, 32))


if __name__ == "__main__":
    unittest.main()
