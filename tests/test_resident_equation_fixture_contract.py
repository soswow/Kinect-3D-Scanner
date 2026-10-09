"""Stdlib-only equation manifest/bits/failure guard contracts, no native math."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts.research import validate_resident_equation_fixture as guard
from scripts.research.benchmark_device_ldlt import difference_evidence


def hash_bits(value):
    return hashlib.sha256(value).hexdigest()


def npy(bits, dtype, shape):
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape}).encode("ascii")
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + bits


class EquationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=guard.ROOT / "benchmark-output", prefix="stdlib-equations-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.solver = self.root / "original.dll"
        self.solver.write_bytes(b"artificial original solve; not a native library")
        self.synthetic, self.bridge = self.root / "synthetic.json", self.root / "bridge.json"
        self.synthetic.write_text('{"artificial":true}', encoding="utf-8")
        self.bridge.write_text('{"artificial":true}', encoding="utf-8")
        self.payload = self.root / "equations.npz"
        self.bits = {"matrices": struct.pack("<72d", *[float(i % 7 == 0) for i in range(72)]),
            "gradients": struct.pack("<12d", *([0.] * 12)),
            "cpu_update": struct.pack("<32d", *[float(i % 5 == 0) for i in range(32)]),
            "cpu_status": struct.pack("<2i", 0, 0)}
        self.layouts = {"matrices": ("<f8", (2, 6, 6)), "gradients": ("<f8", (2, 6)),
                        "cpu_update": ("<f8", (2, 4, 4)), "cpu_status": ("<i4", (2,))}
        self.write_payload()
        fixture = {"tasks": [{"position": 0, "pair": [8, 12], "proposal_sha256": [hash_bits(str(i).encode()) for i in range(5)]},
                              {"position": 4, "pair": [0, 2], "proposal_sha256": [hash_bits(str(i + 5).encode()) for i in range(4)]}]}
        self.fixture_hash = guard.canonical_hash(fixture)
        binding = {"resident_math": {"solve_library_sha256": guard.sha(self.solver)},
                   "artifacts_sha256": {"scripts/tool_paths.py": guard.sha(guard.ROOT / "scripts/tool_paths.py")}}
        proposals = []
        expected = [(task["pair"], index, fingerprint) for task in fixture["tasks"]
                    for index, fingerprint in enumerate(task["proposal_sha256"])]
        for mode in ("native_cpu", "grid"):
            for sequence, (pair, index, fingerprint) in enumerate(expected):
                first = 0 if mode == "native_cpu" or sequence == 0 else 2
                end = 0 if mode == "native_cpu" else 2
                proposals.append({"mode": mode, "sequence": sequence, "pair": pair,
                    "proposal_index": index, "input_sha256": fingerprint, "completed": True,
                    "first_record": first, "end_record": end})
        matches = [{"match_index": i, "source": {"points_sha256": hash_bits(b"source")},
            "target": {"points_sha256": hash_bits(b"target")}, "initial_sha256": hash_bits(b"initial"),
            "completed": True, "first_record": i, "end_record": i + 1} for i in range(2)]
        context0 = {key: proposals[9][key] for key in ("mode", "sequence", "pair", "proposal_index", "input_sha256")}
        rows = []
        for i in range(2):
            rows.append({"record_index": i, "status": 0, "exception": None,
                "matrix_sha256": hash_bits(self.bits["matrices"][i * 288:(i + 1) * 288]),
                "gradient_sha256": hash_bits(self.bits["gradients"][i * 48:(i + 1) * 48]),
                "update_sha256": hash_bits(self.bits["cpu_update"][i * 128:(i + 1) * 128]),
                "context": [copy.deepcopy(context0), {key: matches[i][key] for key in
                    ("match_index", "source", "target", "initial_sha256")}]})
        pairs = [{"position": task["position"], "pair": task["pair"], "pair_verdict_same": True,
            "proposal_results": [{"proposal_index": i, "input_sha256": fingerprint, "quality": {"passed": True}}
                for i, fingerprint in enumerate(task["proposal_sha256"])]} for task in fixture["tasks"]]
        grid = {"mode": "grid", "complete": True, "pairs": copy.deepcopy(pairs),
                "resident_statistics_delta": {"pose_iterations": 2, "calls": 2},
                "statistics_delta": {"query_rows": 20, "device_calls": 2}}
        component = {"status": "passed", "cleanup_passed": True, "fixture_arrays_unchanged": True,
            "cloud_arrays_unchanged": True, "fixture_binding": fixture, "proof_bindings": binding,
            "proof_bindings_after": binding, "solve_metadata": {"artificial": True},
            "real_runs": [{"mode": "native_cpu", "complete": True, "pairs": copy.deepcopy(pairs)}, grid]}
        self.component = self.root / "component.json"
        self.component.write_text(json.dumps(component), encoding="utf-8")
        artifacts = {name: guard.sha(guard.ROOT / name) for name in guard.ARTIFACTS}
        fixed = {str(path): guard.sha(path) for path in (self.solver, self.synthetic, self.bridge)}
        self.value = {"kind": guard.KIND, "status": "passed", "performance_attribution_valid": False,
            "new_solve_authority": False, "previous_pose_is_actual": False, "closure_failures": [],
            "observer_hooks_restored": True, "hook_cleanup_passed": True,
            "hook_cleanup_failures": [],
            "source_sha256": guard.FROZEN, "source_sha256_after": guard.FROZEN,
            "component_source_sha256": hash_bits(b"component"), "component_source_sha256_after": hash_bits(b"component"),
            "artifacts_sha256": artifacts, "artifacts_sha256_after": copy.deepcopy(artifacts),
            "fixed_files_sha256": fixed, "fixed_files_sha256_after": copy.deepcopy(fixed),
            "component_proofs": {"synthetic": {"path": str(self.synthetic), "sha256": guard.sha(self.synthetic)},
                                 "bridge": {"path": str(self.bridge), "sha256": guard.sha(self.bridge)}},
            "proof_synthetic_sha256": guard.sha(self.synthetic), "proof_bridge_sha256": guard.sha(self.bridge),
            "proof_bindings": binding, "fixture_binding": fixture, "fixture_binding_sha256": self.fixture_hash,
            "actual_runtime_binding": binding, "actual_runtime_binding_after": copy.deepcopy(binding),
            "solve_metadata": {"artificial": True}, "component_report": {"path": str(self.component), "sha256": guard.sha(self.component)},
            "recorder": {"count": 2, "latched_failure": None, "delegate_instances": 1, "rows": rows},
            "actual_component_counts": {"actual_solve_calls": 2, "resident_pose_iterations": 2,
                "resident_calls": 2, "query_rows": 20, "nn_device_calls": 2},
            "proposal_scopes": proposals, "match_scopes": matches,
            "equation_artifact": {"path": str(self.payload), "sha256": guard.sha(self.payload),
                "bytes": self.payload.stat().st_size, "arrays": {name: {"dtype": "float64" if dtype == "<f8" else "int32",
                    "shape": list(shape)} for name, (dtype, shape) in self.layouts.items()}}}
        self.manifest = self.root / "capture.json"
        self.patch_proof = patch.object(guard, "validate_grid_proof", return_value=SimpleNamespace(
            fixture_binding_sha256=self.fixture_hash, artifact_sha256=()))
        self.patch_resident = patch.object(guard, "resident_checks", return_value={})
        self.patch_core = patch.object(guard, "current_core_hash", return_value=guard.FROZEN)
        for value in (self.patch_proof, self.patch_resident, self.patch_core):
            value.start()
            self.addCleanup(value.stop)

    def write_payload(self):
        with zipfile.ZipFile(self.payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, (dtype, shape) in self.layouts.items():
                archive.writestr(name + ".npy", npy(self.bits[name], dtype, shape))

    def validate(self):
        self.manifest.write_text(json.dumps(self.value, allow_nan=False), encoding="utf-8")
        return guard.validate_equation_fixture(self.manifest, expected_solver_path=self.solver)

    def test_closed_original_row_and_proposal_membership(self):
        token = self.validate()
        self.assertEqual(token.count, 2)
        token.check_unchanged()
        self.assertEqual(token.manifest, self.value)

    def test_complete_windows_component_artifact_keys_are_normalized(self):
        # Physical Windows captures retain the complete inherited component
        # mapping with backslashes. resident_checks expects normalized keys.
        component_artifacts = {"scripts\\tool_paths.py": guard.sha(guard.ROOT / "scripts/tool_paths.py"),
            "scripts\\tool-catalog.json": guard.sha(guard.ROOT / "scripts/tool-catalog.json")}
        for key in ("proof_bindings", "actual_runtime_binding", "actual_runtime_binding_after"):
            self.value[key]["artifacts_sha256"] = copy.deepcopy(component_artifacts)
        component = json.loads(self.component.read_text(encoding="utf-8"))
        for key in ("proof_bindings", "proof_bindings_after"):
            component[key]["artifacts_sha256"] = copy.deepcopy(component_artifacts)
        self.component.write_text(json.dumps(component), encoding="utf-8")
        self.value["component_report"]["sha256"] = guard.sha(self.component)
        with patch.object(guard, "resident_checks", return_value={}) as resident_check:
            self.validate()
        self.assertEqual(resident_check.call_args.args[2], {
            key.replace("\\", "/"): value for key, value in component_artifacts.items()})
        self.assertEqual(resident_check.call_args.args[1]["artifacts_sha256"], component_artifacts)

    def test_changed_original_dependency_after_validation_is_rejected(self):
        token = self.validate()
        self.solver.write_bytes(b"changed original")
        with self.assertRaisesRegex(guard.EquationFixtureError, "dependency changed"):
            token.check_unchanged()

    def test_failed_observer_and_unclosed_cleanup_are_rejected(self):
        for field, value in (("status", "failed"), ("observer_hooks_restored", False),
                             ("hook_cleanup_passed", False), ("previous_pose_is_actual", True),
                             ("hook_cleanup_failures", [{"message": "fault"}]),
                             ("closure_failures", [{"message": "fault"}])):
            with self.subTest(field=field):
                previous = self.value[field]
                self.value[field] = value
                with self.assertRaises(guard.EquationFixtureError):
                    self.validate()
                self.value[field] = previous

    def test_component_fault_cannot_be_relabelled_as_passed(self):
        component = json.loads(self.component.read_text())
        component["timing_driver_current_failure"] = {"message": "primary failure"}
        self.component.write_text(json.dumps(component), encoding="utf-8")
        self.value["component_report"]["sha256"] = guard.sha(self.component)
        with self.assertRaisesRegex(guard.EquationFixtureError, "fault was hidden"):
            self.validate()

    def test_changed_numeric_row_hash_is_rejected(self):
        self.value["recorder"]["rows"][1]["matrix_sha256"] = hash_bits(b"changed")
        with self.assertRaisesRegex(guard.EquationFixtureError, "bits changed"):
            self.validate()

    def test_status_payload_cannot_be_relabelled_as_accepted(self):
        self.bits["cpu_status"] = struct.pack("<2i", 0, 4)
        self.write_payload()
        self.value["equation_artifact"].update(sha256=guard.sha(self.payload), bytes=self.payload.stat().st_size)
        with self.assertRaisesRegex(guard.EquationFixtureError, "accepted statuses"):
            self.validate()

    def test_changed_runtime_and_missing_current_source_are_rejected(self):
        self.value["actual_runtime_binding_after"]["changed"] = True
        with self.assertRaisesRegex(guard.EquationFixtureError, "runtime"):
            self.validate()
        self.value["actual_runtime_binding_after"].pop("changed")
        self.value["artifacts_sha256"].pop(guard.ARTIFACTS[0])
        with self.assertRaisesRegex(guard.EquationFixtureError, "source bindings"):
            self.validate()

    def test_changed_call_or_seed_context_is_rejected(self):
        self.value["recorder"]["rows"][0]["context"][0]["input_sha256"] = hash_bits(b"new seed")
        with self.assertRaisesRegex(guard.EquationFixtureError, "proposal/seed"):
            self.validate()

    def test_changed_original_match_input_is_rejected(self):
        self.value["recorder"]["rows"][0]["context"][1]["initial_sha256"] = hash_bits(b"new seed")
        with self.assertRaisesRegex(guard.EquationFixtureError, "match inputs"):
            self.validate()

    def test_missing_tail_call_is_rejected(self):
        self.value["match_scopes"][-1]["end_record"] = 1
        with self.assertRaisesRegex(guard.EquationFixtureError, "coverage incomplete"):
            self.validate()

    def test_npz_extra_member_or_object_dtype_is_rejected(self):
        with zipfile.ZipFile(self.payload, "a") as archive:
            archive.writestr("extra.npy", b"unbound")
        with self.assertRaisesRegex(guard.EquationFixtureError, "exactly four"):
            guard.read_numeric_npz(self.payload, 2)
        self.layouts["matrices"] = ("|O", (2, 6, 6))
        self.write_payload()
        with self.assertRaisesRegex(guard.EquationFixtureError, "dtype/shape"):
            guard.read_numeric_npz(self.payload, 2)

    def test_npz_truncated_numeric_bits_are_rejected(self):
        self.bits["matrices"] = self.bits["matrices"][:-1]
        self.write_payload()
        with self.assertRaisesRegex(guard.EquationFixtureError, "payload length"):
            guard.read_numeric_npz(self.payload, 2)

    def test_nonfinite_difference_stays_a_json_failure(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            row = difference_evidence(value)
            self.assertIsNone(row["absolute_delta"])
            self.assertTrue(row["nonfinite_difference"])
            json.dumps(row, allow_nan=False)
        self.assertEqual(difference_evidence(0.), {"absolute_delta": 0., "nonfinite_difference": False})

    def test_shader_uses_headerless_cpp11_binary64_constants(self):
        source = (guard.ROOT / "scripts/research/research_device_ldlt.cu").read_text(encoding="utf-8")
        self.assertNotIn("#include", source)
        self.assertNotIn("NAN", source)
        self.assertNotIn("0x1p-1022", source)
        self.assertIn("__longlong_as_double(0x7ff8000000000000LL)", source)
        self.assertIn("eigen_double_min = 2.2250738585072014e-308", source)
        self.assertEqual(struct.unpack("<Q", struct.pack("<d", 2.2250738585072014e-308))[0], 1 << 52)


if __name__ == "__main__":
    unittest.main()
