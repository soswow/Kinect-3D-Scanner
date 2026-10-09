"""Stdlib-only contracts; this suite never loads the research DLL or NumPy."""
from __future__ import annotations
import copy
import builtins
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.research import benchmark_cached_target_geometry as bench


def case():
    points = struct.pack("<ddd", 0., 0., 0.)
    ids, squared = struct.pack("<q", 0), struct.pack("<d", 0.)
    descriptor = bench.helper.points_descriptor(points, 1)
    return {"record": {"name": "fake", "target": descriptor, "queries": descriptor, "radius": .03},
            "target_bytes": points, "query_bytes": points, "gold": (ids, squared)}


class Clock:
    def __init__(self): self.value = 0.
    def read(self): return self.value
    def advance(self, amount): self.value += amount


class Backend:
    def __init__(self, clock=None):
        self.clock = clock; self.calls = []; self.active = set(); self.next = 1; self.destroy_error = None
    def create(self, points, rows, limit):
        if self.clock: self.clock.advance(2.)
        handle = self.next; self.next += 1; self.active.add(handle)
        self.calls.append(("create", points, rows, limit))
        return handle, limit
    def query(self, handle, points, rows, radius, threads):
        if self.clock: self.clock.advance(5.)
        self.calls.append(("query", handle, points, rows, radius, threads))
        return struct.pack("<q", 0) * rows, struct.pack("<d", 0.) * rows
    def destroy(self, handle):
        if self.clock: self.clock.advance(7.)
        self.calls.append(("destroy", handle))
        if self.destroy_error: raise self.destroy_error
        self.active.remove(handle)


def parity_rows():
    rows = []
    for spec in list(bench.synthetic_specs()) + [{"name": "synthetic-lattice", "queries": [0]}]:
        count = len(spec["queries"])
        rows.append({"name": spec["name"], "complete": True, "queries": {"shape": [count, 3]},
            "original_scalar_reference": {"query_rows": count, "hits": count},
            "parity": [{"threads": threads, "raw_ids_exact": True, "raw_squared_bits_exact": True,
                        "wrapper_ids_exact": True, "wrapper_squared_bits_exact": True, "owners_closed": True}
                       for threads in bench.helper.QUERY_THREADS]})
    return rows


def timing_rows():
    trial = {"complete": True, "warm_query_wall_s": [2., 3.], "cold_constructor_and_first_query_wall_s": 4.,
             "close_wall_s": 1., "whole_trial_wall_s": 10.}
    row = {"name": "fake", "complete": True, "evaluate_registration": [{"wall_s": 1., "canonical_ids_exact": True,
            "helper_rmse_equivalence_claim": False} for _ in range(2)], "raw_cached_native": [], "full_python_cache": []}
    for threads in bench.helper.QUERY_THREADS:
        raw = {"threads": threads, **copy.deepcopy(trial)}; row["raw_cached_native"].append(raw)
        wrapped = copy.deepcopy(raw)
        wrapped["cache"] = {"closed": True, "failure": None, "retained_targets": 0, "owned_reservation_bytes": 0,
                            "statistics": {"builds": 1, "queries": 3}}
        row["full_python_cache"].append(wrapped)
    return [row]


def npz(path, *, dtype="<f8", shape=(2, 3), payload=None, fortran=False):
    raw = repr({"descr": dtype, "fortran_order": fortran, "shape": shape}).encode("latin1") + b"\n"
    values = bytes(24 * shape[0]) if payload is None else payload
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("points.npy", b"\x93NUMPY\x01\x00" + len(raw).to_bytes(2, "little") + raw + values)


class Contracts(unittest.TestCase):
    def args(self, directory):
        return SimpleNamespace(run_allocated=True, output=directory / "fresh.json", library=directory / "fake.dll",
            build_receipt=directory / "build.json", field_geometry=None, field_report=None)

    def receipt(self, args):
        args.library.write_bytes(b"unloaded fake DLL")
        compiler = args.library.with_name("cl.exe"); compiler.write_bytes(b"unexecuted fake compiler")
        receipt = {"kind": bench.helper.KIND, "stage": "built", "actual_exit_code": 0,
            "source_contract": bench.helper.source_contract(), "library_sha256": bench.helper.file_hash(args.library),
            "command": [str(compiler), "/std:c++17", "/fp:strict", "/DEIGEN_DONT_PARALLELIZE"],
            "compiler": {"path": str(compiler), "sha256": bench.helper.file_hash(compiler)},
            "dependencies": {"eigen": {"archive_sha256": bench.helper.EIGEN_ARCHIVE_SHA256,
                "headers_sha256": hashlib.sha256(b"fake Eigen headers").hexdigest(), "files": ["COPYING.MPL2"]},
                "nanoflann": {"archive_sha256": bench.helper.NANOFLANN_ARCHIVE_SHA256,
                "headers_sha256": hashlib.sha256(b"fake nanoflann headers").hexdigest(), "files": ["COPYING"]}}}
        args.build_receipt.write_text(json.dumps(receipt), encoding="utf-8")
        return receipt

    def test_preflight_and_fixed_receipts_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory)); self.receipt(args)
            with patch.object(bench.helper, "NativeLibrary", side_effect=AssertionError("Do not load")):
                binding = bench.preflight(args); bench.check_fixed(binding)
            self.assertFalse(binding["dependency_assets_rehashed_at_runtime"])
            self.assertEqual(binding["recorded_build_dependencies"], json.loads(args.build_receipt.read_text())["dependencies"])
            args.library.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Fixed resource changed"): bench.check_fixed(binding)

    def test_nonfinite_json_literals_and_overflow_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            for number in ("NaN", "Infinity", "1e400"):
                path.write_text('{"value":' + number + '}', encoding="utf-8")
                with self.assertRaises(ValueError): bench.read_json(path)

    def test_preflight_refuses_unallocated_existing_and_bad_build(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory)); receipt = self.receipt(args)
            args.run_allocated = False
            with self.assertRaisesRegex(ValueError, "allocation"): bench.preflight(args)
            args.run_allocated = True; args.output.write_text("preserved", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "existing output"): bench.preflight(args)
            args.output.unlink(); receipt["actual_exit_code"] = 1
            args.build_receipt.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "successful helper"): bench.preflight(args)

    def test_build_policy_and_dependency_license_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory)); baseline = self.receipt(args)
            for mutation in (lambda r: r["command"].append("/fp:fast"),
                             lambda r: r["command"].append("/DNANOFLANN_FIRST_MATCH"),
                             lambda r: r["dependencies"]["nanoflann"].update(files=[])):
                receipt = copy.deepcopy(baseline); mutation(receipt)
                args.build_receipt.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaises(ValueError): bench.preflight(args)

    def test_field_npz_header_numeric_shape_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.npz"; npz(path, fortran=True)
            self.assertEqual(bench.point_header(path), {"dtype": "<f8", "shape": [2, 3], "fortran_order": True})
            for values in ({"dtype": "|O"}, {"shape": (2, 2)}, {"payload": b"short"}):
                npz(path, **values)
                with self.assertRaises(ValueError): bench.point_header(path)

    def test_field_geometry_requires_closed_matching_native_export(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory)); self.receipt(args)
            args.field_geometry = Path(directory) / "geometry.npz"; npz(args.field_geometry)
            args.field_report = Path(directory) / "native.json"
            digest = bench.helper.file_hash(args.field_geometry)
            native = {"kind": bench.FIELD_KIND, "mode": "native", "status": "passed", "failure": None,
                "cleanup_passed": True, "cleanup_failures": [], "binding": {"test": 1}, "binding_after": {"test": 1},
                "candidate_source": {"test": 2}, "candidate_source_after": {"test": 2}, "loaded_owners_unchanged": True,
                "geometry": {"path": str(args.field_geometry), "sha256": digest},
                "profile": {"mesh_built": True, "geometry": {"sha256": digest}}}
            args.field_report.write_text(json.dumps(native), encoding="utf-8")
            self.assertFalse(bench.preflight(args)["field_data"]["actual_finish_queries"])
            for key, value in (("cleanup_passed", False), ("binding_after", {"test": 9}), ("kind", "old-proof")):
                bad = copy.deepcopy(native); bad[key] = value; args.field_report.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError): bench.preflight(args)

    def test_exact_id_and_squared_bit_refusals(self):
        item = case(); bench.exact_result(item, *item["gold"])
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            bench.exact_result(item, struct.pack("<q", -1), struct.pack("<d", math.inf))
        with self.assertRaisesRegex(ValueError, "bits differ"):
            bench.exact_result(item, item["gold"][0], struct.pack("<d", math.nextafter(0., 1.)))

    def test_strict_radius_and_malformed_output_are_refused(self):
        item = case()
        with self.assertRaises(bench.helper.NativeGeometryError):
            bench.exact_result(item, item["gold"][0], struct.pack("<d", .03 * .03))
        with self.assertRaises(bench.helper.NativeGeometryError): bench.exact_result(item, b"", b"")

    def test_raw_owner_retained_on_destroy_failure_and_primary_preserved(self):
        backend = Backend(); owner = bench.OwnedTarget(backend); owner.create(case()["target_bytes"], 1)
        primary = RuntimeError("original failure"); backend.destroy_error = ValueError("cleanup failure")
        with self.assertRaises(RuntimeError) as caught: bench.close_one(owner, primary)
        self.assertIs(caught.exception, primary); self.assertIs(caught.exception.__cause__, backend.destroy_error)
        self.assertIsNotNone(owner.handle); self.assertFalse(owner.closed)
        backend.destroy_error = None; owner.close(); self.assertFalse(backend.active)

    def test_raw_owner_cleanup_uses_captured_backend(self):
        original, foreign = Backend(), Backend(); owner = bench.OwnedTarget(original)
        owner.create(case()["target_bytes"], 1); owner.backend = foreign; owner.close()
        self.assertFalse(original.active); self.assertFalse(foreign.calls)

    def test_cold_clock_charges_constructor_query_and_close(self):
        clock = Clock(); backend = Backend(clock); real = bench.OwnedTarget
        class SlowConstructor(real):
            def __init__(self, backend): clock.advance(3.); super().__init__(backend)
        with patch.object(bench, "OwnedTarget", SlowConstructor), patch.object(bench.time, "perf_counter", clock.read):
            row = bench.timed_lookup(case(), backend, 4, 2, [])
        self.assertEqual(row["cold_constructor_and_first_query_wall_s"], 10.)
        self.assertEqual(row["warm_query_wall_s"], [5., 5.]); self.assertEqual(row["close_wall_s"], 7.)
        self.assertEqual(row["whole_trial_wall_s"], 27.); self.assertFalse(backend.active)

    def test_full_cache_cold_and_warm_owners_are_closed(self):
        backend = Backend(); row = bench.timed_lookup(case(), backend, 20, 2, [], wrapped=True)
        self.assertTrue(row["cache"]["closed"]); self.assertEqual(row["cache"]["statistics"]["builds"], 1)
        self.assertEqual(row["cache"]["statistics"]["queries"], 3); self.assertFalse(backend.active)
        self.assertTrue(all(call[-1] == 20 for call in backend.calls if call[0] == "query"))

    def test_partial_trial_receipt_survives_query_and_cleanup_failure(self):
        backend = Backend(); primary = RuntimeError("query failed"); backend.destroy_error = ValueError("close failed")
        rows = []
        with patch.object(backend, "query", side_effect=primary):
            with self.assertRaises(RuntimeError) as caught: bench.timed_lookup(case(), backend, 1, 2, [], destination=rows)
        self.assertIs(caught.exception, primary); self.assertEqual(len(rows), 1); self.assertFalse(rows[0]["complete"])
        self.assertIn("whole_trial_wall_s", rows[0]); self.assertTrue(backend.active)

    def test_parity_requires_all_cases_threads_and_exact_outputs(self):
        rows = parity_rows(); bench.validate_parity(rows)
        for mutate in (lambda values: values.pop(0), lambda values: values[0]["parity"].pop(),
                       lambda values: values[0]["parity"][0].update(raw_squared_bits_exact=False),
                       lambda values: values[0].update(complete=False)):
            bad = copy.deepcopy(rows); mutate(bad)
            with self.assertRaises(ValueError): bench.validate_parity(bad)

    def test_timing_receipts_require_inclusive_walls_closed_cache_and_distinct_scope(self):
        rows = timing_rows(); bench.validate_timings(rows, 2, ["fake"])
        for mutate in (lambda values: values[0]["raw_cached_native"][0].update(whole_trial_wall_s=8.),
                       lambda values: values[0]["full_python_cache"][0]["cache"].update(closed=False),
                       lambda values: values[0]["evaluate_registration"][0].update(helper_rmse_equivalence_claim=True),
                       lambda values: values[0]["raw_cached_native"][0]["warm_query_wall_s"].append(math.nan)):
            bad = copy.deepcopy(rows); mutate(bad)
            with self.assertRaises(ValueError): bench.validate_timings(bad, 2, ["fake"])

    def test_independent_cleanup_keeps_primary_and_attempts_later_action(self):
        report = {"cleanup_failures": []}; primary = RuntimeError("actual method failure"); calls = []
        def fail(): calls.append("failed"); raise ValueError("receipt getter failed")
        actual, _ = bench.cleanup_step(report, primary, "receipt", fail)
        actual, value = bench.cleanup_step(report, actual, "later", lambda: calls.append("later") or True)
        self.assertIs(actual, primary); self.assertTrue(value); self.assertEqual(calls, ["failed", "later"])
        self.assertEqual(len(report["cleanup_failures"]), 1)

    def test_loaded_function_and_native_alias_ownership_guard(self):
        class PointCloud:
            def transform(self, matrix): return self
        class KDTree:
            def search_hybrid_vector_3d(self, point, radius, cap): return (0, [], [])
        registration = SimpleNamespace(evaluate_registration=lambda *args: None)
        utility = SimpleNamespace(Vector3dVector=lambda values: values, get_max_threads=lambda: 20,
                                  set_max_threads=lambda value: None)
        fake = SimpleNamespace(geometry=SimpleNamespace(PointCloud=PointCloud, KDTreeFlann=KDTree),
                               pipelines=SimpleNamespace(registration=registration), utility=utility)
        guard = bench.LoadedOwners(fake); guard.check(); original_evaluate = registration.evaluate_registration
        original = bench.exact_result.__code__
        try:
            bench.exact_result.__code__ = (lambda *args: None).__code__
            with self.assertRaisesRegex(ValueError, "owner changed"): guard.check()
        finally: bench.exact_result.__code__ = original
        guard.check(); registration.evaluate_registration = lambda *args: None
        with self.assertRaisesRegex(ValueError, "native callable owner changed"): guard.check()
        registration.evaluate_registration = original_evaluate
        # A separately captured guard also checks actual cloud construction/thread control aliases.
        guard = bench.LoadedOwners(fake); utility.set_max_threads = lambda value: None
        with self.assertRaisesRegex(ValueError, "native callable owner changed"): guard.check()

    def test_preflight_failure_saves_failed_diagnostic_before_any_native_load(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory)); receipt = self.receipt(args); receipt["actual_exit_code"] = 1
            args.build_receipt.write_text(json.dumps(receipt), encoding="utf-8")
            with patch.object(bench.helper, "NativeLibrary", side_effect=AssertionError("native load forbidden")):
                with self.assertRaises(ValueError): bench.main(["--library", str(args.library), "--build-receipt",
                    str(args.build_receipt), "--output", str(args.output), "--run-allocated"])
            report = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed"); self.assertIsNotNone(report["failure"])
            self.assertFalse(report["actual_finish_query_coverage"]); self.assertTrue(report["owners_closed"])

    def test_positive_main_preflight_reaches_import_without_self_freshness_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory)); self.receipt(args); original_import = builtins.__import__
            attempted = []
            def blocked_import(name, *values, **keywords):
                if name == "numpy":
                    attempted.append(name); raise RuntimeError("Positive preflight reached numerical boundary")
                return original_import(name, *values, **keywords)
            with patch.object(builtins, "__import__", blocked_import):
                with self.assertRaisesRegex(RuntimeError, "Positive preflight reached"):
                    bench.main(["--library", str(args.library), "--build-receipt", str(args.build_receipt),
                                "--output", str(args.output), "--run-allocated"])
            report = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertEqual(attempted, ["numpy"]); self.assertIn("binding", report)
            self.assertEqual(report["status"], "failed"); self.assertTrue(report["environment_restored"])

    def test_fresh_import_and_help_do_not_import_numerical_or_load_native(self):
        source = "import sys; sys.path.insert(0," + repr(str(ROOT)) + "); "
        source += "from scripts.research import benchmark_cached_target_geometry; "
        source += "assert not any(n.split('.')[0] in {'numpy','open3d','cupy','cv2','ctypes'} for n in sys.modules)"
        result = subprocess.run([sys.executable, "-S", "-c", source], capture_output=True, text=True, cwd=ROOT.parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        help_result = subprocess.run([sys.executable, "-S", str(ROOT / bench.OWN_FILES[0]), "--help"],
                                     capture_output=True, text=True, cwd=ROOT.parent)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)


if __name__ == "__main__": unittest.main()
