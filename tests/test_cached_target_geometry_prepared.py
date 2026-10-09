"""Focused stdlib lease/benchmark contracts; no DLL or numerical package loads."""
from __future__ import annotations
import builtins
import copy
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.research import cached_target_geometry_prepared as lease
from scripts.research import benchmark_cached_target_geometry_prepared as bench
from tests import test_cached_target_geometry_benchmark as fixtures


class CheckedBackend(fixtures.Backend):
    def query(self, handle, points, rows, radius, threads):
        if any(not math.isfinite(value) or abs(value) > lease.native.COORDINATE_LIMIT
               for (value,) in struct.iter_unpack("<d", points)):
            raise lease.native.NativeGeometryError("Native query domain failure")
        return super().query(handle, points, rows, radius, threads)


def prepared_report():
    parity = fixtures.parity_rows()
    for row in parity:
        row["prepared_complete"] = True
        for value in row["parity"]:
            value.update(prepared_ids_exact=True, prepared_squared_bits_exact=True, prepared_owner_closed=True)
    timing = fixtures.timing_rows()
    timing[0]["queries"] = {"shape": [1, 3]}; timing[0]["prepared_immutable_lease"] = []
    for threads in lease.native.QUERY_THREADS:
        timing[0]["prepared_immutable_lease"].append({"threads": threads, "complete": True,
            "prepare_constructor_wall_s": 1., "first_query_wall_s": 1., "cold_constructor_and_first_query_wall_s": 2.,
            "warm_query_wall_s": [2., 3.], "close_wall_s": 1., "whole_trial_wall_s": 8.,
            "lease": {"closed": True, "failure": None, "owned_reservation_bytes": 0, "queries": 3, "query_rows": 3,
                      "native_query_domain_and_result_checks": True, "python_output_scan": False}})
    return {"parity": parity, "benchmarks": timing}


class Contracts(unittest.TestCase):
    def test_target_validation_once_and_fresh_native_queries(self):
        backend = CheckedBackend(); item = fixtures.case()
        with patch.object(lease.native, "points_descriptor", wraps=lease.native.points_descriptor) as scan:
            owner = lease.PreparedTarget(backend, item["target_bytes"], 1)
            for _ in range(3): self.assertEqual(owner.query(item["query_bytes"], 1, .03), item["gold"])
            self.assertEqual(scan.call_count, 1)
        self.assertEqual(owner.report()["queries"], 3); owner.close(); self.assertFalse(backend.active)

    def test_immutable_target_and_caps_refused_before_native_allocation(self):
        for points, rows, options in ((bytearray(fixtures.case()["target_bytes"]), 1, {}),
                                     (fixtures.case()["target_bytes"], 1, {"max_owned_bytes": 65537}),
                                     (fixtures.case()["target_bytes"], 1, {"query_threads": True})):
            backend = CheckedBackend()
            with self.assertRaises(ValueError) as caught: lease.PreparedTarget(backend, points, rows, **options)
            self.assertFalse(backend.calls); caught.exception.prepared_target.close()

    def test_query_immutable_byte_count_checked_before_backend(self):
        for payload, count in ((bytearray(24), 1), (b"short", 1), (bytes(24), True)):
            backend = CheckedBackend(); owner = lease.PreparedTarget(backend, fixtures.case()["target_bytes"], 1)
            with self.assertRaises(ValueError): owner.query(payload, count, .03)
            self.assertEqual(len(backend.calls), 1); owner.close()

    def test_native_query_finite_domain_guard_is_preserved_and_sticky(self):
        backend = CheckedBackend(); owner = lease.PreparedTarget(backend, fixtures.case()["target_bytes"], 1)
        with self.assertRaisesRegex(lease.native.NativeGeometryError, "Native query domain"):
            owner.query(struct.pack("<ddd", math.inf, 0., 0.), 1, .03)
        with self.assertRaisesRegex(lease.native.NativeGeometryError, "faulted"):
            owner.query(fixtures.case()["query_bytes"], 1, .03)
        owner.close(); self.assertTrue(owner.closed)

    def test_malformed_result_transport_is_refused(self):
        class BadBackend(CheckedBackend):
            def query(self, *args): return b"short", b"short"
        owner = lease.PreparedTarget(BadBackend(), fixtures.case()["target_bytes"], 1)
        with self.assertRaisesRegex(lease.native.NativeGeometryError, "transport"):
            owner.query(fixtures.case()["query_bytes"], 1, .03)
        owner.close()

    def test_config_entry_backend_and_callback_mutations_refused(self):
        for mutate in (lambda owner: setattr(owner, "configuration", (4, 256 * 1024 * 1024)),
                       lambda owner: setattr(owner, "backend", CheckedBackend()),
                       lambda owner: setattr(owner, "entry", None),
                       lambda owner: setattr(owner, "_query", lambda *args: fixtures.case()["gold"])):
            backend = CheckedBackend(); owner = lease.PreparedTarget(backend, fixtures.case()["target_bytes"], 1); mutate(owner)
            with self.assertRaisesRegex(lease.native.NativeGeometryError, "owner/configuration"):
                owner.query(fixtures.case()["query_bytes"], 1, .03)
            owner.close(); self.assertFalse(backend.active)

    def test_cleanup_uses_captured_backend_after_public_replacement(self):
        original, foreign = CheckedBackend(), CheckedBackend()
        owner = lease.PreparedTarget(original, fixtures.case()["target_bytes"], 1); owner.backend = foreign
        owner.close(); self.assertFalse(original.active); self.assertFalse(foreign.calls)

    def test_partial_allocated_construction_remains_owned(self):
        class BadReceipt(CheckedBackend):
            def create(self, points, rows, limit):
                handle, owned = super().create(points, rows, limit); return handle, owned + 1
        backend = BadReceipt()
        with self.assertRaises(lease.native.NativeGeometryError) as caught:
            lease.PreparedTarget(backend, fixtures.case()["target_bytes"], 1)
        self.assertTrue(backend.active); owner = caught.exception.prepared_target
        owner.close(); self.assertFalse(backend.active)

    def test_close_failure_retains_handle_and_preserves_primary(self):
        backend = CheckedBackend(); owner = lease.PreparedTarget(backend, fixtures.case()["target_bytes"], 1)
        primary = RuntimeError("first actual query failure"); backend.destroy_error = ValueError("destroy failed")
        with self.assertRaises(RuntimeError) as caught: owner.close(primary)
        self.assertIs(caught.exception, primary); self.assertIs(caught.exception.__cause__, backend.destroy_error)
        self.assertFalse(owner.closed); self.assertIsNotNone(owner.entry); self.assertTrue(backend.active)
        backend.destroy_error = None; owner.close(); self.assertTrue(owner.closed); self.assertFalse(backend.active)

    def test_empty_query_still_uses_native_path(self):
        backend = CheckedBackend(); owner = lease.PreparedTarget(backend, fixtures.case()["target_bytes"], 1)
        self.assertEqual(owner.query(b"", 0, .03), (b"", b"")); self.assertEqual(owner.queries, 1); owner.close()

    def test_exact_parity_check_is_outside_prepared_transport(self):
        class DifferentWinner(CheckedBackend):
            def query(self, *args): return struct.pack("<q", -1), struct.pack("<d", math.inf)
        owner = lease.PreparedTarget(DifferentWinner(), fixtures.case()["target_bytes"], 1)
        result = owner.query(fixtures.case()["query_bytes"], 1, .03)
        with self.assertRaisesRegex(ValueError, "IDs differ"): bench.original.exact_result(fixtures.case(), *result)
        owner.close()

    def test_cold_prepare_and_close_clock_is_inclusive(self):
        clock = fixtures.Clock(); backend = CheckedBackend(clock); rows = []; owners = []
        real = lease.PreparedTarget.__init__
        def slow_constructor(self, *args, **kwargs): clock.advance(3.); real(self, *args, **kwargs)
        with patch.object(lease.PreparedTarget, "__init__", slow_constructor), patch.object(bench.time, "perf_counter", clock.read):
            bench.prepared_trial(fixtures.case(), backend, 1, 2, owners, rows)
        row = rows[0]
        self.assertEqual(row["prepare_constructor_wall_s"], 5.); self.assertEqual(row["first_query_wall_s"], 5.)
        self.assertEqual(row["cold_constructor_and_first_query_wall_s"], 10.)
        self.assertEqual(row["warm_query_wall_s"], [5., 5.]); self.assertEqual(row["close_wall_s"], 7.)
        self.assertEqual(row["whole_trial_wall_s"], 27.); self.assertFalse(backend.active)

    def test_partial_trial_diagnostics_and_owner_survive_failure(self):
        class QueryFailure(CheckedBackend):
            def query(self, *args): raise RuntimeError("actual native query failure")
        backend = QueryFailure(); backend.destroy_error = ValueError("cleanup failed"); rows = []; owners = []
        with self.assertRaises(RuntimeError): bench.prepared_trial(fixtures.case(), backend, 1, 2, owners, rows)
        self.assertFalse(rows[0]["complete"]); self.assertIn("whole_trial_wall_s", rows[0]); self.assertTrue(backend.active)
        backend.destroy_error = None; owners[0].close()

    def test_validation_requires_prepared_parity_native_guards_and_inclusive_walls(self):
        report = prepared_report(); bench.validate_prepared(report, 2, ["fake"])
        for mutation in (lambda value: value["parity"][0]["parity"][0].update(prepared_ids_exact=False),
                         lambda value: value["benchmarks"][0]["prepared_immutable_lease"][0]["lease"].update(native_query_domain_and_result_checks=False),
                         lambda value: value["benchmarks"][0]["prepared_immutable_lease"][0].update(whole_trial_wall_s=7.),
                         lambda value: value["benchmarks"][0]["prepared_immutable_lease"][0]["lease"].update(queries=2)):
            bad = copy.deepcopy(report); mutation(bad)
            with self.assertRaises(ValueError): bench.validate_prepared(bad, 2, ["fake"])

    def test_positive_preflight_and_failed_diagnostic_before_numerical_import(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = fixtures.Contracts(); args = setup.args(Path(directory)); setup.receipt(args)
            binding = bench.preflight(args); self.assertEqual(binding["prepared_source"], lease.source_contract())
            original_import = builtins.__import__
            def stop(name, *args, **kwargs):
                if name == "numpy": raise RuntimeError("source-only positive setup boundary")
                return original_import(name, *args, **kwargs)
            with patch.object(builtins, "__import__", stop):
                with self.assertRaisesRegex(RuntimeError, "positive setup boundary"):
                    bench.main(["--library", str(args.library), "--build-receipt", str(args.build_receipt),
                                "--output", str(args.output), "--run-allocated"])
            report = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed"); self.assertTrue(report["owners_closed"])
            self.assertFalse(report["scanner_speed_claim"]); self.assertFalse(report["actual_finish_query_coverage"])

    def test_cold_loaded_owner_guard_rejects_prepared_body_mutation(self):
        class PointCloud:
            def transform(self, value): return self
        class KDTree:
            def search_hybrid_vector_3d(self, *args): return (0, [], [])
        fake = SimpleNamespace(geometry=SimpleNamespace(PointCloud=PointCloud, KDTreeFlann=KDTree),
            utility=SimpleNamespace(Vector3dVector=lambda value: value, get_max_threads=lambda: 20, set_max_threads=lambda value: None),
            pipelines=SimpleNamespace(registration=SimpleNamespace(evaluate_registration=lambda *args: None)))
        guard = bench.LoadedOwners(fake); code = lease.PreparedTarget.query.__code__
        try:
            lease.PreparedTarget.query.__code__ = (lambda *args: None).__code__
            with self.assertRaisesRegex(ValueError, "owner changed"): guard.check()
        finally: lease.PreparedTarget.query.__code__ = code
        guard.check()

    def test_fresh_import_and_help_are_stdlib_only(self):
        statement = "import sys; sys.path.insert(0," + repr(str(ROOT)) + "); "
        statement += "from scripts.research import benchmark_cached_target_geometry_prepared; "
        statement += "assert not any(n.split('.')[0] in {'numpy','open3d','cupy','cv2','ctypes'} for n in sys.modules)"
        result = subprocess.run([sys.executable, "-S", "-c", statement], capture_output=True, text=True, cwd=ROOT.parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-S", str(ROOT / lease.FILES[1]), "--help"], capture_output=True, text=True, cwd=ROOT.parent)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__": unittest.main()
