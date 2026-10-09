"""Allocated pilot: prepared immutable lease versus frozen lookup baseline.

Fresh original scalar IDs/d2 bits are required before any timing. Lookup warm
clocks exclude separately recorded original PointCloud transformations and all
registration reductions; constructor/close/inclusive trials remain charged.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
from types import CodeType, FunctionType

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.research import benchmark_cached_target_geometry as original
from scripts.research import cached_target_geometry_prepared as prepared

KIND = "cached-target-geometry-prepared-original-api-ab-v1"
HELD_BASELINE = {
    "scripts/research/cached_target_geometry_native.cpp": "0aa7cf7ab0cfc46a502b10fd98e6a3bb19462f080d27bd172ed710084b68beff",
    "scripts/research/cached_target_geometry_native.py": "0c303d87c2a669c6424701b80b20bea80063eefdbe4d87f742695ab70be25a5f",
    "scripts/research/cached_target_geometry_native.md": "f0879c65be7fc6cf930cf6c6766132c58acb9fd1750c9c1440635e75e83a2f90",
    "tests/test_cached_target_geometry_native.py": "56fedf61efb44203af2508adc91fcf7792cd5e1007c183f7bb68af16b0074536",
    "scripts/research/benchmark_cached_target_geometry.py": "035ce0caad8f4d6e65e2a490f974a8932df04ba6dbaad7dae859b0aab996ce6b",
    "tests/test_cached_target_geometry_benchmark.py": "acab2af0867a74054e9a12c709d4ebe5ae484bfa8c135ffcc3a4b5a0caa68c3f"}


def preflight(args):
    for name, expected in HELD_BASELINE.items():
        original.require(original.helper.file_hash(ROOT / name) == expected, "Frozen baseline changed: " + name)
    binding = original.preflight(args)
    binding["prepared_source"] = prepared.source_contract()
    binding["artifacts_sha256"].update(binding["prepared_source"]["artifacts_sha256"])
    return binding


class LoadedOwners(original.LoadedOwners):
    def __init__(self, o3d):
        super().__init__(o3d)
        for module in (prepared, sys.modules[__name__]):
            cold = compile(Path(module.__file__).read_text(encoding="utf-8"), module.__file__, "exec", dont_inherit=True)
            for code in (value for value in cold.co_consts if isinstance(value, CodeType)):
                owner = getattr(module, code.co_name, None); slots = [(module, code.co_name, owner, code)]
                if isinstance(owner, type):
                    slots = [(owner, c.co_name, getattr(owner, c.co_name, None), c) for c in code.co_consts if isinstance(c, CodeType)]
                for slot, name, function, expected in slots:
                    if not isinstance(function, FunctionType): continue
                    original.require(function.__globals__ is module.__dict__ and original.code_key(function.__code__) == original.code_key(expected),
                                     "Loaded prepared body differs from source")
                    aliases = [(key, module.__dict__[key]) for key in function.__code__.co_names if key in module.__dict__]
                    self.records.append((slot, name, function, function.__code__, repr(function.__defaults__), repr(function.__kwdefaults__), module, aliases))
        self.check()


def make_lease(case, backend, threads, owners):
    owner = object.__new__(prepared.PreparedTarget); owners.append(owner)
    prepared.PreparedTarget.__init__(owner, backend, case["target_bytes"], case["record"]["target"]["shape"][0], query_threads=threads)
    return owner


def audit_case(case, backend, np, o3d, owners):
    original.audit_case(case, backend, np, o3d, owners)
    case["record"]["prepared_complete"] = False
    for row in case["record"]["parity"]:
        owner = None; primary = None
        try:
            owner = make_lease(case, backend, row["threads"], owners)
            ids, squared = owner.query(case["query_bytes"], case["record"]["queries"]["shape"][0], case["record"]["radius"])
            original.exact_result(case, ids, squared)
        except BaseException as error: primary = error; raise
        finally:
            if owner is not None: owner.close(primary)
        original.require(owner.closed, "Prepared audit owner did not close")
        row.update(prepared_ids_exact=True, prepared_squared_bits_exact=True, prepared_owner_closed=True)
    original.check_case(case, np); case["record"]["prepared_complete"] = True


def prepared_trial(case, backend, threads, repeats, owners, destination):
    started = time.perf_counter(); owner = None; primary = None
    row = {"threads": threads, "warm_query_wall_s": [], "complete": False}; destination.append(row)
    try:
        owner = make_lease(case, backend, threads, owners)
        before = time.perf_counter(); row["prepare_constructor_wall_s"] = before - started
        def query(): return owner.query(case["query_bytes"], case["record"]["queries"]["shape"][0], case["record"]["radius"])
        ids, squared = query()
        row["first_query_wall_s"] = time.perf_counter() - before
        row["cold_constructor_and_first_query_wall_s"] = time.perf_counter() - started
        original.exact_result(case, ids, squared)
        for _ in range(repeats):
            before = time.perf_counter(); ids, squared = query(); elapsed = time.perf_counter() - before
            row["warm_query_wall_s"].append(elapsed)
            original.exact_result(case, ids, squared)
    except BaseException as error: primary = error; raise
    finally:
        before = time.perf_counter()
        try:
            if owner is not None: owner.close(primary)
        finally:
            row["close_wall_s"] = time.perf_counter() - before
            row["whole_trial_wall_s"] = time.perf_counter() - started
    row["lease"] = owner.report(); original.require(row["lease"]["closed"] and row["lease"]["failure"] is None, "Prepared timing lease not clean")
    row["complete"] = True


def validate_prepared(report, repeats, names):
    original.validate_parity(report["parity"])
    for row in report["parity"]:
        original.require(row.get("prepared_complete") is True and all(value.get(key) is True for value in row["parity"]
                         for key in ("prepared_ids_exact", "prepared_squared_bits_exact", "prepared_owner_closed")), "Incomplete prepared scalar parity")
    original.validate_timings(report["benchmarks"], repeats, names)
    for row in report["benchmarks"]:
        trials = row.get("prepared_immutable_lease", [])
        original.require([value.get("threads") for value in trials] == list(original.helper.QUERY_THREADS), "Prepared timing thread coverage differs")
        for value in trials:
            walls = value.get("warm_query_wall_s", []); lease = value.get("lease", {})
            original.require(value.get("complete") is True and len(walls) == repeats and all(original.finite_wall(wall) for wall in walls)
                and all(original.finite_wall(value.get(key)) for key in ("prepare_constructor_wall_s", "first_query_wall_s",
                    "cold_constructor_and_first_query_wall_s", "close_wall_s", "whole_trial_wall_s")), "Prepared timing incomplete")
            original.require(value["cold_constructor_and_first_query_wall_s"] + 1e-9 >= value["prepare_constructor_wall_s"] + value["first_query_wall_s"]
                and value["whole_trial_wall_s"] + 1e-9 >= value["cold_constructor_and_first_query_wall_s"] + sum(walls) + value["close_wall_s"],
                "Prepared inclusive clock omits work")
            original.require(lease.get("closed") is True and lease.get("failure") is None and lease.get("owned_reservation_bytes") == 0
                and lease.get("queries") == repeats + 1 and lease.get("query_rows") == (repeats + 1) * row["queries"]["shape"][0]
                and lease.get("native_query_domain_and_result_checks") is True and lease.get("python_output_scan") is False,
                "Prepared native guards/owner/counters differ")


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True); parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field-geometry", type=Path); parser.add_argument("--field-report", type=Path)
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=3); parser.add_argument("--run-allocated", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv); original.require(args.run_allocated, "Root allocation requires --run-allocated")
    args.output = args.output.resolve(); original.require(not args.output.exists(), "Refuse an existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); owners = []; primary = None; binding = guard = saved_threads = o3d = None
    old_omp = os.environ.get("OMP_NUM_THREADS")
    report = {"kind": KIND, "status": "running", "failure": None, "cleanup_failures": [], "parity": [], "benchmarks": [],
        "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "scanner_speed_claim": False, "whole_finish_authority": False,
        "actual_finish_query_coverage": False,
        "scopes": {"prepared": "One target Python scan/hash+owned immutable bytes/native copy; every query still native finite/domain/strict NN checked",
            "warm": "No target Python scan/hash/output row scan; ABI input/output copies+native search/thread setup; excludes transform and registration reducer",
            "cold": "Target prepare/constructor+first query, close separately; whole trial includes every exact-output check",
            "comparison": "Original EvaluateRegistration performs transformation/tree rebuild/reduction; search-only clocks are a nonidentical scope",
            "order": "Pilot fixed order: original evaluation/raw/full-cache first, prepared lease afterward; no balanced speed claim"}}
    def save(): args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        binding = preflight(args); report["binding"] = binding; save()
        os.environ["OMP_NUM_THREADS"] = "8"
        import numpy as np
        import open3d as o3d
        saved_threads = o3d.utility.get_max_threads(); o3d.utility.set_max_threads(20)
        original.require(o3d.utility.get_max_threads() == 20, "Original TBB20 policy required")
        before = time.perf_counter(); backend = original.helper.NativeLibrary(args.library, args.build_receipt)
        report["native_library_load_s"] = time.perf_counter() - before
        guard = LoadedOwners(o3d); report["runtime"] = original.runtime_binding(np, o3d, backend)
        specs = list(original.synthetic_specs()); timings = [original.lattice_spec()]
        if args.field_geometry is not None: timings.extend(original.field_specs(args.field_geometry, np))
        cases = []
        for index, spec in enumerate(specs + timings):
            guard.check(); case = original.prepare_case(spec, np, o3d)
            if "selection" in spec: case["record"]["selection"] = spec["selection"]
            report["parity"].append(case["record"]); audit_case(case, backend, np, o3d, owners); guard.check()
            if index >= len(specs): cases.append(case)
        original.validate_parity(report["parity"])
        original.require(all(row.get("prepared_complete") is True for row in report["parity"]), "Fresh prepared parity required before timings")
        report["fresh_parity_passed_before_timing"] = True; save()
        for case in cases:
            guard.check(); row = original.benchmark_case(case, backend, np, o3d, args.repeats, owners, report["benchmarks"])
            row["complete"] = False; row["prepared_immutable_lease"] = []
            for threads in original.helper.QUERY_THREADS: prepared_trial(case, backend, threads, args.repeats, owners, row["prepared_immutable_lease"])
            original.check_case(case, np); row["complete"] = True; guard.check(); save()
        validate_prepared(report, args.repeats, [case["record"]["name"] for case in cases])
        report["runtime_after"] = original.runtime_binding(np, o3d, backend)
        original.require(report["runtime_after"] == report["runtime"], "Native/runtime/thread/floating state changed")
        original.check_fixed(binding); report["binding_after"] = copy.deepcopy(binding)
    except BaseException as error:
        primary = error; report["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for owner in reversed(owners):
            primary, _ = original.cleanup_step(report, primary, "target owner", lambda owner=owner: owner.close() if not getattr(owner, "closed", False) else None)
        for name, action in (("loaded owners", lambda: guard.check() if guard else None),
                             ("fixed source/resources", lambda: original.check_fixed(binding) if binding else None),
                             ("restore threads", lambda: o3d.utility.set_max_threads(saved_threads) if saved_threads is not None else None)):
            primary, _ = original.cleanup_step(report, primary, name, action)
        def restore_environment():
            if old_omp is None: os.environ.pop("OMP_NUM_THREADS", None)
            else: os.environ["OMP_NUM_THREADS"] = old_omp
        primary, _ = original.cleanup_step(report, primary, "restore environment", restore_environment)
        for name, action in (("owners_closed", lambda: all(getattr(owner, "closed", False) for owner in owners)),
                             ("threads_restored", lambda: saved_threads is None or o3d.utility.get_max_threads() == saved_threads),
                             ("environment_restored", lambda: os.environ.get("OMP_NUM_THREADS") == old_omp)):
            primary, value = original.cleanup_step(report, primary, name, action); report[name] = value is True
        report["whole_experiment_wall_s"] = time.perf_counter() - started; report["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        report["status"] = "passed" if primary is None and not report["cleanup_failures"] and all(report[key] for key in ("owners_closed", "threads_restored", "environment_restored")) else "failed"
        if primary is not None and report["failure"] is None: report["failure"] = {"type": type(primary).__name__, "message": str(primary)}
        try: save()
        except BaseException as error:
            if primary is not None: raise primary from error
            raise
    if primary is not None: raise primary
    original.require(report["status"] == "passed", "Prepared pilot closure failed")
    print(json.dumps({"status": report["status"], "parity_cases": len(report["parity"]), "output": str(args.output)}))


if __name__ == "__main__": main()
