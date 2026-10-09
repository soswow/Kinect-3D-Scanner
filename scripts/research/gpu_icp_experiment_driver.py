"""Allocated current-field independent-seed ICP audit, then separately timed runs.

Batch only genuine original proposal prefixes for one fixed prepared pair.
Original reciprocal/camera/held-out/visual/information gates consume each result
in order. All other ICP calls and competing proposals remain original CPU.
This is component evidence, not a full live/Finish/mesh or adaptive frontier run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import gpu_icp_experiment_protocol as protocol
from scripts.research.gpu_icp_experiment_capture import CURRENT, source_hash, sha, fixture_records
PRODUCER_FILES = ("scripts/research/gpu_icp_experiment_driver.py",
    "scripts/research/gpu_icp_experiment_protocol.py", "scripts/research/gpu_icp_experiment_capture.py",
    "tests/test_gpu_icp_experiment_capture.py", "tests/test_gpu_icp_experiment_protocol.py")
ENVIRONMENT = {"OMP_NUM_THREADS": "8", "KINECT_NATIVE": "on", "KINECT_CUDA_REGISTRATION": "cpu"}


def canonical_descriptor(record):
    return {key: record[key] for key in ("shape", "dtype", "nbytes", "sha256")}


def capture_pair(row):
    return {role: {name: canonical_descriptor(value) for name, value in row[role].items()}
            for role in ("source", "target")} | {"seeds": [canonical_descriptor(s) for s in row["seeds"]]}


def equal_evidence(x, y):
    if type(x) is dict and type(y) is dict:
        return x.keys() == y.keys() and all(equal_evidence(x[k], y[k]) for k in x)
    if type(x) is list and type(y) is list:
        return len(x) == len(y) and all(equal_evidence(a, b) for a, b in zip(x, y))
    if type(x) in (int, float) and type(y) in (int, float):
        return math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-8)
    return type(x) is type(y) and x == y


def correspondence_bounds(rows, source_count, target_count):
    if (type(source_count) is not int or type(target_count) is not int or source_count <= 0 or target_count <= 0
            or any(not (0 <= row[0] < source_count and 0 <= row[1] < target_count) for row in rows)):
        raise ValueError("Final original correspondence row/ID is outside consumed cloud bounds")


def result_shadow(native, candidate, target, index, np, module, *, source_count):
    a, b = np.asarray(native.correspondence_set), np.asarray(candidate.correspondence_set)
    def sorted_pairs(value):
        if value.ndim != 2 or value.shape[1] != 2 or value.dtype.kind not in "iu":
            raise ValueError("Malformed original correspondence set")
        if len(np.unique(value[:, 0])) != len(value):
            raise ValueError("A source row has duplicate final correspondences")
        correspondence_bounds(value, source_count, len(target.points))
        return value[np.lexsort((value[:, 1], value[:, 0]))].astype(np.int32, copy=False)
    a, b = sorted_pairs(a), sorted_pairs(b)
    differences = {"transform_max_abs_diff": float(np.max(np.abs(np.asarray(native.transformation)-np.asarray(candidate.transformation)))),
        "fitness_abs_diff": abs(float(native.fitness)-float(candidate.fitness)),
        "inlier_rmse_abs_diff": abs(float(native.inlier_rmse)-float(candidate.inlier_rmse))}
    ids_equal = np.array_equal(a, b)
    gate_equal = bool(module._strong(native, target)) == bool(module._strong(candidate, target))
    passed = ids_equal and gate_equal and all(math.isfinite(v) and v <= 1e-8 for v in differences.values())
    return {"seed_index": index, "passed": bool(passed), "correspondence_ids_equal": bool(ids_equal),
        "correspondence_order_scope": "Canonical original source-row/target-ID set; CPU internal iteration order is not authority",
        "strong_gate_equal": gate_equal, "native_correspondences": len(a), "candidate_correspondences": len(b), **differences}


def restore_matches(fixture, fragments, module):
    for fragment in fragments.values():
        for view in fragment.keys:
            view.match_cache.clear()
    for key, fragment in fragments.items():
        for item, view in zip(fixture["fragments"][key]["keys"], fragment.keys):
            for other_fragment in fragments.values():
                for other in other_fragment.keys:
                    match = item["matches"].get(other.index)
                    if match is not None:
                        module._cache_matches(view, other, match.copy())


def gate_shadow(native, candidate, consumed_once, index, np, baseline):
    x, y = baseline.json_value(native), baseline.json_value(candidate)
    if x is None or y is None:
        passed = x is None and y is None
        return {"seed_index": index, "passed": passed and consumed_once,
            "original_forward_consumed_once": consumed_once, "native_accepted": x is not None,
            "candidate_accepted": y is not None}
    exact_keys = (set(x) | set(y)) - {"transform", "information", "geometry", "visual", "validation"}
    exact = all(x.get(k) == y.get(k) for k in exact_keys)
    transform = float(np.max(np.abs(np.asarray(x["transform"])-np.asarray(y["transform"]))))
    info = float(np.max(np.abs(np.asarray(x["information"])-np.asarray(y["information"]))))
    passed = exact and transform <= 1e-8 and info <= 1e-5 and equal_evidence(x, y)
    return {"seed_index": index, "passed": bool(passed and consumed_once),
        "original_forward_consumed_once": consumed_once, "native_accepted": True, "candidate_accepted": True,
        "exact_witness_fields_equal": exact, "transform_max_abs_diff": transform, "information_max_abs_diff": info,
        "all_original_numeric_gate_evidence_equal_within_declared_tolerance": equal_evidence(x, y)}


def verdict(values, module):
    good = [value for value in values if value is not None]
    ambiguous = bool(good) and any(module._disagrees(good[0]["transform"], value["transform"]) for value in good[1:])
    return {"accepted": bool(good) and not ambiguous, "ambiguous": bool(ambiguous), "verified_proposals": len(good), "proposal_count": len(values)}


def final_closure(report):
    """Both audit and timing close current producer/helper/runtime identities."""
    helpers = report.get("helpers")
    return (report.get("cleanup_passed") is True and report.get("source_unchanged") is True
        and report.get("fixture_unchanged") is True and report.get("input_bytes_unchanged") is True
        and report.get("producer_artifacts_sha256") == report.get("producer_artifacts_sha256_after")
        and report.get("binding") is not None and report.get("binding_after") == report["binding"]
        and report.get("pair_binding") is not None and report.get("pair_binding_after") == report["pair_binding"]
        and type(helpers) is list and bool(helpers)
        and all(type(helper) is dict and helper.get("source_unchanged") is True
            and helper.get("closed") is True and helper.get("failure") is None
            and helper.get("cleanup_failures") == [] and helper.get("binding") == report["binding"] for helper in helpers))


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("audit", "timing"), default="audit")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[2])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--proof", type=Path)
    p.add_argument("--solver-dll", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/resident-icp/resident_solve.dll")
    p.add_argument("--run-allocated", action="store_true")
    args = p.parse_args()
    args.output, args.capture, args.solver_dll = args.output.resolve(), args.capture.resolve(), args.solver_dll.resolve()
    if (not args.run_allocated or not args.output.is_relative_to(ROOT/"benchmark-output") or args.output.exists()
            or args.task < 0 or args.batch_sizes != sorted(set(args.batch_sizes))
            or any(n not in (2, 4) for n in args.batch_sizes) or not 1 <= args.repeats <= 5
            or args.mode == "timing" and args.proof is None):
        p.error("Require allocated exclusive hardware, fresh private output and genuine bounded2/4 seed sizes; timing needs fresh proof")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            p.error("Stop the idle field server before allocated numerical work")
    return args


def run(args):
    report = {"kind": protocol.KIND if args.mode == "audit" else "gpu-icp-seed-microbatch-timing-v1",
        "status": "running", "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "whole_finish_authority": False, "performance_attribution_valid": args.mode == "timing",
        "scope": "One fixed raw-derived pair; genuine ordered proposal prefixes only. Every other original reciprocal/camera/held-out/visual/info call stays CPU. Full original competing-proposal verdict checked; no adaptive frontier, tracking, graph, fusion or mesh authority.",
        "audit_rows": [], "timing_rows": [], "batch_sizes": args.batch_sizes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save()
    failure = None
    helpers = []
    old_env = {k: os.environ.get(k) for k in ENVIRONMENT}
    thread_restore = None
    module = original_match = None
    fixed = {}
    try:
        report["source_sha256"] = source_hash()
        if report["source_sha256"] != CURRENT:
            raise RuntimeError("Current production source changed before experiment")
        capture = json.loads(args.capture.read_text(encoding="utf-8"))
        if (capture.get("kind") != "gpu-icp-current-field-pair-fixture-v2" or capture.get("status") != "passed"
                or capture.get("source_sha256") != CURRENT or capture.get("source_sha256_after") != CURRENT
                or capture.get("cleanup_passed") is not True or capture.get("pose_seeds_used_for_live_tracking") is not False
                or capture.get("artifacts_sha256") != capture.get("artifacts_sha256_after")):
            raise ValueError("Fresh closed current raw field fixture required")
        for path, expected in capture["artifacts_sha256"].items():
            if sha(ROOT/path) != expected:
                raise ValueError("Captured original preparation/helper changed")
        for name in ("fixture", "runtime_snapshot", "session", "profile", "native_extension"):
            record = capture[name]
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("Captured raw/profile/native/fixture bytes changed")
            fixed[record["path"]] = record["sha256"]
        fixed[str(args.capture)] = sha(args.capture)
        for path in (args.solver_dll, args.solver_dll.with_suffix(".build.json"), args.solver_dll.with_suffix(".cpp")):
            fixed[str(path)] = sha(path)
        if args.proof is not None:
            args.proof = args.proof.resolve()
            fixed[str(args.proof)] = sha(args.proof)
        report["fixed_files_sha256"] = dict(fixed)
        report["producer_artifacts_sha256"] = {path: sha(ROOT/path) for path in PRODUCER_FILES}
        os.environ.update(ENVIRONMENT)
        from scripts.research import benchmark_parallel_fragments as baseline
        from scripts.research.microbatch_icp import MicrobatchICP, cloud_record, matrix_record
        from scripts.research.archive.research_resident_icp import EigenSolve
        import cv2
        import open3d as o3d
        import numpy as np
        import scanner_server.fragments as module
        thread_restore = (cv2, o3d, cv2.getNumThreads(), o3d.utility.get_max_threads())
        cv2.setNumThreads(20); o3d.utility.set_max_threads(20)
        original_match = module._match
        with Path(capture["fixture"]["path"]).open("rb") as stream:
            fixture = pickle.load(stream)
        records = fixture_records(fixture)
        if records != capture["tasks"] or not args.task < len(records):
            raise ValueError("Actual fixture task/cloud/seed hashes differ from closed capture")
        task, record = fixture["tasks"][args.task], records[args.task]
        if any(size > record["proposal_count"] for size in args.batch_sizes):
            raise ValueError("Requested batch lacks genuine distinct original seeds; no padding allowed")
        a, b = task["pair"]
        fragments = {key: baseline.unpack_fragment(fixture["fragments"][key]) for key in (a, b)}
        source, target = fragments[a].train, fragments[b].train
        # This explicit layout-only copy preserves every original value bit and
        # is supplied identically to native CPU and GPU. No seed is perturbed.
        seeds = [np.ascontiguousarray(seed) for seed in task["proposals"]]
        pair = {"source": cloud_record(np, source), "target": cloud_record(np, target),
            "seeds": [matrix_record(np, seed) for seed in seeds]}
        if pair != capture_pair(record):
            raise RuntimeError("Consumed numerical values differ from original field fixture")
        report["pair_binding"] = pair
        report["actual_pair"] = {"position": args.task, "fragments": task["pair"], "genuine_seed_count": len(seeds)}
        native_results = []
        native_times = []
        native_gates = []
        for index, seed in enumerate(seeds):
            started = time.perf_counter()
            result = original_match(source, target, seed)
            native_times.append(time.perf_counter()-started)
            native_results.append(result)
            restore_matches(fixture, fragments, module)
            native_gates.append(module._verify_bridge(fragments[a], fragments[b], seed, fixture["camera"]))
            print(f"Native pair{a}/{b} original seed{index}: {time.perf_counter()-started:.3f}s", flush=True)
        report["original_pair_verdict"] = verdict(native_gates, module)
        report["initial_native_icp_s_by_seed"] = native_times
        solve_factory = lambda: EigenSolve(args.solver_dll)
        probe = MicrobatchICP(solve_factory, audit=True)
        helpers.append(probe)
        binding = probe.binding()
        report["binding"] = binding
        permit = None
        if args.mode == "timing":
            permit = protocol.validate_microbatch_audit(args.proof, binding, pair)
            report["audited_proof"] = {"path": str(args.proof), "sha256": sha(args.proof)}
            timed_helper = MicrobatchICP(solve_factory, audit=False, timing_permit=permit)
            helpers.append(timed_helper)
        # Numeric library bytes are closed independently from version labels.
        backend = sys.modules[o3d.geometry.PointCloud.__module__.rsplit(".", 1)[0]]
        numpy_core = sys.modules["numpy._core._multiarray_umath"]
        for path in (Path(backend.__file__), Path(numpy_core.__file__)):
            fixed[str(path)] = sha(path)
        report["fixed_files_sha256"] = dict(fixed)
        rounds = 1 if args.mode == "audit" else args.repeats
        report["native_timing_rows"] = []
        for repeat in range(rounds):
            for count in args.batch_sizes:
                if args.mode == "audit":
                    schedules = ("serial", "concurrent")
                else:
                    order = ["native", "serial", "concurrent"]
                    schedules = order[repeat % 3:] + order[:repeat % 3]
                for schedule in schedules:
                    if schedule == "native":
                        started = time.perf_counter()
                        results = [original_match(source, target, seed) for seed in seeds[:count]]
                        elapsed = time.perf_counter()-started
                        shadows = [result_shadow(native_results[i], value, target, i, np, module, source_count=len(source.points)) for i, value in enumerate(results)]
                        report["native_timing_rows"].append({"repeat": repeat, "batch_size": count,
                            "batch_wall_s": elapsed, "shadows": shadows, "passed": all(s["passed"] for s in shadows)})
                        save()
                        if not all(s["passed"] for s in shadows):
                            raise RuntimeError("Contemporary native CPU result changed across timing rounds")
                        continue
                    helper = probe if args.mode == "audit" else timed_helper
                    if helper.binding() != binding:
                        raise RuntimeError("Actual helper runtime differs from audited configuration")
                    started = time.perf_counter()
                    results = helper.match_seeds(source, target, seeds[:count], schedule=schedule)
                    elapsed = time.perf_counter()-started
                    shadows = [result_shadow(native_results[i], value, target, i, np, module, source_count=len(source.points)) for i, value in enumerate(results)]
                    gates, candidate_values = [], []
                    for index, seed in enumerate(seeds):
                        restore_matches(fixture, fragments, module)
                        consumed = [0]
                        def scoped_forward(ss, tt, initial):
                            if index < count and ss is source and tt is target and consumed[0] == 0:
                                if np.asarray(initial).tobytes() != seed.tobytes():
                                    raise RuntimeError("Original proposal forward seed changed")
                                consumed[0] += 1
                                return results[index]
                            return original_match(ss, tt, initial)
                        module._match = scoped_forward
                        try:
                            value = module._verify_bridge(fragments[a], fragments[b], seed, fixture["camera"])
                        finally:
                            module._match = original_match
                        candidate_values.append(value)
                        if index < count:
                            gates.append(gate_shadow(native_gates[index], value, consumed[0] == 1, index, np, baseline))
                        elif not gate_shadow(native_gates[index], value, True, index, np, baseline)["passed"]:
                            raise RuntimeError("Untouched original competing-proposal gate changed")
                    same_verdict = verdict(candidate_values, module) == report["original_pair_verdict"]
                    row = {"repeat": repeat, "batch_size": count, "schedule": schedule,
                        "pair_binding": protocol.prefix_pair(pair, count), "batch_record": helper.batches[-1],
                        "native_shadows": shadows, "bridge_gate_shadows": gates, "pair_verdict_equal": same_verdict,
                        "full_original_proposal_count": len(seeds), "batch_wall_s": elapsed,
                        "passed": same_verdict and all(s["passed"] for s in shadows) and all(g["passed"] for g in gates)}
                    report["audit_rows" if args.mode == "audit" else "timing_rows"].append(row)
                    save()
                    print(f"{args.mode} pair{a}/{b} {count}seed {schedule}: {elapsed:.6f}s pass={row['passed']}", flush=True)
                    if not row["passed"]:
                        raise RuntimeError("Fresh native/gate shadow failed; retain diagnostics and do not time/promote")
        report["pair_binding_after"] = {"source": cloud_record(np, source), "target": cloud_record(np, target),
            "seeds": [matrix_record(np, seed) for seed in seeds]}
        report["binding_after"] = probe.binding()
        report["input_bytes_unchanged"] = report["pair_binding_after"] == pair
    except BaseException as error:
        failure = error
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        cleanup = []
        def clean(name, action):
            try:
                return action()
            except BaseException as error:
                cleanup.append({"action": name, "type": type(error).__name__, "message": str(error)})
                return None
        if original_match is not None:
            clean("original gate hook", lambda: setattr(module, "_match", original_match))
        for helper in helpers:
            clean("lane stream/cache completion", helper.close)
        report["helpers"] = [clean("helper metadata", helper.report) for helper in helpers]
        if thread_restore is not None:
            cv2, o3d, cv_threads, o3d_threads = thread_restore
            clean("OpenCV restoration", lambda: cv2.setNumThreads(cv_threads))
            clean("Open3D restoration", lambda: o3d.utility.set_max_threads(o3d_threads))
        for key, value in old_env.items():
            clean("environment restoration:"+key, lambda key=key, value=value:
                os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value))
        report["source_sha256_after"] = clean("current source rehash", source_hash)
        report["producer_artifacts_sha256_after"] = clean("producer artifacts", lambda: {p: sha(ROOT/p) for p in PRODUCER_FILES})
        report["fixed_files_sha256_after"] = clean("raw/profile/native/fixture/solver/library bytes", lambda: {p: sha(p) for p in fixed})
        report["source_unchanged"] = report["source_sha256_after"] == report.get("source_sha256") == CURRENT
        report["fixture_unchanged"] = report["fixed_files_sha256_after"] == fixed
        report["cleanup_failures"] = cleanup
        report["cleanup_passed"] = not cleanup and all(os.environ.get(k) == v for k, v in old_env.items())
        report["status"] = "passed" if failure is None and final_closure(report) else "failed"
        report["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if report["status"] == "passed" and args.mode == "audit":
            try:
                protocol.validate_report(report, report["binding"], report["pair_binding"])
            except BaseException as error:
                failure = error
                report["status"] = "failed"
                report["validation_failure"] = {"type": type(error).__name__, "message": str(error)}
        try:
            save()
        except BaseException as write_error:
            if failure is not None:
                failure.add_note("Final diagnostic write failed: " + repr(write_error))
                raise failure from write_error
            raise
    if report["status"] != "passed":
        raise RuntimeError("Microbatch audit/timing failed; private current diagnostics preserved") from failure
    from scripts.process_metrics import finish_cuda_worker
    finish_cuda_worker()


if __name__ == "__main__":
    run(parse())
