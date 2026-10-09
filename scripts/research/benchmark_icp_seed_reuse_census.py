"""Observe original CPU complete proposals; cluster actual seeds offline only.

Every genuine proposal and every dynamic original ICP call executes. Exact
source/target point, normal and color bytes define buckets. Ordered greedy
seed classes compare each member only with their first representative, never
transitive neighbours. This grants no caching, skipped-call or speed authority.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import socket
import sys
import time
import traceback
from types import CodeType, FunctionType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import microbatch_bridge_scope as scope

KIND = "original-cpu-complete-proposal-seed-reuse-census-v1"
CURRENT = "07a948e81127dc742a330ec6bc24c85ceb7a9996128703c2a1cb7f9d0593741c"
THRESHOLDS = (0., 1e-12, 1e-10, 1e-8, 1e-6)
ENVIRONMENT = {"OMP_NUM_THREADS": "8", "KINECT_NATIVE": "on", "KINECT_CUDA_REGISTRATION": "cpu"}
OWN_FILES = ("scripts/research/benchmark_icp_seed_reuse_census.py", "tests/test_icp_seed_reuse_census.py")
HELD_FILES = {
    "scripts/research/microbatch_bridge_scope.py": "a9303b1f85d8eaaa427e8a873b5c32b623ac16d971725c418c8050202014bf80",
    "scripts/research/microbatch_bridge_driver.py": "d048525610ac41d4b591ed5779d3d01ad06876f786d662fe067fcea8e6d2c2ec"}


def require(ok, message):
    if not ok: raise scope.BridgeFailure(message)


def matrix_values(value):
    require(type(value) is list and len(value) == 4 and all(type(r) is list and len(r) == 4 for r in value),
        "Unrounded original 4x4 matrix values required")
    result = [v for row in value for v in row]
    require(all(type(v) is float and math.isfinite(v) for v in result), "Finite FP64 seed/result values required")
    return result


def max_delta(a, b):
    return max(abs(x-y) for x, y in zip(matrix_values(a), matrix_values(b)))


def cloud_bucket(row):
    return scope.canonical({name: row["input_binding"][name] for name in ("source", "target")})


def result_delta(a, b):
    return {"transform_max_abs_delta": max_delta(a["transformation"], b["transformation"]),
        "fitness_abs_delta": abs(a["fitness"]-b["fitness"]),
        "rmse_abs_delta": abs(a["inlier_rmse"]-b["inlier_rmse"]),
        "canonical_ids_equal": a["correspondence_mapping"] == b["correspondence_mapping"],
        "raw_pair_order_equal": a["raw_correspondences"] == b["raw_correspondences"]}


def cluster_rows(rows, threshold):
    """Exact direction/value bucket, first representative, original call order."""
    require(type(threshold) is float and threshold in THRESHOLDS, "Only declared census thresholds")
    buckets, groups = {}, []
    for index, row in enumerate(rows):
        require(row.get("complete") is True and row.get("input_bytes_unchanged") is True,
            "Only complete immutable original calls can enter offline census")
        matrix_values(row["seed_values"])
        key = cloud_bucket(row)
        representatives = buckets.setdefault(key, [])
        found = None
        for group in representatives:
            old = rows[group["representative"]]
            equal = (old["input_binding"]["seed"] == row["input_binding"]["seed"] if threshold == 0.
                else max_delta(old["seed_values"], row["seed_values"]) <= threshold)
            if equal:
                found = group; break
        if found is None:
            found = {"representative": index, "members": [index]}
            representatives.append(found); groups.append(found)
        else: found["members"].append(index)
    repeated, comparisons = [], []
    for group in groups:
        if len(group["members"]) < 2: continue
        representative = rows[group["representative"]]
        members = []
        for index in group["members"][1:]:
            row = rows[index]
            delta = result_delta(representative["result"], row["result"])
            delta.update(call_index=index, seed_max_abs_delta=max_delta(representative["seed_values"], row["seed_values"]))
            comparisons.append(delta); members.append(delta)
        repeated.append({"representative": group["representative"], "members": group["members"], "result_comparisons": members})
    return {"seed_matrix_max_abs_threshold": threshold, "exact_full_seed_bytes": threshold == 0.,
        "original_calls": len(rows), "exact_cloud_value_buckets": len(buckets), "greedy_classes": len(groups),
        "entries_after_first_representative": len(rows)-len(groups), "repeated_classes": len(repeated),
        "canonical_id_changed_comparisons": sum(not r["canonical_ids_equal"] for r in comparisons),
        "max_result_deltas": {name: max((r[name] for r in comparisons), default=0.) for name in
            ("transform_max_abs_delta", "fitness_abs_delta", "rmse_abs_delta")},
        "groups": repeated, "transitive_clustering": False, "skipped_calls": 0,
        "safe_reuse_established": False, "speed_authority": False}


class CpuCallObserver:
    """Delegates exactly once and returns the actual original result object."""
    def __init__(self, np, guard, row, context, route):
        self.np, self.guard, self.row, self.context = np, guard, row, context
        self.route = route
        self.failure = None

    def __call__(self, source, target, initial):
        if self.failure is not None: raise scope.BridgeFailure("Failed census observer cannot resume") from self.failure
        call = {"context": dict(self.context), "call_index": len(self.row["calls"]),
            "caller": sys._getframe(2).f_code.co_name, "complete": False}
        self.row["calls"].append(call)
        try:
            self.guard.check(); self.route()
            call["input_binding"] = scope.input_binding(self.np, source, target, initial)
            call["seed_values"] = self.np.asarray(initial).tolist(); matrix_values(call["seed_values"])
            require(call["input_binding"]["seed"]["dtype"] == "<f8"
                and call["input_binding"]["seed"]["shape"] == [4, 4], "Original FP64 seed required")
            started = time.perf_counter()
            result = self.guard.cpu_match(source, target, initial)
            call["original_cpu_match_wall_s"] = time.perf_counter()-started
            call["result"] = scope.result_evidence(self.np, result, len(source.points), len(target.points))
            call["input_bytes_unchanged"] = call["input_binding"] == scope.input_binding(self.np, source, target, initial)
            require(call["input_bytes_unchanged"], "Original census call changed arrays or seed")
            self.guard.check(); self.route(); call["complete"] = True
            return result
        except BaseException as error:
            self.failure = error
            call["failure"] = {"type": type(error).__name__, "message": str(error)}
            raise scope.BridgeFailure("Original CPU call/evidence census failed; no continuation") from error


class OriginalCpuRoute:
    """Source-bound original dispatch is disabled, not replaced or invoked."""
    def __init__(self, module):
        self.module, self.path = module, Path(module.__file__).resolve()
        self.digest, self.function, self.context = scope.sha(self.path), module.match, module._device
        code = next(c for c in compile(self.path.read_text(encoding="utf-8"), str(self.path), "exec", dont_inherit=True).co_consts
            if isinstance(c, CodeType) and c.co_name == "match")
        require(isinstance(self.function, FunctionType) and self.function.__globals__ is module.__dict__
            and scope.code_identity(code) == scope.code_identity(self.function.__code__), "Original dispatch differs from source")
        self.code = self.function.__code__; self.defaults = repr(self.function.__defaults__)
        self.keywords = repr(self.function.__kwdefaults__); self.check()

    def check(self, *, source=False):
        require(sys.modules.get(self.module.__name__) is self.module and self.module.match is self.function
            and self.function.__code__ is self.code and self.function.__globals__ is self.module.__dict__
            and repr(self.function.__defaults__) == self.defaults and repr(self.function.__kwdefaults__) == self.keywords
            and self.module._device is self.context and self.context.get() is None
            and os.environ.get("KINECT_CUDA_REGISTRATION") == "cpu"
            and Path(self.module.__file__).resolve() == self.path and (not source or scope.sha(self.path) == self.digest),
            "Only unchanged original disabled CUDA dispatch permits CPU census")


class LoadedCensusGuard:
    """Small source/loaded-code boundary for this output-only observer."""
    def __init__(self):
        self.module = sys.modules[__name__]; self.path = Path(__file__).resolve(); self.digest = scope.sha(self.path)
        compiled = compile(self.path.read_text(encoding="utf-8"), str(self.path), "exec", dont_inherit=True)
        self.records = []
        names = {"matrix_values", "max_delta", "cloud_bucket", "result_delta", "cluster_rows", "CpuCallObserver", "OriginalCpuRoute", "run", "capture_preflight"}
        for code in compiled.co_consts:
            if not isinstance(code, CodeType) or code.co_name not in names: continue
            owner = getattr(self.module, code.co_name)
            pairs = [(self.module, code.co_name, owner, code)] if isinstance(owner, FunctionType) else [
                (owner, c.co_name, getattr(owner, c.co_name), c) for c in code.co_consts if isinstance(c, CodeType)]
            for parent, name, fn, expected in pairs:
                require(isinstance(fn, FunctionType) and fn.__globals__ is self.module.__dict__
                    and scope.code_identity(fn.__code__) == scope.code_identity(expected), "Loaded census evidence differs from source")
                aliases = tuple((key, self.module.__dict__[key]) for key in fn.__code__.co_names if key in self.module.__dict__)
                self.records.append((parent, name, fn, fn.__code__, repr(fn.__defaults__), repr(fn.__kwdefaults__), aliases))
        self.classes = {name: getattr(self.module, name) for name in ("CpuCallObserver", "OriginalCpuRoute")}
        self.check()

    def check(self, *, source=False):
        require(sys.modules.get(self.module.__name__) is self.module and Path(self.module.__file__).resolve() == self.path
            and all(getattr(self.module, name) is value for name, value in self.classes.items())
            and (not source or scope.sha(self.path) == self.digest), "Census loaded/source owner changed")
        for owner, name, fn, code, defaults, keywords, aliases in self.records:
            require(getattr(owner, name) is fn and fn.__code__ is code and fn.__globals__ is self.module.__dict__
                and repr(fn.__defaults__) == defaults and repr(fn.__kwdefaults__) == keywords
                and all(self.module.__dict__.get(key) is value for key, value in aliases),
                "Output-only census callback/cluster code/default owner changed")


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args()
    args.captures = [p.resolve() for p in args.captures]; args.output = args.output.resolve()
    if (not args.run_allocated or not 1 <= len(args.captures) <= 2 or len(set(args.captures)) != len(args.captures)
            or args.output.exists() or not args.output.is_relative_to(ROOT/"benchmark-output")):
        parser.error("Exclusive CPU allocation, one/two distinct local captures and fresh private output required")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0: parser.error("Stop idle field server before offline CPU census")
    return args


def capture_preflight(path, file_sha, actual_source):
    before = file_sha(path)
    capture = json.loads(path.read_text(encoding="utf-8"))
    require(file_sha(path) == before, "Capture changed while reading its metadata")
    require(capture.get("kind") == "gpu-icp-current-field-pair-fixture-v2" and capture.get("status") == "passed"
        and capture.get("cleanup_passed") is True and not capture.get("failure") and not capture.get("cleanup_failures")
        and capture.get("pose_seeds_used_for_live_tracking") is False and actual_source == CURRENT
        and capture.get("source_sha256") == actual_source and capture.get("source_sha256_after") == actual_source
        and capture.get("artifacts_sha256") == capture.get("artifacts_sha256_after"), "Closed current fresh capture required")
    fixed = {str(path): before}
    for name in ("fixture", "runtime_snapshot", "session", "profile", "native_extension"):
        record = capture[name]; actual = str(Path(record["path"]).resolve(strict=True))
        require(file_sha(actual) == record["sha256"], "Captured raw/fixture/profile/native resource changed")
        fixed[actual] = record["sha256"]
    require(Path(capture["fixture"]["path"]).resolve().is_relative_to(ROOT/"benchmark-output"), "Only locally owned capture pickle")
    for name, digest in capture["artifacts_sha256"].items():
        require(file_sha(ROOT/name) == digest, "Capture producer/preparation source changed")
    return capture, fixed


def run(args):
    began = time.perf_counter(); report = {"kind": KIND, "status": "running", "failure": None, "captures": [],
        "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "skipped_icp_calls": 0,
        "timing_authority": False, "whole_finish_authority": False,
        "scope": "All captured original-ranked pairs/all genuine proposals; original complete CPU verifier and ordered ambiguity. Offline seed census only, no ICP skipping, numerical changes or cache approximation.",
        "clustering_policy": "Exact directed point/normal/color value buckets; ordered first representative per threshold; max absolute difference over all16 unrounded seed entries. No transitive closure.",
        "timers": "Observed CPU work includes census metadata/input checks and original gate observations; no counterfactual saved time or speed claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save(); primary = None; threads = None; guard = None; env = {k: os.environ.get(k) for k in ENVIRONMENT}
    try:
        os.environ.update(ENVIRONMENT)
        from scripts.research.gpu_icp_experiment_capture import sha as file_sha, source_hash, fixture_records, profile_contract
        from scripts.research.benchmark_parallel_fragments import unpack_fragment, verification_dependencies
        from scripts.research.microbatch_bridge_driver import train_inventory
        for name, digest in HELD_FILES.items(): require(file_sha(ROOT/name) == digest, "Held original observer sources changed")
        captures, fixed, artifact_names = [], {}, set(OWN_FILES+tuple(HELD_FILES))
        for path in args.captures:
            capture, resources = capture_preflight(path, file_sha, source_hash())
            captures.append((path, capture)); fixed.update(resources); artifact_names.update(capture["artifacts_sha256"])
            profile = json.loads(Path(capture["profile"]["path"]).read_text(encoding="utf-8"))
            profile_contract(profile, capture["session"]["path"], source_hash(), capture["session"]["sha256"])
        import numpy as np
        import open3d as o3d
        import cv2
        from scanner_server import fragments as module
        from scanner_server import cuda_registration
        threads = (cv2.getNumThreads(), o3d.utility.get_max_threads())
        cv2.setNumThreads(20); o3d.utility.set_max_threads(20)
        guard = scope.OriginalVerifierGuard(module)
        route = OriginalCpuRoute(cuda_registration)
        observer_guard = LoadedCensusGuard()
        numerical = importlib.import_module("numpy._core._multiarray_umath")
        backend = sys.modules[o3d.geometry.PointCloud.__module__.split(".geometry")[0]]
        modules = (np, o3d, cv2, numerical, backend, module, scope, cuda_registration)
        owners = {m.__name__: (m, str(Path(m.__file__).resolve())) for m in modules}
        binaries = {str(Path(m.__file__).resolve()): file_sha(m.__file__) for m in (numerical, backend, cv2)}
        for path in Path(cv2.__file__).resolve().parent.rglob("cv2*.pyd"): binaries[str(path.resolve())] = file_sha(path)
        fixed.update(binaries); executable = str(Path(sys.executable).resolve()); fixed[executable] = file_sha(executable)
        def runtime():
            return {"executable": str(Path(sys.executable).resolve()), "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__},
                "binary_sha256": binaries, "numpy_build_sha256": hashlib.sha256(scope.canonical(getattr(np.__config__, "CONFIG", {})).encode()).hexdigest(),
                "numpy_cpu_features": {k: bool(v) for k, v in getattr(numerical, "__cpu_features__", {}).items()},
                "thread_policy": {"opencv": cv2.getNumThreads(), "open3d": o3d.utility.get_max_threads(), "omp": os.environ.get("OMP_NUM_THREADS")},
                "environment": {k: os.environ.get(k) for k in ENVIRONMENT}, "registration": "original legacy CPU"}
        def binding():
            return {"source_sha256": source_hash(), "artifacts_sha256": {n: file_sha(ROOT/n) for n in sorted(artifact_names)},
                "fixed_files": {p: file_sha(p) for p in fixed}, "runtime": runtime(), "verification_dependencies": verification_dependencies()}
        report["binding"] = binding()
        def boundary():
            guard.check(source=True); route.check(source=True); observer_guard.check()
            require(runtime() == report["binding"]["runtime"], "Actual census numerical/thread/environment policy changed")
            require(all(sys.modules.get(k) is m and str(Path(m.__file__).resolve()) == path for k, (m, path) in owners.items()), "Loaded census numerical/helper owner changed")
        all_calls = []
        for capture_index, (path, capture) in enumerate(captures):
            require(runtime()["versions"] == capture["versions"], "Actual versions differ from capture")
            with Path(capture["fixture"]["path"]).open("rb") as stream: fixture = pickle.load(stream)
            require(fixture_records(fixture) == capture["tasks"], "Materialized capture task arrays/seeds changed")
            require(fixture["metadata"]["local_pose_source"] == "measured Finish fragment report; no archived ZIP poses", "No archived Live seeds")
            receipt = {"capture": str(path), "capture_sha256": fixed[str(path)], "tasks": []}; report["captures"].append(receipt)
            for position, task in enumerate(fixture["tasks"]):
                require(task["position"] == position and 1 <= len(task["proposals"]) <= 6, "All genuine ordered proposals required")
                a, b = task["pair"]; source, target = (unpack_fragment(fixture["fragments"][i]) for i in (a, b))
                for key, fragment in ((a, source), (b, target)):
                    for original, view in zip(fixture["fragments"][key]["keys"], fragment.keys):
                        for other in source.keys+target.keys:
                            matches = original["matches"].get(other.index)
                            if matches is not None: module._cache_matches(view, other, matches.copy())
                before = train_inventory(np, source, target)
                seeds = [scope.descriptor(np, s) for s in task["proposals"]]
                pair = {"position": position, "pair": [a, b], "proposals": []}; receipt["tasks"].append(pair)
                values = []; started = time.perf_counter()
                for proposal_index, seed in enumerate(task["proposals"]):
                    boundary(); row = {"proposal_index": proposal_index, "calls": [], "gates": [], "complete": False}
                    pair["proposals"].append(row)
                    observer = CpuCallObserver(np, guard, row, {"capture_index": capture_index, "task": position, "pair": [a, b], "proposal_index": proposal_index}, route.check)
                    private_source, private_target = scope.private_fragments(source, target)
                    verifier = scope.PrivateBridgeScope(module, observer, trace=row["gates"], guard=guard)
                    value = verifier.verify(private_source, private_target, seed, fixture["camera"])
                    verifier.healthy(); require(observer.failure is None and all(c["complete"] for c in row["calls"]), "Original dynamic call suffix failed")
                    row["result"] = scope.semantic(np, value); row["complete"] = True
                    values.append(value); all_calls.extend(row["calls"]); boundary(); save()
                    print(f"CPU census capture{capture_index} pair{a}/{b} proposal{proposal_index}: {len(row['calls'])} original calls", flush=True)
                pair["original_pair_verdict"] = scope.original_pair_verdict(module, values)
                pair["observed_whole_proposals_wall_s"] = time.perf_counter()-started
                pair["input_bytes_unchanged"] = before == train_inventory(np, source, target)
                pair["seed_bytes_unchanged"] = seeds == [scope.descriptor(np, s) for s in task["proposals"]]
                require(pair["input_bytes_unchanged"] and pair["seed_bytes_unchanged"], "Original pair arrays/seeds changed")
        boundary(); report["call_count"] = len(all_calls)
        report["clustering"] = [cluster_rows(all_calls, t) for t in THRESHOLDS]
        report["ordered_calls_sha256"] = hashlib.sha256(scope.canonical(all_calls).encode()).hexdigest()
        report["original_cpu_results_returned_unchanged"] = True
    except BaseException as error:
        primary = error; report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        failures = []
        def clean(label, action):
            try: return action()
            except BaseException as error:
                failures.append({"action": label, "type": type(error).__name__, "message": str(error)}); return None
        if guard is not None: clean("original loaded/source owners", lambda: guard.check(source=True))
        if "binding" in report:
            clean("all actual census owner closure", boundary); report["binding_after"] = clean("source/raw/runtime/binary closure", binding)
        if threads is not None:
            clean("OpenCV restore", lambda: cv2.setNumThreads(threads[0])); clean("Open3D restore", lambda: o3d.utility.set_max_threads(threads[1]))
            if clean("actual restored threads", lambda: (cv2.getNumThreads(), o3d.utility.get_max_threads())) != threads:
                failures.append({"action": "threads restored", "message": "Original values differ"})
        for key, value in env.items(): clean("environment "+key, lambda key=key, value=value: os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value))
        report["cleanup_failures"] = failures; report["cleanup_passed"] = not failures and all(os.environ.get(k) == v for k, v in env.items())
        report["status"] = "passed" if (primary is None and report["cleanup_passed"] and report.get("binding_after") == report.get("binding")
            and report.get("original_cpu_results_returned_unchanged") is True and report.get("call_count", 0) > 0) else "failed"
        report["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(); report["producer_wall_s"] = time.perf_counter()-began
        try: save()
        except BaseException as error:
            if primary is not None: raise primary from error
            raise
    if report["status"] != "passed": raise RuntimeError("CPU seed census failed; private diagnostics retained") from primary


if __name__ == "__main__": run(parse())
