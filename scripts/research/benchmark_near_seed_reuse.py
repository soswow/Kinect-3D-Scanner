"""Fresh complete-proposal CPU near-seed reuse audit, then matched timing.

This explicitly changes the registration method. No old CUDA/Finish token is
accepted. Every audit hit receives a fresh original CPU same-input shadow;
all proposals/gates/ambiguity are compared with a separate original CPU run.
Timing requires that new closed audit and its exact own ordered trajectory.
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
import struct
import sys
import time
import traceback
from types import CodeType, FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import benchmark_icp_seed_reuse_census as census
from scripts.research import near_seed_icp as method
scope = census.scope
KIND = "complete-original-proposals-near-seed-cpu-method-audit-v1"
TIMING_KIND = "complete-original-proposals-near-seed-cpu-method-timing-v1"
OWN_FILES = ("scripts/research/near_seed_icp.py", "scripts/research/benchmark_near_seed_reuse.py", "tests/test_near_seed_reuse.py")
CONFIGURATION = {"threshold": 1e-10, "max_entries": 256, "max_clouds": 256, "max_bytes": 64*1024**2}


def require(value, message): census.require(value, message)


class LoadedMethodGuard:
    """Cold source comparison, then cheap code/default/global-owner checks."""
    def __init__(self):
        self.modules, self.functions, self.classes = [], [], []
        for module in (method, sys.modules[__name__], census, scope):
            path = Path(module.__file__).resolve(); self.modules.append((module, path))
            compiled = compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
            for code in compiled.co_consts:
                if not isinstance(code, CodeType): continue
                owner = getattr(module, code.co_name)
                if isinstance(owner, type): self.classes.append((module, code.co_name, owner))
                pairs = [(module, code.co_name, owner, code)] if isinstance(owner, FunctionType) else [
                    (owner, c.co_name, getattr(owner, c.co_name), c) for c in code.co_consts if isinstance(c, CodeType)]
                for parent, name, value, expected in pairs:
                    fn = value.fget if isinstance(value, property) else value
                    require(isinstance(fn, FunctionType) and fn.__globals__ is module.__dict__
                        and scope.code_identity(fn.__code__) == scope.code_identity(expected), "Loaded method/experiment differs from actual source")
                    aliases = tuple((key, module.__dict__[key]) for key in fn.__code__.co_names if key in module.__dict__)
                    self.functions.append((parent, name, value, fn, fn.__code__, repr(fn.__defaults__), repr(fn.__kwdefaults__), module, aliases))
        self.check()

    def check(self):
        require(all(sys.modules.get(m.__name__) is m and Path(m.__file__).resolve() == path for m, path in self.modules)
            and all(getattr(m, name) is cls for m, name, cls in self.classes), "Method/experiment class/module owner changed")
        for owner, name, value, fn, code, defaults, keywords, module, aliases in self.functions:
            require(getattr(owner, name) is value and fn.__code__ is code and fn.__globals__ is module.__dict__
                and repr(fn.__defaults__) == defaults and repr(fn.__kwdefaults__) == keywords
                and all(module.__dict__.get(key) is v for key, v in aliases), "Method/experiment code/default/global owner changed")


def result_reference(result):
    return (scope.canonical(result["pose"]), result["fitness"].hex(), result["inlier_rmse"].hex(),
        scope.canonical(result["correspondence_mapping"]))


def class_reference(receipt):
    return tuple(receipt[name] for name in ("kind", "class_id", "cloud_value_sha256", "representative_seed_sha256",
        "seed_max_abs_delta", "threshold", "evictions"))


def call_reference(call):
    return (scope.canonical(call["context"]), call["call_index"], call["caller"], scope.canonical(call["input_binding"]),
        result_reference(call["result"]), class_reference(call["method_receipt"]))


def phase_calls(phase):
    return [call for pair in phase["pairs"] for proposal in pair["proposals"] for call in proposal["calls"]]


def validate_scope(phase, captures):
    expected = [(ci, task) for ci, capture in enumerate(captures) for task in capture["tasks"]]
    require(phase.get("complete") is True and len(phase["pairs"]) == len(expected), "Entire captured pair scope required")
    for pair, (ci, task) in zip(phase["pairs"], expected):
        require(pair["identity"] == [ci, task["position"]]+task["pair"] and len(pair["proposals"]) == task["proposal_count"], "Original captured pair/all genuine proposal order changed")
        require(pair["original_seed_descriptors"] == [{k: s[k] for k in ("dtype", "shape", "nbytes", "sha256")} for s in task["seeds"]], "Original genuine seeds changed")
        for pi, proposal in enumerate(pair["proposals"]):
            require(proposal["proposal_index"] == pi and proposal.get("complete") is True, "Original genuine proposal suffix required")
            for call_index, call in enumerate(proposal["calls"]):
                require(call["call_index"] == call_index and call["context"] == {"capture_index": ci, "task": task["position"], "pair": task["pair"], "proposal_index": pi}, "Ordered actual dynamic call context changed")


def compare_phases(native, candidate):
    require(len(native["pairs"]) == len(candidate["pairs"]), "Every original pair must be exhausted")
    rows = []
    for a, b in zip(native["pairs"], candidate["pairs"]):
        differences = []
        require(a["identity"] == b["identity"] and len(a["proposals"]) == len(b["proposals"]), "Original pair/proposal order changed")
        for old, new in zip(a["proposals"], b["proposals"]):
            require(len(old["calls"]) == len(new["calls"]), "Original complete proposal dynamic call count changed")
            scope.compare_evidence(old["result"], new["result"], path="result", differences=differences)
            left = [{k: v for k, v in r.items() if k != "wall_s_inclusive"} for r in old["gates"]]
            right = [{k: v for k, v in r.items() if k != "wall_s_inclusive"} for r in new["gates"]]
            if len(left) != len(right): differences.append({"path": "gates", "reason": "length"})
            else:
                for i, (x, y) in enumerate(zip(left, right)):
                    scope.compare_evidence(x, y, tolerance=1e-5 if x["name"] ==
                        "REG.get_information_matrix_from_point_clouds" else 1e-8, path=f"gates[{i}]", differences=differences)
        scope.compare_evidence(a["original_pair_verdict"], b["original_pair_verdict"], path="ambiguity", differences=differences)
        rows.append({"identity": a["identity"], "passed": not differences, "difference_count": len(differences), "differences": differences[:100]})
    return {"passed": all(row["passed"] for row in rows), "pairs": rows,
        "gate_numeric_policy": "Original discrete/canonical membership exact; original full-proposal small matrices1e-8, information1e-5. Actual hit CPU shadows separately matrix1e-10/metrics1e-12."}


def validate_hit_shadow(call):
    shadow = call["method_receipt"]["shadow"]
    require(shadow.get("collected") is True and shadow.get("passed") is True
        and shadow.get("correspondence_ids_equal") is True and shadow["candidate"] == call["result"],
        "Fresh same-input original CPU hit evidence required")
    a, b = shadow["native"], shadow["candidate"]
    matrices = []
    for value in (a, b):
        rows = value["transformation"]
        require(len(rows) == 4 and all(len(row) == 4 for row in rows), "Full original hit-shadow pose required")
        flat = tuple(v for row in rows for v in row)
        require(all(type(v) in (int, float) and math.isfinite(v) for v in flat)
            and value["pose"] == {"dtype": "<f8", "shape": [4, 4], "nbytes": 128,
                "sha256": hashlib.sha256(struct.pack("<16d", *flat)).hexdigest()}
            and type(value["fitness"]) in (int, float) and math.isfinite(value["fitness"]) and 0 <= value["fitness"] <= 1
            and type(value["inlier_rmse"]) in (int, float) and math.isfinite(value["inlier_rmse"]) and value["inlier_rmse"] >= 0,
            "Finite original hit-shadow result bits required")
        matrices.append(flat)
    actual = (max(abs(p-q) for p, q in zip(*matrices)), abs(a["fitness"]-b["fitness"]), abs(a["inlier_rmse"]-b["inlier_rmse"]))
    keys = ("transform_max_abs_delta", "fitness_abs_delta", "rmse_abs_delta")
    require(a["correspondence_mapping"] == b["correspondence_mapping"] and all(
        type(shadow[k]) in (int, float) and math.isfinite(shadow[k]) and 0 <= shadow[k] == delta <= limit
        for k, delta, limit in zip(keys, actual, (1e-10, 1e-12, 1e-12))),
        "Actual hit evidence exceeded own tight method quality bounds")


def load_audit(path, binding, captures):
    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest(); report = json.loads(raw.decode("utf-8"))
    require(report.get("kind") == KIND and report.get("mode") == "audit" and report.get("status") == "passed"
        and report.get("failure") is None and report.get("cleanup_failures") == [] and report.get("cleanup_passed") is True
        and report.get("binding") == report.get("binding_after") == binding and report.get("captures") == captures,
        "Only fresh closed current-source own-method audit may omit hit CPU shadows")
    require(len(report.get("rounds", [])) == 1 and report["rounds"][0]["quality"]["passed"] is True, "Complete native-versus-method proposal audit required")
    native = report["rounds"][0]["native"]; candidate = report["rounds"][0]["cache"]
    validate_scope(native, captures); validate_scope(candidate, captures)
    require(compare_phases(native, candidate) == report["rounds"][0]["quality"], "Own audit full original gate/ambiguity closure disagrees")
    stats = candidate["cache"]["statistics"]; calls = phase_calls(candidate)
    require(candidate["cache"]["configuration"] == binding["configuration"]
        and candidate["cache"]["policy"] == method.POLICY
        and candidate["cache"]["owned_payload_bytes"] == 0
        and all(type(v) is int and v >= 0 for v in stats.values()), "Effective closed audit cache configuration/counts changed")
    kinds = [call["method_receipt"]["kind"] for call in calls]
    require(all(kind in ("hit", "miss") for kind in kinds)
        and kinds.count("hit") == stats["hits"] and kinds.count("miss") == stats["misses"]
        and sum(call["method_receipt"]["kind"] == "miss" and call["method_receipt"]["class_id"] is None for call in calls) == stats["unretained_misses"],
        "Actual hit/miss receipts disagree with audit statistics")
    require(candidate["cache"]["audit"] is True and candidate["cache"]["closed"] is True and candidate["cache"]["failure"] is None
        and stats["calls"] == len(calls) and stats["hits"] > 0 and stats["audited_hits"] == stats["hits"]
        and stats["hits"]+stats["misses"] == stats["calls"] == stats["original_cpu_calls"], "Every actual audit hit must have fresh original CPU shadow")
    for phase in (native, candidate):
        require(phase.get("cleanup_passed") is True and phase.get("cleanup_failures") == [] and phase.get("failure") is None
            and phase["call_count"] == len(phase_calls(phase)), "Complete native/method phase cleanup required")
        for call in phase_calls(phase):
            require(call.get("complete") is True and call.get("input_bytes_unchanged") is True and not call.get("failure"), "Original actual call closure required")
    for pair in native["pairs"]+candidate["pairs"]:
        require(pair.get("complete") is True and pair.get("input_bytes_unchanged") is True and pair.get("seed_bytes_unchanged") is True,
            "Original whole pair closure required")
        require(all(p.get("complete") is True and all(g.get("complete") is True for g in p["gates"]) for p in pair["proposals"]), "Original proposal/gate suffix incomplete")
    for call in calls:
        receipt = call["method_receipt"]
        require(receipt["threshold"] == binding["configuration"]["threshold"]
            and type(receipt["seed_max_abs_delta"]) in (int, float) and math.isfinite(receipt["seed_max_abs_delta"])
            and 0 <= receipt["seed_max_abs_delta"] <= receipt["threshold"]
            and (receipt["class_id"] is None and receipt["kind"] == "miss" or type(receipt["class_id"]) is int and receipt["class_id"] >= 0)
            and receipt["cloud_value_sha256"] == hashlib.sha256(scope.canonical({k: call["input_binding"][k] for k in ("source", "target")}).encode()).hexdigest(),
            "Actual first-representative receipt configuration/input disagrees")
        if call["method_receipt"]["kind"] == "hit":
            validate_hit_shadow(call)
        else:
            require(receipt["shadow"].get("collected") is False and receipt["seed_max_abs_delta"] == 0
                and receipt["representative_seed_sha256"] == call["input_binding"]["seed"]["sha256"], "Audit miss must execute actual original seed")
    require(path.read_bytes() == raw, "Own audit changed while loading")
    # Immutable scalar tuples, no repeated whole-proof JSON or I/O per call.
    return digest, tuple(call_reference(call) for call in calls), candidate


class Observer:
    def __init__(self, np, guard, route, row, context, cache, expected, ordinal):
        self.row, self.cache, self.expected, self.ordinal, self.failure = row, cache, expected, ordinal, None
        original = SimpleNamespace(check=guard.check, cpu_match=guard.cpu_match if cache is None else cache.match)
        self.delegate = census.CpuCallObserver(np, original, row, context, route.check)
        self.np = np

    def __call__(self, source, target, initial):
        caller = sys._getframe(2).f_code.co_name; index = self.ordinal[0]
        try:
            if self.expected is not None:
                require(index < len(self.expected), "Own timed dynamic call suffix exceeded audit")
                ref = self.expected[index]
                require((scope.canonical(self.delegate.context), len(self.row["calls"]), caller,
                    scope.canonical(scope.input_binding(self.np, source, target, initial))) == ref[:4],
                    "Own timed actual source/target/seed/context changed before CPU work")
            result = self.delegate(source, target, initial)
            call = self.row["calls"][-1]; call["caller"] = caller
            if self.cache is not None:
                call["method_receipt"] = dict(self.cache.last_receipt)
                if self.expected is not None: require(call_reference(call) == self.expected[index], "Own timed result/canonical IDs/cache representative trajectory changed")
            self.ordinal[0] += 1
            return result
        except BaseException as error:
            self.failure = error
            if self.row["calls"]: self.row["calls"][-1]["complete"] = False
            raise scope.BridgeFailure("Near-seed experiment actual call/reference failed") from error


def execute_phase(np, module, guard, route, fixtures, cache, expected, boundary, phase):
    from scripts.research.benchmark_parallel_fragments import unpack_fragment
    from scripts.research.microbatch_bridge_driver import train_inventory
    started = time.perf_counter(); ordinal = [0]; primary = None
    try:
        for capture_index, fixture in enumerate(fixtures):
            for position, task in enumerate(fixture["tasks"]):
                pair_started = time.perf_counter(); a, b = task["pair"]
                pair = {"identity": [capture_index, position, a, b], "proposals": [], "complete": False}; phase["pairs"].append(pair)
                try:
                    source, target = (unpack_fragment(fixture["fragments"][i]) for i in (a, b))
                    for key, fragment in ((a, source), (b, target)):
                        for original, view in zip(fixture["fragments"][key]["keys"], fragment.keys):
                            for other in source.keys+target.keys:
                                matches = original["matches"].get(other.index)
                                if matches is not None: module._cache_matches(view, other, matches.copy())
                    before = train_inventory(np, source, target); seeds = [scope.descriptor(np, s) for s in task["proposals"]]
                    pair["original_seed_descriptors"] = seeds
                    results = []
                    for proposal_index, seed in enumerate(task["proposals"]):
                        boundary(); row = {"proposal_index": proposal_index, "calls": [], "gates": [], "complete": False}; pair["proposals"].append(row)
                        observer = Observer(np, guard, route, row, {"capture_index": capture_index, "task": position,
                            "pair": [a, b], "proposal_index": proposal_index}, cache, expected, ordinal)
                        private_source, private_target = scope.private_fragments(source, target)
                        verifier = scope.PrivateBridgeScope(module, observer, trace=row["gates"], guard=guard)
                        value = verifier.verify(private_source, private_target, seed, fixture["camera"])
                        verifier.healthy(); require(observer.failure is None and all(c["complete"] for c in row["calls"]), "Actual original call suffix failed")
                        row["result"] = scope.semantic(np, value); row["complete"] = True; results.append(value)
                    pair["original_pair_verdict"] = scope.original_pair_verdict(module, results)
                    pair["input_bytes_unchanged"] = before == train_inventory(np, source, target)
                    pair["seed_bytes_unchanged"] = seeds == [scope.descriptor(np, s) for s in task["proposals"]]
                    require(pair["input_bytes_unchanged"] and pair["seed_bytes_unchanged"], "Actual complete original pair inputs changed")
                    boundary()
                    if cache is not None: cache.clear_pair()
                    pair["complete"] = True
                finally:
                    pair["inclusive_pair_wall_s"] = time.perf_counter()-pair_started
                print(f"{phase['method']} complete pair{a}/{b}: {sum(len(r['calls']) for r in pair['proposals'])} calls", flush=True)
        if expected is not None: require(ordinal[0] == len(expected), "Own timed dynamic call suffix missing")
        phase["complete"] = True
    except BaseException as error:
        primary = error
        raise
    finally:
        failures = []
        for label, action in (() if cache is None else (("cache close", cache.close), ("cache report", lambda: phase.__setitem__("cache", cache.report())))):
            try: action()
            except BaseException as error: failures.append((label, error))
        phase["whole_phase_wall_s"] = time.perf_counter()-started
        phase["call_count"] = ordinal[0]
        phase["cleanup_failures"] = [{"action": label, "type": type(error).__name__, "message": str(error)} for label, error in failures]
        phase["cleanup_passed"] = not failures
        phase["failure"] = None if primary is None else {"type": type(primary).__name__, "message": str(primary)}
        if primary is not None or failures: phase["complete"] = False
        if failures:
            if primary is not None: raise primary from failures[0][1]
            raise scope.BridgeFailure("Near-seed phase cleanup failed") from failures[0][1]


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("audit", "timing"), default="audit")
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--threshold", type=float, choices=method.THRESHOLDS, default=1e-10)
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args(); args.output = args.output.resolve(); args.captures = [p.resolve() for p in args.captures]
    if (not args.run_allocated or len(args.captures) != 2 or len(set(args.captures)) != 2 or args.output.exists()
        or not args.output.is_relative_to(ROOT/"benchmark-output") or args.mode == "timing" and args.proof is None):
        parser.error("Exclusive CPU slot, both genuine captures, fresh private output and own timing proof required")
    if args.proof is not None: args.proof = args.proof.resolve()
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0: parser.error("Stop idle field server first")
    return args


def run(args):
    began = time.perf_counter(); report = {"kind": KIND if args.mode == "audit" else TIMING_KIND, "mode": args.mode,
        "status": "running", "failure": None, "cleanup_failures": [], "captures": [], "rounds": [],
        "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "whole_finish_authority": False,
        "method_change": "CPU near-seed result reuse, actual seeds unrounded, first immutable representative; no CUDA changes",
        "scope": "Both fresh captures, all nine original-ranked pairs/all30 genuine complete proposals. No adaptive frontier/optimizer/Live/fusion/mesh authority.",
        "timers": "Primary cold_whole_phase_wall_s includes cache constructor, cloud reconstruction, full input hashing, lookup/result copying, original CPU solves/gates, exact own timing references including terminal gate/verdict validation, metadata and cache cleanup. Nested whole_phase_wall_s starts after cache construction; disjoint inclusive_pair_wall_s spans each pair's unpack/setup through final input/seed/boundary checks and cold cache reset. Audit adds fresh original hit shadows; no audit subtraction. Cross-method quality and raw/fixture/library provenance/imports are separate producer validation/setup/closure, not a whole Finish claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save(); primary = None; threads = None; env = {k: os.environ.get(k) for k in census.ENVIRONMENT}; guard = None
    try:
        os.environ.update(census.ENVIRONMENT)
        from scripts.research.gpu_icp_experiment_capture import source_hash, sha, fixture_records, profile_contract
        from scripts.research.benchmark_parallel_fragments import verification_dependencies
        captures, fixtures, fixed, names = [], [], {}, set(OWN_FILES+census.OWN_FILES+tuple(census.HELD_FILES))
        for path in args.captures:
            capture, resources = census.capture_preflight(path, sha, source_hash()); fixed.update(resources); names.update(capture["artifacts_sha256"])
            profile_contract(json.loads(Path(capture["profile"]["path"]).read_text(encoding="utf-8")), capture["session"]["path"], source_hash(), capture["session"]["sha256"])
            captures.append(capture); report["captures"].append({"path": str(path), "sha256": fixed[str(path)], "tasks": capture["tasks"]})
        import numpy as np
        import open3d as o3d
        import cv2
        from scanner_server import fragments as module, cuda_registration
        threads = (cv2.getNumThreads(), o3d.utility.get_max_threads()); cv2.setNumThreads(20); o3d.utility.set_max_threads(20)
        guard = scope.OriginalVerifierGuard(module); route = census.OriginalCpuRoute(cuda_registration)
        method_guard = LoadedMethodGuard()
        backend = sys.modules[o3d.geometry.PointCloud.__module__.split(".geometry")[0]]; numerical = importlib.import_module("numpy._core._multiarray_umath")
        owners = tuple((m, m.__name__, str(Path(m.__file__).resolve())) for m in (np, o3d, cv2, backend, numerical, module, cuda_registration, method, census, scope))
        for m, _, path in owners: fixed[path] = sha(path)
        for path in Path(cv2.__file__).resolve().parent.rglob("cv2*.pyd"): fixed[str(path.resolve())] = sha(path)
        fixed[str(Path(sys.executable).resolve())] = sha(sys.executable)
        configuration = dict(CONFIGURATION, threshold=args.threshold)
        configuration_snapshot = tuple(sorted(configuration.items()))
        def runtime():
            return {"versions": {"python": sys.version.split()[0], "numpy": np.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__},
                "executable": str(Path(sys.executable).resolve()), "thread_policy": [cv2.getNumThreads(), o3d.utility.get_max_threads(), os.environ.get("OMP_NUM_THREADS")],
                "environment": {k: os.environ.get(k) for k in census.ENVIRONMENT},
                "numpy_build_sha256": hashlib.sha256(scope.canonical(getattr(np.__config__, "CONFIG", {})).encode()).hexdigest(),
                "numpy_cpu_features": {k: bool(v) for k, v in getattr(numerical, "__cpu_features__", {}).items()}}
        def binding():
            return {"source_sha256": source_hash(), "artifacts_sha256": {n: sha(ROOT/n) for n in sorted(names)}, "fixed_files": {p: sha(p) for p in fixed},
                "runtime": runtime(), "configuration": dict(configuration), "method_policy": method.POLICY, "verification_dependencies": verification_dependencies()}
        report["binding"] = binding()
        def boundary():
            guard.check(source=True); route.check(source=True); method_guard.check()
            require(tuple(sorted(configuration.items())) == configuration_snapshot, "Declared effective method configuration changed")
            require(runtime() == report["binding"]["runtime"] and all(sys.modules.get(n) is m and str(Path(m.__file__).resolve()) == p for m, n, p in owners), "Actual numerical/CPU route/owner policy changed")
        for capture in captures:
            require(capture["versions"] == runtime()["versions"], "Actual numerical versions differ from capture")
            with Path(capture["fixture"]["path"]).open("rb") as stream: fixture = pickle.load(stream)
            require(fixture_records(fixture) == capture["tasks"] and fixture["metadata"]["local_pose_source"] == "measured Finish fragment report; no archived ZIP poses", "Own current prepared inputs changed")
            fixtures.append(fixture)
        require([len(f["tasks"]) for f in fixtures] == [1, 8] and sum(len(t["proposals"]) for f in fixtures for t in f["tasks"]) == 30, "Exact current C6one+C7eight/30 genuine proposal scope required")
        expected = reference = proof_sha = None
        if args.mode == "timing":
            proof_sha, expected, reference = load_audit(args.proof, report["binding"], report["captures"])
            report["audit_proof"] = {"path": str(args.proof), "sha256": proof_sha}
        report["producer_setup_wall_s"] = time.perf_counter()-began
        for repetition in range(1 if args.mode == "audit" else 3):
            row = {"repetition": repetition, "order": ["native", "cache"] if repetition % 2 == 0 else ["cache", "native"]}; report["rounds"].append(row)
            for mode in row["order"]:
                phase = {"method": mode, "pairs": [], "complete": False}; row[mode] = phase
                start = time.perf_counter()
                index = [0]
                def authorize_actual_input(payload):
                    require(expected is not None and index[0] < len(expected)
                        and scope.canonical(payload) == expected[index[0]][3], "Only actual same-input own audited timing can omit original hit shadows")
                    index[0] += 1
                try:
                    cache = method.NearSeedMatcher(np, guard, route.check, audit=args.mode == "audit",
                        timing_authorizer=authorize_actual_input if args.mode == "timing" else None, **configuration) if mode == "cache" else None
                    execute_phase(np, module, guard, route, fixtures, cache, expected if mode == "cache" else None, boundary, phase)
                    if mode == "cache" and reference is not None: require(compare_phases(reference, phase)["passed"], "Timed own cache complete gates/verdict changed")
                finally: phase["cold_whole_phase_wall_s"] = time.perf_counter()-start
                save()
            row["quality"] = compare_phases(row["native"], row["cache"])
            require(row["quality"]["passed"], "Near-seed method changed original complete proposal/gate/ambiguity quality")
            if args.mode == "audit": require(row["cache"]["cache"]["statistics"]["hits"] > 0, "No actual reuse coverage")
        boundary()
    except BaseException as error:
        primary = error; report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        failures = []
        def clean(label, action):
            try: return action()
            except BaseException as error: failures.append({"action": label, "type": type(error).__name__, "message": str(error)}); return None
        if "binding" in report:
            clean("original loaded/current owners", boundary); report["binding_after"] = clean("source/raw/binary/runtime closure", binding)
        if args.mode == "timing" and report.get("audit_proof"):
            actual = clean("own audit terminal hash", lambda: sha(args.proof))
            if actual != proof_sha: failures.append({"action": "own audit changed", "message": "Audit SHA differs at closure"})
        if threads is not None:
            clean("OpenCV restore", lambda: cv2.setNumThreads(threads[0])); clean("Open3D restore", lambda: o3d.utility.set_max_threads(threads[1]))
            if clean("actual restored threads", lambda: (cv2.getNumThreads(), o3d.utility.get_max_threads())) != threads: failures.append({"action": "thread restore", "message": "Actual policy differs"})
        for k, v in env.items(): clean("environment "+k, lambda k=k, v=v: os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v))
        report["cleanup_failures"] = failures; report["cleanup_passed"] = not failures and all(os.environ.get(k) == v for k, v in env.items())
        report["status"] = "passed" if primary is None and report["cleanup_passed"] and report.get("binding_after") == report.get("binding") else "failed"
        report["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(); report["producer_wall_s"] = time.perf_counter()-began
        try: save()
        except BaseException as error:
            if primary is not None: raise primary from error
            raise
    if report["status"] != "passed": raise RuntimeError("Near-seed experiment failed; diagnostics retained") from primary


if __name__ == "__main__": run(parse())
