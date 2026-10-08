"""Distinct stdlib-only authority for one freshly audited seed microbatch.

This token permits omission of complete CPU query/result shadows on exactly
the audited original point arrays, ordered seed prefix and scheduling policy.
It supplies no whole-Finish, new field, graph, mesh or device-loop authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[2]
CURRENT = "07a948e81127dc742a330ec6bc24c85ceb7a9996128703c2a1cb7f9d0593741c"
KIND = "gpu-icp-seed-microbatch-audit-v1"
POLICY = "independent-seed-explicit-stream-original-resident-cpu-solve-v1"
PRODUCERS = ("scripts/research/gpu_icp_experiment_driver.py", "scripts/research/gpu_icp_experiment_protocol.py",
    "scripts/research/gpu_icp_experiment_capture.py", "tests/test_gpu_icp_experiment_capture.py",
    "tests/test_gpu_icp_experiment_protocol.py")
_REGISTERED = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_string(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def array_descriptor(value, shape_tail):
    require(type(value) is dict and set(value) == {"shape", "dtype", "nbytes", "sha256"}, "Exact canonical array descriptor required")
    shape = value["shape"]
    require(type(shape) is list and len(shape) == 2 and all(type(n) is int and n >= 0 for n in shape)
        and shape[1] == shape_tail and value["dtype"] in ("<f8", "=f8")
        and value["nbytes"] == math.prod(shape) * 8 and digest_string(value["sha256"]), "Malformed original FP64 descriptor")


def pair_contract(pair):
    require(type(pair) is dict and set(pair) == {"source", "target", "seeds"}, "Exact original pair/seed binding required")
    for role in ("source", "target"):
        require(type(pair[role]) is dict and set(pair[role]) == {"points", "normals", "colors"}, "Original complete cloud descriptor required")
        for value in pair[role].values():
            array_descriptor(value, 3)
        require(0 < pair[role]["points"]["shape"][0] <= 1000000, "Bounded nonempty original cloud required")
    require(pair["target"]["normals"]["shape"] == pair["target"]["points"]["shape"], "Original complete target normals required")
    require(type(pair["seeds"]) is list and 2 <= len(pair["seeds"]) <= 6, "Genuine original proposal seed list required")
    for seed in pair["seeds"]:
        array_descriptor(seed, 4)
        require(seed["shape"] == [4, 4], "Original 4x4 seed required")
    require(len({seed["sha256"] for seed in pair["seeds"]}) == len(pair["seeds"]), "No duplicate or padded seeds")


def prefix_pair(pair, count):
    return {"source": pair["source"], "target": pair["target"], "seeds": pair["seeds"][:count]}


def binding_contract(binding):
    require(type(binding) is dict and set(binding) == {"source_sha256", "artifacts_sha256", "runtime", "configuration", "solve_metadata", "method_policy"}, "Exact fresh microbatch runtime binding required")
    require(binding["source_sha256"] == CURRENT and binding["method_policy"] == POLICY, "Wrong current source or seed scheduling method")
    artifacts = binding["artifacts_sha256"]
    require(type(artifacts) is dict and artifacts and all(digest_string(h) for h in artifacts.values()), "Bound current helper/kernel artifacts")
    for name, expected in artifacts.items():
        require(type(name) is str and "\\" not in name and not Path(name).is_absolute() and ":" not in name
            and ".." not in Path(name).parts and (ROOT / name).resolve().is_relative_to(ROOT), "Safe canonical source artifact name required")
        require(sha(ROOT / name) == expected, "A measured helper/kernel artifact changed")
    cfg, runtime = binding["configuration"], binding["runtime"]
    require(cfg.get("device") == runtime.get("device") and cfg.get("gpu_timing") is False
        and cfg.get("stages") == [[.12, 40], [.06, 30], [.03, 20]], "Original ICP stages/explicit device/timer policy required")
    require(runtime.get("open3d") == "0.20.0" and runtime.get("thread_policy") ==
        {"open3d_threads": 20, "opencv_threads": 20, "omp_threads": "8"}, "Original native/thread semantics required")
    for key in ("max_points", "max_query_bytes", "max_cache_bytes", "max_total_bytes"):
        require(type(cfg.get(key)) is int and cfg[key] > 0, "Explicit bounded batch memory required")
    require(cfg["max_points"] <= 1000000 and cfg["max_query_bytes"] <= 136 * 1024**2
        and cfg["max_cache_bytes"] <= 256 * 1024**2 and cfg["max_total_bytes"] <= 1024 * 1024**2,
        "Memory policy exceeds measured bounded domain")
    canonical(binding)


def audit_lane(lane):
    nn, icp = lane["retrieval_statistics"], lane["statistics"]
    for name in ("query_rows", "direct_gpu_hits", "declared_gpu_misses", "audited_hits", "audited_misses", "exact_cpu_queries"):
        require(type(nn.get(name)) is int and nn[name] >= 0, "Measured nonnegative integer query accounting required")
    require(nn["query_rows"] > 0 and nn["direct_gpu_hits"] == nn["audited_hits"]
        and nn["declared_gpu_misses"] == nn["audited_misses"]
        and nn["query_rows"] == nn["audited_hits"] + nn["audited_misses"] + nn["exact_cpu_queries"],
        "Every actual direct hit/miss must be shadowed before correction")
    require(all(nn.get(name) == 0 for name in ("audit_index_mismatches", "audit_false_misses", "device_malformed_results"))
        and icp.get("cpu_fallback_calls") == 0 and icp.get("calls") == 1,
        "No hidden full-call fallback, mismatch or malformed device result")


def validate_report(report, expected_binding, expected_pair):
    binding_contract(expected_binding)
    pair_contract(expected_pair)
    require(report.get("kind") == KIND and report.get("status") == "passed", "Fresh completed microbatch audit required")
    require(report.get("failure") is None and report.get("validation_failure") is None and report.get("cleanup_failures") == [],
        "A contradictory recorded primary/cleanup failure cannot authorize timing")
    require(report.get("binding") == report.get("binding_after") == expected_binding
        and report.get("pair_binding") == report.get("pair_binding_after") == expected_pair, "Actual original arrays/runtime changed")
    producer = report.get("producer_artifacts_sha256")
    require(type(producer) is dict and producer == report.get("producer_artifacts_sha256_after") and set(PRODUCERS) <= set(producer),
        "Closed current producer/validator artifact identity required")
    for name, expected in producer.items():
        require(type(name) is str and "\\" not in name and not Path(name).is_absolute() and ":" not in name
            and ".." not in Path(name).parts and digest_string(expected) and sha(ROOT/name) == expected,
            "Producer/protocol source changed")
    fixed = report.get("fixed_files_sha256")
    require(type(fixed) is dict and fixed and fixed == report.get("fixed_files_sha256_after"),
        "Native binaries/solver/ZIP/profile/fixture before-after closure required")
    helpers = report.get("helpers")
    require(type(helpers) is list and helpers and all(type(h) is dict and h.get("audit") is True
        and h.get("binding") == expected_binding and h.get("source_unchanged") is True and h.get("closed") is True
        and h.get("failure") is None and h.get("cleanup_failures") == [] for h in helpers),
        "All actual audited lane/cache owners must close without failure or source drift")
    require(all(report.get(name) is True for name in ("cleanup_passed", "source_unchanged", "fixture_unchanged", "input_bytes_unchanged")), "Closed cleanup/input/source audit required")
    require(report.get("whole_finish_authority") is False and report.get("performance_attribution_valid") is False,
        "CPU-shadow audit cannot supply whole-Finish or timing attribution")
    sizes = report.get("batch_sizes")
    require(type(sizes) is list and sizes and sizes == sorted(set(sizes)) and all(type(n) is int and n in (2, 4) and n <= len(expected_pair["seeds"]) for n in sizes), "Only genuine audited ordered2/4 seed prefixes")
    rows = report.get("audit_rows")
    require(type(rows) is list and len(rows) == 2 * len(sizes), "Both serial and concurrent audits required")
    allowed, coverage = [], set()
    direct_hits = direct_misses = 0
    for row in rows:
        count, schedule = row.get("batch_size"), row.get("schedule")
        require(count in sizes and schedule in ("serial", "concurrent") and (count, schedule) not in coverage, "Unique actual batch audit rows required")
        coverage.add((count, schedule))
        actual_pair = prefix_pair(expected_pair, count)
        require(row.get("pair_binding") == actual_pair and row.get("passed") is True, "Exact original seed prefix and completed audit required")
        require(row.get("pair_verdict_equal") is True and row.get("full_original_proposal_count") == len(expected_pair["seeds"]),
            "Complete original competing-proposal consumption and ambiguity verdict required")
        batch = row["batch_record"]
        require(batch.get("inputs") == actual_pair and batch.get("schedule") == schedule and batch.get("lanes") == count
            and batch.get("input_bytes_unchanged") is True and batch.get("full_call_fallbacks") == 0,
            "Actual scoped lane batch differs from claimed audit")
        lanes = batch.get("lane_deltas")
        require(type(lanes) is list and [lane.get("index") for lane in lanes] == list(range(count)), "Original ordered lane consumption required")
        for lane in lanes:
            audit_lane(lane)
            direct_hits += lane["retrieval_statistics"]["audited_hits"]
            direct_misses += lane["retrieval_statistics"]["audited_misses"]
        shadows = row.get("native_shadows")
        require(type(shadows) is list and [s.get("seed_index") for s in shadows] == list(range(count)), "Each actual original seed needs fresh native ICP shadow")
        for shadow in shadows:
            require(shadow.get("passed") is True and shadow.get("correspondence_ids_equal") is True
                and shadow.get("strong_gate_equal") is True, "Native result/correspondence/normal-diversity shadow failed")
            for name in ("transform_max_abs_diff", "fitness_abs_diff", "inlier_rmse_abs_diff"):
                require(type(shadow.get(name)) in (int, float) and math.isfinite(shadow[name]) and 0 <= shadow[name] <= 1e-8,
                    "Fresh native result tolerance failed")
        gates = row.get("bridge_gate_shadows")
        require(type(gates) is list and [g.get("seed_index") for g in gates] == list(range(count))
            and all(g.get("passed") is True and g.get("original_forward_consumed_once") is True for g in gates),
            "Original reciprocal/held-out/independent-camera/visual/info proposal gates required")
        allowed.append({"pair_binding": actual_pair, "schedule": schedule})
    require(coverage == {(size, schedule) for size in sizes for schedule in ("serial", "concurrent")}
        and direct_hits > 0 and direct_misses > 0, "Positive complete actual hit/miss and both scheduling coverage required")
    return allowed


@dataclass(frozen=True)
class MicrobatchTimingPermit:
    report_path: str
    report_sha256: str
    binding_json: str
    pair_json: str
    allowed_json: str


def validate_microbatch_audit(path, expected_binding, expected_pair_binding):
    path = Path(path).resolve(strict=True)
    report = json.loads(path.read_text(encoding="utf-8"))
    allowed = validate_report(report, expected_binding, expected_pair_binding)
    require(all(sha(name) == expected for name, expected in report["fixed_files_sha256"].items()),
        "A fixed raw fixture/native library or solver resource changed since audit")
    token = MicrobatchTimingPermit(str(path), sha(path), canonical(expected_binding), canonical(expected_pair_binding), canonical(allowed))
    _REGISTERED[id(token)] = token
    return token


def assert_registered_microbatch_permit(token):
    require(type(token) is MicrobatchTimingPermit and _REGISTERED.get(id(token)) is token,
        "Require a freshly validated distinct microbatch permit, never an old Device/Finish token")
    require(sha(token.report_path) == token.report_sha256, "Closed audit report changed")


def validate_microbatch_permit(token, *, binding, pair_binding, schedule):
    assert_registered_microbatch_permit(token)
    require(canonical(binding) == token.binding_json, "Actual runtime/config/source differs from audited batch")
    binding_contract(binding)
    require({"pair_binding": pair_binding, "schedule": schedule} in json.loads(token.allowed_json),
        "Foreign arrays, seed order/count or scheduling policy were not audited")
    # Revalidate current report contents as well as its hash, including complete
    # native and original gate evidence; the registry alone is not authority.
    report = json.loads(Path(token.report_path).read_text(encoding="utf-8"))
    validate_report(report, binding, json.loads(token.pair_json))
    return MappingProxyType({"scope": "same-array ordered-seed microbatch timing only", "whole_finish_authority": False,
        "fixed_resource_closure": "Full raw/native/solver/fixture hashes at permit construction and producer before/after. Call-time exact consumed arrays/current helper/runtime binding; no raw ZIP rehash in timed batch."})
