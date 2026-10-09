"""Scoped authority for fresh actual-input complete-device ICP experiments.

No old/Microbatch/Finish permit is accepted. Audit and timing must use exactly
the same array values, device evaluator, graph/chunk and runtime configuration.
The terminal GPU reference is bitwise; original native CPU shadows are separate
acceptance evidence. This authority supplies no reconstruction or speed claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIND = "gpu-icp-current-input-device-loop-audit-v3"
TIMING_KIND = "gpu-icp-current-input-device-loop-timing-v3"
POLICY = "explicit-stream-chunked-device-icp-owner-bound-v2"
CURRENT = "07a948e81127dc742a330ec6bc24c85ceb7a9996128703c2a1cb7f9d0593741c"
_REGISTRY = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            result.update(part)
    return result.hexdigest()


def is_digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def descriptor(value, tail, dtype):
    require(type(value) is dict and set(value) == {"dtype", "shape", "sha256"}, "Exact array value descriptor required")
    shape = value["shape"]
    require(type(shape) is list and len(shape) == 2 and all(type(n) is int and n >= 0 for n in shape)
        and shape[1] == tail and value["dtype"] == dtype and is_digest(value["sha256"]), "Malformed original numeric descriptor")


def normalized_input(value):
    """Only audit instrumentation flags differ between the two declared modes."""
    require(type(value) is dict and set(value) == {"source", "target", "normals", "seed", "configuration"},
        "Complete exact consumed arrays/seed/configuration required")
    for key in ("source", "target", "normals"):
        descriptor(value[key], 3, "<f8")
    descriptor(value["seed"], 4, "<f8")
    require(value["seed"]["shape"] == [4, 4] and value["normals"]["shape"] == value["target"]["shape"]
        and 0 < value["source"]["shape"][0] <= 1_000_000 and 0 < value["target"]["shape"][0] <= 1_000_000,
        "Bounded original cloud/normal/seed shapes required")
    cfg = value["configuration"]
    require(type(cfg) is dict and set(cfg) == {"device", "max_points", "max_scratch_bytes", "max_total_bytes",
        "stages", "cuda_graph", "chunk_iterations", "audit_nearest", "audit_misses"}, "Exact lane execution configuration required")
    require(cfg["device"] == "CUDA:0" and cfg["stages"] == [[.12, 40], [.06, 30], [.03, 20]]
        and type(cfg["cuda_graph"]) is bool and type(cfg["chunk_iterations"]) is int
        and cfg["chunk_iterations"] in (1, 2, 4) and type(cfg["audit_nearest"]) is bool
        and cfg["audit_nearest"] is cfg["audit_misses"], "Original stages, selected graph/chunk and complete audit flags required")
    for key, limit in (("max_points", 1_000_000), ("max_scratch_bytes", 512*1024**2), ("max_total_bytes", 1024*1024**2)):
        require(type(cfg[key]) is int and 0 < cfg[key] <= limit, "Invalid bounded device-loop policy")
    normalized = json.loads(canonical(value))
    del normalized["configuration"]["audit_nearest"]
    del normalized["configuration"]["audit_misses"]
    return normalized


def normalized_source(value):
    require(type(value) is dict and value.get("policy") == POLICY and value.get("original_math_guard_inverse") is True,
        "Fresh complete-device evaluator source required")
    require(value.get("stages") == [[.12, 40], [.06, 30], [.03, 20]] and value.get("graph_option_supported") is True
        and value.get("options") == ["--std=c++11", "--fmad=false"], "Declared original reduction/stage/compiler policy required")
    artifacts = value.get("artifacts")
    require(type(artifacts) is dict and artifacts and all(is_digest(h) for h in artifacts.values())
        and is_digest(value.get("generated_sha256")), "Closed mathematical helper/shader bytes required")
    normalized = json.loads(canonical(value))
    normalized.pop("stream_ptr", None)  # Distinct owned streams are intentionally allowed.
    return normalized


def terminal_contract(value, rows, targets):
    require(type(value) is dict and set(value) == {"pose", "correspondences", "fitness", "inlier_rmse", "queries", "updates"},
        "Exact terminal pose/correspondence/metric/iteration reference required")
    descriptor(value["pose"], 4, "<f8")
    descriptor(value["correspondences"], 2, "<i4")
    require(value["pose"]["shape"] == [4, 4] and value["correspondences"]["shape"][0] <= rows,
        "Invalid bounded terminal output shape")
    for key in ("fitness", "inlier_rmse"):
        require(type(value[key]) in (int, float) and math.isfinite(value[key]) and value[key] >= 0,
            "Nonfinite terminal metric")
    require(value["fitness"] <= 1 and type(value["queries"]) is int and 3 <= value["queries"] <= 93
        and type(value["updates"]) is int and 3 <= value["updates"] <= 90,
        "Incomplete or out-of-bounds three-stage trajectory")
    require(value["fitness"] == value["correspondences"]["shape"][0]/rows,
        "Terminal fitness must count the actual source-to-target correspondence rows")
    require(value["queries"] == value["updates"]+3,
        "Each original scale needs exactly one initial query in addition to update evaluations")


def actual_resource_closure(binding):
    require(type(binding) is dict and set(binding) == {"source_sha256", "method_source", "artifacts_sha256", "runtime", "fixed_files"},
        "Exact current experiment runtime/source/input binding required")
    require(binding["source_sha256"] == CURRENT and binding["method_source"].get("policy") == POLICY,
        "Wrong current core or separately named device method")
    runtime = binding["runtime"]
    require(runtime.get("device") == "CUDA:0" and runtime.get("open3d") == "0.20.0"
        and runtime.get("thread_policy") == {"open3d": 20, "opencv": 20, "omp": "8"}, "Original loaded native/thread/device policy required")
    require(type(runtime.get("binaries")) is dict and set(runtime["binaries"]) ==
        {"open3d", "numpy", "cupy", "native"}, "Actual loaded numerical/native binary owners required")
    for record in runtime["binaries"].values():
        require(type(record) is dict and set(record) == {"path", "sha256"} and is_digest(record["sha256"]),
            "Malformed loaded binary identity")
    artifacts, fixed = binding["artifacts_sha256"], binding["fixed_files"]
    require(type(artifacts) is dict and artifacts and type(fixed) is dict and fixed,
        "Complete experiment and raw/capture/fixture resources required")
    for name, digest in binding["method_source"]["artifacts"].items():
        require(artifacts.get(name) == digest, "Every actual mathematical helper must be included in current source closure")
    for name, digest in artifacts.items():
        path = Path(name)
        require(type(name) is str and "\\" not in name and ":" not in name and not path.is_absolute()
            and ".." not in path.parts and (ROOT/path).resolve().is_relative_to(ROOT) and is_digest(digest)
            and sha(ROOT/path) == digest, "A current experiment/math source changed")
    for name, digest in fixed.items():
        require(type(name) is str and is_digest(digest) and sha(name) == digest, "A bound raw/capture/fixture/binary file changed")
    for record in runtime["binaries"].values():
        require(fixed.get(record["path"]) == record["sha256"], "Loaded binary must be rehashed in actual file closure")


def validate_audit_report(report, binding):
    actual_resource_closure(binding)
    require(report.get("kind") == KIND and report.get("status") == "passed" and report.get("mode") == "audit",
        "Newly closed own-loop audit required; no historical or microbatch authority")
    require(report.get("binding") == report.get("binding_after") == binding and report.get("failure") is None
        and report.get("cleanup_failures") == [] and report.get("cleanup_passed") is True
        and report.get("input_bytes_unchanged") is True and report.get("loaded_owners_unchanged") is True,
        "Audit input/source/runtime/owner/cleanup closure failed")
    require(report.get("performance_attribution_valid") is False and report.get("whole_finish_authority") is False,
        "CPU-shadow audit cannot claim timing or reconstruction authority")
    rows = report.get("rows")
    require(type(rows) is list and rows, "Positive complete actual trajectories required")
    scopes, hit_count, miss_count = {}, 0, 0
    for row in rows:
        consumed = row["input_binding"]
        require(consumed["configuration"]["audit_nearest"] is True and consumed["configuration"]["audit_misses"] is True,
            "Every actual direct hit/miss must be CPU-audited")
        scope = normalized_input(consumed)
        source = normalized_source(row["source_binding"])
        require({k: source[k] for k in binding["method_source"]} == binding["method_source"],
            "Actual lane shader/helpers differ from current declared method")
        require(source.get("device") == consumed["configuration"]["device"]
            and source.get("cuda_graph") is consumed["configuration"]["cuda_graph"],
            "Actual lane graph/device provenance differs from consumed configuration")
        n, m = scope["source"]["shape"][0], scope["target"]["shape"][0]
        terminal_contract(row["terminal"], n, m)
        stats = row["loop_report"]["statistics"]
        require(row["loop_report"].get("closed") is True and row["loop_report"].get("failure") is None
            and row["loop_report"].get("input_binding") == consumed
            and normalized_source(row["loop_report"].get("provenance")) == source
            and row["native_shadow"].get("passed") is True and row["native_shadow"].get("correspondence_ids_equal") is True,
            "Fresh complete original CPU shadow and selected-stream cleanup required")
        for name in ("transform_max_abs_delta", "fitness_abs_delta", "rmse_abs_delta"):
            value = row["native_shadow"].get(name)
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e-8,
                "Original native result shadow failed declared tight bounds")
        required = ("queries", "query_rows", "updates", "direct_hits", "direct_misses", "audited_hits",
                    "audited_misses", "cpu_ambiguity_rows", "solve_blocks")
        require(all(type(stats.get(k)) is int and stats[k] >= 0 for k in required), "Complete actual integer accounting required")
        require(stats["queries"] == row["terminal"]["queries"] and stats["updates"] == row["terminal"]["updates"]
            and stats["query_rows"] == stats["queries"]*n
            and stats["direct_hits"] == stats["audited_hits"] and stats["direct_misses"] == stats["audited_misses"]
            and stats["query_rows"] == stats["audited_hits"]+stats["audited_misses"]+stats["cpu_ambiguity_rows"]
            and stats["solve_blocks"] == 0, "Incomplete actual NN or hidden solve/full-call fallback")
        trace = row.get("query_trace")
        require(type(trace) is list and len(trace) == stats["queries"], "Every actual NN evaluation must have closed CPU-shadow evidence")
        for index, query in enumerate(trace):
            require(query.get("query_index") == index and type(query.get("stage")) is int and query["stage"] in (0, 1, 2)
                and query.get("radius") == (.12, .06, .03)[query["stage"]]
                and query.get("target_sha256") == consumed["target"]["sha256"], "Actual NN stage/target query trace changed")
            descriptor(query.get("packet"), 8, "<f8")
            require(query["packet"]["shape"][0] == n and is_digest(query.get("corrected_ids_sha256")),
                "Every original query row must be CPU-shadowed before correction")
        stages = [query["stage"] for query in trace]
        require(stages == sorted(stages) and set(stages) == {0,1,2}
            and all(2 <= stages.count(stage) <= maximum for stage,maximum in enumerate((41,31,21))),
            "Original scale order and per-stage query bounds must close the complete trajectory")
        require(row.get("query_trace_sha256") == hashlib.sha256(canonical(trace).encode()).hexdigest(), "Actual query trace hash changed")
        key = canonical(scope)
        require(key not in scopes, "Duplicate scope cannot substitute for missing actual configuration")
        scopes[key] = {"input": scope, "source": source, "terminal": row["terminal"]}
        hit_count += stats["audited_hits"]; miss_count += stats["audited_misses"]
    require(hit_count > 0 and miss_count > 0, "Positive actual direct hit AND miss audit coverage required")
    return list(scopes.values())


@dataclass(frozen=True)
class DeviceLoopTimingPermit:
    report_path: str
    report_sha256: str
    binding_json: str
    scopes_json: str


def validate_device_loop_audit(path, binding):
    path = Path(path).resolve(strict=True)
    before = sha(path)
    scopes = validate_audit_report(json.loads(path.read_text(encoding="utf-8")), binding)
    require(sha(path) == before, "Audit report changed during fresh validation")
    token = DeviceLoopTimingPermit(str(path), before, canonical(binding), canonical(scopes))
    _REGISTRY[id(token)] = token
    return token


def registered(token):
    require(type(token) is DeviceLoopTimingPermit and _REGISTRY.get(id(token)) is token,
        "Require own freshly validated device-loop permit, not a constructed/old/Microbatch token")
    require(sha(token.report_path) == token.report_sha256, "Closed audit report changed")


def validate_device_loop_start(token, consumed, source, binding):
    registered(token)
    require(canonical(binding) == token.binding_json, "Current complete experiment/runtime scope changed")
    require(consumed["configuration"]["audit_nearest"] is False and consumed["configuration"]["audit_misses"] is False,
        "Timing consumer must explicitly omit observational shadows")
    normalized = normalized_input(consumed)
    actual_source = normalized_source(source)
    matches = [s for s in json.loads(token.scopes_json) if s["input"] == normalized and s["source"] == actual_source]
    require(len(matches) == 1, "Foreign exact arrays/seed/graph/chunk/source configuration was not audited")
    return token


def validate_device_loop_terminal(token, consumed, source, terminal, binding):
    validate_device_loop_start(token, consumed, source, binding)
    normalized = normalized_input(consumed)
    record = next(s for s in json.loads(token.scopes_json) if s["input"] == normalized)
    terminal_contract(terminal, normalized["source"]["shape"][0], normalized["target"]["shape"][0])
    require(canonical(terminal) == canonical(record["terminal"]),
        "Timed pose/correspondence/metrics/query/update bytes differ from own actual audited trajectory")
    return True


def validate_timing_accounting(stats, source_rows):
    keys = ("queries","query_rows","direct_hits","direct_misses","cpu_ambiguity_rows",
        "audited_hits","audited_misses","flagged_rows","packet_bytes","solve_blocks")
    require(all(type(stats.get(k)) is int and stats[k]>=0 for k in keys), "Complete timed integer NN accounting required")
    require(stats["query_rows"] == stats["queries"]*source_rows
        and stats["direct_hits"]+stats["direct_misses"]+stats["cpu_ambiguity_rows"] == stats["query_rows"]
        and stats["audited_hits"] == stats["audited_misses"] == stats["solve_blocks"] == 0
        and stats["flagged_rows"] == stats["cpu_ambiguity_rows"]
        and stats["packet_bytes"] == 64*stats["cpu_ambiguity_rows"],
        "Timing must omit observational audits while explicitly counting original ambiguity resolution")
