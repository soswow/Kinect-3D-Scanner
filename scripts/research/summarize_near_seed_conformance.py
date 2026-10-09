"""Publish path-free fixed-pair conformance receipts; no numerical authority."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import CodeType, FunctionType
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import benchmark_near_seed_conformance as driver
from scripts.research import benchmark_near_seed_reuse as strict
scope = driver.scope
KIND = "near-seed-fixed-pair-conformance-scalar-summary-v2"
PINS = {"scripts/research/benchmark_near_seed_conformance.py": "22660ef23db176ed9de7018f8755a9b519b962a21a77b71d924ea991db9d76db",
    "tests/test_near_seed_conformance.py": "5d64359c2ee21caae3095c7166082b67179ecfe38b90fc9dfe92077ef262acc2"}


def require(ok, text): driver.require(ok, text)


def read(path):
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def artifact_receipt(binding, pins):
    require(type(binding.get("artifacts_sha256")) is dict and all(binding["artifacts_sha256"].get(name) == digest for name, digest in pins.items()),
        "Recorded producer artifacts must bind the actual frozen validation source family")


def guard_artifacts(guards, binding):
    for guard in guards:
        guard.check()
        for _, path in guard.modules:
            name = path.relative_to(ROOT).as_posix()
            artifact_receipt(binding, {name: scope.sha(path)})


def phase_closed(phase, captures):
    driver.validate_scope(phase, captures)
    calls = driver.phase_calls(phase)
    require(phase.get("cleanup_passed") is True and phase.get("cleanup_failures") == [] and phase.get("failure") is None
        and phase["call_count"] == len(calls) == 714, "Entire closed actual 714-call phase required")
    for pair in phase["pairs"]:
        require(pair.get("input_bytes_unchanged") is True and pair.get("seed_bytes_unchanged") is True
            and pair.get("complete") is True, "Whole exact pair input/top-seed closure required")
        for proposal in pair["proposals"]:
            require(all(g.get("complete") is True for g in proposal["gates"]), "Complete original gate suffix required")
    require(all(c.get("complete") is True and c.get("input_bytes_unchanged") is True and not c.get("failure") for c in calls),
        "All actual CPU calls must close")
    for value in [phase["cold_whole_phase_wall_s"], phase["whole_phase_wall_s"]]+[p["inclusive_pair_wall_s"] for p in phase["pairs"]]:
        require(type(value) is float and 0 <= value < float("inf"), "Finite nonnegative inclusive wall time required")
    require(phase["cold_whole_phase_wall_s"] >= phase["whole_phase_wall_s"]
        >= sum(p["inclusive_pair_wall_s"] for p in phase["pairs"]), "Inclusive cold/phase/pair clocks contradict their nesting")
    return calls


def strict_failure(audit_path, failure_path):
    audit, audit_sha = read(audit_path); failure, failure_sha = read(failure_path)
    artifact_receipt(audit["binding"], driver.V1_PINS)
    guard = strict.LoadedMethodGuard(); guard_artifacts((guard,), audit["binding"])
    strict.load_audit(audit_path, audit["binding"], audit["captures"])
    require(failure.get("kind") == strict.TIMING_KIND and failure.get("status") == "failed" and failure.get("failure")
        and failure.get("cleanup_passed") is True and failure.get("cleanup_failures") == []
        and failure.get("binding") == failure.get("binding_after") == audit["binding"]
        and failure.get("captures") == audit["captures"]
        and failure["audit_proof"]["sha256"] == audit_sha, "Preserved own strict v1 timing refusal/cleanup required")
    phase = failure["rounds"][0]["cache"]; stats = phase["cache"]["statistics"]
    require(len(driver.phase_calls(phase)) == 1 and stats["calls"] == stats["misses"] == stats["original_cpu_calls"] == 1
        and stats["hits"] == stats["audited_hits"] == 0 and phase["cache"]["closed"] is True, "Strict refusal must be the first original CPU MISS")
    old, actual = driver.phase_calls(audit["rounds"][0]["cache"])[0], driver.phase_calls(phase)[0]
    require(old["input_binding"] == actual["input_binding"] and actual["method_receipt"]["kind"] == "miss"
        and all(old[k] == actual[k] for k in ("context", "call_index", "caller"))
        and strict.class_reference(old["method_receipt"]) == strict.class_reference(actual["method_receipt"])
        and strict.result_reference(old["result"]) != strict.result_reference(actual["result"])
        and actual.get("input_bytes_unchanged") is True and actual.get("complete") is False
        and old["result"]["correspondence_mapping"] == actual["result"]["correspondence_mapping"], "Strict original identical-input canonical-ID evidence required")
    deltas = {"pose_max_abs_delta": driver.method.seed_distance(driver.validate_result(old["result"]), driver.validate_result(actual["result"])),
        "fitness_abs_delta": abs(old["result"]["fitness"]-actual["result"]["fitness"]),
        "rmse_abs_delta": abs(old["result"]["inlier_rmse"]-actual["result"]["inlier_rmse"])}
    require(deltas["pose_max_abs_delta"] > 0 and max(deltas.values()) <= 1e-12, "Recorded refusal must show bounded original CPU last-bit noise")
    require(read(audit_path)[1] == audit_sha and read(failure_path)[1] == failure_sha, "Preserved strict receipts changed during publication")
    guard_artifacts((guard,), audit["binding"])
    return {"audit_sha256": audit_sha, "failed_timing_sha256": failure_sha, "first_call_kind": "miss", "reused_calls": 0,
        "initial_seed_bytes_identical": True, "canonical_ids_identical": True, "cleanup_passed": True, **deltas,
        "independent_process_exit_observed": False}


def compact(audit_path, timing_path, strict_audit_path, strict_failure_path):
    driver.derivation_contract()
    require(all(scope.sha(ROOT/name) == digest for name, digest in PINS.items()), "Frozen v2 publisher dependencies changed")
    guards = (driver.LoadedMethodGuard(), strict.LoadedMethodGuard())
    audit, audit_sha = read(audit_path); timing, timing_sha = read(timing_path)
    artifact_receipt(audit["binding"], dict(driver.V1_PINS, **PINS))
    guard_artifacts(guards, audit["binding"])
    digest, refs, audited_phase = driver.load_audit(audit_path, audit["binding"], audit["captures"])
    require(digest == audit_sha and timing.get("kind") == driver.TIMING_KIND and timing.get("mode") == "timing"
        and timing.get("status") == "passed" and timing.get("failure") is None and timing.get("cleanup_passed") is True
        and timing.get("cleanup_failures") == [] and timing["binding"] == timing["binding_after"] == audit["binding"]
        and timing["captures"] == audit["captures"] and timing["audit_proof"]["sha256"] == audit_sha
        and timing.get("conformance_policy") == dict(driver.CONFORMANCE_POLICY)
        and all(timing.get(k) is False for k in ("whole_finish_authority", "native_history_bitwise_equivalent", "neighborhood_authority", "performance_authority")),
        "Only closed matching own v2 audit/timing receipts may be published")
    require(len(audit["captures"]) == 2 and sum(len(c["tasks"]) for c in audit["captures"]) == 9
        and sum(t["proposal_count"] for c in audit["captures"] for t in c["tasks"]) == 30,
        "Exact own two-capture nine-pair thirty-proposal scope required")
    phase_closed(audit["rounds"][0]["native"], audit["captures"]); phase_closed(audited_phase, audit["captures"])
    stats = audited_phase["cache"]["statistics"]
    require(stats["calls"] == 714 and stats["hits"] == stats["audited_hits"] == 172 and stats["misses"] == 542,
        "All actual 172 audit hits must be freshly CPU-shadowed")
    require(len(timing["rounds"]) == 3, "Three independently controlled rotating rounds required")
    samples, deltas = [], []
    for i, row in enumerate(timing["rounds"]):
        require(row["repetition"] == i and row["order"] == (["native", "cache"] if i%2 == 0 else ["cache", "native"]), "Declared rotating order changed")
        native_calls = phase_closed(row["native"], timing["captures"]); calls = phase_closed(row["cache"], timing["captures"])
        quality = driver.compare_phases(row["native"], row["cache"])
        require(quality == row["quality"] and quality["passed"] and driver.compare_phases(audited_phase, row["cache"])["passed"],
            "Every timing complete proposal/gate/ambiguity control must pass")
        cache = row["cache"]["cache"]; actual = cache["statistics"]
        require(cache["audit"] is False and cache["closed"] is True and cache["failure"] is None and cache["owned_payload_bytes"] == 0
            and cache["configuration"] == audit["binding"]["configuration"] and cache["policy"] == driver.method.POLICY
            and actual["calls"] == 714 and actual["hits"] == 172 and actual["misses"] == actual["original_cpu_calls"] == 542
            and actual["audited_hits"] == 0, "Timing may omit only authorized original hit shadows")
        trajectory = driver.TimingTrajectory(refs)
        for original in calls:
            value = copy.deepcopy(original); saved = value.pop("native_noise_conformance")
            trajectory.prepare(value["input_binding"], value["seed_values"], value["context"], value["call_index"], value["caller"])
            trajectory.authorize(value["input_binding"]); trajectory.finish(value)
            require(value["native_noise_conformance"] == saved and value["method_receipt"]["shadow"].get("collected") is False,
                "Actual timing conformance must independently recompute; omitted hit shadow must be explicit")
            deltas.append(saved)
        samples.append({"round": i, "order": row["order"], "native_cold_whole_phase_s": row["native"]["cold_whole_phase_wall_s"],
            "reuse_cold_whole_phase_s": row["cache"]["cold_whole_phase_wall_s"],
            "pairs": [{"ordinal": n, "native_inclusive_s": a["inclusive_pair_wall_s"], "reuse_inclusive_s": b["inclusive_pair_wall_s"]}
                for n, (a, b) in enumerate(zip(row["native"]["pairs"], row["cache"]["pairs"]))]})
    strict_receipt = strict_failure(strict_audit_path, strict_failure_path)
    guard_artifacts(guards, audit["binding"])
    require(read(audit_path)[1] == audit_sha and read(timing_path)[1] == timing_sha
        and all(scope.sha(ROOT/name) == digest for name, digest in PINS.items()), "Publication dependencies/receipts changed")
    return {"kind": KIND, "status": "passed", "audit_sha256": audit_sha, "timing_sha256": timing_sha,
        "source_sha256": audit["binding"]["source_sha256"], "producer_sources_sha256": {Path(name).name: digest for name, digest in dict(driver.V1_PINS, **PINS).items()},
        "scope": {"captures": 2, "pairs": 9, "proposals": 30, "calls_per_phase": 714, "audit_hits_cpu_shadowed": 172,
            "audit_misses": 542, "timing_rounds": 3, "timing_calls": 2142, "timing_hits": 516},
        "conformance_policy": dict(driver.CONFORMANCE_POLICY), "maximum_actual_noise": {
            k: max(d[k] for d in deltas) for k in ("actual_seed_max_abs_delta", "representative_seed_max_abs_delta",
                "result_pose_max_abs_delta", "fitness_abs_delta", "rmse_abs_delta")},
        "wall_samples": samples, "median_native_cold_whole_phase_s": statistics.median(s["native_cold_whole_phase_s"] for s in samples),
        "median_reuse_cold_whole_phase_s": statistics.median(s["reuse_cold_whole_phase_s"] for s in samples),
        "strict_v1_refusal": strict_receipt, "whole_finish_authority": False, "production_authority": False,
        "general_field_authority": False, "native_history_bitwise_equivalent": False, "performance_authority": False,
        "limits": "Recorded receipts only; no raw/binary rescan or independent process-exit observation. Cold phase charges hashing/setup/reconstruction/copying/all original gates and cleanup; nested pair walls are not subtracted. Fixed-pair empirical conformance, not a near-seed neighborhood proof."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "timing", "strict-audit", "strict-failure", "output"): parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Fresh output required")
    primary = None
    try: value = compact(args.audit, args.timing, args.strict_audit, args.strict_failure)
    except BaseException as error:
        primary = error
        value = {"kind": KIND, "status": "failed", "failure": {"type": type(error).__name__, "message": str(error)},
            "whole_finish_authority": False, "production_authority": False, "performance_authority": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try: args.output.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    except BaseException as error:
        if primary is not None: raise primary from error
        raise
    if primary is not None: raise primary
    print("Published fixed-pair scalar receipts; no Finish authority")


if __name__ == "__main__": main()
