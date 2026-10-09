"""Publish scalar receipts from a closed complete-proposal audit and timing.

Stdlib only, with no token creation or numerical dispatch. Current source bytes
and the reports' resource receipts are checked; private ZIP/library bytes and
OS child-exit metadata are not independently rescanned by this publisher.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import microbatch_bridge_protocol as protocol
from scripts.research import microbatch_bridge_scope as scope
from scripts.research import device_loop_workspace as workspace

KIND = "gpu-icp-complete-bridge-scalar-summary-v1"
TIMING_KIND = "gpu-icp-complete-bridge-proposal-timing-v1"
CURRENT = "07a948e81127dc742a330ec6bc24c85ceb7a9996128703c2a1cb7f9d0593741c"
OWN_FILES = ("scripts/research/summarize_gpu_icp_complete_bridge.py",
    "tests/test_gpu_icp_complete_bridge_summary.py")


def require(value, message):
    if not value: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(scope.canonical(value).encode()).hexdigest()


def wall(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
        "Positive finite measured wall required")
    return float(value)


def closed(report, mode):
    require(report.get("kind") == (protocol.KIND if mode == "audit" else TIMING_KIND)
        and report.get("mode") == mode and report.get("status") == "passed"
        and report.get("failure") is None and report.get("cleanup_failures") == [],
        "Only closed own complete-bridge receipts can be published")
    require(all(report.get(key) is True for key in ("cleanup_passed", "input_bytes_unchanged",
        "original_seed_bytes_unchanged", "loaded_owners_unchanged", "pair_quality_passed"))
        and report.get("whole_finish_authority") is False
        and report.get("performance_attribution_valid") is (mode == "timing"),
        "Recorded complete input/source/quality/owner closure required")
    binding = report["binding"]
    require(binding == report.get("binding_after") and binding.get("source_sha256") == CURRENT
        and type(binding.get("fixed_files")) is dict and bool(binding["fixed_files"]),
        "Exact current source and complete before/after fixed-resource receipts required")
    artifacts = binding["artifacts_sha256"]
    require(type(artifacts) is dict and 1 <= len(artifacts) <= 200,
        "Bounded nonempty source family required")
    for name, fingerprint in artifacts.items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == fingerprint,
            "Recorded producer/math/protocol source changed")
    for name in protocol.BRIDGE_FILES:
        require(artifacts.get(name) == sha(ROOT/name), "Complete-proposal source family omitted")
    contract = workspace.source_contract()
    require(binding.get("workspace_source") == contract
        and all(artifacts.get(name) == fingerprint for name, fingerprint in contract["artifacts"].items())
        and binding.get("configuration", {}).get("graph") is True
        and binding.get("configuration", {}).get("chunk_iterations") == 4,
        "Exact pristine workspace source family and declared graph4 policy required")
    owner, cache = report["workspace"], report["shared_cache"]
    require(owner.get("policy") == workspace.POLICY and owner.get("closed") is True and owner.get("failure") is None
        and owner.get("audit") is (mode == "audit") and owner.get("active_lane") is False
        and owner.get("retained_failed_lanes") == 0 and owner.get("template_started") is False
        and owner.get("provenance") == binding.get("workspace_source")
        and cache.get("closed") is True and cache.get("failure") is None,
        "Every private lane and immutable shared cache must close")
    return binding


def gates_equal(a, b):
    require(type(a) is list and a and len(a) == len(b), "Ordered original gate suffix required")
    for old, new in zip(a, b):
        require(old.get("complete") is True and new.get("complete") is True,
            "Incomplete original gate event")
        old = {k: v for k, v in old.items() if k != "wall_s_inclusive"}
        new = {k: v for k, v in new.items() if k != "wall_s_inclusive"}
        tolerance = 1e-5 if old.get("name") == "REG.get_information_matrix_from_point_clouds" else 1e-8
        require(not scope.compare_evidence(old, new, tolerance=tolerance),
            "Original ordered gate outcome/witness/numeric evidence changed")


def proposals(report):
    native, gpu = report["native_proposals"], report["gpu_proposals"]
    require(type(native) is list and 2 <= len(native) == len(gpu) <= 6
        and len(report["pair_binding"]["seeds"]) == len(gpu)
        and report["native_pair_verdict"] == report["gpu_pair_verdict"],
        "All genuine proposals and original ordered ambiguity verdict required")
    calls = []
    for index, (old, new) in enumerate(zip(native, gpu)):
        require(old.get("proposal_index") == new.get("proposal_index") == index
            and old.get("complete") is True and new.get("complete") is True
            and not scope.compare_evidence(old["result"], new["result"], path="result"),
            "Actual complete proposal result/ordering changed")
        gates_equal(old["gates"], new["gates"])
        require(type(new.get("calls")) is list and 1 <= len(new["calls"]) <= 1024,
            "Bounded complete actual call suffix required")
        for ci, call in enumerate(new["calls"]):
            require(call.get("proposal_index") == index and call.get("call_index") == ci
                and call.get("complete") is True and call.get("failure") is None
                and call.get("cleanup_failure") is None and call.get("input_bytes_unchanged") is True,
                "Actual dynamic call order/closure changed")
            loop, consumed = call["loop_report"], call["consumed_input_binding"]
            require(loop.get("closed") is True and loop.get("failure") is None
                and loop.get("input_binding") == consumed and loop.get("provenance") == call["source_binding"]
                and call["source_binding"].get("setup_reuse") == report["binding"]["workspace_source"],
                "Original-class lane input/source/completion did not close")
            protocol._full_input(call["input_binding"], consumed)
            calls.append(call)
    jobs = report["workspace"]["jobs"]
    require(len(jobs) == len(calls) and all(job.get("index") == i
        and job.get("closed") is True and job.get("failure") is None
        and job.get("report") == calls[i]["loop_report"] for i, job in enumerate(jobs)),
        "Each actual ordered call needs its exact completed workspace job receipt")
    return calls


def compact(audit, timing, audit_sha):
    binding = closed(audit, "audit")
    require(closed(timing, "timing") == binding and timing["pair_binding"] == audit["pair_binding"]
        and timing.get("audit_proof", {}).get("sha256") == audit_sha,
        "Timing must repeat this closed audit's exact pair/source/runtime receipt")
    audited, timed = proposals(audit), proposals(timing)
    require(len(audited) == len(timed), "Missing actual dynamic call suffix")
    totals = {key: 0 for key in ("queries", "query_rows", "updates", "audited_hits", "audited_misses", "cpu_ambiguity_rows")}
    maxima = {key: 0. for key in ("transform_max_abs_delta", "fitness_abs_delta", "rmse_abs_delta")}
    for old, new in zip(audited, timed):
        require(old["consumed_input_binding"]["configuration"].get("audit_nearest") is True
            and old["consumed_input_binding"]["configuration"].get("audit_misses") is True
            and new["consumed_input_binding"]["configuration"].get("audit_nearest") is False
            and new["consumed_input_binding"]["configuration"].get("audit_misses") is False
            and new.get("query_trace") == [] and new.get("native_shadow", {}).get("collected") is False,
            "Timing omits only the already-proven observational shadows")
        protocol._validate_one({"kind": protocol.values.KIND, "status": "passed", "mode": "audit",
            "binding": protocol._resource_binding(binding), "binding_after": protocol._resource_binding(binding),
            "failure": None, "cleanup_failures": [], "cleanup_passed": True,
            "input_bytes_unchanged": True, "loaded_owners_unchanged": True,
            "performance_attribution_valid": False, "whole_finish_authority": False,
            "rows": [protocol._call_for_common(old)]}, protocol._resource_binding(binding))
        require(old["input_binding"] == new["input_binding"]
            and protocol.values.normalized_input(old["consumed_input_binding"]) == protocol.values.normalized_input(new["consumed_input_binding"])
            and protocol.values.normalized_source(old["source_binding"]) == protocol.values.normalized_source(new["source_binding"])
            and old["terminal"] == new["terminal"]
            and not scope.compare_evidence(old["result"], new["result"], path="result"),
            "Timed actual inputs/terminal/query/update/result differ from own audit")
        protocol.values.validate_timing_accounting(new["loop_report"]["statistics"],
            new["consumed_input_binding"]["source"]["shape"][0])
        for key in totals: totals[key] += old["loop_report"]["statistics"][key]
        for key in maxima: maxima[key] = max(maxima[key], old["native_shadow"][key])
    require(totals["audited_hits"] > 0 and totals["audited_misses"] > 0,
        "Positive complete actual hit/miss CPU-shadow coverage required")
    require(timing["gpu_pair_verdict"] == audit["gpu_pair_verdict"], "Own audited pair verdict changed")
    for old, new in zip(audit["gpu_proposals"], timing["gpu_proposals"]):
        gates_equal(old["gates"], new["gates"])
    cpu, setup, gpu = (wall(timing[k]) for k in ("native_whole_proposals_wall_s",
        "shared_constructor_prepare_wall_s", "gpu_whole_proposals_wall_s"))
    subtotal, cleanup = wall(timing["gpu_cold_setup_and_proposals_wall_s"]), wall(timing["owner_cleanup_wall_s"])
    require(subtotal == setup+gpu, "Recorded cold subtotal must exactly equal setup plus proposals")
    return {"status": "passed", "source_sha256": CURRENT, "binding_sha256": digest(binding),
        "source_family_sha256": digest(binding["artifacts_sha256"]),
        "pair_binding_sha256": digest(audit["pair_binding"]),
        "proposal_count": len(audit["gpu_proposals"]), "actual_gpu_calls": len(audited),
        "native_calls": sum(len(p["calls"]) for p in timing["native_proposals"]),
        "original_gate_events": sum(len(p["gates"]) for p in timing["gpu_proposals"]),
        "audit_coverage": totals, "original_native_result_shadow_maxima": maxima,
        "canonical_correspondence_ids_equal_every_call": True,
        "original_complete_gate_witness_and_ambiguity_evidence_passed": True,
        "timing_repeated_exact_audited_inputs_terminals_and_gate_evidence": True,
        "timing": {"native_proposals_wall_s": cpu, "gpu_cold_setup_wall_s": setup,
            "gpu_proposals_wall_s": gpu, "gpu_setup_and_proposals_subtotal_s": subtotal,
            "gpu_final_owner_cleanup_s": cleanup, "gpu_subtotal_plus_cleanup_s": subtotal+cleanup,
            "gpu_subtotal_plus_cleanup_to_native_wall_ratio": (subtotal+cleanup)/cpu,
            "producer_closure_s": wall(timing["producer_closure_wall_s"]),
            "producer_wall_s_before_final_publication": wall(timing["producer_wall_s_before_final_publication"])},
        "whole_finish_authority": False}


def run(args):
    output, a = (Path(value).resolve() for value in (args.output, args.audit))
    timings = [Path(value).resolve() for value in args.timing]
    require(output.is_relative_to(ROOT/"benchmark-output") and not output.exists(), "Fresh private summary path required")
    require(1 <= len(timings) <= 5 and len(set(timings)) == len(timings)
        and all(p.is_relative_to(ROOT/"benchmark-output") for p in [a]+timings),
        "Only bounded unique locally owned report receipts accepted")
    record = {"kind": KIND, "status": "running", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "performance_authority_token_created": False, "whole_finish_authority": False,
        "fixed_resource_bytes_rehashed_by_summary": False, "os_child_exit_independently_verified": False,
        "timer_scope": "Cold setup+proposals is a subtotal. Adding final GPU owner cleanup gives the declared numerator of the native wall ratio; preflight, proof validation and final producer closure remain separate. Proposal timers include original gates, per-call graph/state/copies/completion, validation and progress publication. Nested timers are not subtracted for a speed claim.",
        "limitations": ["One prepared original-ranked pair; no Live, frontier, optimizer, fusion, mesh or whole-Finish speed authority.",
            "Only recorded before/after private resource receipts and current source bytes are checked here. Root-owned execution metadata establishes actual process exit."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    def save(): output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save()
    try:
        ah = sha(a)
        audit = json.loads(a.read_text(encoding="utf-8"))
        samples, fingerprints = [], []
        for index, t in enumerate(timings):
            th = sha(t)
            timing = json.loads(t.read_text(encoding="utf-8"))
            require(Path(timing["audit_proof"]["path"]).resolve() == a, "Timing names another closed audit")
            case = compact(audit, timing, ah)
            measurement = case.pop("timing")
            if index == 0: record["case"] = case
            else: require(case == record["case"], "Matched timing scope/count/quality changed between samples")
            require(sha(a) == ah and sha(t) == th, "Measured report receipt changed during publication")
            samples.append({"sample_index": index, **measurement}); fingerprints.append(th)
        record["timing_samples"] = samples
        record["timing_group"] = {"samples": len(samples),
            "median_native_proposals_wall_s": statistics.median(s["native_proposals_wall_s"] for s in samples),
            "median_gpu_subtotal_plus_cleanup_s": statistics.median(s["gpu_subtotal_plus_cleanup_s"] for s in samples),
            "median_matched_gpu_to_native_wall_ratio": statistics.median(s["gpu_subtotal_plus_cleanup_to_native_wall_ratio"] for s in samples)}
        record["report_sha256"] = {"audit": ah, "timing": fingerprints}
        record["publisher_source_family_sha256"] = digest({p: sha(ROOT/p) for p in OWN_FILES})
        record["status"] = "passed"; save()
    except BaseException as error:
        record["status"] = "failed"; record["failure"] = {"type": type(error).__name__, "message": str(error)}
        try: save()
        except BaseException as secondary: raise error from secondary
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--timing", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
