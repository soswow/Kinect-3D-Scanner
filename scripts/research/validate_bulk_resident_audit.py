"""Separate CPU-auditor authority, never field pose or GPU-target authority.

The complete original nine-proposal trajectory must close simultaneous scalar
and bulk original-native shadows. Later field queries still all receive native
CPU shadows, scalar ambiguity handling and complete original CPU ICP shadows.
This module uses only the standard library.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from dataclasses import dataclass
import json
import math
from scripts.research import validate_device_flat_grid_proof as device_proof
from scripts.research.validate_finish_resident_proof import read_json,closed_file
from scripts.research.archive.validate_uniform_grid_proof import (
    GridProofError,canonical_hash,file_hash,require,positive,zero)

KIND = "complete-old-resident-scalar-bulk-dual-audit"
TRAVERSAL = "device-flat-resident-dual-cpu-shadow-v1"
POLICY = "original-legacy-bulk-shadow-v1"
EXPECTED_QUERIES = 104123989
BULK_ARTIFACTS = ("scripts/research/bulk_legacy_nn_audit.py",
    "scripts/research/cuda_bulk_audit_grid_registration.py",
    "scripts/research/benchmark_bulk_resident_audit.py",
    "scripts/research/validate_bulk_resident_audit.py")
NATIVE_KEYS = ("python","python_executable","numpy","numpy_file","open3d",
    "selected_modules","native_binaries","numpy_native_binaries")


def finite_nonnegative(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value >= 0


def validate_api_cases(api, tracked):
    require(api.get("new_field_bulk_only_shadow_authorized") is False and api.get("gpu_queries_run") is False,
            "The bounded API report cannot claim field/GPU authority")
    require(api.get("trace_binding") == api.get("trace_binding_after"),"Retained old real trace changed")
    for prefix in ("trace","manifest","capture"):
        row=api["trace_binding"]
        closed_file({"path":row[prefix+"_path"],"sha256":row[prefix+"_sha256"]},tracked)
    reference=api["source_reference"]
    closed_file({"path":reference["manifest"],"sha256":reference["sha256"]},tracked)
    require(len(reference["sources"]) == 4,"Official source reference coverage absent")
    for source in reference["sources"]:
        closed_file({"path":source["snapshot"],"sha256":source["sha256"]},tracked)
    require(api.get("normalization_contracts",{}).get("passed") is True,"CPU audit normalization/lifetime contracts absent")
    synthetic=api["synthetic_cases"];real=api["real_cases"]
    require(len(synthetic) == 932 and len(real) == 12,"Initial actual scalar/bulk cases incomplete")
    require(sum(x["query_rows"] for x in synthetic) == 5924 and sum(x["query_rows"] for x in real) == 24576,
            "Initial scalar/bulk query coverage incomplete")
    for group, chunks in ((synthetic,[65536]),(real,[257,4096,65536])):
        for case in group:
            require(case.get("all_original_rows_checked") is True and case.get("arrays_unchanged") is True
                    and positive(case.get("query_rows")) and positive(case.get("target_rows"))
                    and finite_nonnegative(case.get("scalar_wall_s")),"Actual original scalar case absent")
            require([x.get("chunk_rows") for x in case["bulk_runs"]] == chunks,"Actual bulk chunk coverage absent")
            for row in case["bulk_runs"]:
                require(zero(row.get("id_errors")) and zero(row.get("diagnostic_metric_bit_errors"))
                        and row.get("query_rows") == case["query_rows"] and row.get("target_rows") == case["target_rows"]
                        and positive(row.get("evaluate_calls")) and finite_nonnegative(row.get("all_in_wall_s")),
                        "Actual original scalar/bulk IDs or metric bits changed")


def validate_dual_synthetics(report):
    cases=report["synthetic_indices"]
    require(len(cases)>=16 and all(positive(c.get("queries")) and zero(c.get("index_mismatches"))
        and zero(c.get("device_adapter_index_mismatches"))
        and finite_nonnegative(c.get("device_adapter_max_squared_distance_delta"))
        and c["device_adapter_max_squared_distance_delta"]<=1e-12 for c in cases),
        "Actual new host/device nearest synthetic coverage incomplete")
    boundaries=report["synthetic_grid_boundaries"]
    require(len(boundaries)>=43 and all(positive(c.get("queries")) and zero(c.get("index_mismatches")) for c in boundaries)
        and any(c.get("case")=="original-double-subnormal-and-minimum-radius" for c in boundaries),
        "Actual signed/minimum-radius/subnormal proof absent")
    device_proof.device_audit_coverage(report["synthetic_statistics"])
    checks=report["device_flat_synthetic"]
    require(checks.get("passed") is True and set(checks.get("required_checks",[]))==set(device_proof.DEVICE_CHECKS)
        and all(checks.get(k) is True for k in device_proof.DEVICE_CHECKS),"Device classifier/ownership proof failed")
    require(checks.get("classifier_cases")==32 and checks.get("classifier_rows")==(127+128+129+257)*8
        and positive(checks.get("packet_rows")) and all(positive(checks.get("positive_mask_coverage",{}).get(k))
        for k in ("unsupported","uncertain","audited_hits","audited_misses")),"Actual classifier packet/role coverage absent")
    owned=report["flat_cache_ownership"]
    require(owned.get("passed") is True and positive(owned.get("cloud_evictions")) and all(owned.get(k) is True for k in
        ("bounded_retained_bytes","clear_releases_all_owned_arrays","one_byte_budget_original_cpu_fallback")),
        "Inherited ownership/eviction/budget proof absent")


@dataclass(frozen=True)
class BulkResidentAuditAuthority:
    dual_report_path: str
    dual_report_sha256: str
    api_report_path: str
    api_report_sha256: str
    runtime_binding_json: str
    fixture_binding_sha256: str
    artifact_sha256: tuple

    @property
    def runtime_binding(self):
        return json.loads(self.runtime_binding_json)


def validate_native_runtime(runtime,tracked):
    require(isinstance(runtime,dict) and all(runtime.get(k) for k in NATIVE_KEYS),"Actual selected CPU/NumPy/Python identities absent")
    for name in ("native_binaries","numpy_native_binaries"):
        binaries = runtime[name]
        require(isinstance(binaries,dict) and binaries,"Selected loaded native binaries absent")
        for path,item in binaries.items():
            actual = closed_file(dict(item,path=path),tracked)
            require(type(item.get("bytes")) is int and item["bytes"] > 0 and actual.stat().st_size == item["bytes"],
                    "Selected loaded native binary extent changed")


def validate_bulk_records(records,stats,targets):
    """Every actual old hit and miss must agree before correction, bit for bit."""
    require(isinstance(records,list) and records,"Complete old dual records absent")
    count = hits = misses = calls = 0
    for record in records:
        require(record.get("complete") is True and record.get("bulk_domain_supported") is True
                and record.get("target_unchanged") is True,"Unsupported, incomplete or changed dual target")
        query = record["audit_query_descriptor"]
        require(set(query) == {"dtype","shape","bytes","sha256"} and query["dtype"] == "<f8"
                and isinstance(query["shape"],list) and len(query["shape"]) == 2
                and all(type(x) is int for x in query["shape"])
                and positive(query["shape"][0]) and query["shape"][0] <= 1_000_000 and query["shape"][1] == 3
                and type(query["bytes"]) is int and query["bytes"] == 24*query["shape"][0]
                and device_proof.digest(query["sha256"]),"Malformed original dual query descriptor")
        require(record.get("target_digest") in targets,"Old dual trajectory target membership incomplete")
        require(all(device_proof.digest(record.get(key)) for key in
            ("scalar_ids_sha256","bulk_ids_sha256","scalar_squared_sha256","bulk_squared_sha256"))
            and record["scalar_ids_sha256"] == record["bulk_ids_sha256"]
            and record["scalar_squared_sha256"] == record["bulk_squared_sha256"]
            and zero(record.get("id_mismatches")) and zero(record.get("metric_bit_mismatches")),
            "Original scalar/bulk IDs or diagnostic metric bits changed")
        a,b = record["audit_hit_rows"],record["audit_miss_rows"]
        require(type(a) is int and type(b) is int and a >= 0 and b >= 0 and a+b == query["shape"][0],"Dual hit/miss coverage incomplete")
        require(isinstance(record.get("radius_m"),(int,float)) and not isinstance(record["radius_m"],bool)
                and math.isfinite(record["radius_m"]) and record["radius_m"] in (.03,.06,.12),"Original stage radius changed")
        require(positive(record.get("evaluate_calls")) and finite_nonnegative(record.get("all_in_wall_s")),
                "Native bulk execution accounting absent")
        count += a+b; hits += a; misses += b; calls += record["evaluate_calls"]
    require(count == stats["query_rows"] == stats.get("dual_scalar_bulk_queries") == stats.get("bulk_shadow_queries")
            and hits == stats["audited_hits"] == stats.get("bulk_shadow_hits")
            and misses == stats["audited_misses"] == stats.get("bulk_shadow_misses")
            and calls == stats.get("bulk_evaluate_calls"),"Complete dual row/hit/miss/native-call accounting differs")
    require(all(zero(stats.get(key)) for key in ("dual_id_mismatches","dual_metric_bit_mismatches",
        "bulk_domain_scalar_queries","exact_cpu_queries","device_malformed_results")),"Old full dual proof contains unsupported/fallback/error rows")


def validate_bulk_resident_audit(path,expected_bindings,expected_fixture_binding):
    try:
        path = Path(path).resolve(strict=True)
        before = file_hash(path)
        tracked = {path:before}
        report = read_json(path)
        require(report.get("kind") == KIND and report.get("traversal") == TRAVERSAL,
                "Old component/API-only report cannot authorize the new CPU auditor")
        artifacts = device_proof.validate_configuration(expected_bindings)
        require(set(BULK_ARTIFACTS).issubset(artifacts),"New helper/adapter/producer/validator pins absent")
        solve_manifest = device_proof.validate_current_artifacts(expected_bindings,artifacts)
        # Explicit new-kind check precedes reuse of unchanged common numerical,
        # cleanup, runtime and original source guards. No saved file is relabelled.
        device_proof.common_checks(dict(report,kind=device_proof.KIND,traversal=device_proof.TRAVERSAL),
            expected_bindings,artifacts,solve_manifest)
        cfg = expected_bindings["bulk_audit_configuration"]
        require(cfg == {"policy":POLICY,"audit_mode":"dual","chunk_rows":65536},"Complete old dual policy changed")
        bulk = report["bulk_audit"]
        require(all(bulk.get(k) == v for k,v in cfg.items()) and bulk.get("new_field_bulk_shadow_authorized_by_report_alone") is False,
                "CPU auditor evidence must not claim field pose or GPU authority")
        runtime = expected_bindings["bulk_cpu_runtime"]
        validate_native_runtime(runtime,tracked)
        require(report.get("bulk_api_proof") == report.get("bulk_api_proof_after") == expected_bindings["bulk_api_proof"],"API proof changed")
        api_path = closed_file(report["bulk_api_proof"],tracked)
        api = read_json(api_path)
        require(api.get("kind") == "scalar-vs-original-legacy-bulk-nearest-api-crosscheck" and api.get("status") == "passed"
                and api.get("script_sha256") == api.get("script_sha256_after") == artifacts[BULK_ARTIFACTS[0]]
                and api.get("runtime") == api.get("runtime_after") and not api.get("failure"),"Initial actual-native API proof failed or changed")
        require(all(api["runtime"].get(k) == runtime.get(k) for k in NATIVE_KEYS),"Complete dual audit uses different loaded native binaries")
        require(api["synthetic_summary"].get("query_rows") == 5924 and api["synthetic_summary"].get("all_ids_and_metric_bits_exact") is True
                and api["real_summary"].get("query_rows") == 24576 and all(api["real_summary"].get(key) is True for key in
                    ("all_retained_rows_scalar_checked","stored_original_gold_equal","all_ids_and_metric_bits_exact")),"Actual original CPU API crosscheck lacks exact coverage")
        validate_native_runtime(api["runtime"],tracked)
        validate_api_cases(api,tracked)
        # The API pilot recorded its own environment. The full dual trajectory
        # freshly proves the canonical OMP8 policy; binary identity stays exact.
        require(runtime["thread_environment"].get("OMP_NUM_THREADS") == "8"
                and bulk.get("api_native_identity_match") is True,"Complete dual policy/binary crosscheck absent")
        require(report.get("synthetic_only") is False and report.get("synthetic_device_adapter_checked") is True,
                "A complete old real trajectory is mandatory")
        validate_dual_synthetics(report)
        require(report.get("fixture_binding") == expected_fixture_binding
                and report.get("fixture_binding_sha256") == canonical_hash(expected_fixture_binding),"Old raw/fixture/task scope differs")
        for key in ("fixture_sha256","reference_sha256","original_raw_input_sha256"):
            require(report.get(key) == report.get(key+"_after") == expected_fixture_binding[key],"Fixture/raw/reference changed")
        require(report.get("fixture_inputs")==report.get("fixture_inputs_after"),"Original input path/hash closure changed")
        for name,key in (("fixture","fixture_sha256"),("reference","reference_sha256"),("raw","original_raw_input_sha256")):
            require(report["fixture_inputs"][name]["sha256"]==expected_fixture_binding[key],"Original input hash binding differs")
            closed_file(report["fixture_inputs"][name],tracked)
        require(report.get("fixture_arrays_unchanged") is True and report.get("cloud_arrays_unchanged") is True
                and report.get("original_fixture_authority",{}).get("same_decisions_and_support") is True
                and report.get("fixture_provenance",{}).get("local_pose_source") == "measured Finish fragment report; no archived ZIP poses",
                "Original native raw-derived fixture authority incomplete")
        tasks = expected_fixture_binding["tasks"]
        require([(t["position"],t["pair"],len(t["proposal_sha256"])) for t in tasks] == [(0,[8,12],5),(4,[0,2],4)],"Original nine proposals changed")
        runs = report["real_runs"]
        require([(r.get("mode"),r.get("repeat")) for r in runs] == [("native_cpu",0),("grid",0)],"Complete dual native/grid run order differs")
        targets = report["real_target_membership_sha256"]
        require(isinstance(targets,list) and targets and len(set(targets)) == len(targets)
                and all(device_proof.digest(x) for x in targets),"Old dual target membership absent")
        for run in runs:
            require(run.get("complete") is True and len(run["pairs"]) == len(tasks),"Incomplete old original pair run")
            for pair,task in zip(run["pairs"],tasks):
                proposals = pair["proposal_results"]
                require(pair.get("position") == task["position"] and pair.get("pair") == task["pair"]
                        and [p.get("proposal_index") for p in proposals] == list(range(len(task["proposal_sha256"])))
                        and [p.get("input_sha256") for p in proposals] == task["proposal_sha256"]
                        and pair.get("verdict",{}).get("accepted") is (task["position"] == 0)
                        and pair.get("pair_verdict_same") is True,"Original ordered proposal/gate authority changed")
                if run["mode"] == "grid":
                    require(all(p.get("quality",{}).get("passed") is True for p in proposals),"Original witness/pose/information gates failed")
            if run["mode"] == "grid":
                stats = run["statistics_delta"]
                device_proof.device_audit_coverage(stats,complete_trajectory=True)
                require(stats["query_rows"] == EXPECTED_QUERIES,"Not every original trajectory query was doubly shadowed")
                require(device_proof.resident_checks(run,expected_bindings,artifacts)==report["solve_metadata"],
                        "Actual resident solve metadata changed within the dual trajectory")
                require(run.get("bulk_records_sha256") == canonical_hash(run["bulk_records"]),"Closed dual record fingerprint differs")
                validate_bulk_records(run["bulk_records"],stats,set(targets))
                require(positive(stats.get("resident_xyz_uploads")),"Original-order resident XYZ never uploaded")
        require(bulk.get("records_sha256") == canonical_hash(bulk["records"])
                and bulk["records"][-len(runs[1]["bulk_records"]):] == runs[1]["bulk_records"],"Closed complete dual records differ from actual real suffix")
        for name,value in artifacts.items():
            require(file_hash(ROOT/name) == value,"Bulk auditor dependency changed")
        require(all(file_hash(p) == value for p,value in tracked.items()),"Closed dual/API/native bytes changed")
        return BulkResidentAuditAuthority(str(path),before,str(api_path),report["bulk_api_proof"]["sha256"],
            json.dumps(expected_bindings,sort_keys=True,separators=(",",":"),allow_nan=False),
            canonical_hash(expected_fixture_binding),tuple(sorted(artifacts.items())))
    except GridProofError:
        raise
    except (KeyError,TypeError,ValueError,AttributeError,IndexError,OSError) as error:
        raise GridProofError(f"Incomplete/malformed/stale dual CPU auditor proof: {error}") from error


def validate_authority_for_bulk_shadow(authority,*,device,chunk_rows,native_runtime):
    """Before GPU init: compatibility only, no old field target whitelist."""
    require(type(authority) is BulkResidentAuditAuthority,"A distinct complete-old dual token is required")
    binding = authority.runtime_binding
    require(str(device) == binding["resident_configuration"]["device"] == "CUDA:0"
            and type(chunk_rows) is int and chunk_rows == binding["bulk_audit_configuration"]["chunk_rows"] == 65536,
            "Bulk CPU auditor device/chunk policy changed")
    require(native_runtime == binding["bulk_cpu_runtime"],"Actual loaded CPU/NumPy/Python policy differs from dual proof")
    require(file_hash(Path(authority.dual_report_path)) == authority.dual_report_sha256
            and file_hash(Path(authority.api_report_path)) == authority.api_report_sha256,"Closed auditor reports changed")
    require(dict(authority.artifact_sha256) == device_proof.normalized_artifacts(binding["artifacts_sha256"]),
            "Auditor token/dependency pins differ")
    device_proof.validate_current_artifacts(binding,device_proof.validate_configuration(binding))
    validate_native_runtime(native_runtime,{})
    fresh=validate_bulk_resident_audit(Path(authority.dual_report_path),binding,
        read_json(Path(authority.dual_report_path))["fixture_binding"])
    require(fresh==authority,"A reconstructed or malformed token cannot bypass complete dual proof validation")
    return True
