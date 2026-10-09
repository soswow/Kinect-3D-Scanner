"""Actual-input candidate method evidence and independent Final observables.

A positive exhaustive audit qualifies only these current source/runtime/budget
bytes for this research experiment. Later calls have newly consumed inputs;
they are not claimed to replay that audit or to prove a general domain. Shadow
mode adds an original CPU result on every new actual input but does not audit
nearest queries. Final quality uses a separate unobserved native reference.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import socket
import struct
import sys
from types import CodeType, FunctionType, MappingProxyType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import compare_gpu_icp_finishes as values
from scripts.research import gpu_icp_device_loop_protocol as loop

KIND = "gpu-icp-whole-finish-candidate-observable-quality-v1"
PREFIX = "gpu-icp-whole-finish-candidate-"
CONFIGURATION = MappingProxyType({"device": 0, "cuda_graph": True, "chunk_iterations": 4,
    "max_points": 1_000_000, "max_scratch_bytes": 256*1024**2,
    "max_total_bytes": 512*1024**2, "pair_cache_bytes": 256*1024**2, "max_jobs": 4096})
FILES = ("scripts/research/gpu_icp_finish_candidate.py", "tests/test_gpu_icp_finish_candidate.py",
    "scripts/research/profile_gpu_icp_finish_candidate.py", "tests/test_gpu_icp_finish_candidate_profile.py",
    "scripts/research/compare_gpu_icp_candidate_finishes.py", "tests/test_gpu_icp_candidate_finish_quality.py")
PURE_FILES = ("scripts/research/compare_gpu_icp_finishes.py", "scripts/research/gpu_icp_device_loop_protocol.py",
    "shared/surface_metrics.py")
_REGISTRY = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            result.update(part)
    return result.hexdigest()


def code_state(code):
    return (code.co_code, tuple(code_state(c) if isinstance(c, CodeType) else c for c in code.co_consts),
        code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
        code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags)


class LoadedOwners:
    """Small source-compiled pure-helper guard, without numerical imports."""
    def __init__(self):
        self.records = []
        for module, names in ((sys.modules[__name__], None),
                (values, ("matrix", "pose_values", "pose_deltas", "retained_graphs", "number", "canonical", "require")),
                (loop, ("normalized_input", "normalized_source", "descriptor", "terminal_contract",
                    "validate_timing_accounting", "is_digest", "require", "canonical"))):
            compiled = compile(Path(module.__file__).read_text(encoding="utf-8"), module.__file__, "exec", dont_inherit=True)
            codes = {c.co_name: c for c in compiled.co_consts if isinstance(c, CodeType)}
            for name in (names or tuple(codes)):
                owner = getattr(module, name, None)
                methods = [(module,name,owner,codes[name])]
                if isinstance(owner,type):
                    methods = [(owner,c.co_name,getattr(owner,c.co_name,None),c)
                        for c in codes[name].co_consts if isinstance(c,CodeType)]
                for slot,method_name,fn,expected in methods:
                    if not isinstance(fn,FunctionType):continue
                    require(fn.__globals__ is module.__dict__ and code_state(fn.__code__) == code_state(expected),
                        "Loaded candidate qualification helper differs from its source")
                    aliases=tuple((key,module.__dict__[key]) for key in fn.__code__.co_names if key in module.__dict__)
                    self.records.append((module,slot,method_name,fn,fn.__code__,repr(fn.__defaults__),repr(fn.__kwdefaults__),aliases))
        self.records=tuple(self.records)
        self.check()

    def check(self):
        for module, slot, name, fn, code, defaults, keywords, aliases in self.records:
            require(sys.modules.get(module.__name__) is module and getattr(slot,name) is fn
                and fn.__globals__ is module.__dict__ and fn.__code__ is code
                and repr(fn.__defaults__) == defaults and repr(fn.__kwdefaults__) == keywords
                and all(module.__dict__.get(key) is owner for key,owner in aliases),
                "Loaded candidate qualification owner/code/default changed")


def normalized_artifacts(binding):
    result = {}
    for name, digest in binding["artifacts_sha256"].items():
        normalized = name.replace("\\", "/")
        path = Path(normalized)
        require(not path.is_absolute() and ":" not in normalized and ".." not in path.parts
            and normalized not in result and loop.is_digest(digest), "Bounded unique source paths required")
        result[normalized] = digest
    return result


def check_sources(binding):
    from scripts.profile_session import source_hash
    require(loop.is_digest(binding.get("source_sha256")) and binding["source_sha256"] == source_hash(), "Cold-bound current production source changed")
    artifacts = normalized_artifacts(binding)
    require(all(name in artifacts for name in FILES+PURE_FILES), "Every new candidate and consumed quality helper must be pinned")
    for name, digest in artifacts.items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == digest, "Current candidate helper bytes changed: "+name)
    fixed = binding["fixed_files"]
    require(fixed and all(loop.is_digest(h) and sha(p) == h for p, h in fixed.items()), "Actual raw/native/compiler resources changed")
    runtime = binding["runtime"]
    require(runtime.get("device") == "CUDA:0" and runtime.get("thread_policy") ==
        {"open3d": 20, "opencv": 20, "omp": "8"}, "Original actual selected runtime/threads required")
    for record in runtime["binaries"].values():
        require(fixed.get(record["path"]) == record["sha256"], "Loaded native binary must belong to rehashed resources")


def closed_candidate(report):
    mode, registration = report.get("mode"), report.get("registration", {})
    require(mode in ("audit", "shadow", "measure") and report.get("kind") == PREFIX+mode+"-v1"
        and report.get("status") == "passed" and report.get("failure") is None
        and report.get("cleanup_passed") is True and report.get("cleanup_failures") == []
        and report.get("binding") == report.get("binding_after")
        and registration.get("mode") == mode and registration.get("complete") is True
        and registration.get("restored") is True and registration.get("closed") is True
        and registration.get("failure") is None and registration.get("cleanup_failures") == []
        and registration.get("successful_builds") == 1 and registration.get("default_evidence") is True,
        "Only fully closed new candidate actual-input Finish reports are evidence")
    require(loop.is_digest(report["binding"].get("source_sha256")), "Closed actual candidate core digest required")
    configuration_contract(registration.get("configuration"))
    contract=report.get("candidate_source",{})
    method=contract.get("candidate_method")
    require(type(contract) is dict and contract == report.get("candidate_source_after")
        and type(method) is dict and contract.get("configuration") == dict(CONFIGURATION) and method.get("configuration") == dict(CONFIGURATION)
        and method.get("policy") == "bridge-only-original-device-loop-minimal-candidate-v1"
        and method.get("historical_trajectory_authority") is False and method.get("general_domain_proof") is False,
        "Candidate source contract must close with the actual method configuration")
    require(report.get("calls") == registration.get("calls"), "Actual candidate call receipt copies differ")
    if mode=="measure":
        receipt=report.get("method_qualification",{})
        require(receipt.get("required") is True and receipt.get("closed") is True and receipt.get("failure") is None
            and loop.is_digest(receipt.get("report_sha256")) and receipt["report_sha256"]==receipt.get("report_sha256_after")
            and receipt.get("historical_trajectory_authority") is False and receipt.get("general_domain_authority") is False,
            "Measure requires its own closed unchanged exhaustive method qualification receipt")
    return registration


def native_shadow(row):
    shadow = row["native_shadow"]
    require(type(shadow) is dict and shadow.get("passed") is True and shadow.get("correspondence_ids_equal") is True,
        "Every actual audited/shadowed GPU call requires original CPU correspondence equality")
    a, b = shadow["native"], shadow["candidate"]
    left, right = values.matrix(a["transformation"]), values.matrix(b["transformation"])
    differences = {"transform_max_abs_delta": max(abs(x-y) for ar, br in zip(left, right) for x, y in zip(ar, br)),
        "fitness_abs_delta": abs(values.number(a["fitness"])-values.number(b["fitness"])),
        "rmse_abs_delta": abs(values.number(a["inlier_rmse"])-values.number(b["inlier_rmse"]))}
    require(a["correspondence_mapping"] == b["correspondence_mapping"]
        and all(shadow.get(key) == delta and delta <= 1e-8 for key, delta in differences.items()),
        "Original CPU result metrics/canonical IDs fail the existing tight evidence bounds")
    return differences


def consumed_domain(consumed, source, configuration):
    configuration_contract(configuration)
    normalized = loop.normalized_input(consumed)
    cfg = normalized["configuration"]
    require(cfg["device"] == "CUDA:0" and cfg["max_points"] == configuration["max_points"]
        and cfg["max_scratch_bytes"] == configuration["max_scratch_bytes"]
        and cfg["max_total_bytes"] == configuration["max_total_bytes"]
        and cfg["cuda_graph"] is configuration["cuda_graph"]
        and cfg["chunk_iterations"] == configuration["chunk_iterations"], "Current actual lane budgets/configuration changed")
    normalized_source = loop.normalized_source(source)
    require(normalized_source.get("device") == "CUDA:0" and normalized_source.get("cuda_graph") is True,
        "Actual current selected evaluator provenance changed")
    return normalized, normalized_source


def configuration_contract(configuration):
    expected={"device":0,"cuda_graph":True,"chunk_iterations":4,"max_points":1_000_000,
        "max_scratch_bytes":268435456,"max_total_bytes":536870912,"pair_cache_bytes":268435456,"max_jobs":4096}
    require(canonical(dict(CONFIGURATION))==canonical(expected) and type(configuration) is dict
        and canonical(configuration)==canonical(expected), "Candidate effective numeric/boolean configuration changed")
    return configuration


def call_evidence(row, mode):
    require(row.get("complete") is True and row.get("input_bytes_unchanged") is True
        and row.get("cleanup_failures") == [], "Actual GPU call/input/cleanup must close")
    consumed = row["consumed_input_binding"]
    normalized, source = consumed_domain(consumed, row["source_binding"], dict(CONFIGURATION))
    n, m = normalized["source"]["shape"][0], normalized["target"]["shape"][0]
    full=row["input_binding"]
    for side in ("source","target"):
        points=full[side]["points"]
        require({k:points[k] for k in ("dtype","shape","sha256")}==consumed[side], "Original full cloud values differ from actually consumed points")
        for kind in ("points","normals","colors"):
            descriptor=full[side][kind]
            loop.descriptor({k:descriptor[k] for k in ("dtype","shape","sha256")},3,"<f8")
            require(descriptor["nbytes"]==math.prod(descriptor["shape"])*8
                and descriptor["shape"][0] in (0,points["shape"][0]), "Original full cloud descriptor malformed")
    require({k:full["target"]["normals"][k] for k in ("dtype","shape","sha256")}==consumed["normals"]
        and {k:full["seed"][k] for k in ("dtype","shape","sha256")}==consumed["seed"]
        and full["seed"]["nbytes"]==128, "Original target normals/unrounded seed differ from consumed values")
    loop.terminal_contract(row["terminal"], n, m)
    result=row["result"]
    pose=values.matrix(result["transformation"])
    require({k:result["pose"][k] for k in ("dtype","shape","sha256")}==row["terminal"]["pose"]
        and result["pose"]["nbytes"]==128 and result["pose"]["sha256"]==hashlib.sha256(struct.pack("<16d",*(v for r in pose for v in r))).hexdigest()
        and {k:result["raw_correspondences"][k] for k in ("dtype","shape","sha256")}==row["terminal"]["correspondences"]
        and result["raw_correspondences"]["nbytes"]==8*row["terminal"]["correspondences"]["shape"][0]
        and result["fitness"]==row["terminal"]["fitness"] and result["inlier_rmse"]==row["terminal"]["inlier_rmse"],
        "Actual returned pose/pairs/metrics do not match terminal bytes")
    report, stats = row["loop_report"], row["loop_report"]["statistics"]
    require(report.get("closed") is True and report.get("failure") is None
        and report.get("input_binding") == consumed and loop.normalized_source(report["provenance"]) == source,
        "Selected lane closure/input/provenance mismatch")
    require(stats.get("queries") == row["terminal"]["queries"] and stats.get("updates") == row["terminal"]["updates"],
        "Actual query/update counts differ from terminal receipt")
    if mode == "audit":
        require(consumed["configuration"]["audit_nearest"] is True and consumed["configuration"]["audit_misses"] is True,
            "Exhaustive audit flags required")
        keys = ("query_rows", "direct_hits", "direct_misses", "audited_hits", "audited_misses", "cpu_ambiguity_rows", "solve_blocks", "flagged_rows", "packet_bytes")
        require(all(type(stats.get(k)) is int and stats[k] >= 0 for k in keys)
            and stats["query_rows"] == stats["queries"]*n
            and stats["direct_hits"] == stats["audited_hits"] and stats["direct_misses"] == stats["audited_misses"]
            and stats["query_rows"] == stats["audited_hits"]+stats["audited_misses"]+stats["cpu_ambiguity_rows"]
            and stats["flagged_rows"]==stats["query_rows"] and stats["packet_bytes"]==64*stats["query_rows"]
            and stats["solve_blocks"] == 0, "Incomplete exhaustive nearest/ambiguity audit accounting")
        trace = row["query_trace"]
        require(len(trace) == stats["queries"] and row["query_trace_sha256"] == hashlib.sha256(canonical(trace).encode()).hexdigest(),
            "Exhaustive actual query trace incomplete")
        for index, query in enumerate(trace):
            require(query["query_index"] == index and type(query["stage"]) is int and query["stage"] in (0, 1, 2)
                and query["radius"] == (.12,.06,.03)[query["stage"]]
                and query["target_sha256"] == consumed["target"]["sha256"], "Actual query stage/target mismatch")
            packet=query["packet"]
            loop.descriptor({k:packet[k] for k in ("dtype","shape","sha256")}, 8, "<f8")
            require(packet["nbytes"]==64*n,"Actual NN packet byte count changed")
            require(query["packet"]["shape"][0] == n and loop.is_digest(query["corrected_ids_sha256"]), "Actual query rows missing CPU audit")
        stages = [q["stage"] for q in trace]
        require(stages == sorted(stages) and set(stages) == {0,1,2}
            and all(2 <= stages.count(s) <= maximum for s, maximum in enumerate((41,31,21))), "Original stage order/budgets changed")
    else:
        require(consumed["configuration"]["audit_nearest"] is False and row["query_trace"] == [], "Nonaudit route cannot claim NN shadow traces")
        loop.validate_timing_accounting(stats, n)
    if mode in ("audit", "shadow"):
        native_shadow(row)
        require(row["result"]==row["native_shadow"]["candidate"],"Actual returned candidate differs from CPU-shadow candidate")
        if mode == "shadow":
            certificate=row.get("cpu_first_certificate",{})
            require(row["authorization"].get("cpu_first") is True and certificate.get("completed") is True
                and certificate.get("input_bytes_unchanged") is True and certificate.get("input")==consumed
                and certificate.get("result")==row["native_shadow"]["native"], "Native-shadow mode requires fresh CPU result before GPU")
    else:
        require(row.get("native_shadow") in (None,{"collected":False}), "Measure cannot claim unperformed CPU shadows")
    return stats


def report_evidence(report, positive=False):
    registration = closed_candidate(report)
    rows, mode = registration["calls"], report["mode"]
    require(type(rows) is list and len(rows) <= CONFIGURATION["max_jobs"]
        and [r.get("call_index") for r in rows] == list(range(len(rows))), "Complete bounded actual call order required")
    counts = {k: 0 for k in ("queries", "query_rows", "audited_hits", "audited_misses", "cpu_ambiguity_rows")}
    for row in rows:
        stats = call_evidence(row, mode)
        actual_source=loop.normalized_source(row["source_binding"])
        require(all(actual_source.get(key)==value for key,value in report["binding"]["method_source"].items()),
            "Actual evaluated helper/shader differs from the closed declared method")
        artifacts=normalized_artifacts(report["binding"])
        require(all(artifacts.get(name.replace("\\","/"))==digest for name,digest in actual_source["artifacts"].items()),
            "Actual evaluated helper/shader is absent from current source closure")
        for key in counts: counts[key] += stats[key]
    workspace = registration.get("workspace")
    require((not rows and workspace is None and registration.get("cache_receipts") == [])
        or (rows and type(workspace) is dict and workspace.get("closed") is True
            and workspace.get("failure") is None and len(workspace.get("jobs", [])) == len(rows)
            and workspace.get("template_started") is False and workspace.get("active_lane") is False
            and all(job.get("index")==index and job.get("closed") is True and job.get("report")==rows[index]["loop_report"]
                for index,job in enumerate(workspace["jobs"]))
            and registration["cache_receipts"] and all(cache.get("closed") is True and cache.get("failure") is None
                and cache.get("active_streams")==0 and type(cache.get("owned_numeric_bytes")) is int
                and 0<=cache["owned_numeric_bytes"]<=CONFIGURATION["pair_cache_bytes"]
                for cache in registration["cache_receipts"])),
        "Actual workspace/pair-cache completion required before accepted evidence")
    if positive:
        require(mode == "audit" and rows and counts["audited_hits"] > 0 and counts["audited_misses"] > 0,
            "Method qualification requires a positive exhaustive actual hit AND miss audit")
    return {"actual_gpu_calls": len(rows), **counts,
        "exhaustive_nn_audit": mode == "audit", "actual_cpu_terminal_shadows": len(rows) if mode != "measure" else 0,
        "new_inputs_not_exact_audit_replay": mode != "audit", "general_domain_authority": False}


@dataclass(frozen=True)
class CandidateMethodQualification:
    report_path: str
    report_sha256: str
    binding_json: str
    configuration_json: str


def current_candidate_contract():
    from scripts.research import profile_gpu_icp_finish_candidate as controller
    return controller.source_contract()


def qualify_candidate_method(path, expected_binding, configuration=None):
    configuration = dict(CONFIGURATION) if configuration is None else configuration
    configuration_contract(configuration)
    owners = LoadedOwners()
    check_sources(expected_binding)
    path = Path(path).resolve(strict=True)
    before = sha(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report.get("binding") == expected_binding, "Own exhaustive method audit uses different source/runtime")
    require(report.get("candidate_source") == current_candidate_contract(),
        "Own exhaustive method audit differs from the actual current controller source contract")
    report_evidence(report, positive=True)
    require(sha(path) == before, "Exhaustive method audit changed during qualification")
    owners.check()
    token = CandidateMethodQualification(str(path), before, canonical(expected_binding), canonical(configuration))
    _REGISTRY[id(token)] = (token, tuple(token.__dict__.items()), owners)
    return token


def assert_registered_candidate_qualification(token):
    state = _REGISTRY.get(id(token))
    require(type(token) is CandidateMethodQualification and state is not None and state[0] is token
        and tuple(token.__dict__.items()) == state[1], "Require own unchanged freshly qualified candidate method, not an old/constructed token")
    state[2].check()
    return token


def validate_candidate_method(token, *, binding, configuration):
    assert_registered_candidate_qualification(token)
    require(canonical(binding) == token.binding_json and canonical(configuration) == token.configuration_json,
        "Current candidate source/runtime/effective policy differs from own exhaustive method audit")
    return token


def validate_candidate_call(token, *, consumed, source, configuration):
    assert_registered_candidate_qualification(token)
    require(canonical(configuration) == token.configuration_json, "Current candidate call configuration drifted")
    normalized, actual_source = consumed_domain(consumed, source, configuration)
    require(consumed["configuration"]["audit_nearest"] is False, "Method qualification only authorizes explicitly unaudited observation flags")
    expected = json.loads(token.binding_json)["method_source"]
    require(all(actual_source.get(k) == v for k, v in expected.items()), "Actual candidate evaluator source changed")
    return normalized


def qualification_receipt(token, *, close=False):
    """Hash once at terminal closure, never per actual trajectory or gate."""
    assert_registered_candidate_qualification(token)
    after=sha(token.report_path) if close else None
    require(not close or after==token.report_sha256,"Exhaustive method audit changed before candidate cleanup completed")
    return {"required":True,"report_path":token.report_path,"report_sha256":token.report_sha256,
        "report_sha256_after":after,"closed":close,"failure":None,
        "historical_trajectory_authority":False,"general_domain_authority":False,
        "scope":"Observed actual exhaustive audit of current method/source/runtime/configuration; later inputs are new."}


def published_fragment_memberships(profile):
    """Discrete published witnesses/frontier; not an internal optimizer trace."""
    report=profile.get("fragment_reconnection")
    require(type(report) is dict,"Original published fragment report missing")
    selected=profile["selected_indices"]
    require(type(selected) is list and len(selected)<=4096 and len(set(selected))==len(selected)
        and all(type(i) is int and i>=0 for i in selected),"Bounded original raw observation IDs required")
    raw=set(selected)
    def ids(items,allowed):
        require(type(items) is list and len(items)<=4096 and len(set(items))==len(items)
            and all(type(i) is int and i in allowed for i in items),"Malformed published membership IDs")
        return list(items)
    def support(items):
        require(type(items) is list and len(items)<=4096,"Bounded original witness pairs required")
        require(all(type(pair) is list and len(pair)==2 and all(type(i) is int and i in raw for i in pair)
            for pair in items),"Malformed original witness endpoints")
        return [list(pair) for pair in items]
    known=("verified_bridges","ambiguous_pairs","invalid_indices","unassigned_indices","excluded_frames",
        "connected_fragments","unconnected_fragments","recent_reference_links","loop_closures","fragments",
        "pair_budget","fragment_limit","budget_limited","applied","failed")
    result={"available_fields":[key for key in known if key in report]}
    fragments=report.get("fragments",[])
    require(type(fragments) is list and len(fragments)<=32,"Bounded original fragment inventory required")
    fragment_ids=[r["id"] for r in fragments]
    require(len(set(fragment_ids))==len(fragment_ids) and all(type(i) is int and 0<=i<32 for i in fragment_ids),"Malformed fragment IDs")
    fragments_set=set(fragment_ids)
    if "fragments" in report:
        result["fragments"]=[]
        for row in fragments:
            require(type(row["connected"]) is bool,"Original fragment connection flag required")
            result["fragments"].append({"id":row["id"],"connected":row["connected"],
                "frame_indices":ids(row["frame_indices"],raw),"context_frame_indices":ids(row["context_frame_indices"],raw)})
    def edge(row,verified):
        require(type(row) is dict and type(row["source"]) is int and type(row["target"]) is int
            and row["source"] in fragments_set and row["target"] in fragments_set and row["source"]!=row["target"],
            "Malformed published bridge fragment endpoints")
        item={"source":row["source"],"target":row["target"],"support":support(row["support"])}
        if verified:
            require(type(row["validation_scope"]) is str and type(row["connected_to_scan"]) is bool,
                "Original bridge validation/connection scope required")
            item.update(validation_scope=row["validation_scope"],connected_to_scan=row["connected_to_scan"],
                visual_constraint_available="visual_constraint" in row)
            if "visual_constraint" in row:
                require(type(row["visual_constraint"]) is bool,"Original visual constraint flag required")
                item["visual_constraint"]=row["visual_constraint"]
        return item
    for key in ("verified_bridges","loop_closures"):
        if key in report:
            require(type(report[key]) is list and len(report[key])<=512,"Bounded published bridge inventory required")
            result[key]=[edge(row,key=="verified_bridges") for row in report[key]]
    for key in ("invalid_indices","unassigned_indices","excluded_frames","connected_fragments","unconnected_fragments"):
        if key in report:result[key]=ids(report[key],fragments_set if key.endswith("fragments") else raw)
    if "ambiguous_pairs" in report:
        rows=report["ambiguous_pairs"]
        require(type(rows) is list and len(rows)<=256 and all(type(r) is list and len(r)==2
            and all(type(i) is int and i in fragments_set for i in r) and r[0]!=r[1] for r in rows),"Malformed published ambiguous pairs")
        result["ambiguous_pairs"]=[list(row) for row in rows]
    if "recent_reference_links" in report:
        rows=report["recent_reference_links"]
        require(type(rows) is list and len(rows)<=4096,"Bounded recent reference links required")
        result["recent_reference_links"]=[]
        for row in rows:
            require(type(row["source_index"]) is int and type(row["target_index"]) is int
                and row["source_index"] in raw and row["target_index"] in raw
                and type(row["fragment_id"]) is int and row["fragment_id"] in fragments_set,
                "Malformed published recent-reference membership")
            result["recent_reference_links"].append({k:row[k] for k in ("source_index","target_index","fragment_id")})
    for key in ("pair_budget","fragment_limit"):
        if key in report:
            require(type(report[key]) is int and 0<report[key]<=(256 if key=="pair_budget" else 32),"Original published budget bound required")
            result[key]=report[key]
    for key in ("budget_limited","applied","failed"):
        if key in report:
            require(type(report[key]) is bool,"Published original outcome flag required")
            result[key]=report[key]
    return result


def observable_quality(native_report, candidate_report):
    require((native_report.get("kind"),native_report.get("mode")) in (
        ("gpu-icp-whole-finish-unobserved-control-v1","control"), (PREFIX+"native-v1","native"))
        and native_report.get("status") == "passed"
        and native_report.get("failure") is None and native_report.get("cleanup_passed") is True
        and native_report.get("cleanup_failures") == [] and native_report.get("binding") == native_report.get("binding_after"),
        "Independent closed unobserved original Finish control required")
    registration = native_report["registration"]
    require(registration.get("complete") is True and registration.get("restored") is True
        and registration.get("closed") is True and registration.get("failure") is None
        and registration.get("cleanup_failures") == [] and registration.get("original_build_calls") == 1,
        "Original control build/resource closure required")
    evidence = report_evidence(candidate_report)
    require(native_report["scope_binding"] == candidate_report["scope_binding"], "Different actual fresh Live checkpoint scope")
    a, b = native_report["profile"], candidate_report["profile"]
    for key in ("input_sha256", "selected_indices", "seed", "settings", "pipeline_options", "thread_policy", "accepted_indices"):
        require(a[key] == b[key], "Actual raw/settings/coverage differs: "+key)
    require(loop.is_digest(a["source_sha256"]) and a["source_sha256"] == b["source_sha256"] == candidate_report["binding"]["source_sha256"]
        == native_report["binding"]["source_sha256"]
        and native_report["binding"]["runtime"] == candidate_report["binding"]["runtime"], "Current core or actual numerical runtime differs")
    require(native_report["binding"]["method_source"]==candidate_report["binding"]["method_source"], "Original device evaluator source differs")
    for kind in ("artifacts_sha256","fixed_files"):
        left=native_report["binding"][kind];right=candidate_report["binding"][kind]
        if kind=="artifacts_sha256":left=normalized_artifacts(native_report["binding"]);right=normalized_artifacts(candidate_report["binding"])
        require(all(left[key]==right[key] for key in left.keys()&right.keys()), "Common current source/math/raw/native resource differs: "+kind)
    for profile in (a,b):
        require(profile.get("mesh_built") is True and profile.get("finish_requested") is True
            and profile.get("pose_seeds_used") is False and profile.get("input_changed_during_profile") is False
            and profile.get("source_changed_during_profile") is False, "Successful unchanged original raw Finish required")
    poses = []
    for profile in (a,b):
        require(0 < len(profile["accepted_indices"]) <= 4096
            and len(set(profile["accepted_indices"])) == len(profile["accepted_indices"])
            and all(type(i) is int and i>=0 for i in profile["accepted_indices"])
            and [r["index"] for r in profile["poses"]] == profile["accepted_indices"], "Complete original unrounded pose membership required")
        poses.append({r["index"]: values.matrix(r["camera_to_world"]) for r in profile["poses"]})
    inventory=candidate_report["registration"]["final_pose_inventory"]
    require(candidate_report.get("final_pose_inventory")==inventory,"Actual candidate Final inventory copies differ")
    require(values.pose_values(candidate_report,b)==poses[1],"Actual unrounded candidate Final inventory differs from exported values")
    translation, rotation = values.pose_deltas(*poses)
    native_memberships=published_fragment_memberships(a)
    candidate_memberships=published_fragment_memberships(b)
    require(native_memberships==candidate_memberships,"Published original bridge/witness/frontier memberships changed")
    graph_available = isinstance(registration.get("graphs"), list) and registration.get("registration_evidence_available") is True
    graph_equal = None
    if graph_available:
        require(isinstance(candidate_report["registration"].get("graphs"), list), "Candidate graph evidence absent")
        graph_equal = values.retained_graphs(native_report) == values.retained_graphs(candidate_report)
        require(graph_equal, "Actual retained graph endpoint/component membership changed")
    return {"checks": {"same_checkpoint_raw_settings_and_accepted_ids": True,
            "same_published_bridge_witness_frontier_memberships":True,
            "pose_bounds": translation <= .0005 and rotation <= .1},
        "metrics": {"pose_translation_max_m": translation, "pose_rotation_max_deg": rotation},
        "candidate_evidence": evidence, "retained_graph_comparison_available": graph_available,
        "published_fragment_membership_fields":native_memberships["available_fields"],
        "published_memberships_are_internal_optimizer_trace":False,
        "retained_graph_equal": graph_equal, "ordered_native_gate_history_comparison_available": False,
        "native_ransac_history_equivalence_claimed": False, "production_authority": False,
        "scope": "Actual new candidate inputs; independent native reference is not ground truth. No exact prior trajectory, ordered gate, general-domain or production permission."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_report", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    require(args.run_allocated and output.is_relative_to(ROOT/"benchmark-output") and not output.exists(), "Fresh private allocated quality output required")
    with socket.socket() as sock:
        sock.settimeout(.25)
        require(sock.connect_ex(("127.0.0.1",8000)) != 0, "Field server must be idle for physical surface comparison")
    paths = [p.resolve(strict=True) for p in (args.native_report,args.candidate_report)]
    require(paths[0] != paths[1] and output not in paths, "Independent controls and fresh output required")
    hashes = [sha(p) for p in paths]
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    result = {"kind": KIND,"status":"failed","failure":None,
        "native_report":{"path":str(paths[0]),"sha256":hashes[0]},
        "candidate_report":{"path":str(paths[1]),"sha256":hashes[1]}, "comparator_sha256":sha(__file__)}
    primary = None
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        owners = LoadedOwners()
        check_sources(reports[1]["binding"])
        result.update(observable_quality(*reports))
        require(all(result["checks"].values()), "Candidate Final pose observables failed declared bounds")
        geometries=[]
        for report in reports:
            record=report["profile"]["geometry"]
            path=Path(record["artifact"]).resolve(strict=True)
            require(path.is_relative_to(ROOT/"benchmark-output") and sha(path)==record["sha256"], "Actual closed geometry artifact changed")
            geometries.append({"path":str(path),"sha256":record["sha256"]})
        import numpy as np
        import open3d as o3d
        from shared.surface_metrics import surface_metrics
        def mesh(record):
            with np.load(record["path"],allow_pickle=False) as arrays:
                value=o3d.geometry.TriangleMesh()
                value.vertices=o3d.utility.Vector3dVector(arrays["points"])
                value.triangles=o3d.utility.Vector3iVector(arrays["faces"])
            return value
        surface=surface_metrics(mesh(geometries[1]),mesh(geometries[0]),threshold_m=.005,samples=30000)
        result["surface"]=surface
        result["checks"]["fixed_coordinate_surface_bounds"]=surface["surface_p95_m"]<=.0005 and surface["precision"]>=.999 and surface["completeness"]>=.999
        result["reference_geometry"],result["candidate_geometry"]=geometries
        require(all(sha(p)==h for p,h in zip(paths,hashes)) and all(sha(r["path"])==r["sha256"] for r in geometries), "Quality report/geometry bytes changed")
        check_sources(reports[1]["binding"]);owners.check()
        require(all(result["checks"].values()), "Physical candidate Final surface failed")
        result["status"]="passed"
    except BaseException as error:
        primary=error;result["failure"]={"type":type(error).__name__,"message":str(error)}
        raise
    finally:
        try:output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        except BaseException as write_error:
            if primary is not None:raise primary from write_error
            raise


if __name__ == "__main__":
    main()
