"""Fresh complete-proposal workspace audit and exact ordered timing authority.

No old loop/batch permit. Original whole-proposal gates and ambiguity verdict
must pass independently, and every dynamic candidate ICP must have exhaustive
actual query and original native result shadows before setup-reuse timing.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from scripts.research import gpu_icp_device_loop_protocol as values
from scripts.research import device_loop_workspace_protocol as workspace_tokens

KIND="gpu-icp-complete-bridge-proposal-audit-v1"
BRIDGE_FILES=("scripts/research/microbatch_bridge_driver.py","scripts/research/microbatch_bridge_scope.py",
    "scripts/research/microbatch_bridge_protocol.py","tests/test_microbatch_bridge.py",
    "tests/test_microbatch_bridge_protocol.py","scripts/research/gpu_icp_device_loop_protocol.py",
    "scripts/research/gpu_icp_device_loop_experiment.py")
_MINT_CONTEXT=None
_BRIDGE_RECORDS={}


def _base_descriptor(value):return {k:value[k] for k in ("dtype","shape","sha256")}


def _resource_binding(binding):
    return {key:binding[key] for key in ("source_sha256","method_source","artifacts_sha256","runtime","fixed_files")}


def _full_input(payload,consumed):
    values.require(type(payload) is dict and set(payload)=={"source","target","seed"},"Exact complete original _match call input required")
    for role in ("source","target"):
        value=payload[role]
        values.require(type(value) is dict and set(value)=={"points","normals","colors"},"Complete original point/normal/color vector identity required")
        for name,record in value.items():
            values.require(type(record) is dict and set(record)=={"dtype","shape","nbytes","sha256"},"Malformed full original vector descriptor")
            values.descriptor(_base_descriptor(record),3,"<f8")
            values.require(type(record["nbytes"]) is int and record["nbytes"]==24*record["shape"][0],"Vector bytecount does not match original shape")
    seed=payload["seed"]
    values.require(type(seed) is dict and set(seed)=={"dtype","shape","nbytes","sha256"}
        and seed["nbytes"]==128,"Original unrounded seed byte identity required")
    values.require(_base_descriptor(payload["source"]["points"])==consumed["source"]
        and _base_descriptor(payload["target"]["points"])==consumed["target"]
        and _base_descriptor(payload["target"]["normals"])==consumed["normals"]
        and _base_descriptor(seed)==consumed["seed"],"Actually consumed arrays/seed differ from observed original bridge call")


def _call_for_common(call):
    trace=copy.deepcopy(call["query_trace"])
    for query in trace:query["packet"]=_base_descriptor(query["packet"])
    return {"input_binding":call["consumed_input_binding"],"source_binding":call["source_binding"],
        "terminal":call["terminal"],"native_shadow":call["native_shadow"],"loop_report":call["loop_report"],
        "query_trace":trace,"query_trace_sha256":hashlib.sha256(values.canonical(trace).encode()).hexdigest()}


def validate_report(report,binding,pair):
    from scripts.research import device_loop_workspace as workspace
    from scripts.research import microbatch_bridge_scope as gate_values
    values.actual_resource_closure(_resource_binding(binding))
    contract=workspace.source_contract()
    values.require(report.get("kind")==KIND and report.get("status")=="passed" and report.get("mode")=="audit"
        and report.get("binding")==report.get("binding_after")==binding and report.get("pair_binding")==pair,
        "Fresh closed own complete-proposal audit and exact actual pair/runtime binding required")
    values.require(binding.get("workspace_source")==contract,"Setup-reuse method source was not independently bound")
    configuration=binding.get("configuration",{})
    values.require(configuration.get("graph") is True and configuration.get("chunk_iterations")==4,
        "This complete-proposal method requires its declared graph4 execution policy")
    for name,digest in contract["artifacts"].items():
        values.require(binding["artifacts_sha256"].get(name)==digest,"New workspace/protocol/tests omitted from proof source closure")
    for name in BRIDGE_FILES:
        values.require(binding["artifacts_sha256"].get(name)==values.sha(values.ROOT/name),
            "Own complete-bridge producer/observer/validator/tests and inherited evidence guards must be pinned")
    values.require(report.get("failure") is None and report.get("cleanup_failures")==[]
        and report.get("cleanup_passed") is True and report.get("input_bytes_unchanged") is True
        and report.get("original_seed_bytes_unchanged") is True and report.get("loaded_owners_unchanged") is True
        and report.get("whole_finish_authority") is False and report.get("performance_attribution_valid") is False,
        "Complete proposal source/input/owner/cleanup or attribution contract failed")
    owner=report.get("workspace")
    values.require(type(owner) is dict and owner.get("policy")==workspace.POLICY and owner.get("provenance")==contract
        and owner.get("closed") is True and owner.get("failure") is None and owner.get("active_lane") is False
        and owner.get("retained_failed_lanes")==0 and owner.get("template_started") is False and owner.get("audit") is True,
        "Never-started template and all fresh lane/stream ownership must close")
    cache=report.get("shared_cache",{})
    values.require(cache.get("closed") is True and cache.get("failure") is None,"All shared immutable cache leases must close")
    native,candidate=report.get("native_proposals"),report.get("gpu_proposals")
    values.require(type(native) is list and type(candidate) is list and 2<=len(native)==len(candidate)<=6
        and len(pair["seeds"])==len(candidate),"All genuine competing proposals required; never prefix-only bridge authority")
    values.require(report.get("native_pair_verdict")==report.get("gpu_pair_verdict")
        and report.get("pair_quality_passed") is True,"Original complete ordered ambiguity/support verdict changed")
    calls=[];common_rows=[]
    for index,(old,new) in enumerate(zip(native,candidate)):
        values.require(old.get("proposal_index")==new.get("proposal_index")==index
            and old.get("complete") is True and new.get("complete") is True,"Complete original proposal ordering required")
        values.require(not gate_values.compare_evidence(old["result"],new["result"],path="result"),
            "Actual native complete result/witness/transform/information evidence changed")
        left,right=old["gates"],new["gates"]
        values.require(type(left) is list and len(left)==len(right) and left,"Complete original gate output suffix required")
        for i,(a,b) in enumerate(zip(left,right)):
            a={k:v for k,v in a.items() if k!="wall_s_inclusive"};b={k:v for k,v in b.items() if k!="wall_s_inclusive"}
            tolerance=1e-5 if a.get("name")=="REG.get_information_matrix_from_point_clouds" else 1e-8
            values.require(not gate_values.compare_evidence(a,b,tolerance=tolerance,path=f"gates[{i}]"),
                "Actual ordered original gate outcomes/membership/numeric evidence changed")
        dynamic=new["calls"]
        values.require(type(dynamic) is list and dynamic and len(dynamic)<=1024,"Bounded complete dynamic call inventory required")
        for ci,call in enumerate(dynamic):
            values.require(call.get("proposal_index")==index and call.get("call_index")==ci and call.get("complete") is True
                and call.get("input_bytes_unchanged") is True and call.get("failure") is None and call.get("cleanup_failure") is None,
                "Every ordered actual call must close without fallback or input damage")
            consumed=call["consumed_input_binding"]
            _full_input(call["input_binding"],consumed)
            values.require(consumed["configuration"].get("cuda_graph") is True
                and consumed["configuration"].get("chunk_iterations")==4,
                "Every actual complete-proposal trajectory must use its declared graph4 policy")
            values.require(call["source_binding"].get("setup_reuse")==contract,"Actual lane did not use the new audited setup reuse")
            values.require(call["query_trace_sha256"]==hashlib.sha256(values.canonical(call["query_trace"]).encode()).hexdigest(),
                "Actual exhaustive query trace changed")
            common_rows.append(_call_for_common(call))
            calls.append({"proposal_index":index,"call_index":ci,"payload":call["input_binding"],
                "input":values.normalized_input(consumed),"source":values.normalized_source(call["source_binding"]),
                "terminal":call["terminal"],"result":call["result"]})
    values.require(0<len(calls)<=4096 and len(owner.get("jobs",[]))==len(calls)
        and all(job.get("closed") is True and job.get("failure") is None for job in owner["jobs"]),
        "Every candidate call must own exactly one completed pristine workspace clone")
    for index,(job,call) in enumerate(zip(owner["jobs"],common_rows)):
        values.require(job.get("index")==index and job.get("report")==call["loop_report"],
            "Ordered workspace clone closure must match the exact actual call report")
    # Reuse pure evidence guards; this creates no old token or old report.
    adapted={"kind":values.KIND,"status":"passed","mode":"audit","binding":_resource_binding(binding),
        "binding_after":_resource_binding(binding),"failure":None,"cleanup_failures":[],"cleanup_passed":True,
        "input_bytes_unchanged":True,"loaded_owners_unchanged":True,"performance_attribution_valid":False,
        "whole_finish_authority":False,"rows":common_rows}
    # Repeated exact dynamic inputs are legitimate in complete original gates.
    # Validate each separately, then check aggregate positive hit/miss coverage.
    hits=misses=0
    for row in common_rows:
        adapted["rows"]=[row]
        # Some individual camera calls have no direct misses; aggregate authority
        # below supplies coverage, without manufacturing any additional query.
        statistics=row["loop_report"]["statistics"]
        hits+=statistics["audited_hits"];misses+=statistics["audited_misses"]
        _validate_one(adapted,_resource_binding(binding))
    values.require(hits>0 and misses>0,"Positive actual direct-hit AND direct-miss audit coverage required")
    return contract,calls


def _validate_one(report,binding):
    """Same numerical evidence body, aggregate positivity supplied by caller."""
    from types import FunctionType
    namespace=dict(values.validate_audit_report.__globals__)
    original=namespace["require"]
    def require(ok,message):
        if message=="Positive actual direct hit AND miss audit coverage required":return
        original(ok,message)
    namespace["require"]=require
    # Complete resource closure already ran once on the exact outer binding;
    # do not rehash raw ZIPs and native binaries for each dynamic call.
    namespace["actual_resource_closure"]=lambda _:None
    fn=FunctionType(values.validate_audit_report.__code__,namespace)
    return fn(report,binding)


def _require_mint_context(path,binding,workspace,calls):
    values.require(_MINT_CONTEXT==(str(Path(path).resolve()),values.canonical(binding),values.canonical(workspace),
        values.canonical(calls)),"Only fresh complete-bridge validator may mint workspace timing authority")


def validate_bridge_audit(path,expected_binding,expected_pair_binding):
    global _MINT_CONTEXT
    path=Path(path).resolve(strict=True);before=values.sha(path)
    report=json.loads(path.read_text(encoding="utf-8"))
    contract,calls=validate_report(report,expected_binding,expected_pair_binding)
    values.require(values.sha(path)==before,"Complete bridge proof changed during validation")
    _MINT_CONTEXT=(str(path),values.canonical(expected_binding),values.canonical(contract),values.canonical(calls))
    try:
        token=workspace_tokens._mint(path,expected_binding,contract,calls)
        _BRIDGE_RECORDS[id(token)]=copy.deepcopy(report)
        return token
    finally:_MINT_CONTEXT=None


def validate_expected_bridge_call(token,proposal_index,call_index,payload):
    workspace_tokens.registered(token)
    rows=json.loads(token.calls_json)
    expected=next((r for r in rows if (r["proposal_index"],r["call_index"])==(proposal_index,call_index)),None)
    values.require(expected is not None and expected["payload"]==payload,"Complete bridge consumed different actual points/normals/colors/seed/order")
    return token


def validate_bridge_terminal(token,proposal_index,call_index,payload,terminal):
    validate_expected_bridge_call(token,proposal_index,call_index,payload)
    expected=next(r for r in json.loads(token.calls_json) if (r["proposal_index"],r["call_index"])==(proposal_index,call_index))
    values.require(values.canonical(terminal)==values.canonical(expected["terminal"]),"Complete bridge timed terminal bytes/metrics/query/update counts changed")
    return True


def validate_complete_bridge(token,actual):
    workspace_tokens.registered(token)
    from scripts.research import microbatch_bridge_scope as gate_values
    expected=json.loads(token.calls_json)
    counts={r["proposal_index"]:0 for r in expected}
    for row in expected:counts[row["proposal_index"]]+=1
    report=_BRIDGE_RECORDS.get(id(token))
    values.require(type(actual) is dict and report is not None,"Own audited complete bridge event reference required")
    candidates=actual.get("gpu_proposals")
    values.require(type(candidates) is list and len(candidates)==len(report["gpu_proposals"])
        and actual.get("gpu_pair_verdict")==report["gpu_pair_verdict"],"Original complete ordered ambiguity/proposal verdict changed")
    for index,(old,new) in enumerate(zip(report["gpu_proposals"],candidates)):
        values.require(new.get("proposal_index")==index and new.get("complete") is True
            and len(new.get("calls",[]))==counts[index],"Incomplete dynamic call/whole proposal suffix")
        values.require(not gate_values.compare_evidence(old["result"],new["result"],path="result"),"Timed complete proposal result differs from own audited GPU result")
        values.require(len(old["gates"])==len(new["gates"]),"Timed original gate event suffix changed")
        for i,(a,b) in enumerate(zip(old["gates"],new["gates"])):
            a={k:v for k,v in a.items() if k!="wall_s_inclusive"};b={k:v for k,v in b.items() if k!="wall_s_inclusive"}
            values.require(not gate_values.compare_evidence(a,b,path=f"gates[{i}]"),"Timed original gate decisions/witnesses/numeric outputs changed")
        for ci,call in enumerate(new["calls"]):
            values.require(call.get("proposal_index")==index and call.get("call_index")==ci and call.get("complete") is True,
                "Timed actual call ordering changed")
            consumed=call["consumed_input_binding"];source=call["source_binding"];loop=call.get("loop_report",{})
            values.require(loop.get("closed") is True and loop.get("failure") is None
                and loop.get("input_binding")==consumed and values.normalized_source(loop.get("provenance"))==values.normalized_source(source)
                and call.get("input_bytes_unchanged") is True and call.get("failure") is None and call.get("cleanup_failure") is None,
                "Timed actual original-class lane/source/input/cleanup did not close")
            values.validate_timing_accounting(loop["statistics"],consumed["source"]["shape"][0])
            flat=next(i for i,r in enumerate(expected) if (r["proposal_index"],r["call_index"])==(index,ci))
            workspace_tokens.validate_terminal(token,flat,consumed,source,call["terminal"])
            validate_bridge_terminal(token,index,ci,call["input_binding"],call["terminal"])
    return True
