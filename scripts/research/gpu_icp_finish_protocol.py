"""Fresh whole-Finish device-loop proof with Windows-owned immutable receipts.

Numerical modules are never imported here. This registry cannot consume any
component, pair or historical Finish token. Every dynamic call and its terminal
reference belongs to one freshly closed actual-input audit and native quality
comparison. Audit/quality handles deny ordinary write/delete until cleanup.
"""
from __future__ import annotations
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import FunctionType
from typing import NamedTuple
from scripts.research import gpu_icp_device_loop_protocol as values
from scripts.research import microbatch_bridge_protocol as numerical
from scripts.research import microbatch_bridge_scope as gates
from scripts.research.microbatch_bridge_owned_protocol import _freeze, _thaw

ROOT = values.ROOT
KIND = "gpu-icp-whole-finish-audit-v1"
TIMING_KIND = "gpu-icp-whole-finish-timing-v1"
NATIVE_KIND = "gpu-icp-whole-finish-native-v1"
QUALITY_KIND = "gpu-icp-whole-finish-quality-v1"
POLICY = "windows-owned-global-device-loop-whole-finish-proof-v1"
MAX_PROOF_BYTES = 256 * 1024**2
OWN_FILES = ("scripts/research/device_loop_finish_workspace.py",
    "scripts/research/gpu_icp_finish_protocol.py", "tests/test_device_loop_finish_workspace.py",
    "tests/test_gpu_icp_finish_protocol.py")
FINISH_FILES = ("scripts/research/profile_gpu_icp_finish.py",
    "scripts/research/gpu_icp_finish_scope.py", "scripts/research/compare_gpu_icp_finishes.py",
    "tests/test_gpu_icp_finish_scope.py", "tests/test_gpu_icp_finish_quality.py",
    "tests/test_gpu_icp_finish_profile.py", "scripts/research/compare_resident_finishes.py",
    "shared/surface_metrics.py", "scripts/research/private_live_checkpoint.py")
QUALITY_CHECKS = ("same_checkpoint", "same_accepted_indices", "same_retained_graph",
    "same_ordered_gate_memberships", "same_mesh_success", "pose_bounds", "surface_bounds",
    "actual_gpu_calls_shadowed", "actual_gpu_query_rows_shadowed")
_REGISTRY = {}


class LoadedGuard:
    """Cold source/code checks followed by cheap immutable loaded-owner checks."""
    def __init__(self, modules):
        from scripts.research.device_loop_workspace import code_state
        self.records, self.modules, self.classes = [], [], []
        for module in modules:
            path = Path(module.__file__).resolve()
            compiled = compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
            self.modules.append((module, path))
            for code in compiled.co_consts:
                if not hasattr(code, "co_code"):
                    continue
                owner = getattr(module, code.co_name)
                if isinstance(owner, type):
                    self.classes.append((module, code.co_name, owner))
                pairs = [(module, code.co_name, owner, code)] if isinstance(owner, FunctionType) else [
                    (owner, c.co_name, getattr(owner, c.co_name), c)
                    for c in code.co_consts if hasattr(c, "co_code")]
                for parent, name, fn, expected in pairs:
                    values.require(isinstance(fn, FunctionType) and fn.__globals__ is module.__dict__
                        and code_state(fn.__code__) == code_state(expected), "Loaded Finish guard differs from source")
                    self.records.append((parent, name, fn, fn.__code__, repr(fn.__defaults__), repr(fn.__kwdefaults__)))
        self.aliases = [(module,name,getattr(module,name)) for module in modules for name in
            ("values", "numerical", "gates", "_freeze", "_thaw", "semantic_agreement", "original")
            if hasattr(module,name)]

    def check(self):
        for module, path in self.modules:
            values.require(sys.modules.get(module.__name__) is module and Path(module.__file__).resolve() == path,
                           "Finish proof module owner changed")
        for module, name, owner in self.classes:
            values.require(getattr(module, name) is owner, "Finish proof class owner changed")
        for owner, name, fn, code, defaults, kwdefaults in self.records:
            values.require(getattr(owner, name) is fn and fn.__code__ is code
                and repr(fn.__defaults__) == defaults and repr(fn.__kwdefaults__) == kwdefaults,
                "Finish proof code/default owner changed")
        for owner, name, value in self.aliases:
            values.require(getattr(owner, name) is value, "Finish proof imported owner changed")


def source_contract():
    from scripts.research import device_loop_finish_workspace as workspace
    pins = dict(workspace.OLD_PINS)
    values.require(all(values.sha(ROOT/name) == digest for name, digest in pins.items()), "Held fourteen sources changed")
    pins.update({name: values.sha(ROOT/name) for name in OWN_FILES + FINISH_FILES})
    return {"policy": POLICY, "artifacts": pins, "maximum_proof_bytes": MAX_PROOF_BYTES,
        "proof_share": "GENERIC_READ / FILE_SHARE_READ; denies ordinary writes/deletes",
        "proof_hashes": "initial and final read under each retained handle; no hot proof reads",
        "references": "recursively immutable typed tuples; global call order",
        "fresh_whole_finish_audit_and_native_quality_required": True}


class WindowsProofLock:
    """One owned handle; both hashes read through it, never reopen the path."""
    def __init__(self, path):
        values.require(os.name == "nt", "Whole-Finish proof ownership requires Windows")
        from ctypes import wintypes as w
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.path, self.closed = str(path), False
        self.api.CreateFileW.argtypes = (w.LPCWSTR,w.DWORD,w.DWORD,ctypes.c_void_p,w.DWORD,w.DWORD,w.HANDLE)
        self.api.CreateFileW.restype = w.HANDLE
        self.api.ReadFile.argtypes = (w.HANDLE,ctypes.c_void_p,w.DWORD,ctypes.POINTER(w.DWORD),ctypes.c_void_p)
        self.api.ReadFile.restype = w.BOOL
        self.api.SetFilePointerEx.argtypes = (w.HANDLE,ctypes.c_longlong,ctypes.POINTER(ctypes.c_longlong),w.DWORD)
        self.api.SetFilePointerEx.restype = w.BOOL
        self.api.GetFileSizeEx.argtypes = (w.HANDLE,ctypes.POINTER(ctypes.c_longlong))
        self.api.GetFileSizeEx.restype = w.BOOL
        self.api.CloseHandle.argtypes, self.api.CloseHandle.restype = (w.HANDLE,), w.BOOL
        self.handle = self.api.CreateFileW(self.path,0x80000000,1,None,3,0x80,None)
        if self.handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())

    def read(self):
        values.require(not self.closed, "Closed Finish proof owner")
        size = ctypes.c_longlong()
        if not self.api.GetFileSizeEx(self.handle, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        values.require(0 < size.value <= MAX_PROOF_BYTES, "Bounded complete Finish proof required")
        if not self.api.SetFilePointerEx(self.handle,0,None,0):
            raise ctypes.WinError(ctypes.get_last_error())
        chunks, remaining = [], size.value
        while remaining:
            amount = min(1024*1024,remaining)
            buffer, read = ctypes.create_string_buffer(amount), ctypes.c_ulong()
            if not self.api.ReadFile(self.handle,buffer,amount,ctypes.byref(read),None):
                raise ctypes.WinError(ctypes.get_last_error())
            values.require(0 < read.value <= amount, "Incomplete Finish proof read")
            chunks.append(buffer.raw[:read.value]); remaining -= read.value
        return b"".join(chunks)

    def close(self):
        if self.closed:
            return
        if not self.api.CloseHandle(self.handle):
            raise ctypes.WinError(ctypes.get_last_error())
        self.closed = True


def _resources(binding):
    return {key: binding[key] for key in ("source_sha256", "method_source", "artifacts_sha256", "runtime", "fixed_files")}


def _scope(report):
    return report.get("registration", report)


def _diagnostic_free(value):
    """Only explicit measured durations are excluded; budget/outcome values stay."""
    if type(value) is list:
        return [_diagnostic_free(v) for v in value]
    if type(value) is dict:
        result = {key: _diagnostic_free(v) for key, v in value.items()
                if key not in ("wall_s", "wall_s_inclusive", "registration_wall_s", "original_cpu_shadow_wall_s")}
        if result.get("function") == "bundle_adjustment.propose_bundle_poses":
            output=result.get("result")
            if type(output) is list and len(output)==2 and type(output[1]) is dict:
                output[1].pop("elapsed_ms",None)
        return result
    return value


def _closed(report, mode, binding, scope):
    kind = KIND if mode == "audit" else NATIVE_KIND
    values.require(report.get("kind") == kind and report.get("mode") == mode and report.get("status") == "passed"
        and report.get("binding") == report.get("binding_after") == binding
        and report.get("scope_binding") == scope and report.get("failure") is None
        and report.get("cleanup_failures") == [] and report.get("cleanup_passed") is True,
        "Fresh current whole-Finish envelope did not close")
    record = _scope(report)
    values.require(record.get("complete") is True and record.get("restored") is True
        and record.get("closed") is True and record.get("failure") is None and record.get("cleanup_failures") == []
        and record.get("successful_builds") == 1 and record.get("final_pose_inventory") == report.get("final_pose_inventory"),
        "Original Finish build/dispatch/observers did not close healthy and restored")
    return record


def _workspace(owner, contract, audit):
    values.require(type(owner) is dict and owner.get("provenance") == contract
        and owner.get("policy") == contract["policy"] and owner.get("closed") is True
        and owner.get("failure") is None and owner.get("active_lane") is False
        and owner.get("retained_failed_lanes") == 0 and owner.get("template_started") is False
        and owner.get("audit") is audit, "Whole-Finish workspace did not prove all stream completion")


def _events(record, calls):
    ordered=record.get("events")
    values.require(type(ordered) is list and ordered, "Complete original Finish event inventory required")
    domains={"match":len(calls),"gate":len(record["gates"]),"graph_optimization":len(record["graphs"])}
    found={key:[] for key in domains}
    for row in ordered:
        values.require(type(row) is dict and set(row)=={"event","index"} and row["event"] in domains
            and type(row["index"]) is int and 0<=row["index"]<domains[row["event"]],
            "Unknown or unbounded original Finish event")
        found[row["event"]].append(row["index"])
    values.require(all(sorted(indices)==list(range(domains[key])) for key,indices in found.items()),
                   "Original event/call suffix duplicated or omitted")


def validate_report(report, binding, scope):
    from scripts.research import device_loop_finish_workspace as workspace
    from scripts.research import gpu_icp_finish_scope as injection
    values.actual_resource_closure(_resources(binding))
    contract = workspace.source_contract()
    scope_contract = injection.source_contract()
    values.require(binding.get("workspace_source") == contract and binding.get("finish_proof_source") == source_contract()
        and binding.get("finish_scope_source") == scope_contract,
        "New whole-Finish workspace/proof sources were not bound")
    pins = dict(contract["artifacts"])
    pins.update(source_contract()["artifacts"]); pins.update(scope_contract["artifacts"])
    for name, digest in pins.items():
        values.require(binding["artifacts_sha256"].get(name) == digest, "Whole-Finish source dependency omitted")
    cfg = binding.get("configuration", {})
    values.require(cfg.get("graph") is True and cfg.get("chunk_iterations") == 4, "Declared graph4 Finish policy required")
    record = _closed(report, "audit", binding, scope)
    owner = report["workspace"]
    caches = report.get("cache_receipts")
    calls = report.get("calls")
    values.require(type(calls) is list and record.get("calls") == calls, "Whole-Finish global call inventory differs")
    values.require(record.get("workspace") == owner and record.get("cache_receipts") == caches,
                   "Whole-Finish top and registration ownership receipts differ")
    _events(record,calls)
    if not calls:
        values.require(owner is None and caches==[], "Zero-GPU original path cannot claim a fabricated workspace/cache")
        return contract, []
    _workspace(owner, contract, True)
    values.require(type(caches) is list and caches and all(c.get("closed") is True and c.get("failure") is None
        and c.get("active_streams") == 0 for c in caches), "Every directed-pair cache/lease must close")
    values.require(type(calls) is list and 0 < len(calls) <= 4096 and record.get("calls") == calls
        and len(owner.get("jobs", [])) == len(calls), "Complete globally ordered dynamic call inventory required")
    refs, hits, misses = [], 0, 0
    for index, (call, job) in enumerate(zip(calls, owner["jobs"])):
        values.require(call.get("call_index") == index and call.get("complete") is True
            and call.get("input_bytes_unchanged") is True and call.get("failure") is None
            and call.get("cleanup_failures") == [], "An actual whole-Finish call failed closure")
        consumed, source = call["consumed_input_binding"], call["source_binding"]
        payload = call["payload"]
        values.require(type(payload) is dict and set(payload) == {"pair_index", "pair", "proposal_index", "caller", "gate_context", "input_binding"}
            and type(payload["pair_index"]) is int and payload["pair_index"] >= 0
            and type(payload["proposal_index"]) is int and payload["proposal_index"] >= 0
            and type(payload["pair"]) is list and len(payload["pair"]) == 2
            and all(type(i) is int and i >= 0 for i in payload["pair"])
            and type(payload["caller"]) is str and type(payload["gate_context"]) is list
            and payload["input_binding"] == call["input_binding"], "Actual original pair/proposal/caller/context changed")
        numerical._full_input(call["input_binding"], consumed)
        values.require(source.get("setup_reuse") == contract and consumed["configuration"].get("cuda_graph") is True
            and consumed["configuration"].get("chunk_iterations") == 4, "Actual lane graph/workspace changed")
        values.require(call["query_trace_sha256"] == hashlib.sha256(values.canonical(call["query_trace"]).encode()).hexdigest(),
            "Exhaustive actual query trace changed")
        common = numerical._call_for_common(call)
        adapted = {"kind": values.KIND, "status": "passed", "mode": "audit", "binding": _resources(binding),
            "binding_after": _resources(binding), "failure": None, "cleanup_failures": [], "cleanup_passed": True,
            "input_bytes_unchanged": True, "loaded_owners_unchanged": True, "performance_attribution_valid": False,
            "whole_finish_authority": False, "rows": [common]}
        numerical._validate_one(adapted, _resources(binding))
        stats = call["loop_report"]["statistics"]
        hits += stats["audited_hits"]; misses += stats["audited_misses"]
        values.require(job.get("index") == index and job.get("closed") is True and job.get("failure") is None
            and job.get("report") == call["loop_report"], "Workspace job differs from actual closed call")
        refs.append({"payload": payload, "input": values.normalized_input(consumed),
            "source": values.normalized_source(source), "terminal": call["terminal"], "result": call["result"]})
    values.require(hits > 0 and misses > 0, "Positive actual original CPU hit AND miss shadows required")
    return contract, refs


def validate_quality(quality, native, audit, binding, scope, audit_hash, native_hash):
    from scripts.research import compare_gpu_icp_finishes as comparator
    values.require(quality.get("kind") == QUALITY_KIND and quality.get("status") == "passed"
        and quality.get("binding") == binding and quality.get("scope_binding") == scope
        and quality.get("audit_report", {}).get("sha256") == audit_hash
        and quality.get("native_report", {}).get("sha256") == native_hash
        and quality.get("comparator_source_sha256") == values.sha(Path(comparator.__file__)),
        "Independent new native/Final quality record is missing or stale")
    _closed(native, "native", binding, scope)
    computed = comparator.quality_record(native, audit)
    surface = quality.get("surface", {})
    for role, report in (("candidate",audit),("reference",native)):
        record = quality.get(role+"_geometry", {})
        values.require(type(record) is dict and set(record)=={"path","sha256","sha256_after"}
            and record["sha256"] == record["sha256_after"] == values.sha(record["path"])
            and surface.get(role+"_geometry_sha256") == record["sha256"]
            and report.get("geometry") == {"path":record["path"],"sha256":record["sha256"]},
            "Recorded physical surface certificate differs from its current geometry")
    metrics = dict(computed["metrics"], **{key:surface.get(key) for key in
        ("surface_p95_m","precision","completeness","threshold_m","samples")})
    for name, limit in (("pose_translation_max_m", .0005), ("pose_rotation_max_deg", .1), ("surface_p95_m", .0005)):
        value = metrics.get(name)
        values.require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= limit,
                       "Fixed physical Final bound failed")
    values.require(metrics.get("threshold_m") == .005 and metrics.get("samples") == 30000
        and all(type(metrics.get(k)) in (int, float) and math.isfinite(metrics[k]) and .999 <= metrics[k] <= 1
                for k in ("precision", "completeness")), "Fixed-coordinate 30k surface scope failed")
    checks = dict(computed["checks"], surface_bounds=True)
    values.require(quality.get("checks") == checks and quality.get("metrics") == metrics
        and set(checks) == set(QUALITY_CHECKS) and all(value is True for value in checks.values()),
        "Original final graph/witness/coverage quality failed")
    return {"checks":checks,"metrics":metrics,
        "surface_scope":"Recorded current-geometry certificate; native 30k surface computation is separately allocated, not redispatched during proof validation"}


@dataclass(frozen=True)
class FinishTimingPermit:
    report_path: str
    report_sha256: str
    quality_path: str
    quality_sha256: str
    policy: str = POLICY


class References(NamedTuple):
    workspace: tuple
    calls: tuple
    scope: tuple
    binding: tuple
    events: tuple
    quality: tuple
    ordered: tuple


def validate_finish_audit(path, expected_binding, expected_scope, *, quality_path):
    from scripts.research import device_loop_finish_workspace as workspace
    from scripts.research import compare_gpu_icp_finishes as comparator
    from scripts.research import microbatch_bridge_owned_protocol as inherited
    from scripts.research import gpu_icp_finish_scope as injection
    semantic_owner=sys.modules.get(comparator.semantic_agreement.__module__)
    values.require(semantic_owner is not None and getattr(semantic_owner,"semantic_agreement",None) is comparator.semantic_agreement,
                   "Original semantic quality helper owner changed")
    guard = LoadedGuard((sys.modules[__name__], numerical, values, gates, inherited, workspace, injection, comparator,semantic_owner))
    locks, primary = [], None
    try:
        path, quality_path = Path(path).resolve(strict=True), Path(quality_path).resolve(strict=True)
        for p in (path, quality_path):
            locks.append(WindowsProofLock(p))
        raw, quality_raw = locks[0].read(), locks[1].read()
        digest, quality_digest = hashlib.sha256(raw).hexdigest(), hashlib.sha256(quality_raw).hexdigest()
        report, quality = json.loads(raw.decode("utf-8")), json.loads(quality_raw.decode("utf-8"))
        contract, calls = validate_report(report, expected_binding, expected_scope)
        native_path = Path(quality["native_report"]["path"]).resolve(strict=True)
        native_bytes = native_path.read_bytes(); native_hash = hashlib.sha256(native_bytes).hexdigest()
        validate_quality(quality, json.loads(native_bytes.decode("utf-8")), report, expected_binding, expected_scope, digest, native_hash)
        record = _scope(report)
        refs = References(_freeze(contract), _freeze(calls), _freeze(expected_scope), _freeze(expected_binding),
            _freeze({key: _diagnostic_free(record[key]) for key in ("pairs", "gates", "graphs", "final_pose_inventory")}),
            _freeze(quality),_freeze(record["events"]))
        token = FinishTimingPermit(str(path), digest, str(quality_path), quality_digest)
        guard.check()
        owned_locks = tuple(locks)
        _REGISTRY[id(token)] = {"token": token, "fields": tuple(token.__dict__.items()), "refs": refs, "refs_owner":refs,
            "ref_fields": tuple(refs), "locks": owned_locks, "locks_owner":owned_locks,
            "lock_fields": tuple((l.path,l.handle,l.api) for l in locks),
            "guard": guard, "closed": False, "failure": None, "cleanup_failures": [],
            "hash_initial": (digest, quality_digest), "hash_final": None, "reads": [1,1],
            "next": 0, "active": None, "started": False, "completed": 0,"event_next":0,
            "native_file": (str(native_path), native_hash),
            "geometry_files":tuple((quality[role+"_geometry"]["path"],quality[role+"_geometry"]["sha256"])
                                   for role in ("candidate","reference"))}
        return token
    except BaseException as error:
        primary = error
        for lock in reversed(locks):
            try:
                lock.close()
            except BaseException as cleanup:
                if hasattr(primary, "add_note"):
                    primary.add_note("Finish proof entry cleanup: " + repr(cleanup))
                primary.__cause__ = cleanup
        raise primary


def _state(token):
    state = _REGISTRY.get(id(token))
    values.require(type(token) is FinishTimingPermit and state is not None and state["token"] is token
        and tuple(token.__dict__.items()) == state["fields"] and type(state["refs"]) is References
        and state["refs"] is state["refs_owner"] and state["locks"] is state["locks_owner"]
        and all(value is saved for value, saved in zip(state["refs"],state["ref_fields"])),
        "Only this registered immutable full-Finish token can authorize work")
    return state


def registered(token):
    state = _state(token); state["guard"].check()
    values.require(not state["closed"] and state["failure"] is None, "Closed/damaged full-Finish proof owner")
    for lock, (path, handle, api) in zip(state["locks"], state["lock_fields"]):
        values.require(lock.path == path and lock.handle == handle and lock.api is api and lock.closed is False,
                       "Full-Finish proof handle/path/API owner changed")
    return token


def _call(token, index):
    calls = _state(token)["refs"].calls[1]
    values.require(type(index) is int and 0 <= index < len(calls), "Exact global Finish call index required")
    return _thaw(calls[index])


def _frozen_field(value, key):
    values.require(value[0]=="dict", "Immutable evidence mapping required")
    for name, field in value[1]:
        if name==key:return field
    raise ValueError("Missing immutable evidence field: "+key)


def constructor_authority(token, workspace, configuration):
    registered(token)
    values.require(len(_state(token)["refs"].calls[1])>0, "Zero-GPU negative scope cannot construct a timing GPU workspace")
    values.require(_freeze(workspace) == _state(token)["refs"].workspace, "Finish workspace source/configuration changed")
    for index in range(len(_state(token)["refs"].calls[1])):
        cfg = _call(token,index)["input"]["configuration"]
        values.require(all(cfg.get(k) == v for k,v in configuration.items()), "Finish template device/graph/budgets changed")
    return token


def next_workspace_job(token, payload):
    registered(token); state = _state(token); index = state["next"]
    values.require(state["active"] is None and _freeze(payload) == _freeze(_call(token,index)["payload"]),
                   "Actual global Finish call/cloud/seed/context/order differs before GPU")
    state["active"], state["started"] = index, False
    return index


def current_workspace_job(token):
    registered(token)
    return _state(token)["active"]


def validate_start(token, index, consumed, source):
    registered(token); state = _state(token); expected = _call(token,index)
    values.require(state["active"] == index and state["started"] is False,
                   "Only the exact next scheduled Finish lane may start once")
    values.require(consumed["configuration"]["audit_nearest"] is False
        and consumed["configuration"]["audit_misses"] is False
        and _freeze(values.normalized_input(consumed)) == _freeze(expected["input"])
        and _freeze(values.normalized_source(source)) == _freeze(expected["source"]),
        "Actual current arrays/seed/source/graph was not fully audited")
    state["started"] = True
    return token


def workspace_authorizer(token, index):
    registered(token)
    return lambda consumed, source: validate_start(token,index,consumed,source)


def validate_finish_terminal(token, index, payload, terminal, loop_report):
    registered(token); state = _state(token); expected = _call(token,index)
    consumed, source = loop_report.get("input_binding"), loop_report.get("provenance")
    values.require(state["active"] == index and state["started"] is True
        and _freeze(payload) == _freeze(expected["payload"]) and _freeze(terminal) == _freeze(expected["terminal"]),
        "Timed Finish terminal bytes/metrics/query/update counts differ from its own audit")
    values.require(loop_report.get("closed") is True and loop_report.get("failure") is None
        and _freeze(values.normalized_input(consumed)) == _freeze(expected["input"])
        and _freeze(values.normalized_source(source)) == _freeze(expected["source"]), "Timed Finish lane did not close")
    values.validate_timing_accounting(loop_report["statistics"],consumed["source"]["shape"][0])
    state["active"], state["started"] = None, False
    state["next"] += 1; state["completed"] += 1
    return True


def validate_finish_event(token, row):
    registered(token);state=_state(token)
    ordered=state["refs"].ordered[1];position=state["event_next"]
    values.require(type(row) is dict and row.get("complete") is True and position<len(ordered),
                   "Unexpected or incomplete original Finish event")
    kind=row.get("event");key={"match":"call_index","gate":"gate_index","graph_optimization":"graph_index"}.get(kind)
    values.require(key is not None and _thaw(ordered[position])=={"event":kind,"index":row.get(key)},
                   "Original complete Finish event order/context changed")
    index=row[key]
    if kind=="match":
        expected=_call(token,index)
        values.require(state["completed"]>index and _freeze(row["payload"])==_freeze(expected["payload"])
            and _freeze(row["terminal"])==_freeze(expected["terminal"])
            and _freeze(row["result"])==_freeze(expected["result"]),"Timed match event changed after terminal closure")
    else:
        category="gates" if kind=="gate" else "graphs"
        records=_frozen_field(state["refs"].events,category)[1]
        values.require(type(index) is int and 0<=index<len(records)
            and _freeze(_diagnostic_free(row))==records[index],
            "Original gate/graph result, options or discrete memberships changed")
    state["event_next"]+=1
    return True


def validate_complete_finish(token, actual):
    registered(token); state = _state(token); record = _scope(actual)
    calls = record.get("calls")
    values.require(state["active"] is None and state["completed"] == len(state["refs"].calls[1])
        and state["event_next"] == len(state["refs"].ordered[1]) and _freeze(record.get("events"))==state["refs"].ordered
        and type(calls) is list and len(calls) == state["completed"]
        and record.get("complete") is True and record.get("restored") is True and record.get("closed") is True
        and record.get("failure") is None
        and record.get("cleanup_failures") == [] and record.get("successful_builds") == 1,
        "Whole-Finish actual event/call suffix or healthy build is incomplete")
    owner, caches = record.get("workspace"), record.get("cache_receipts")
    if calls:
        _workspace(owner, _thaw(state["refs"].workspace), False)
        values.require(type(caches) is list and caches and all(c.get("closed") is True and c.get("failure") is None
            and c.get("active_streams") == 0 for c in caches), "Timed pair/cache completion did not close")
        values.require(len(owner.get("jobs", [])) == len(calls), "Timed workspace job suffix differs")
    else:
        values.require(owner is None and caches == [], "Zero-GPU timing cannot claim GPU ownership")
    for index, call in enumerate(calls):
        expected = _call(token,index)
        values.require(call.get("call_index") == index and call.get("complete") is True
            and call.get("input_bytes_unchanged") is True and call.get("cleanup_failures") == []
            and call.get("failure") is None
            and _freeze(call["payload"]) == _freeze(expected["payload"])
            and _freeze(call["terminal"]) == _freeze(expected["terminal"])
            and _freeze(call["result"]) == _freeze(expected["result"]), "Timed actual call receipt changed")
        loop=call["loop_report"];consumed=loop.get("input_binding")
        values.require(loop.get("closed") is True and loop.get("failure") is None
            and _freeze(values.normalized_input(consumed))==_freeze(expected["input"])
            and _freeze(values.normalized_source(loop.get("provenance")))==_freeze(expected["source"]),
            "Timed Finish lane receipt changed after completion")
        values.validate_timing_accounting(loop["statistics"],consumed["source"]["shape"][0])
        job = owner["jobs"][index]
        values.require(job.get("index") == index and job.get("closed") is True and job.get("failure") is None
            and job.get("report") == loop, "Timed workspace job differs from actual closed lane")
    actual_events = {key: _diagnostic_free(record[key]) for key in ("pairs", "gates", "graphs", "final_pose_inventory")}
    values.require(_freeze(actual_events) == state["refs"].events,
        "Original complete gate/graph/witness/verdict/final-pose event trajectory changed")
    return True


def close_permit(token, primary=None):
    state = _REGISTRY.get(id(token))
    values.require(type(token) is FinishTimingPermit and state is not None and state["token"] is token,
                   "Own full-Finish cleanup owner required")
    if state["closed"]:
        return
    cleanup = None
    try:
        registered(token)
        hashes = []
        for index, lock in enumerate(state["locks"]):
            hashes.append(hashlib.sha256(lock.read()).hexdigest()); state["reads"][index] += 1
        state["hash_final"] = tuple(hashes)
        values.require(state["hash_final"] == state["hash_initial"], "Owned whole-Finish proof changed before cleanup")
        values.require(values.sha(state["native_file"][0]) == state["native_file"][1], "Independent native reference changed")
        values.require(all(values.sha(path)==digest for path,digest in state["geometry_files"]),
                       "Recorded surface geometry changed during timing")
    except BaseException as error:
        cleanup = error; state["failure"] = state["failure"] or repr(error)
        state["cleanup_failures"].append(repr(error))
    closed = True
    for lock, (path,handle,api) in zip(state["locks_owner"],state["lock_fields"]):
        try:
            lock.path,lock.handle,lock.api = path,handle,api
            lock.close()
        except BaseException as error:
            closed = False; state["failure"] = state["failure"] or repr(error)
            state["cleanup_failures"].append(repr(error))
            if cleanup is None:
                cleanup = error
    state["closed"] = closed
    if cleanup is not None:
        if primary is not None:
            raise primary from cleanup
        raise cleanup


def permit_report(token):
    if token is None:
        return {"required": False}
    state = _state(token)
    return {"required": True, "policy": POLICY, "closed": state["closed"], "failure": state["failure"],
        "hash_initial": list(state["hash_initial"]), "hash_final": None if state["hash_final"] is None else list(state["hash_final"]),
        "held_handle_reads": list(state["reads"]), "share_read_only": True, "decoded_immutable": True,
        "completed_calls": state["completed"], "cleanup_failures": list(state["cleanup_failures"]), "old_token_consumed": False}


def load_finish_permit(audit_path, quality_path, *, binding, scope_binding):
    return validate_finish_audit(audit_path,binding,scope_binding,quality_path=quality_path)
