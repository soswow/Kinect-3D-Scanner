"""New complete-bridge authority with one Windows-owned read-only proof.

No v1 token. Exact original audit/terminal/gate bodies are source-derived;
only the proof owner and decoded-reference storage change. The initial proof
is hashed under GENERIC_READ/FILE_SHARE_READ, denying ordinary writes/deletes.
Its immutable cached references remain valid until end-hash and handle close.
"""
from __future__ import annotations
import ast
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import FunctionType, MappingProxyType
from scripts.research import microbatch_bridge_protocol as original
from scripts.research import gpu_icp_device_loop_protocol as values

ROOT=values.ROOT
KIND="gpu-icp-complete-bridge-owned-proof-audit-v2"
TIMING_KIND="gpu-icp-complete-bridge-owned-proof-timing-v2"
POLICY="windows-owned-readonly-proof-immutable-ordered-bridge-v2"
MAX_PROOF_BYTES=64*1024**2
NEW_FILES=("scripts/research/microbatch_bridge_owned_protocol.py","tests/test_microbatch_bridge_owned_protocol.py",
    "scripts/research/device_loop_owned_workspace.py","tests/test_device_loop_owned_workspace.py",
    "scripts/research/microbatch_bridge_owned_driver.py","tests/test_microbatch_bridge_owned_driver.py")
BASE_FILES={
    "scripts/research/device_loop_workspace.py":"45ee20f5d020fd34c1b9e45886cf57d86f6c4221b3453bccdcf727bb47ab5d4d",
    "scripts/research/device_loop_workspace_protocol.py":"12d5995b19b3d525fd3cf119628936f7bcee8db611b9a1011f0d2c18d6d2214d",
    "scripts/research/microbatch_bridge_protocol.py":"1d5a150e634bd21d92264452052123f490fdc1c32174fe282b107d57cc26a04b",
    "tests/test_device_loop_workspace.py":"116005ef59d7182c8c6886024cddeccab7d3d689e8527d8abc983bd9a7269b52",
    "tests/test_microbatch_bridge_protocol.py":"b2feff4314ec4f996231c2612a978b96a2fd18f1a1b002dd18a86466fb79f0d6",
    "scripts/research/microbatch_bridge_driver.py":"d048525610ac41d4b591ed5779d3d01ad06876f786d662fe067fcea8e6d2c2ec",
    "scripts/research/microbatch_bridge_scope.py":"a9303b1f85d8eaaa427e8a873b5c32b623ac16d971725c418c8050202014bf80",
    "tests/test_microbatch_bridge.py":"72e840905e36fcce7409a668695993e79a4aa0509dc80088b85fab04251aacb4"}
_REGISTRY={}


class _OwnerGuard:
    """Pinned loaded guard code/globals/default owners; no hot source reads."""
    def __init__(self):
        from scripts.research.device_loop_workspace import code_state
        self.records=[];self.modules=[];self.classes=[]
        for module in (sys.modules[__name__],original,values):
            path=Path(module.__file__).resolve();compiled=compile(path.read_text(encoding="utf-8"),str(path),"exec",dont_inherit=True)
            self.modules.append((module,path))
            for code in compiled.co_consts:
                if not hasattr(code,"co_code"):continue
                owner=getattr(module,code.co_name)
                if isinstance(owner,type):self.classes.append((module,code.co_name,owner))
                pairs=[(module,code.co_name,owner,code)] if isinstance(owner,FunctionType) else [
                    (owner,c.co_name,getattr(owner,c.co_name),c) for c in code.co_consts if hasattr(c,"co_code")]
                for parent,name,fn,expected in pairs:
                    values.require(isinstance(fn,FunctionType) and fn.__globals__ is module.__dict__
                        and code_state(fn.__code__)==code_state(expected),"Loaded proof guard differs from exact source owner")
                    self.records.append((parent,name,fn,fn.__code__,repr(fn.__defaults__),repr(fn.__kwdefaults__)))

    def check(self):
        for module,path in self.modules:
            values.require(sys.modules.get(module.__name__) is module and Path(module.__file__).resolve()==path,"Proof guard module owner changed")
        for module,name,owner in self.classes:
            values.require(getattr(module,name) is owner,"Loaded proof guard class owner changed")
        for owner,name,fn,code,defaults,kwdefaults in self.records:
            values.require(getattr(owner,name) is fn and fn.__code__ is code and repr(fn.__defaults__)==defaults
                and repr(fn.__kwdefaults__)==kwdefaults,"Loaded proof guard code/default owner changed")


def source_contract():
    for name,digest in BASE_FILES.items():
        values.require(values.sha(ROOT/name)==digest,"Frozen v1 mathematical/ownership evidence guard changed")
    return {"policy":POLICY,"artifacts":dict(BASE_FILES,**{name:values.sha(ROOT/name) for name in NEW_FILES}),
        "proof_share":"GENERIC_READ / FILE_SHARE_READ; denies ordinary write/delete handles",
        "max_proof_bytes":MAX_PROOF_BYTES,"decoded_references":"recursively immutable typed tuples",
        "hash_policy":"initial held-handle read and final held-handle read; no per-call proof I/O",
        "new_audit_required":True,"whole_finish_authority":False}


class WindowsProofLock:
    """One independently owned kernel32 handle; no path reopen for proof reads."""
    def __init__(self,path):
        values.require(os.name=="nt","This ownership method requires Windows share-mode semantics")
        from ctypes import wintypes as w
        self.api=ctypes.WinDLL("kernel32",use_last_error=True);self.path=str(path);self.closed=False
        self.api.CreateFileW.argtypes=(w.LPCWSTR,w.DWORD,w.DWORD,ctypes.c_void_p,w.DWORD,w.DWORD,w.HANDLE)
        self.api.CreateFileW.restype=w.HANDLE
        self.api.ReadFile.argtypes=(w.HANDLE,ctypes.c_void_p,w.DWORD,ctypes.POINTER(w.DWORD),ctypes.c_void_p)
        self.api.ReadFile.restype=w.BOOL
        self.api.SetFilePointerEx.argtypes=(w.HANDLE,ctypes.c_longlong,ctypes.POINTER(ctypes.c_longlong),w.DWORD)
        self.api.SetFilePointerEx.restype=w.BOOL
        self.api.GetFileSizeEx.argtypes=(w.HANDLE,ctypes.POINTER(ctypes.c_longlong));self.api.GetFileSizeEx.restype=w.BOOL
        self.api.CloseHandle.argtypes=(w.HANDLE,);self.api.CloseHandle.restype=w.BOOL
        self.handle=self.api.CreateFileW(self.path,0x80000000,1,None,3,0x80,None)
        if self.handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())

    def read(self):
        values.require(not self.closed,"Closed proof owner cannot supply evidence")
        size=ctypes.c_longlong()
        if not self.api.GetFileSizeEx(self.handle,ctypes.byref(size)):raise ctypes.WinError(ctypes.get_last_error())
        values.require(0<size.value<=MAX_PROOF_BYTES,"Bounded owned proof size required")
        if not self.api.SetFilePointerEx(self.handle,0,None,0):raise ctypes.WinError(ctypes.get_last_error())
        chunks=[];remaining=size.value
        while remaining:
            amount=min(1024*1024,remaining);buffer=ctypes.create_string_buffer(amount);read=ctypes.c_ulong()
            if not self.api.ReadFile(self.handle,buffer,amount,ctypes.byref(read),None):raise ctypes.WinError(ctypes.get_last_error())
            values.require(0<read.value<=amount,"Incomplete owned proof read")
            chunks.append(buffer.raw[:read.value]);remaining-=read.value
        return b"".join(chunks)

    def close(self):
        if self.closed:return
        if not self.api.CloseHandle(self.handle):raise ctypes.WinError(ctypes.get_last_error())
        self.closed=True


def _freeze(value):
    if value is None:return ("none",)
    if type(value) is bool:return ("bool",value)
    if type(value) is int:return ("int",value)
    if type(value) is float:
        values.require(math.isfinite(value),"Nonfinite immutable proof value")
        return ("float",value.hex())
    if type(value) is str:return ("str",value)
    if type(value) is list:return ("list",tuple(_freeze(v) for v in value))
    values.require(type(value) is dict and all(type(k) is str for k in value),"Unknown mutable proof value")
    return ("dict",tuple((k,_freeze(value[k])) for k in sorted(value)))


def _thaw(value):
    kind=value[0]
    if kind=="none":return None
    if kind=="float":return float.fromhex(value[1])
    if kind in ("bool","int","str"):return value[1]
    if kind=="list":return [_thaw(v) for v in value[1]]
    return {k:_thaw(v) for k,v in value[1]}


def _derived(name,edits):
    """Only declared ownership seams; inverse recovers the complete old body."""
    path=ROOT/"scripts/research/microbatch_bridge_protocol.py"
    values.require(values.sha(path)==BASE_FILES[str(path.relative_to(ROOT)).replace("\\","/")],"Wrong inherited evidence source")
    text=path.read_text(encoding="utf-8");tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    body=ast.get_source_segment(text,node);changed=body
    for old,new in edits:
        values.require(changed.count(old)==1,"Ambiguous new ownership seam")
        changed=changed.replace(old,new)
    recovered=changed
    for old,new in reversed(edits):recovered=recovered.replace(new,old)
    values.require(recovered==body,"New proof owner changed the original complete evidence guards")
    namespace=dict(original.__dict__)
    namespace.update(KIND=KIND,BRIDGE_FILES=tuple(original.BRIDGE_FILES)+NEW_FILES,
        _state=_state,_thaw=_thaw,registered=registered,validate_terminal=validate_terminal,
        validate_bridge_terminal=validate_bridge_terminal)
    exec(compile(changed,str(Path(__file__).resolve()),"exec",dont_inherit=True),namespace)
    return namespace[name]


def validate_report(report,binding,pair):
    values.require(binding.get("owned_proof_source")==source_contract(),"New proof ownership sources/configuration omitted")
    function=_derived("validate_report",(("from scripts.research import device_loop_workspace as workspace",
        "from scripts.research import device_loop_owned_workspace as workspace"),))
    return function(report,binding,pair)


@dataclass(frozen=True)
class OwnedBridgeTimingPermit:
    report_path:str
    report_sha256:str
    policy:str=POLICY


@dataclass(frozen=True)
class _References:
    workspace:tuple
    calls:tuple
    report:tuple
    indexed:object


def validate_bridge_audit(path,expected_binding,expected_pair_binding):
    path=Path(path).resolve(strict=True);guard=_OwnerGuard();lock=WindowsProofLock(path)
    try:
        raw=lock.read();digest=hashlib.sha256(raw).hexdigest()
        report=json.loads(raw.decode("utf-8"))
        workspace,calls=validate_report(report,expected_binding,expected_pair_binding)
        refs=_References(_freeze(workspace),_freeze(calls),_freeze({"gpu_proposals":[{
            key:p[key] for key in ("proposal_index","complete","result","gates")}
            for p in report["gpu_proposals"]],"gpu_pair_verdict":report["gpu_pair_verdict"]}),
            MappingProxyType({(r["proposal_index"],r["call_index"]):i for i,r in enumerate(calls)}))
        token=OwnedBridgeTimingPermit(str(path),digest)
        guard.check()
        _REGISTRY[id(token)]={"token":token,"fields":tuple(token.__dict__.values()),"lock":lock,"lock_owner":lock,
            "lock_fields":(lock.path,lock.handle,lock.api),"refs":refs,"refs_owner":refs,
            "ref_fields":(refs.workspace,refs.calls,refs.report,refs.indexed),"guard":guard,
            "closed":False,"failure":None,"cleanup_failures":[],"hash_initial":digest,"hash_final":None,"reads":1}
        return token
    except BaseException as primary:
        try:lock.close()
        except BaseException as cleanup:raise primary from cleanup
        raise


def _state(token):
    state=_REGISTRY.get(id(token))
    values.require(type(token) is OwnedBridgeTimingPermit and state is not None and state["token"] is token
        and tuple(token.__dict__.values())==state["fields"] and state["refs"] is state["refs_owner"]
        and all(getattr(state["refs"],key) is owner for key,owner in zip(("workspace","calls","report","indexed"),state["ref_fields"]))
        and state["lock"] is state["lock_owner"],
        "Own registered v2 authority required; no v1/constructed/modified token/cache")
    return state


def registered(token):
    state=_state(token)
    state["guard"].check()
    _check_lock(state)
    values.require(not state["closed"] and state["failure"] is None and not state["lock"].closed,"Closed/damaged proof owner cannot authorize work")
    return token


def _call(token,index):
    state=_state(token);calls=state["refs"].calls[1]
    values.require(type(index) is int and 0<=index<len(calls),"Exact ordered dynamic call index required")
    return _thaw(calls[index])


def constructor_authority(token,workspace,configuration):
    registered(token);state=_state(token)
    values.require(_freeze(workspace)==state["refs"].workspace,"New workspace setup source/configuration differs from its own audit")
    values.require(all(_call(token,index)["input"]["configuration"][key]==value
        for index in range(len(state["refs"].calls[1])) for key,value in configuration.items()),"Template graph/device/budget changed")
    return token


def validate_start(token,index,consumed,source):
    registered(token);expected=_call(token,index)
    values.require(consumed["configuration"]["audit_nearest"] is False and consumed["configuration"]["audit_misses"] is False,"Timing omits only observational shadows")
    values.require(_freeze(values.normalized_input(consumed))==_freeze(expected["input"])
        and _freeze(values.normalized_source(source))==_freeze(expected["source"]),"Actual arrays/seed/graph/source/order were not audited")
    return token


def validate_terminal(token,index,consumed,source,terminal):
    validate_start(token,index,consumed,source)
    values.require(_freeze(terminal)==_freeze(_call(token,index)["terminal"]),"Timed terminal bytes/metrics/query/update counts changed")
    return True


def validate_expected_bridge_call(token,proposal_index,call_index,payload):
    registered(token);index=_state(token)["refs"].indexed.get((proposal_index,call_index))
    values.require(index is not None and _freeze(payload)==_freeze(_call(token,index)["payload"]),"Complete bridge consumed different original arrays/colors/seed/order")
    return token


def validate_bridge_terminal(token,proposal_index,call_index,payload,terminal):
    validate_expected_bridge_call(token,proposal_index,call_index,payload)
    index=_state(token)["refs"].indexed[(proposal_index,call_index)]
    values.require(_freeze(terminal)==_freeze(_call(token,index)["terminal"]),"Complete bridge timed terminal differs from own immutable audit")
    return True


def validate_complete_bridge(token,actual):
    registered(token)
    fn=_derived("validate_complete_bridge",(("workspace_tokens.registered(token)","registered(token)"),
        ("expected=json.loads(token.calls_json)","expected=_thaw(_state(token)['refs'].calls)"),
        ("report=_BRIDGE_RECORDS.get(id(token))","report=_thaw(_state(token)['refs'].report)"),
        ("workspace_tokens.validate_terminal(","validate_terminal(")))
    return fn(token,actual)


def close_permit(token,primary=None):
    state=_REGISTRY.get(id(token))
    values.require(type(token) is OwnedBridgeTimingPermit and state is not None and state["token"] is token,"Own registered cleanup owner required")
    if state["closed"]:return
    cleanup=None
    try:
        _state(token);state["guard"].check();_check_lock(state)
        raw=state["lock_owner"].read();state["reads"]+=1;state["hash_final"]=hashlib.sha256(raw).hexdigest()
        values.require(state["hash_final"]==state["hash_initial"],"Owned proof changed before terminal closure")
    except BaseException as error:
        cleanup=error;state["failure"]=state["failure"] or repr(error);state["cleanup_failures"].append(repr(error))
    try:
        lock=state["lock_owner"]
        lock.path,lock.handle,lock.api=state["lock_fields"]
        lock.close();state["closed"]=True
    except BaseException as error:
        state["failure"]=state["failure"] or repr(error)
        state["cleanup_failures"].append(repr(error))
        if cleanup is None:cleanup=error
    if cleanup is not None:
        if primary is not None:raise primary from cleanup
        raise cleanup


def _check_lock(state):
    lock=state["lock_owner"];path,handle,api=state["lock_fields"]
    values.require(lock.path==path and lock.handle==handle and lock.api is api,"Owned proof path/handle/API owner changed")


def permit_report(token):
    if token is None:return {"required":False}
    state=_state(token)
    return {"required":True,"policy":POLICY,"closed":state["closed"],"failure":state["failure"],
        "hash_initial":state["hash_initial"],"hash_final":state["hash_final"],"share_read_only":True,
        "decoded_immutable":True,"held_handle_reads":state["reads"],"cleanup_failures":list(state["cleanup_failures"]),"old_token_consumed":False}
