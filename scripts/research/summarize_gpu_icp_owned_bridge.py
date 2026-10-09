"""Scalar publication of the fresh owned-proof bridge receipts, without authority.

Original numerical/count/order/terminal/gate checks run in private unchanged
function clones. Only source metadata, receipt versions and the owned-lock
envelope are added. No raw ZIP/library rescan, registry or native dispatch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research import summarize_gpu_icp_complete_bridge as original
from scripts.research import microbatch_bridge_protocol as guards
from scripts.research import microbatch_bridge_owned_protocol as owned
from scripts.research import microbatch_bridge_owned_driver as driver
from scripts.research import device_loop_owned_workspace as workspace
from scripts.research.device_loop_workspace import code_state

ORIGINAL_PATH=ROOT/"scripts/research/summarize_gpu_icp_complete_bridge.py"
ORIGINAL_SHA256="fe1a8b7a30a8b56fac6162b0b9855ea4ee067abe20a51f02c0f8a06c057f25dd"
KIND="gpu-icp-complete-bridge-owned-proof-scalar-summary-v2"
TIMING_KIND=owned.TIMING_KIND
OWN_FILES=("scripts/research/summarize_gpu_icp_owned_bridge.py",
    "tests/test_gpu_icp_owned_bridge_summary.py",
    "scripts/research/summarize_gpu_icp_complete_bridge.py",
    "tests/test_gpu_icp_complete_bridge_summary.py")


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive():
    require(Path(original.__file__).resolve()==ORIGINAL_PATH and sha(ORIGINAL_PATH)==ORIGINAL_SHA256,
        "Exact measured original scalar publisher required")
    compiled=compile(ORIGINAL_PATH.read_text(encoding="utf-8"),str(ORIGINAL_PATH),"exec",dont_inherit=True)
    expected={code.co_name:code for code in compiled.co_consts if hasattr(code,"co_code")}
    # These are pure guards only. No constructor, mint, registry, lock or exit API.
    protocol=SimpleNamespace(KIND=owned.KIND,BRIDGE_FILES=tuple(driver.OLD_FILES+driver.NEW_FILES),
        _resource_binding=guards._resource_binding,_call_for_common=guards._call_for_common,
        _validate_one=guards._validate_one,values=guards.values,_full_input=guards._full_input)
    namespace=dict(original.__dict__)
    namespace.update(KIND=KIND,TIMING_KIND=TIMING_KIND,OWN_FILES=OWN_FILES,
        protocol=protocol,workspace=workspace,__name__=__name__,__file__=__file__)
    records=[]
    for name,value in original.__dict__.items():
        if isinstance(value,FunctionType) and value.__globals__ is original.__dict__:
            require(value.__closure__ is None and code_state(value.__code__)==code_state(expected[name]),
                "Original loaded scalar function differs from source")
            clone=FunctionType(value.__code__,namespace,name,value.__defaults__)
            clone.__kwdefaults__=value.__kwdefaults__
            namespace[name]=clone
            global_owners=tuple((key,original.__dict__[key]) for key in value.__code__.co_names if key in original.__dict__)
            records.append((name,value,value.__code__,repr(value.__defaults__),repr(value.__kwdefaults__),global_owners,clone))
    return namespace,tuple(records),protocol


_NAMESPACE,_RECORDS,_PURE_PROTOCOL=derive()
_ORIGINAL_CLOSED=_NAMESPACE["closed"]
_ORIGINAL_COMPACT=_NAMESPACE["compact"]
_ORIGINAL_RUN=_NAMESPACE["run"]
_PURE_FIELDS=tuple(vars(_PURE_PROTOCOL).items())
_MODULE_OWNERS=tuple((m,m.__name__,Path(m.__file__).resolve()) for m in (original,guards,owned,driver,workspace))


def pure_guard_records():
    records=[];sources=[]
    for module in (guards,guards.values):
        path=Path(module.__file__).resolve();sources.append((module,path,sha(path)))
        compiled=compile(path.read_text(encoding="utf-8"),str(path),"exec",dont_inherit=True)
        expected={code.co_name:code for code in compiled.co_consts if hasattr(code,"co_code")}
        for name,value in vars(module).items():
            if isinstance(value,FunctionType) and value.__globals__ is module.__dict__:
                require(code_state(value.__code__)==code_state(expected[name]),"Loaded pure scalar guard differs from source")
                owners=tuple((key,module.__dict__[key]) for key in value.__code__.co_names if key in module.__dict__)
                records.append((module,name,value,value.__code__,repr(value.__defaults__),repr(value.__kwdefaults__),owners))
    return tuple(records),tuple(sources)


_PURE_RECORDS,_PURE_SOURCES=pure_guard_records()


def check_loaded():
    require(sha(ORIGINAL_PATH)==ORIGINAL_SHA256,"Measured publisher bytes changed")
    for module,name,path in _MODULE_OWNERS:
        require(sys.modules.get(name) is module and Path(module.__file__).resolve()==path,"Loaded scalar guard module owner changed")
    for name,value,code,defaults,keywords,globals_owners,clone in _RECORDS:
        require(getattr(original,name) is value and value.__code__ is code and value.__globals__ is original.__dict__
            and repr(value.__defaults__)==defaults and repr(value.__kwdefaults__)==keywords
            and all(original.__dict__.get(key) is owner for key,owner in globals_owners),
            "Original scalar function/code/default/global owner changed")
        require(clone.__code__ is code and clone.__globals__ is _NAMESPACE
            and repr(clone.__defaults__)==defaults and repr(clone.__kwdefaults__)==keywords,
            "Private scalar clone/code/default owner changed")
        require(_NAMESPACE.get(name) is ({"closed":closed,"compact":compact}.get(name,clone)),
            "Private scalar guard alias changed")
    require(all(getattr(_PURE_PROTOCOL,key) is value for key,value in _PURE_FIELDS),"Pure guard exposure changed")
    require(_NAMESPACE.get("protocol") is _PURE_PROTOCOL and _NAMESPACE.get("workspace") is workspace,
        "Private pure guard/workspace namespace changed")
    for module,path,fingerprint in _PURE_SOURCES:
        require(sys.modules.get(module.__name__) is module and Path(module.__file__).resolve()==path and sha(path)==fingerprint,
            "Pure scalar guard module/source owner changed")
    for module,name,value,code,defaults,keywords,owners in _PURE_RECORDS:
        require(getattr(module,name) is value and value.__code__ is code and value.__globals__ is module.__dict__
            and repr(value.__defaults__)==defaults and repr(value.__kwdefaults__)==keywords
            and all(module.__dict__.get(key) is owner for key,owner in owners),"Pure scalar guard code/default/global owner changed")


def closed(report,mode):
    binding=_ORIGINAL_CLOSED(report,mode)
    proof_contract=owned.source_contract();driver_contract=driver.source_contract()
    require(binding.get("owned_proof_source")==proof_contract and binding.get("owned_driver_source")==driver_contract
        and binding.get("workspace_source")==workspace.source_contract(),"Exact owned protocol/driver/workspace envelope required")
    require(all(binding["artifacts_sha256"].get(name)==value for contract in (proof_contract,driver_contract)
        for name,value in contract["artifacts"].items()),"Owned envelope source family omitted")
    if mode=="audit":require(report.get("owned_proof")=={"required":False},"Audit must not claim an owned timing lock")
    return binding


def compact(audit,timing,audit_sha):
    check_loaded()
    receipt=timing.get("owned_proof",{})
    require(receipt.get("policy")==owned.POLICY and receipt.get("required") is True and receipt.get("closed") is True and receipt.get("failure") is None
        and receipt.get("hash_initial")==receipt.get("hash_final")==audit_sha
        and type(receipt.get("held_handle_reads")) is int and receipt["held_handle_reads"]==2
        and all(receipt.get(key) is True for key in ("share_read_only","decoded_immutable"))
        and receipt.get("old_token_consumed") is False and receipt.get("cleanup_failures")==[],
        "Timing needs its exact two-read closed owned immutable proof receipt")
    result=_ORIGINAL_COMPACT(audit,timing,audit_sha)
    result["owned_proof_receipt"]={"policy":owned.POLICY,"closed":True,"audit_sha256":audit_sha,"held_handle_reads":2,
        "share_read_only":True,"decoded_immutable":True,"old_token_consumed":False}
    check_loaded()
    return result


_NAMESPACE.update(closed=closed,compact=compact)


def run(args):
    output=Path(args.output).resolve()
    require(output.is_relative_to(ROOT/"benchmark-output") and not output.exists(),"Fresh private summary path required")
    before={name:sha(ROOT/name) for name in OWN_FILES}
    check_loaded()
    try:
        _ORIGINAL_RUN(args)
        check_loaded()
        require(before=={name:sha(ROOT/name) for name in OWN_FILES},"Publisher family changed during publication")
    except BaseException as primary:
        # Existing-output refusal is retained; never relabel an older report.
        if output.exists() and output.is_relative_to(ROOT/"benchmark-output"):
            try:
                report=json.loads(output.read_text(encoding="utf-8"))
                if report.get("kind")==KIND:
                    report.update(status="failed",failure={"type":type(primary).__name__,"message":str(primary)})
                    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
            except BaseException as secondary:raise primary from secondary
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--timing",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
