"""Path-free scalar census; no reuse, speed, quality token or native dispatch."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research import benchmark_icp_seed_reuse_census as census
from scripts.research.device_loop_workspace import code_state

KIND="original-cpu-seed-reuse-scalar-census-v1"
DRIVER="scripts/research/benchmark_icp_seed_reuse_census.py"
DRIVER_SHA256="a02bddc6e2559d712c0c2f8b5963057b010a1cb805b70aa8c7c1a792c6c6eb49"
OWN_FILES=("scripts/research/summarize_icp_seed_reuse_census.py","tests/test_icp_seed_census_summary.py")


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helper_records():
    path=ROOT/DRIVER;require(sha(path)==DRIVER_SHA256,"Frozen original census source required")
    expected={code.co_name:code for code in compile(path.read_text(encoding="utf-8"),str(path),"exec",dont_inherit=True).co_consts if hasattr(code,"co_code")}
    records=[]
    for name in ("require","matrix_values","max_delta","cloud_bucket","result_delta","cluster_rows"):
        fn=getattr(census,name)
        require(isinstance(fn,FunctionType) and fn.__globals__ is census.__dict__ and code_state(fn.__code__)==code_state(expected[name]),
            "Loaded offline clustering helper differs from original source")
        records.append((name,fn,fn.__code__,repr(fn.__defaults__),tuple((k,census.__dict__[k]) for k in fn.__code__.co_names if k in census.__dict__)))
    return tuple(records)


_HELPERS=helper_records()
_CANONICAL=census.scope.canonical
_SCOPE_PATH=ROOT/"scripts/research/microbatch_bridge_scope.py"
_SCOPE_CODE=compile(_SCOPE_PATH.read_text(encoding="utf-8"),str(_SCOPE_PATH),"exec",dont_inherit=True)
_EXPECTED_CANONICAL=next(code for code in _SCOPE_CODE.co_consts if hasattr(code,"co_code") and code.co_name=="canonical")
require(sha(_SCOPE_PATH)==census.HELD_FILES["scripts/research/microbatch_bridge_scope.py"]
    and _CANONICAL.__globals__ is census.scope.__dict__ and code_state(_CANONICAL.__code__)==code_state(_EXPECTED_CANONICAL),
    "Initial canonical receipt serializer must match pinned original source")
_CANONICAL_CODE=_CANONICAL.__code__
_CANONICAL_DEFAULTS=repr(_CANONICAL.__defaults__)
_CANONICAL_GLOBALS=tuple((key,_CANONICAL.__globals__[key]) for key in _CANONICAL_CODE.co_names if key in _CANONICAL.__globals__)


def check_helpers():
    require(sha(ROOT/DRIVER)==DRIVER_SHA256 and Path(census.__file__).resolve()==ROOT/DRIVER,
        "Frozen census source/module changed")
    for name,fn,code,defaults,owners in _HELPERS:
        require(getattr(census,name) is fn and fn.__code__ is code and fn.__globals__ is census.__dict__
            and repr(fn.__defaults__)==defaults and all(census.__dict__.get(k) is owner for k,owner in owners),
            "Offline clustering helper code/default/global owner changed")
    require(census.scope.canonical is _CANONICAL and _CANONICAL.__code__ is _CANONICAL_CODE
        and repr(_CANONICAL.__defaults__)==_CANONICAL_DEFAULTS
        and all(_CANONICAL.__globals__.get(key) is owner for key,owner in _CANONICAL_GLOBALS),
        "Recorded canonical receipt serializer owner changed")


def compact(report):
    check_helpers()
    require(report.get("kind")==census.KIND and report.get("status")=="passed" and report.get("failure") is None
        and report.get("cleanup_passed") is True and report.get("cleanup_failures")==[]
        and report.get("original_cpu_results_returned_unchanged") is True
        and report.get("skipped_icp_calls")==0 and report.get("timing_authority") is False and report.get("whole_finish_authority") is False,
        "Closed original CPU census without skipped work/authority required")
    binding=report["binding"]
    require(binding==report.get("binding_after") and binding.get("source_sha256")==census.CURRENT
        and type(binding.get("fixed_files")) is dict and binding["fixed_files"]
        and type(binding.get("verification_dependencies")) is dict and binding["verification_dependencies"],"Recorded source/input/runtime/dependency closure required")
    artifacts=binding["artifacts_sha256"]
    require(type(artifacts) is dict and 4<=len(artifacts)<=200 and artifacts.get(DRIVER)==DRIVER_SHA256,"Bounded source family/current original driver required")
    for name,digest in artifacts.items():
        path=(ROOT/name).resolve();require(path.is_relative_to(ROOT) and sha(path)==digest,"Recorded census helper source changed")
    for name,digest in census.HELD_FILES.items():require(artifacts.get(name)==digest,"Original verifier source family omitted")
    runtime=binding["runtime"]
    require(runtime.get("registration")=="original legacy CPU" and runtime.get("thread_policy")=={"opencv":20,"open3d":20,"omp":"8"}
        and runtime.get("environment")==census.ENVIRONMENT and runtime.get("binary_sha256")
        and all(binding["fixed_files"].get(path)==digest for path,digest in runtime["binary_sha256"].items()),"Recorded original native/runtime/thread policy required")
    calls=[];pairs=proposals=gates=0
    captures=report["captures"];require(type(captures) is list and len(captures)==2,"Original two capture groups required")
    for capture_index,capture in enumerate(captures):
        require(binding["fixed_files"].get(capture["capture"])==capture["capture_sha256"],"Original capture receipt differs from input closure")
        for position,pair in enumerate(capture["tasks"]):
            pairs+=1
            require(pair.get("position")==position and pair.get("input_bytes_unchanged") is True and pair.get("seed_bytes_unchanged") is True,
                "Original ordered pair/seed/input closure required")
            rows=pair["proposals"];require(type(rows) is list and 1<=len(rows)<=6 and pair["original_pair_verdict"].get("proposal_count")==len(rows),"Every genuine proposal required")
            for proposal_index,row in enumerate(rows):
                proposals+=1
                require(row.get("proposal_index")==proposal_index and row.get("complete") is True and "result" in row
                    and type(row.get("gates")) is list and row["gates"] and all(g.get("complete") is True for g in row["gates"]),"Complete original proposal/gate receipts required")
                gates+=len(row["gates"])
                require(type(row.get("calls")) is list and row["calls"],"Original dynamic CPU calls required")
                for call_index,call in enumerate(row["calls"]):
                    require(call.get("call_index")==call_index and call.get("context")=={"capture_index":capture_index,"task":position,"pair":pair["pair"],"proposal_index":proposal_index}
                        and call.get("complete") is True and call.get("input_bytes_unchanged") is True,"Ordered unchanged original call scope required")
                    census.matrix_values(call["result"]["transformation"])
                    require(all(type(call["result"].get(k)) in (int,float) and math.isfinite(call["result"][k]) for k in ("fitness","inlier_rmse")),"Finite original result metrics required")
                    calls.append(call)
    require((pairs,proposals,len(calls),report.get("call_count"))==(9,30,714,714),"Complete original nine-pair/30-proposal/714-call scope required")
    require(hashlib.sha256(census.scope.canonical(calls).encode()).hexdigest()==report.get("ordered_calls_sha256"),"Original ordered results/inputs receipt changed")
    fresh=[census.cluster_rows(calls,threshold) for threshold in census.THRESHOLDS]
    require(fresh==report.get("clustering") and all(row["exact_cloud_value_buckets"]==292 for row in fresh),"Exact offline first-representative groups/counts/results changed")
    check_helpers()
    return {"status":"passed","source_sha256":census.CURRENT,"pairs":pairs,"genuine_proposals":proposals,"original_cpu_calls":len(calls),
        "original_gate_events":gates,"exact_directed_cloud_value_buckets":292,
        "thresholds":[{k:v for k,v in row.items() if k!="groups"} for row in fresh],
        "skipped_calls":0,"safe_reuse_established":False,"speed_authority":False,"quality_authority":False,"whole_finish_authority":False,
        "scope":"Offline original-CPU observations only; threshold groups are candidates for a separate reuse experiment, not saved calls or guaranteed equivalent future results."}


def run(args):
    source,output=(Path(value).resolve() for value in (args.report,args.output))
    require(source.is_relative_to(ROOT/"benchmark-output") and output.is_relative_to(ROOT/"benchmark-output") and not output.exists(),"Private local receipt and fresh output required")
    record={"kind":KIND,"status":"running","performance_authority_token_created":False,"fixed_private_resources_rescanned":False,"os_child_exit_independently_verified":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():output.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    save()
    try:
        require(source.stat().st_size<=32*1024**2,"Bounded census receipt required")
        before=sha(source);own={name:sha(ROOT/name) for name in OWN_FILES}
        record.update(compact(json.loads(source.read_text(encoding="utf-8"))))
        require(sha(source)==before and own=={name:sha(ROOT/name) for name in OWN_FILES},"Census/publisher source changed during publication")
        record.update(source_report_sha256=before,publisher_source_family_sha256=hashlib.sha256(census.scope.canonical(own).encode()).hexdigest(),status="passed");save()
    except BaseException as primary:
        record.update(status="failed",failure={"type":type(primary).__name__,"message":str(primary)})
        try:save()
        except BaseException as secondary:raise primary from secondary
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
