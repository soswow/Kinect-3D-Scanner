"""Component-only causal A/B source guard timing with unchanged CUDA math.

Run separately with --source-check-mode original-ast and whole-file-sha.
Both inherit the proved original all-nine three-round native/device/combined
protocol, fresh proof validation, caps, streams, CPU ambiguity resolution and
original Eigen/convergence. Identical nested host clocks apply in both modes.
No full-Finish or production registration authority is created.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research import benchmark_combined_sync_timing as parent
from scripts.research import combined_source_guard_adapter as adapter
from scripts.research.validate_combined_source_guard_timing import (
    ARTIFACTS,MODES,POLICY,validate_source_guard_authority,validate_source_guard_activation)
from scripts.research.archive.validate_uniform_grid_proof import file_hash

KIND="standalone-original-double-combined-source-guard-component-timing-v1"


def parser():
    value=parent.parser()
    value.description=__doc__
    value.add_argument("--source-check-mode",choices=MODES,default="whole-file-sha")
    return value


def owned_bridge_constructor(factory,original_bridge=None):
    """Keep the entire original bridge body, removing only its local class import.

    The imported constructor is supplied as a private global factory bound to
    the new authority. Reverse insertion recovers the complete original AST;
    no hooks/globals in the imported adapter are changed.
    """
    captured=parent.bridge_class if original_bridge is None else original_bridge
    tree=ast.parse(textwrap.dedent(inspect.getsource(captured)))
    original=ast.dump(tree,include_attributes=False)
    node=tree.body[0]
    imports=[n for n in node.body if isinstance(n,ast.ImportFrom)
        and n.module=="scripts.research.combined_sync_timing_adapter"]
    if len(imports)!=1:raise RuntimeError("Original bridge constructor import seam changed")
    imported=imports[0];before=copy.deepcopy(imported.names)
    if [(n.name,n.asname) for n in before]!=[("CombinedSyncTimedResidentICP",None),("CombinedSyncTimingFailure",None)]:
        raise RuntimeError("Original bridge constructor aliases changed")
    imported.names=[copy.deepcopy(before[1])]
    recovered=copy.deepcopy(tree)
    restored=next(n for n in recovered.body[0].body if isinstance(n,ast.ImportFrom)
        and n.module=="scripts.research.combined_sync_timing_adapter")
    restored.names=before
    if ast.dump(recovered,include_attributes=False)!=original:
        raise RuntimeError("Constructor injection changed original bridge control/math")
    namespace=dict(parent.__dict__,CombinedSyncTimedResidentICP=factory)
    exec(compile(ast.fix_missing_locations(tree),__file__,"exec"),namespace)
    return namespace["bridge_class"],{
        "original_bridge_ast_sha256":hashlib.sha256(original.encode()).hexdigest(),
        "constructor_import_only":True,"original_entire_bridge_body_recoverable":True}


def restore_hooks(owned):
    errors=[]
    for module,name,original in reversed(owned):
        try:
            setattr(module,name,original)
            if getattr(module,name) is not original:raise RuntimeError("Owned hook identity was not restored")
        except BaseException as error:
            errors.append({"action":name,"type":type(error).__name__,"message":str(error)})
    return errors


def close_report(path,mode,pins,tokens,constructor_contract,cleanup_errors):
    report=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report,dict):raise RuntimeError("Inherited timing report is not an object")
    after={name:file_hash(ROOT/name) for name in ARTIFACTS}
    rows=[row for row in report.get("real_runs",[]) if row.get("mode")=="combined_sync"]
    previous={"source_checks":0,"iteration_config_checks":0,"preenqueue_calls":0,
        "source_check_s":0.,"iteration_config_check_s":0.,"preenqueue_bookkeeping_s":0.}
    healthy=(report.get("status")=="passed" and report.get("kind")==KIND and pins==after
        and not cleanup_errors and len(tokens)==1 and len(rows)==3)
    for row in rows:
        actual=row["device_resident"]["source_guard_timing"]
        statistics=actual["statistics"]
        difference={key:statistics[key]-previous[key] for key in previous}
        previous=dict(statistics)
        row["source_guard_statistics_delta"]=difference
        healthy=healthy and (actual["mode"]==mode and actual["source_guard_sources"]==pins
            and actual["source_unchanged"] is True and actual["original_contract_equality_preserved"] is True
            and difference["source_checks"]==236 and difference["iteration_config_checks"]==11569
            and difference["preenqueue_calls"]==11569
            and all(type(value) in (float,int) and value>=0 for value in difference.values()))
    for token in tokens:
        validate_source_guard_activation(token,token.original_timing_authority,mode)
    report["source_guard_scope"]={"policy":POLICY,"mode":mode,"artifacts_sha256":pins,
        "artifacts_sha256_after":after,"sources_unchanged":pins==after,
        "owned_hooks_restored":not cleanup_errors,"cleanup_failures":cleanup_errors,
        "constructor_contract":constructor_contract,"new_kernel":False,"graph_capture":False,
        "whole_finish_authority":False,"identical_clocks_in_both_modes":True,
        "timer_scope":"Source checks and preenqueue wall are inclusive nested host timers. Both modes retain the complete original all-nine protocol. No timer subtraction establishes registration throughput."}
    if not healthy:
        report["status"]="failed"
        report["source_guard_closure_failure"]="New source-guard component counters/source/restore closure failed"
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if not healthy:raise RuntimeError(report["source_guard_closure_failure"])
    return report


def preserve_failure(path,error,mode=None,pins=None,cleanup=None):
    # The inherited helper preserves malformed partial bytes and the primary
    # failure. Its KIND has already been restored, so label this current scope
    # explicitly even when the inherited run failed before its first save.
    parent.preserve_failure(path,error)
    try:
        report=json.loads(path.read_text(encoding="utf-8"))
        report.update(kind=KIND,status="failed")
        report.setdefault("source_guard_scope",{}).update(policy=POLICY,mode=mode,
            whole_finish_authority=False,current_failure=True)
        if pins is not None:report["source_guard_scope"]["artifacts_sha256"]=pins
        if cleanup is not None:report["source_guard_scope"]["cleanup_failures"]=cleanup
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    except BaseException as secondary:
        error.add_note(f"New source-guard failed-scope annotation also failed: {secondary}")
        raise error from secondary


def run(args):
    if not args.run_allocated:raise ValueError("Require an explicitly allocated exclusive hardware slot")
    if args.output.exists():raise ValueError("Require a fresh output path; preserve previous reports")
    # Deferred worker is standard-library-only and is restored before execution.
    from scripts import process_metrics
    pins={name:file_hash(ROOT/name) for name in ARTIFACTS}
    owned=[];tokens=[];deferred=[];failure=None;cleanup=[];contract=None
    original_bridge=parent.bridge_class
    original_worker=process_metrics.finish_cuda_worker
    def bridge(library,scratch,query,authority,combined_mode):
        nonlocal contract
        if not combined_mode:return original_bridge(library,scratch,query,authority,False)
        token=validate_source_guard_authority(authority,args.source_check_mode,pins)
        tokens.append(token)
        def construct(*values,**options):
            return adapter.SourceGuardResidentICP(*values,source_guard_authority=token,
                source_check_mode=args.source_check_mode,**options)
        constructor,contract=owned_bridge_constructor(construct,original_bridge)
        return constructor(library,scratch,query,authority,True)
    def install(module,name,value):
        # Enroll before setattr: even a partially failing setter must be undone.
        owned.append((module,name,getattr(module,name)))
        setattr(module,name,value)
    try:
        install(parent,"bridge_class",bridge)
        install(parent,"KIND",KIND)
        install(process_metrics,"finish_cuda_worker",lambda:deferred.append(True))
        parent.run(args)
    except BaseException as error:
        failure=error
    finally:
        cleanup=restore_hooks(owned)
    try:
        if failure is not None:raise failure
        if deferred!=[True]:raise RuntimeError("Inherited worker completion was not deferred exactly once")
        report=close_report(args.output,args.source_check_mode,pins,tokens,contract,cleanup)
        print(json.dumps({"output":str(args.output),"status":report["status"],
            "source_check_mode":args.source_check_mode}),flush=True)
    except BaseException as error:
        if failure is not None and error is not failure:failure.add_note(f"Additional source guard closure error: {error}")
        primary=failure or error
        for item in cleanup:primary.add_note(f"Owned source-guard cleanup: {item}")
        preserve_failure(args.output,primary,args.source_check_mode,pins,cleanup)
        raise primary
    # Must follow final report closure and restoration of all owned hooks.
    original_worker()


def main():
    args=parser().parse_args()
    if args.output.exists():parser().error("Require fresh output; preserve all previous reports")
    try:run(args)
    except BaseException as error:
        if isinstance(error,SystemExit):raise
        preserve_failure(args.output,error,args.source_check_mode)
        raise


if __name__=="__main__":main()
