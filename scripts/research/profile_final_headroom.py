"""Separate reserve-headroom supervisor of the frozen original Final producer.

Both native-original and exact-rightsized retain original CPU registration.
The second mode changes physical initial capacity, keeping the unique hardcap.
All inherited session/checkpoint/component/fresh-output options remain required.
Use --physical-block-budget explicitly; no old allocation/Finish authority is
relabelled. Windows worker termination is deferred until this envelope closes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
KIND = "offline-original-cpu-finish-reserve-headroom-v2"
NEW_ARTIFACTS = ("scripts/research/final_allocation_headroom.py","scripts/research/profile_final_headroom.py")
ORIGINAL_PRODUCER = "scripts/research/profile_final_allocation.py"
ORIGINAL_PRODUCER_SHA256 = "ce9376d8003c570ba3e5584253a8c7cd616f3324ab7c5ab412a737948d0cd042"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(hooks,saved_argv):
    errors = []
    for owner,name,value in reversed(hooks):
        try:
            setattr(owner,name,value)
            if getattr(owner,name) is not value:
                raise RuntimeError("Headroom supervisor hook failed to restore original object")
        except BaseException as error:
            errors.append(error)
    try:
        sys.argv = saved_argv
    except BaseException as error:
        errors.append(error)
    return errors


def run(args,forwarded,*,base=None,metrics=None):
    if type(args.physical_block_budget) is not int or not 1 <= args.physical_block_budget <= 100000:
        raise ValueError("Require explicit bounded physical capacity budget")
    from scripts.research.final_allocation_headroom import HeadroomFinalScope,ObservedOriginalHeadroomScope
    if base is None:
        from scripts.research import profile_final_allocation as base
    if metrics is None:
        from scripts import process_metrics as metrics
    if sha(ROOT/ORIGINAL_PRODUCER) != ORIGINAL_PRODUCER_SHA256:
        raise RuntimeError("Frozen original allocation producer changed")
    pins = {name:sha(ROOT/name) for name in (*base.EXPERIMENT_ARTIFACTS,*NEW_ARTIFACTS)}
    envelope = args.output.with_suffix(".headroom.json")
    sidecar = args.output.with_suffix(".allocation.json")
    if envelope.exists() or args.output.exists() or sidecar.exists():
        raise RuntimeError("Require fresh headroom envelope/profile/sidecar; preserve earlier artifacts")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report = {"kind":KIND,"mode":args.mode,"status":"running","physical_block_budget":args.physical_block_budget,
        "artifacts_sha256":pins,"original_producer_sha256":ORIGINAL_PRODUCER_SHA256,
        "scope":"Unchanged original CPU Finish/gates/checkpoint producer with separately declared reserve-headroom allocator. Both modes charge matched exact frustum observation inside Finish. Original unique limit unchanged; physical capacity distinct.",
        "geometry_quality_proven":False,"performance_authority":False,"cleanup_failures":[]}
    def save():
        envelope.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    save()
    saved_argv = sys.argv
    hooks,deferred,primary = [],[],None
    original_worker = metrics.finish_cuda_worker
    def replace(owner,name,value):
        old = getattr(owner,name)
        hooks.append((owner,name,old))
        setattr(owner,name,value)
    def allocator(engine,**kwargs):
        return HeadroomFinalScope(engine,physical_budget=args.physical_block_budget,**kwargs)
    def control(engine,**kwargs):
        return ObservedOriginalHeadroomScope(engine,physical_budget=args.physical_block_budget,**kwargs)
    try:
        replace(base,"KIND",KIND)
        replace(base,"EXPERIMENT_ARTIFACTS",tuple(pins))
        replace(base,"RightSizedFinalScope",allocator)
        replace(base,"ObservedOriginalFinalScope",control)
        replace(metrics,"finish_cuda_worker",lambda:deferred.append(True))
        sys.argv = [str(ROOT/ORIGINAL_PRODUCER),"--mode",args.mode,"--output",str(args.output),"--run-allocated",*forwarded]
        base.main()
        if deferred != [True]:
            raise RuntimeError("Expected exactly one deferred original terminal worker request")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if (data.get("kind") != KIND or data.get("status") != "complete" or data.get("artifacts_sha256") != pins
                or data.get("artifacts_sha256_after") != pins):
            raise RuntimeError("Delegated new-kind allocation sidecar/source closure is incomplete")
        if data.get("allocation",{}).get("kind") != "offline-original-final-reserve-headroom-v2":
            raise RuntimeError("Delegated allocator did not record the new reserve-headroom policy")
        report["delegated_sidecar"] = {"path":str(sidecar),"sha256":sha(sidecar)}
        report["profile"] = {"path":str(args.output),"sha256":sha(args.output)}
        report["outcome"] = data.get("outcome")
        report["allocation"] = data["allocation"]
        report["status"] = "complete"
    except BaseException as error:
        primary = error
        report.update(status="failed",failure={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()})
        raise
    finally:
        errors = restore(hooks,saved_argv)
        report["supervisor_restored"] = not errors
        report["deferred_terminal_worker_calls"] = len(deferred)
        try:
            report["artifacts_sha256_after"] = {name:sha(ROOT/name) for name in pins}
            if report["artifacts_sha256_after"] != pins:
                raise RuntimeError("Headroom original/new sources changed during execution")
            if sidecar.exists():
                report["delegated_sidecar_after"] = {"path":str(sidecar),"sha256":sha(sidecar)}
                if "delegated_sidecar" in report and report["delegated_sidecar_after"] != report["delegated_sidecar"]:
                    raise RuntimeError("Delegated allocation report changed during supervisor closure")
            if "profile" in report:
                report["profile_after"] = {"path":str(args.output),"sha256":sha(args.output)}
                if report["profile_after"] != report["profile"]:
                    raise RuntimeError("Delegated profile changed during supervisor closure")
        except BaseException as error:
            errors.append(error)
        report["cleanup_failures"] = [{"type":type(error).__name__,"message":str(error)} for error in errors]
        if errors:
            report["status"] = "failed"
        try:
            save()
        except BaseException as write_error:
            if primary is not None:
                primary.add_note(f"Headroom envelope write also failed: {write_error}")
                raise primary from write_error
            if errors:
                errors[0].add_note(f"Headroom envelope write also failed: {write_error}")
                raise errors[0] from write_error
            raise
        if errors:
            if primary is not None:
                primary.add_note(f"Headroom supervisor cleanup also failed: {errors}")
            else:
                raise errors[0]
    # All original numerical cleanup, new source/report closure and hook restore
    # have succeeded before the known Windows DLL finalizer workaround executes.
    original_worker()


def main():
    parser = argparse.ArgumentParser(description=__doc__,epilog="Forward session, checkpoint, component proofs and logical-budget/expected-failure options from profile_final_allocation.py.")
    parser.add_argument("--mode",choices=("native-original","exact-rightsized"),required=True)
    parser.add_argument("--physical-block-budget",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run-allocated",action="store_true")
    args,forwarded = parser.parse_known_args()
    if not args.run_allocated or not 1 <= args.physical_block_budget <= 100000:
        parser.error("Require exclusive allocation and explicit physical capacity budget1–100000")
    args.output = args.output.resolve()
    try:
        args.output.relative_to(ROOT/"benchmark-output")
    except ValueError:
        parser.error("Keep fresh private reports inside ignored benchmark-output")
    run(args,forwarded)


if __name__ == "__main__":
    main()
