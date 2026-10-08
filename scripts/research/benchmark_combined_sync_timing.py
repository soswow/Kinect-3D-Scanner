"""Three-way all-nine component timing under separate fresh proof authority.

Fresh combined and original Device proofs are revalidated against actual
installed libraries/configuration before any unaudited registration begins.
Every round retains the original fixture proposals/order/bridge gates, with
contemporary native decisions/witnesses/poses/information as the comparison.
This is neither a complete Finish experiment nor a production backend.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
KIND="standalone-original-double-combined-sync-component-timing-v1"
FROZEN="9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a"
from scripts.research.archive.validate_uniform_grid_proof import file_hash,canonical_hash
from scripts.research.validate_combined_sync_timing import ARTIFACTS,POLICY,validate_timing_authority


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-allocated",action="store_true")
    value.add_argument("--output",type=Path,required=True)
    value.add_argument("--combined-synthetic",type=Path,default=ROOT/"benchmark-output/field-cuda-study/combined-sync-v1/synthetic-proof.json")
    value.add_argument("--combined-bridge",type=Path,default=ROOT/"benchmark-output/field-cuda-study/combined-sync-v1/bridge-audit.json")
    value.add_argument("--device-synthetic",type=Path,default=ROOT/"benchmark-output/field-cuda-study/device-layout-v2/synthetic-proof.json")
    value.add_argument("--device-bridge",type=Path,default=ROOT/"benchmark-output/field-cuda-study/device-layout-v2/bridge-audit.json")
    value.add_argument("--fixture",type=Path,default=ROOT/"benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.fixture.pickle")
    value.add_argument("--reference",type=Path,default=ROOT/"benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.json")
    value.add_argument("--solver-dll",type=Path,default=ROOT/"benchmark-output/cuda-pipeline/resident-icp/resident_solve.dll")
    value.add_argument("--repeats",type=int,choices=(3,),default=3)
    return value


def rebuild_runtime(path,cp,np,o3d,cv2,configuration,cache):
    """Report supplies only the declared source set; actual hashes are rebuilt."""
    from scripts.profile_session import source_hash
    from scripts.research.benchmark_parallel_fragments import numerical_source_hash
    from scripts.process_metrics import gpu_info
    from scripts.research import benchmark_device_grid_resident as device
    from scripts.research import benchmark_combined_sync_icp as combined
    from scripts.research.cuda_device_flat_grid_registration import DOMAIN
    before=file_hash(path)
    saved=json.loads(path.read_text(encoding="utf-8"))
    actual=copy.deepcopy(saved)
    actual.update(source_sha256=source_hash(),component_source_sha256=numerical_source_hash(),gpu=gpu_info(),
        artifacts_sha256={name:file_hash(ROOT/name) for name in saved["artifacts_sha256"]},domain=DOMAIN,
        versions={"numpy":np.__version__,"open3d":o3d.__version__,"opencv":cv2.__version__,"cupy":cp.__version__},
        thread_policy={"open3d":o3d.utility.get_max_threads(),"opencv":cv2.getNumThreads(),"omp":os.environ["OMP_NUM_THREADS"]},
        resident_configuration=dict(configuration),cache_policy=dict(cache))
    if saved["kind"]==combined.__dict__.get("KIND","standalone-original-double-combined-sync-resident"):
        audit=Path(saved["cpu_audit_proof"]["path"]).resolve(strict=True)
        actual["cpu_audit_proof"]={"path":str(audit),"sha256":file_hash(audit)}
        binding=combined.current_runtime_binding(actual,cp,np,o3d)
        # Independently rebuild actual bulk runtime/API/artifact configuration;
        # no CPU shadow is run and no timing registration consumes this auditor.
        combined.current_bulk_authority(audit,cp,np,o3d,cv2)
    else:
        binding=device.current_runtime_binding(actual,cp,np,o3d)
    if file_hash(path)!=before:
        raise RuntimeError("Closed component report changed during runtime reconstruction")
    return binding


def bridge_class(library,scratch,query,authority,combined_mode):
    from scripts.research.cuda_device_flat_grid_registration import DeviceFlatGridICP
    from scripts.research.research_device_grid_resident_icp import DeviceGridResidentICP
    from scripts.research.combined_sync_timing_adapter import CombinedSyncTimedResidentICP,CombinedSyncTimingFailure
    from scripts.research.archive.research_resident_icp import EigenSolve
    class TimingBridge(DeviceFlatGridICP):
        def __init__(self,**kwargs):
            super().__init__(max_query_bytes=query,**kwargs)
            self.resident=None
            self.failure=None
            try:
                self.solve=EigenSolve(library)
                options={"max_points":1000000,"max_scratch_bytes":scratch,"gpu_timing":False,"owns_retrieval":False}
                self.resident=(CombinedSyncTimedResidentICP(self,self.solve,authority=authority,**options)
                    if combined_mode else DeviceGridResidentICP(self,self.solve,**options))
                self.resident.cpu_fallback=self.forbid_full_fallback
            except BaseException as error:
                for action in ((self.resident.close,) if self.resident is not None else ())+(super().close,):
                    try:action()
                    except BaseException as cleanup:error.add_note(f"Partial timing construction cleanup: {cleanup}")
                raise
        @staticmethod
        def forbid_full_fallback(*args,**kwargs):
            raise CombinedSyncTimingFailure("Unsupported complete component call; CPU retry prohibited")
        def match(self,source,target,initial):
            if self.failure is not None:raise self.failure
            started=time.perf_counter()
            try:
                return self.resident.match(source,target,initial)
            except BaseException as error:
                self.failure=CombinedSyncTimingFailure("Device/combined component registration failed")
                raise self.failure from error
            finally:
                self.statistics["icp_calls"]+=1
                self.statistics["icp_wall_s"]+=time.perf_counter()-started
        def close(self):
            primary=sys.exception()
            errors=[]
            for action in ((self.resident.close,) if self.resident is not None else ())+(super().close,):
                try:action()
                except BaseException as error:errors.append(error)
            if errors:
                if primary is not None:primary.add_note(f"Timing owned cleanup also failed: {errors}")
                else:
                    for error in errors[1:]:errors[0].add_note(f"Additional cleanup failure: {error}")
                    raise errors[0]
    return TimingBridge


def delta(after,before):
    return {key:value-before[key] for key,value in after.items()}


def compare_round(runs,values,reference,baseline,agreement):
    """Compare every rotating-order result only after contemporary CPU closes."""
    native=next(run for run in runs if run["mode"]=="native_cpu")
    lookup={pair["position"]:pair for pair in native["pairs"]}
    original=[row for result in reference["results"] if result["configuration"]=="1x20"
        for row in result["rows"] if row["position"] in (0,4)]
    rows=[{"pair":pair["pair"],"proposal_count":len(pair["proposal_results"]),**pair["verdict"]} for pair in native["pairs"]]
    authority=baseline.compare_rows(original,rows)
    if authority.get("same_decisions_and_support") is not True:
        raise RuntimeError("Contemporary native control diverged from original fixture gates/support")
    for run in runs:
        for pair in run["pairs"]:
            expected=lookup[pair["position"]]
            pair["pair_verdict_same"]=all(pair["verdict"][key]==expected["verdict"][key]
                for key in ("accepted","ambiguous","verified_proposals"))
            for proposal in pair["proposal_results"]:
                key=(pair["position"],proposal["proposal_index"])
                proposal["quality"]=(None if run["mode"]=="native_cpu" else
                    agreement(values[("native_cpu",)+key],values[(run["mode"],)+key]))
            if (not pair["pair_verdict_same"] or any(p["quality"] and not p["quality"]["passed"] for p in pair["proposal_results"])):
                raise RuntimeError("Original proposal pose/information/witness/gate evidence changed")
        run["quality_complete"]=True
    return authority


def run(args):
    if not args.run_allocated:raise ValueError("Require an explicitly allocated exclusive hardware slot")
    if args.output.exists():raise ValueError("Require a fresh output path; preserve prior results")
    if os.environ.get("OMP_NUM_THREADS","8")!="8":raise ValueError("Require original OMP8 policy")
    os.environ["OMP_NUM_THREADS"]="8"
    os.environ["KINECT_CUDA_REGISTRATION"]="cpu"
    from scripts.profile_session import source_hash
    from scripts.research import benchmark_parallel_fragments as baseline
    if source_hash()!=FROZEN:raise RuntimeError("Production source9331 required before numerical imports")
    from scripts.research.archive.benchmark_uniform_grid_nn import load_fixture_scope,restore_matches
    # Reject a different pickle before deserializing the private producer's
    # current raw-derived fixture. Unknown external pickle input is unsupported.
    declared=json.loads(args.combined_bridge.read_text(encoding="utf-8"))["fixture_binding"]
    if (file_hash(args.fixture)!=declared["fixture_sha256"] or file_hash(args.reference)!=declared["reference_sha256"]):
        raise RuntimeError("Private fixture/reference differs before materialization")
    fixture,reference,raw,tasks,fixture_binding=load_fixture_scope(args,baseline.numerical_source_hash())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={"kind":KIND,"status":"running","policy":POLICY,"source_sha256":FROZEN,
        "fixture_binding":fixture_binding,"performance_attribution_valid":True,"new_whole_finish_authority":False,
        "cpu_query_shadows":False,"cpu_result_shadows":False,"gpu_equation_shadows":False,
        "order":[["native_cpu","device_resident","combined_sync"],["combined_sync","native_cpu","device_resident"],
            ["device_resident","combined_sync","native_cpu"]],"real_runs":[],"setup":{},
        "timer_scope":"Complete original all-nine bridge component including original gates and in-loop bookkeeping. Imports/proof preflight/fresh-cloud setup excluded and separately measured; cold target index builds remain in first registration. Timers nested, not additive.",
        "retrieval_scope":"Original Device counters apply to all Device queries; combined counters apply to every provisional query, with inherited retrieval counters restricted to flagged CPU-resolution reruns."}
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    save()
    import cupy as cp
    import cv2
    import numpy as np
    import open3d as o3d
    from scanner_server import fragments as module
    from scripts.research.archive.benchmark_selective_fragment_threads import fresh_fragments
    from scripts.research.archive.benchmark_optix_nn import evidence_agreement,canonical_evidence,pair_verdict,gpu_memory_snapshot
    from scripts.benchmarks.profile_fragment_verification import array_digest,cloud_digest
    from scripts.process_metrics import finish_cuda_worker,gpu_info,peak_rss_bytes
    o3d.utility.set_max_threads(20);cv2.setNumThreads(20)
    template=json.loads(args.combined_bridge.read_text(encoding="utf-8"))
    config=dict(template["resident_configuration"]);cache=dict(template["cache_policy"])
    if str(args.solver_dll.resolve())!=str(Path(template["solver_library_path"]).resolve()):
        raise ValueError("Require the exact independently proved original Eigen library")
    artifacts={name:file_hash(ROOT/name) for name in ARTIFACTS}
    original_match=module._match
    solvers={};packed={};failure=None
    paths=[args.combined_synthetic,args.combined_bridge,args.device_synthetic,args.device_bridge,args.fixture,args.reference,raw]
    report["input_files"]={str(path.resolve()):file_hash(path) for path in paths}
    report["artifacts_sha256"]=artifacts
    try:
        started=time.perf_counter()
        combined_binding=rebuild_runtime(args.combined_bridge,cp,np,o3d,cv2,config,cache)
        device_binding=rebuild_runtime(args.device_bridge,cp,np,o3d,cv2,config,cache)
        authority=validate_timing_authority(args.combined_synthetic,args.combined_bridge,args.device_synthetic,args.device_bridge,
            combined_binding,device_binding,fixture_binding,artifacts)
        report.update(preflight_s=time.perf_counter()-started,combined_runtime_binding=combined_binding,
            device_runtime_binding=device_binding,gpu=gpu_info(),component_source_sha256=baseline.numerical_source_hash(),
            authority={"combined_synthetic_sha256":authority.combined_authority.synthetic_report_sha256,
                "combined_bridge_sha256":authority.combined_authority.bridge_report_sha256,
                "device_synthetic_sha256":authority.device_authority.synthetic_report_sha256,
                "device_bridge_sha256":authority.device_authority.bridge_report_sha256,
                "target_membership_count":len(authority.combined_authority.target_digests),
                "new_timing_source_sha256":dict(authority.timing_artifact_sha256)})
        for mode in ("device_resident","combined_sync"):
            started=time.perf_counter()
            cls=bridge_class(args.solver_dll,config["max_scratch_bytes"],config["max_query_bytes"],authority,mode=="combined_sync")
            solvers[mode]=cls(max_clouds=cache["max_clouds"],max_cache_bytes=cache["retained_gpu_bytes"],
                audit_nearest=False,audit_misses=False,miss_policy="direct-miss-research-v1",proof_authority=authority.device_authority)
            report["setup"][mode]={"wall_s":time.perf_counter()-started,"solve_metadata":dict(solvers[mode].solve.metadata),
                "configuration":{"device":str(solvers[mode].device),"max_points":solvers[mode].resident.max_points,
                    "max_scratch_bytes":solvers[mode].resident.max_scratch_bytes,"max_query_bytes":solvers[mode].max_query_bytes,
                    "gpu_timing":solvers[mode].resident.gpu_timing}}
            if report["setup"][mode]["configuration"]!=config:raise RuntimeError("Actual solver configuration differs")
        started=time.perf_counter()
        packed={mode:fresh_fragments(fixture,baseline,module) for mode in ("native_cpu","device_resident","combined_sync")}
        report["fresh_cloud_setup_s"]=time.perf_counter()-started
        fixture_before=array_digest(fixture);cloud_before={mode:cloud_digest(value) for mode,value in packed.items()}
        save()
        for repeat,order in enumerate(report["order"]):
            values={};round_runs=[]
            for mode in order:
                cloud=packed[mode];restore_matches(fixture,cloud,module)
                solver=solvers.get(mode)
                module._match=original_match if solver is None else solver.match
                before=dict(solver.statistics) if solver else None
                resident_before=dict(solver.resident.statistics) if solver else None
                combined_before=dict(solver.resident.combined_statistics) if mode=="combined_sync" else None
                row={"mode":mode,"repeat":repeat,"pairs":[],"complete":False,"quality_complete":False}
                report["real_runs"].append(row);round_runs.append(row)
                started=time.perf_counter()
                for task in tasks:
                    a,b=task["pair"];found=[];proposals=[];pair_started=time.perf_counter()
                    for index,pose in enumerate(task["proposals"]):
                        proposal_started=time.perf_counter()
                        value=module._verify_bridge(cloud[a],cloud[b],pose,fixture["camera"])
                        found.append(value);values[(mode,task["position"],index)]=value
                        proposals.append({"proposal_index":index,"input_sha256":hashlib.sha256(pose.tobytes()).hexdigest(),
                            "elapsed_s":time.perf_counter()-proposal_started,"evidence":canonical_evidence(value),"quality":None})
                        print(f"{mode} round{repeat} pair{a}/{b} proposal{index}: {proposals[-1]['elapsed_s']:.3f}s",flush=True)
                    row["pairs"].append({"position":task["position"],"pair":task["pair"],"proposal_results":proposals,
                        "verdict":pair_verdict(found),"elapsed_s":time.perf_counter()-pair_started})
                    save()
                row.update(complete=True,elapsed_s=time.perf_counter()-started)
                if solver:
                    row.update(statistics_delta=delta(solver.statistics,before),
                        resident_statistics_delta=delta(solver.resident.statistics,resident_before),device_resident=solver.resident.report(),
                        target_membership_sha256=sorted(solver.target_membership_sha256),gpu_memory_after=gpu_memory_snapshot(solver.device_id))
                    for key in ("gpu_transform_ms","gpu_equations_ms"):
                        row["resident_statistics_delta"].pop(key,None)
                    row["uncollected_resident_statistics"]=["gpu_transform_ms","gpu_equations_ms"]
                    stats=row["resident_statistics_delta"]
                    if (stats["calls"]!=236 or stats["pose_iterations"]!=10861 or stats["cpu_fallback_calls"]
                            or solver.failure is not None or row["statistics_delta"]["device_malformed_results"]):
                        raise RuntimeError("Original call/iteration/fault/fallback closure differs")
                    if mode=="combined_sync":
                        counts=delta(solver.resident.combined_statistics,combined_before)
                        row["combined_statistics_delta"]=counts
                        row["uncollected_resident_statistics"] += ["nn_wall_s"]
                        row["resident_statistics_delta"].pop("nn_wall_s",None)
                        if (counts["provisional_queries"]!=104123989 or counts["iteration_calls"]!=11569
                                or counts["primary_counter_term_syncs"]!=11569 or counts["primary_download_bytes"]!=3702080
                                or counts["common_calls"]+counts["flagged_calls"]!=11569 or counts["malformed_results"]
                                or counts["discarded_placeholder_calls"]!=counts["flagged_calls"]):
                            raise RuntimeError("Single-copy complete query/counter transport differs from its independent audit")
                    elif (row["statistics_delta"]["query_rows"]!=104123989 or row["statistics_delta"]["device_calls"]!=11569):
                        raise RuntimeError("Original resident complete query coverage differs")
                    if set(solver.target_membership_sha256)!=set(authority.device_authority.target_digests):
                        raise RuntimeError("Actual target membership differs from complete both proofs")
                save()
            report.setdefault("native_fixture_authority",[]).append(compare_round(round_runs,values,reference,baseline,evidence_agreement))
            save()
        report["fixture_arrays_unchanged"]=array_digest(fixture)==fixture_before
        report["cloud_arrays_unchanged"]={mode:cloud_digest(value)==cloud_before[mode] for mode,value in packed.items()}
        if not report["fixture_arrays_unchanged"] or not all(report["cloud_arrays_unchanged"].values()):
            raise RuntimeError("Original fixture/cloud arrays changed during component reproduction")
    except BaseException as error:
        failure=error
        report["failure"]={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
    finally:
        cleanup_errors=[]
        def cleanup(name,action):
            try:return action()
            except BaseException as error:
                cleanup_errors.append({"action":name,"type":type(error).__name__,"message":str(error)})
                return None
        cleanup("restore original match",lambda:setattr(module,"_match",original_match))
        report["original_hook_restored"]=module._match is original_match
        report["source_sha256_after"]=cleanup("production source",source_hash)
        report["component_source_sha256_after"]=cleanup("numerical component source",baseline.numerical_source_hash)
        report["artifacts_sha256_after"]=cleanup("timing artifacts",lambda:{name:file_hash(ROOT/name) for name in ARTIFACTS})
        report["input_files_after"]=cleanup("raw/fixture/reference/proof bytes",lambda:{name:file_hash(Path(name)) for name in report["input_files"]})
        report["combined_runtime_binding_after"]=cleanup("actual combined runtime",lambda:rebuild_runtime(args.combined_bridge,cp,np,o3d,cv2,config,cache))
        report["device_runtime_binding_after"]=cleanup("actual Device runtime",lambda:rebuild_runtime(args.device_bridge,cp,np,o3d,cv2,config,cache))
        if "authority" in locals():
            again=cleanup("complete proof authority revalidation",lambda:validate_timing_authority(args.combined_synthetic,args.combined_bridge,
                args.device_synthetic,args.device_bridge,report["combined_runtime_binding_after"],report["device_runtime_binding_after"],fixture_binding,artifacts))
            report["authority_unchanged"]=again==authority
        for mode,solver in solvers.items():
            report.setdefault("final_device_reports",{})[mode]=cleanup(mode+" report",solver.resident.report)
            cleanup(mode+" selected-stream owned cleanup",solver.close)
        report["gpu_after"]=cleanup("GPU identity",gpu_info)
        report["peak_process_rss_bytes"]=cleanup("peak RSS",peak_rss_bytes)
        report["cleanup_failures"]=cleanup_errors
        passed=(failure is None and not cleanup_errors and report["original_hook_restored"] and report.get("authority_unchanged") is True
            and report["source_sha256_after"]==FROZEN and report["component_source_sha256_after"]==report.get("component_source_sha256")
            and report["artifacts_sha256_after"]==artifacts and report["input_files_after"]==report["input_files"]
            and report["combined_runtime_binding_after"]==report.get("combined_runtime_binding")
            and report["device_runtime_binding_after"]==report.get("device_runtime_binding") and report["gpu_after"]==report.get("gpu")
            and len(report["real_runs"])==9 and all(row["complete"] and row["quality_complete"] for row in report["real_runs"]))
        report["status"]="passed" if passed else "failed"
        try:save()
        except BaseException as error:
            if failure is not None:
                failure.add_note(f"Final timing diagnostics write also failed: {error}")
                raise failure from error
            raise
    if not passed:raise RuntimeError("New component timing quality/provenance gates failed; diagnostics preserved") from failure
    print(json.dumps({"output":str(args.output),"status":report["status"],"runs":[
        {key:row[key] for key in ("mode","repeat","elapsed_s")} for row in report["real_runs"]]}),flush=True)
    finish_cuda_worker()


def preserve_failure(path,error):
    value={"kind":KIND,"status":"failed"};secondary=[]
    if path.exists():
        try:
            value=json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value,dict):raise ValueError("Partial report is not an object")
        except BaseException as read_error:
            secondary.append({"action":"partial read","message":str(read_error)})
            value={"kind":KIND,"status":"failed"}
            backup=path.with_name(path.name+".partial-"+str(time.time_ns()))
            try:backup.write_bytes(path.read_bytes());value["partial_report_preserved_at"]=str(backup)
            except BaseException as backup_error:secondary.append({"action":"partial preserve","message":str(backup_error)})
    value.update(status="failed",timing_driver_current_failure={"type":type(error).__name__,"message":str(error),
        "traceback":traceback.format_exc()},timing_driver_secondary_errors=secondary)
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    except BaseException as write_error:
        error.add_note(f"Failed timing diagnostic write also failed: {write_error}; {secondary}")
        raise error from write_error


def main():
    args=parser().parse_args()
    if args.output.exists():parser().error("Require fresh output; preserve all previous reports")
    try:run(args)
    except BaseException as error:
        if isinstance(error,SystemExit):raise
        preserve_failure(args.output,error)
        raise


if __name__=="__main__":main()
