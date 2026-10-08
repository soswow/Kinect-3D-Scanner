"""Focused fresh device classifier/transport/cache proof, separate from NN audits.

All numerical imports and hardware work are inside run_device_checks. Deliberate
faults use separately owned solvers, never the producer's successful audit
counters. Artificial raw minima test classification only; original host/device
NN cases and the complete original nine-proposal trajectory remain mandatory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json


REQUIRED = ("classifier_masks_exact", "packet_exact", "scatter_metrics_exact",
    "audit_before_correction", "finite_transport_hard_failure", "malformed_raw_hard_failure",
    "policy_mutation_rejected", "query_budget_rejected", "input_contract_rejected",
    "cache_xyz_exact", "cache_release_clears_xyz", "mutation_rebuilds_xyz",
    "budget_cpu_only_no_xyz", "no_full_query_download")


def _classifier_cases(solver):
    import numpy as np
    cp=solver.cp
    eps=np.finfo(np.float64).eps; tiny=np.finfo(np.float64).tiny
    radius=.12; r2=radius*radius
    patterns=[[0.,-1.,0.,np.inf,3.],[-1.,-1.,np.inf,np.inf,3.],[-3.,-1.,np.inf,np.inf,0.]]
    for shift in (-130,-65,-64,-63,0,63,64,65,130):
        a=r2*(1+shift*eps)
        patterns.append([0.,-1.,a,np.inf,3.])
        for separation in (0,32,63,64,65,130):
            b=a+separation*eps*max(abs(a),r2)
            patterns.append([0.,1.,a,b,3.])
    cases=0; packet_rows=0; maxima={"unsupported":0,"uncertain":0,"audited_hits":0,"audited_misses":0}
    for count in (127,128,129,257):
        raw=np.asarray([patterns[index%len(patterns)] for index in range(count)],np.float64)
        query=np.column_stack((np.arange(count,dtype=np.float64)*.001,np.zeros(count),np.zeros(count)))
        first=raw[:,0].astype(np.int32);a=raw[:,2];b=raw[:,3]
        finite=np.isfinite(a);unsupported=first==-3
        with np.errstate(invalid="ignore"):
            boundary=finite&(np.abs(a-r2)<=64*eps*np.maximum(np.maximum(np.abs(a),abs(r2)),tiny))
            tie=finite&np.isfinite(b)&(b-a<=64*eps*np.maximum(np.maximum(np.abs(a),np.abs(b)),abs(r2)))
        uncertain=boundary|tie;missing=~unsupported&(~finite|(a>=r2))
        for direct in (False,True):
            for audit_hit in (False,True):
                for audit_miss in (False,True):
                    fallback=unsupported|uncertain|((not direct)&missing)
                    found=first.copy();found[missing]=-1
                    hit=~fallback&(found>=0);miss=direct&missing&~uncertain
                    reason=(unsupported.astype(np.uint32)|uncertain.astype(np.uint32)*2|
                        ((not direct)&missing).astype(np.uint32)*4|
                        (hit&audit_hit).astype(np.uint32)*8|(miss&audit_miss).astype(np.uint32)*16)
                    expected=np.array([np.count_nonzero(reason),np.count_nonzero(hit),np.count_nonzero(miss),
                        np.count_nonzero(fallback),np.count_nonzero(unsupported),np.count_nonzero(uncertain),
                        int(raw[:,4].sum()),0,np.count_nonzero(hit&audit_hit),np.count_nonzero(miss&audit_miss)],np.uint64)
                    with cp.cuda.Device(solver.device_id),cp.cuda.Stream.null:
                        device_raw=cp.asarray(raw);device_query=cp.asarray(query)
                        ids=cp.empty(count,cp.int32);metrics=cp.empty(count,cp.float64)
                        reasons=cp.empty(count,cp.uint32);rows=cp.empty(count,cp.uint32);counters=cp.zeros(10,cp.uint64)
                        solver.classify(((count+127)//128,),(128,),(device_raw,np.uint32(count),np.uint32(3),
                            np.float64(r2),np.int32(direct),np.int32(audit_hit),np.int32(audit_miss),
                            ids,metrics,reasons,rows,counters))
                        observed=cp.asnumpy(counters)
                        if not np.array_equal(observed,expected) or not np.array_equal(cp.asnumpy(ids),found) or not np.array_equal(cp.asnumpy(reasons),reason):
                            raise RuntimeError("Resident device masks/counters differ from original CPU resolver formulas")
                        expected_metrics=np.where(found>=0,a,np.inf)
                        if not np.array_equal(cp.asnumpy(metrics).view(np.uint64),expected_metrics.view(np.uint64)):
                            raise RuntimeError("Device classifier changed exact original raw squared-distance bits")
                        m=int(expected[0]);flagged=cp.asnumpy(rows[:m])
                        if len(np.unique(flagged))!=m or not np.array_equal(np.sort(flagged),np.flatnonzero(reason)):
                            raise RuntimeError("Partial-block/warp compaction omitted, duplicated or invented a flagged row")
                        if m:
                            packet=cp.empty((m,8),cp.float64)
                            solver.pack(((m+127)//128,),(128,),(device_query,device_raw,ids,reasons,rows,np.uint32(m),packet))
                            host=cp.asnumpy(packet)
                            expected_packet=np.column_stack((flagged,query[flagged],found[flagged],reason[flagged],raw[flagged,2:4])).astype(np.float64)
                            if not np.array_equal(host.view(np.uint64),expected_packet.view(np.uint64)):
                                raise RuntimeError("Flagged transport changed original row/query/raw metric bits")
                            target=np.array([[0.,0.,0.],[.1,.02,.03],[.2,.04,.06]],np.float64)
                            corrected=(flagged%4).astype(np.int32);corrected[corrected==3]=-1
                            corrected_device=cp.asarray(corrected);target_device=cp.asarray(target)
                            solver.scatter(((m+127)//128,),(128,),(rows,corrected_device,np.uint32(m),device_query,target_device,ids,metrics))
                            expected_ids=found.copy();expected_ids[flagged]=corrected
                            expected_distances=expected_metrics.copy();expected_distances[flagged]=np.inf
                            selected=corrected>=0
                            delta=query[flagged[selected]]-target[corrected[selected]]
                            exact=(delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1])+delta[:,2]*delta[:,2]
                            expected_distances[flagged[selected]]=exact
                            if not np.array_equal(cp.asnumpy(ids),expected_ids) or not np.array_equal(cp.asnumpy(metrics).view(np.uint64),expected_distances.view(np.uint64)):
                                raise RuntimeError("Original-order CPU correction scatter changed ID or RN64 distance bits")
                            packet_rows+=m
                    cases+=1
                    for name,index in (("unsupported",4),("uncertain",5),("audited_hits",8),("audited_misses",9)):
                        maxima[name]=max(maxima[name],int(expected[index]))
    if cases!=32 or packet_rows<=0 or any(value<=0 for value in maxima.values()):
        raise RuntimeError("Classifier proof lacks positive supported/unsupported/tie/audit coverage")
    return {"classifier_masks_exact":True,"packet_exact":True,"scatter_metrics_exact":True,
        "classifier_cases":cases,"classifier_rows":sum((127,128,129,257))*8,
        "packet_rows":packet_rows,"positive_mask_coverage":maxima,
        "scope":"Artificial raw metrics verify masks/packing/scatter, not NN identity; actual NN audits remain separate"}


def run_device_checks(_main_solver):
    import numpy as np
    import open3d as o3d
    from scripts.research.cuda_device_flat_grid_registration import DeviceFlatGridICP
    owned=[];result={}
    def make(**kwargs):
        value=DeviceFlatGridICP(**kwargs);owned.append(value);return value
    def cloud(offset=0.):
        points=np.array([[.013,0.,0.],[.173,.02,0.],[.333,0.,.04]],np.float64)
        points[:,0]+=offset
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    try:
        solver=make(max_clouds=1,max_cache_bytes=256*1024,audit_nearest=False,audit_misses=False)
        result.update(_classifier_cases(solver))
        cp=solver.cp;target=cloud();points=np.asarray(target.points).copy();item=solver._dataset(target,.12)
        with cp.cuda.Device(solver.device_id),cp.cuda.Stream.null:
            queries=cp.asarray(points)
            ids,metrics=solver.nearest_device(queries,item,.12)
            result["no_full_query_download"]=(np.array_equal(cp.asnumpy(ids),np.arange(3,dtype=np.int32)) and
                np.array_equal(cp.asnumpy(metrics),np.zeros(3)) and solver.statistics["device_query_download_rows"]==0 and
                solver.statistics["device_query_download_bytes"]==0 and solver.statistics["device_counter_syncs"]==1)
            result["cache_xyz_exact"]=np.array_equal(cp.asnumpy(item["data"]).view(np.uint64),points.view(np.uint64))
            budget=solver.max_query_bytes;solver.max_query_bytes=1
            try:
                solver.nearest_device(queries,item,.12)
                raise AssertionError("Query scratch limit was ignored")
            except ValueError as error:
                result["query_budget_rejected"]="byte cap" in str(error)
            finally:solver.max_query_bytes=budget
            solver.miss_policy="direct-miss-research-v1"
            try:
                solver.nearest_device(queries,item,.12)
                raise AssertionError("Mutated unaudited direct-miss policy was authorized")
            except RuntimeError as error:
                result["policy_mutation_rejected"]="complete CPU hit/miss audits" in str(error)
            finally:solver.miss_policy="cpu-fallback"
            rejected=0
            for bad in (queries.astype(cp.float32),cp.empty((3,4),cp.float64),cp.empty((6,3),cp.float64)[::2]):
                try:solver.nearest_device(bad,item,.12)
                except ValueError:rejected+=1
            result["input_contract_rejected"]=rejected==3
            previous_data=item["data"]
            np.asarray(target.points)[0,0]+=.007
            rebuilt=solver._dataset(target,.12)
            result["mutation_rebuilds_xyz"]=(rebuilt is not item and item["data"] is None and not item["grids"] and
                rebuilt["data"] is not previous_data and np.array_equal(cp.asnumpy(rebuilt["data"]),np.asarray(target.points)))
            evicted=rebuilt;solver._dataset(cloud(.5),.12)
            solver.clear_cache()
            result["cache_release_clears_xyz"]=(evicted["data"] is None and not evicted["grids"] and evicted["gpu_bytes"]==0 and
                solver.cache_bytes==0 and not solver.cache and solver.statistics["cache_evictions"]>0)
        # A fresh fault solver prevents intentional errors from contaminating
        # the producer's positive direct-hit/miss audit counters.
        fault=make(audit_nearest=True,audit_misses=True,miss_policy="direct-miss-research-v1")
        fault_target=cloud();fault_item=fault._dataset(fault_target,.12)
        class NeverCPU:
            calls=0
            def search_hybrid_vector_3d(self,*args):
                self.calls+=1
                raise AssertionError("Invalid device transport reached CPU FLANN")
        never=NeverCPU();original_cpu=fault_item["cpu"]
        with cp.cuda.Device(fault.device_id),cp.cuda.Stream.null:
            query=cp.asarray(np.asarray(fault_target.points)[:1].copy())
            original_raw=fault._raw_device;original_scatter=fault.scatter
            scatters=[]
            def no_scatter(*args):
                scatters.append(True)
                raise AssertionError("Audit mismatch was corrected before failure")
            fault.scatter=no_scatter
            try:
                fault._raw_device=lambda *_:cp.asarray([[1.,0.,0.,.01,3.]],dtype=cp.float64)
                try:
                    fault.nearest_device(query,fault_item,.12)
                    raise AssertionError("Wrong audited device ID was hidden")
                except RuntimeError as error:
                    result["audit_before_correction"]=("audited original CPU IDs" in str(error) and not scatters and
                        fault.statistics["audit_index_mismatches"]==1 and fault.statistics["audit_false_misses"]==0)
                fault_item["cpu"]=never
                fault._raw_device=lambda *_:cp.asarray([[.5,-1.,0.,np.inf,3.]],dtype=cp.float64)
                try:
                    fault.nearest_device(query,fault_item,.12)
                    raise AssertionError("Malformed raw ID was recovered")
                except RuntimeError as error:
                    result["malformed_raw_hard_failure"]=("Malformed/inconsistent" in str(error) and not scatters and never.calls==0)
                fault._raw_device=original_raw
                try:
                    fault.nearest_device(cp.asarray([[np.nan,0.,0.]],dtype=cp.float64),fault_item,.12)
                    raise AssertionError("Nonfinite query transport reached CPU")
                except RuntimeError as error:
                    result["finite_transport_hard_failure"]=("Nonfinite flagged query/CPU transport" in str(error) and never.calls==0)
            finally:
                fault._raw_device=original_raw;fault.scatter=original_scatter;fault_item["cpu"]=original_cpu
        tiny=make(max_clouds=1,max_cache_bytes=1,audit_nearest=True,audit_misses=True,miss_policy="direct-miss-research-v1")
        tiny_target=cloud();tiny_item=tiny._dataset(tiny_target,.12)
        with cp.cuda.Device(tiny.device_id),cp.cuda.Stream.null:
            ids,metrics=tiny.nearest_device(cp.asarray(np.asarray(tiny_target.points).copy()),tiny_item,.12)
            result["budget_cpu_only_no_xyz"]=(tiny_item["cpu_only"] is True and tiny_item["data"] is None and
                tiny.cache_bytes==0 and tiny.statistics["cache_budget_bypasses"]>0 and
                np.array_equal(cp.asnumpy(ids),np.arange(3,dtype=np.int32)) and np.array_equal(cp.asnumpy(metrics),np.zeros(3)))
        result["required_checks"]=list(REQUIRED)
        result["passed"]=all(result.get(name) is True for name in REQUIRED)
        result["fault_statistics_scope"]="Intentional failed audit/malformed/nonfinite cases use separate owned solver; excluded from successful producer statistics"
        if not result["passed"]:raise RuntimeError("Focused device classifier/transport/ownership proof failed")
        return result
    except BaseException as error:
        error.add_note("Partial focused device proof: "+json.dumps(result,sort_keys=True,allow_nan=False))
        raise
    finally:
        primary=sys.exception();errors=[]
        for solver in reversed(owned):
            try:solver.close()
            except BaseException as error:errors.append(f"{type(error).__name__}: {error}")
        if errors:
            if primary is not None:primary.add_note(f"Focused device cleanup also failed: {errors}")
            else:raise RuntimeError(f"Focused device cleanup failed: {errors}")
