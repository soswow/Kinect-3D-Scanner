"""Separate true-device flat grid adapter; no production imports or selection.

Original raw flat minima stay resident. Only rows requiring the original CPU
resolver, or explicitly requested CPU shadow audits, leave the device. A small
counter transfer synchronizes classification before any CPU work. Device faults
and malformed output propagate; they are never retried as numerical fallback.
Coordinates must be finite for CPU FLANN resolution. Nonfinite device values
are rejected from the flagged packet before CPU search, without full-query copy.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hashlib
import math
import time

import numpy as np
from scripts.research.archive.cuda_flat_grid_registration import FlatUniformGridICP, DOMAIN as FLAT_DOMAIN
from scripts.research.validate_device_flat_grid_proof import DeviceFlatGridProofAuthority

CLASSIFIER_SOURCE=Path(__file__).with_name("research_device_flat_grid_nn.cu")
DOMAIN=dict(FLAT_DOMAIN,version="resident-device-flat-dyadic27-original-double-v1",
    device_resolution="original64eps masks on raw original-double minima; original-radius CPU resolves only flagged rows; audit rows copied explicitly; no full-query roundtrip in unaudited supported path",
    target_layout="original-order unsorted float64 XYZ retained with sorted grids under unified byte/cache eviction accounting",
    metrics="resident direct squared distance from original RN flat minima; corrected IDs use original-order target and explicit RN distance",
    temporary_memory="at most136 array bytes per query plus80B counters including CPU-only provided-metric scatter; maximum1million queries and configured136MiB query-buffer cap; cache cap/pools separately observed")


class DeviceFlatGridICP(FlatUniformGridICP):
    def __init__(self,device="CUDA:0",max_clouds=64,max_cache_bytes=256*1024**2,
                 audit_nearest=False,audit_misses=False,miss_policy="cpu-fallback",proof_authority=None,
                 max_query_bytes=136*1024**2):
        if max_query_bytes<1:
            raise ValueError("Require positive separately bounded device query scratch")
        if proof_authority is not None:
            if not isinstance(proof_authority,DeviceFlatGridProofAuthority):
                raise ValueError("Resident device path requires its own synthetic/trajectory proof authority")
            runtime=proof_authority.runtime_binding
            config=runtime["resident_configuration"];cache=runtime["cache_policy"]
            if (config["device"]!=str(device) or config["max_query_bytes"]!=max_query_bytes
                    or cache["max_clouds"]!=max_clouds or cache["retained_gpu_bytes"]!=max_cache_bytes):
                raise ValueError("Device/query/cache configuration differs from audited resident trajectory")
            expected=dict(proof_authority.artifact_sha256)
            for path in (Path(__file__),CLASSIFIER_SOURCE,Path(__file__).with_name("validate_device_flat_grid_proof.py"),
                         ROOT/"scripts/tool_paths.py",ROOT/"scripts/tool-catalog.json"):
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected.get(str(path.relative_to(ROOT))):
                    raise ValueError("Device adapter/classifier/validator differs from independently audited proof")
        super().__init__(device,max_clouds,max_cache_bytes,audit_nearest,audit_misses,miss_policy,proof_authority)
        self.max_query_bytes=max_query_bytes
        for key in ("resident_xyz_uploads","resident_xyz_upload_bytes","device_calls","device_counter_syncs",
                    "device_flagged_rows","device_query_download_rows","device_query_download_bytes",
                    "device_correction_upload_bytes","peak_device_query_bytes","device_malformed_results"):
            self.statistics[key]=0
        for key in ("resident_xyz_enqueue_s","device_total_wall_s","device_raw_classify_counter_s",
                    "device_flagged_copy_s","device_cpu_resolution_s","device_correction_enqueue_s"):
            self.statistics[key]=0.
        with self.cp.cuda.Device(self.device_id),self.cp.cuda.Stream.null:
            text=CLASSIFIER_SOURCE.read_text(encoding="utf-8")
            self.classify=self.cp.RawKernel(text,"classify_flat_results",options=("--std=c++11","--fmad=false"))
            self.pack=self.cp.RawKernel(text,"pack_flat_cpu_rows",options=("--std=c++11","--fmad=false"))
            self.scatter=self.cp.RawKernel(text,"scatter_flat_cpu_results",options=("--std=c++11","--fmad=false"))
            self.scatter_metrics=self.cp.RawKernel(text,"scatter_flat_provided_metrics",options=("--std=c++11","--fmad=false"))
            for kernel in (self.classify,self.pack,self.scatter,self.scatter_metrics):kernel.compile()

    def _release(self,item):
        # Parent synchronizes before subtracting ALL gpu_bytes and clearing grids.
        # Resident XYZ is included in gpu_bytes and released after the same sync.
        super()._release(item)
        item["data"]=None

    def _dataset_impl(self,target,radius):
        item=super()._dataset_impl(target,radius)
        item.setdefault("data",None)
        if item["data"] is None and not item["cpu_only"] and any(grid is not None for grid in item["grids"].values()):
            points=np.asarray(target.points)
            needed=points.nbytes
            while self.cache_bytes+needed>self.max_cache_bytes and len(self.cache)>1:
                _,old=self.cache.popitem(last=False)
                self._release(old)
                self.statistics["cache_evictions"]+=1
            if item["gpu_bytes"]+needed>self.max_cache_bytes:
                self._release(item)
                item["cpu_only"]=True
                self.statistics["cache_budget_bypasses"]+=1
            else:
                started=time.perf_counter()
                with self.cp.cuda.Device(self.device_id),self.cp.cuda.Stream.null:
                    item["data"]=self.cp.asarray(np.ascontiguousarray(points))
                item["gpu_bytes"]+=needed
                self.cache_bytes+=needed
                self.statistics["resident_xyz_uploads"]+=1
                self.statistics["resident_xyz_upload_bytes"]+=needed
                self.statistics["resident_xyz_enqueue_s"]+=time.perf_counter()-started
                self.statistics["peak_retained_gpu_bytes"]=max(self.statistics["peak_retained_gpu_bytes"],self.cache_bytes)
        return item

    def nearest_device(self,queries,item,radius):
        cp=self.cp
        if self.miss_policy=="direct-miss-research-v1" and not (self.audit_nearest and self.audit_misses) and self.proof_authority is None:
            raise RuntimeError("Device direct misses require complete CPU hit/miss audits or independently validated trajectory authority")
        if (not isinstance(queries,cp.ndarray) or queries.dtype!=cp.float64 or queries.ndim!=2 or queries.shape[1]!=3
                or not queries.flags.c_contiguous or queries.device.id!=self.device_id or len(queries)>1000000):
            raise ValueError("Device flat queries must be contiguous original float64(N,3) on selected device within count bound")
        if not math.isfinite(radius) or radius<=0 or not math.isfinite(radius*radius):
            raise ValueError("Require a finite positive original radius and squared radius")
        # Worst case includes raw40N, IDs4N, metrics8N, flags4N, rows4N,
        # packed CPU/audit64N, corrected IDs4N and CPU-only provided metrics8N
        # =136N, plus80B counters. No advanced-index temporary allocation.
        needed=len(queries)*136+80
        if needed>self.max_query_bytes:
            raise ValueError("Device nearest scratch exceeds its explicit byte cap")
        self.statistics["peak_device_query_bytes"]=max(self.statistics["peak_device_query_bytes"],needed)
        started=time.perf_counter()
        try:
            with cp.cuda.Device(self.device_id),cp.cuda.Stream.null:
                return self._nearest_device_impl(queries,item,radius)
        finally:
            self.statistics["device_calls"]+=1
            self.statistics["device_total_wall_s"]+=time.perf_counter()-started

    def _nearest_device_impl(self,queries,item,radius):
        cp=self.cp
        n=len(queries)
        ids=cp.empty(n,cp.int32);squared=cp.empty(n,cp.float64)
        if not n:return ids,squared
        started=time.perf_counter()
        raw=self._raw_device(queries,item,radius)
        reasons=cp.empty(n,cp.uint32);flagged=cp.empty(n,cp.uint32);counters=cp.zeros(10,cp.uint64)
        self.classify(((n+127)//128,),(128,),(raw,np.uint32(n),np.uint32(len(item["target"].points)),
            np.float64(radius*radius),np.int32(self.miss_policy=="direct-miss-research-v1"),np.int32(self.audit_nearest),
            np.int32(self.audit_misses),ids,squared,reasons,flagged,counters))
        values=cp.asnumpy(counters)
        self.statistics["device_counter_syncs"]+=1
        self.statistics["device_raw_classify_counter_s"]+=time.perf_counter()-started
        m,hits,misses,fallback,unsupported,uncertain,visits,invalid,audited_hits,audited_misses=map(int,values)
        if (invalid or m>n or m!=fallback+audited_hits+audited_misses or hits+misses+fallback!=n
                or audited_hits>hits or audited_misses>misses or unsupported>fallback or uncertain>fallback):
            self.statistics["device_malformed_results"]+=invalid or 1
            raise RuntimeError("Malformed/inconsistent device NN classification; CPU recovery is prohibited")
        for key,value in (("query_rows",n),("direct_gpu_hits",hits),("declared_gpu_misses",misses),
                ("unsupported_queries",unsupported),("uncertain_queries",uncertain),("candidate_visits",visits),
                ("exact_cpu_queries",fallback),("audited_hits",audited_hits),("audited_misses",audited_misses)):
            self.statistics[key]+=value
        self.statistics["device_flagged_rows"]+=m
        if not m:return ids,squared
        started=time.perf_counter()
        packet=cp.empty((m,8),cp.float64)
        self.pack(((m+127)//128,),(128,),(queries,raw,ids,reasons,flagged,np.uint32(m),packet))
        host=cp.asnumpy(packet)
        self.statistics["device_flagged_copy_s"]+=time.perf_counter()-started
        self.statistics["device_query_download_rows"]+=m
        self.statistics["device_query_download_bytes"]+=host.nbytes
        if not np.isfinite(host[:,:6]).all():
            raise RuntimeError("Nonfinite flagged query/CPU transport; no FLANN call or unsafe recovery")
        rows=host[:,0].astype(np.uint32);reason=host[:,5].astype(np.uint32)
        if (np.any(host[:,0]!=rows) or np.any(rows>=n)
                or len(np.unique(rows))!=m or np.any(host[:,5]!=reason) or np.any((reason==0)|(reason>31))):
            raise RuntimeError("Invalid flagged CPU transport; do not mask numerical/device failure")
        cpu_mask=(reason&7)!=0
        audit_hit_mask=(reason&8)!=0
        audit_miss_mask=(reason&16)!=0
        if (np.count_nonzero(cpu_mask)!=fallback or np.count_nonzero(audit_hit_mask)!=audited_hits
                or np.count_nonzero(audit_miss_mask)!=audited_misses
                or np.count_nonzero(reason&1)!=unsupported or np.count_nonzero(reason&2)!=uncertain
                or np.any(cpu_mask&(audit_hit_mask|audit_miss_mask)) or np.any(audit_hit_mask&audit_miss_mask)
                or np.any(host[:,4]!=host[:,4].astype(np.int32))
                or np.any((host[:,4]<-3)|(host[:,4]==-2)|(host[:,4]>=len(item["target"].points)))):
            raise RuntimeError("Flagged-row masks/IDs disagree with device counters")
        corrected=np.empty(m,np.int32)
        mismatches=false_misses=0
        started=time.perf_counter()
        for index in range(m):
            query_started=time.perf_counter()
            count,found,_=item["cpu"].search_hybrid_vector_3d(host[index,1:4],radius,1)
            query_elapsed=time.perf_counter()-query_started
            self.statistics["cpu_fallback_s" if cpu_mask[index] else "cpu_audit_s"]+=query_elapsed
            expected=found[0] if count else -1
            observed=int(host[index,4])
            if reason[index]&24:
                mismatches+=int(observed!=expected)
                false_misses+=int(observed<0 and expected>=0)
            corrected[index]=expected
        self.statistics["device_cpu_resolution_s"]+=time.perf_counter()-started
        self.statistics["audit_index_mismatches"]+=mismatches
        self.statistics["audit_false_misses"]+=false_misses
        if mismatches:
            raise RuntimeError(f"Device flat trajectory changed {mismatches} audited original CPU IDs, including {false_misses} false misses")
        started=time.perf_counter()
        if item["data"] is not None:
            resident_corrected=cp.asarray(corrected)
            self.scatter(((m+127)//128,),(128,),(flagged,resident_corrected,np.uint32(m),queries,item["data"],ids,squared))
            self.statistics["device_correction_upload_bytes"]+=corrected.nbytes
        else:
            # Explicit unsupported/cache CPU-only path; ResidentICP rejects
            # missing resident data and redoes its full original CPU call.
            distances=np.full(m,np.inf)
            selected=np.flatnonzero(corrected>=0)
            delta=host[selected,1:4]-np.asarray(item["target"].points)[corrected[selected]]
            distances[selected]=np.sum(delta*delta,axis=1)
            resident_corrected=cp.asarray(corrected);resident_distances=cp.asarray(distances)
            self.scatter_metrics(((m+127)//128,),(128,),(flagged,resident_corrected,resident_distances,np.uint32(m),ids,squared))
            self.statistics["device_correction_upload_bytes"]+=corrected.nbytes+distances.nbytes
        self.statistics["device_correction_enqueue_s"]+=time.perf_counter()-started
        return ids,squared
