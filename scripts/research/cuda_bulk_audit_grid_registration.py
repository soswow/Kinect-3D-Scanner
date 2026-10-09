"""Separate exact CPU-shadow experiment; original GPU math stays inherited.

The complete old trajectory first uses dual scalar/bulk shadows. A subsequent
field experiment may use bulk shadows only with its distinct validated token.
Ambiguous/unsupported GPU rows always retain original scalar FLANN resolution.
No production selection, old proof editing, or unaudited timing is enabled here.
"""

import ast
import hashlib
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from scripts.research.cuda_device_flat_grid_registration import DeviceFlatGridICP
from scripts.research.bulk_legacy_nn_audit import (
    UnsupportedBulkDomain, array_binding, bulk_legacy_ids)

ROOT = Path(__file__).resolve().parents[2]
POLICY = "original-legacy-bulk-shadow-v1"


def unchanged_transport_scope():
    """Source-only check: the copied method differs only in CPU resolution."""
    def method(path, name):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == name)
        node = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_nearest_device_impl")
        # Normalize the one permitted block, ending immediately before the
        # inherited CPU-resolution timer and audit/scatter authority checks.
        start = next(i for i, value in enumerate(node.body)
                     if isinstance(value, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == "corrected"
                         for target in value.targets)) if name == "DeviceFlatGridICP" else next(
                         i for i, value in enumerate(node.body)
                         if isinstance(value, ast.Assign) and isinstance(value.value, ast.Call)
                         and isinstance(value.value.func, ast.Attribute)
                         and value.value.func.attr == "perf_counter"
                         and i + 1 < len(node.body) and isinstance(node.body[i + 1], ast.Assign)
                         and isinstance(node.body[i + 1].value, ast.Call)
                         and isinstance(node.body[i + 1].value.func, ast.Attribute)
                         and node.body[i + 1].value.func.attr == "_resolve_cpu_packet")
        stop = next(i for i, value in enumerate(node.body[start:], start)
                    if isinstance(value, ast.AugAssign) and isinstance(value.target, ast.Subscript)
                    and isinstance(value.target.slice, ast.Constant)
                    and value.target.slice.value == "device_cpu_resolution_s")
        node.body[start:stop] = [ast.Expr(value=ast.Constant(value="ORIGINAL_CPU_SHADOW_BLOCK"))]
        return ast.dump(node, include_attributes=False)
    old = ROOT / "scripts/research/cuda_device_flat_grid_registration.py"
    if method(old, "DeviceFlatGridICP") != method(Path(__file__), "BulkAuditDeviceFlatGridICP"):
        raise RuntimeError("Bulk adapter changed GPU transport/classification/audit/scatter outside its CPU-shadow block")
    return True


class BulkAuditDeviceFlatGridICP(DeviceFlatGridICP):
    def __init__(self, *args, audit_mode="dual", bulk_authority=None,
                 bulk_chunk_rows=65536, **kwargs):
        if audit_mode not in ("dual", "bulk-shadow"):
            raise ValueError("Unknown exact CPU-shadow experiment policy")
        if kwargs.get("audit_nearest") is not True or kwargs.get("audit_misses") is not True:
            raise ValueError("Bulk research is audit-only and requires complete CPU hit/miss shadows")
        if not isinstance(bulk_chunk_rows, int) or isinstance(bulk_chunk_rows, bool) or not 1 <= bulk_chunk_rows <= 65536:
            raise ValueError("Require bounded bulk query chunks")
        if audit_mode == "dual" and bulk_authority is not None:
            raise ValueError("The original dual proof must not consume previous authority")
        if audit_mode == "bulk-shadow":
            from scripts.research.validate_bulk_resident_audit import validate_authority_for_bulk_shadow
            from scripts.research.bulk_legacy_nn_audit import runtime_binding
            validate_authority_for_bulk_shadow(bulk_authority,
                device=args[0] if args else kwargs.get("device", "CUDA:0"), chunk_rows=bulk_chunk_rows,
                native_runtime=runtime_binding(np, o3d))
        unchanged_transport_scope()
        self.audit_mode, self.bulk_authority, self.bulk_chunk_rows = audit_mode, bulk_authority, bulk_chunk_rows
        self._bulk_configuration = (audit_mode, bulk_chunk_rows, bulk_authority)
        super().__init__(*args, **kwargs)
        self.bulk_records = []
        self.statistics.update({key: 0 for key in (
            "bulk_shadow_queries", "bulk_shadow_hits", "bulk_shadow_misses", "dual_scalar_bulk_queries",
            "dual_id_mismatches", "dual_metric_bit_mismatches", "bulk_domain_scalar_queries", "bulk_evaluate_calls")})
        self.statistics.update({key: 0. for key in ("bulk_all_in_s", "bulk_evaluate_s", "bulk_normalize_s")})

    def nearest_device(self, queries, item, radius):
        # A later settings mutation cannot turn a source-only dual constructor
        # into a bulk-only or unaudited path without its validated authority.
        mode, chunk_rows, authority = self._bulk_configuration
        if (self.audit_mode != mode or self.bulk_chunk_rows != chunk_rows
                or self.bulk_authority is not authority
                or self.audit_nearest is not True or self.audit_misses is not True):
            raise RuntimeError("Bulk audit mode, authority, chunk size or complete shadow flags changed after validation")
        return super().nearest_device(queries, item, radius)

    def _resolve_cpu_packet(self, host, reason, cpu_mask, item, radius):
        m = len(host)
        audit_mask = (reason & 24) != 0
        audit_rows = np.flatnonzero(audit_mask)
        corrected = np.empty(m, np.int32)
        scalar_squared = np.full(m, np.inf, np.float64)
        # Dual mode queries every original row exactly as the inherited loop;
        # bulk-only mode retains scalar handling for reason&7 without alteration.
        for index in range(m):
            if self.audit_mode == "bulk-shadow" and not cpu_mask[index]:
                continue
            query_started = time.perf_counter()
            count, found, distances = item["cpu"].search_hybrid_vector_3d(host[index, 1:4], radius, 1)
            elapsed = time.perf_counter() - query_started
            self.statistics["cpu_fallback_s" if cpu_mask[index] else "cpu_audit_s"] += elapsed
            if count not in (0, 1) or len(found) != count or len(distances) != count:
                raise RuntimeError("Malformed original scalar result in dual CPU shadow")
            corrected[index] = found[0] if count else -1
            if count:
                scalar_squared[index] = distances[0]
        if len(audit_rows):
            queries = np.ascontiguousarray(host[audit_rows, 1:4])
            target_before = array_binding(np, np.asarray(item["target"].points))
            if target_before["sha256"] != item["digest"].hex():
                raise RuntimeError("Cached original scalar tree disagrees with the bulk target vector")
            try:
                bulk = bulk_legacy_ids(np, o3d, queries, item["target"], radius,
                    expected_target_digest=item["digest"], chunk_rows=self.bulk_chunk_rows)
            except UnsupportedBulkDomain:
                # No rows are omitted. Unsupported bulk domains preserve the
                # entire original scalar audit, separately declared in proof.
                self.statistics["bulk_domain_scalar_queries"] += len(audit_rows)
                if self.audit_mode == "bulk-shadow":
                    for index in audit_rows:
                        query_started = time.perf_counter()
                        count, found, distances = item["cpu"].search_hybrid_vector_3d(host[index, 1:4], radius, 1)
                        self.statistics["cpu_audit_s"] += time.perf_counter() - query_started
                        if count not in (0, 1) or len(found) != count or len(distances) != count:
                            raise RuntimeError("Malformed original scalar domain-fallback result")
                        corrected[index] = found[0] if count else -1
                self.bulk_records.append({"complete": True, "bulk_domain_supported": False,
                    "audit_query_descriptor": array_binding(np, queries), "target_digest": item["digest"].hex(),
                    "radius_m": float(radius), "scalar_domain_queries": len(audit_rows)})
            else:
                hits = int(np.count_nonzero((reason[audit_rows] & 8) != 0))
                misses = int(np.count_nonzero((reason[audit_rows] & 16) != 0))
                self.statistics["bulk_shadow_queries"] += len(audit_rows)
                self.statistics["bulk_shadow_hits"] += hits
                self.statistics["bulk_shadow_misses"] += misses
                for key, value in (("bulk_evaluate_calls", bulk.statistics["evaluate_calls"]),
                        ("bulk_all_in_s", bulk.statistics["all_in_wall_s"]),
                        ("bulk_evaluate_s", bulk.statistics["evaluate_wall_s"]),
                        ("bulk_normalize_s", bulk.statistics["normalize_metric_s"])):
                    self.statistics[key] += value
                record = {"complete": False, "bulk_domain_supported": True,
                    "audit_query_descriptor": array_binding(np, queries), "target_digest": item["digest"].hex(),
                    "radius_m": float(radius), "audit_hit_rows": hits, "audit_miss_rows": misses,
                    "bulk_ids_sha256": array_binding(np, bulk.ids)["sha256"],
                    "bulk_squared_sha256": array_binding(np, bulk.diagnostic_squared)["sha256"],
                    "all_in_wall_s": bulk.statistics["all_in_wall_s"],
                    "evaluate_calls": bulk.statistics["evaluate_calls"]}
                self.bulk_records.append(record)
                if self.audit_mode == "dual":
                    expected = corrected[audit_rows]
                    expected_squared = scalar_squared[audit_rows]
                    id_errors = int(np.count_nonzero(expected != bulk.ids))
                    metric_errors = int(np.count_nonzero(expected_squared.view(np.uint64)
                                                        != bulk.diagnostic_squared.view(np.uint64)))
                    self.statistics["dual_scalar_bulk_queries"] += len(audit_rows)
                    self.statistics["dual_id_mismatches"] += id_errors
                    self.statistics["dual_metric_bit_mismatches"] += metric_errors
                    record.update(scalar_ids_sha256=array_binding(np, expected)["sha256"],
                        scalar_squared_sha256=array_binding(np, expected_squared)["sha256"],
                        id_mismatches=id_errors, metric_bit_mismatches=metric_errors)
                    if id_errors or metric_errors:
                        raise RuntimeError(f"Bulk CPU shadow changed {id_errors} original IDs or {metric_errors} diagnostic metric bits")
                else:
                    # Native bulk IDs alone establish membership. Diagnostics
                    # are never used to reclassify a correspondence or radius.
                    corrected[audit_rows] = bulk.ids
                target_after = array_binding(np, np.asarray(item["target"].points))
                record["target_unchanged"] = target_after == target_before
                if not record["target_unchanged"]:
                    raise RuntimeError("Original target changed before correction/scatter")
                record["complete"] = True
        observed = host[:, 4].astype(np.int32)
        mismatches = int(np.count_nonzero(audit_mask & (observed != corrected)))
        false_misses = int(np.count_nonzero(audit_mask & (observed < 0) & (corrected >= 0)))
        return corrected, mismatches, false_misses

    # Copied inherited method follows. Its original GPU/raw/classifier/packet/
    # counter/audit/scatter AST is checked by unchanged_transport_scope().

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
        started=time.perf_counter()
        corrected,mismatches,false_misses=self._resolve_cpu_packet(host,reason,cpu_mask,item,radius)
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
