"""Proved single-copy component timing; original numerical kernels stay intact.

Only observational full NN, equation-bit and CPU-result shadows are omitted.
Original ties/boundaries/unsupported rows still resolve on CPU before original
equations, and no malformed/failed complete call can silently recover.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from scripts.research import research_combined_sync_icp as audited
from scripts.research.research_device_grid_resident_icp import DeviceGridResidentICP
from scripts.research.validate_combined_sync_timing import validate_activation, POLICY


def timed_iteration():
    """Reuse original audited control/math statements; remove only observations.

    The provisional enqueue/copy/counter prefix is recovered byte-for-AST-byte.
    Flagged original CPU handling and common original normal normalization are
    retained verbatim, with explicit original parent delegation for super().
    No new shader, metric, filtered ID, solve or convergence expression exists.
    """
    source = textwrap.dedent(inspect.getsource(audited.CombinedSyncResidentICP._iteration_data))
    original = ast.parse(source).body[0]
    def dump(value):
        if isinstance(value,list):
            value=ast.Module(body=value,type_ignores=[])
        return ast.dump(value,include_attributes=False)
    def expression(value): return ast.unparse(value)
    prefix_end = next(i+1 for i,node in enumerate(original.body) if isinstance(node,ast.AugAssign)
        and expression(node.target) == expression(ast.parse('stats["provisional_candidate_visits"]',mode="eval").body))
    nearest_index = next(i for i,node in enumerate(original.body) if isinstance(node,ast.Assign)
        and isinstance(node.targets[0],ast.Tuple) and [n.id for n in node.targets[0].elts] == ["nearest","distances"])
    nearest = copy.deepcopy(original.body[nearest_index])
    validate_ids = next(copy.deepcopy(node) for node in original.body if isinstance(node,ast.If)
        and "nearest.dtype" in ast.unparse(node.test))
    flagged = next(copy.deepcopy(node) for node in original.body if isinstance(node,ast.If)
        and dump(node.test) == dump(ast.parse('counters["flagged"]',mode="eval").body))
    flagged.body.insert(0,validate_ids)
    flagged.body.insert(0,nearest)
    # The original class's parent is the same unchanged Device wrapper. Making
    # delegation explicit avoids manufacturing an implicit __class__ closure.
    call = flagged.body[-1].value
    if not (isinstance(flagged.body[-1],ast.Return) and isinstance(call,ast.Call)
            and isinstance(call.func,ast.Attribute) and call.func.attr == "_equations"
            and ast.unparse(call.func.value) == "super()"):
        raise RuntimeError("Original flagged CPU-before-equation seam changed")
    call.func.value = ast.Name(id="DeviceGridResidentICP",ctx=ast.Load())
    call.args.insert(0,ast.Name(id="self",ctx=ast.Load()))
    common_begin = next(i for i,node in enumerate(original.body) if isinstance(node,ast.AugAssign)
        and expression(node.target) == expression(ast.parse('stats["common_calls"]',mode="eval").body))
    common_end = next(i+1 for i,node in enumerate(original.body) if i >= common_begin and isinstance(node,ast.AugAssign)
        and expression(node.target) == expression(ast.parse('stats["primary_normalize_s"]',mode="eval").body))
    common = copy.deepcopy(original.body[common_begin:common_end])
    tree = ast.Module(body=[copy.deepcopy(original)],type_ignores=[])
    tree.body[0].body = copy.deepcopy(original.body[:prefix_end])+[flagged]+common+[
        ast.Return(value=ast.Name(id="result",ctx=ast.Load()))]
    ast.fix_missing_locations(tree)
    if dump(tree.body[0].body[:prefix_end]) != dump(original.body[:prefix_end]):
        raise RuntimeError("Timing changed the original provisional source/counter/math prefix")
    if dump(common) != dump(original.body[common_begin:common_end]):
        raise RuntimeError("Timing changed common original normal normalization")
    scope = dict(audited.__dict__,DeviceGridResidentICP=DeviceGridResidentICP)
    exec(compile(tree,__file__,"exec"),scope)
    contract = {"original_iteration_ast_sha256":hashlib.sha256(dump(original).encode()).hexdigest(),
        "timed_iteration_ast_sha256":hashlib.sha256(dump(tree.body[0]).encode()).hexdigest(),
        "original_provisional_prefix_unchanged":True,"original_common_normalization_unchanged":True,
        "flagged_original_cpu_before_equations":True,
        "omitted_observations":["full original CPU nearest shadows on common rows",
            "common original equation replay and ID/distance/filter/term bit copies",
            "full original CPU ICP result shadows supplied only by audited producer"],
        "new_math":False,"graph_capture":False}
    return scope["_iteration_data"],contract


_TIMED_ITERATION, ITERATION_CONTRACT = timed_iteration()


class CombinedSyncTimingFailure(BaseException):
    """Timing failures cannot become an ordinary retained/rejected bridge."""


class CombinedSyncTimedResidentICP(audited.CombinedSyncResidentICP):
    _iteration_data = _TIMED_ITERATION

    def __init__(self,retrieval,solve,*,authority,**kwargs):
        config = {"device":f"CUDA:{retrieval.device_id}","max_points":kwargs.get("max_points",1000000),
            "max_scratch_bytes":kwargs.get("max_scratch_bytes",256*1024**2),
            "max_query_bytes":retrieval.max_query_bytes,"gpu_timing":kwargs.get("gpu_timing",True)}
        validate_activation(authority,retrieval.proof_authority,config,
            {"max_clouds":retrieval.max_clouds,"retained_gpu_bytes":retrieval.max_cache_bytes})
        if retrieval.audit_nearest or retrieval.audit_misses or kwargs.get("gpu_timing",True) is not False:
            raise ValueError("Timing must omit full audits and retain the canonical False-event policy")
        self.authority = authority
        self.contract = audited.source_contract()
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._scratch = None
        # The only skipped constructor policy is the frozen audited-only guard.
        # Original device/resident math and its kernel construction remain used.
        DeviceGridResidentICP.__init__(self,retrieval,solve,**kwargs)
        self._configuration = self._current_configuration()
        self.combined_statistics = {key:0 for key in ("iteration_calls","primary_counter_term_syncs",
            "primary_download_bytes","provisional_queries","provisional_candidate_visits","common_calls",
            "common_queries","flagged_calls","flagged_queries","discarded_placeholder_calls",
            "malformed_results","peak_combined_owned_query_bytes")}
        self.combined_statistics.update({key:0. for key in ("primary_enqueue_s","primary_copy_wait_s","primary_normalize_s")})
        cp=self.cp
        with cp.cuda.Device(self.device_id),cp.cuda.Stream.null:
            text=audited.guarded_gpu_source()
            self.combined_partial_kernel=cp.RawKernel(text,"combined_guarded_normal_partials",options=("--fmad=false",))
            self.combined_collapse_kernel=cp.RawKernel(text,"combined_guarded_collapse_partials",options=("--fmad=false",))
            self.combined_partial_kernel.compile()
            self.combined_collapse_kernel.compile()
        # Unsupported complete calls stop; legitimate per-query CPU ambiguity
        # remains in unchanged retrieval.nearest_device when flagged.
        self.cpu_fallback = self._forbid_full_fallback

    @staticmethod
    def _forbid_full_fallback(*args,**kwargs):
        raise CombinedSyncTimingFailure("Unsupported complete timed resident call; CPU retry forbidden")

    def report(self):
        value=DeviceGridResidentICP.report(self)
        value["statistics"].pop("nn_wall_s",None)
        value["uncollected_statistics"].append("nn_wall_s")
        value.update(combined_sync_timing={"policy":POLICY,"source_contract":dict(self.contract),
            "iteration_contract":dict(ITERATION_CONTRACT),"statistics":dict(self.combined_statistics),
            "audit_only":False,"new_timing_authority":True,"graph_capture":False,
            "source_sha256_before":self.source_sha256,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "observational_shadows":"omitted only after independent full combined and Device proofs",
            "uncollected_statistics":["full_query_audited_calls","full_query_audited_rows","id_bit_comparison_rows",
                "metric_bit_comparison_rows","filtered_bit_comparison_rows","term_bit_comparison_calls"],
            "retrieval_statistics_scope":"Original retrieval counters describe flagged rerun/CPU-resolution path only; every provisional query/counter/normal copy is separately counted above",
            "transfer_counter_scope":"Inherited resident device_to_host_bytes covers final correspondence and flagged original equation copies only. Add primary_download_bytes for every combined counter/normal copy; inherited retrieval device_counter_syncs and device_query_download_bytes separately describe flagged rerun counter/packet copies. These are explicit payload bytes, not PCIe bus traffic.",
            "timer_scope":"Primary copy wait includes queued original NN and normal equations.320B per common iteration; flagged iterations additionally retain original CPU resolution and original equation transfer. No shader-only or additive nested-time claim.",
            "memory_scope":"Same conservative audited forecast retained:20N+320B persistent plus136N+80B query bound; raw40N released after completed primary copy. Pools/host packets/external buffers additional."})
        value["performance_attribution_valid"]=True
        return value
