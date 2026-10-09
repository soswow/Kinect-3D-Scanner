"""Causal host source-check sibling; every numerical/control statement retained.

Only original full-source AST reconstruction may use equivalent whole-file
SHA equality plus cached-contract equality. Both modes add identical nested
source-check/pre-enqueue clocks; unchanged kernels/caps/CPU ambiguity/cleanup
and original Eigen/convergence are inherited from the separately proved path.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import FunctionType

from scripts.research import combined_sync_timing_adapter as parent
from scripts.research import research_combined_sync_icp as audited
from scripts.research.validate_combined_source_guard_timing import (
    POLICY,SOURCE,ROOT,validate_source_guard_activation)


def measured_iteration():
    # Recover the already reviewed generated function via its construction
    # source. Its complete AST is available in the equivalent function's code
    # metadata only indirectly, so capture compile(tree) without numerical work.
    captured=[]
    real_compile=compile
    scope=dict(parent.timed_iteration.__globals__)
    def capture(value,*args,**kwargs):
        if isinstance(value,ast.Module):captured.append(copy.deepcopy(value))
        return real_compile(value,*args,**kwargs)
    scope["compile"]=capture
    constructor=FunctionType(parent.timed_iteration.__code__,scope,
        parent.timed_iteration.__name__,parent.timed_iteration.__defaults__,parent.timed_iteration.__closure__)
    constructor()
    if len(captured)!=1:raise RuntimeError("Original timed iteration construction changed")
    tree=captured[0];function=tree.body[0]
    before=ast.dump(function,include_attributes=False)
    seam=next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign)
        and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="started")
    first=ast.parse("_source_guard_preenqueue_started = time.perf_counter()").body[0]
    additions=ast.parse('self.source_guard_statistics["preenqueue_bookkeeping_s"] += time.perf_counter() - _source_guard_preenqueue_started\nself.source_guard_statistics["preenqueue_calls"] += 1').body
    function.body.insert(seam,additions[1]);function.body.insert(seam,additions[0]);function.body.insert(0,first)
    recovered=copy.deepcopy(function)
    del recovered.body[0];del recovered.body[seam:seam+2]
    if ast.dump(recovered,include_attributes=False)!=before:
        raise RuntimeError("Host clocks changed original timed control/math statements")
    ast.fix_missing_locations(tree)
    namespace=dict(parent._TIMED_ITERATION.__globals__)
    exec(compile(tree,__file__,"exec"),namespace)
    return namespace["_iteration_data"],{"original_timed_iteration_ast_sha256":hashlib.sha256(before.encode()).hexdigest(),
        "only_preenqueue_clock_insertions":True,"original_numerical_and_control_statements_unchanged":True,
        "new_kernel":False,"graph_capture":False}


_ITERATION,ITERATION_CONTRACT=measured_iteration()


def frozen_generator_value(value):
    if value is None or type(value) in (bool,int,str,bytes):return (type(value).__name__,value)
    if type(value) is float and math.isfinite(value):return ("float",value)
    if isinstance(value,Path):return ("path",type(value).__name__,str(value))
    if type(value) in (tuple,list):
        return (type(value).__name__,tuple(frozen_generator_value(item) for item in value))
    if type(value) is dict:
        # Preserve dictionary insertion order as well as every key/value.
        return ("dict",tuple((frozen_generator_value(key),frozen_generator_value(item)) for key,item in value.items()))
    raise RuntimeError("Unsupported source generator default/closure state")


def function_state(function):
    # Retain the immutable code object plus its identity, so replacing it with
    # equal-looking code also stops and the original object cannot be recycled.
    return (function,function.__code__,id(function.__code__),
        frozen_generator_value(function.__defaults__),frozen_generator_value(function.__kwdefaults__),
        tuple(frozen_generator_value(cell.cell_contents) for cell in (function.__closure__ or ())))


def generator_state():
    return (frozen_generator_value(audited._EDITS),audited.POLICY,audited.ORIGINAL_RESIDENT,
        *(function_state(function) for function in (
            audited.original_gpu_source,audited.guarded_gpu_source,audited.source_contract)))


class SourceGuardResidentICP(parent.CombinedSyncTimedResidentICP):
    _iteration_data=_ITERATION
    def __init__(self,retrieval,solve,*,authority,source_guard_authority,source_check_mode,**kwargs):
        validate_source_guard_activation(source_guard_authority,authority,source_check_mode)
        self.source_guard_authority=source_guard_authority
        self.source_check_mode=source_check_mode
        self._source_guard_generator_state=generator_state()
        self.source_guard_statistics={"source_checks":0,"iteration_config_checks":0,"preenqueue_calls":0,
            "source_check_s":0.,"iteration_config_check_s":0.,"preenqueue_bookkeeping_s":0.}
        super().__init__(retrieval,solve,authority=authority,**kwargs)
        if json.dumps(self.contract,sort_keys=True,separators=(",",":"),allow_nan=False)!=source_guard_authority.source_contract_json:
            raise RuntimeError("Constructor's actual full source contract differs from separate authority")

    def _check_configuration(self,*,check_source=False):
        started=time.perf_counter()
        try:
            if self.source_check_mode!=self.source_guard_authority.source_check_mode:
                raise RuntimeError("Source-check policy changed after construction")
            if self.source_check_mode=="original-ast" or not check_source:
                return super()._check_configuration(check_source=check_source)
            # Exact original source contract was reconstructed at construction.
            # Whole-file equality implies the same literal and generated guards;
            # cached contract equality also rejects mutable-contract corruption.
            super()._check_configuration(check_source=False)
            if (hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()!=self.source_guard_authority.original_source_sha256
                    or generator_state()!=self._source_guard_generator_state
                    or json.dumps(self.contract,sort_keys=True,separators=(",",":"),allow_nan=False)
                        !=self.source_guard_authority.source_contract_json):
                raise RuntimeError("Original full source bytes or source contract changed during component timing")
        finally:
            key="source" if check_source else "iteration_config"
            self.source_guard_statistics[key+"_checks"]+=1
            self.source_guard_statistics[key+"_check_s"]+=time.perf_counter()-started

    def report(self):
        value=super().report()
        value["source_guard_timing"]={"policy":POLICY,"mode":self.source_check_mode,
            "statistics":dict(self.source_guard_statistics),"iteration_contract":dict(ITERATION_CONTRACT),
            "source_guard_sources":dict(self.source_guard_authority.artifact_sha256),
            "original_whole_source_sha256":self.source_guard_authority.original_source_sha256,
            "original_contract_equality_preserved":True,
            "source_unchanged":all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
                for name,digest in self.source_guard_authority.artifact_sha256),
            "timer_scope":"Nested inclusive host wall; source-check inside resident call, preenqueue includes iteration config/buffer lookup/reset/view. Same clocks in AST and SHA modes; not shader-only, additive or subtractive attribution."}
        return value
