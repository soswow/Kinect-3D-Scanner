"""Prospective CPU-first routing; no native imports or Finish qualification.

The caller supplies source-bound snapshot/check adapters and a bounded GPU
owner. A pair snapshot inventories every immutable train cloud in that directed
fragment pair, with point/normal/color descriptors, and its original configuration.
This helper never infers a route from archived timings, seeds, or acceptance.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from types import MappingProxyType

POLICY = "cpu-first-40ms-v1"
THRESHOLD_S = .040
LIMITS = MappingProxyType({"max_pairs":256,"max_jobs":4096,"max_points":1_000_000,
    "max_scratch_bytes":268435456,"pair_cache_bytes":268435456,"max_total_bytes":536870912})


class RoutingFailure(BaseException):
    """Routing or GPU faults cannot become an ordinary proposal rejection."""


def require(ok, message):
    if not ok: raise RoutingFailure(message)


def canonical(value):
    # A detached finite JSON value binds exact supplied descriptor/configuration
    # values; it is not numerical normalization or rounding.
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def pair_contract(binding):
    require(type(binding) is dict and type(binding.get("pair")) is list
        and len(binding["pair"])==2 and all(type(i) is int and 0<=i<32 for i in binding["pair"])
        and binding["pair"][0]!=binding["pair"][1], "Bounded directed fragment IDs required")
    for side in ("source","target"):
        clouds=binding.get(side)
        require(type(clouds) is list and 0<len(clouds)<=32,"Complete bounded immutable pair cloud inventory required")
        for cloud in clouds:
            require(type(cloud) is dict and set(cloud)=={"points","normals","colors"},"Point/normal/color inventory required")
            count=None
            for name in ("points","normals","colors"):
                record=cloud[name]
                require(type(record) is dict and set(record)=={"dtype","shape","nbytes","sha256"},"Exact verified cloud descriptor required")
                shape,digest=record["shape"],record["sha256"]
                require(record["dtype"]=="<f8" and type(shape) is list and len(shape)==2
                    and all(type(i) is int for i in shape) and shape[1]==3 and 0<=shape[0]<=LIMITS["max_points"]
                    and type(record["nbytes"]) is int and record["nbytes"]==24*shape[0]
                    and type(digest) is str and len(digest)==64 and all(c in "0123456789abcdef" for c in digest),"Malformed verified immutable cloud descriptor")
                if name=="points": count=shape[0];require(count>0,"Nonempty train clouds required")
                else: require(shape[0] in ((count,) if name=="normals" else (0,count)),"Original cloud attribute rows differ")
    configuration=binding.get("configuration")
    require(type(configuration) is dict and configuration,"Original effective configuration required")
    for key in ("max_points","max_scratch_bytes","pair_cache_bytes","max_total_bytes"):
        require(type(configuration.get(key)) is int and 0<configuration[key]<=LIMITS[key],"Bounded GPU configuration required: "+key)
    canonical(binding)
    return tuple(binding["pair"])


class CpuFirstPairRouter:
    """One Finish owner; one lazy GPU adapter and no successful error fallback.

    snapshot_pair(pair_source_owner,pair_target_owner,configuration_owner) must
    freshly verify actual owner bytes and return the pair_contract schema.
    snapshot_inputs(source,target,initial) returns exact verified current input
    descriptors. check_result(result,source,target,initial) raises on invalid
    results and must preserve result/input objects. gpu_factory(configuration)
    returns an existing bounded adapter exposing match(), close(), closed and
    configuration. A partially failing factory attaches routing_owner to its
    exception so this owner remains available for independent cleanup.
    """
    def __init__(self, original_cpu, gpu_factory, *, snapshot_pair, snapshot_inputs,
                 check_result, clock=time.perf_counter):
        self.original_cpu,self.gpu_factory=original_cpu,gpu_factory
        self.snapshot_pair,self.snapshot_inputs,self.check_result,self.clock=snapshot_pair,snapshot_inputs,check_result,clock
        callbacks=(original_cpu,gpu_factory,snapshot_pair,snapshot_inputs,check_result,clock)
        require(all(callable(f) for f in callbacks),"Captured original and verified caller adapters required")
        self.callback_owners=callbacks
        self.callback_states=tuple((getattr(f,"__code__",None),repr(getattr(f,"__defaults__",None)),
            repr(getattr(f,"__kwdefaults__",None))) for f in callbacks)
        self.limit_owner=LIMITS;self.limit_values=canonical(dict(LIMITS))
        self.pairs={};self.calls=[];self.gpu_owner=self.gpu_owner_held=None;self.gpu_configuration=self.gpu_config_owner=None
        self.factory_incomplete=False
        self.failure=None;self.cleanup_failures=[];self.closed=False;self.factory_calls=0

    def _fail(self,error):
        if self.failure is None:
            self.failure=error if isinstance(error,RoutingFailure) else RoutingFailure("CPU-first routing failed: "+repr(error))
            if self.failure is not error:self.failure.__cause__=error
        raise self.failure

    def _healthy(self):
        if self.failure is not None:raise self.failure
        require(not self.closed,"Closed router cannot dispatch")
        require(POLICY=="cpu-first-40ms-v1" and type(THRESHOLD_S) is float and THRESHOLD_S==.040
            and LIMITS is self.limit_owner and canonical(dict(LIMITS))==self.limit_values,"Effective routing policy changed")
        actual=(self.original_cpu,self.gpu_factory,self.snapshot_pair,self.snapshot_inputs,self.check_result,self.clock)
        require(all(a is b for a,b in zip(actual,self.callback_owners)),"Original caller/adapter owner changed")
        require(tuple((getattr(f,"__code__",None),repr(getattr(f,"__defaults__",None)),
            repr(getattr(f,"__kwdefaults__",None))) for f in actual)==self.callback_states,"Original caller/adapter code/default changed")
        require(self.gpu_owner is self.gpu_owner_held,"GPU adapter owner replaced")

    def match(self,source,target,initial,*,pair_source_owner,pair_target_owner,configuration_owner):
        record=None
        try:
            started=self.clock();self._healthy()
            require(len(self.calls)<LIMITS["max_jobs"],"Routing actual job cap exceeded")
            binding=self.snapshot_pair(pair_source_owner,pair_target_owner,configuration_owner)
            pair=pair_contract(binding);bound=canonical(binding)
            owners=(pair_source_owner,pair_target_owner,configuration_owner)
            state=self.pairs.get(pair)
            if state is None:
                require(len(self.pairs)<LIMITS["max_pairs"],"Routing directed pair cap exceeded")
                state={"owners":owners,"binding":bound,"route":None,"first_elapsed_s":None,"calls":0}
                self.pairs[pair]=state
            require(all(a is b for a,b in zip(state["owners"],owners)) and state["binding"]==bound,
                "Directed pair original owners/cloud bytes/configuration changed")
            first=state["route"] is None
            route="cpu" if first else state["route"]
            record={"call_index":len(self.calls),"pair":list(pair),"route":route,"first":first,"complete":False}
            self.calls.append(record)
            inputs=canonical(self.snapshot_inputs(source,target,initial))
            self._healthy()
            if route=="cpu": result=self.original_cpu(source,target,initial)
            else:
                if self.gpu_owner is None:
                    self.factory_calls+=1
                    configuration=json.loads(bound)["configuration"]
                    self.factory_incomplete=True
                    try:self.gpu_owner=self.gpu_factory(configuration)
                    except BaseException as error:
                        self.gpu_owner=self.gpu_owner_held=getattr(error,"routing_owner",None);raise
                    self.gpu_owner_held=self.gpu_owner
                    self.factory_incomplete=False
                    self.gpu_config_owner=self.gpu_owner.configuration
                    self.gpu_configuration=canonical(configuration)
                require(self.gpu_owner is not None and self.gpu_owner.closed is False
                    and self.gpu_owner.configuration is self.gpu_config_owner
                    and canonical(dict(self.gpu_owner.configuration))==self.gpu_configuration==canonical(binding["configuration"]),
                    "Active bounded GPU adapter/configuration required")
                self._healthy()
                result=self.gpu_owner.match(source,target,initial)
            self.check_result(result,source,target,initial)
            require(canonical(self.snapshot_inputs(source,target,initial))==inputs,"Actual call mutated original inputs")
            require(canonical(self.snapshot_pair(*owners))==bound,"Actual call mutated pair/configuration bytes")
            self._healthy();ended=self.clock()
            require(type(started) in (int,float) and type(ended) in (int,float)
                and math.isfinite(started) and math.isfinite(ended) and ended>=started,"Unknown/nonmonotonic complete call timing")
            elapsed=ended-started
            require(math.isfinite(elapsed),"Unknown complete call elapsed")
            if first: state.update(first_elapsed_s=elapsed,route="gpu" if elapsed>=THRESHOLD_S else "cpu")
            state["calls"]+=1;record.update(complete=True,complete_call_wall_s=elapsed)
            return result  # Original CPU first result and all later returns are unchanged objects.
        except BaseException as error:
            if record is not None:record["failure"]=repr(error)
            self._fail(error)

    def close(self,primary=None):
        cleanup=None
        if not self.closed and self.gpu_owner_held is not None:
            try:
                self.gpu_owner_held.close()
                require(self.gpu_owner_held.closed is True,"GPU selected completion unproved; owner retained")
                self.factory_incomplete=False
            except BaseException as error:cleanup=error;self.cleanup_failures.append(repr(error))
        if self.factory_incomplete and cleanup is None:
            cleanup=RoutingFailure("Failed factory completion unknown; no complete retained owner receipt")
            self.cleanup_failures.append(repr(cleanup))
        self.closed=cleanup is None
        if cleanup is not None and self.failure is None:
            self.failure=cleanup if isinstance(cleanup,RoutingFailure) else RoutingFailure("GPU cleanup failed: "+repr(cleanup))
            if self.failure is not cleanup:self.failure.__cause__=cleanup
        fault=primary or self.failure
        if fault is not None:
            if cleanup is not None:raise fault from cleanup
            raise fault

    def report(self):
        return {"policy":POLICY,"threshold_s":THRESHOLD_S,"closed":self.closed,
            "failure":None if self.failure is None else repr(self.failure),"cleanup_failures":list(self.cleanup_failures),
            "factory_calls":self.factory_calls,"calls":json.loads(canonical(self.calls)),
            "pairs":[{"pair":list(pair),"binding_sha256":hashlib.sha256(s["binding"].encode()).hexdigest(),
                "route":s["route"],"first_elapsed_s":s["first_elapsed_s"],"calls":s["calls"]} for pair,s in self.pairs.items()],
            "limits":dict(LIMITS),"memory_enforcement":"Existing bounded GPU adapter; no allocation in routing helper",
            "first_timer_scope":"Pair/input checks, actual original CPU match, result/input checks and owner checks",
            "whole_finish_validated":False,"performance_claim":False,"production_authority":False}
