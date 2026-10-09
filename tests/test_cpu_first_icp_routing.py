"""Artificial routing/state tests; no native work or Finish qualification."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import cpu_first_icp_routing as routing


def descriptor(rows=2,digest="a"):
    return {"dtype":"<f8","shape":[rows,3],"nbytes":rows*24,"sha256":digest*64}


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now


class GPU:
    def __init__(self,configuration):
        self.configuration=configuration;self.closed=False;self.calls=0;self.close_calls=0
        self.result=object();self.error=None;self.close_error=None
    def match(self,*args):
        self.calls+=1
        if self.error is not None:raise self.error
        return self.result
    def close(self):
        self.close_calls+=1
        if self.close_error is not None:raise self.close_error
        self.closed=True


class Fixture:
    def __init__(self,elapsed=.04):
        self.clock=Clock();self.elapsed=elapsed;self.cpu_calls=0;self.factory_calls=0
        self.cpu_result=object();self.gpu=None;self.factory_error=None
        self.owners=[SimpleNamespace(index=0),SimpleNamespace(index=1),SimpleNamespace(index=2)]
        self.configuration_owner=object()
        self.configuration={k:routing.LIMITS[k] for k in ("max_points","max_scratch_bytes","pair_cache_bytes","max_total_bytes")}
        self.configuration["stages"]=[[.12,40],[.06,30],[.03,20]]
        self.cloud={"points":descriptor(),"normals":descriptor(digest="b"),"colors":descriptor(digest="c")}
        self.inputs={"source":descriptor(),"target":descriptor(digest="b"),"seed_sha256":"d"*64}
        self.checks=[];self.boundary_cost=0.
        self.router=routing.CpuFirstPairRouter(self.cpu,self.factory,snapshot_pair=self.pair,
            snapshot_inputs=self.input,check_result=self.check,clock=self.clock)
    def pair(self,a,b,configuration):
        self.clock.now+=self.boundary_cost
        return {"pair":[a.index,b.index],"source":[copy.deepcopy(self.cloud)],"target":[copy.deepcopy(self.cloud)],
            "configuration":copy.deepcopy(self.configuration)}
    def input(self,*args):
        self.clock.now+=self.boundary_cost;return copy.deepcopy(self.inputs)
    def check(self,result,*args):
        self.clock.now+=self.boundary_cost;self.checks.append(result)
    def cpu(self,*args):
        self.cpu_calls+=1;self.clock.now+=self.elapsed;return self.cpu_result
    def factory(self,configuration):
        self.factory_calls+=1
        if self.factory_error is not None:raise self.factory_error
        self.gpu=GPU(configuration);return self.gpu
    def call(self,a=0,b=1,**kwargs):
        return self.router.match("source","target","unrounded seed",pair_source_owner=self.owners[a],
            pair_target_owner=self.owners[b],configuration_owner=self.configuration_owner,**kwargs)


class CpuFirstRoutingTests(unittest.TestCase):
    def test_first_result_identity_and_no_eager_gpu(self):
        f=Fixture()
        self.assertIs(f.call(),f.cpu_result)
        self.assertEqual((f.cpu_calls,f.factory_calls),(1,0))
        self.assertIs(f.call(),f.gpu.result)
        self.assertEqual((f.cpu_calls,f.factory_calls,f.gpu.calls),(1,1,1))

    def test_complete_timer_includes_pair_input_and_result_checks(self):
        f=Fixture(.03);f.boundary_cost=.002
        f.call();row=f.router.report()["pairs"][0]
        self.assertAlmostEqual(row["first_elapsed_s"],.04)
        self.assertEqual(row["route"],"gpu")
        self.assertIs(f.checks[0],f.cpu_result)

    def test_exact_threshold_and_adjacent_below(self):
        for elapsed,route in ((math.nextafter(.04,0.),"cpu"),(.04,"gpu"),(math.nextafter(.04,1.),"gpu")):
            f=Fixture(elapsed);f.call();self.assertEqual(f.router.report()["pairs"][0]["route"],route)
            f.call();self.assertEqual(f.factory_calls,route=="gpu")

    def test_reverse_and_different_pair_need_their_own_first_cpu(self):
        f=Fixture(.05);f.call();f.call(1,0);f.call(0,2)
        self.assertEqual((f.cpu_calls,f.factory_calls),(3,0))
        f.call(1,0);f.call(0,2)
        self.assertEqual((f.factory_calls,f.gpu.calls),(1,2))

    def test_pair_cloud_normal_color_and_config_changes_refuse_before_work(self):
        for name in ("points","normals","colors","configuration"):
            f=Fixture();f.call()
            if name=="configuration":f.configuration["stages"][0][1]=39
            else:f.cloud[name]["sha256"]="e"*64
            with self.assertRaisesRegex(routing.RoutingFailure,"cloud bytes/configuration changed"):f.call()
            self.assertEqual((f.cpu_calls,f.factory_calls),(1,0))

    def test_same_endpoint_numbers_cannot_replace_original_owner(self):
        f=Fixture();f.call();f.owners[0]=SimpleNamespace(index=0)
        with self.assertRaisesRegex(routing.RoutingFailure,"original owners"):f.call()
        f=Fixture();f.call();f.configuration_owner=object()
        with self.assertRaisesRegex(routing.RoutingFailure,"original owners"):f.call()

    def test_invalid_or_unknown_first_timing_latches_without_gpu(self):
        for elapsed in (float("nan"),float("inf"),-.01):
            f=Fixture(elapsed)
            with self.assertRaisesRegex(routing.RoutingFailure,"timing") as caught:f.call()
            with self.assertRaises(routing.RoutingFailure) as again:f.call()
            self.assertIs(caught.exception,again.exception)
            self.assertEqual((f.cpu_calls,f.factory_calls),(1,0))

    def test_gpu_failure_is_sticky_and_never_retries_cpu(self):
        f=Fixture();f.call();f.call();original=RuntimeError("device fault");f.gpu.error=original
        with self.assertRaises(routing.RoutingFailure) as caught:f.call()
        self.assertIs(caught.exception.__cause__,original)
        with self.assertRaises(routing.RoutingFailure) as again:f.call()
        self.assertIs(caught.exception,again.exception)
        self.assertEqual((f.cpu_calls,f.gpu.calls),(1,2))
        with self.assertRaises(routing.RoutingFailure):f.router.close()
        self.assertTrue(f.router.closed);self.assertTrue(f.gpu.closed)

    def test_partial_factory_owner_retained_for_independent_cleanup(self):
        f=Fixture();f.call();owner=GPU(dict(f.configuration));error=RuntimeError("partial factory")
        error.routing_owner=owner;f.factory_error=error
        with self.assertRaises(routing.RoutingFailure) as caught:f.call()
        self.assertIs(f.router.gpu_owner_held,owner)
        with self.assertRaises(routing.RoutingFailure) as cleanup:f.router.close()
        self.assertIs(cleanup.exception,caught.exception);self.assertTrue(owner.closed)

    def test_unknown_factory_completion_cannot_claim_closed(self):
        f=Fixture();f.call();f.factory_error=RuntimeError("unknown construction")
        with self.assertRaises(routing.RoutingFailure) as caught:f.call()
        with self.assertRaises(routing.RoutingFailure) as cleanup:f.router.close()
        self.assertIs(cleanup.exception,caught.exception);self.assertFalse(f.router.closed)
        self.assertIn("completion unknown",str(cleanup.exception.__cause__))

    def test_cleanup_failure_preserves_primary_and_retains_owner_for_retry(self):
        f=Fixture();f.call();f.call();primary=RuntimeError("body failure");cleanup=RuntimeError("completion failure")
        f.gpu.close_error=cleanup
        with self.assertRaises(RuntimeError) as caught:f.router.close(primary)
        self.assertIs(caught.exception,primary);self.assertIs(caught.exception.__cause__,cleanup)
        self.assertFalse(f.router.closed);self.assertIs(f.router.gpu_owner_held,f.gpu)
        calls=(f.cpu_calls,f.gpu.calls)
        with self.assertRaisesRegex(routing.RoutingFailure,"GPU cleanup failed"):f.call()
        self.assertEqual((f.cpu_calls,f.gpu.calls),calls)
        f.gpu.close_error=None
        with self.assertRaises(RuntimeError) as caught:f.router.close(primary)
        self.assertIs(caught.exception,primary);self.assertTrue(f.router.closed)

    def test_owner_replacement_cleanup_uses_captured_gpu_owner(self):
        f=Fixture();f.call();f.call();original=f.gpu;foreign=GPU(dict(f.configuration));f.router.gpu_owner=foreign
        with self.assertRaisesRegex(routing.RoutingFailure,"owner replaced"):f.call()
        with self.assertRaises(routing.RoutingFailure):f.router.close()
        self.assertTrue(original.closed);self.assertEqual(foreign.close_calls,0)

    def test_configuration_caps_and_actual_job_cap_refuse_before_work(self):
        f=Fixture();f.configuration["max_total_bytes"]+=1
        with self.assertRaisesRegex(routing.RoutingFailure,"Bounded GPU configuration"):f.call()
        self.assertEqual(f.cpu_calls,0)
        f=Fixture();f.router.calls=[{}]*routing.LIMITS["max_jobs"]
        with self.assertRaisesRegex(routing.RoutingFailure,"job cap"):f.call()
        self.assertEqual(f.cpu_calls,0)
        f=Fixture();f.router.pairs={key:{} for key in [(a,b) for a in range(32) for b in range(32) if a!=b][:routing.LIMITS["max_pairs"]]}
        with self.assertRaisesRegex(routing.RoutingFailure,"pair cap"):
            f.router.match("source","target","seed",pair_source_owner=SimpleNamespace(index=31),
                pair_target_owner=SimpleNamespace(index=30),configuration_owner=f.configuration_owner)
        self.assertEqual(f.cpu_calls,0)

    def test_original_callable_replacement_and_policy_loosen_refuse(self):
        f=Fixture();f.router.original_cpu=lambda *args:object()
        with self.assertRaisesRegex(routing.RoutingFailure,"owner changed"):f.call()
        f=Fixture()
        with patch.object(routing,"THRESHOLD_S",.039):
            with self.assertRaisesRegex(routing.RoutingFailure,"policy changed"):f.call()

    def test_closed_route_and_report_are_no_authority_and_detached(self):
        f=Fixture(.01);f.call();f.router.close()
        report=f.router.report();report["calls"][0]["pair"][0]=99
        self.assertEqual(f.router.report()["calls"][0]["pair"],[0,1])
        for key in ("whole_finish_validated","performance_claim","production_authority"):
            self.assertFalse(report[key])
        with self.assertRaisesRegex(routing.RoutingFailure,"Closed router"):f.call()


if __name__=="__main__":unittest.main()
