"""Stdlib supervisor patch/closure/worker/forwarding fault contracts."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import profile_final_headroom as producer
from scripts.research.final_allocation_headroom import HeadroomFinalScope,ObservedOriginalHeadroomScope
from scripts.research.validate_checkpoint_finish_proof import FINISH_ARTIFACTS


class Harness:
    def __init__(self,args,metrics):
        self.KIND = "frozen-original-kind"
        self.EXPERIMENT_ARTIFACTS = (*FINISH_ARTIFACTS,"scripts/research/final_allocation_scope.py","scripts/research/profile_final_allocation.py")
        self.RightSizedFinalScope,self.ObservedOriginalFinalScope = object(),object()
        self.args,self.metrics = args,metrics
        self.failure = None
        self.worker_requests = 1
        self.armed,self.fail_restore = False,False
        self.original_allocator = self.RightSizedFinalScope
        self.received = None
        self.bad_status = False

    def __setattr__(self,name,value):
        if (name == "RightSizedFinalScope" and getattr(self,"armed",False) and getattr(self,"fail_restore",False)
                and value is self.original_allocator):
            raise RuntimeError("injected supervisor restore failure")
        object.__setattr__(self,name,value)

    def main(self):
        self.received = list(sys.argv)
        selected = self.RightSizedFinalScope if self.args.mode == "exact-rightsized" else self.ObservedOriginalFinalScope
        value = selected(object(),original_final=lambda *a:None)
        assert isinstance(value,HeadroomFinalScope if self.args.mode == "exact-rightsized" else ObservedOriginalHeadroomScope)
        assert value.physical_budget == self.args.physical_block_budget
        if self.failure is not None:
            raise self.failure
        pins = {name:producer.sha(producer.ROOT/name) for name in self.EXPERIMENT_ARTIFACTS}
        self.args.output.write_text("{}",encoding="utf-8")
        self.args.output.with_suffix(".allocation.json").write_text(json.dumps({"kind":self.KIND,
            "status":"failed" if self.bad_status else "complete","artifacts_sha256":pins,"artifacts_sha256_after":pins,
            "allocation":{"kind":"offline-original-final-reserve-headroom-v2","physical_budget_blocks":self.args.physical_block_budget},
            "outcome":"expected-early-capacity-failure" if "--expect-capacity-failure" in sys.argv else "mesh-built-quality-unproven"}),encoding="utf-8")
        for _ in range(self.worker_requests):
            self.metrics.finish_cuda_worker()
        self.armed = True


class SupervisorContracts(unittest.TestCase):
    def setup_run(self,folder,*,mode="exact-rightsized"):
        args = SimpleNamespace(output=Path(folder)/"report.json",mode=mode,physical_block_budget=10000)
        metrics = SimpleNamespace(finish_cuda_worker=None)
        base = Harness(args,metrics)
        original_kind,original_argv = base.KIND,sys.argv
        calls = []
        def worker():
            report = json.loads(args.output.with_suffix(".headroom.json").read_text())
            self.assertEqual(report["status"],"complete")
            self.assertTrue(report["supervisor_restored"])
            self.assertEqual(base.KIND,original_kind)
            self.assertIs(sys.argv,original_argv)
            self.assertIs(metrics.finish_cuda_worker,worker)
            calls.append(True)
        metrics.finish_cuda_worker = worker
        return args,base,metrics,calls

    def test_both_modes_forward_original_options_and_close_before_worker(self):
        for mode in ("native-original","exact-rightsized"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                args,base,metrics,calls = self.setup_run(folder,mode=mode)
                forwarded = ["capture.zip","--checkpoint","checkpoint.json","--final-block-count","20000"]
                producer.run(args,forwarded,base=base,metrics=metrics)
                self.assertEqual(calls,[True])
                self.assertEqual(base.received[-len(forwarded):],forwarded)
                self.assertNotIn("--physical-block-budget",base.received)
                report = json.loads(args.output.with_suffix(".headroom.json").read_text())
                self.assertEqual(len(report["artifacts_sha256"]),19)
                self.assertEqual(report["physical_block_budget"],10000)
                self.assertEqual(report["deferred_terminal_worker_calls"],1)
                self.assertFalse(report["geometry_quality_proven"])

    def test_expected_unique_overflow_flag_delegated_without_geometry_relabel(self):
        with tempfile.TemporaryDirectory() as folder:
            args,base,metrics,_ = self.setup_run(folder)
            producer.run(args,["capture.zip","--expect-capacity-failure"],base=base,metrics=metrics)
            report = json.loads(args.output.with_suffix(".headroom.json").read_text())
            self.assertEqual(report["outcome"],"expected-early-capacity-failure")
            self.assertFalse(report["performance_authority"])

    def test_primary_failure_restores_all_and_never_terminates_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            args,base,metrics,calls = self.setup_run(folder)
            primary = ValueError("original reconstruction failed")
            base.failure = primary
            old = (base.KIND,base.EXPERIMENT_ARTIFACTS,base.RightSizedFinalScope,base.ObservedOriginalFinalScope,sys.argv)
            with self.assertRaises(ValueError) as caught:
                producer.run(args,["capture.zip"],base=base,metrics=metrics)
            self.assertIs(caught.exception,primary)
            self.assertEqual((base.KIND,base.EXPERIMENT_ARTIFACTS,base.RightSizedFinalScope,base.ObservedOriginalFinalScope,sys.argv),old)
            self.assertEqual(calls,[])
            self.assertEqual(json.loads(args.output.with_suffix(".headroom.json").read_text())["status"],"failed")

    def test_missing_or_duplicate_terminal_request_rejects_saved_success(self):
        for count in (0,2):
            with self.subTest(count=count),tempfile.TemporaryDirectory() as folder:
                args,base,metrics,calls = self.setup_run(folder)
                base.worker_requests = count
                with self.assertRaises(RuntimeError):
                    producer.run(args,[],base=base,metrics=metrics)
                self.assertEqual(calls,[])
                self.assertEqual(json.loads(args.output.with_suffix(".headroom.json").read_text())["status"],"failed")

    def test_delegated_failed_status_cannot_become_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            args,base,metrics,calls = self.setup_run(folder)
            base.bad_status = True
            with self.assertRaises(RuntimeError):
                producer.run(args,[],base=base,metrics=metrics)
            self.assertEqual(calls,[])

    def test_restore_failure_attempts_other_hooks_and_argv_prevents_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            args,base,metrics,calls = self.setup_run(folder)
            base.fail_restore = True
            original_worker,argv = metrics.finish_cuda_worker,sys.argv
            with self.assertRaises(RuntimeError):
                producer.run(args,[],base=base,metrics=metrics)
            self.assertEqual(base.KIND,"frozen-original-kind")
            self.assertIs(metrics.finish_cuda_worker,original_worker)
            self.assertIs(sys.argv,argv)
            self.assertEqual(calls,[])
            report = json.loads(args.output.with_suffix(".headroom.json").read_text())
            self.assertFalse(report["supervisor_restored"])
            self.assertEqual(report["status"],"failed")

    def test_primary_preserved_when_envelope_closure_write_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            args,base,metrics,calls = self.setup_run(folder)
            primary = ValueError("native original failed")
            base.failure = primary
            write,counts = Path.write_text,[0]
            def injected(path,*values,**kwargs):
                if path == args.output.with_suffix(".headroom.json"):
                    counts[0] += 1
                    if counts[0] == 2:
                        raise OSError("injected envelope write failure")
                return write(path,*values,**kwargs)
            with patch.object(Path,"write_text",injected),self.assertRaises(ValueError) as caught:
                producer.run(args,[],base=base,metrics=metrics)
            self.assertIs(caught.exception,primary)
            self.assertIsInstance(caught.exception.__cause__,OSError)
            self.assertEqual(calls,[])

    def test_existing_reports_preserved_before_any_patch(self):
        with tempfile.TemporaryDirectory() as folder:
            args,base,metrics,calls = self.setup_run(folder)
            args.output.write_bytes(b"original measured bytes")
            with self.assertRaises(RuntimeError):
                producer.run(args,[],base=base,metrics=metrics)
            self.assertEqual(args.output.read_bytes(),b"original measured bytes")
            self.assertEqual(base.KIND,"frozen-original-kind")
            self.assertEqual(calls,[])

    def test_bound_budget_boolean_rejected_before_any_patch(self):
        with tempfile.TemporaryDirectory() as folder:
            args,base,metrics,calls = self.setup_run(folder)
            args.physical_block_budget = True
            with self.assertRaises(ValueError):
                producer.run(args,[],base=base,metrics=metrics)
            self.assertEqual(base.KIND,"frozen-original-kind")
            self.assertEqual(calls,[])

    def test_cli_help_from_temporary_directory_without_site_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable,"-S",str(producer.ROOT/"scripts/research/profile_final_headroom.py"),"--help"],
                cwd=folder,text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("--physical-block-budget",result.stdout)
            self.assertEqual(list(Path(folder).iterdir()),[])


if __name__ == "__main__":
    unittest.main()
