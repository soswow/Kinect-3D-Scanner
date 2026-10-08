"""Offline allocation producer dispatch/closure; no numerical runtime needed."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import profile_final_allocation as producer


class BoundaryContracts(unittest.TestCase):
    def engine(self, original):
        class Engine:
            def _final_volume(self, progress_cb=None):
                return original(self, progress_cb)
        return Engine()

    def test_original_result_and_arguments_are_untouched_and_instance_restores(self):
        value, callback, calls = object(), object(), []
        engine = self.engine(lambda owner, progress: calls.append((owner,progress)) or value)
        # The real volume API is the only observation after native return.
        volume = SimpleNamespace(hashmap=lambda: SimpleNamespace(capacity=lambda:3,size=lambda:2))
        engine._final_volume = lambda progress: calls.append((engine,progress)) or volume
        previous = engine._final_volume
        with producer.FinalBoundaryObserver(engine, lambda owner:{"owner":id(owner)}) as scope:
            self.assertIs(engine._final_volume(callback), volume)
        self.assertEqual(calls, [(engine, callback)])
        self.assertIs(engine._final_volume, previous)
        self.assertTrue(scope.restored)
        self.assertTrue(scope.calls[0]["owner_and_pose_preserved"])
        self.assertEqual(scope.calls[0]["candidate_capacity"], 3)

    def test_original_error_is_retained_when_post_observation_also_fails(self):
        primary, secondary = RuntimeError("native failure"), RuntimeError("observation failure")
        def fail(owner, progress):
            raise primary
        engine = self.engine(fail)
        calls = []
        def state(owner):
            calls.append(owner)
            if len(calls) == 2:
                raise secondary
            return {"owner":id(owner)}
        with self.assertRaises(RuntimeError) as caught:
            with producer.FinalBoundaryObserver(engine, state) as scope:
                engine._final_volume()
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, secondary)
        self.assertTrue(any("observation failure" in note for note in primary.__notes__))
        self.assertTrue(scope.restored)
        self.assertEqual(scope.calls[0]["failure"]["message"], "native failure")

    def test_changed_live_boundary_is_not_a_successful_allocation_experiment(self):
        engine = self.engine(lambda owner, progress: None)
        volume = SimpleNamespace(hashmap=lambda: SimpleNamespace(capacity=lambda:3,size=lambda:2))
        counter = []
        engine._final_volume = lambda progress: counter.append("changed") or volume
        with self.assertRaisesRegex(producer.FinalAllocationContractError, "Live ownership"):
            with producer.FinalBoundaryObserver(engine, lambda owner:{"state":len(counter)}) as scope:
                engine._final_volume()
        self.assertFalse(scope.calls[0]["owner_and_pose_preserved"])
        self.assertTrue(scope.restored)

    def test_partial_installation_failure_restores_previous_class_method(self):
        primary = RuntimeError("partial hook install")
        class Engine:
            def __setattr__(self,name,value):
                object.__setattr__(self,name,value)
                if name == "_final_volume":
                    raise primary
            def _final_volume(self,progress_cb=None):
                return "original"
        engine = Engine()
        with self.assertRaises(RuntimeError) as caught:
            producer.FinalBoundaryObserver(engine, lambda owner:{}).__enter__()
        self.assertIs(caught.exception, primary)
        self.assertNotIn("_final_volume", engine.__dict__)
        self.assertEqual(engine._final_volume(), "original")

    def test_wrong_instance_is_rejected_before_delegate(self):
        engine = self.engine(lambda owner,progress: "unexpected")
        with producer.FinalBoundaryObserver(engine, lambda owner:{}):
            with self.assertRaisesRegex(producer.FinalAllocationContractError, "different engine"):
                engine._final_volume.__func__(object())


class ProducerCliContracts(unittest.TestCase):
    def test_real_help_and_hardware_permission_guard_work_from_unrelated_directory(self):
        script = producer.ROOT/"scripts/research/profile_final_allocation.py"
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable,"-S",str(script),"--help"],cwd=folder,
                                    capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("exact-rightsized",result.stdout)
            result = subprocess.run([sys.executable,"-S",str(script),"raw.zip","--mode","exact-rightsized",
                                    "--checkpoint","checkpoint.json","--component-synthetic","s.json",
                                    "--component-bridge","b.json","--output","out.json"],cwd=folder,
                                    capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,2)
            self.assertIn("Require an explicitly allocated exclusive hardware slot",result.stderr)
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_import_does_not_load_native_runtime_or_create_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            code = ("import sys,json;sys.path.insert(0,"+repr(str(producer.ROOT))+");"
                    "import scripts.research.profile_final_allocation;"
                    "print(json.dumps([n for n in sys.modules if n.split('.')[0] in "
                    "('numpy','open3d','cupy') or n=='scanner_server.engine']))")
            result = subprocess.run([sys.executable,"-S","-c",code],cwd=folder,
                                    capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),[])
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_experiment_pins_do_not_relabel_checkpoint_producer_or_old_proofs(self):
        self.assertEqual(len(producer.EXPERIMENT_ARTIFACTS),len(producer.FINISH_ARTIFACTS)+2)
        self.assertEqual(producer.EXPERIMENT_ARTIFACTS[:len(producer.FINISH_ARTIFACTS)],producer.FINISH_ARTIFACTS)
        self.assertNotIn(producer.KIND,("offline-full-finish-device-flat-resident-checkpoint-v2",
                                       "offline-full-finish-device-flat-resident-checkpoint-proposal-threads-v3"))


if __name__ == "__main__":
    unittest.main()
