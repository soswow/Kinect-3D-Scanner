"""External worker exit and overlap receipts, without Windows/native imports."""
import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.research import monitor_gpu_icp_worker as monitor


def row(pid, parent, stamp=100):
    return {"pid": pid, "parent_pid": parent, "created_100ns": stamp, "name": "python.exe"}


class Child:
    pid = 2

    def __init__(self, code=0):
        self.code = code
        self.polls = 0
        self.waited = False

    def poll(self):
        self.polls += 1
        return None if self.polls <= 2 else self.code

    def wait(self):
        self.waited = True
        return self.code


class MonitorTests(unittest.TestCase):
    def test_descendant_survives_launcher_exit_but_pid_reuse_is_foreign(self):
        known = set()
        self.assertEqual(monitor.classify([row(1, 0), row(2, 1), row(3, 2)], 1, 2, known), ([], []))
        self.assertEqual(monitor.classify([row(3, 0)], 1, None, known), ([], []))
        self.assertEqual(monitor.classify([row(3, 0, 101)], 1, None, known)[0], [row(3, 0, 101)])

    def test_nested_descendants_do_not_depend_on_enumeration_order(self):
        self.assertEqual(monitor.classify([row(4, 3), row(3, 2), row(2, 1)], 1, 2, set()), ([], []))

    def test_child_predating_current_parent_generation_is_foreign(self):
        rows = [row(1, 0, 90), row(2, 1, 110), row(3, 2, 100)]
        self.assertEqual(monitor.classify(rows, 1, 2, set())[0], [row(3, 2, 100)])

    def test_absent_parent_generation_cannot_authorize_an_unknown_child(self):
        self.assertEqual(monitor.classify([row(3, 2)], 1, 2, set())[0], [row(3, 2)])

    def test_unknown_creation_identity_never_proves_isolation(self):
        unknown = row(2, 1, None)
        self.assertEqual(monitor.classify([unknown], 1, 2, set()), ([], [unknown]))

    def execute(self, inventory, *, child=None, idle=lambda: True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        receipt, log = Path(temporary.name) / "receipt.json", Path(temporary.name) / "log.txt"
        worker = Child() if child is None else child
        def launch(*args, **kwargs):
            kwargs["stdout"].write(b"actual worker log\n")
            return worker
        with patch.object(monitor.time, "sleep"):
            code = monitor.supervise(["python", "worker"], receipt, log,
                inventory=inventory, idle=idle, launch=launch, own_pid=1)
        return code, json.loads(receipt.read_text()), log, worker

    def test_foreign_job_refuses_before_launch(self):
        code, receipt, log, child = self.execute(lambda: [row(9, 8)])
        self.assertEqual(code, 2)
        self.assertEqual(receipt["status"], "refused")
        self.assertIsNone(receipt["worker_exit_code"])
        self.assertFalse(receipt["root_waited"])
        self.assertFalse(log.exists())
        self.assertFalse(child.waited)

    def test_running_server_refuses_before_launch(self):
        code, receipt, log, _ = self.execute(lambda: [row(1, 0)], idle=lambda: False)
        self.assertEqual(code, 2)
        self.assertFalse(log.exists())
        self.assertEqual(receipt["status"], "refused")

    def test_own_worker_exit_recorded_and_log_closed(self):
        samples = iter([[row(1, 0)], [row(1, 0), row(2, 1), row(3, 2)], [row(1, 0)]])
        code, receipt, log, child = self.execute(lambda: next(samples))
        self.assertEqual(code, 0)
        self.assertTrue(child.waited)
        self.assertTrue(receipt["root_waited"])
        self.assertEqual(receipt["worker_exit_code"], 0)
        self.assertEqual(receipt["status"], "complete")
        self.assertTrue(receipt["observed_process_isolation"])
        self.assertTrue(receipt["worker_tree_closed"])
        self.assertEqual(log.read_bytes(), b"actual worker log\n")

    def test_launcher_wait_alone_cannot_claim_live_descendant_closure(self):
        samples = iter([[row(1, 0)], [row(1, 0), row(2, 1), row(3, 2)], [row(1, 0), row(3, 0)]])
        code, receipt, _, child = self.execute(lambda: next(samples))
        self.assertEqual(code, 4)
        self.assertEqual(receipt["status"], "worker-tree-still-active")
        self.assertFalse(receipt["worker_tree_closed"])
        self.assertEqual(receipt["worker_exit_code"], 0)
        self.assertTrue(child.waited)

    def test_mid_worker_external_job_retained_and_invalidates_timing(self):
        samples = iter([[row(1, 0)], [row(1, 0), row(9, 8)], [row(1, 0)]])
        code, receipt, _, child = self.execute(lambda: next(samples))
        self.assertEqual(code, 3)
        self.assertEqual(receipt["status"], "overlap-observed")
        self.assertEqual(receipt["worker_exit_code"], 0)
        self.assertTrue(child.waited)
        self.assertFalse(receipt["observed_process_isolation"])
        self.assertEqual(receipt["samples"][1]["foreign_jobs"], [row(9, 8)])

    def test_actual_nonzero_worker_exit_is_not_masked(self):
        code, receipt, _, child = self.execute(lambda: [row(1, 0)], child=Child(7))
        self.assertEqual(code, 7)
        self.assertEqual(receipt["worker_exit_code"], 7)
        self.assertEqual(receipt["status"], "failed")
        self.assertTrue(child.waited)

    def test_monitor_fault_still_waits_and_publishes_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt, log = Path(folder) / "receipt.json", Path(folder) / "log.txt"
            child = Child()
            samples = iter([[row(1, 0)]])
            def inventory():
                try:
                    return next(samples)
                except StopIteration:
                    raise OSError("enumeration failed")
            with self.assertRaisesRegex(OSError, "enumeration failed"):
                monitor.supervise([], receipt, log, inventory=inventory,
                    idle=lambda: True, launch=lambda *a, **k: child, own_pid=1)
            value = json.loads(receipt.read_text())
            self.assertEqual(value["status"], "supervisor-failed")
            self.assertTrue(value["root_waited"])
            self.assertEqual(value["worker_exit_code"], 0)
            self.assertTrue(child.waited)

    def test_secondary_wait_and_publication_faults_preserve_primary_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            primary = RuntimeError("enumeration primary")
            samples = iter([[row(1, 0)]])
            def inventory():
                try:
                    return next(samples)
                except StopIteration:
                    raise primary
            child = Child()
            with patch.object(child, "wait", side_effect=OSError("secondary wait")), \
                 patch.object(monitor.json, "dump", side_effect=OSError("secondary publication")):
                with self.assertRaises(RuntimeError) as caught:
                    monitor.supervise([], Path(folder)/"receipt.json", Path(folder)/"log.txt",
                        inventory=inventory, idle=lambda: True,
                        launch=lambda *a, **k: child, own_pid=1)
            self.assertIs(caught.exception, primary)
            self.assertIn("secondary wait", " ".join(primary.__notes__))
            self.assertIn("secondary publication", " ".join(primary.__notes__))

    def test_worker_failure_survives_receipt_and_diagnostic_publication_faults(self):
        for broken_stderr in (False, True):
            with self.subTest(broken_stderr=broken_stderr), tempfile.TemporaryDirectory() as folder:
                child = Child(7)
                stream = io.StringIO()
                with patch.object(monitor.json, "dump", side_effect=OSError("receipt write")), \
                     patch.object(monitor.sys, "stderr", stream), patch.object(monitor.time, "sleep"):
                    context = patch.object(stream, "write", side_effect=BrokenPipeError("stderr")) \
                        if broken_stderr else patch.object(stream, "write", wraps=stream.write)
                    with context:
                        code = monitor.supervise([], Path(folder)/"receipt.json", Path(folder)/"log.txt",
                            inventory=lambda: [row(1, 0)], idle=lambda: True,
                            launch=lambda *a, **k: child, own_pid=1)
                self.assertEqual(code, 7)
                self.assertTrue(child.waited)
                if not broken_stderr:
                    diagnostic = json.loads(stream.getvalue())
                    self.assertEqual(diagnostic["worker_exit_code"], 7)
                    self.assertTrue(diagnostic["root_waited"])


if __name__ == "__main__":
    unittest.main()
