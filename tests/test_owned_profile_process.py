"""Actual Windows ownership tests; no native scanner or numerical imports."""

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.research.owned_profile_process import run_owned


@unittest.skipUnless(sys.platform == "win32", "Windows job ownership contracts")
class OwnedProcessTests(unittest.TestCase):
    def run_case(self, code, timeout=10, on_started=None):
        with tempfile.TemporaryDirectory(prefix="owned-profile-contract-") as directory:
            folder = Path(directory)
            record = {}
            with (folder/"worker.log").open("wb") as stream:
                return run_owned([sys.executable, "-S", "-c", code], cwd=folder,
                    env=dict(os.environ), stream=stream, timeout=timeout,
                    record=record, on_started=on_started), record

    def test_success_waits_and_closes_owned_job(self):
        code, record = self.run_case('print("contract")')
        self.assertEqual(code, 0)
        self.assertTrue(record["owned_tree_closed"])
        self.assertTrue(record["child_wait_completed"])
        self.assertEqual(record["process_cleanup_failures"], [])

    def test_timeout_closes_real_grandchild_retained_handle(self):
        from ctypes import wintypes as w
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [w.DWORD,w.BOOL,w.DWORD], w.HANDLE
        kernel.WaitForSingleObject.argtypes, kernel.WaitForSingleObject.restype = [w.HANDLE,w.DWORD], w.DWORD
        kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [w.HANDLE],w.BOOL
        with tempfile.TemporaryDirectory(prefix="owned-tree-contract-") as directory:
            folder = Path(directory)
            child_file = folder/"grandchild.pid"
            code = ('import subprocess,sys,time,pathlib; '
                'p=subprocess.Popen([sys.executable,"-S","-c","import time; time.sleep(120)"]); '
                f'pathlib.Path({str(child_file)!r}).write_text(str(p.pid)); time.sleep(120)')
            record, retained = {}, []
            def observe():
                import time
                deadline = time.monotonic()+5
                while not child_file.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(child_file.exists(), "Worker did not spawn the actual grandchild")
                handle = kernel.OpenProcess(0x100000,False,int(child_file.read_text()))
                self.assertTrue(handle)
                retained.append(handle)
                self.assertEqual(kernel.WaitForSingleObject(handle,0),258)
            try:
                with (folder/"timeout.log").open("wb") as stream:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        run_owned([sys.executable,"-S","-c",code],cwd=folder,env=dict(os.environ),
                            stream=stream,timeout=.1,record=record,on_started=observe)
                self.assertEqual(len(retained),1)
                self.assertEqual(kernel.WaitForSingleObject(retained[0],1000),0)
                self.assertTrue(record["owned_tree_closed"])
                self.assertTrue(record["child_wait_completed"])
                self.assertEqual(record["process_cleanup_failures"],[])
            finally:
                for handle in retained:
                    self.assertTrue(kernel.CloseHandle(handle))

    def test_parent_reporting_fault_closes_worker(self):
        def fault():
            raise ValueError("report-write-fault")
        with self.assertRaisesRegex(ValueError,"report-write-fault"):
            self.run_case('import time; time.sleep(120)',on_started=fault)

    def test_job_api_fault_still_kills_and_waits_retained_unassigned_child(self):
        actions = []
        class FaultJob:
            def attach_and_resume(self, child):
                raise ValueError("assignment-fault")
            def active(self):
                raise OSError("query-fault")
            def close(self):
                actions.append("close-job")
        class Child:
            pid = 123
            def poll(self):
                return None
            def kill(self):
                actions.append("kill-retained")
            def wait(self, timeout):
                actions.append("wait-retained")
                return 1
        record = {}
        with patch("scripts.research.owned_profile_process.WindowsJob",return_value=FaultJob()), \
                patch("scripts.research.owned_profile_process.subprocess.Popen",return_value=Child()):
            with self.assertRaisesRegex(ValueError,"assignment-fault"):
                run_owned(["unused"],cwd=".",env={},stream=None,timeout=1,record=record)
        self.assertEqual(actions,["kill-retained","wait-retained","close-job"])
        self.assertTrue(record["child_wait_completed"])
        self.assertFalse(record["owned_tree_closed"])
        self.assertEqual(len(record["process_cleanup_failures"]),2)


if __name__ == "__main__":
    unittest.main()
