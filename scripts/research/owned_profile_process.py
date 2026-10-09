"""Run an isolated profile worker and close its entire owned process tree.

Windows venv executables can launch another Python process. A retained job
handle owns the launcher and descendants; no PID-based termination is used.
The launcher starts suspended and joins the job before it can spawn a child.
"""

from __future__ import annotations

import subprocess
import sys
import time


class WindowsJob:
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w
        self.c, self.w = c, w
        self.kernel = k = c.WinDLL("kernel32", use_last_error=True)
        class BasicLimits(c.Structure):
            _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64),
                ("flags", w.DWORD), ("min_working_set", c.c_size_t),
                ("max_working_set", c.c_size_t), ("active_limit", w.DWORD),
                ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class IoCounters(c.Structure):
            _fields_ = [(name, c.c_uint64) for name in (
                "read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class ExtendedLimits(c.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IoCounters)] + [
                (name, c.c_size_t) for name in ("process_memory", "job_memory", "peak_process", "peak_job")]
        class Accounting(c.Structure):
            _fields_ = [(name, c.c_int64) for name in ("user", "kernel", "period_user", "period_kernel")] + [
                (name, w.DWORD) for name in ("faults", "total", "active", "terminated")]
        self.Accounting = Accounting
        for name, arguments, result in (
            ("CreateJobObjectW", [c.c_void_p, w.LPCWSTR], w.HANDLE),
            ("SetInformationJobObject", [w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            ("QueryInformationJobObject", [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL),
            ("AssignProcessToJobObject", [w.HANDLE, w.HANDLE], w.BOOL),
            ("TerminateJobObject", [w.HANDLE, w.UINT], w.BOOL),
            ("CloseHandle", [w.HANDLE], w.BOOL),
            ("CreateToolhelp32Snapshot", [w.DWORD, w.DWORD], w.HANDLE),
            ("OpenThread", [w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            ("GetProcessIdOfThread", [w.HANDLE], w.DWORD),
            ("ResumeThread", [w.HANDLE], w.DWORD)):
            method = getattr(k, name)
            method.argtypes, method.restype = arguments, result
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            self.check(k.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)))
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                primary.add_note(f"Partial job construction cleanup also failed: {cleanup}")
            raise

    def check(self, result):
        if not result:
            raise self.c.WinError(self.c.get_last_error())

    def attach_and_resume(self, child):
        c, w, k = self.c, self.w, self.kernel
        self.check(k.AssignProcessToJobObject(self.handle, int(child._handle)))
        class ThreadEntry(c.Structure):
            _fields_ = [(name, w.DWORD) for name in ("size", "usage", "tid", "pid")] + [
                ("base_priority", w.LONG), ("delta_priority", w.LONG), ("flags", w.DWORD)]
        k.Thread32First.argtypes = k.Thread32Next.argtypes = [w.HANDLE, c.POINTER(ThreadEntry)]
        k.Thread32First.restype = k.Thread32Next.restype = w.BOOL
        snapshot = k.CreateToolhelp32Snapshot(4, 0)
        if snapshot == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())
        tids = []
        try:
            entry = ThreadEntry()
            entry.size = c.sizeof(entry)
            self.check(k.Thread32First(snapshot, c.byref(entry)))
            while True:
                if entry.pid == child.pid:
                    tids.append(entry.tid)
                if not k.Thread32Next(snapshot, c.byref(entry)):
                    if c.get_last_error() != 18:  # ERROR_NO_MORE_FILES
                        raise c.WinError(c.get_last_error())
                    break
        finally:
            self.check(k.CloseHandle(snapshot))
        if len(tids) != 1:
            raise RuntimeError("Suspended newborn profile process must have exactly one thread")
        thread = k.OpenThread(0x802, False, tids[0])  # QUERY_LIMITED_INFORMATION | SUSPEND_RESUME
        if not thread:
            raise c.WinError(c.get_last_error())
        try:
            if k.GetProcessIdOfThread(thread) != child.pid:
                raise RuntimeError("Retained newborn thread no longer belongs to the owned child")
            if k.ResumeThread(thread) != 1:
                raise RuntimeError("Newborn profile thread did not have the expected suspension count")
        finally:
            self.check(k.CloseHandle(thread))

    def active(self):
        value = self.Accounting()
        self.check(self.kernel.QueryInformationJobObject(self.handle, 1,
            self.c.byref(value), self.c.sizeof(value), None))
        return value.active

    def terminate(self):
        self.check(self.kernel.TerminateJobObject(self.handle, 1))

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            self.check(self.kernel.CloseHandle(handle))


def run_owned(command, *, cwd, env, stream, timeout, record, on_started=None):
    """Wait for the worker and verify no descendants remain; record cleanup.

    A timeout or interruption terminates only members of the retained Windows
    job. Ownership begins before the worker executes user code. Other platforms
    fail before launching until their descendant ownership is implemented.
    """
    if sys.platform != "win32":
        raise NotImplementedError("This isolated profile launcher requires Windows job ownership")
    child = None
    job = WindowsJob()
    primary = None
    try:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
            stderr=subprocess.STDOUT, creationflags=4)
        record.update(child_pid=child.pid, process_ownership="windows-kill-on-close-job")
        job.attach_and_resume(child)
        if on_started:
            on_started()
        code = child.wait(timeout=timeout)
        if job.active():
            raise RuntimeError("Profile launcher exited while an owned descendant remained")
        record.update(child_wait_completed=True, owned_tree_closed=True, exit_code=code)
        return code
    except BaseException as error:
        primary = error
        raise
    finally:
        errors, verified = [], False
        record["owned_tree_closed"] = False
        try:
            if job.active():
                job.terminate()
        except BaseException as error:
            errors.append(error)
        # Never skip the retained direct-child handle after a job API fault.
        # This also covers failed assignment of a still-suspended nonmember.
        try:
            if child and child.poll() is None:
                child.kill()  # retained Popen handle, not a looked-up PID
        except BaseException as error:
            errors.append(error)
        try:
            if child:
                child.wait(timeout=15)
                record["child_wait_completed"] = True
        except BaseException as error:
            errors.append(error)
        try:
            deadline = time.monotonic() + 15
            while job.active() and time.monotonic() < deadline:
                time.sleep(.01)
            if job.active():
                raise RuntimeError("Owned profile process tree did not terminate")
            verified = True
        except BaseException as error:
            errors.append(error)
        try:
            job.close()
        except BaseException as error:
            errors.append(error)
        record["owned_tree_closed"] = verified and not errors
        record["process_cleanup_failures"] = [{"type":type(error).__name__, "message":str(error)} for error in errors]
        if errors:
            if primary:
                for error in errors:
                    primary.add_note(f"Owned process cleanup also failed: {error}")
            else:
                raise errors[0]
