"""Run one frozen candidate worker and sample external Windows job overlap.

This supervisor imports no numerical libraries and changes no worker code.
Its receipt is process-isolation evidence, never a timing/quality permission.
Other work is observed, never killed. Sampling cannot rule out shorter jobs
between snapshots or GPU activity in arbitrary unrecognized executables.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import datetime as dt
import hashlib
import json
from pathlib import Path
import os
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts/research/profile_gpu_icp_finish_candidate.py"


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def fresh_private(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / "benchmark-output") or path.exists():
        raise ValueError("Require fresh private output under benchmark-output")
    return path


def server_idle():
    with socket.socket() as probe:
        probe.settimeout(.2)
        return probe.connect_ex(("127.0.0.1", 8000)) != 0


def windows_processes():
    """Toolhelp enumeration plus creation ticks protects against PID reuse."""
    if sys.platform != "win32":
        raise RuntimeError("This external supervisor requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32FirstW.restype = wintypes.BOOL
    kernel.Process32NextW.argtypes = kernel.Process32FirstW.argtypes
    kernel.Process32NextW.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    result = []
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(entry)
        present = kernel.Process32FirstW(handle, ctypes.byref(entry))
        if not present:
            raise ctypes.WinError(ctypes.get_last_error())
        while present:
            name = entry.szExeFile.lower()
            if name.startswith("python") or name in ("cl.exe", "nvcc.exe", "kinect-3d-scanner.exe"):
                process = kernel.OpenProcess(0x1000, False, entry.th32ProcessID)
                created = None
                if process:
                    try:
                        stamps = [wintypes.FILETIME() for _ in range(4)]
                        if kernel.GetProcessTimes(process, *(ctypes.byref(s) for s in stamps)):
                            created = stamps[0].dwLowDateTime + (stamps[0].dwHighDateTime << 32)
                    finally:
                        kernel.CloseHandle(process)
                result.append({"pid": int(entry.th32ProcessID),
                    "parent_pid": int(entry.th32ParentProcessID), "name": name,
                    "created_100ns": created})
            present = kernel.Process32NextW(handle, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)
    return result


def classify(rows, own_pid, worker_pid, known):
    """Learn descendants while their parent is observed; never reuse a PID."""
    owned = {own_pid}
    if worker_pid is not None:
        owned.add(worker_pid)
    for row in rows:
        if row["created_100ns"] is not None and (row["pid"], row["created_100ns"]) in known:
            owned.add(row["pid"])
    changed = True
    generations = {row["pid"]: row["created_100ns"] for row in rows}
    while changed:
        changed = False
        for row in rows:
            parent_created = generations.get(row["parent_pid"])
            if (row["pid"] not in owned and row["parent_pid"] in owned
                    and parent_created is not None and row["created_100ns"] is not None
                    and row["created_100ns"] >= parent_created):
                owned.add(row["pid"])
                changed = True
    for row in rows:
        if row["pid"] in owned and row["created_100ns"] is not None:
            known.add((row["pid"], row["created_100ns"]))
    foreign = [row for row in rows if row["pid"] not in owned]
    unknown = [row for row in rows if row["created_100ns"] is None]
    return foreign, unknown


def supervise(command, receipt_path, log_path, *, interval=.5, inventory=windows_processes,
              idle=server_idle, launch=subprocess.Popen, own_pid=None):
    own_pid = os.getpid() if own_pid is None else own_pid
    known = set()
    source_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"kind": "gpu-icp-external-worker-observation-v1", "started_utc": utc(),
        "command": command, "supervisor_pid": own_pid, "poll_interval_s": interval,
        "status": "preflight", "worker_exit_code": None, "root_waited": False,
        "samples": [], "observed_process_isolation": False, "worker_tree_closed": False,
        "supervisor_source_sha256": source_before, "cleanup_failures": [],
        "scope": "Sampled named Windows jobs and port 8000; no all-process GPU exclusivity, no worker quality or timing authority"}
    child = None
    primary = None
    try:
        def sample():
            rows = inventory()
            worker_pid = child.pid if child is not None and child.poll() is None else None
            foreign, unknown = classify(rows, own_pid, worker_pid, known)
            value = {"utc": utc(), "server_idle": idle(), "processes": rows,
                     "foreign_jobs": foreign, "unidentified_creation_times": unknown,
                     "owned_jobs": [r for r in rows if r not in foreign and r["pid"] != own_pid]}
            report["samples"].append(value)
            return not foreign and not unknown and value["server_idle"]
        if not sample():
            report["status"] = "refused"
            return 2
        with open(log_path, "xb") as log:
            child = launch(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            report.update(worker_pid=child.pid, status="running", worker_started_utc=utc())
            while child.poll() is None:
                sample()
                time.sleep(interval)
            code = child.wait()
            report.update(worker_exit_code=code, root_waited=True, worker_ended_utc=utc())
            sample()
        clean = all(not s["foreign_jobs"] and not s["unidentified_creation_times"] and s["server_idle"]
                    for s in report["samples"])
        tree_closed = not report["samples"][-1]["owned_jobs"]
        report.update(status="failed" if code != 0 else "worker-tree-still-active" if not tree_closed
                      else "complete" if clean else "overlap-observed",
                      observed_process_isolation=clean, worker_tree_closed=tree_closed)
        return code if code != 0 else 4 if not tree_closed else 0 if clean else 3
    except BaseException as error:
        primary = error
        report.update(status="supervisor-failed", failure={"type": type(error).__name__, "message": str(error)})
        # Never orphan a numerical worker on a monitoring/publication fault.
        if child is not None:
            try:
                report.update(worker_exit_code=child.wait(), root_waited=True, worker_ended_utc=utc())
            except BaseException as wait_error:
                report["cleanup_failures"].append({"action": "worker wait", "type": type(wait_error).__name__,
                                                   "message": str(wait_error)})
                error.add_note("Worker wait failed: " + repr(wait_error))
        raise
    finally:
        report["ended_utc"] = utc()
        try:
            report["supervisor_source_sha256_after"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            if report["supervisor_source_sha256_after"] != source_before:
                raise RuntimeError("Supervisor source changed during observation")
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(receipt_path, "x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
        except BaseException as publication_error:
            if primary is not None:
                primary.add_note("Supervisor receipt publication failed: " + repr(publication_error))
            elif report["worker_exit_code"] not in (None, 0):
                # Preserve the actual worker failure code even if its receipt
                # cannot be saved. The calling root receives this explicit fact.
                try:
                    sys.stderr.write(json.dumps({"worker_exit_code": report["worker_exit_code"],
                        "root_waited": report["root_waited"], "receipt_publication_failure": repr(publication_error)}) + "\n")
                except BaseException:
                    pass  # A broken diagnostic stream cannot replace worker failure.
            else:
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    receipt, log = fresh_private(args.receipt), fresh_private(args.log)
    if receipt == log or not args.python.resolve(strict=True).is_file():
        parser.error("Distinct private receipt/log and existing Python required")
    worker_args = args.worker_args[1:] if args.worker_args[:1] == ["--"] else args.worker_args
    if not worker_args:
        parser.error("Supply the candidate worker arguments after --")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.python.resolve()), "-s", "-u", str(WORKER)] + worker_args
    return supervise(command, receipt, log)


if __name__ == "__main__":
    raise SystemExit(main())
