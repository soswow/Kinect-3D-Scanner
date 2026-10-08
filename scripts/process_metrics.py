"""Peak resident process memory on Windows, macOS and Linux."""

import sys


def gpu_info():
    """Record installed GPU/driver identity outside measured processing."""
    import csv
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return [{"name": row[0].strip(), "driver": row[1].strip()}
            for row in csv.reader(result.stdout.splitlines()) if len(row) == 2]


def finish_cuda_worker():
    """Exit a completed Windows CUDA profiling worker without DLL finalizers.

    Call only after device synchronization and after closing all artifacts.
    Windows CRT exit still invokes DLL detach routines; the preview CUDA wheel
    can free static GPU allocations after its driver DLL has detached. The OS
    releases this isolated worker's resources when it terminates.
    """
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    sys.stdout.flush()
    sys.stderr.flush()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateProcess.restype = wintypes.BOOL
    if not kernel.TerminateProcess(kernel.GetCurrentProcess(), 0):
        raise ctypes.WinError(ctypes.get_last_error())


def peak_rss_bytes():
    if sys.platform != "win32":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak * (1 if sys.platform == "darwin" else 1024)

    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize
