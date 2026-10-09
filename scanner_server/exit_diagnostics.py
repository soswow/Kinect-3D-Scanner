"""Exit reporting independent of the API, Open3D and logging configuration."""

from datetime import datetime, timezone
import faulthandler
import os
from pathlib import Path
import sys
import threading
import traceback


class ExitDiagnostics:
    def __init__(self, role="server", native_faults=False):
        self.role = role
        self.log = None
        self.previous_thread_hook = None
        directory = Path(os.environ.get("KINECT_LOG_DIR", "logs"))
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self.log = (directory / "server.lifecycle.log").open("a", encoding="utf-8", buffering=1)
        except (OSError, RuntimeError) as exc:
            self.record(f"Unable to open diagnostic files: {exc}; using stderr")
        if native_faults:
            # Keep the OS stderr descriptor active through interpreter teardown:
            # Open3D/native destructors can crash after main() has returned.
            # The detached launcher redirects this to server.stderr.log.
            if not faulthandler.is_enabled():
                try:
                    faulthandler.enable(file=sys.__stderr__, all_threads=True)
                except (OSError, RuntimeError, ValueError) as exc:
                    self.record(f"Unable to enable native fault diagnostics: {exc}")
            self.previous_thread_hook = threading.excepthook
            threading.excepthook = self.thread_exception

    def record(self, message):
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        line = f"{timestamp} pid={os.getpid()} {self.role}: {message}\n"
        # Flush each event so a subsequent native crash does not lose it.
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except (OSError, ValueError):
            pass
        if self.log is not None:
            try:
                self.log.write(line)
                self.log.flush()
            except (OSError, ValueError):
                pass

    def exception(self, message, exc):
        self.record(message + "\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    def thread_exception(self, args):
        self.exception(f"Unhandled exception in thread {args.thread.name}; process may still be running",
                       args.exc_value)
        self.previous_thread_hook(args)

    def close(self):
        if self.previous_thread_hook is not None:
            threading.excepthook = self.previous_thread_hook
        if self.log is not None:
            self.log.close()


def describe_exit(code, platform=sys.platform):
    if code == 0:
        return "normal process exit"
    if platform != "win32" and code < 0:
        import signal
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = str(-code)
        return f"terminated by signal {name}"
    if platform == "win32":
        status = code & 0xFFFFFFFF
        reasons = {
            0xC0000005: "native access violation",
            0xC0000017: "native out of memory",
            0xC000013A: "console control interruption",
            0xC0000409: "native fail-fast / stack buffer overrun",
        }
        reason = reasons.get(status, "nonzero process exit; cause not identified by exit code alone")
        return f"{reason} (Windows status 0x{status:08X})"
    return "nonzero process exit; see server traceback and fault logs"
