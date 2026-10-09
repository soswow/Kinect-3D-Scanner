"""Keep observing a detached server after its launcher has returned."""

import os
import signal
import subprocess
import sys

from .exit_diagnostics import ExitDiagnostics, describe_exit


def main(argv=None):
    diagnostics = ExitDiagnostics(role="supervisor")
    child = None
    try:
        arguments = list(sys.argv[1:] if argv is None else argv)
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {}
        child = subprocess.Popen([sys.executable, "-m", "scanner_server", *arguments], **options)
        diagnostics.record(f"Started server child pid={child.pid}; cwd={os.getcwd()}")
        try:
            code = child.wait()
        except KeyboardInterrupt:
            diagnostics.record(f"Supervisor interrupted; requesting shutdown of child pid={child.pid}")
            if child.poll() is None:
                child.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
            try:
                code = child.wait(timeout=30)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                diagnostics.record(f"Child pid={child.pid} did not stop; supervisor forcing termination")
                child.kill()
                code = child.wait()
        diagnostics.record(f"Server child pid={child.pid} exited: code={code}; {describe_exit(code)}")
        return 0 if code == 0 else 1
    except Exception as exc:
        diagnostics.exception("Supervisor failed", exc)
        if child is not None and child.poll() is None:
            diagnostics.record(f"Stopping child pid={child.pid} after supervisor failure")
            child.kill()
            child.wait()
        return 1
    finally:
        diagnostics.close()


if __name__ == "__main__":
    sys.exit(main())
