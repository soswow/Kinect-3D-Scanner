"""Exit causes survive normal return, exceptions and externally killed children."""

import contextlib
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scanner_server.__main__ import main
from scanner_server.exit_diagnostics import ExitDiagnostics, describe_exit
from scanner_server import supervisor


class ServerExitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        environment = patch.dict(os.environ, {"KINECT_LOG_DIR": self.directory.name})
        environment.start()
        self.addCleanup(environment.stop)
        self.log = Path(self.directory.name) / "server.lifecycle.log"

    def run_server(self, action):
        class Server:
            def __init__(self, config):
                self.started = True
            def run(self):
                action(self)
            def handle_exit(self, sig, frame):
                self.should_exit = True
        with patch.dict(sys.modules, {"uvicorn": SimpleNamespace(Config=lambda *a, **k: None, Server=Server)}), \
                contextlib.redirect_stderr(io.StringIO()):
            return main([])

    def test_normal_return_and_signal_request_are_logged_without_changing_shutdown(self):
        def stop(server):
            server.handle_exit(signal.SIGTERM, None)
            self.assertTrue(server.should_exit)
        self.assertEqual(self.run_server(stop), 0)
        contents = self.log.read_text(encoding="utf-8")
        self.assertIn("Shutdown requested by SIGTERM", contents)
        self.assertIn("reason=SIGTERM; exit_code=0", contents)

    def test_exception_keeps_full_traceback_in_persistent_log(self):
        def fail(server):
            raise RuntimeError("deliberate server failure")
        self.assertEqual(self.run_server(fail), 1)
        contents = self.log.read_text(encoding="utf-8")
        self.assertIn("Traceback (most recent call last)", contents)
        self.assertIn("RuntimeError: deliberate server failure", contents)
        self.assertIn("exit_code=1", contents)

    def test_system_exit_is_logged_and_preserves_exit_code(self):
        def fail(server):
            raise SystemExit(3)
        with self.assertRaises(SystemExit) as caught:
            self.run_server(fail)
        self.assertEqual(caught.exception.code, 3)
        self.assertIn("Server raised SystemExit; exit_code=3", self.log.read_text(encoding="utf-8"))

    def test_startup_failure_is_not_reported_as_success(self):
        self.assertEqual(self.run_server(lambda server: setattr(server, "started", False)), 3)
        self.assertIn("before startup completed", self.log.read_text(encoding="utf-8"))

    def test_supervisor_reports_external_termination_even_without_child_log(self):
        # A real process dies outside Python's exception/finally handling. Use a
        # tiny child instead of starting Open3D or allocating a scan volume.
        original_popen = subprocess.Popen
        def launch(command, **kwargs):
            child = original_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            child.kill()
            return child
        with patch.object(supervisor.subprocess, "Popen", side_effect=launch), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(supervisor.main([]), 1)
        contents = self.log.read_text(encoding="utf-8")
        self.assertIn("Server child pid=", contents)
        self.assertIn("exited: code=", contents)
        self.assertNotIn("normal process exit", contents)

    def test_supervisor_records_cli_failure_from_real_server_command(self):
        result = subprocess.run([sys.executable, "-m", "scanner_server.supervisor", "--port", "0"],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 1)
        self.assertIn("exited: code=2", self.log.read_text(encoding="utf-8"))

    def test_real_uvicorn_graceful_shutdown_records_signal_and_completion(self):
        # A minimal lifespan app exercises Uvicorn's real signal handling and
        # signal replay without importing the reconstruction engine.
        code = '''
import asyncio, signal, sys, types
from scanner_server.__main__ import main
async def app(scope, receive, send):
    if scope['type'] != 'lifespan':
        return
    assert (await receive())['type'] == 'lifespan.startup'
    await send({'type': 'lifespan.startup.complete'})
    asyncio.get_running_loop().call_later(.2, signal.raise_signal, signal.SIGINT)
    assert (await receive())['type'] == 'lifespan.shutdown'
    await send({'type': 'lifespan.shutdown.complete'})
sys.modules['scanner_server.app'] = types.SimpleNamespace(app=app)
sys.exit(main(['--port', '54321', '--host', '127.0.0.1']))
'''
        # Choose a free port immediately before starting the isolated process.
        import socket
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            code = code.replace("54321", str(probe.getsockname()[1]))
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        contents = self.log.read_text(encoding="utf-8")
        self.assertIn("Shutdown requested by SIGINT", contents)
        self.assertIn("Shutdown completed; reason=SIGINT", contents)


    def test_fault_handler_stays_enabled_for_native_interpreter_teardown(self):
        with patch("scanner_server.exit_diagnostics.faulthandler.is_enabled", return_value=False), \
                patch("scanner_server.exit_diagnostics.faulthandler.enable") as enable, \
                patch("scanner_server.exit_diagnostics.faulthandler.disable") as disable:
            diagnostics = ExitDiagnostics(native_faults=True)
            enable.assert_called_once_with(file=sys.__stderr__, all_threads=True)
            diagnostics.close()
            disable.assert_not_called()

    def test_log_files_append_across_restarts(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for message in ("first run", "second run"):
                diagnostics = ExitDiagnostics()
                diagnostics.record(message)
                diagnostics.close()
        contents = self.log.read_text(encoding="utf-8")
        self.assertIn("first run", contents)
        self.assertIn("second run", contents)


class ExitCodeTests(unittest.TestCase):
    def test_native_windows_fault_and_signed_status_are_identified(self):
        for status in (0xC0000005, -1073741819):
            self.assertIn("native access violation", describe_exit(status, "win32"))
            self.assertIn("0xC0000005", describe_exit(status, "win32"))

    def test_unknown_exit_does_not_invent_a_cause(self):
        self.assertIn("cause not identified", describe_exit(1, "win32"))
        self.assertIn("terminated by signal", describe_exit(-9, "linux"))


if __name__ == "__main__":
    unittest.main()
