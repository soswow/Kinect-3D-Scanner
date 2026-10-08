"""Independent startup policy, configuration failures and frozen helper dispatch."""

import contextlib
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scanner_server.__main__ import configure, main
from kinect_scanner import runtime


class ServerStartupTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_defaults_do_not_require_cuda_or_native_extension(self):
        args = configure([])
        self.assertEqual((args.device, args.tracking, args.native), ("auto", "auto", "auto"))
        self.assertEqual((args.port, args.threads, args.block_count), (8000, 4, 5000))

    def test_cli_overrides_environment_before_api_import(self):
        os.environ.update(KINECT_DEVICE="cuda", KINECT_SERVER_PORT="8001")
        calls = []
        def run(app, **kwargs):
            calls.append((app, kwargs, os.environ.copy()))
        with patch.dict(sys.modules, {"uvicorn": SimpleNamespace(run=run)}):
            self.assertEqual(main(["--device", "cpu", "--port", "8002", "--threads", "2"]), 0)
        self.assertEqual(calls[0][1], {"host": "0.0.0.0", "port": 8002})
        self.assertEqual(calls[0][2]["KINECT_DEVICE"], "cpu")
        self.assertEqual(calls[0][2]["OMP_NUM_THREADS"], "2")

    def test_bad_configuration_fails_before_importing_server(self):
        for arguments in (["--port", "0"], ["--port", "65536"], ["--threads", "-1"],
                          ["--block-count", "oops"], ["--host", ""], ["--device", "metal"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    configure(arguments)
                self.assertEqual(error.exception.code, 2)
        for variable, value in (("KINECT_DEVICE", "metal"), ("KINECT_SERVER_PORT", "bad"),
                                ("OMP_NUM_THREADS", "0")):
            with self.subTest(variable=variable), patch.dict(os.environ, {variable: value}), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    configure([])

    def test_help_needs_no_processing_dependencies(self):
        with patch.dict(sys.modules, {"uvicorn": None}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--help"])
        self.assertEqual(error.exception.code, 0)

    def test_dependency_error_returns_failure(self):
        with patch.dict(sys.modules, {"uvicorn": None}), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main([]), 1)


class ClientRuntimeTests(unittest.TestCase):
    def test_bundle_outputs_stay_outside_read_only_resources(self):
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "platform", "darwin"), \
                patch.object(Path, "home", return_value=Path("/Users/scanner")):
            self.assertEqual(runtime.data_root(), Path("/Users/scanner/Library/Application Support/Kinect3DScanner"))
            self.assertEqual(runtime.export_root(), Path("/Users/scanner/Documents/Kinect 3D Scanner/export"))
            self.assertEqual(runtime.log_path(), Path("/Users/scanner/Library/Logs/Kinect3DScanner/client.log"))

    def test_preview_reuses_frozen_executable_with_helper_dispatch(self):
        with patch.object(sys, "frozen", True, create=True):
            self.assertEqual(runtime.viewer_command('{"filepath":"a b.ply"}'),
                             [sys.executable, "--viewer", '{"filepath":"a b.ply"}'])
        with patch.object(sys, "frozen", False, create=True):
            self.assertEqual(runtime.viewer_command("{}"),
                             [sys.executable, "-m", "kinect_scanner.viewer", "{}"])

    def test_installed_package_uses_user_data_not_site_packages(self):
        with patch.object(sys, "frozen", False, create=True), \
                patch.object(Path, "is_file", return_value=False):
            self.assertNotEqual(runtime.data_root(), Path(runtime.__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
