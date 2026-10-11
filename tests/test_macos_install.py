"""Bundle replacement must not corrupt live clients or lose the previous app."""

import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import install_macos_client as installer


class MacInstallTests(unittest.TestCase):
    def setUp(self):
        self.real_run = subprocess.run
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "dist/Kinect 3D Scanner.app"
        self.destination = self.root / "Applications/Kinect 3D Scanner.app"
        for bundle, content in ((self.source, b"new"), (self.destination, b"previous")):
            binary = bundle / installer.EXECUTABLE
            binary.parent.mkdir(parents=True)
            binary.write_bytes(content)
            resources = bundle / "Contents/Resources"
            resources.mkdir()
            (resources / "calibration.bin").write_bytes(content + b"-resource")
            (resources / "alias").symlink_to("calibration.bin")

    def shell(self, args, **kwargs):
        if args[0] == "/usr/bin/ditto":
            shutil.copytree(args[1], args[2], symlinks=True)
        elif args[0] == "/usr/bin/diff":
            return self.real_run(args, **kwargs)
        else:
            self.fail(f"Unexpected command: {args}")
        return SimpleNamespace(returncode=0)

    def install(self):
        with patch.object(installer.subprocess, "run", side_effect=self.shell), contextlib.redirect_stdout(io.StringIO()):
            installer.install_bundle(self.source, self.destination)

    def test_running_client_or_helper_blocks_before_touching_bundle(self):
        binary = self.destination / installer.EXECUTABLE
        for suffix in ("", " --viewer {}", " -B -S -I -c multiprocessing"):
            with self.subTest(suffix=suffix), \
                    patch.object(installer.subprocess, "run", return_value=SimpleNamespace(stdout=f"123 {binary}{suffix}\n")):
                with self.assertRaisesRegex(RuntimeError, "PIDs: 123"):
                    installer.install_bundle(self.source, self.destination)
                self.assertEqual(binary.read_bytes(), b"previous")
                self.assertEqual(list(self.destination.parent.glob(".kinect-scanner-install-*")), [])

    def test_process_matching_ignores_other_bundles_and_non_executable_arguments(self):
        commands = (f"42 /other/Kinect 3D Scanner.app/Contents/MacOS/Kinect 3D Scanner\n"
                    f"43 /bin/echo {self.destination}/Contents/MacOS/Kinect 3D Scanner\n"
                    f"44 {self.destination}-other/Contents/MacOS/Kinect 3D Scanner\n")
        with patch.object(installer.subprocess, "run", return_value=SimpleNamespace(stdout=commands)):
            installer.require_stopped(self.destination)

    def test_process_snapshot_failure_leaves_bundle_untouched(self):
        with patch.object(installer.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "ps")):
            with self.assertRaises(subprocess.CalledProcessError):
                installer.install_bundle(self.source, self.destination)
        self.assertEqual((self.destination / installer.EXECUTABLE).read_bytes(), b"previous")

    def test_client_launched_during_staging_blocks_replacement(self):
        with patch.object(installer, "require_stopped", side_effect=[None, RuntimeError("client launched")]), \
                patch.object(installer, "check_bundle"):
            with self.assertRaisesRegex(RuntimeError, "client launched"):
                self.install()
        self.assertEqual((self.destination / installer.EXECUTABLE).read_bytes(), b"previous")
        self.assertEqual(list(self.destination.parent.glob(".kinect-scanner-install-*")), [])

    def test_staged_check_failure_preserves_previous_bundle(self):
        with patch.object(installer, "require_stopped"), \
                patch.object(installer, "check_bundle", side_effect=RuntimeError("check failed")):
            with self.assertRaisesRegex(RuntimeError, "check failed"):
                self.install()
        self.assertEqual((self.destination / installer.EXECUTABLE).read_bytes(), b"previous")

    def test_installed_check_failure_restores_entire_previous_bundle(self):
        with patch.object(installer, "require_stopped"), \
                patch.object(installer, "check_bundle", side_effect=[None, RuntimeError("installed check failed")]):
            with self.assertRaisesRegex(RuntimeError, "installed check failed"):
                self.install()
        self.assertEqual((self.destination / installer.EXECUTABLE).read_bytes(), b"previous")
        self.assertEqual((self.destination / "Contents/Resources/alias").read_bytes(), b"previous-resource")
        self.assertEqual(list(self.destination.parent.glob(".kinect-scanner-install-*")), [])

    def test_changed_executable_fails_verification_and_preserves_previous_bundle(self):
        for installed in (False, True):
            def corrupt(bundle):
                if (bundle == self.destination) == installed:
                    (bundle / installer.EXECUTABLE).write_bytes(b"unexpected replacement")
            with self.subTest(installed=installed), patch.object(installer, "require_stopped"), \
                    patch.object(installer, "check_bundle", side_effect=corrupt):
                with self.assertRaisesRegex(RuntimeError, "executable differs"):
                    self.install()
                self.assertEqual((self.destination / installer.EXECUTABLE).read_bytes(), b"previous")


    def test_success_checks_both_locations_and_retains_previous_bundle(self):
        with patch.object(installer, "require_stopped") as stopped, \
                patch.object(installer, "check_bundle") as checked:
            self.install()
        self.assertEqual(stopped.call_count, 2)
        self.assertEqual(checked.call_count, 2)
        self.assertEqual(checked.call_args.args, (self.destination,))
        self.assertEqual((self.destination / installer.EXECUTABLE).read_bytes(), b"new")
        backup, = self.destination.parent.glob(".kinect-scanner-install-*/previous.app")
        self.assertEqual((backup / installer.EXECUTABLE).read_bytes(), b"previous")
        self.assertTrue((self.destination / "Contents/Resources/alias").is_symlink())

    def test_fresh_install_and_failed_fresh_install(self):
        shutil.rmtree(self.destination)
        with patch.object(installer, "require_stopped"), patch.object(installer, "check_bundle"):
            self.install()
        self.assertEqual(list(self.destination.parent.glob(".kinect-scanner-install-*")), [])
        shutil.rmtree(self.destination)
        with patch.object(installer, "require_stopped"), \
                patch.object(installer, "check_bundle", side_effect=[None, RuntimeError("failed")]):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                self.install()
        self.assertFalse(self.destination.exists())

    def test_overlapping_paths_and_symlink_destination_are_rejected(self):
        for path in (self.source, self.source / "nested.app", self.source.parent):
            with self.subTest(path=path), self.assertRaises(ValueError):
                installer.install_bundle(self.source, path)
        link = self.root / "symlink.app"
        link.symlink_to(self.destination)
        with self.assertRaises(ValueError):
            installer.install_bundle(self.source, link)

    def test_installer_lock_rejects_concurrent_installation(self):
        with installer.install_lock(self.destination), self.assertRaisesRegex(RuntimeError, "in progress"):
            installer.install_bundle(self.source, self.destination)


if __name__ == "__main__":
    unittest.main()
