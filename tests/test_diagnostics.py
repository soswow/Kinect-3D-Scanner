import logging
import tempfile
import unittest
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shared.diagnostics import ResourceMonitor, configure_logging


class DiagnosticsTests(unittest.TestCase):
    def test_disk_warning_reports_threshold_crossings_without_repeating_samples(self):
        monitor = ResourceMonitor("/tmp")
        samples = [500, 400, 2048, 300, 200]
        with patch("shared.diagnostics.shutil.disk_usage", side_effect=[
                SimpleNamespace(free=value * 1024**2) for value in samples]), \
                self.assertLogs("shared.diagnostics", level="INFO") as logs:
            for _ in samples:
                monitor.sample()
        self.assertEqual(2, sum("Low disk space" in message for message in logs.output))
        self.assertEqual(1, sum("Disk space recovered" in message for message in logs.output))
        self.assertEqual(5, sum("Resources" in message for message in logs.output))

    def test_persistent_timestamps_traceback_and_resources(self):
        root = logging.getLogger()
        old_handlers, old_level = list(root.handlers), root.level
        with tempfile.TemporaryDirectory() as directory:
            try:
                path = (Path(directory) / "client.log").resolve()
                configure_logging(path)
                configure_logging(path)
                matching = [h for h in root.handlers if isinstance(h, RotatingFileHandler)
                            and h.baseFilename == str(path)]
                self.assertEqual(len(matching), 1)
                self.assertEqual(matching[0].backupCount, 3)
                try:
                    raise RuntimeError("simulated build failure")
                except RuntimeError:
                    logging.getLogger(__name__).exception("Build failed session=test")
                sample = ResourceMonitor(directory).sample()
                self.assertGreaterEqual(sample["disk_free_mib"], 0)
                contents = path.read_text()
                self.assertRegex(contents, r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}Z pid=")
                self.assertIn("Traceback", contents)
                self.assertIn("session=test", contents)
                self.assertIn("Resources", contents)
            finally:
                for handler in list(root.handlers):
                    if handler not in old_handlers:
                        root.removeHandler(handler)
                        handler.close()
                root.setLevel(old_level)
