"""The real relocated comparator must close preflight failures without native imports."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research/compare_checkpoint_resident_finishes.py"


class CheckpointComparatorCliContracts(unittest.TestCase):
    def test_real_cli_hashes_its_path_and_saves_actionable_preflight_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            native, audit, output = (folder/name for name in ("native.json", "audit.json", "quality.json"))
            native.write_text("{}\n", encoding="utf-8")
            audit.write_text('{"artifacts_sha256":{}}\n', encoding="utf-8")
            before = {path: path.read_bytes() for path in (native, audit)}
            result = subprocess.run([sys.executable, "-S", str(SCRIPT), str(native), str(audit),
                                     "--output", str(output), "--run-allocated"], cwd=folder,
                                    capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Incomplete current quality producer pins", result.stderr)
            self.assertNotIn("has no attribute", result.stderr)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "failed")
            self.assertEqual(value["comparison_artifact_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
            self.assertIn("Incomplete current quality producer pins", value["failure"]["message"])
            self.assertEqual(value["native_sidecar"]["sha256"], hashlib.sha256(before[native]).hexdigest())
            self.assertEqual(value["audit_report_sha256"], hashlib.sha256(before[audit]).hexdigest())
            self.assertEqual({path: path.read_bytes() for path in before}, before)
            self.assertEqual(set(path.name for path in folder.iterdir()), {"native.json", "audit.json", "quality.json"})

    def test_help_from_unrelated_directory_needs_no_site_or_numerical_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], cwd=folder,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
