"""Stdlib fake-fixture preflight, byte restoration and rollback contracts."""
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.research import restore_measured_source as helper


class ReproducibilityContracts(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self, *, measured=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            originals = {"scanner_server/a.py": b"alpha\nbeta\n",
                         "shared/b.py": b"alpha\r\nbeta\ngamma\r\n",
                         "scripts/reference.py": b"reference\r\n"}
            styles = ("lf", "mixed", "crlf")
            rows = []
            for (name, raw), style in zip(originals.items(), styles):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw if measured else raw.replace(b"\r\n", b"\n"))
                row = {"path": name, "raw_sha256": helper.digest(raw),
                       "normalized_sha256": helper.digest(raw.replace(b"\r\n", b"\n")),
                       "raw_bytes": len(raw), "newlines": raw.count(b"\n"), "style": style}
                if style == "mixed":
                    row["lf_lines"] = [[2, 2]]
                rows.append(row)
            value = {"schema": 1, "source_sha256": "9"*64,
                     "core_inventory": list(originals)[:2], "files": rows}
            manifest = root/"private-manifest.json"
            raw = (json.dumps(value)+"\n").encode()
            manifest.write_bytes(raw)
            with patch.object(helper, "_MANIFEST", manifest), patch.object(helper, "_MANIFEST_SHA256", helper.digest(raw)), patch.object(helper, "_COUNTS", (3, 2)):
                yield root, originals, value, manifest

    def repin_fixture(self, value, manifest):
        raw = json.dumps(value).encode()
        manifest.write_bytes(raw)
        return patch.object(helper, "_MANIFEST_SHA256", helper.digest(raw))

    def bytes_now(self, root, originals):
        return {name: (root/name).read_bytes() for name in originals}

    def test_normalized_preflight_reconstructs_all_bytes_without_writing(self):
        with self.fixture() as (root, originals, _, _):
            before = self.bytes_now(root, originals)
            _, items = helper.plan(root)
            self.assertEqual(self.bytes_now(root, originals), before)
            self.assertEqual({p.relative_to(root).as_posix(): after for p, _, after, _ in items}, originals)

    def test_already_measured_baseline_is_exact_noop(self):
        with self.fixture(measured=True) as (root, originals, _, _):
            _, items = helper.plan(root)
            with patch.object(helper, "atomic_write", side_effect=AssertionError("No-op must not write")):
                helper.restore(items)
            self.assertEqual(self.bytes_now(root, originals), originals)

    def test_explicit_fake_restore_preserves_mixed_and_final_newline_bytes(self):
        with self.fixture() as (root, originals, _, _):
            _, items = helper.plan(root)
            helper.restore(items)
            self.assertEqual(self.bytes_now(root, originals), originals)
            self.assertEqual(list(root.rglob("*.tmp")), [])

    def test_source_change_rejected_before_any_write(self):
        with self.fixture() as (root, originals, _, _):
            (root/"scripts/reference.py").write_bytes(b"changed code\n")
            before = self.bytes_now(root, originals)
            with patch.object(helper, "atomic_write", side_effect=AssertionError("Preflight must be read-only")), self.assertRaisesRegex(ValueError, "Normalized source changed"):
                helper.plan(root)
            self.assertEqual(self.bytes_now(root, originals), before)

    def test_new_production_file_or_missing_core_is_refused(self):
        for extra in (True, False):
            with self.subTest(extra=extra), self.fixture() as (root, originals, _, _):
                if extra:
                    (root/"scanner_server/fusion_allocation.py").write_bytes(b"production\n")
                else:
                    (root/"shared/b.py").unlink()
                with self.assertRaisesRegex(ValueError, "Core inventory changed"):
                    helper.plan(root)

    def test_private_manifest_changes_require_exact_bundled_pin(self):
        with self.fixture() as (root, _, _, manifest):
            manifest.write_bytes(manifest.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError, "Bundled measured-source manifest changed"):
                helper.plan(root)

    def test_unsafe_or_duplicate_paths_refused_even_in_authorized_fake_manifest(self):
        for name in ("../outside.py", "scanner_server/../../outside.py", "C:/outside.py", "scanner_server\\outside.py", "scanner_server/a.py"):
            with self.subTest(name=name), self.fixture() as (root, _, value, manifest):
                value["files"][2]["path"] = name
                with self.repin_fixture(value, manifest), self.assertRaises(ValueError):
                    helper.plan(root)

    def test_symlink_owner_and_resolved_escape_refused(self):
        for fault in ("symlink", "escape"):
            with self.subTest(fault=fault), self.fixture() as (root, _, _, _):
                selected = root/"scripts/reference.py"
                method = Path.is_symlink if fault == "symlink" else Path.resolve
                def altered(path, *args, **kwargs):
                    if path == selected:
                        return True if fault == "symlink" else root.parent/"outside.py"
                    return method(path, *args, **kwargs)
                with patch.object(Path, "is_symlink" if fault == "symlink" else "resolve", altered), self.assertRaisesRegex(ValueError, "Unsafe/missing"):
                    helper.plan(root)

    def test_malformed_mixed_ranges_and_forged_raw_hash_refused(self):
        changes = [lambda row: row.update(lf_lines=[]), lambda row: row.update(lf_lines=[[0, 1]]),
                   lambda row: row.update(lf_lines=[[2, 2], [2, 2]]), lambda row: row.update(raw_sha256="0"*64),
                   lambda row: row.update(raw_bytes=-1), lambda row: row.update(newlines=4)]
        for change in changes:
            with self.subTest(change=change), self.fixture() as (root, originals, value, manifest):
                change(value["files"][1])
                before = self.bytes_now(root, originals)
                with self.repin_fixture(value, manifest), self.assertRaises(ValueError):
                    helper.plan(root)
                self.assertEqual(self.bytes_now(root, originals), before)

    def test_mid_restore_failure_rolls_back_prior_atomic_replacement(self):
        with self.fixture() as (root, originals, _, _):
            before = self.bytes_now(root, originals)
            _, items = helper.plan(root)
            primary = OSError("Second file native replace failed")
            original = helper.atomic_write
            def injected(path, raw, mode):
                if path.name == "reference.py":
                    raise primary
                return original(path, raw, mode)
            with patch.object(helper, "atomic_write", injected), self.assertRaises(OSError) as result:
                helper.restore(items)
            self.assertIs(result.exception, primary)
            self.assertEqual(self.bytes_now(root, originals), before)

    def test_error_after_replace_is_enrolled_and_rolled_back(self):
        with self.fixture() as (root, originals, _, _):
            before = self.bytes_now(root, originals)
            _, items = helper.plan(root)
            primary = OSError("Cleanup failed after replacement")
            original = helper.atomic_write
            failed = False
            def injected(path, raw, mode):
                nonlocal failed
                original(path, raw, mode)
                if not failed:
                    failed = True
                    raise primary
            with patch.object(helper, "atomic_write", injected), self.assertRaises(OSError) as result:
                helper.restore(items)
            self.assertIs(result.exception, primary)
            self.assertEqual(self.bytes_now(root, originals), before)

    def test_concurrent_edit_is_preserved_and_other_writes_rolled_back(self):
        with self.fixture() as (root, originals, _, _):
            before = self.bytes_now(root, originals)
            _, items = helper.plan(root)
            selected = root/"scripts/reference.py"
            selected.write_bytes(b"external edit\n")
            with self.assertRaisesRegex(ValueError, "Source changed after preflight"):
                helper.restore(items)
            self.assertEqual((root/"shared/b.py").read_bytes(), before["shared/b.py"])
            self.assertEqual(selected.read_bytes(), b"external edit\n")

    def test_rollback_error_keeps_primary_and_reports_unrestored_path(self):
        with self.fixture() as (root, _, _, _):
            _, items = helper.plan(root)
            original = helper.atomic_write
            primary = OSError("primary write")
            calls = 0
            def injected(path, raw, mode):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise primary
                if calls == 3:
                    raise OSError("rollback fault")
                return original(path, raw, mode)
            with patch.object(helper, "atomic_write", injected), self.assertRaises(OSError) as result:
                helper.restore(items)
            self.assertIs(result.exception, primary)
            self.assertTrue(any("Rollback also failed" in note for note in primary.__notes__))

    def test_real_bundled_manifest_is_compact_exact74_core48(self):
        value = helper.load_manifest()
        self.assertEqual((len(value["files"]), len(value["core_inventory"])), (74, 48))
        self.assertEqual(sum(row["style"] == "mixed" for row in value["files"]), 9)

    def test_relocated_cli_help_has_no_restore_or_numerical_imports(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-S", str(Path(helper.__file__).resolve()), "--help"],
                                    cwd=folder, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--restore", result.stdout)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
