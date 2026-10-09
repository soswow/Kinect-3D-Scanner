"""Small source/ownership contracts; no build, DLL load or numerical import."""
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.research import cached_target_geometry_native as method


def xyz(*values):
    return struct.pack("<" + "d" * len(values), *values)


class FakeBackend:
    def __init__(self):
        self.created = []; self.queries = []; self.destroyed = []
        self.bad_result = False; self.destroy_failure = None

    def create(self, points, rows, limit):
        self.created.append((points, rows, limit))
        return len(self.created), method.reservation(rows)

    def query(self, handle, points, rows, radius, threads):
        self.queries.append((handle, points, rows, radius, threads))
        if self.bad_result:
            return struct.pack("<q", 99) * rows, xyz(0) * rows
        return struct.pack("<q", 0) * rows, xyz(0) * rows

    def destroy(self, handle):
        self.destroyed.append(handle)
        if handle == self.destroy_failure:
            raise RuntimeError("injected destroy failure")


class CacheContracts(unittest.TestCase):
    def test_index_reused_but_actual_queries_always_executed(self):
        backend = FakeBackend(); cache = method.CachedTargets(backend, query_threads=20)
        points = xyz(0, 0, 0)
        first = cache.query(points, 1, xyz(0, 0, 0), 1, .03)
        second = cache.query(points, 1, xyz(.01, 0, 0), 1, .03)
        self.assertEqual(len(backend.created), 1)
        self.assertEqual([row[1] for row in backend.queries], [xyz(0, 0, 0), xyz(.01, 0, 0)])
        self.assertEqual([row[-1] for row in backend.queries], [20, 20])
        self.assertNotEqual(first["queries"], second["queries"])
        self.assertFalse(second["qualified"]); cache.close()
        self.assertEqual(backend.destroyed, [1]); self.assertTrue(cache.report()["closed"])

    def test_target_replacement_signed_zero_and_row_order_create_new_indexes(self):
        backend = FakeBackend(); cache = method.CachedTargets(backend)
        for value in (xyz(0., 0, 0), xyz(-0., 0, 0), xyz(1, 0, 0)):
            cache.query(value, 1, xyz(0, 0, 0), 1, .03)
        cache.query(xyz(0, 0, 0, 1, 0, 0), 2, b"", 0, .03)
        cache.query(xyz(1, 0, 0, 0, 0, 0), 2, b"", 0, .03)
        self.assertEqual(len(backend.created), 5); cache.close()

    def test_lru_eviction_is_completed_before_new_native_allocation(self):
        backend = FakeBackend(); cache = method.CachedTargets(backend, max_targets=1)
        cache.query(xyz(0, 0, 0), 1, b"", 0, .03)
        cache.query(xyz(1, 0, 0), 1, b"", 0, .03)
        self.assertEqual(backend.destroyed, [1]); self.assertEqual(cache.report()["statistics"]["evictions"], 1)
        self.assertEqual(cache.report()["retained_targets"], 1); cache.close()

    def test_budget_domain_refusal_occurs_before_native_create(self):
        backend = FakeBackend(); cache = method.CachedTargets(backend, max_owned_bytes=66000)
        with self.assertRaisesRegex(method.NativeGeometryError, "before allocation"):
            cache.query(xyz(0, 0, 0, 1, 0, 0), 2, b"", 0, .03)
        self.assertEqual(backend.created, []); cache.close()
        for bad, rows in ((bytearray(xyz(0, 0, 0)), 1), (xyz(float("nan"), 0, 0), 1),
                          (xyz(2 ** 20 + 1, 0, 0), 1), (b"", False)):
            with self.assertRaises(ValueError): method.points_descriptor(bad, rows)

    def test_effective_configuration_or_backend_change_is_sticky(self):
        cache = method.CachedTargets(FakeBackend()); cache.configuration = (4, 256 * 1024 * 1024, 2)
        with self.assertRaisesRegex(method.NativeGeometryError, "changed"):
            cache.query(xyz(0, 0, 0), 1, b"", 0, .03)
        self.assertIsNotNone(cache.failure); cache.close()
        cache = method.CachedTargets(FakeBackend()); cache.backend = FakeBackend()
        with self.assertRaisesRegex(method.NativeGeometryError, "changed"):
            cache.query(xyz(0, 0, 0), 1, b"", 0, .03)

    def test_malformed_native_output_never_escapes_and_owner_is_retained(self):
        backend = FakeBackend(); backend.bad_result = True; cache = method.CachedTargets(backend)
        with self.assertRaisesRegex(method.NativeGeometryError, "Malformed"):
            cache.query(xyz(0, 0, 0), 1, xyz(0, 0, 0), 1, .03)
        self.assertEqual(cache.report()["retained_targets"], 1)
        self.assertEqual(cache.report()["statistics"]["queries"], 0); cache.close()

    def test_backend_replacement_cleanup_uses_original_allocating_owner(self):
        original, foreign = FakeBackend(), FakeBackend()
        cache = method.CachedTargets(original)
        cache.query(xyz(0, 0, 0), 1, b"", 0, .03)
        cache.backend = foreign
        with self.assertRaisesRegex(method.NativeGeometryError, "changed"):
            cache.query(xyz(0, 0, 0), 1, b"", 0, .03)
        cache.close()
        self.assertEqual(original.destroyed, [1]); self.assertEqual(foreign.destroyed, [])
        self.assertTrue(cache.closed)

    def test_failed_destroy_preserves_primary_and_other_cleanup_attempts(self):
        backend = FakeBackend(); cache = method.CachedTargets(backend)
        for x in (0, 1): cache.query(xyz(x, 0, 0), 1, b"", 0, .03)
        backend.destroy_failure = 1; primary = ValueError("primary search fault")
        with self.assertRaises(ValueError) as caught: cache.close(primary)
        self.assertIs(caught.exception, primary); self.assertIsInstance(primary.__cause__, RuntimeError)
        self.assertEqual(backend.destroyed, [1, 2]); self.assertEqual(cache.report()["retained_targets"], 1)
        self.assertFalse(cache.closed)
        backend.destroy_failure = None; cache.close()
        self.assertTrue(cache.closed); self.assertIsNotNone(cache.failure)


class SourceContracts(unittest.TestCase):
    def test_direct_abi_methods_check_bounds_before_accessing_ctypes(self):
        # A never-loaded shell has no ct/dll attributes. Refusal must precede them.
        native = object.__new__(method.NativeLibrary)
        for rows in (False, -1, method.MAX_ROWS + 1):
            with self.assertRaises(ValueError): native.create(b"", rows, method.NATIVE_MAX_BYTES)
            with self.assertRaises(ValueError): native.query(1, b"", rows, .03, 1)
        with self.assertRaises(ValueError): native.create(xyz(0, 0, 0), 1, 1)
        for handle, threads in ((0, 1), (2 ** 64, 1), (1, 3), (1, True)):
            with self.assertRaises(ValueError): native.query(handle, b"", 0, .03, threads)
        with self.assertRaises(ValueError): native.destroy(2 ** 64)

    def test_strict_radius_hit_boundary_and_miss_encoding(self):
        r = .125; boundary = r * r
        method.validate_output(struct.pack("<q", 0), xyz(boundary - 2 ** -59), 1, 1, r)
        method.validate_output(struct.pack("<q", -1), xyz(float("inf")), 1, 1, r)
        for index, distance in ((0, boundary), (-1, 0), (1, 0), (0, float("nan")), (0, -1)):
            with self.assertRaises(method.NativeGeometryError):
                method.validate_output(struct.pack("<q", index), xyz(distance), 1, 1, r)

    def test_cpp_declared_search_and_lifetime_seams(self):
        source = Path(method.__file__).with_suffix(".cpp").read_text(encoding="utf-8")
        for required in ("const Eigen::MatrixXd, -1, nanoflann::metric_L2, false",
                         "std::make_unique<Tree>(data.rows(), data, 15)",
                         "knnSearch(queries + 3 * row, 1, &index, &d2)",
                         "std::lower_bound(&d2, &d2 + 1, radius2)",
                         "#ifdef NANOFLANN_FIRST_MATCH", "worker.join()", "fp_valid(target.fp)"):
            self.assertIn(required, source)
        self.assertNotIn("Transform(", source); self.assertNotIn("tbb::", source)
        self.assertNotIn("radiusSearch(", source)

    def test_dependency_bytes_license_and_pinned_archive_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive = root / "source.tar.gz"
            target = root / "headers"; (target / "include").mkdir(parents=True)
            for name, data in (("include/nanoflann.hpp", b"header"), ("COPYING", b"license")):
                (target / name).write_bytes(data)
            with tarfile.open(archive, "w:gz") as stream:
                for name in ("include/nanoflann.hpp", "COPYING"):
                    data = (target / name).read_bytes(); member = tarfile.TarInfo("nanoflann-1.5.0/" + name)
                    member.size = len(data); stream.addfile(member, io.BytesIO(data))
            select = lambda name: str(name) in ("COPYING", "include/nanoflann.hpp")
            record = method._verify_archive(archive, target, method.file_hash(archive), select)
            self.assertEqual(record["files"], ["COPYING", "include/nanoflann.hpp"])
            with self.assertRaises(ValueError): method._verify_archive(archive, target, "0" * 64, select)
            (target / "include/nanoflann.hpp").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                method._verify_archive(archive, target, method.file_hash(archive), select)

    def test_recipe_is_unbuilt_strict_and_never_executes_a_compiler(self):
        eigen = {"files": ["COPYING.MPL2"] + [f"Eigen/f{i}" for i in range(10)]}
        nano = {"files": ["COPYING", "include/nanoflann.hpp"]}
        with tempfile.TemporaryDirectory() as directory:
            compiler = Path(directory) / "cl.exe"; compiler.write_bytes(b"source-only fake")
            with patch.object(method, "_verify_archive", side_effect=[eigen, nano]):
                receipt = method.build_recipe(eigen_root=directory, eigen_archive="unused", nanoflann_root=directory,
                    nanoflann_archive="unused", compiler=compiler, output=Path(directory) / "new.dll")
            self.assertEqual(receipt["stage"], "verified recipe only; not built")
            self.assertIn("/fp:strict", receipt["command"]); self.assertFalse((Path(directory) / "new.dll").exists())
            self.assertFalse(receipt["source_contract"]["qualified"])

    def test_fresh_import_and_help_do_not_load_numerical_or_native_packages(self):
        code = "import sys; sys.path.insert(0, %r); from scripts.research import cached_target_geometry_native; assert not set(('numpy','cupy','open3d','cv2','ctypes')) & set(sys.modules)" % str(method.ROOT)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-S", "-c", code], cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, "-S", method.__file__, "--help"], cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("does not build", result.stdout)


if __name__ == "__main__":
    unittest.main()
