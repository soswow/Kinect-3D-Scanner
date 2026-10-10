"""Stdlib-only script relocation contracts; no native/GPU modules imported.

Published JSON is compared in Git's LF text form so an ordinary autocrlf
checkout cannot invalidate the snapshot. Original raw proof byte fingerprints
inside those reports are never rewritten or treated as new-layout authority.
"""
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "scripts/tool-catalog.json"
PUBLISHED_REPORTS = {
    "docs/benchmarks/accelerometer-implementation.json": "02e4bef8fa13ebc4c998091e9462602d670d82f6958179b7d2a5dd826f29160b",
    "docs/benchmarks/accelerometer-probe.json": "98e151c741f7d37458cea8c11acd195a44c3868f4463079ff29bb0e5ad65d6c4",
    "docs/benchmarks/adaptive-feature-tracking.json": "cdd23451cea39370031e74a70113e4ab4693934d5b2de4bf03a55df25484ec42",
    "docs/benchmarks/algorithm-review.json": "1af2563ce090cb3b7f438a47bf5f03b156c3c518646ae025e3553f97369e6a8a",
    "docs/benchmarks/archived-adaptive-feature-tracking.json": "225a7f1675d4de43c2f6fdf70d3ce1280c268d566408d18c3f1c671ffcb5c025",
    "docs/benchmarks/backend-profile.json": "fe90c4d07523049d342c33251596fb248ae0affe3402d3eb73ddadb82ad3ea62",
    "docs/benchmarks/capture-profile.json": "7e5728220a6860d55065040677793a79d32ee5bd15ddecf3652245135791031f",
    "docs/benchmarks/chest-3-summary.json": "075c1217a01e440f9e4eaabcaa73cf713e0fe8919c738f73a2490bc07d7c0cf1",
    "docs/benchmarks/chest-3-visual-tracking.json": "323574841f7cb777d4aee88f530a0c508cc12bf75118bc17a39da22b8158eb7a",
    "docs/benchmarks/chest-4-tracking-recovery.json": "0095eebe2fff1016f0e34d7e5910c4fcefeb06a9a19df4e3330f56fe6a9574d3",
    "docs/benchmarks/confidence-fusion.json": "81959f2f75a17ba72fb19233d1ed14df1f833ca033c2ec618d649c5d1719f111",
    "docs/benchmarks/cuda-pipeline/current-release-validation.json": "8e9e6f8f27198c93e9bec9943edce741ce26164adc65276218dc4bd196d52ed9",
    "docs/benchmarks/cuda-pipeline/grid-lookup-ablation-order-v2/host-diagnostics-summary.json": "5b918681795b68ead4b3e908c251fafa7125a0a5b194617a9d1d329b89c96c79",
    "docs/benchmarks/cuda-pipeline/grid-lookup-ablation-v1/research-summary.json": "28ed16c54252bebbe89feba8109268d21f88ce680631770e98fb8f344a4916b8",
    "docs/benchmarks/cuda-pipeline/native-icp-signatures/research-summary.json": "e80e0aa9a419ea2ee3b091d59598c461ab578a77be8b1741339015182ec134de",
    "docs/benchmarks/cuda-pipeline/optix-nearest/research-summary.json": "647c21666ed2114f0a5ef731bfc88a8f9eeb8d3bd235629c79ba13bb4d89eebb",
    "docs/benchmarks/cuda-pipeline/report-manifest.json": "955d1f66ab60f52a65a1b653fe6b5256c3d9355c48c68a1b40c15d15c139c451",
    "docs/benchmarks/cuda-pipeline/resident-icp/research-summary.json": "739c8ebc0b9678845c63576ca553f0bbda10638f56918c1bfce11385bded4210",
    "docs/benchmarks/cuda-pipeline/selective-fragment-threads/research-summary.json": "01b7ada5952deb40ffbe195742ab417a663a75c148d0bab561ae12d574ca34ea",
    "docs/benchmarks/cuda-pipeline/summary-provenance-validation.json": "3fbdaff30516dced665d0b199371366643cfdafce022942c679f4f6b8bfe1e26",
    "docs/benchmarks/cuda-pipeline/uniform-grid-nearest/device-resident-v1/research-summary.json": "c506c84dc64efd251a574b711dd02974d2612dabdd8ccd2b0a557e2212bf20aa",
    "docs/benchmarks/cuda-pipeline/uniform-grid-nearest/flat-v1/research-summary.json": "59aa7e18e6a76684bbbcbf4957107e791e93f8ea1cfe13f6c66194689a9f1c41",
    "docs/benchmarks/cuda-pipeline/uniform-grid-nearest/research-summary.json": "b7eb0ccfa80f95884ccf94db667e0c9400d879a3fbe5ca22c6f17e550fa76578",
    "docs/benchmarks/cuda-pipeline/uniform-grid-nearest/staged-pruned-v1/research-summary.json": "b6415982ce1707c1098fabdf19280c0cdd4176195cd56c890b922c577ceec678",
    "docs/benchmarks/cuda-pipeline/uniform-grid-nearest/summary.json": "b7eb0ccfa80f95884ccf94db667e0c9400d879a3fbe5ca22c6f17e550fa76578",
    "docs/benchmarks/cuda-pipeline/validation.json": "aa392f9b04725b9b3b49abb5c242c37d3f6f44d1f4f016bae8610016c1bc8a07",
    "docs/benchmarks/cuda-pipeline/verification-profile/chest-3-summary.json": "52ab76745b94ca10dc3a8e686fb103f3da3c001e40c72af9db0ae742b90d8b90",
    "docs/benchmarks/cuda-session-performance.json": "fc1797b0b1f3bf680d12c7e03a2b6b77cb2edad4a68a8751bb7d167c190904d2",
    "docs/benchmarks/cuda-study/report-manifest.json": "bf4567c3dc7383c5908b484a38932de4c60a4b575e3f75a0438cb84e70ff3499",
    "docs/benchmarks/desk-supported-heldout-summary.json": "54a4ab264c6bd4a12e527eeb0fb3fe2ee5198306ef8f1bb895d5dd9ec06dd88d",
    "docs/benchmarks/feature-summary.json": "8ca718943d44fdba8b735cfb889807472fc004aa9756a761e0af885c1a32f237",
    "docs/benchmarks/icp-source-cache.json": "4f3c23134f2f277f2085d7abd93c55113c047aa9c10a94063a85170e9730a23b",
    "docs/benchmarks/live-tracking-depth-cache.json": "14609f29b4b96ccc6339fa12d3945d201295f2783b698a5eebb51f83b743789b",
    "docs/benchmarks/milestone-validation.json": "0da8722b473479fdabdf5d4d2c87719549079d3e9bd6e7b1a8db6d9f008ffbfa",
    "docs/benchmarks/native-performance.json": "ac82d81a7e246992496a974756c8a6901499da490b4a3b3da64aaddbe852670e",
    "docs/benchmarks/recorded-bundle-proposals.json": "457131625cbbcf798d1b1a3b5078512a12a1dfb1bb5061ec1f47707bb95e7aed",
    "docs/benchmarks/recorded-confidence-fusion-10000-blocks.json": "d61d8e526d73a7778fa80c5c94fabccc1014ac1f3f35d9ade96f6c403d31ba2a",
    "docs/benchmarks/recorded-confidence-fusion.json": "5866c94183257cbb3ecf0a6c6b8a53e3d56e20f3c6df0673a6e57e5160ead56c",
    "docs/benchmarks/recorded-tracking-repeatability.json": "4c2b1d80ac86ba8ace21379d3a3f967aa294655f8448e77646b8b609602a31eb",
    "docs/benchmarks/recorded-tracking-speed.json": "e10bd3d410b7caae7b3305170fff8a443e0b1ce344ef988ac4c2db2f390e1087",
    "docs/benchmarks/rgbd-bundle-adjustment.json": "f5e857370db1d4451001367a390dc935ab6aa8139bfebbc8b54cce7e0370f867",
    "docs/benchmarks/rgbd-summary.json": "c53eabc475701b7f99776ccfa0951b5c3908deb2faeef36fcaeedef21cf0a02e",
    "docs/benchmarks/xyz-heldout-summary.json": "175a5e717b13cc6431361b8974ae559d5f2330dcb01deeef950db9d6597254ee"
}


def static_path(node, filename, values=None):
    """Interpret only pathlib expressions; never execute a script or import it."""
    values = {} if values is None else values
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return str(filename)
        if node.id in values:
            return values[node.id]
        raise ValueError(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -static_path(node.operand, filename, values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return static_path(node.left, filename, values) / static_path(node.right, filename, values)
    if isinstance(node, ast.Subscript):
        return static_path(node.value, filename, values)[static_path(node.slice, filename, values)]
    if isinstance(node, ast.Attribute) and node.attr in ("parents", "parent", "stem", "name"):
        return getattr(static_path(node.value, filename, values), node.attr)
    if isinstance(node, ast.Call):
        args = [static_path(arg, filename, values) for arg in node.args]
        if isinstance(node.func, ast.Name) and node.func.id == "Path":
            return Path(*args)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("resolve", "with_name", "with_suffix", "joinpath"):
            return getattr(static_path(node.func.value, filename, values), node.func.attr)(*args)
    raise ValueError(ast.dump(node))


def bindings(tree, filename):
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = static_path(node.value, filename, values)
        except (ValueError, TypeError, AttributeError, IndexError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def is_main(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
        and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in node.test.comparators))


class ScriptLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        # Earlier component proofs pin the original relocation catalog bytes.
        # New, unrelocated field tools use an additive current-path inventory.
        field_catalog = json.loads((ROOT / "scripts/research/field-tool-catalog.json").read_text(encoding="utf-8"))
        if field_catalog["schema_version"] != 1:
            raise ValueError("Unsupported additive field-tool catalog")
        cls.entries = cls.catalog["entries"] + field_catalog["entries"]
        cls.field_paths = {e["path"] for e in field_catalog["entries"]}
        cls.mapping = {e["legacy_path"]: e["path"] for e in cls.entries}

    def python_sources(self):
        for entry in self.entries:
            path = ROOT / entry["path"]
            if path.suffix == ".py":
                text = path.read_text(encoding="utf-8")
                yield path, text, ast.parse(text, filename=entry["path"])

    def test_catalog_paths_are_unique_bounded_and_grouped(self):
        self.assertEqual(1, self.catalog["schema_version"])
        self.assertEqual(len(self.entries), len(self.mapping))
        self.assertEqual(len(self.entries), len({e["path"] for e in self.entries}))
        self.assertEqual({"maintained", "benchmarks", "active", "archive"}, {e["group"] for e in self.entries})
        for entry in self.entries:
            with self.subTest(entry=entry):
                for key in ("legacy_path", "path"):
                    value = PurePosixPath(entry[key])
                    self.assertFalse(value.is_absolute())
                    self.assertNotIn("..", value.parts)
                    self.assertEqual("scripts", value.parts[0])
                self.assertEqual(PurePosixPath(entry["legacy_path"]).name, PurePosixPath(entry["path"]).name)
                folder = {"maintained": "scripts", "benchmarks": "scripts/benchmarks",
                          "active": "scripts/research", "archive": "scripts/research/archive"}[entry["group"]]
                self.assertEqual(folder, str(PurePosixPath(entry["path"]).parent))
                self.assertTrue((ROOT / entry["path"]).is_file())

    def test_catalog_covers_relocated_folders_without_duplicate_old_entrypoints(self):
        expected = {e["path"] for e in self.entries if e["group"] != "maintained"}
        actual = {p.relative_to(ROOT).as_posix() for folder in ("scripts/benchmarks", "scripts/research")
                  for p in (ROOT / folder).rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.name not in ("__init__.py", "README.md")}
        self.assertEqual(expected, actual)
        for old, new in self.mapping.items():
            if old != new:
                self.assertFalse((ROOT / old).exists(), f"Stale old entrypoint: {old}")

    def test_root_path_evaluation_uses_repository_not_scripts_folder(self):
        checked = 0
        for path, _, tree in self.python_sources():
            values = bindings(tree, path)
            if "ROOT" in values:
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertEqual(ROOT, values["ROOT"].resolve())
                checked += 1
        self.assertGreaterEqual(checked, 10)

    def test_direct_entrypoints_bootstrap_before_local_imports(self):
        checked = 0
        for path, _, tree in self.python_sources():
            if not any(is_main(node) for node in tree.body):
                continue
            local = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
                     and n.module.split(".")[0] in ("scripts", "scanner_server", "shared") and n.col_offset == 0]
            if not local:
                continue
            bootstrap = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "insert"
                and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "path"
                and isinstance(n.func.value.value, ast.Name) and n.func.value.value.id == "sys"]
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(bootstrap, "Direct execution cannot find repository imports")
                self.assertLess(min(n.lineno for n in bootstrap), min(n.lineno for n in local))
                values = bindings(tree, path)
                candidates = []
                for call in bootstrap:
                    expr = call.args[1]
                    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "str":
                        expr = expr.args[0]
                    try:
                        candidates.append(static_path(expr, path, values).resolve())
                    except (ValueError, AttributeError, TypeError):
                        pass
                self.assertIn(ROOT, candidates)
            checked += 1
        self.assertGreater(checked, 0)

    def test_all_absolute_scripts_imports_resolve_and_use_current_modules(self):
        relocated_modules = {PurePosixPath(old).with_suffix("").as_posix().replace("/", "."):
                             PurePosixPath(new).with_suffix("").as_posix().replace("/", ".")
                             for old, new in self.mapping.items() if old.endswith(".py") and old != new}
        for path, _, tree in self.python_sources():
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names if alias.name.startswith("scripts."))
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scripts"):
                    names.append(node.module)
                    if node.module == "scripts" or (ROOT / node.module.replace(".", "/")).is_dir():
                        names.extend(node.module+"."+alias.name for alias in node.names if alias.name != "*")
                for name in names:
                    with self.subTest(path=path.relative_to(ROOT), module=name):
                        self.assertNotIn(name, relocated_modules, f"Old import should be {relocated_modules.get(name)}")
                        module_path = ROOT / name.replace(".", "/")
                        self.assertTrue(module_path.is_dir() or module_path.with_suffix(".py").is_file(),
                                        f"Local import target missing: {name}")

    def test_static_shader_and_sibling_resources_exist(self):
        checked = 0
        for path, _, tree in self.python_sources():
            values = bindings(tree, path)
            for name, value in values.items():
                if isinstance(value, Path) and value.suffix in (".cu", ".cpp") and value.is_relative_to(ROOT / "scripts"):
                    with self.subTest(path=path.relative_to(ROOT), binding=name):
                        self.assertTrue(value.is_file(), f"Missing local CUDA/C++ resource: {value}")
                    checked += 1
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "with_name":
                    continue
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    continue
                name = node.args[0].value
                if not name.endswith((".py", ".cu", ".cpp", ".ps1", ".md")):
                    continue
                try:
                    resource = static_path(node, path, values)
                except (ValueError, TypeError, AttributeError):
                    continue
                with self.subTest(path=path.relative_to(ROOT), sibling=name):
                    self.assertTrue(resource.is_file(), f"Relocated sibling resource missing: {resource}")
                checked += 1
        self.assertGreater(checked, 0)

    def test_legacy_literal_dependencies_are_explicitly_resolved(self):
        # Historical identities in comments/docstrings are allowed; executable
        # bare legacy file strings must pass through the central resolver.
        moved = {old for old, new in self.mapping.items() if old != new}
        for path, text, tree in self.python_sources():
            parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str) or node.value not in moved:
                    continue
                parent = parents.get(node)
                if isinstance(parent, ast.Expr):
                    continue
                ancestors = []
                while parent is not None:
                    ancestors.append(parent)
                    parent = parents.get(parent)
                resolved = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                               and n.func.id in ("tool_path", "tool_relative", "resolve_tool", "resolve_tool_path") for n in ancestors)
                # Resource lists may intentionally hold historical strings;
                # require an explicit resolver in that source instead of a
                # second implicit filename-fallback implementation.
                has_resolver = bool(re.search(r"\b(tool_path|tool_relative|resolve_tool|resolve_tool_path)\s*\(", text))
                with self.subTest(path=path.relative_to(ROOT), legacy=node.value):
                    self.assertTrue(resolved or has_resolver, "Bare stale resource path bypasses relocation mapping")

    def test_published_measurement_json_remains_unchanged(self):
        self.assertGreaterEqual(len(PUBLISHED_REPORTS), 2)
        for relative, expected in PUBLISHED_REPORTS.items():
            with self.subTest(report=relative):
                payload = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
                self.assertEqual(expected, hashlib.sha256(payload).hexdigest())

    def test_full_historical_matrix_stays_out_of_generated_documentation(self):
        # The full 4.2 MB matrix is available at the indexed historical commit.
        # Bounded published evidence retains the immutable hashes above.
        self.assertFalse((ROOT / "docs/benchmarks/cuda-pipeline-experiments.json").exists())
        for name in ("summarize_cuda_pipeline.py", "summarize_cuda_study.py"):
            tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
            summary = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--summary")
            default = next(keyword.value for keyword in summary.keywords if keyword.arg == "default")
            path = static_path(default, ROOT / "scripts" / name)
            self.assertTrue(path.is_relative_to(Path("benchmark-output")))

    def test_tool_finder_covers_sources_without_importing_or_running_them(self):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        environment.pop("PYTHONPATH", None)
        finder = ROOT / "scripts/list_tools.py"
        with tempfile.TemporaryDirectory(prefix="tool-finder-") as folder:
            def query(*args):
                result = subprocess.run([sys.executable, "-S", str(finder), "--json", *args],
                    cwd=folder, env=environment, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=15, check=False)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual([], list(Path(folder).iterdir()))
                return json.loads(result.stdout)

            rows = query()
            expected = {path.relative_to(ROOT).as_posix() for path in (ROOT / "scripts").rglob("*")
                if path.is_file() and path.suffix in {".py", ".ps1", ".cu", ".cpp", ".md", ".json"}
                and "__pycache__" not in path.parts and path.name not in {"__init__.py", "README.md"}}
            self.assertEqual(expected, {row["path"] for row in rows})
            by_path = {row["path"]: row for row in rows}
            self.assertEqual("cli", by_path["scripts/check_cuda_backend.py"]["kind"])
            self.assertEqual("helper", by_path["scripts/http_check_safety.py"]["kind"])
            self.assertEqual("native", by_path["scripts/research/device_loop_control.cu"]["kind"])
            session = query("--search", "SESSION", "--group", "workflow", "--kind", "cli")
            self.assertTrue(session)
            self.assertTrue(all(row["group"] == "workflow" and row["kind"] == "cli"
                and "session" in (row["path"] + " " + row["purpose"]).lower() for row in session))

    def test_report_formatter_help_needs_no_optional_packages_or_outputs(self):
        with tempfile.TemporaryDirectory(prefix="formatter-help-") as folder:
            for name in ("summarize_cuda_pipeline.py", "summarize_cuda_study.py", "list_tools.py"):
                result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts" / name), "--help"],
                    cwd=folder, capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=15, check=False)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn("usage:", result.stdout.lower())
                self.assertEqual([], list(Path(folder).iterdir()))

    def test_resource_resolver_maps_every_catalog_entry_from_other_cwd(self):
        helper = ROOT / "scripts/tool_paths.py"
        self.assertTrue(helper.is_file())
        spec = importlib.util.spec_from_file_location("layout_tool_paths", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        resolver = next((getattr(module, name, None) for name in
                         ("tool_path", "resolve_tool_path", "resolve_tool") if callable(getattr(module, name, None))), None)
        self.assertIsNotNone(resolver, "Central catalog resolver API missing")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="layout-cwd-") as folder:
            try:
                os.chdir(folder)
                for old, new in self.mapping.items():
                    # New field tools have current paths, without historical basename aliases.
                    spellings = (new, new.replace("/", "\\")) if new in self.field_paths else (
                        old, new, PurePosixPath(old).name, old.replace("/", "\\"))
                    for spelling in spellings:
                        with self.subTest(resource=spelling):
                            self.assertEqual((ROOT / new).resolve(), Path(resolver(spelling)).resolve())
            finally:
                os.chdir(previous)
        for name in ("..", "../outside.py", "scripts/../outside.py", "C:/outside.py",
                     "scripts\\..\\outside.py", "/outside.py", "//host/share/tool.py", ROOT):
            with self.subTest(rejected_resource=name):
                with self.assertRaises(ValueError):
                    resolver(name)

    def test_catalog_resolver_shader_and_module_literals_exist(self):
        spec = importlib.util.spec_from_file_location("layout_literal_paths", ROOT / "scripts/tool_paths.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        checked = 0
        for path, _, tree in self.python_sources():
            has_resolver = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                               and n.func.id == "tool_path" for n in ast.walk(tree))
            if not has_resolver:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                name = node.value
                known = name in self.mapping or name in self.mapping.values() or name in {
                    PurePosixPath(old).name for old in self.mapping}
                if not known:
                    continue
                with self.subTest(path=path.relative_to(ROOT), resource=name):
                    self.assertTrue(helper.tool_path(name).is_file(), f"Missing catalog resource: {name}")
                checked += 1
        self.assertGreater(checked, 0)

    def test_relocated_cli_help_without_site_packages_or_working_directory(self):
        entrypoints = (
            "scripts/benchmark_device_grid_resident.py", "scripts/benchmark_device_grid_resident_timing.py",
            "scripts/research_resident_icp.py", "scripts/benchmark_flat_grid_nn.py",
            "scripts/benchmark_flat_grid_timing.py", "scripts/benchmark_uniform_grid_timing.py",
            "scripts/benchmark_staged_grid_timing.py", "scripts/benchmark_parallel_staged_grid_timing.py",
            "scripts/benchmark_grid_lookup_ablation.py", "scripts/profile_cuda_pipeline.py")
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        # Prevent a developer PYTHONPATH from making optional numerical
        # packages available despite -S; only the script bootstrap supplies
        # repository modules. Help must finish before allocation/API/work.
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="layout-help-") as folder:
            scratch = Path(folder)
            for legacy in entrypoints:
                script = ROOT / self.mapping[legacy]
                with self.subTest(entrypoint=script.relative_to(ROOT)):
                    result = subprocess.run([sys.executable, "-S", str(script), "--help"],
                        cwd=scratch, env=environment, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=15, check=False)
                    self.assertEqual(0, result.returncode,
                        f"Help failed before numerical work:\n{result.stdout}\n{result.stderr}")
                    self.assertIn("usage:", result.stdout.lower())
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual([], list(scratch.iterdir()), "--help generated work/report files")

    def test_direct_static_script_dependency_paths_exist(self):
        checked = 0
        for path, _, tree in self.python_sources():
            values = bindings(tree, path)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.BinOp, ast.Call)):
                    continue
                try:
                    resource = static_path(node, path, values)
                except (ValueError, TypeError, AttributeError, IndexError):
                    continue
                if (not isinstance(resource, Path) or not resource.is_absolute()
                        or not resource.is_relative_to(ROOT / "scripts")
                        or resource.suffix not in (".py", ".cu", ".cpp", ".ps1", ".md")):
                    continue
                with self.subTest(path=path.relative_to(ROOT), resource=resource.relative_to(ROOT)):
                    self.assertTrue(resource.is_file(), "Static source dependency bypasses catalog relocation")
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
