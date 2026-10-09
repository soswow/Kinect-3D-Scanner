"""Bounded CPU-method reuse/source/reference contracts; no numerical imports."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.research import near_seed_icp as method
from scripts.research import benchmark_near_seed_reuse as driver


def seed(x=0.):
    return struct.pack("<16d", 1., 0., 0., x, 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.)


def evidence(x=0.):
    pose = {"dtype": "<f8", "shape": [4, 4], "nbytes": 128, "sha256": hashlib.sha256(seed(x)).hexdigest()}
    return {"pose": pose, "transformation": [list(struct.unpack("<16d", seed(x))[i:i+4]) for i in (0, 4, 8, 12)],
        "fitness": .8, "inlier_rmse": .002, "correspondence_mapping": {"sha256": "canonical"}, "raw_correspondences": {"sha256": "raw"}}


def payload():
    return method.FrozenResult(seed(), struct.pack("<4i", 0, 1, 1, 0), "<i4", (2, 2), json.dumps(evidence()), .8, .002)


class CacheTests(unittest.TestCase):
    def test_threshold_original_representative_and_no_transitive_chain(self):
        cache = method.RepresentativeCache(); a = cache.insert("cloud", seed(), payload())
        self.assertEqual(cache.find("cloud", seed(.75e-10))[0], a)
        self.assertIsNone(cache.find("cloud", seed(1.5e-10)))
        b = cache.insert("cloud", seed(1.5e-10), payload())
        self.assertEqual(cache.find("cloud", seed(.9e-10))[0], a)
        self.assertNotEqual(a, b); self.assertEqual(cache.entries[a][1], seed())

    def test_all16entries_inclusive_bound_and_one_ulp_outside(self):
        cache = method.RepresentativeCache(); cache.insert("cloud", seed(), payload())
        self.assertIsNotNone(cache.find("cloud", seed(1e-10)))
        self.assertIsNone(cache.find("cloud", seed(1.0000000000000002e-10)))
        values = list(struct.unpack("<16d", seed())); values[14] = 2e-10
        self.assertIsNone(cache.find("cloud", struct.pack("<16d", *values)))

    def test_exact_cloud_direction_normal_color_bucket_never_approximated(self):
        cache = method.RepresentativeCache(); cache.insert("source/target/points/normals/colors", seed(), payload())
        for key in ("target/source/points/normals/colors", "source/target/changednormals/colors", "source/target/points/normals/changedcolors"):
            self.assertIsNone(cache.find(key, seed()))

    def test_lru_entry_eviction_and_first_rep_order_are_distinct(self):
        cache = method.RepresentativeCache(max_entries=2)
        a = cache.insert("a", seed(), payload()); b = cache.insert("b", seed(), payload())
        cache.find("a", seed()); c = cache.insert("c", seed(), payload())
        self.assertIn(a, cache.entries); self.assertNotIn(b, cache.entries); self.assertIn(c, cache.entries)
        self.assertEqual(cache.evictions, 1)

    def test_cloud_cap_evicts_all_old_bucket_entries(self):
        cache = method.RepresentativeCache(max_clouds=1)
        cache.insert("a", seed(), payload()); cache.insert("a", seed(1.), payload()); cache.insert("b", seed(), payload())
        self.assertEqual(list(cache.buckets), ["b"]); self.assertEqual(len(cache.entries), 1); self.assertEqual(cache.evictions, 2)

    def test_byte_cap_and_oversized_payload_bypass_without_unbounded_storage(self):
        one = method.RepresentativeCache(); one.insert("a", seed(), payload()); limit = one.bytes
        cache = method.RepresentativeCache(max_bytes=limit)
        cache.insert("a", seed(), payload()); cache.insert("b", seed(), payload())
        self.assertLessEqual(cache.bytes, limit); self.assertEqual(len(cache.entries), 1)
        tiny = method.RepresentativeCache(max_bytes=1); self.assertIsNone(tiny.insert("a", seed(), payload())); self.assertFalse(tiny.entries)

    def test_configuration_immutable_and_owner_replacement_refused(self):
        cache = method.RepresentativeCache()
        with self.assertRaises(TypeError): cache.configuration["threshold"] = 1.
        cache.configuration = dict(cache.configuration, threshold=1e-8)
        with self.assertRaises(method.scope.BridgeFailure): cache.find("a", seed())

    def test_cached_payload_is_tuple_slots_and_bytes_not_mutable_dataclass(self):
        value = payload()
        with self.assertRaises(AttributeError): object.__setattr__(value, "pose", seed(1.))
        with self.assertRaises(TypeError): value.pose[0] = 1
        cache = method.RepresentativeCache()
        changed = value._replace(pose=bytearray(value.pose))
        with self.assertRaises(method.scope.BridgeFailure): cache.insert("a", seed(), changed)

    def test_hit_reconstruction_has_new_writable_owners_each_time(self):
        value = payload()
        fake = SimpleNamespace(frombuffer=lambda raw, dtype: SimpleNamespace(reshape=lambda *shape:
            SimpleNamespace(copy=lambda: bytearray(raw))))
        first = method.restore_result(fake, value); first.transformation[0] ^= 1; first.correspondence_set[0] ^= 1
        second = method.restore_result(fake, value)
        self.assertEqual(bytes(second.transformation), value.pose); self.assertEqual(bytes(second.correspondence_set), value.raw_pairs)
        self.assertIsNot(first.transformation, second.transformation); self.assertIsNot(first.correspondence_set, second.correspondence_set)

    def test_invalid_policy_seed_and_closed_cache_fail(self):
        for kwargs in ({"threshold": 1e-6}, {"max_entries": True}, {"max_clouds": 0}, {"max_bytes": -1}):
            with self.assertRaises(method.scope.BridgeFailure): method.RepresentativeCache(**kwargs)
        cache = method.RepresentativeCache()
        for value in (b"bad", seed(float("nan"))):
            with self.assertRaises(method.scope.BridgeFailure): cache.insert("a", value, payload())
        cache.close()
        with self.assertRaises(method.scope.BridgeFailure): cache.find("a", seed())


class MethodTests(unittest.TestCase):
    def setup_method(self, audit=True, authorizer=None):
        self.native_calls, self.objects = [], []
        def cpu(source, target, initial):
            self.native_calls.append(initial); value = object(); self.objects.append(value); return value
        self.original = SimpleNamespace(cpu_match=cpu, check=lambda: None)
        self.np = SimpleNamespace(asarray=lambda value: SimpleNamespace(tobytes=lambda order: value))
        self.source = SimpleNamespace(points=[0, 1]); self.target = SimpleNamespace(points=[2, 3])
        def input_binding(np, source, target, initial):
            cloud = {k: {"dtype": "<f8", "shape": [2, 3], "nbytes": 48, "sha256": k} for k in ("points", "normals", "colors")}
            return {"source": cloud, "target": cloud, "seed": {"dtype": "<f8", "shape": [4, 4], "sha256": hashlib.sha256(initial).hexdigest()}}
        self.shadow = {"correspondence_ids_equal": True, "transform_max_abs_delta": 1e-15, "fitness_abs_delta": 0., "rmse_abs_delta": 1e-16}
        for mock in (patch.object(method.scope, "input_binding", side_effect=input_binding), patch.object(method, "freeze_result", return_value=payload()),
            patch.object(method, "restore_result", side_effect=lambda np, value: object()), patch.object(method.scope, "result_shadow", side_effect=lambda *args: dict(self.shadow))):
            mock.start(); self.addCleanup(mock.stop)
        return method.NearSeedMatcher(self.np, self.original, lambda: None, audit=audit, timing_authorizer=authorizer)

    def test_miss_returns_original_object_hit_is_fresh_and_actual_seed_shadowed(self):
        cache = self.setup_method(); a = cache.match(self.source, self.target, seed()); b = cache.match(self.source, self.target, seed(1e-12))
        self.assertIs(a, self.objects[0]); self.assertIsNot(b, self.objects[0]); self.assertIsNot(b, self.objects[1])
        self.assertEqual(self.native_calls, [seed(), seed(1e-12)])
        self.assertEqual(cache.statistics["hits"], 1); self.assertEqual(cache.statistics["audited_hits"], 1)

    def test_wrong_ids_or_tight_numeric_hit_bound_latches_without_retry(self):
        for name, value in (("correspondence_ids_equal", False), ("transform_max_abs_delta", 1.01e-10), ("fitness_abs_delta", 1.01e-12), ("rmse_abs_delta", 1.01e-12)):
            cache = self.setup_method(); cache.match(self.source, self.target, seed()); self.shadow[name] = value
            with self.assertRaises(method.scope.BridgeFailure): cache.match(self.source, self.target, seed(1e-12))
            count = len(self.native_calls)
            with self.assertRaises(method.scope.BridgeFailure): cache.match(self.source, self.target, seed())
            self.assertEqual(len(self.native_calls), count)

    def test_timing_requires_own_actual_input_authorizer_before_work(self):
        with self.assertRaises(method.scope.BridgeFailure): self.setup_method(audit=False)
        seen = []; cache = self.setup_method(audit=False, authorizer=lambda value: seen.append(value))
        cache.match(self.source, self.target, seed()); cache.match(self.source, self.target, seed(1e-12))
        self.assertEqual(len(seen), 2); self.assertEqual(len(self.native_calls), 1); self.assertEqual(cache.statistics["audited_hits"], 0)

    def test_changed_audit_or_owner_is_rejected_before_original_call(self):
        cache = self.setup_method(); cache.audit = False
        with self.assertRaises(method.scope.BridgeFailure): cache.match(self.source, self.target, seed())
        self.assertFalse(self.native_calls)

    def test_replaced_healthy_cache_configuration_is_rejected_before_call_or_clear(self):
        for action in ("call", "clear"):
            cache = self.setup_method(); cache.cache = method.RepresentativeCache(threshold=1e-8)
            with self.assertRaises(method.scope.BridgeFailure):
                if action == "call": cache.match(self.source, self.target, seed())
                else: cache.clear_pair()
            self.assertFalse(self.native_calls)

    def test_cold_pair_clear_retains_counters_but_not_prior_classes(self):
        cache = self.setup_method(); cache.match(self.source, self.target, seed()); cache.clear_pair()
        cache.match(self.source, self.target, seed()); cache.close()
        self.assertEqual(cache.statistics["misses"], 2); self.assertEqual(cache.statistics["hits"], 0)
        self.assertEqual(cache.report()["cleared_pairs"], 1); self.assertEqual(cache.report()["owned_payload_bytes"], 0)


class SourceTests(unittest.TestCase):
    def test_loaded_method_class_code_defaults_and_global_mutations_fail(self):
        guard = driver.LoadedMethodGuard(); guard.check()
        fn = method.RepresentativeCache.find; original = fn.__code__
        try:
            fn.__code__ = (lambda *args: None).__code__
            with self.assertRaises(method.scope.BridgeFailure): guard.check()
        finally: fn.__code__ = original
        with patch.object(method, "THRESHOLDS", (1.,)):
            with self.assertRaises(method.scope.BridgeFailure): guard.check()
        with patch.object(driver, "Observer", object):
            with self.assertRaises(method.scope.BridgeFailure): guard.check()
        fn = method.RepresentativeCache.__init__; original = fn.__defaults__
        try:
            fn.__defaults__ = (1e-8, 256, 64, 64*1024**2)
            with self.assertRaises(method.scope.BridgeFailure): guard.check()
        finally: fn.__defaults__ = original
        fn = method.scope.result_shadow; original = fn.__code__
        try:
            fn.__code__ = (lambda *args: None).__code__
            with self.assertRaises(method.scope.BridgeFailure): guard.check()
        finally: fn.__code__ = original
        guard.check()

    def test_actual_input_reference_and_result_class_fingerprints_are_exact(self):
        a = evidence(); b = evidence(1e-15)
        self.assertNotEqual(driver.result_reference(a), driver.result_reference(b))
        receipt = {"kind": "hit", "class_id": 0, "cloud_value_sha256": "a", "representative_seed_sha256": "b", "seed_max_abs_delta": 1e-12, "threshold": 1e-10, "evictions": 0}
        other = dict(receipt, representative_seed_sha256="changed")
        self.assertNotEqual(driver.class_reference(receipt), driver.class_reference(other))

    def test_foreign_old_or_failed_proof_refused_without_native_imports(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"report.json"
            for value in ({"kind": "gpu-icp-complete-bridge-owned-proof-audit-v2"}, {"kind": driver.KIND, "mode": "audit", "status": "failed"}):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(method.scope.BridgeFailure): driver.load_audit(path, {}, [])

    def test_othercwd_help_and_fresh_import_have_no_numerical_modules(self):
        with tempfile.TemporaryDirectory() as folder:
            code = "import sys;sys.path.insert(0,"+repr(str(driver.ROOT))+ ");import scripts.research.benchmark_near_seed_reuse;assert not any(k in sys.modules for k in ('numpy','cupy','open3d','cv2'))"
            value = subprocess.run([sys.executable, "-S", "-c", code], cwd=folder, capture_output=True, text=True)
            self.assertEqual(value.returncode, 0, value.stderr)
            value = subprocess.run([sys.executable, "-S", str(driver.ROOT/driver.OWN_FILES[1]), "--help"], cwd=folder, capture_output=True, text=True)
            self.assertEqual(value.returncode, 0, value.stderr)


def closed_audit():
    configuration = dict(driver.CONFIGURATION)
    descriptors = [{"dtype": "<f8", "shape": [4, 4], "nbytes": 128, "sha256": hashlib.sha256(seed(x)).hexdigest()} for x in (0., 1e-12)]
    captures = [{"tasks": [{"position": 0, "pair": [0, 1], "proposal_count": 2, "seeds": descriptors}]}]
    proposals = []
    cloud = {k: {"dtype": "<f8", "shape": [2, 3], "nbytes": 48, "sha256": k} for k in ("points", "normals", "colors")}
    for i in range(2):
        inputs = {"source": copy.deepcopy(cloud), "target": copy.deepcopy(cloud), "seed": descriptors[i]}
        call = {"call_index": 0, "context": {"capture_index": 0, "task": 0, "pair": [0, 1], "proposal_index": i},
            "caller": "_verify_bridge", "input_binding": inputs, "result": evidence(), "complete": True,
            "input_bytes_unchanged": True, "failure": None}
        proposals.append({"proposal_index": i, "calls": [call], "gates": [{"name": "_strong", "complete": True, "result": True}], "result": None, "complete": True})
    pair = {"identity": [0, 0, 0, 1], "original_seed_descriptors": descriptors, "proposals": proposals,
        "original_pair_verdict": {"accepted": True}, "complete": True, "input_bytes_unchanged": True, "seed_bytes_unchanged": True}
    native = {"pairs": [pair], "complete": True, "cleanup_passed": True, "cleanup_failures": [], "failure": None, "call_count": 2}
    candidate = copy.deepcopy(native)
    for i, call in enumerate(driver.phase_calls(candidate)):
        a, b = evidence(1e-15), evidence()
        shadow = {"collected": False} if i == 0 else {"collected": True, "passed": True, "native": a, "candidate": b,
            "correspondence_ids_equal": True, "transform_max_abs_delta": 1e-15, "fitness_abs_delta": 0., "rmse_abs_delta": 0.}
        call["method_receipt"] = {"kind": "miss" if i == 0 else "hit", "class_id": 0,
            "cloud_value_sha256": hashlib.sha256(driver.scope.canonical({k: call["input_binding"][k] for k in ("source", "target")}).encode()).hexdigest(),
            "representative_seed_sha256": descriptors[0]["sha256"], "seed_max_abs_delta": 0. if i == 0 else 1e-12,
            "threshold": 1e-10, "evictions": 0, "shadow": shadow}
    candidate["cache"] = {"audit": True, "closed": True, "failure": None, "configuration": configuration,
        "policy": method.POLICY, "owned_payload_bytes": 0, "statistics": {"calls": 2, "hits": 1, "misses": 1,
            "audited_hits": 1, "original_cpu_calls": 2, "unretained_misses": 0}}
    binding = {"configuration": configuration}
    return {"kind": driver.KIND, "mode": "audit", "status": "passed", "failure": None, "cleanup_passed": True,
        "cleanup_failures": [], "binding": binding, "binding_after": copy.deepcopy(binding), "captures": captures,
        "rounds": [{"native": native, "cache": candidate, "quality": driver.compare_phases(native, candidate)}]}, binding, captures


class ReceiptTests(unittest.TestCase):
    def load(self, report, binding, captures):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"audit.json"; path.write_text(json.dumps(report), encoding="utf-8")
            return driver.load_audit(path, binding, captures)

    def test_complete_own_receipt_and_ordered_immutable_references_pass(self):
        report, binding, captures = closed_audit(); digest, calls, phase = self.load(report, binding, captures)
        self.assertEqual(len(digest), 64); self.assertIs(type(calls), tuple); self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], driver.call_reference(driver.phase_calls(phase)[1]))

    def test_actual_kinds_counts_unretained_and_configuration_are_required(self):
        for change in ("unknown", "counts", "unretained", "configuration"):
            report, binding, captures = closed_audit(); phase = report["rounds"][0]["cache"]
            if change == "unknown": driver.phase_calls(phase)[0]["method_receipt"]["kind"] = "other"
            elif change == "counts": phase["cache"]["statistics"].update(hits=2, misses=0, audited_hits=2)
            elif change == "unretained": phase["cache"]["statistics"]["unretained_misses"] = 1
            else: phase["cache"]["configuration"] = dict(binding["configuration"], threshold=1e-8)
            with self.assertRaises(method.scope.BridgeFailure): self.load(report, binding, captures)

    def test_shadow_numeric_evidence_is_independently_recomputed(self):
        for key, value in (("transform_max_abs_delta", -1.), ("fitness_abs_delta", float("nan")), ("rmse_abs_delta", 1e-13)):
            report, binding, captures = closed_audit(); call = driver.phase_calls(report["rounds"][0]["cache"])[1]
            call["method_receipt"]["shadow"][key] = value
            with self.assertRaises(method.scope.BridgeFailure): self.load(report, binding, captures)
        report, binding, captures = closed_audit(); call = driver.phase_calls(report["rounds"][0]["cache"])[1]
        call["method_receipt"]["shadow"]["candidate"] = evidence(1e-15)
        with self.assertRaises(method.scope.BridgeFailure): self.load(report, binding, captures)

    def test_native_suffix_cleanup_and_unchanged_inputs_are_required(self):
        for change in ("suffix", "cleanup", "inputs"):
            report, binding, captures = closed_audit(); phase = report["rounds"][0]["native"]
            if change == "suffix": phase["pairs"][0]["proposals"].pop()
            elif change == "cleanup": phase["cleanup_passed"] = False
            else: driver.phase_calls(phase)[0]["input_bytes_unchanged"] = False
            with self.assertRaises(method.scope.BridgeFailure): self.load(report, binding, captures)

    def test_information_gate_uses_only_declared_original_separate_tolerance(self):
        report, _, _ = closed_audit(); a, b = report["rounds"][0]["native"], report["rounds"][0]["cache"]
        for phase, number in ((a, 0.), (b, 5e-6)):
            phase["pairs"][0]["proposals"][0]["gates"][0] = {"name": "REG.get_information_matrix_from_point_clouds", "complete": True, "result": number}
        self.assertTrue(driver.compare_phases(a, b)["passed"])
        for phase in (a, b): phase["pairs"][0]["proposals"][0]["gates"][0]["name"] = "_strong"
        self.assertFalse(driver.compare_phases(a, b)["passed"])

    def test_phase_preserves_primary_and_attempts_independent_cleanup_with_partial_timers(self):
        primary = method.scope.BridgeFailure("original method fault"); secondary = RuntimeError("close fault"); attempted = []
        def close(): attempted.append("close"); raise secondary
        def report(): attempted.append("report"); raise RuntimeError("report fault")
        cache = SimpleNamespace(close=close, report=report)
        fragment = SimpleNamespace(keys=[])
        fixtures = [{"tasks": [{"pair": [0, 1], "proposals": []}], "fragments": [{"keys": []}, {"keys": []}], "camera": {}}]
        phase = {"method": "cache", "pairs": [], "complete": False}
        imports = {"scripts.research.benchmark_parallel_fragments": SimpleNamespace(unpack_fragment=lambda v: fragment),
            "scripts.research.microbatch_bridge_driver": SimpleNamespace(train_inventory=lambda *v: {})}
        def boundary(): raise primary
        with patch.dict(sys.modules, imports), patch.object(driver.scope, "original_pair_verdict", return_value={}):
            with self.assertRaises(method.scope.BridgeFailure) as caught: driver.execute_phase(None, None, None, None, fixtures, cache, None, boundary, phase)
        self.assertIs(caught.exception, primary); self.assertIs(caught.exception.__cause__, secondary); self.assertEqual(attempted, ["close", "report"])
        self.assertFalse(phase["complete"]); self.assertFalse(phase["cleanup_passed"]); self.assertEqual(len(phase["cleanup_failures"]), 2)
        self.assertGreaterEqual(phase["whole_phase_wall_s"], phase["pairs"][0]["inclusive_pair_wall_s"])
        self.assertEqual(phase["call_count"], 0)


if __name__ == "__main__": unittest.main()
