"""Offline full-Finish proof regressions; no numerical or device imports."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts.research import validate_finish_resident_proof as proof
from scripts.research.validate_device_flat_grid_proof import DeviceFlatGridProofAuthority


def payload(marker="a"):
    array = lambda shape: {"sha256": marker * 64, "dtype": "<f8", "shape": shape}
    return {"source_points": array([5, 3]), "source_normals": array([0, 3]),
            "target_points": array([7, 3]), "target_normals": array([7, 3]), "seed": array([4, 4]),
            "stages": copy.deepcopy(proof.STAGES), "huber_m": .01,
            "relative_fitness": 1e-6, "relative_rmse": 1e-6, "thread_policy": dict(proof.THREAD_POLICY),
            "site": {"file": "scanner_server/refinement.py", "function": "_match", "line": 24},
            "gate_context": ["reconnect", "proposal:0"]}


def component():
    return DeviceFlatGridProofAuthority("b" * 64, "c" * 64, "d" * 64, "e" * 64,
                                       frozenset({"0" * 64}), (), json.dumps({"resident": "fixture"}))


def authority(calls):
    base = component().__dict__.copy()
    base["target_digests"] = frozenset(call["target_points"]["sha256"] for call in calls)
    return proof.FinishResidentProofAuthority(**base, finish_scope_json=json.dumps({"archive": "fresh"}),
        expected_call_signatures=tuple(proof.call_signature(call) for call in calls),
        finish_audit_report_sha256="f" * 64, quality_proof_sha256="1" * 64, full_finish_artifact_sha256=())


class FakeSolver:
    def __init__(self, result=None, error=None, close_error=None):
        self.result = result if result is not None else object()
        self.error, self.close_error = error, close_error
        self.statistics, self.closed, self.match_calls = {}, False, 0
        self.cpu_fallback = lambda *args: "unused original fallback"

    def match(self, *args):
        self.match_calls += 1
        if self.error:
            raise self.error
        return self.result

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


def fake_capture(source, target, initial, site, gate_context, thread_policy):
    value = payload(source)
    value["gate_context"] = list(gate_context)
    return value


def good_comparison(*args):
    return {"passed": True, "transformation_max_abs_delta": 0., "fitness_abs_delta": 0.,
            "rmse_abs_delta": 0., "correspondence_ids_equal": True}


class FinishCallContractTests(unittest.TestCase):
    def test_fixed_original_call_recipe_and_empty_source_normals_are_supported(self):
        value = payload()
        proof.validate_call_inputs(value)
        token = authority([value])
        self.assertEqual(proof.call_signature(value), proof.validate_expected_call(token, 0, value))
        proof.validate_complete_calls(token, 1)

    def test_byte_shapes_policy_and_unrounded_seeds_fail_closed(self):
        mutations = (
            lambda p: p["source_points"].update(dtype="<f4"),
            lambda p: p["target_normals"].update(shape=[6, 3]),
            lambda p: p["seed"].update(shape=[3, 4]),
            lambda p: p["target_points"].update(sha256="unknown"),
            lambda p: p["source_points"].update(shape=[True, 3]),
            lambda p: p["source_points"].update(shape=[1_000_001, 3]),
            lambda p: p["source_normals"].update(shape=[3, 3]),
            lambda p: p.update(stages=[[.12, 39], [.06, 30], [.03, 20]]),
            lambda p: p.update(huber_m=.02),
            lambda p: p.update(relative_fitness=1e-5),
            lambda p: p.update(thread_policy={"open3d": 20, "opencv": 20, "omp": "20"}),
            lambda p: p["site"].update(line=True),
            lambda p: p.update(gate_context=None),
            lambda p: p.update(gate_context=[float("nan")]),
        )
        for mutate in mutations:
            value = payload()
            mutate(value)
            with self.subTest(value=value), self.assertRaises(proof.GridProofError):
                proof.validate_call_inputs(value)
        token = authority([payload()])
        changed = payload()
        changed["seed"]["sha256"] = "2" * 64
        with self.assertRaisesRegex(proof.GridProofError, "unrounded seed"):
            proof.validate_expected_call(token, 0, changed)

    def test_full_finish_requires_distinct_authority_and_exact_complete_order(self):
        first, second = payload("a"), payload("b")
        token = authority([first, second])
        with self.assertRaisesRegex(proof.GridProofError, "Old component"):
            proof.validate_expected_call(component(), 0, first)
        for index, call in ((0, second), (1, first), (2, first), (-1, first), (True, first)):
            with self.subTest(index=index), self.assertRaises(proof.GridProofError):
                proof.validate_expected_call(token, index, call)
        for count in (0, 1, 3, True):
            with self.subTest(count=count), self.assertRaises(proof.GridProofError):
                proof.validate_complete_calls(token, count)
        proof.validate_complete_calls(token, 2)

    def test_scope_and_runtime_properties_do_not_expose_mutable_authority(self):
        token = authority([payload()])
        token.finish_scope["archive"] = "different"
        token.runtime_binding["resident"] = "different"
        self.assertEqual(token.finish_scope, {"archive": "fresh"})
        self.assertEqual(token.runtime_binding, {"resident": "fixture"})
        self.assertEqual(token.target_digests, frozenset({"a" * 64}))

    def test_import_does_not_load_numerical_libraries(self):
        code = ("import sys,json;import scripts.research.validate_finish_resident_proof;"
                "print(json.dumps(sorted({'numpy','open3d','cupy','cv2'}&set(sys.modules))))")
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=proof.ROOT,
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])

    def test_original_cpu_shadow_cannot_change_seed_inputs_or_hide_full_fallback(self):
        original = payload()
        signature = proof.call_signature(original)
        row = {"complete": True, "call_inputs": original, "call_signature": signature,
               "inputs_after_resident": signature, "inputs_before_shadow": signature,
               "inputs_after_shadow": signature, "resident_full_call_fallbacks": 0,
               "cpu_shadow": {"passed": True, "correspondence_ids_equal": True,
                              "transformation_max_abs_delta": 0., "fitness_abs_delta": 0., "rmse_abs_delta": 0.}}
        self.assertEqual(proof.validate_shadow(row), signature)
        mutations = [lambda r: r.update(inputs_before_shadow="2" * 64),
                     lambda r: r.update(inputs_after_shadow="2" * 64),
                     lambda r: r.update(inputs_after_resident="2" * 64),
                     lambda r: r.update(resident_full_call_fallbacks=1),
                     lambda r: r["cpu_shadow"].update(correspondence_ids_equal=False)]
        for field in ("transformation_max_abs_delta", "fitness_abs_delta", "rmse_abs_delta"):
            for value in (True, None, -1e-10, float("inf"), float("nan"), 1.01e-8):
                mutations.append(lambda r, f=field, v=value: r["cpu_shadow"].update({f: v}))
        for mutate in mutations:
            value = copy.deepcopy(row)
            mutate(value)
            with self.subTest(value=value), self.assertRaises(proof.GridProofError):
                proof.validate_shadow(value)

    def test_closed_json_rejects_duplicate_and_nonfinite_evidence(self):
        for text in ('{"passed":false,"passed":true}', '{"delta":NaN}', '{"delta":Infinity}'):
            with self.subTest(text=text), self.assertRaises(proof.GridProofError):
                proof.strict_json(text)

    def test_only_bundle_measured_duration_is_excluded_from_gate_semantics(self):
        from scripts.research.compare_resident_finishes import semantic_agreement
        left = {"elapsed_ms": 15., "time_budget_s": 30., "applied": True, "reason": "validated"}
        right = dict(left, elapsed_ms=39.)
        path = "bundle_adjustment.propose_bundle_poses/result"
        self.assertTrue(semantic_agreement(left, right, path))
        for changed in (dict(right, time_budget_s=40.), dict(right, applied=False), dict(right, reason="timeout")):
            with self.subTest(changed=changed):
                self.assertFalse(semantic_agreement(left, changed, path))
        self.assertFalse(semantic_agreement(left, right, "fragments.propose_fragment_poses/result"))

    def test_canonical_mapping_preserves_exact_ids_with_raw_order_diagnostic(self):
        from scripts.research.compare_resident_finishes import semantic_agreement
        left = {"correspondence_set": {"dtype": "<i4", "shape": [100, 2], "sha256": "a" * 64},
                "correspondence_mapping": {"dtype": "<i4", "shape": [100, 2], "sha256": "b" * 64},
                "fitness": .9}
        reordered = copy.deepcopy(left)
        reordered["correspondence_set"]["sha256"] = "c" * 64
        self.assertTrue(semantic_agreement(left, reordered))
        changed_id = copy.deepcopy(reordered)
        changed_id["correspondence_mapping"]["sha256"] = "d" * 64
        self.assertFalse(semantic_agreement(left, changed_id))


class FinishScopedDispatchTests(unittest.TestCase):
    def setUp(self):
        from scripts.research import finish_resident_registration as scope_module
        self.scope_module = scope_module
        self.original_hook = lambda *args: None
        self.registration = SimpleNamespace(match=self.original_hook)
        self.native_calls = []
        self.native_result = object()

        def original_match(*args):
            routed = self.registration.match(*args)
            if routed is not None:
                return routed
            self.native_calls.append(args)
            return self.native_result

        self.original_match = original_match

    def make_scope(self, solver, **kwargs):
        return self.scope_module.FinishRegistrationScope(
            self.registration, self.original_match, solver, capture_inputs=fake_capture,
            result_summary=lambda *args: {"mocked_result": "finite"}, compare_results=good_comparison, **kwargs)

    def test_cpu_shadow_bypasses_resident_recursion_and_preserves_return_identity(self):
        solver, rows = FakeSolver(), []
        original_fallback = solver.cpu_fallback
        with self.make_scope(solver, trace=rows.append) as context:
            actual = self.registration.match("a", "target", "seed")
            self.assertIs(actual, solver.result)
            self.assertEqual(solver.match_calls, 1)
            self.assertEqual(self.native_calls, [("a", "target", "seed")])
            context.finish()
        self.assertIs(self.registration.match, self.original_hook)
        self.assertIs(solver.cpu_fallback, original_fallback)
        self.assertTrue(solver.closed)
        self.assertTrue(context.report()["complete"])
        self.assertEqual(rows[0]["inputs_after_shadow"], rows[0]["call_signature"])

    def test_native_mode_runs_original_once_without_resident_or_cpu_recursion(self):
        rows = []
        with self.make_scope(None, mode="native", trace=rows.append) as context:
            self.assertIs(self.registration.match("a", "target", "seed"), self.native_result)
            context.finish()
        self.assertEqual(len(self.native_calls), 1)
        self.assertEqual(rows[0]["query_statistics_delta"], {})
        self.assertIs(self.registration.match, self.original_hook)

    def test_timing_rejects_scope_drift_before_solver_and_missing_suffix_at_exit(self):
        first = fake_capture("a", None, None, None, (), None)
        second = fake_capture("b", None, None, None, (), None)
        for wrong in ("b",):
            solver = FakeSolver()
            with self.assertRaises(self.scope_module.FinishResearchFailure):
                with self.make_scope(solver, mode="timing", authority=authority([first])):
                    self.registration.match(wrong, "target", "seed")
            self.assertEqual(solver.match_calls, 0)
            self.assertIs(self.registration.match, self.original_hook)
        solver = FakeSolver()
        with self.assertRaises(self.scope_module.FinishResearchFailure):
            with self.make_scope(solver, mode="timing", authority=authority([first, second])) as context:
                self.registration.match("a", "target", "seed")
                context.finish()
        self.assertEqual(solver.match_calls, 1)
        self.assertIs(self.registration.match, self.original_hook)

    def test_swallowed_fault_and_sync_cleanup_cannot_produce_successful_context(self):
        primary = ValueError("GPU work failed")
        solver = FakeSolver(error=primary, close_error=RuntimeError("cleanup sync failed"))
        context = self.make_scope(solver)
        with self.assertRaises(self.scope_module.FinishResearchFailure) as caught:
            with context:
                try:
                    self.registration.match("a", "target", "seed")
                except BaseException as fault:
                    self.assertIs(fault, context.failure)
                    # Model _stage cleanup masking the research error followed by
                    # the engine's ordinary registration recovery swallowing it.
                    try:
                        raise RuntimeError("post-body CUDA synchronize failed")
                    except Exception:
                        pass
                try:
                    context.finish()
                except BaseException:
                    pass
        self.assertIs(caught.exception, context.failure)
        self.assertIs(caught.exception.__cause__, primary)
        self.assertTrue(any("cleanup sync failed" in note for note in caught.exception.__notes__))
        self.assertIs(self.registration.match, self.original_hook)
        self.assertFalse(context.report()["complete"])

    def test_full_call_retry_is_hard_failure_and_latch_prevents_further_gpu_work(self):
        solver = FakeSolver()
        solver.match = lambda *args: solver.cpu_fallback(*args)
        context = self.make_scope(solver)
        with self.assertRaises(self.scope_module.FinishResearchFailure):
            with context:
                try:
                    self.registration.match("a", "target", "seed")
                except self.scope_module.FinishResearchFailure:
                    pass
                with self.assertRaises(self.scope_module.FinishResearchFailure):
                    self.registration.match("b", "target", "seed")
        self.assertEqual(context.full_cpu_fallback_calls, 1)
        self.assertEqual(self.native_calls, [])
        self.assertIs(self.registration.match, self.original_hook)

    def test_failed_entry_restores_hook_and_closes_owned_solver(self):
        class BrokenSolver:
            closed = False

            @property
            def cpu_fallback(self):
                raise ValueError("cannot read fallback")

            def close(self):
                self.closed = True

        solver = BrokenSolver()
        with self.assertRaises(BaseException):
            with self.make_scope(solver):
                self.fail("Broken context entry was accepted")
        self.assertIs(self.registration.match, self.original_hook)
        self.assertTrue(solver.closed)

    def test_gate_recording_failure_latches_even_when_engine_catches_exception(self):
        solver = FakeSolver()
        gate = SimpleNamespace(verify=lambda: {"accepted": True})
        original_gate = gate.verify
        context = self.make_scope(solver)
        with patch.object(self.scope_module, "evidence", side_effect=ValueError("cannot record evidence")):
            with self.assertRaises(self.scope_module.FinishResearchFailure):
                with context:
                    context.observe(gate, "verify")
                    self.registration.match("a", "target", "seed")
                    try:
                        gate.verify()
                    except Exception:
                        pass
                    context.finish()
        self.assertIs(gate.verify, original_gate)
        self.assertIs(self.registration.match, self.original_hook)

    def test_original_gate_exception_is_not_relabeled_as_cuda_failure(self):
        original_error = ValueError("original ordinary rejection")

        def reject():
            raise original_error

        gate = SimpleNamespace(verify=reject)
        solver = FakeSolver()
        with self.make_scope(solver) as context:
            context.observe(gate, "verify")
            self.registration.match("a", "target", "seed")
            with self.assertRaises(ValueError) as caught:
                gate.verify()
            self.assertIs(caught.exception, original_error)
            self.assertIsNone(context.failure)
            context.finish()
        self.assertIs(gate.verify, reject)


class FinishClosedProofTests(unittest.TestCase):
    """Exercise field closure; separately tested device proofs are mocked inputs."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.artifacts = {}
        for name in proof.FINISH_ARTIFACTS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("artificial source contract fixture: " + name, encoding="utf-8")
            self.artifacts[name] = proof.file_hash(path)
        runtime = {"source_sha256": "9" * 64, "artifacts_sha256": {}}
        synthetic, bridge = self.write("synthetic.json", {"artificial_component": True}), self.write("bridge.json", {})
        self.component_proof = {"synthetic": synthetic, "bridge": bridge}
        self.token = DeviceFlatGridProofAuthority(proof.canonical_hash(runtime), "1" * 64,
            synthetic["sha256"], bridge["sha256"], frozenset({"0" * 64}), (), json.dumps(runtime))
        archive = self.root / "raw.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("manifest.json", json.dumps({"frames": [{}, {}, {}]}))
        self.scope = {"archive": {"path": str(archive), "sha256": proof.file_hash(archive)},
            "selected_indices": [0, 1, 2], "seed": 0, "runtime_binding": runtime,
            "settings": {"camera": {}, "sensor_calibration": None, "voxel_m": .01,
                         "final_voxel_m": .005, "final_block_count": 10000},
            "pipeline_options": {"KINECT_CUDA_REGISTRATION": "cpu", "research_final_preplan": False},
            "thread_policy": dict(proof.THREAD_POLICY),
            "live": {"accepted_indices": [0, 1, 2], "unprocessed_count": 0,
                     "decisions_sha256": "2" * 64, "poses_sha256": "3" * 64, "pose_source": "fresh raw Live replay"}}
        call = payload()
        signature = proof.call_signature(call)
        counts = {key: 0 for key in proof.QUERY_COUNTERS}
        counts.update(query_rows=10, direct_gpu_hits=6, declared_gpu_misses=4, audited_hits=6, audited_misses=4,
                      device_calls=1, device_flagged_rows=10, device_query_download_rows=10, device_query_download_bytes=640)
        self.match = {"event": "match", "complete": True, "call_index": 0, "call_inputs": call,
            "call_signature": signature, "inputs_after_resident": signature, "inputs_before_shadow": signature,
            "inputs_after_shadow": signature, "resident_full_call_fallbacks": 0,
            "cpu_shadow": good_comparison(), "query_statistics_delta": counts}
        gate = {"event": "gate", "gate_index": 0, "function": "verify", "complete": True, "result": None}
        candidate_trace = self.write_trace("candidate.jsonl", [self.match, gate])
        native_trace = self.write_trace("native.jsonl", [dict(self.match), gate])
        registration = {"complete": True, "restored": True, "calls": 1, "call_signatures": [signature],
            "input_immutability_passed": True, "cpu_shadow_calls": 1, "cpu_shadow_failures": 0,
            "full_cpu_fallback_calls": 0, "target_digests": ["a" * 64], "failure": None, "cleanup_failures": [],
            "retrieval_statistics": counts, "resident": {"statistics": {"calls": 1}}, "gate_calls": 1}
        candidate, native = self.profile("candidate"), self.profile("native")
        self.audit = self.sidecar("audit", candidate, candidate_trace, registration)
        native_registration = dict(registration, cpu_shadow_calls=0, resident=None, retrieval_statistics={})
        native_sidecar = self.write("native.resident.json", self.sidecar("native", native, native_trace, native_registration))
        self.audit_record = self.write("audit.resident.json", self.audit)
        self.quality = {"kind": "offline-full-finish-device-flat-quality-v1", "status": "passed",
            "audit_report_sha256": self.audit_record["sha256"], "audit": self.audit_record,
            "candidate": candidate, "native": native, "native_sidecar": native_sidecar,
            "candidate_trace": candidate_trace, "native_trace": native_trace,
            "live_pose_signatures_equal": True, "live_decision_signatures_equal": True,
            "same_ordered_gate_decisions": True, "ordered_gate_evidence_passed": True,
            "comparison": {"candidate_report_sha256": candidate["sha256"], "baseline_report_sha256": native["sha256"],
                "same_mesh_success": True, "same_accepted_indices": True,
                "triangle_surface_metrics_vs_cpu": {"threshold_m": .005, "samples_per_surface": 30000,
                    "alignment": "fixed input coordinate frame; no scale or trajectory fitting",
                    "surface_p95_m": .0001, "precision": 1., "completeness": 1.}}}
        self.quality_record = self.write("quality.json", self.quality)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
        return {"path": str(path), "sha256": proof.file_hash(path)}

    def write_trace(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return {"path": str(path), "sha256": proof.file_hash(path), "rows": len(rows)}

    def profile(self, name):
        geometry = self.root / (name + ".geometry.npz")
        geometry.write_bytes(b"artificial geometry contract fixture")
        value = {"input_sha256": self.scope["archive"]["sha256"], "source_sha256": "9" * 64,
            "input_changed_during_profile": False, "source_changed_during_profile": False,
            "finish_requested": True, "mesh_built": True, "pose_seeds_used": False, "frames": 3,
            "selected_indices": [0, 1, 2], "seed": 0, "settings": self.scope["settings"],
            "research_pipeline_options": self.scope["pipeline_options"], "pipeline_options": {"KINECT_CUDA_REGISTRATION": "cpu"},
            "accepted_indices_before_finish": [0, 1, 2], "accepted_indices": [0, 1, 2],
            "final_reconstruction": {"voxel_m": .005},
            "geometry": {"vertices": 3, "triangles": 1, "artifact": str(geometry)}}
        result = self.write(name + ".json", value)
        result.update(geometry_path=str(geometry), geometry_sha256=proof.file_hash(geometry))
        return result

    def sidecar(self, mode, profile, trace, registration):
        runtime = self.token.runtime_binding
        return {"kind": proof.KIND, "mode": mode, "status": "complete", "performance_attribution_valid": mode != "audit",
            "runner_hooks_restored": True, "source_sha256": "9" * 64, "source_sha256_after": "9" * 64,
            "archive_sha256": self.scope["archive"]["sha256"], "archive_sha256_after": self.scope["archive"]["sha256"],
            "runtime_binding": runtime, "runtime_binding_after": runtime,
            "scope_binding": self.scope, "scope_binding_sha256": proof.canonical_hash(self.scope),
            "artifacts_sha256": self.artifacts, "artifacts_sha256_after": self.artifacts,
            "component_proof": self.component_proof, "component_proof_after": self.component_proof,
            "registration": registration, "trace": trace, "profile": profile}

    def validate(self):
        with patch.object(proof, "ROOT", self.root), \
             patch.object(proof, "validate_configuration", return_value={}), \
             patch.object(proof, "validate_current_artifacts"), \
             patch.object(proof, "resident_report_checks", return_value={}):
            return proof.validate_finish_proof(self.audit_record["path"], self.token, self.scope,
                                               self.artifacts, quality_path=self.quality_record["path"])

    def rewrite_audit(self):
        self.audit_record = self.write("audit.resident.json", self.audit)
        self.quality.update(audit=self.audit_record, audit_report_sha256=self.audit_record["sha256"])
        self.quality_record = self.write("quality.json", self.quality)

    def test_complete_closed_scope_mints_only_fresh_targets_and_order(self):
        token = self.validate()
        self.assertIs(type(token), proof.FinishResidentProofAuthority)
        self.assertEqual(token.target_digests, frozenset({"a" * 64}))
        self.assertNotIn("0" * 64, token.target_digests)
        self.assertEqual(token.expected_call_signatures, (self.match["call_signature"],))

    def test_old_target_partial_query_or_failed_cleanup_cannot_mint_authority(self):
        original = copy.deepcopy(self.audit)
        mutations = (
            lambda r: r["registration"].update(target_digests=["0" * 64]),
            lambda r: r["registration"].update(restored=False),
            lambda r: r.update(runner_hooks_restored=False),
            lambda r: r["registration"].update(cleanup_failures=["CUDA synchronize failed"]),
            lambda r: r["registration"].update(cpu_shadow_calls=0),
            lambda r: r["registration"]["retrieval_statistics"].update(audited_hits=5),
        )
        for mutate in mutations:
            self.audit = copy.deepcopy(original)
            mutate(self.audit)
            self.rewrite_audit()
            with self.subTest(audit=self.audit), self.assertRaises(proof.GridProofError):
                self.validate()

    def test_quality_cannot_weaken_fixed_gate_or_use_stale_geometry(self):
        original = copy.deepcopy(self.quality)
        changes = ({"surface_p95_m": .0005001}, {"precision": .9989}, {"completeness": .9989},
                   {"threshold_m": .01}, {"samples_per_surface": 1000}, {"alignment": "best-fit ICP"})
        for change in changes:
            self.quality = copy.deepcopy(original)
            self.quality["comparison"]["triangle_surface_metrics_vs_cpu"].update(change)
            self.quality_record = self.write("quality.json", self.quality)
            with self.subTest(change=change), self.assertRaises(proof.GridProofError):
                self.validate()
        self.quality = original
        self.quality_record = self.write("quality.json", self.quality)
        Path(self.quality["native"]["geometry_path"]).write_bytes(b"changed after quality closed")
        with self.assertRaises(proof.GridProofError):
            self.validate()

    def test_replay_subset_or_archived_pose_provenance_cannot_claim_full_scope(self):
        for changed in ({"selected_indices": [0, 1]},
                        {"live": dict(self.scope["live"], pose_source="archived ZIP poses")},
                        {"live": dict(self.scope["live"], unprocessed_count=1)}):
            scope = copy.deepcopy(self.scope)
            scope.update(changed)
            with self.subTest(changed=changed), self.assertRaises(proof.GridProofError):
                proof.validate_scope(scope, self.token.runtime_binding, {})

    def test_output_changes_during_validation_are_detected(self):
        def mutate_after_profile_checks(*args):
            Path(self.quality["candidate"]["geometry_path"]).write_bytes(b"changed during validation")

        calls = 0

        def current(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                mutate_after_profile_checks()

        with patch.object(proof, "ROOT", self.root), \
             patch.object(proof, "validate_configuration", return_value={}), \
             patch.object(proof, "resident_report_checks", return_value={}), \
             patch.object(proof, "validate_current_artifacts", side_effect=current):
            with self.assertRaisesRegex(proof.GridProofError, "changed during validation"):
                proof.validate_finish_proof(self.audit_record["path"], self.token, self.scope,
                                           self.artifacts, quality_path=self.quality_record["path"])


if __name__ == "__main__":
    unittest.main()
