"""Stdlib-only mocked proof-contract checks; never numerical/GPU authority.

Only current production/artifact I/O is mocked in report-contract tests. The
actual report validator, metadata, nine-proposal and audit checks execute.
Pinned-build checks use tiny temporary files and mock the broad core scan.
"""
import argparse
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_device_flat_grid_proof as guard


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def specimen():
    artifacts = {name: sha(name) for name in guard.REQUIRED_ARTIFACTS}
    cfg = dict(device="CUDA:0", max_points=1_000_000, max_scratch_bytes=256*1024**2,
               max_query_bytes=136*1024**2, gpu_timing=False)
    math = dict(script_sha256=artifacts["scripts/research_resident_icp.py"],
                gpu_source_sha256=sha("original GPU literal"),
                cpu_bridge_source_sha256=artifacts["benchmark-output/cuda-pipeline/resident-icp/resident_solve.cpp"],
                solve_library_sha256=artifacts[guard.SOLVE_DLL], solve_manifest_sha256=artifacts[guard.SOLVE_MANIFEST])
    binding = dict(source_sha256=guard.FROZEN, component_source_sha256=sha("component math"),
                   domain=dict(version=guard.DOMAIN_VERSION), gpu=dict(name="mock GPU", driver="mock"),
                   thread_policy=dict(open3d=20, opencv=20, omp="8"),
                   cache_policy=dict(max_clouds=64, retained_gpu_bytes=256*1024**2),
                   resident_configuration=cfg, resident_math=math, artifacts_sha256=artifacts)
    manifest = dict(eigen_commit=guard.EIGEN_COMMIT, eigen_archive_sha256=guard.EIGEN_ARCHIVE,
                    eigen_headers_sha256=guard.EIGEN_HEADERS, verified_header_files=366,
                    bridge_source_sha256=math["cpu_bridge_source_sha256"], library_sha256=math["solve_library_sha256"])
    metadata = dict(manifest, reported_eigen_version="3.4.0")
    resident = dict(device=cfg["device"], gpu_timing=False, scratch_limit_bytes=cfg["max_scratch_bytes"],
                    query_scratch_limit_bytes=cfg["max_query_bytes"], max_points=cfg["max_points"],
                    statistics=dict(cpu_fallback_calls=0), fallback_reasons={},
                    uncollected_statistics=list(guard.UNCOLLECTED_RESIDENT_EVENTS),
                    provenance=dict(source_unchanged=True, actual_retrieval_unchanged=True,
                        original_resident_math_unchanged=True, source_sha256=guard.FROZEN,
                        script_sha256=math["script_sha256"], gpu_source_sha256=math["gpu_source_sha256"],
                        original_gpu_source_sha256=math["gpu_source_sha256"],
                        cpu_bridge_source_sha256=math["cpu_bridge_source_sha256"],
                        original_cpu_bridge_source_sha256=math["cpu_bridge_source_sha256"], solve_metadata=metadata,
                        actual_retrieval_sha256={name: artifacts[name] for name in (
                            "scripts/cuda_device_flat_grid_registration.py", "scripts/research_device_flat_grid_nn.cu",
                            "scripts/cuda_flat_grid_registration.py", "scripts/research_flat_grid_nn.cu",
                            "scripts/validate_device_flat_grid_proof.py")},
                        original_resident_math_sha256={name: artifacts[name] for name in (
                            "scripts/research_resident_icp.py", "scripts/research_device_grid_resident_icp.py")}))
    common = dict(kind=guard.KIND, traversal=guard.TRAVERSAL, status="passed",
        performance_attribution_valid=False, cpu_hit_audit=True, cpu_miss_audit=True,
        miss_policy="direct-miss-research-v1", proof_bindings=binding, proof_bindings_after=binding,
        proof_bindings_sha256=guard.canonical_hash(binding), source_sha256=guard.FROZEN,
        source_sha256_after=guard.FROZEN, component_source_sha256=binding["component_source_sha256"],
        component_source_sha256_after=binding["component_source_sha256"], artifacts_sha256=artifacts,
        artifacts_sha256_after=artifacts, gpu=binding["gpu"], gpu_after=binding["gpu"], domain=binding["domain"],
        cleanup_passed=True, host_enclosure=dict(unsupported_guard_passed=True, packing_unique_cases=1,
            domain_boundary_cases=[dict(radius=r, passed=True, supported_boundary_pairs=1)
                                   for r in (2.**-20, .03, .06, .12, .125, 1., .25)]),
        device="CUDA:0", resident_configuration=cfg, resident_configuration_after=cfg,
        resident_math=math, resident_math_after=math, solve_metadata=metadata, device_resident=resident)
    stats = dict(query_rows=10, direct_gpu_hits=4, declared_gpu_misses=4, audited_hits=4, audited_misses=4,
                 audit_index_mismatches=0, audit_false_misses=0, device_calls=1, device_malformed_results=0,
                 device_flagged_rows=10, device_query_download_rows=10, device_query_download_bytes=640,
                 resident_xyz_uploads=1)
    synthetic = copy.deepcopy(common)
    synthetic.update(synthetic_only=True, real_runs=[], synthetic_device_adapter_checked=True,
        synthetic_indices=[dict(queries=1, index_mismatches=0, device_adapter_index_mismatches=0,
                                device_adapter_max_squared_distance_delta=0.) for _ in range(16)],
        synthetic_grid_boundaries=[dict(queries=1, index_mismatches=0,
            case="original-double-subnormal-and-minimum-radius" if i == 0 else str(i)) for i in range(43)],
        synthetic_statistics=copy.deepcopy(stats), device_flat_synthetic=dict(
            passed=True, required_checks=list(guard.DEVICE_CHECKS), classifier_cases=32,
            classifier_rows=5128, packet_rows=1,
            positive_mask_coverage=dict(unsupported=1, uncertain=1, audited_hits=1, audited_misses=1),
            **{name: True for name in guard.DEVICE_CHECKS}),
        flat_cache_ownership=dict(passed=True, cloud_evictions=1, bounded_retained_bytes=True,
            clear_releases_all_owned_arrays=True, one_byte_budget_original_cpu_fallback=True))
    fixture = dict(fixture_sha256=sha("fixture"), reference_sha256=sha("reference"),
                   original_raw_input_sha256=sha("raw"), tasks=[
                       dict(position=p, pair=pair, proposal_sha256=[sha(f"{p}/{i}") for i in range(n)])
                       for p, pair, n in ((0, [8, 12], 5), (4, [0, 2], 4))])
    pairs = [dict(position=t["position"], pair=t["pair"], pair_verdict_same=True,
        verdict=dict(accepted=t["position"] == 0), proposal_results=[dict(
            proposal_index=i, input_sha256=d, quality=dict(passed=True))
            for i, d in enumerate(t["proposal_sha256"])]) for t in fixture["tasks"]]
    bridge = copy.deepcopy(common)
    bridge.update(synthetic_only=False, fixture_binding=fixture, fixture_binding_sha256=guard.canonical_hash(fixture),
        fixture_arrays_unchanged=True, cloud_arrays_unchanged=True,
        fixture_provenance=dict(local_pose_source="measured Finish fragment report; no archived ZIP poses"),
        original_fixture_authority=dict(same_decisions_and_support=True), real_target_membership_sha256=[sha("target")],
        real_runs=[dict(mode="native_cpu", complete=True, pairs=copy.deepcopy(pairs)),
            dict(mode="grid", complete=True, pairs=copy.deepcopy(pairs), statistics_delta=copy.deepcopy(stats),
                 resident_statistics_delta=dict(calls=1, pose_iterations=1, cpu_fallback_calls=0),
                 resident_uncollected_statistics=list(guard.UNCOLLECTED_RESIDENT_EVENTS), device_resident=copy.deepcopy(resident))])
    for field in ("fixture_sha256", "reference_sha256", "original_raw_input_sha256"):
        bridge[field] = bridge[field+"_after"] = fixture[field]
    return binding, fixture, manifest, synthetic, bridge


class Contract(unittest.TestCase):
    def validate(self, mutation=None, io_effect=None):
        binding, fixture, manifest, synthetic, bridge = specimen()
        if mutation:
            mutation(binding, fixture, synthetic, bridge)
        with tempfile.TemporaryDirectory(prefix="device-proof-contract-") as folder:
            sp, bp = Path(folder)/"synthetic.json", Path(folder)/"bridge.json"
            sp.write_text(json.dumps(synthetic), encoding="utf-8")
            bp.write_text(json.dumps(bridge), encoding="utf-8")
            with patch.object(guard, "validate_current_artifacts", return_value=manifest, side_effect=io_effect):
                return guard.validate_grid_proof(sp, bp, binding, fixture)

    def test_valid_distinct_immutable_authority(self):
        authority = self.validate()
        self.assertIs(type(authority), guard.DeviceFlatGridProofAuthority)
        bound = authority.runtime_binding
        bound["resident_configuration"]["device"] = "CUDA:1"
        self.assertEqual(authority.runtime_binding["resident_configuration"]["device"], "CUDA:0")
        with self.assertRaises(FrozenInstanceError):
            authority.runtime_binding_json = "{}"

    def test_mixed_host_device_synthetic_not_all_query_copy_required(self):
        def mixed(_, __, synthetic, ___):
            synthetic["synthetic_statistics"].update(device_flagged_rows=2, device_query_download_rows=2,
                                                      device_query_download_bytes=128)
        self.validate(mixed)

    def test_warm_resident_run_can_upload_zero(self):
        def warm(_, __, ___, bridge):
            repeat = copy.deepcopy(bridge["real_runs"][1])
            repeat["statistics_delta"]["resident_xyz_uploads"] = 0
            bridge["real_runs"].append(repeat)
        self.validate(warm)

    def test_artifact_change_after_work_rejected(self):
        with self.assertRaisesRegex(guard.GridProofError, "changed after work"):
            self.validate(io_effect=[specimen()[2], guard.GridProofError("changed after work")])

    def test_proof_bytes_change_rejected(self):
        original = guard.file_hash
        count = 0
        def changed(path):
            nonlocal count
            count += 1
            return original(path) if count <= 2 else sha("changed report")
        with patch.object(guard, "file_hash", side_effect=changed):
            with self.assertRaisesRegex(guard.GridProofError, "Proof files changed"):
                self.validate()

    def test_pinned_build_and_current_bytes(self):
        binding, _, manifest, _, _ = specimen()
        with tempfile.TemporaryDirectory(prefix="device-proof-build-") as folder:
            root = Path(folder)
            mp = root/guard.SOLVE_MANIFEST
            mp.parent.mkdir(parents=True)
            mp.write_text(json.dumps(manifest), encoding="utf-8")
            tiny = root/"tiny-source.py"
            tiny.write_text("original bytes", encoding="utf-8")
            artifacts = {"tiny-source.py": guard.file_hash(tiny)}
            with patch.object(guard, "ROOT", root), patch.object(guard, "current_core_hash", return_value=guard.FROZEN):
                self.assertEqual(guard.validate_current_artifacts(binding, artifacts), manifest)
                tiny.write_text("changed bytes", encoding="utf-8")
                with self.assertRaisesRegex(guard.GridProofError, "artifact changed"):
                    guard.validate_current_artifacts(binding, artifacts)
                artifacts["tiny-source.py"] = guard.file_hash(tiny)
                manifest["eigen_headers_sha256"] = sha("wrong headers")
                mp.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(guard.GridProofError, "Pinned official Eigen"):
                    guard.validate_current_artifacts(binding, artifacts)


def negative(name, mutation):
    def check(self):
        with self.assertRaises(guard.GridProofError):
            self.validate(mutation)
    check.__name__ = "test_reject_"+name
    setattr(Contract, check.__name__, check)


def set_path(root, path, value):
    for key in path[:-1]:
        root = root[key]
    root[path[-1]] = value


for name, scope, path, value in (
    ("old_host_kind", 2, ("kind",), "standalone-original-double-flat-grid"),
    ("old_traversal", 3, ("traversal",), "flat-full-radius-v1"),
    ("event_policy", 0, ("resident_configuration", "gpu_timing"), True),
    ("wrong_device", 0, ("resident_configuration", "device"), "CUDA:1"),
    ("too_many_points", 0, ("resident_configuration", "max_points"), 1_000_001),
    ("boolean_points", 0, ("resident_configuration", "max_points"), True),
    ("scratch_cap", 0, ("resident_configuration", "max_scratch_bytes"), 256*1024**2+1),
    ("query_cap", 0, ("resident_configuration", "max_query_bytes"), 136*1024**2+1),
    ("cache_cap", 0, ("cache_policy", "retained_gpu_bytes"), 256*1024**2+1),
    ("thread_policy", 0, ("thread_policy", "omp"), "4"),
    ("unaudited_hit", 3, ("cpu_hit_audit",), False),
    ("unaudited_miss", 3, ("cpu_miss_audit",), False),
    ("synthetic_adapter_unchecked", 2, ("synthetic_device_adapter_checked",), False),
    ("synthetic_index_error", 2, ("synthetic_indices", 0, "device_adapter_index_mismatches"), 1),
    ("synthetic_nan_metric", 2, ("synthetic_indices", 0, "device_adapter_max_squared_distance_delta"), float("nan")),
    ("classifier_check_absent", 2, ("device_flat_synthetic", "audit_before_correction"), False),
    ("classifier_tail_missing", 2, ("device_flat_synthetic", "classifier_rows"), 5127),
    ("classifier_zero_miss_coverage", 2, ("device_flat_synthetic", "positive_mask_coverage", "audited_misses"), 0),
    ("missing_cache_ownership", 2, ("flat_cache_ownership", "clear_releases_all_owned_arrays"), False),
    ("archived_seed", 3, ("fixture_provenance", "local_pose_source"), "ZIP archived pose"),
    ("changed_raw", 3, ("original_raw_input_sha256_after",), sha("changed raw")),
    ("unknown_mode", 3, ("real_runs", 0, "mode"), "approximate"),
    ("proposal_reordered", 3, ("real_runs", 1, "pairs", 0, "proposal_results", 0, "proposal_index"), 4),
    ("rejected_pair_accepted", 3, ("real_runs", 1, "pairs", 1, "verdict", "accepted"), True),
    ("witness_gate", 3, ("real_runs", 1, "pairs", 0, "proposal_results", 0, "quality", "passed"), False),
    ("hit_audit_short", 3, ("real_runs", 1, "statistics_delta", "audited_hits"), 3),
    ("false_miss", 3, ("real_runs", 1, "statistics_delta", "audit_false_misses"), 1),
    ("malformed_device_result", 3, ("real_runs", 1, "statistics_delta", "device_malformed_results"), 1),
    ("incomplete_query_shadow", 3, ("real_runs", 1, "statistics_delta", "device_query_download_rows"), 9),
    ("wrong_packet_bytes", 3, ("real_runs", 1, "statistics_delta", "device_query_download_bytes"), 240),
    ("zero_xyz_uploads", 3, ("real_runs", 1, "statistics_delta", "resident_xyz_uploads"), 0),
    ("no_resident_iterations", 3, ("real_runs", 1, "resident_statistics_delta", "pose_iterations"), 0),
    ("full_call_fallback", 3, ("real_runs", 1, "resident_statistics_delta", "cpu_fallback_calls"), 1),
    ("disabled_event_zero_in_delta", 3, ("real_runs", 1, "resident_statistics_delta", "gpu_transform_ms"), 0.),
    ("delta_uncollected_missing", 3, ("real_runs", 1, "resident_uncollected_statistics"), []),
    ("disabled_event_zero_cumulative", 3, ("device_resident", "statistics", "gpu_equations_ms"), 0.),
    ("original_math_changed", 3, ("device_resident", "provenance", "original_resident_math_unchanged"), False),
    ("actual_solve_version_missing", 3, ("solve_metadata", "reported_eigen_version"), ""),
    ("target_membership_absent", 3, ("real_target_membership_sha256",), []),
    ("old_domain", 0, ("domain", "version"), "flat-dyadic27-original-double-v1"),
    ("escaping_artifact", 0, ("artifacts_sha256", "../outside.py"), sha("outside")),
    ("missing_cpp_pin", 0, ("artifacts_sha256", "benchmark-output/cuda-pipeline/resident-icp/resident_solve.cpp"), "bad"),
):
    negative(name, lambda *items, scope=scope, path=path, value=value: set_path(items[scope], path, value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Require fresh contract report; preserve earlier evidence")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contract)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = dict(kind="mocked-stdlib-device-resident-proof-contract", status="passed" if result.wasSuccessful() else "failed",
                  tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  numerical_authority=False, numerical_imports=False, hardware_calls=False,
                  mocked_scope="Current artifact/core I/O in report tests only; tiny temporary files for pinned-build tests",
                  script_sha256=guard.file_hash(Path(__file__)), validator_sha256=guard.file_hash(Path(guard.__file__)))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
