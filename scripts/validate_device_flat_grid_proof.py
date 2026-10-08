"""Pure-stdlib authority for a freshly audited device-flat resident trajectory.

Old host-grid/OptiX results and kernel-only samples cannot authorize this path.
The caller supplies freshly reconstructed actual runtime/configuration bindings;
this validator also verifies current on-disk sources and the pinned solve files.
No numerical imports, CUDA initialization, compilation or source writes occur.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from scripts.validate_flat_grid_proof import FlatGridProofAuthority
from scripts.validate_uniform_grid_proof import (
    GridProofError, audit_coverage, canonical_hash, common_checks as original_common_checks,
    file_hash, positive, require, zero)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a"
KIND = "standalone-original-double-device-flat-resident"
TRAVERSAL = "device-flat-resident-v1"
DOMAIN_VERSION = "resident-device-flat-dyadic27-original-double-v1"
SOLVE_DLL = "benchmark-output/cuda-pipeline/resident-icp/resident_solve.dll"
SOLVE_MANIFEST = "benchmark-output/cuda-pipeline/resident-icp/resident_solve.build.json"
EIGEN_COMMIT = "da7909592376c893dabbc4b6453a8ffe46b1eb8e"
EIGEN_ARCHIVE = "37f71e1d7c408e2cc29ef90dcda265e2de0ad5ed6249d7552f6c33897eafd674"
EIGEN_HEADERS = "043d5bf81ab89615257f297638bfe749867f1593c656610178be4c5a611f7b7a"
DEVICE_CHECKS = (
    "classifier_masks_exact", "packet_exact", "scatter_metrics_exact", "audit_before_correction",
    "finite_transport_hard_failure", "malformed_raw_hard_failure", "policy_mutation_rejected",
    "query_budget_rejected", "input_contract_rejected", "cache_xyz_exact", "cache_release_clears_xyz",
    "mutation_rebuilds_xyz", "budget_cpu_only_no_xyz", "no_full_query_download")
UNCOLLECTED_RESIDENT_EVENTS = ("gpu_transform_ms", "gpu_equations_ms")
REQUIRED_ARTIFACTS = (
    "scripts/cuda_device_flat_grid_registration.py", "scripts/research_device_flat_grid_nn.cu",
    "scripts/device_flat_grid_synthetic.py", "scripts/research_device_grid_resident_icp.py",
    "scripts/validate_device_flat_grid_proof.py", "scripts/benchmark_device_grid_resident.py",
    "scripts/benchmark_device_grid_resident_timing.py", "scripts/DEVICE_FLAT_GRID_RESEARCH.md",
    "scripts/research_resident_icp.py", "scripts/cuda_flat_grid_registration.py",
    "scripts/research_flat_grid_nn.cu", "scripts/validate_flat_grid_proof.py",
    "scripts/cuda_uniform_grid_registration.py", "scripts/research_uniform_grid_nn.cu",
    "scripts/validate_uniform_grid_proof.py", "scanner_server/cuda_nn_registration.py",
    SOLVE_DLL, SOLVE_MANIFEST, "benchmark-output/cuda-pipeline/resident-icp/resident_solve.cpp")


@dataclass(frozen=True)
class DeviceFlatGridProofAuthority(FlatGridProofAuthority):
    """Separate immutable token; inherited flat authority is insufficient."""
    runtime_binding_json: str

    @property
    def runtime_binding(self):
        # Return a fresh object: modifying it cannot change the frozen authority.
        return json.loads(self.runtime_binding_json)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def normalized_artifacts(mapping):
    require(isinstance(mapping, dict) and mapping, "Actual source/solve artifacts absent")
    result = {}
    for name, value in mapping.items():
        require(isinstance(name, str) and digest(value), "Malformed source/solve fingerprint")
        canonical = name.replace("\\", "/")
        require(canonical not in result, "Aliased duplicate artifact names")
        path = (ROOT / canonical).resolve()
        require(path.is_relative_to(ROOT.resolve()), "Artifact path escapes the research workspace")
        result[canonical] = value
    return result


def validate_configuration(binding):
    require(binding.get("source_sha256") == FROZEN, "Require the frozen9331 source")
    require(binding.get("domain", {}).get("version") == DOMAIN_VERSION, "Old retrieval domain cannot authorize device residency")
    require(binding.get("thread_policy") == {"open3d": 20, "opencv": 20, "omp": "8"}, "Original thread policy changed")
    cfg = binding["resident_configuration"]
    require(cfg.get("device") == "CUDA:0" and cfg.get("gpu_timing") is False,
            "Require actual CUDA0 and the proven event-disabled resident policy")
    require(positive(cfg.get("max_points")) and cfg["max_points"] <= 1_000_000, "Invalid resident point bound")
    require(positive(cfg.get("max_scratch_bytes")) and cfg["max_scratch_bytes"] <= 256 * 1024**2,
            "Invalid combined resident scratch bound")
    require(positive(cfg.get("max_query_bytes")) and cfg["max_query_bytes"] <= 136 * 1024**2,
            "Invalid separately bounded device query scratch")
    cache = binding["cache_policy"]
    require(positive(cache.get("max_clouds")) and cache["max_clouds"] <= 64
            and positive(cache.get("retained_gpu_bytes")) and cache["retained_gpu_bytes"] <= 256 * 1024**2,
            "Invalid retained cache policy")
    math = binding["resident_math"]
    require(all(digest(math.get(key)) for key in ("script_sha256", "gpu_source_sha256", "cpu_bridge_source_sha256",
                "solve_library_sha256", "solve_manifest_sha256")), "Original resident math/solve fingerprints absent")
    artifacts = normalized_artifacts(binding["artifacts_sha256"])
    require(set(REQUIRED_ARTIFACTS).issubset(artifacts), "Device, inherited math, producer, validator or pinned solve artifact missing")
    require(artifacts["scripts/research_resident_icp.py"] == math["script_sha256"]
            and artifacts[SOLVE_DLL] == math["solve_library_sha256"]
            and artifacts[SOLVE_MANIFEST] == math["solve_manifest_sha256"]
            and artifacts["benchmark-output/cuda-pipeline/resident-icp/resident_solve.cpp"] == math["cpu_bridge_source_sha256"],
            "Resident source/solve bindings disagree with their actual artifact bytes")
    return artifacts


def validate_current_artifacts(binding, artifacts):
    require(current_core_hash() == binding["source_sha256"], "Current production source differs from frozen proof")
    for name, expected in artifacts.items():
        require(file_hash(ROOT / name) == expected, f"Current source/solve artifact changed: {name}")
    manifest = json.loads((ROOT / SOLVE_MANIFEST).read_text(encoding="utf-8"))
    require(manifest.get("eigen_commit") == EIGEN_COMMIT and manifest.get("eigen_archive_sha256") == EIGEN_ARCHIVE
            and manifest.get("eigen_headers_sha256") == EIGEN_HEADERS and manifest.get("verified_header_files") == 366,
            "Pinned official Eigen commit/archive/header tree is not the verified build")
    require(manifest.get("bridge_source_sha256") == binding["resident_math"]["cpu_bridge_source_sha256"]
            and manifest.get("library_sha256") == binding["resident_math"]["solve_library_sha256"],
            "Pinned solve manifest disagrees with the current source/library")
    return manifest


def current_core_hash():
    """Exact profile_session source hash algorithm, with no numerical imports."""
    value = hashlib.sha256()
    for folder in (ROOT / "scanner_server", ROOT / "shared", ROOT / "native"):
        for path in sorted(folder.rglob("*")):
            if path.suffix in (".py", ".cpp", ".cu", ".h", ".hpp", ".toml") and "build" not in path.parts:
                value.update(str(path.relative_to(ROOT)).encode())
                value.update(path.read_bytes())
    return value.hexdigest()


def common_checks(report, binding, artifacts, solve_manifest):
    require(report.get("kind") == KIND and report.get("traversal") == TRAVERSAL,
            "Require separate measured device-flat resident proof, not a host/OptiX/sample result")
    original_common_checks(dict(report, kind="standalone-original-double-uniform-grid"), binding)
    require(report.get("proof_bindings_after") == binding, "Installed runtime, configuration or solve changed")
    require(report.get("device") == binding["resident_configuration"]["device"]
            and report.get("resident_configuration") == report.get("resident_configuration_after") == binding["resident_configuration"]
            and report.get("resident_math") == report.get("resident_math_after") == binding["resident_math"],
            "Reported actual resident configuration/math differs from current binding")
    metadata = report["solve_metadata"]
    require(isinstance(metadata, dict) and all(metadata.get(k) == v for k, v in solve_manifest.items()),
            "Actual loaded Eigen solve metadata differs from its pinned build manifest")
    require(isinstance(metadata.get("reported_eigen_version"), str) and bool(metadata["reported_eigen_version"]),
            "Actual loaded Eigen version was not recorded")
    require(resident_report_checks(report["device_resident"], binding, artifacts) == metadata,
            "Final actual resident solve/source metadata changed")


def device_audit_coverage(stats, complete_trajectory=False):
    audit_coverage(stats)
    require(positive(stats.get("device_calls")) and zero(stats.get("device_malformed_results")),
            "Proof did not run clean device-resident retrieval")
    if complete_trajectory:
        require(stats.get("device_flagged_rows") == stats["query_rows"]
                and stats.get("device_query_download_rows") == stats["query_rows"]
                and stats.get("device_query_download_bytes") == 64 * stats["query_rows"],
                "Every actual proof query must be copied once for complete CPU fallback/hit/miss shadow authority")


def resident_checks(run, binding, artifacts):
    delta = run["resident_statistics_delta"]
    require(set(run.get("resident_uncollected_statistics", [])) == set(UNCOLLECTED_RESIDENT_EVENTS)
            and all(key not in delta for key in UNCOLLECTED_RESIDENT_EVENTS),
            "Disabled resident event totals must be omitted and explicitly uncollected")
    require(positive(delta.get("calls")) and positive(delta.get("pose_iterations"))
            and zero(delta.get("cpu_fallback_calls")), "New trajectory must use resident iterations without full original-CPU call fallback")
    return resident_report_checks(run["device_resident"], binding, artifacts)


def resident_report_checks(report, binding, artifacts):
    cfg, math = binding["resident_configuration"], binding["resident_math"]
    require(report.get("device") == cfg["device"] and report.get("gpu_timing") is cfg["gpu_timing"]
            and report.get("scratch_limit_bytes") == cfg["max_scratch_bytes"]
            and report.get("query_scratch_limit_bytes") == cfg["max_query_bytes"]
            and report.get("max_points") == cfg["max_points"], "Actual resident device/resource policy differs")
    require(set(report.get("uncollected_statistics", [])) == set(UNCOLLECTED_RESIDENT_EVENTS)
            and all(key not in report.get("statistics", {}) for key in UNCOLLECTED_RESIDENT_EVENTS),
            "Disabled resident event counters must not be reported as measured zeroes")
    provenance = report["provenance"]
    require(all(provenance.get(key) is True for key in
                ("source_unchanged", "actual_retrieval_unchanged", "original_resident_math_unchanged")),
            "Resident source/math/retrieval changed")
    require(provenance.get("source_sha256") == FROZEN and provenance.get("script_sha256") == math["script_sha256"]
            and provenance.get("gpu_source_sha256") == provenance.get("original_gpu_source_sha256") == math["gpu_source_sha256"]
            and provenance.get("cpu_bridge_source_sha256") == provenance.get("original_cpu_bridge_source_sha256") == math["cpu_bridge_source_sha256"],
            "Resident equations/transforms/Eigen bridge differ from original bound research math")
    for field, names in (("actual_retrieval_sha256", ("scripts/cuda_device_flat_grid_registration.py",
            "scripts/research_device_flat_grid_nn.cu", "scripts/cuda_flat_grid_registration.py",
            "scripts/research_flat_grid_nn.cu", "scripts/validate_device_flat_grid_proof.py")),
            ("original_resident_math_sha256", ("scripts/research_resident_icp.py", "scripts/research_device_grid_resident_icp.py"))):
        mapping = normalized_artifacts(provenance[field])
        require(all(mapping.get(name) == artifacts[name] for name in names), "Actual resident dependency fingerprints disagree")
    require(zero(report.get("statistics", {}).get("cpu_fallback_calls")) and not report.get("fallback_reasons"),
            "A prior full-call CPU fallback was hidden in cumulative resident state")
    return provenance.get("solve_metadata")


def validate_grid_proof(synthetic_path, bridge_path, expected_bindings, expected_fixture_binding):
    synthetic_path, bridge_path = Path(synthetic_path).resolve(), Path(bridge_path).resolve()
    require(synthetic_path != bridge_path, "Synthetic and NEW resident-trajectory proof must be separate artifacts")
    before = (file_hash(synthetic_path), file_hash(bridge_path))
    try:
        artifacts = validate_configuration(expected_bindings)
        solve_manifest = validate_current_artifacts(expected_bindings, artifacts)
        synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        common_checks(synthetic, expected_bindings, artifacts, solve_manifest)
        common_checks(bridge, expected_bindings, artifacts, solve_manifest)
        require(synthetic.get("synthetic_only") is True and not synthetic.get("real_runs"), "Require fresh dedicated device synthetic proof")
        require(synthetic.get("synthetic_device_adapter_checked") is True, "Actual new device adapter was not tested")
        cases = synthetic.get("synthetic_indices", [])
        require(len(cases) >= 16 and all(positive(c.get("queries")) and zero(c.get("index_mismatches"))
                and zero(c.get("device_adapter_index_mismatches"))
                and isinstance(c.get("device_adapter_max_squared_distance_delta"), (int, float))
                and not isinstance(c["device_adapter_max_squared_distance_delta"], bool)
                and 0 <= c["device_adapter_max_squared_distance_delta"] <= 1e-12 for c in cases),
                "Fresh actual host/device nearest-index proof incomplete")
        boundaries = synthetic.get("synthetic_grid_boundaries", [])
        require(len(boundaries) >= 43 and all(positive(c.get("queries")) and zero(c.get("index_mismatches")) for c in boundaries)
                and any(c.get("case") == "original-double-subnormal-and-minimum-radius" for c in boundaries),
                "Fresh signed/minimum-radius/subnormal nearest proof absent")
        device_audit_coverage(synthetic["synthetic_statistics"])
        checks = synthetic["device_flat_synthetic"]
        require(checks.get("passed") is True and set(checks.get("required_checks", [])) == set(DEVICE_CHECKS)
                and all(checks.get(name) is True for name in DEVICE_CHECKS), "Fresh classifier/transport/cache/fault checks incomplete")
        require(checks.get("classifier_cases") == 32 and checks.get("classifier_rows") == (127 + 128 + 129 + 257) * 8
                and positive(checks.get("packet_rows"))
                and all(positive(checks.get("positive_mask_coverage", {}).get(name)) for name in
                        ("unsupported", "uncertain", "audited_hits", "audited_misses")),
                "Classifier tail/warp/block and positive ambiguity/audit coverage incomplete")
        owned = synthetic["flat_cache_ownership"]
        require(owned.get("passed") is True and positive(owned.get("cloud_evictions"))
                and all(owned.get(name) is True for name in ("bounded_retained_bytes", "clear_releases_all_owned_arrays",
                        "one_byte_budget_original_cpu_fallback")), "Inherited cache ownership proof absent")
        require(bridge.get("synthetic_only") is False, "Require separate actual resident-trajectory proof")
        require(bridge.get("fixture_binding") == expected_fixture_binding
                and bridge.get("fixture_binding_sha256") == canonical_hash(expected_fixture_binding), "Fixture/raw/tasks/proposals differ")
        for field in ("fixture_sha256", "reference_sha256", "original_raw_input_sha256"):
            require(bridge.get(field) == expected_fixture_binding[field] == bridge.get(field + "_after"), "Raw/fixture/reference changed")
        require(bridge.get("fixture_arrays_unchanged") is True and bridge.get("cloud_arrays_unchanged") is True,
                "Original native input arrays changed")
        require(bridge.get("fixture_provenance", {}).get("local_pose_source") == "measured Finish fragment report; no archived ZIP poses"
                and bridge.get("original_fixture_authority", {}).get("same_decisions_and_support") is True,
                "Native raw-derived measured fixture authority absent")
        tasks = expected_fixture_binding["tasks"]
        require([(t["position"], t["pair"], len(t["proposal_sha256"])) for t in tasks] == [(0, [8, 12], 5), (4, [0, 2], 4)],
                "Require original accepted/rejected nine-proposal scope")
        runs = bridge["real_runs"]
        cpu = [r for r in runs if r.get("mode") == "native_cpu"]
        gpu = [r for r in runs if r.get("mode") == "grid"]
        require(len(cpu) == 1 and gpu and all(r.get("mode") in ("native_cpu", "grid") for r in runs),
                "Contemporary original native baseline or actual resident runs missing")
        for run in runs:
            require(run.get("complete") is True and len(run.get("pairs", [])) == 2, "Incomplete original pair run")
            for pair, task in zip(run["pairs"], tasks):
                require(pair.get("position") == task["position"] and pair.get("pair") == task["pair"], "Original pair order differs")
                proposals = pair.get("proposal_results", [])
                require([p.get("proposal_index") for p in proposals] == list(range(len(task["proposal_sha256"])))
                        and [p.get("input_sha256") for p in proposals] == task["proposal_sha256"], "Original proposal order/bytes differ")
                require(pair.get("verdict", {}).get("accepted") is (task["position"] == 0)
                        and pair.get("pair_verdict_same") is True, "Original acceptance/ambiguity/support gates changed")
                if run["mode"] == "grid":
                    require(all(p.get("quality", {}).get("passed") is True for p in proposals), "Original witnesses/pose/information gates failed")
            if run["mode"] == "grid":
                device_audit_coverage(run["statistics_delta"], complete_trajectory=True)
                require(zero(run["statistics_delta"].get("resident_xyz_uploads"))
                        or positive(run["statistics_delta"].get("resident_xyz_uploads")), "Invalid original-order XYZ upload count")
                require(resident_checks(run, expected_bindings, artifacts) == bridge["solve_metadata"], "Actual resident solve metadata changed")
        require(positive(sum(run["statistics_delta"].get("resident_xyz_uploads", 0) for run in gpu)),
                "The actual resident trajectory never uploaded original-order target XYZ")
        targets = bridge.get("real_target_membership_sha256", [])
        require(targets and len(set(targets)) == len(targets) and all(digest(d) for d in targets), "Audited original target point-byte membership absent")
        validate_current_artifacts(expected_bindings, artifacts)
        require(before == (file_hash(synthetic_path), file_hash(bridge_path)), "Proof files changed during validation")
        return DeviceFlatGridProofAuthority(canonical_hash(expected_bindings), canonical_hash(expected_fixture_binding),
            before[0], before[1], frozenset(targets), tuple(sorted(expected_bindings["artifacts_sha256"].items())),
            json.dumps(expected_bindings, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except GridProofError:
        raise
    except (KeyError, TypeError, IndexError, ValueError, AttributeError, OSError) as error:
        raise GridProofError(f"Incomplete/malformed/stale device proof: {error}") from error
