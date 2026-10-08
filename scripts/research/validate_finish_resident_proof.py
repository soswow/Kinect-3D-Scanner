"""Pure-stdlib proof boundaries for offline, matched full-Finish research.

Component proof does not authorize new field targets or live trajectories.
This module neither imports numerical packages nor initializes a GPU.  Its
token authorizes only the ordered calls and exact input scope of a successful
fresh full-Finish audit; it is not a production policy or an accuracy claim.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass

from scripts.research.validate_device_flat_grid_proof import (
    DeviceFlatGridProofAuthority, digest, normalized_artifacts,
    validate_configuration, validate_current_artifacts, resident_report_checks, device_audit_coverage)
from scripts.research.archive.validate_uniform_grid_proof import (
    GridProofError, canonical_hash, file_hash, positive, require, zero)

KIND = "offline-full-finish-device-flat-resident-v1"
STAGES = [[.12, 40], [.06, 30], [.03, 20]]
THREAD_POLICY = {"open3d": 20, "opencv": 20, "omp": "8"}
ARRAY_KEYS = ("source_points", "source_normals", "target_points", "target_normals", "seed")
CALL_KEYS = set(ARRAY_KEYS) | {
    "stages", "huber_m", "relative_fitness", "relative_rmse", "thread_policy", "site", "gate_context"}
FINISH_ARTIFACTS = (
    "scripts/research/finish_resident_registration.py",
    "scripts/research/profile_resident_finish.py",
    "scripts/research/validate_finish_resident_proof.py",
    "scripts/research/compare_resident_finishes.py", "scripts/profile_session.py",
    "scripts/compare_session_profiles.py", "shared/surface_metrics.py")
QUERY_COUNTERS = ("query_rows", "direct_gpu_hits", "declared_gpu_misses", "audited_hits", "audited_misses",
    "audit_index_mismatches", "audit_false_misses", "device_calls", "device_flagged_rows",
    "device_query_download_rows", "device_query_download_bytes", "device_malformed_results", "exact_cpu_queries")


@dataclass(frozen=True)
class FinishResidentProofAuthority(DeviceFlatGridProofAuthority):
    """A separate scope token, with fresh field targets and ordered input bytes."""
    finish_scope_json: str
    expected_call_signatures: tuple
    finish_audit_report_sha256: str
    quality_proof_sha256: str
    full_finish_artifact_sha256: tuple

    @property
    def finish_scope(self):
        return json.loads(self.finish_scope_json)


def finite_nonnegative(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def validate_call_inputs(payload):
    """Validate original FP64 array descriptors and the unchanged ICP recipe."""
    require(isinstance(payload, dict) and set(payload) == CALL_KEYS,
            "Incomplete original-call input signature")
    for name in ARRAY_KEYS:
        item = payload[name]
        require(isinstance(item, dict) and set(item) == {"sha256", "dtype", "shape"}
                and digest(item["sha256"]) and item["dtype"] == "<f8",
                f"Call must bind the original FP64 {name} bytes")
        shape = item["shape"]
        require(isinstance(shape, list) and len(shape) == 2
                and all(type(n) is int for n in shape), "Malformed original array shape")
        if name == "seed":
            require(shape == [4, 4], "Require an unrounded original 4x4 FP64 seed")
        else:
            require(0 <= shape[0] <= 1_000_000 and shape[1] == 3,
                    "Original cloud exceeds the validated point bound")
    require(payload["source_normals"]["shape"][0] in (0, payload["source_points"]["shape"][0])
            and payload["target_normals"]["shape"] == payload["target_points"]["shape"],
            "Original point/normal shapes disagree")
    require(payload["stages"] == STAGES and payload["huber_m"] == .01
            and payload["relative_fitness"] == payload["relative_rmse"] == 1e-6
            and payload["thread_policy"] == THREAD_POLICY,
            "Original radius, iteration, Huber, convergence or thread policy changed")
    site = payload["site"]
    require(isinstance(site, dict) and set(site) == {"file", "function", "line"}
            and all(isinstance(site[k], str) and 0 < len(site[k]) <= 1024 for k in ("file", "function"))
            and positive(site["line"]), "Original call site is absent")
    try:
        encoded = json.dumps(payload["gate_context"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise GridProofError("Malformed ordered gate context") from error
    require(payload["gate_context"] is not None and len(encoded) <= 8192,
            "Original ordered gate context is absent or unbounded")


def call_signature(payload):
    validate_call_inputs(payload)
    return canonical_hash(payload)


def validate_expected_call(authority, index, payload):
    """Call before GPU work.  The context must turn failure into a hard latch."""
    require(type(authority) is FinishResidentProofAuthority, "Old component authority cannot authorize full Finish")
    require(type(index) is int and 0 <= index < len(authority.expected_call_signatures),
            "Full-Finish call count exceeds the audited scope")
    signature = call_signature(payload)
    require(signature == authority.expected_call_signatures[index],
            "Original call order, cloud/normal bytes, gate context or unrounded seed changed")
    require(payload["target_points"]["sha256"] in authority.target_digests,
            "Full-Finish target was not audited")
    return signature


def validate_complete_calls(authority, count):
    """Call after Finish, before accepting its mesh or timing report."""
    require(type(authority) is FinishResidentProofAuthority
            and type(count) is int and count == len(authority.expected_call_signatures),
            "Full Finish omitted an audited call or failed to complete")


def validate_shadow(row):
    """Reject an unchecked or mutated per-call original CPU result shadow."""
    signature = call_signature(row["call_inputs"])
    require(row.get("call_signature") == signature
            and all(row.get(name) == signature for name in
                    ("inputs_after_resident", "inputs_before_shadow", "inputs_after_shadow")),
            "Resident or original CPU shadow changed the original call inputs")
    require(zero(row.get("resident_full_call_fallbacks")),
            "Whole-call CPU fallback cannot establish resident trajectory authority")
    shadow = row["cpu_shadow"]
    require(shadow.get("passed") is True and shadow.get("correspondence_ids_equal") is True,
            "Original CPU result or exact correspondence shadow failed")
    for field in ("transformation_max_abs_delta", "fitness_abs_delta", "rmse_abs_delta"):
        require(finite_nonnegative(shadow.get(field)) and shadow[field] <= 1e-8,
                "Original CPU result shadow is missing, nonfinite or outside the fixed tolerance")
    return signature


def strict_json(text):
    """Closed evidence cannot contain duplicate keys, NaN or Infinity."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate key in closed proof evidence")
            result[key] = value
        return result

    def constant(value):
        raise GridProofError(f"Nonfinite value in closed proof evidence: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def closed_file(record, tracked):
    require(isinstance(record, dict) and isinstance(record.get("path"), str) and digest(record.get("sha256")),
            "Closed evidence path/hash absent")
    path = Path(record["path"]).resolve()
    actual = file_hash(path)
    require(actual == record["sha256"], "Closed evidence file changed")
    if path in tracked:
        require(tracked[path] == actual, "One evidence path has contradictory fingerprints")
    tracked[path] = actual
    return path


def validate_scope(scope, runtime, tracked):
    require(isinstance(scope, dict) and scope.get("runtime_binding") == runtime
            and scope.get("thread_policy") == THREAD_POLICY, "Field runtime/thread policy differs from component proof")
    archive_path = closed_file(scope["archive"], tracked)
    with zipfile.ZipFile(archive_path) as archive:
        manifest_entry = archive.getinfo("manifest.json")
        require(manifest_entry.file_size <= 16 * 1024**2, "Raw manifest exceeds the bounded research scope")
        manifest = strict_json(archive.read(manifest_entry).decode("utf-8"))
    require(isinstance(manifest.get("frames"), list), "Raw frame manifest is absent")
    selected = scope["selected_indices"]
    require(isinstance(selected, list) and selected and len(selected) <= 3000 and len(selected) == len(manifest["frames"])
            and selected == list(range(len(selected))) and all(type(index) is int for index in selected),
            "Full-Finish scope requires the complete original raw frame order")
    require(type(scope.get("seed")) is int and isinstance(scope.get("settings"), dict)
            and isinstance(scope.get("pipeline_options"), dict), "Effective settings, environment or seed absent")
    settings = scope["settings"]
    require(isinstance(settings.get("camera"), dict) and "sensor_calibration" in settings,
            "Effective field camera/sensor calibration is absent")
    require(type(settings.get("final_block_count")) is int and 1 <= settings["final_block_count"] <= 50000,
            "Invalid matched Final block budget")
    live = scope["live"]
    accepted = live["accepted_indices"]
    require(isinstance(accepted, list) and accepted and accepted == sorted(set(accepted))
            and all(type(index) is int and index in selected for index in accepted), "Fresh Live decisions are incomplete")
    require(zero(live.get("unprocessed_count")) and live.get("pose_source") == "fresh raw Live replay"
            and digest(live.get("decisions_sha256")) and digest(live.get("poses_sha256")),
            "Archived, rounded or unprocessed Live state cannot authorize Finish")
    if "live_fixture" in scope:
        fixture = scope["live_fixture"]
        closed_file(fixture, tracked)
        require(fixture.get("pose_source") == "fresh raw Live replay" and digest(fixture.get("producer_sha256"))
                and fixture.get("raw_archive_sha256") == scope["archive"]["sha256"]
                and fixture.get("scope_sha256") == canonical_hash({k: v for k, v in scope.items() if k != "live_fixture"}),
                "Frozen Live fixture does not bind freshly measured raw Live scope")
    # This also rejects nonfinite hidden settings/calibration/environment values.
    canonical_hash(scope)


def validate_profile(record, scope, tracked):
    path = closed_file(record, tracked)
    geometry = closed_file({"path": record["geometry_path"], "sha256": record["geometry_sha256"]}, tracked)
    profile = read_json(path)
    require(profile.get("input_sha256") == scope["archive"]["sha256"]
            and profile.get("source_sha256") == scope["runtime_binding"]["source_sha256"]
            and profile.get("input_changed_during_profile") is False
            and profile.get("source_changed_during_profile") is False,
            "Profile archive/source changed or differs from the closed field scope")
    require(profile.get("finish_requested") is True and profile.get("mesh_built") is True
            and profile.get("pose_seeds_used") is False, "Successful full Finish without archived ZIP seeds is required")
    require(profile.get("selected_indices") == scope["selected_indices"] and profile.get("seed") == scope["seed"]
            and profile.get("settings") == scope["settings"] and profile.get("research_pipeline_options") == scope["pipeline_options"]
            and isinstance(profile.get("pipeline_options"), dict)
            and all(profile["pipeline_options"][key] == scope["pipeline_options"].get(key) for key in profile["pipeline_options"])
            and profile.get("accepted_indices_before_finish") == scope["live"]["accepted_indices"],
            "Profile settings/selection/environment/seed or freshly measured Live decisions differ")
    require(profile.get("frames") == len(scope["selected_indices"]), "Profile omitted raw frames")
    geometry_record = profile["geometry"]
    require(positive(geometry_record.get("vertices")) and positive(geometry_record.get("triangles"))
            and Path(geometry_record["artifact"]).resolve() == geometry, "Profile does not bind its actual successful triangle mesh")
    require(profile.get("final_reconstruction", {}).get("voxel_m") ==
            (scope["settings"].get("final_voxel_m") or scope["settings"]["voxel_m"]),
            "Actual Final resolution differs from requested effective settings")
    return profile


def validate_quality(path, audit_sha256, candidate_record, scope, tracked, *, artifacts, component_proof, candidate_trace):
    path = Path(path).resolve()
    quality_hash = file_hash(path)
    tracked[path] = quality_hash
    quality = read_json(path)
    require(quality.get("kind") == "offline-full-finish-device-flat-quality-v1" and quality.get("status") == "passed"
            and quality.get("audit_report_sha256") == audit_sha256, "Separate successful closed field quality proof is absent or stale")
    require(quality.get("candidate") == candidate_record, "Quality proof does not bind the audited candidate profile/geometry")
    audit_path = closed_file(quality["audit"], tracked)
    require(quality["audit"]["sha256"] == audit_sha256 and file_hash(audit_path) == audit_sha256,
            "Quality closure does not bind the actual field audit")
    require(quality.get("candidate_trace") == candidate_trace, "Quality gate evidence used a different candidate trace")
    closed_file(quality["candidate_trace"], tracked)
    native_sidecar_path = closed_file(quality["native_sidecar"], tracked)
    native_sidecar = read_json(native_sidecar_path)
    require(native_sidecar.get("kind") == KIND and native_sidecar.get("mode") == "native"
            and native_sidecar.get("status") == "complete" and native_sidecar.get("performance_attribution_valid") is True
            and native_sidecar.get("runner_hooks_restored") is True and not native_sidecar.get("failure")
            and not native_sidecar.get("cleanup_failures"), "Original CPU Finish reference did not close cleanly")
    require(native_sidecar.get("source_sha256") == native_sidecar.get("source_sha256_after") == scope["runtime_binding"]["source_sha256"]
            and native_sidecar.get("archive_sha256") == native_sidecar.get("archive_sha256_after") == scope["archive"]["sha256"]
            and native_sidecar.get("runtime_binding") == native_sidecar.get("runtime_binding_after") == scope["runtime_binding"]
            and native_sidecar.get("artifacts_sha256") == native_sidecar.get("artifacts_sha256_after") == artifacts
            and native_sidecar.get("component_proof") == native_sidecar.get("component_proof_after") == component_proof,
            "Original CPU reference source/runtime/input/helpers/component proof differs")
    native_scope = native_sidecar["scope_binding"]
    require(native_sidecar.get("scope_binding_sha256") == canonical_hash(native_scope), "Native field scope fingerprint differs")
    validate_scope(native_scope, scope["runtime_binding"], tracked)
    require({k: v for k, v in native_scope.items() if k != "live"} == {k: v for k, v in scope.items() if k != "live"}
            and native_scope["live"]["accepted_indices"] == scope["live"]["accepted_indices"],
            "Original CPU reference used different archive/settings/selection/environment or Live acceptance")
    require(quality.get("live_pose_signatures_equal") is (native_scope["live"]["poses_sha256"] == scope["live"]["poses_sha256"])
            and quality.get("live_decision_signatures_equal") is (native_scope["live"]["decisions_sha256"] == scope["live"]["decisions_sha256"]),
            "Whole-trajectory Live roundoff drift was hidden")
    require(native_sidecar.get("profile") == quality["native"] and native_sidecar.get("trace") == quality.get("native_trace"),
            "Original CPU mesh/gate trace differs from its closed native reference")
    closed_file(quality["native_trace"], tracked)
    native_registration = native_sidecar["registration"]
    require(native_registration.get("complete") is True and native_registration.get("restored") is True
            and native_registration.get("input_immutability_passed") is True and positive(native_registration.get("calls"))
            and zero(native_registration.get("cpu_shadow_calls")) and zero(native_registration.get("full_cpu_fallback_calls"))
            and native_registration.get("resident") is None and native_registration.get("retrieval_statistics") == {}
            and not native_registration.get("failure") and not native_registration.get("cleanup_failures"),
            "Original CPU reference was replaced, incomplete or failed")
    candidate = validate_profile(quality["candidate"], scope, tracked)
    native = validate_profile(quality["native"], scope, tracked)
    require(Path(quality["candidate"]["path"]).resolve() != Path(quality["native"]["path"]).resolve(),
            "Candidate cannot serve as its own original CPU mesh reference")
    require(quality.get("same_ordered_gate_decisions") is True and quality.get("ordered_gate_evidence_passed") is True,
            "Original ordered Finish decisions/support/pose/information evidence changed")
    comparison = quality["comparison"]
    require(comparison.get("candidate_report_sha256") == quality["candidate"]["sha256"]
            and comparison.get("baseline_report_sha256") == quality["native"]["sha256"]
            and comparison.get("same_mesh_success") is True and comparison.get("same_accepted_indices") is True
            and candidate.get("accepted_indices") == native.get("accepted_indices"),
            "Mesh comparison is stale or changed final accepted frames")
    surface = comparison["triangle_surface_metrics_vs_cpu"]
    require(surface.get("threshold_m") == .005 and surface.get("samples_per_surface") == 30000
            and surface.get("alignment") == "fixed input coordinate frame; no scale or trajectory fitting",
            "Surface agreement used a relaxed, fitted or undersampled contract")
    require(finite_nonnegative(surface.get("surface_p95_m")) and surface["surface_p95_m"] <= .0005
            and all(finite_nonnegative(surface.get(key)) and .999 <= surface[key] <= 1 for key in ("precision", "completeness")),
            "Final triangle surface agreement failed the fixed 0.5mm p95/99.9% within5mm gate")
    return quality_hash


def validate_finish_proof(audit_path, component_authority, expected_scope_binding, expected_artifacts, *, quality_path):
    """Mint timing authority only after separate component, field and quality proofs.

    The component token must have been validated against the actual current
    runtime by its own validator.  Here we recheck its current artifact bytes,
    exact proof files, the new scope, every call, and immutable closed outputs.
    """
    try:
        require(type(component_authority) is DeviceFlatGridProofAuthority,
                "Require fresh separately validated component authority")
        runtime = component_authority.runtime_binding
        component_artifacts = validate_configuration(runtime)
        require(component_authority.bindings_sha256 == canonical_hash(runtime)
                and dict(component_authority.artifact_sha256) == runtime["artifacts_sha256"],
                "Component token/runtime artifact bindings disagree")
        validate_current_artifacts(runtime, component_artifacts)
        finish_artifacts = normalized_artifacts(expected_artifacts)
        require(set(FINISH_ARTIFACTS).issubset(finish_artifacts), "New Finish context/runner/validator artifacts are missing")
        for name, value in finish_artifacts.items():
            require(file_hash(ROOT / name) == value, f"Current full-Finish helper changed: {name}")
        audit_path = Path(audit_path).resolve()
        audit_hash = file_hash(audit_path)
        tracked = {audit_path: audit_hash}
        report = read_json(audit_path)
        require(report.get("kind") == KIND and report.get("mode") == "audit" and report.get("status") == "complete"
                and report.get("performance_attribution_valid") is False and report.get("runner_hooks_restored") is True
                and not report.get("failure") and not report.get("cleanup_failures"),
                "Require a complete correctness-only field audit with clean hook restoration")
        require(report.get("runtime_binding") == report.get("runtime_binding_after") == runtime
                and report.get("scope_binding") == expected_scope_binding
                and report.get("scope_binding_sha256") == canonical_hash(expected_scope_binding),
                "Field archive/settings/Live trajectory/runtime scope differs")
        validate_scope(expected_scope_binding, runtime, tracked)
        require(report.get("source_sha256") == report.get("source_sha256_after") == runtime["source_sha256"]
                and report.get("archive_sha256") == report.get("archive_sha256_after") == expected_scope_binding["archive"]["sha256"]
                and report.get("artifacts_sha256") == report.get("artifacts_sha256_after") == finish_artifacts,
                "Field source/archive or new helper artifacts changed")
        for name, expected in (("synthetic", component_authority.synthetic_report_sha256),
                               ("bridge", component_authority.bridge_report_sha256)):
            record = report["component_proof"][name]
            require(record.get("sha256") == expected, "Component proof differs from validated current token")
            closed_file(record, tracked)
        require(report.get("component_proof_after") == report["component_proof"], "Component proof changed during field Finish")
        require(report["component_proof"]["synthetic"]["path"] != report["component_proof"]["bridge"]["path"],
                "Synthetic and component trajectory proofs must be separate")
        trace = report["trace"]
        trace_path = closed_file(trace, tracked)
        rows = []
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                # Reuse the strict duplicate/nonfinite parser without importing any numerical package.
                rows.append(strict_json(line))
        require(trace.get("rows") == len(rows) and rows
                and all(isinstance(row, dict) and row.get("event") in ("match", "gate") for row in rows),
                "Closed original-call/gate trace is incomplete")
        matches = [row for row in rows if row["event"] == "match"]
        require(0 < len(matches) <= 100000 and [row.get("call_index") for row in matches] == list(range(len(matches)))
                and all(type(row["call_index"]) is int and row.get("complete") is True for row in matches),
                "Original call order/count is incomplete")
        signatures = tuple(validate_shadow(row) for row in matches)
        registration = report["registration"]
        require(positive(registration.get("calls")) and positive(registration.get("cpu_shadow_calls"))
                and registration.get("complete") is True and registration.get("restored") is True
                and registration.get("input_immutability_passed") is True
                and registration.get("calls") == registration.get("cpu_shadow_calls") == len(matches)
                and registration.get("call_signatures") == list(signatures)
                and zero(registration.get("cpu_shadow_failures")) and zero(registration.get("full_cpu_fallback_calls"))
                and not registration.get("failure") and not registration.get("cleanup_failures"),
                "Full-Finish work, original shadows, patch restoration or cleanup failed")
        targets = frozenset(row["call_inputs"]["target_points"]["sha256"] for row in matches)
        declared_targets = registration.get("target_digests")
        require(isinstance(declared_targets, list) and len(declared_targets) == len(targets)
                and set(declared_targets) == targets, "Fresh full-Finish target membership differs from the complete trace")
        totals = {key: 0 for key in QUERY_COUNTERS}
        for row in matches:
            stats = row["query_statistics_delta"]
            require(all(type(stats.get(key)) is int and stats[key] >= 0 for key in QUERY_COUNTERS),
                    "Per-call native-query shadow accounting is missing or invalid")
            require(stats["direct_gpu_hits"] == stats["audited_hits"] and stats["declared_gpu_misses"] == stats["audited_misses"]
                    and stats["query_rows"] == stats["device_flagged_rows"] == stats["device_query_download_rows"]
                    and stats["device_query_download_bytes"] == 64 * stats["query_rows"]
                    and stats["query_rows"] == stats["direct_gpu_hits"] + stats["declared_gpu_misses"] + stats["exact_cpu_queries"]
                    and all(zero(stats[key]) for key in ("audit_index_mismatches", "audit_false_misses", "device_malformed_results")),
                    "Every actual field query must have its original CPU hit/miss/ambiguity resolution")
            for key in QUERY_COUNTERS:
                totals[key] += stats[key]
        retrieval = registration["retrieval_statistics"]
        require(all(retrieval.get(key) == value for key, value in totals.items()), "Incomplete field-query audit accounting")
        device_audit_coverage(retrieval, complete_trajectory=True)
        resident_report_checks(registration["resident"], runtime, component_artifacts)
        require(registration["resident"]["statistics"].get("calls") == len(matches), "Full resident call count differs")
        validate_profile(report["profile"], expected_scope_binding, tracked)
        quality_hash = validate_quality(quality_path, audit_hash, report["profile"], expected_scope_binding, tracked,
            artifacts=finish_artifacts, component_proof=report["component_proof"], candidate_trace=report["trace"])
        validate_current_artifacts(runtime, component_artifacts)
        for name, expected in finish_artifacts.items():
            require(file_hash(ROOT / name) == expected, "Full-Finish source changed during validation")
        require(all(file_hash(path) == expected for path, expected in tracked.items()), "Closed proof/raw/output changed during validation")
        token = dict(component_authority.__dict__, target_digests=targets)
        return FinishResidentProofAuthority(**token,
            finish_scope_json=json.dumps(expected_scope_binding, sort_keys=True, separators=(",", ":"), allow_nan=False),
            expected_call_signatures=signatures, finish_audit_report_sha256=audit_hash,
            quality_proof_sha256=quality_hash, full_finish_artifact_sha256=tuple(sorted(finish_artifacts.items())))
    except GridProofError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OSError) as error:
        raise GridProofError(f"Incomplete/malformed/stale full-Finish proof: {error}") from error
