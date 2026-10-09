"""Metadata supplement to an already passed physical candidate Final comparison.

This compares published temporal diagnostics only. It performs no reconstruction,
surface sampling, numerical imports, qualification or worker-exit attestation.
Missing optional fields are retained as missing, never converted to false/zero.
If pose_candidates is imported, the original candidate controller's alias checks
do not independently guard that module's loaded bodies. This supplement does not
repair or extend those execution guards; current file/checkpoint closure remains
the original producer's evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import source_hash

KIND = "gpu-icp-current-temporal-diagnostic-supplement-v1"
PRIOR_KIND = "gpu-icp-whole-finish-candidate-observable-quality-v1"
COMPARATOR = ROOT/"scripts/research/compare_gpu_icp_candidate_finishes.py"
COMPARATOR_SHA256 = "cc2ee95ad65733341b9bc1041e5c86dcdad8f15b4e53f979ec8deddc302d1ae7"
FIELDS = ("temporal_bridges", "ambiguous_temporal_pairs",
          "rejected_optimized_boundaries", "rejected_fallback_boundaries")
OWN_FILES = ("scripts/research/compare_gpu_icp_temporal_diagnostics.py",
             "tests/test_gpu_icp_temporal_diagnostics.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "Finite numeric evidence required")
    return value


def digest(value):
    require(type(value) is str and len(value) == 64
        and all(c in "0123456789abcdef" for c in value), "Exact SHA256 digest required")
    return value


def closed(report, mode):
    require(type(report) is dict and mode in ("native", "audit", "shadow", "measure"),
        "Original candidate report/mode unavailable")
    registration = report.get("registration", {})
    require(type(registration) is dict, "Original registration envelope unavailable")
    actual = report.get("mode")
    require(actual == mode and report.get("kind") == "gpu-icp-whole-finish-candidate-"+mode+"-v1"
        and report.get("status") == "passed" and report.get("failure") is None
        and report.get("cleanup_passed") is True and report.get("cleanup_failures") == []
        and type(report.get("binding")) is dict and report["binding"] == report.get("binding_after")
        and registration.get("mode") == mode and registration.get("complete") is True
        and registration.get("closed") is True and registration.get("restored") is True
        and registration.get("failure") is None and registration.get("cleanup_failures") == [],
        "Closed original current candidate report envelope required")
    if mode == "native":
        require(registration.get("original_build_calls") == 1, "One unchanged original native Finish required")
    else:
        require(mode in ("audit", "shadow", "measure") and registration.get("successful_builds") == 1
            and registration.get("default_evidence") is True
            and type(report.get("candidate_source")) is dict
            and report["candidate_source"] == report.get("candidate_source_after"),
            "Closed original candidate method evidence required")
    profile = report.get("profile")
    require(type(profile) is dict, "Original profile unavailable")
    require(profile.get("mesh_built") is True and profile.get("finish_requested") is True
        and profile.get("pose_seeds_used") is False
        and profile.get("input_changed_during_profile") is False
        and profile.get("source_changed_during_profile") is False,
        "Successful unchanged full raw Finish required")
    return profile


def pairs(value, known):
    require(type(value) is list and len(value) <= 512, "Bounded published temporal pairs required")
    result = []
    for row in value:
        require(type(row) is list and len(row) == 2 and all(type(i) is int and i in known for i in row)
            and row[0] != row[1], "Malformed published temporal pair")
        result.append(list(row))
    require(len({tuple(row) for row in result}) == len(result), "Duplicate published temporal pair")
    return result


def diagnostics(profile):
    report = profile.get("fragment_reconnection")
    require(type(report) is dict, "Original published fragment report unavailable")
    fragments = report.get("fragments")
    bridges = report.get("verified_bridges")
    require(type(fragments) is list and len(fragments) <= 32
        and type(bridges) is list and len(bridges) <= 512, "Bounded published graph inventory required")
    require(all(type(row) is dict and type(row.get("id")) is int for row in fragments),
        "Original fragment IDs unavailable")
    ids = [row["id"] for row in fragments]
    require(len(set(ids)) == len(ids) and all(type(i) is int and 0 <= i < 32 for i in ids),
        "Original fragment IDs malformed")
    known = set(ids)
    flags = []
    for index, edge in enumerate(bridges):
        require(type(edge) is dict and type(edge.get("source")) is int and type(edge.get("target")) is int
            and edge["source"] in known and edge["target"] in known and edge["source"] != edge["target"],
            "Original published bridge endpoints malformed")
        item = {"bridge_index": index, "source": edge["source"], "target": edge["target"],
                "presence": "present" if "temporal_constraint" in edge else "missing"}
        if "temporal_constraint" in edge:
            require(type(edge["temporal_constraint"]) is bool, "Temporal constraint must be an actual bool")
            item.update(availability="reported", value=edge["temporal_constraint"])
        else:
            item["availability"] = "optional_flag_not_reported"
        require(edge.get("validation_scope") != "temporal camera pair"
            or edge.get("temporal_constraint") is True, "Temporal camera-pair edge lost its original true flag")
        flags.append(item)
    # Current source emits both pass fields before candidate_pairs and before
    # any final fragment inventory. Their absence after those receipts is a gap,
    # rather than an unavailable conditional conflict diagnostic.
    reached = bool(fragments or bridges) or "candidate_pairs" in report or any(k in report for k in FIELDS[:2])
    result = {"verified_bridge_temporal_flags": flags}
    for name in FIELDS:
        if name not in report:
            require(name not in FIELDS[:2] or not reached, "Missing required temporal-pass field: "+name)
            result[name] = {"presence": "missing", "availability":
                "conditional_conflict_field_not_reported" if name in FIELDS[2:]
                else "temporal_pass_not_published"}
            continue
        value = report[name]
        if name == "temporal_bridges":
            require(type(value) is int and 0 <= value <= 512, "Temporal bridge count must be a bounded actual int")
        else:
            value = pairs(value, known)
        result[name] = {"presence": "present", "availability": "reported", "value": value}
    return result


def reference(record, expected):
    require(type(record) is dict and type(expected) is dict
        and type(record.get("path")) is str and type(expected.get("path")) is str,
        "Original report/geometry reference unavailable")
    require(Path(record["path"]).resolve() == Path(expected["path"]).resolve()
        and digest(record.get("sha256")) == digest(expected.get("sha256")),
        "Prior physical comparison consumes different report bytes")


def compare_documents(native, candidate, prior, *, native_ref, candidate_ref, current_source):
    digest(current_source)
    require(type(candidate) is dict and type(prior) is dict, "Original evidence documents unavailable")
    require(candidate.get("mode") in ("audit", "shadow", "measure"), "Original GPU candidate mode required")
    a, b = closed(native, "native"), closed(candidate, candidate.get("mode"))
    require(type(native.get("scope_binding")) is dict and bool(native["scope_binding"])
        and native["binding"] == candidate["binding"] and native["scope_binding"] == candidate["scope_binding"],
        "Actual source/runtime/checkpoint identity differs")
    require(native["binding"].get("source_sha256") == a.get("source_sha256")
        == b.get("source_sha256") == current_source, "Require actual currently recorded production source")
    for key in ("input_sha256", "selected_indices", "seed", "settings", "pipeline_options", "thread_policy", "accepted_indices"):
        require(a[key] == b[key], "Original full raw Finish scope differs: "+key)
    require(prior.get("kind") == PRIOR_KIND and prior.get("status") == "passed" and prior.get("failure") is None
        and prior.get("comparator_sha256") == COMPARATOR_SHA256, "Passed frozen physical Final comparator required first")
    reference(prior.get("native_report"), native_ref)
    reference(prior.get("candidate_report"), candidate_ref)
    checks = prior.get("checks", {})
    required = ("same_checkpoint_raw_settings_and_accepted_ids", "same_published_bridge_witness_frontier_memberships",
                "pose_bounds", "fixed_coordinate_surface_bounds")
    require(type(checks) is dict and all(checks.get(key) is True for key in required)
        and all(value is True for value in checks.values()),
        "Prior original pose/surface/coverage comparison must pass every check")
    metrics, surface = prior["metrics"], prior["surface"]
    require(0 <= finite(metrics["pose_translation_max_m"]) <= .0005
        and 0 <= finite(metrics["pose_rotation_max_deg"]) <= .1
        and 0 <= finite(surface["surface_p95_m"]) <= .0005
        and .999 <= finite(surface["precision"]) <= 1
        and .999 <= finite(surface["completeness"]) <= 1, "Prior physical Final bounds failed")
    for profile, key in ((a, "reference_geometry"), (b, "candidate_geometry")):
        geometry = profile["geometry"]
        reference(prior[key], {"path": geometry["artifact"], "sha256": geometry["sha256"]})
    left, right = diagnostics(a), diagnostics(b)
    require(left == right, "Original published temporal diagnostics differ")
    return {"source_sha256": current_source, "checks": {"prior_physical_final_comparison_passed": True,
        "same_current_source_runtime_checkpoint": True, "temporal_diagnostics_equal": True},
        "diagnostics": left, "limits": {"physical_surface_recomputed": False,
            "numerical_replay_performed": False, "worker_exits_independently_verified": False,
            "missing_optional_fields_mean_no_conflicts": False, "qualification_minted": False,
            "independent_pose_candidates_loaded_body_guard": False,
            "production_authority": False, "whole_finish_quality_authority": False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native_report", "candidate_report", "final_comparison"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = [getattr(args, name).resolve(strict=True) for name in ("native_report", "candidate_report", "final_comparison")]
    output = args.output.resolve()
    require(output.is_relative_to(ROOT/"benchmark-output") and not output.exists()
        and len(set(paths)) == 3 and output not in paths, "Independent inputs and fresh private output required")
    result = {"kind": KIND, "status": "failed", "failure": None}
    primary = None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        before = [sha(path) for path in paths]
        own = {name: sha(ROOT/name) for name in OWN_FILES}
        require(sha(COMPARATOR) == COMPARATOR_SHA256, "Frozen original Final comparator changed")
        source = source_hash()
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        refs = [{"path": str(path), "sha256": digest} for path, digest in zip(paths, before)]
        result.update(native_report=refs[0], candidate_report=refs[1], prior_final_comparison=refs[2],
                      artifacts_sha256=own, comparator_sha256=COMPARATOR_SHA256)
        result.update(compare_documents(*reports, native_ref=refs[0], candidate_ref=refs[1], current_source=source))
        # Prior geometry was physically compared; only rehash its exact bytes here.
        for report in reports[:2]:
            geometry = report["profile"]["geometry"]
            require(sha(geometry["artifact"]) == geometry["sha256"], "Previously compared geometry bytes changed")
        require([sha(path) for path in paths] == before and source_hash() == source
            and {name: sha(ROOT/name) for name in OWN_FILES} == own and sha(COMPARATOR) == COMPARATOR_SHA256,
            "Supplement input/source bytes changed")
        result["limits"]["previously_compared_geometry_bytes_rehashed"] = True
        result["status"] = "passed"
    except BaseException as error:
        primary = error
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            with output.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
        except BaseException as write_error:
            if primary is not None:
                raise primary from write_error
            raise


if __name__ == "__main__":
    main()
