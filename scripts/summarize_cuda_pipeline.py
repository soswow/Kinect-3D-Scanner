"""Reproduce the October 2026 CUDA pipeline report from local closed profiles.

Contains the original study's fixed hardware/date labels and mode matrix. For
new studies, adapt that metadata before publishing. Full matrices stay under
benchmark-output; publish only selected summaries/charts after review.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def surface_contract_valid(surface):
    if not isinstance(surface, dict):
        return False
    metrics = [surface.get(key) for key in ("surface_p95_m", "precision", "completeness")]
    return (surface.get("threshold_m") == .005
            and surface.get("samples_per_surface") == 30000
            and surface.get("alignment") == "fixed input coordinate frame; no scale or trajectory fitting"
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in metrics)
            and metrics[0] >= 0 and 0 <= metrics[1] <= 1 and 0 <= metrics[2] <= 1)


def quality_status(report, quality):
    if not report["finish_requested"]:
        return "not_requested"
    if not report["mesh_built"]:
        return "failed"
    if quality is None or "triangle_surface_metrics_vs_cpu" not in quality:
        return "missing"
    surface = quality["triangle_surface_metrics_vs_cpu"]
    if not surface_contract_valid(surface):
        return "invalid_contract"
    metrics = [surface.get(key) for key in ("surface_p95_m", "precision", "completeness")]
    return "passed" if (quality.get("same_mesh_success") and quality["same_accepted_indices"]
                        and metrics[0] <= .005 and metrics[1] >= .99 and metrics[2] >= .99) else "failed"


def validate_matched_rows(rows, *, input_only=False, confidence_only=False):
    """Reject incomplete or mismatched data before publishing measured gains."""
    for session in {row["session"] for row in rows}:
        selected = [row for row in rows if row["session"] == session]
        for row in selected:
            if (row["source_unchanged"] is not True or row["input_unchanged"] is not True
                    or row["experimental_policy_unchanged"] is not True
                    or row["pose_seeds_used"] is not False
                    or not row["finish_requested"] or not row["mesh_built"]
                    or row["seed"] is None or row["original_settings_sha256"] is None
                    or row["selected_indices_valid"] is not True
                    or row["comparison_settings_sha256"] is None
                    or row["quality_status"] not in ("passed", "failed")):
                raise ValueError(f"Incomplete immutable/finished quality evidence: {session} / {row['mode']}")
        keys = ("input_sha256", "selected_indices_sha256", "seed", "original_settings_sha256",
                "comparison_settings_sha256")
        if any(len({row[key] for row in selected}) != 1 for key in keys):
            raise ValueError(f"Matched profiles changed observations/settings: {session}")
        # CPU and CUDA variants of one named policy must keep exact settings.
        for family in ({row["mode"].removeprefix("cpu-").removeprefix("legacy-") for row in selected}
                       if not input_only else ("input",)):
            matched = selected if input_only else [row for row in selected
                if row["mode"].removeprefix("cpu-").removeprefix("legacy-") == family]
            if len({row["settings_sha256"] for row in matched}) != 1:
                raise ValueError(f"Matched policy changed exact settings: {session} / {family}")
        if input_only or confidence_only:
            changed_option = "KINECT_CUDA_INPUT" if input_only else "KINECT_CUDA_CONFIDENCE"
            options = [{k: v for k, v in row["pipeline_options"].items()
                        if k != changed_option} for row in selected]
            if len({fingerprint(value) for value in options}) != 1:
                raise ValueError(f"Single-stage experiment changed other pipeline options: {session}")
            if (len({fingerprint(row["thread_policy"]) for row in selected}) != 1
                    or len({row["omp_threads"] for row in selected}) != 1):
                raise ValueError(f"Single-stage experiment changed thread resources: {session}")
        if input_only:
            for row in selected:
                expected = "on" if row["mode"].endswith("-gpu-input") else "off"
                backend = row["backend"].get("cuda_input", {})
                if (row["pipeline_options"].get("KINECT_CUDA_INPUT") != expected
                        or backend.get("requested") != expected
                        or backend.get("confidence_device") != "CPU:0"):
                    raise ValueError(f"Input-mode label does not match execution: {session} / {row['mode']}")
                if expected == "on" and (backend.get("implementation") != "cuda"
                        or backend.get("gpu_batches", 0) <= 0
                        or backend.get("cpu_batches", 0) != 0
                        or backend.get("fallback_batches", 0) != 0
                        or not backend.get("probe_passed")):
                    raise ValueError(f"Input-on profile did not exercise compatible CUDA: {session}")
                if expected == "off" and (backend.get("implementation") != "cpu"
                        or backend.get("gpu_batches", 0) != 0):
                    raise ValueError(f"Input-off profile did not exercise CPU preparation: {session}")
        if confidence_only:
            if len({row["settings_sha256"] for row in selected}) != 1:
                raise ValueError(f"Confidence-only profiles changed settings: {session}")
            for row in selected:
                expected = "on" if row["mode"].endswith("-confidence") else "off"
                backend = row["backend"].get("depth_confidence", {})
                if (row["pipeline_options"].get("KINECT_CUDA_CONFIDENCE") != expected
                        or backend.get("requested") != expected):
                    raise ValueError(f"Confidence-mode label does not match execution: {session}")
                if expected == "on" and (backend.get("implementation") != "cuda"
                        or backend.get("gpu_calls", 0) <= 0
                        or backend.get("cpu_calls", 0) != 0
                        or backend.get("fallback_calls", 0) != 0
                        or backend.get("compatibility_probes", 0) <= 0
                        or backend.get("probe_passed") is not True
                        or backend.get("device") != row["backend"].get("device")):
                    raise ValueError(f"Confidence-on did not exercise compatible CUDA: {session}")
                if expected == "off" and (backend.get("implementation") != "cpu"
                        or backend.get("gpu_calls", 0) != 0):
                    raise ValueError(f"Confidence-off did not exercise CPU: {session}")


def fragment_summary(report):
    data = report.get("fragment_reconnection", {})
    result = {key: data[key] for key in ("applied", "reason", "candidate_pairs", "tested_pairs",
        "budget_limited", "recovered_frames", "corrected_frames", "excluded_frames",
        "fusion_required_blocks", "fusion_block_limit", "fusion_voxel_m", "elapsed_ms") if key in data}
    for key in ("verified_bridges", "loop_closures", "connected_fragments", "unconnected_fragments"):
        if key in data:
            result[key + "_count"] = len(data[key])
    return result


def compact(run):
    import numpy as np

    report_path = Path(run["report"])
    report_bytes = report_path.read_bytes()
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    report = json.loads(report_bytes)
    quality_path = Path(run["report"]).with_name(Path(run["report"]).stem + "-quality.json")
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else None
    status = quality_status(report, quality)
    if quality and quality.get("candidate_report_sha256") is not None and quality["candidate_report_sha256"] != report_sha256:
        status = "stale_report"
    if quality and quality.get("baseline_report_sha256") is not None:
        baseline_path = Path(quality["baseline"])
        if (not baseline_path.exists()
                or hashlib.sha256(baseline_path.read_bytes()).hexdigest() != quality["baseline_report_sha256"]):
            status = "stale_reference"
    elapsed = [row["elapsed_ms"] for row in report["live_diagnostics"]]
    good = [row["elapsed_ms"] for row in report["live_diagnostics"] if row["success"]]
    failed = [row["elapsed_ms"] for row in report["live_diagnostics"] if not row["success"]]
    settings = dict(report["settings"])
    final_voxel = settings.get("final_voxel_m") or settings["voxel_m"]
    normalized = dict(settings)
    normalized.pop("voxel_m")
    normalized.pop("final_voxel_m", None)
    normalized["effective_final_voxel_m"] = final_voxel
    final_matches = (report.get("final_reconstruction", {}).get("voxel_m") == final_voxel)
    indices = report.get("selected_indices")
    indices_valid = (isinstance(indices, list) and len(indices) == report["frames"]
                     and all(type(index) is int and index >= 0 for index in indices)
                     and len(set(indices)) == len(indices))
    return {"session": run["session"], "mode": run["mode"], "repeat": run["repeat"],
            "input_sha256": report["input_sha256"], "source_sha256": report["source_sha256"],
            "source_unchanged": not report["source_changed_during_profile"],
            "input_unchanged": not report["input_changed_during_profile"] if "input_changed_during_profile" in report else None,
            "experimental_policy_unchanged": not report["experimental_policy_changed"] if "experimental_policy_changed" in report else None,
            "finish_requested": report["finish_requested"], "seed": report.get("seed"),
            "pose_seeds_used": report.get("pose_seeds_used"),
            "selected_indices_sha256": fingerprint(indices) if indices_valid else None,
            "selected_indices_valid": indices_valid,
            "original_settings_sha256": report.get("original_settings_sha256"),
            "settings_sha256": fingerprint(settings),
            "comparison_settings_sha256": fingerprint(normalized) if final_matches else None,
            "pipeline_options": report.get("pipeline_options", {}),
            "report_sha256": report_sha256,
            "quality_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest() if quality else None,
            "quality_candidate_report_sha256": quality.get("candidate_report_sha256") if quality else None,
            "quality_baseline_report_sha256": quality.get("baseline_report_sha256") if quality else None,
            "quality_status": status,
            "experimental_visual_fallback": report.get("experimental_visual_fallback", False),
            "experimental_policy_sha256": report.get("experimental_policy_sha256"),
            "frames": report["frames"], "live_s": report["live_s"],
            "processing_fps": report["frames"]/report["live_s"],
            "frame_median_ms": float(np.median(elapsed)), "frame_p95_ms": float(np.percentile(elapsed,95)),
            "first_frame_ms": elapsed[0],
            "accepted_frame_median_ms": float(np.median(good)),
            "rejected_frame_median_ms": float(np.median(failed)) if failed else None,
            "live_accepted": report["accepted_before_finish"],
            "live_accepted_indices": report["accepted_indices_before_finish"],
            "finish_s": report["finish_s"], "final_accepted": report["accepted"] if report["finish_requested"] else None,
            "final_accepted_indices": report["accepted_indices"] if report["finish_requested"] else None,
            "mesh_built": report["mesh_built"], "backend": report["backend"],
            "thread_policy": report.get("thread_policy"),
            "omp_threads": report.get("omp_threads"),
            "gpu_hardware": report.get("gpu_hardware"),
            "surface_agreement": quality.get("triangle_surface_metrics_vs_cpu") if quality else None,
            "same_final_indices_as_cpu": quality.get("same_accepted_indices") if quality else None,
            "preview_resolution_comparison": quality.get("preview_resolution_comparison", False) if quality else False,
            "settings_overrides": report["settings_overrides"],
            "stage_seconds": {name: data["total_ms"]/1000 for name,data in report["live_stages"].items()},
            "tracking_methods": report["tracking_methods"],
            "fragment_reconnection": fragment_summary(report),
            "peak_host_rss_gib": report["peak_process_rss_bytes"]/2**30}


def aggregate(rows, baseline_mode="baseline"):
    import numpy as np

    aggregated = []
    for session in sorted({row["session"] for row in rows}):
        baseline = [row for row in rows if row["session"] == session and row["mode"] == baseline_mode]
        if not baseline:
            continue
        base_time = float(np.median([row["live_s"] for row in baseline]))
        base_accepted = float(np.median([row["live_accepted"] for row in baseline]))
        baseline_finished = [row for row in baseline if row["mesh_built"]
                             and isinstance(row["finish_s"], (int, float))
                             and math.isfinite(row["finish_s"]) and row["finish_s"] >= 0
                             and row["quality_status"] in ("passed", "failed")
                             and surface_contract_valid(row["surface_agreement"])]
        base_complete = float(np.median([row["live_s"] + row["finish_s"] for row in baseline_finished])) if baseline_finished else None
        for mode in dict.fromkeys(row["mode"] for row in rows):
            selected = [row for row in rows if row["session"] == session and row["mode"] == mode]
            if not selected:
                continue
            elapsed = [row["live_s"] for row in selected]
            finished = [row for row in selected if row["mesh_built"]
                        and isinstance(row["finish_s"], (int, float))
                        and math.isfinite(row["finish_s"]) and row["finish_s"] >= 0
                        and row["quality_status"] in ("passed", "failed")
                        and surface_contract_valid(row["surface_agreement"])]
            finish_times = [row["finish_s"] for row in finished]
            quality_unknown = any(row["finish_requested"] and row["quality_status"] in
                                  ("missing", "invalid_contract", "stale_report", "stale_reference") for row in selected)
            quality_failed = any(row["finish_requested"] and
                (row["quality_status"] == "failed" or not row["mesh_built"]) for row in selected)
            aggregated.append({"session": session, "mode": mode, "repeats": len(selected),
                "frames": selected[0]["frames"], "median_live_s": float(np.median(elapsed)),
                "live_range_s": [min(elapsed), max(elapsed)], "baseline_mode": baseline_mode,
                "speedup_vs_baseline": base_time/float(np.median(elapsed)),
                "complete_speedup_vs_baseline": base_complete / float(np.median([row["live_s"] + row["finish_s"] for row in finished])) if finished and base_complete is not None and not quality_unknown else None,
                "processing_fps": selected[0]["frames"]/float(np.median(elapsed)),
                "accepted_range": [min(row["live_accepted"] for row in selected), max(row["live_accepted"] for row in selected)],
                "severe_live_coverage_loss": min(row["live_accepted"] for row in selected) < base_accepted / 2,
                "finished_quality_checks": len(finished), "finished_quality_failed": True if quality_failed else None if quality_unknown else False,
                "finished_quality_unknown": quality_unknown,
                "median_finish_s": float(np.median(finish_times)) if finished else None,
                "median_complete_s": float(np.median([row["live_s"] + row["finish_s"] for row in finished])) if finished else None,
                "finish_range_s": [min(finish_times), max(finish_times)] if finished else None,
                "complete_range_s": [min(row["live_s"] + row["finish_s"] for row in finished),
                                     max(row["live_s"] + row["finish_s"] for row in finished)] if finished else None,
                "final_accepted_range": [min(row["final_accepted"] for row in finished),
                                         max(row["final_accepted"] for row in finished)] if finished else None,
                "worst_surface_p95_mm": max(row["surface_agreement"]["surface_p95_m"] * 1000 for row in finished) if finished else None,
                "minimum_precision": min(row["surface_agreement"]["precision"] for row in finished) if finished else None,
                "minimum_completeness": min(row["surface_agreement"]["completeness"] for row in finished) if finished else None,
                "source_sha256": sorted({row["source_sha256"] for row in selected}),
                "frame_median_ms": float(np.median([row["frame_median_ms"] for row in selected])),
                "frame_p95_ms": float(np.median([row["frame_p95_ms"] for row in selected])),
                "stage_seconds": {name: float(np.median([row["stage_seconds"].get(name,0) for row in selected]))
                                  for name in selected[0]["stage_seconds"]}})
    return aggregated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("benchmark-output/cuda-pipeline"))
    parser.add_argument("--summary", type=Path, default=Path("benchmark-output/cuda-pipeline-experiments.json"))
    args = parser.parse_args()
    import numpy as np

    groups = {}
    for subdir in ("controlled-live", "finish", "phase2-pilot", "phase2-live", "phase2-finish",
                   "phase3-finish", "phase4-finish", "legacy-before-restart", "legacy-finish", "legacy-repeat", "legacy-sift", "adaptive-preview", "adaptive-repeat", "adaptive-fine", "cpu-adaptive-fine", "production-control", "preview-repeat", "cpu-control", "validated-live", "gpu-input-integration", "gpu-confidence-integration", "current-release-control"):
        path = args.directory/subdir/"experiments.json"
        if path.exists():
            groups[subdir] = [compact(run) for run in json.loads(path.read_text())["runs"]]
    if not groups.get("controlled-live"):
        parser.error("Require completed controlled live reports")
    aggregated = aggregate(groups["controlled-live"])
    phase2 = aggregate(groups.get("phase2-live", []))
    validated = aggregate(groups.get("validated-live", []))
    # The first post-update chest-3 baseline paid a 98 s cold GPU startup.
    # Preserve it in cold_hybrid_summary, rather than inflating warmed gains.
    warm = (groups.get("legacy-repeat", []) + groups.get("preview-repeat", []) + groups.get("legacy-sift", [])
            + groups.get("adaptive-preview", []))
    hybrid = aggregate(warm) or aggregate(groups.get("legacy-finish", []))
    current = (groups.get("cpu-control", []) + groups.get("production-control", [])
               + groups.get("adaptive-repeat", []) + groups.get("adaptive-fine", [])
               + groups.get("cpu-adaptive-fine", []))
    current_sources = sorted({row["source_sha256"] for row in current})
    current_drivers = {json.dumps(row["gpu_hardware"], sort_keys=True) for row in current}
    if len(current_sources) > 1 or len(current_drivers) > 1:
        raise ValueError("Primary CPU/CUDA comparison requires one unchanged source and driver")
    validate_matched_rows(current)
    input_rows = groups.get("gpu-input-integration", [])
    if (len({row["source_sha256"] for row in input_rows}) > 1
            or len({json.dumps(row["gpu_hardware"], sort_keys=True) for row in input_rows}) > 1):
        raise ValueError("Input acceleration comparison requires one unchanged source and driver")
    validate_matched_rows(input_rows, input_only=True)
    confidence_rows = groups.get("gpu-confidence-integration", [])
    if (len({row["source_sha256"] for row in confidence_rows}) > 1
            or len({json.dumps(row["gpu_hardware"], sort_keys=True) for row in confidence_rows}) > 1):
        raise ValueError("Confidence comparison requires one unchanged source and driver")
    validate_matched_rows(confidence_rows, confidence_only=True)
    release_rows = []
    if groups.get("current-release-control"):
        release_rows = groups["current-release-control"] + confidence_rows
        expected_modes = {"cpu-baseline": 1, "cpu-adaptive-preview-10mm": 1,
                          "legacy-adaptive-preview-10mm-gpu-input": 2,
                          "legacy-adaptive-preview-10mm-gpu-input-confidence": 2}
        sessions = {row["session"] for row in groups["current-release-control"]}
        if len(sessions) != 2 or sessions != {row["session"] for row in confidence_rows}:
            raise ValueError("Release comparison requires both original archives in CPU and CUDA modes")
        for session in sessions:
            selected = [row for row in release_rows if row["session"] == session]
            if {mode: sum(row["mode"] == mode for row in selected) for mode in expected_modes} != expected_modes:
                raise ValueError(f"Release comparison lacks complete control/reversed repeats: {session}")
            if {row["mode"] for row in selected} != set(expected_modes):
                raise ValueError(f"Unexpected mode in release comparison: {session}")
            for mode, count in expected_modes.items():
                repeated = [row for row in selected if row["mode"] == mode]
                if ({row["repeat"] for row in repeated} != set(range(count))
                        or len({row["report_sha256"] for row in repeated}) != count):
                    raise ValueError(f"Release comparison duplicated a repetition: {session} / {mode}")
        validation = json.loads((args.directory / "validation.json").read_text())
        if (not validation["success"] or validation["source_changed_during_validation"]
                or {row["source_sha256"] for row in release_rows} != {validation["source_sha256"]}
                or {fingerprint(row["gpu_hardware"]) for row in release_rows} != {fingerprint(validation["gpu_hardware"])}
                or len({fingerprint(row["thread_policy"]) for row in release_rows}) != 1
                or len({row["omp_threads"] for row in release_rows}) != 1):
            raise ValueError("Release comparison changed validated source, driver or thread resources")
        validate_matched_rows(release_rows)
        for session in sessions:
            preview_rows = [row for row in release_rows if row["session"] == session and row["mode"] != "cpu-baseline"]
            if len({row["settings_sha256"] for row in preview_rows}) != 1:
                raise ValueError(f"Matched release CUDA backend changed exact preview settings: {session}")
            backend_options = [{key: value for key, value in row["pipeline_options"].items()
                if key not in ("KINECT_CUDA_MATCHING", "KINECT_CUDA_INPUT", "KINECT_CUDA_CONFIDENCE")}
                for row in preview_rows]
            if len({fingerprint(options) for options in backend_options}) != 1:
                raise ValueError(f"Matched release CUDA backend changed tracking policy: {session}")
            for row in preview_rows:
                if row["mode"].startswith("legacy-"):
                    preparation = row["backend"].get("cuda_input", {})
                    if (preparation.get("requested") != "on" or preparation.get("implementation") != "cuda"
                            or preparation.get("gpu_batches", 0) <= 0 or preparation.get("cpu_batches", 0) != 0
                            or preparation.get("fallback_batches", 0) != 0 or not preparation.get("probe_passed")):
                        raise ValueError(f"Release CUDA mode did not exercise exact GPU input: {session}")
        if any(row["quality_status"] != "passed" or row["quality_candidate_report_sha256"] != row["report_sha256"]
               or row["quality_baseline_report_sha256"] is None or row["backend"]["device"] !=
               ("CPU:0" if row["mode"].startswith("cpu-") else "CUDA:0") for row in release_rows):
            raise ValueError("Release comparison requires passed surface gates and the named devices")
    matched = aggregate(current, baseline_mode="cpu-baseline")
    # Prior-source hybrid runs verify repeatability, but do not enter the
    # primary same-source warm timing comparison.
    hybrid_checks = [row for row in groups.get("legacy-finish", []) + warm
                     if not (row["mode"] == "baseline" and row["first_frame_ms"] > 10000)]
    auxiliary = {}
    for name in ("orb-matching", "sift-matching", "registration-options", "multiscale-registration", "visual-refinement",
                 "visual-fine-refinement", "visual-fine-gpu64-normals", "nearest-registration", "reranked-registration",
                 "opencv-threads", "thread-pilot", "thread-cpu-pilot", "sift-distances-fp32", "sift-distances-tf32",
                 "dense-model-synthetic", "dense-model-hybrid", "dense-model-stationary",
                 "exhaustive-matching", "exhaustive-tie-matching", "sift-bound-census",
                 "kdtree-registration", "kdtree-check", "anchored-model",
                 "anchored-model-dense", "anchored-model-measured", "anchored-model-submaps", "anchored-model-submaps-repeat",
                 "parallel-fragments/chest-3", "parallel-fragments/chest-3-dynamic",
                 "parallel-fragments/chest-3-proposals", "parallel-fragments/capture-overlap-estimate",
                 "gpu-prepare/without-confidence", "gpu-prepare/with-confidence", "gpu-prepare/production-parity",
                 "gpu-prepare/with-confidence-exact-reductions", "gpu-prepare/with-confidence-exact-fused-filters",
                 "gpu-prepare/confidence-ablation", "gpu-prepare/production-confidence-parity",
                 "gpu-prepare/production-confidence-boundaries", "confidence-cache/chest-reuse",
                 "gpu-confidence-integration/diagnosis-first-order", "current-release-activity",
                 "verification-profile/chest-3-summary", "optix-nearest/research-summary",
                 "resident-icp/research-summary", "selective-fragment-threads/research-summary",
                 "uniform-grid-nearest/research-summary",
                 "uniform-grid-nearest/staged-pruned-v1/research-summary",
                 "uniform-grid-nearest/flat-v1/research-summary",
                 "uniform-grid-nearest/device-resident-v1/research-summary",
                 "grid-lookup-ablation-v1/research-summary",
                 "grid-lookup-ablation-order-v2/host-diagnostics-summary",
                 "native-icp-signatures/research-summary"):
        path = args.directory/(name+".json")
        if path.exists():
            data = json.loads(path.read_text())
            # Dense prototype poses are diagnostic artifacts, not summary data.
            if "records" in data:
                data.pop("records")
            for run in data.get("runs", []):
                run.pop("records", None)
            auxiliary[name] = data
    quality = {str(path.relative_to(args.directory)): {k:v for k,v in json.loads(path.read_text()).items() if k != "baseline"}
               for folder in ("finish", "phase2-finish", "phase3-finish", "phase4-finish", "legacy-finish", "legacy-repeat", "legacy-sift", "adaptive-preview", "adaptive-repeat", "adaptive-fine", "cpu-adaptive-fine", "production-control", "preview-repeat", "cpu-control", "gpu-input-integration", "gpu-confidence-integration", "current-release-control", "current-release-clean-control")
               for path in (args.directory/folder).glob("*-quality.json")}
    summary = {"date": "2026-10-08", "hardware": "RTX 3080 Ti 12 GB / Intel i7-12700F / Windows 11 / 64 GiB RAM",
        "measurement": "Sequential isolated replays; all original views, calibrations and verification thresholds. Primary CPU/CUDA controls preserve 5 mm live/final fusion. Explicit adaptive preview tradeoff uses 10 mm live fusion and the same 5 mm final reconstruction/budget. Timing includes processing rejected views, but excludes imports, ZIP decoding, USB and network. Final agreement uses original coordinates without alignment; CPU reference is not independent ground truth. The primary matched-source matrix shares one source fingerprint and driver 610.88. The frozen release has its own same-source CPU baseline and alternating-order CUDA repetitions, with matching thread resources. Historical controls and the scripted adaptive prototype retain their separate provenance.",
        "primary_source_sha256": current_sources,
        "runs": groups, "summary": aggregated, "phase2_summary": phase2, "validated_summary": validated,
        "hybrid_summary": hybrid,
        "matched_processing_summary": matched,
        "matched_backend_summary": aggregate([row for row in current
            if row["mode"] in ("cpu-cached", "legacy-cached")], baseline_mode="cpu-cached"),
        "matched_preview_backend_summary": aggregate([row for row in groups.get("cpu-control", []) + warm
            if row["mode"] in ("cpu-sift-preview-10mm", "legacy-sift-preview-10mm")], baseline_mode="cpu-sift-preview-10mm"),
        "matched_adaptive_backend_summary": aggregate([row for row in current
            if row["mode"] in ("cpu-adaptive-preview-10mm", "legacy-adaptive-preview-10mm")], baseline_mode="cpu-adaptive-preview-10mm"),
        "matched_fine_adaptive_backend_summary": aggregate([row for row in current
            if row["mode"] in ("cpu-adaptive", "legacy-adaptive")], baseline_mode="cpu-adaptive"),
        "gpu_input_processing_summary": aggregate(groups.get("gpu-input-integration", []),
            baseline_mode="legacy-adaptive-preview-10mm"),
        "gpu_confidence_processing_summary": aggregate(confidence_rows,
            baseline_mode="legacy-adaptive-preview-10mm-gpu-input"),
        "current_release_processing_summary": aggregate(release_rows, baseline_mode="cpu-baseline"),
        "current_release_matched_backend_summary": aggregate([row for row in release_rows
            if row["mode"] != "cpu-baseline"], baseline_mode="cpu-adaptive-preview-10mm"),
        "hybrid_repeatability_summary": aggregate(hybrid_checks),
        "cold_hybrid_summary": aggregate(groups.get("legacy-finish", [])),
        "agreement_gate": "Same CPU final indices; triangle surface p95 <=5 mm and precision/completeness >=99% within 5 mm. Agreement is not absolute accuracy.",
        "component_experiments": auxiliary, "quality": quality}
    validation = args.directory / "validation.json"
    if validation.exists():
        summary["validation"] = json.loads(validation.read_text())
    measured_validation = args.directory / "validation-measured.json"
    if measured_validation.exists():
        summary["measured_runtime_validation"] = json.loads(measured_validation.read_text())
    input_validation = args.directory / "validation-input-measured.json"
    if input_validation.exists():
        summary["input_runtime_validation"] = json.loads(input_validation.read_text())
    summary["integration_validation"] = {}
    for name in ("gpu-input-integration-validation", "gpu-confidence-integration-validation",
                 "current-release-validation", "summary-provenance-validation"):
        path = args.directory / (name + ".json")
        if path.exists():
            summary["integration_validation"][name] = json.loads(path.read_text())
    summary["server_integration_checks"] = {}
    for name in ("server-check", "server-native-check", "server-ready"):
        path = args.directory / (name + ".json")
        if path.exists():
            summary["server_integration_checks"][name] = json.loads(path.read_text())
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"baseline": "CUDA fusion + GPU tracking (control)", "cpu-baseline": "CPU pipeline (control)", "cpu-cached": "CPU tracking + caching/preparation", "cached": "Caching + CUDA matching",
              "sift": "SIFT + GPU geometry (rejected)", "sift-deferred": "Deferred recovery (rejected)",
              "sift-cpu": "SIFT + CPU verification", "measured": "Measured live poses",
              "measured-final": "Measured live + Finish", "measured-visual-final": "Measured live + visual bridges",
              "orb-fine": "Feature-seeded fine ICP (experimental)", "sift-fine": "SIFT fine ICP (experimental)",
              "legacy-cached": "CPU tracking + CUDA fusion/retrieval"}
    labels["legacy-preview-10mm"] = "10 mm preview / 5 mm final (tradeoff)"
    labels["legacy-sift"] = "CPU tracking + CUDA SIFT retrieval"
    labels["legacy-sift-preview-10mm"] = "SIFT + 10 mm preview / 5 mm final"
    labels["cpu-sift-preview-10mm"] = "CPU SIFT + 10 mm preview / 5 mm final"
    labels["legacy-adaptive-preview-10mm"] = "CUDA adaptive / 10 mm preview"
    labels["cpu-adaptive-preview-10mm"] = "CPU ORB + SIFT fallback / 10 mm preview"
    labels["legacy-adaptive"] = "CUDA adaptive / 5 mm preview"
    labels["cpu-adaptive"] = "CPU adaptive / 5 mm preview"
    colors = {"baseline": "#75869a", "cpu-baseline": "#b0b8c5", "cpu-cached": "#81a1bf", "cached": "#5492b7", "sift": "#209582", "sift-deferred": "#d4943d"}
    colors.update({"sift-cpu": "#5492b7", "measured": "#209582", "measured-final": "#209582",
                   "measured-visual-final": "#6e60a7", "orb-fine": "#d4943d", "sift-fine": "#d4943d",
                   "legacy-cached": "#209582"})
    colors["legacy-preview-10mm"] = "#6e60a7"
    colors["legacy-sift"] = "#d4943d"
    colors["legacy-sift-preview-10mm"] = "#b980b9"
    colors["cpu-sift-preview-10mm"] = "#bca79c"
    colors["legacy-adaptive-preview-10mm"] = "#456bb1"
    colors["cpu-adaptive-preview-10mm"] = "#7f9fc7"
    colors["legacy-adaptive"] = "#456bb1"
    colors["cpu-adaptive"] = "#7f9fc7"
    plot_rows = matched or hybrid or validated or phase2 or aggregated
    mode_order = ("cpu-baseline", "cpu-cached", "legacy-cached", "cpu-adaptive", "legacy-adaptive",
                  "cpu-adaptive-preview-10mm", "legacy-adaptive-preview-10mm")
    plot_rows = sorted(plot_rows, key=lambda row: (row["session"],
        mode_order.index(row["mode"]) if row["mode"] in mode_order else len(mode_order)))
    fig, axes = plt.subplots(1,2,figsize=(18,7),sharex=True,sharey=True)
    for ax,session in zip(axes, sorted({row["session"] for row in plot_rows})):
        selected = [row for row in plot_rows if row["session"] == session]
        y = np.arange(len(selected))
        times = [row["median_live_s"] for row in selected]
        bars = ax.barh(y,times,color=[colors[row["mode"]] for row in selected])
        for bar, row in zip(bars, selected):
            if row["finished_quality_unknown"] or row["severe_live_coverage_loss"] or row["finished_quality_failed"]:
                bar.set_hatch("///")
                bar.set_alpha(.5)
        ax.errorbar(times,y,xerr=np.array([[row["median_live_s"]-row["live_range_s"][0],row["live_range_s"][1]-row["median_live_s"]] for row in selected]).T,
                    fmt="none",ecolor="#253746",capsize=4)
        ax.set_yticks(y,[labels[row["mode"]] for row in selected]);ax.set_ylim(len(selected)-.5,-.5)
        for i,row in enumerate(selected):
            accepted = "–".join(map(str,row["accepted_range"])) if row["accepted_range"][0] != row["accepted_range"][1] else str(row["accepted_range"][0])
            gain = "QUALITY UNKNOWN" if row["finished_quality_unknown"] else "FAILED live coverage" if row["severe_live_coverage_loss"] else "FAILED geometry agreement" if row["finished_quality_failed"] else f"{row['speedup_vs_baseline']:.2f}×"
            ax.text(row["median_live_s"]+3,i,f"{row['median_live_s']:.1f}s · {gain} · {accepted}/{row['frames']} views",va="center",fontsize=9)
        ax.set_title(session[:7]); ax.set_xlabel("Live replay seconds (lower is faster)")
        ax.set_xlim(0,max(times)*1.65); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Earlier pipeline: live processing and retained views",fontsize=15)
    fig.text(.5,.015,"Matched archive replays · imports, USB and network excluded · failed coverage/geometry explicitly marked",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(args.directory/"pipeline-performance.png",dpi=150)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    for ax, session in zip(axes, sorted({row["session"] for row in plot_rows})):
        selected = [row for row in plot_rows if row["session"] == session and row["median_finish_s"] is not None]
        y = np.arange(len(selected))
        live = [row["median_live_s"] for row in selected]
        finish = [row["median_finish_s"] for row in selected]
        live_bars = ax.barh(y, live, color="#456bb1", label="Live processing")
        finish_bars = ax.barh(y, finish, left=live, color="#c2ccd6", label="Finish")
        ax.set_yticks(y, [labels[row["mode"]] for row in selected]); ax.set_ylim(len(selected)-.5,-.5)
        for i, (row, live_bar, finish_bar) in enumerate(zip(selected, live_bars, finish_bars)):
            failed = row["severe_live_coverage_loss"] or row["finished_quality_failed"]
            if failed:
                live_bar.set_hatch("///"); finish_bar.set_hatch("///")
            complete = row["median_complete_s"]
            gain = "FAILED" if failed else f"{row['complete_speedup_vs_baseline']:.2f}×" if row["complete_speedup_vs_baseline"] else ""
            ax.text(complete + 5, i, f"{complete:.1f}s · {gain}", va="center", fontsize=10)
        ax.set_title(session[:7]); ax.set_xlabel("Live processing + Finish seconds (lower is faster)")
        if selected:
            ax.set_xlim(0, max(row["median_complete_s"] for row in selected) * 1.27)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[1]:
            ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Earlier pipeline: complete reconstruction and Finish cost", fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(.5, .035))
    fig.text(.5, .015, "Sequential complete archive replays · 5 mm final reconstruction · USB/network and export excluded", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .94)); fig.savefig(args.directory / "pipeline-complete.png", dpi=150)
    stage_rows = summary["gpu_confidence_processing_summary"]
    if stage_rows:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        y = np.arange(len(stage_rows))
        names = [f"{row['session'][:7]} · {'CUDA' if row['mode'].endswith('-confidence') else 'CPU'} confidence"
                 for row in stage_rows]
        live = [row["median_live_s"] for row in stage_rows]
        finish = [row["median_finish_s"] for row in stage_rows]
        axes[0].barh(y, live, color=["#23856d" if row["mode"].endswith("-confidence") else "#7f9fc7" for row in stage_rows])
        axes[1].barh(y, live, color="#456bb1", label="Live processing")
        axes[1].barh(y, finish, left=live, color="#c2ccd6", label="Finish")
        for i, row in enumerate(stage_rows):
            gate = "PASS" if row["finished_quality_checks"] == row["repeats"] and row["finished_quality_failed"] is False else "FAILED / UNKNOWN"
            axes[0].text(live[i] + 1, i, f"{live[i]:.2f}s", va="center", fontsize=9)
            axes[1].text(live[i] + finish[i] + 3, i, f"{live[i] + finish[i]:.2f}s · {gate}", va="center", fontsize=9)
        for ax in axes:
            ax.set_yticks(y, names)
            ax.set_ylim(len(stage_rows) - .5, -.5)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_xlim(0, max(live) * 1.25)
        axes[1].set_xlim(0, max(a + b for a, b in zip(live, finish)) * 1.3)
        axes[0].set_xlabel("Live processing seconds")
        axes[1].set_xlabel("Complete processing seconds")
        fig.suptitle("Additional exact CUDA confidence: matched complete archive replays", fontsize=13)
        fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(.5, .045))
        repeat_counts = sorted({row["repeats"] for row in stage_rows})
        repetition = str(repeat_counts[0]) if len(repeat_counts) == 1 else "–".join(map(str, repeat_counts))
        fig.text(.5, .015, f"GPU RGB/depth input in both modes · 10 mm preview / 5 mm Finish · {repetition} repeats per mode, alternating order", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .13, 1, .94))
        fig.savefig(args.directory / "pipeline-confidence.png", dpi=150)
    release_summary = summary["current_release_processing_summary"]
    release_labels = {"cpu-baseline": "CPU baseline · 5 mm live",
                      "cpu-adaptive-preview-10mm": "CPU adaptive · 10 mm live",
                      "legacy-adaptive-preview-10mm-gpu-input": "CUDA input · 10 mm live",
                      "legacy-adaptive-preview-10mm-gpu-input-confidence": "CUDA input + confidence · 10 mm live"}
    if release_summary:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        y = np.arange(len(release_summary))
        names = [f"{row['session'][:7]} · {release_labels[row['mode']]}" for row in release_summary]
        live = [row["median_live_s"] for row in release_summary]
        finish = [row["median_finish_s"] for row in release_summary]
        axes[0].barh(y, live, color=["#7f9fc7" if row["mode"].startswith("cpu-") else "#23856d" for row in release_summary])
        axes[1].barh(y, live, color="#456bb1", label="Live processing")
        axes[1].barh(y, finish, left=live, color="#c2ccd6", label="Finish")
        for i, row in enumerate(release_summary):
            axes[0].text(live[i] + 2, i, f"{live[i]:.1f}s · {row['speedup_vs_baseline']:.2f}×", va="center", fontsize=9)
            axes[1].text(row["median_complete_s"] + 4, i, f"{row['median_complete_s']:.1f}s · PASS", va="center", fontsize=9)
        for ax in axes:
            ax.set_yticks(y, names)
            ax.set_ylim(len(release_summary) - .5, -.5)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_xlim(0, max(live) * 1.35)
        axes[1].set_xlim(0, max(row["median_complete_s"] for row in release_summary) * 1.3)
        axes[0].set_xlabel("Live processing seconds")
        axes[1].set_xlabel("Live processing + Finish seconds")
        fig.suptitle("Frozen release: CPU baseline and CUDA-assisted fast preview", fontsize=14)
        fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(.5, .055))
        fig.text(.5, .015, "One repetition per CPU mode / two per CUDA mode · all Finish at 5 mm · compare 10 mm rows to isolate the CUDA backend", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .13, 1, .94))
        fig.savefig(args.directory / "pipeline-current-release.png", dpi=150)
    report = ["# CUDA pipeline experiments", "", summary["measurement"], "",
              "| Session | Mode | Live median (range) | Live speedup | Processing rate | Live retained | Typical / p95 frame | Finish | Complete | Surface p95 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    if release_summary:
        release_report = ["## Frozen release compared with the original CPU workflow", "",
            "One repetition per CPU mode and two alternating-order repetitions per CUDA mode and archive, using the same tested source, driver and thread resources. The original CPU baseline uses 5 mm live fusion; all other rows use the same adaptive 10 mm preview policy. Every row Finishes at 5 mm with unchanged observations, memory budget and surface gates. Comparison with the original baseline measures combined workflow gains; comparison among 10 mm rows measures the additional CUDA backend benefit.", "",
            "| Archive | Workflow | Repeats | Live median (range) | Finish median | Complete median (range) | Live capacity | Typical / p95 view | Final views | Surface p95 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in release_summary:
            release_report.append(f"| {row['session'][:7]} | {release_labels[row['mode']]} | {row['repeats']} | {row['median_live_s']:.2f}s ({row['live_range_s'][0]:.2f}–{row['live_range_s'][1]:.2f}) | {row['median_finish_s']:.2f}s | {row['median_complete_s']:.2f}s ({row['complete_range_s'][0]:.2f}–{row['complete_range_s'][1]:.2f}) | {row['processing_fps']:.2f} views/s | {row['frame_median_ms']:.0f} / {row['frame_p95_ms']:.0f} ms | {row['final_accepted_range'][0]}/{row['frames']} | {row['worst_surface_p95_mm']:.3f} mm |")
        release_report += ["", "![Current release processing](pipeline-current-release.png)", "", "## Earlier matched-policy CUDA attribution", ""]
        release_report[-2:-2] = ["| Archive | CUDA preparation | Matched CPU live time reduction | Matched CPU complete time reduction |",
            "| --- | --- | ---: | ---: |"] + [
            f"| {row['session'][:7]} | {'input + confidence' if row['mode'].endswith('-confidence') else 'input'} | {(1 - 1 / row['speedup_vs_baseline']) * 100:.1f}% | {(1 - 1 / row['complete_speedup_vs_baseline']) * 100:.1f}% |"
            for row in summary["current_release_matched_backend_summary"] if row["mode"].startswith("legacy-")
        ] + [""]
        report[4:4] = release_report
    for row in plot_rows:
        low,high=row["accepted_range"]
        accepted=str(low) if low==high else f"{low}–{high}"
        gain = "QUALITY UNKNOWN" if row["finished_quality_unknown"] else "FAILED live coverage" if row["severe_live_coverage_loss"] else "FAILED geometry agreement" if row["finished_quality_failed"] else f"{row['speedup_vs_baseline']:.2f}×"
        finish = f"{row['median_finish_s']:.1f}s" if row["median_finish_s"] is not None else "—"
        complete = f"{row['median_complete_s']:.1f}s" if row["median_complete_s"] is not None else "—"
        surface = f"{row['worst_surface_p95_mm']:.3f} mm" if row["worst_surface_p95_mm"] is not None else "—"
        report.append(f"| {row['session'][:7]} | {labels[row['mode']]} | {row['median_live_s']:.1f}s ({row['live_range_s'][0]:.1f}–{row['live_range_s'][1]:.1f}) | {gain} | {row['processing_fps']:.2f} views/s | {accepted}/{row['frames']} | {row['frame_median_ms']:.0f} / {row['frame_p95_ms']:.0f} ms | {finish} | {complete} | {surface} |")
    report += ["", "![Processing time](pipeline-performance.png)", "", "![Complete reconstruction](pipeline-complete.png)"]
    if input_rows:
        report += ["", "## Matched native GPU input processing", "",
                   "Separate release-source comparison: the same CUDA adaptive 10 mm preview / 5 mm Finish, with GPU RGB/depth input preparation off or on. Confidence remains on CPU. Includes compatibility/setup work; excludes ZIP decoding and transport.", "",
                   "| Session | GPU input | Live | Finish | Complete | Live gain | Complete gain | Final surface gate |",
                   "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in summary["gpu_input_processing_summary"]:
            passed = row["finished_quality_checks"] == row["repeats"] and not row["finished_quality_failed"]
            report.append(f"| {row['session'][:7]} | {'on' if row['mode'].endswith('-gpu-input') else 'off'} | {row['median_live_s']:.2f}s | {row['median_finish_s']:.2f}s | {row['median_complete_s']:.2f}s | {row['speedup_vs_baseline']:.3f}× | {row['complete_speedup_vs_baseline']:.3f}× | {'PASS' if passed else 'FAIL / missing'} |")
    if confidence_rows:
        report += ["", "## Matched exact CUDA confidence", "",
                   "Separate matched-source comparison with GPU RGB/depth input enabled in both cases. Only the confidence processor changes; all raw observations, pose gates, fusion settings and CPU-reference surface gates remain fixed.", "",
                   "| Session | GPU confidence | Live | Finish | Complete | Live gain | Complete gain | Final surface gate |",
                   "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in summary["gpu_confidence_processing_summary"]:
            passed = row["finished_quality_checks"] == row["repeats"] and not row["finished_quality_failed"]
            report.append(f"| {row['session'][:7]} | {'on' if row['mode'].endswith('-confidence') else 'off'} | {row['median_live_s']:.2f}s | {row['median_finish_s']:.2f}s | {row['median_complete_s']:.2f}s | {row['speedup_vs_baseline']:.3f}× | {row['complete_speedup_vs_baseline']:.3f}× | {'PASS' if passed else 'FAIL / missing'} |")
        report += ["", "![Exact CUDA confidence comparison](pipeline-confidence.png)"]
    report += ["", "## Finish and reconstruction agreement", ""]
    for folder in ("cpu-control", "cpu-adaptive-fine", "production-control", "legacy-finish", "legacy-repeat", "legacy-sift", "adaptive-preview", "adaptive-repeat", "adaptive-fine", "preview-repeat", "phase4-finish", "phase3-finish", "phase2-finish", "finish", "gpu-input-integration", "gpu-confidence-integration", "current-release-control"):
        for row in groups.get(folder, []):
            report.append(f"- {row['session'][:7]} / {row['mode']}: Finish {row['finish_s']:.1f}s; final {row['final_accepted']}/{row['frames']} views; mesh success {row['mesh_built']}.")
    for name,comparison in quality.items():
        surface=comparison.get("triangle_surface_metrics_vs_cpu",{})
        metric = lambda key, scale, unit: (f"{surface[key]*scale:.3f}{unit}"
            if isinstance(surface.get(key), (int, float)) and math.isfinite(surface[key]) else "missing")
        report.append(f"- {name}: same retained indices {comparison.get('same_accepted_indices', 'missing')}; surface p95 {metric('surface_p95_m',1000,' mm')}; precision/completeness within 5 mm {metric('precision',100,'%')} / {metric('completeness',100,'%')}. CPU reference is not independent ground truth.")
    report += ["", "See `docs/CUDA_EXPERIMENTS.md` for rejected experiments, mode selection, architectural options and reproduction."]
    (args.directory/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
