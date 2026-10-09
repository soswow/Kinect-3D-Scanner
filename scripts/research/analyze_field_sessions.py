"""Inspect field ZIP metadata without importing scanner, Open3D, CUDA or NumPy.

The original archives remain untouched. No image pixels or archived pose arrays
are exported. Camera poses are read only for diagnostic motion magnitudes; they
never seed or authorize tracking. ZIP inventories use CRC/size plus exact JSON
hashes by default; --hash-archives optionally adds a complete archive SHA256.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[2]
FINISH_STAGES = (
    "fragment_reconnection", "final_refinement", "bundle_adjustment",
    "final_reintegration",
)
INTERPRETATION_SOURCES = (
    "scanner_server/engine.py", "scanner_server/fragments.py",
    "shared/capture.py", "shared/sensor_recording.py",
    "shared/visual_tracking.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def file_digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def distribution(values):
    values = sorted(float(x) for x in values if finite(x))
    if not values:
        return {"count": 0}

    def quantile(fraction):
        position = fraction * (len(values) - 1)
        lo = math.floor(position)
        hi = math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return {"count": len(values), "min": values[0],
            "median": statistics.median(values), "mean": statistics.mean(values),
            "p95": quantile(.95), "max": values[-1]}


def counts(values):
    return dict(sorted(collections.Counter(str(x) for x in values).items()))


def interval_stats(values):
    valid = [x for x in values if finite(x)]
    gaps = [b - a for a, b in zip(valid, valid[1:])]
    span = valid[-1] - valid[0] if len(valid) > 1 else None
    return {"timestamp_count": len(valid), "span_s": span,
            "interval_s": distribution(gaps),
            "nonpositive_intervals": sum(x <= 0 for x in gaps),
            "selected_rate_hz": ((len(valid) - 1) / span
                                 if span is not None and span > 0 else None)}


def prior_success(row):
    if row.get("recovered_offline"):
        return False
    if row.get("excluded_offline"):
        return True
    return row.get("success") is True


def failure_category(message):
    message = message or "Missing diagnostic message"
    prefixes = (
        ("Pose jump:", "pose_jump"), ("Weak alignment:", "weak_alignment"),
        ("Tracking lost", "tracking_lost"),
        ("Unverified tracking transition", "unverified_transition"),
        ("Excluded:", "excluded_without_connection"),
        ("Frame failed:", "frame_exception"),
    )
    return next((name for prefix, name in prefixes if message.startswith(prefix)),
                message)


def motion(a, b):
    """Diagnostic relative translation/angle; no pose is returned or consumed."""
    for matrix in (a, b):
        if (not isinstance(matrix, list) or len(matrix) != 4
                or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
                or not all(finite(x) for row in matrix for x in row)):
            return None
        if any(abs(matrix[3][i] - (i == 3)) > 1e-6 for i in range(4)):
            return None
        if max(abs(sum(matrix[k][i] * matrix[k][j] for k in range(3))
                   - (i == j)) for i in range(3) for j in range(3)) > 1e-3:
            return None
    translation = math.sqrt(sum((a[i][3] - b[i][3]) ** 2 for i in range(3)))
    trace = sum(a[i][j] * b[i][j] for i in range(3) for j in range(3))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1) / 2))))
    return translation, angle


def path_extents(poses):
    translations = [row.get("camera_to_world") for row in poses or []]
    valid = [matrix for matrix in translations if motion(matrix, matrix) is not None]
    if not valid:
        return {"valid_pose_count": 0}
    bounds = [[min(matrix[i][3] for matrix in valid),
               max(matrix[i][3] for matrix in valid)] for i in range(3)]
    return {"valid_pose_count": len(valid), "xyz_span_m": [b - a for a, b in bounds],
            "scope": "Camera trajectory extent in the saved coordinate frame; not surface dimensions or ground truth."}


def png_header(archive, name):
    with archive.open(name) as stream:
        data = stream.read(33)
    if len(data) != 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError(f"Expected a PNG header: {name}")
    width, height, bits, color = struct.unpack(">IIBB", data[16:26])
    return {"width": width, "height": height, "bits": bits, "color_type": color}


def sensor_segment(archive, prefix):
    config_name = prefix + "/configuration.json"
    status_name = prefix + "/status.json"
    config = json.loads(archive.read(config_name))
    status = json.loads(archive.read(status_name))
    logs = {}
    for kind in ("rgb", "depth", "accelerometer", "events"):
        name = prefix + "/" + kind + ".jsonl"
        if name not in archive.namelist():
            logs[kind] = {"present": False}
            continue
        info = archive.getinfo(name)
        if info.file_size > 64 * 1024 * 1024:
            raise ValueError(f"Metadata index exceeds the 64 MiB inspection limit: {name}")
        data = archive.read(name)
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        stamps = [x.get("host_monotonic_s") for x in rows]
        logs[kind] = {
            "present": True, "rows": len(rows), "bytes": len(data),
            "sha256": digest(data), "intervals": interval_stats(stamps),
            "status_count_matches": status.get("counts", {}).get(kind) == len(rows),
        }
        if kind == "accelerometer":
            logs[kind].update(
                valid_rows=sum(x.get("valid") is True for x in rows),
                gravity_valid_rows=sum(x.get("gravity", {}).get("valid") is True for x in rows),
                gravity_calibration_verified_rows=sum(
                    x.get("gravity", {}).get("calibration_verified") is True for x in rows),
                gravity_confidence=distribution(
                    [x.get("gravity", {}).get("confidence") for x in rows]),
                reason_counts=counts(x.get("reason", x.get("gravity", {}).get("reason"))
                                     for x in rows),
            )
        if kind == "events":
            logs[kind]["type_counts"] = counts(x.get("type") for x in rows)
            logs[kind]["capability_events"] = [
                {k: x.get(k) for k in ("enabled", "reason", "host_monotonic_s")}
                for x in rows if x.get("type") == "accelerometer_capability"
            ]
    return {
        "segment": prefix.rsplit("/", 1)[-1],
        "configuration_sha256": digest(archive.read(config_name)),
        "status_sha256": digest(archive.read(status_name)),
        "settings_sha256": digest(canonical(config.get("settings"))),
        "record_full_camera_streams": status.get("record_full_camera_streams"),
        "streams": config.get("streams"),
        "clock": config.get("clock"),
        "closed": status.get("closed"), "complete": status.get("complete"),
        "error": status.get("error"), "dropped": status.get("dropped"),
        "logs": logs,
    }


def analyze(path, *, hash_archive=False, inspect_headers=True):
    before = path.stat()
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [x.filename for x in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate ZIP member names: {path}")
        inventory = [{"name": x.filename, "bytes": x.file_size,
                      "compressed_bytes": x.compress_size, "crc32": f"{x.CRC:08x}"}
                     for x in infos]
        manifest_bytes = archive.read("manifest.json")
        manifest = json.loads(manifest_bytes)
        reconstruction_name = manifest.get("reconstruction", "reconstruction.json")
        reconstruction_bytes = archive.read(reconstruction_name)
        reconstruction = json.loads(reconstruction_bytes)
        frames = manifest["frames"]
        rows = reconstruction.get("frames", [])
        settings = manifest["settings"]
        calibration = settings.get("sensor_calibration")
        metadata = [x.get("metadata", {}) for x in frames]
        selected_names = [x[k] for x in frames for k in ("rgb", "depth")]
        missing = [x for x in selected_names if x not in names]
        if missing:
            raise ValueError(f"Missing selected image members: {missing}")
        selected_inventory = [x for x in inventory if x["name"] in set(selected_names)]
        header_counts = {}
        if inspect_headers:
            for kind in ("rgb", "depth"):
                header_counts[kind] = counts(
                    json.dumps(png_header(archive, x[kind]), sort_keys=True) for x in frames)
        visuals = [x.get("visual_tracking", {}) for x in metadata]
        motion_rows = []
        for i, (a, b) in enumerate(zip(visuals, visuals[1:]), 1):
            if not (a.get("valid") and b.get("valid") and a.get("segment")
                    and a.get("segment") == b.get("segment")):
                continue
            result = motion(a.get("camera_to_local"), b.get("camera_to_local"))
            if result is not None:
                motion_rows.append({"index": i, "translation_m": result[0],
                                    "rotation_deg": result[1]})
        live_stage = collections.defaultdict(float)
        frame_summary = []
        for row in rows:
            for key, value in row.get("timings_ms", {}).items():
                if finite(value):
                    live_stage[key] += value
            frame_summary.append({
                "index": row.get("index"), "live_success": prior_success(row),
                "final_success": row.get("success"),
                "recovered_offline": bool(row.get("recovered_offline")),
                "excluded_offline": bool(row.get("excluded_offline")),
                "live_failure_category": (None if prior_success(row) else failure_category(
                    row.get("message_before_reconnection", row.get("message")))),
                "elapsed_ms": row.get("elapsed_ms"), "timings_ms": row.get("timings_ms"),
            })
        stages = reconstruction.get("stage_totals_ms", {})
        finish_ms = sum(stages.get(k, 0) for k in FINISH_STAGES)
        reconnection = reconstruction.get("fragment_reconnection", {})
        fragments = reconnection.get("fragments", [])
        final = reconstruction.get("final_reconstruction", {})
        live_indices = [x["index"] for x in rows if prior_success(x)]
        original_indices = ([x["index"] for x in reconstruction["original_poses"]]
                            if reconstruction.get("original_poses") is not None else None)
        retained_indices = [x["index"] for x in reconstruction.get("poses", [])]
        segment_prefixes = sorted({x.rsplit("/", 1)[0] for x in names
                                   if x.startswith("sensors/") and x.endswith("/status.json")})
        sensors = [sensor_segment(archive, prefix) for prefix in segment_prefixes]
        continuous_members = [x for x in inventory if x["name"].startswith("sensors/")
                              and x["name"].endswith(".npy")]
        timestamps = [x.get("timestamp_s") for x in frames]
        selected_capture = interval_stats(timestamps)
        live_elapsed_ms = sum(x["elapsed_ms"] for x in rows if finite(x.get("elapsed_ms")))
        row = {
            "archive": {"name": path.name, "path": str(path.resolve()),
                        "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                        "member_count": len(inventory),
                        "uncompressed_bytes": sum(x["bytes"] for x in inventory),
                        "inventory_sha256": digest(canonical(inventory)),
                        "selected_image_inventory_sha256": digest(canonical(selected_inventory)),
                        "manifest_sha256": digest(manifest_bytes),
                        "reconstruction_sha256": digest(reconstruction_bytes),
                        "sha256": file_digest(path) if hash_archive else None,
                        "identity_scope": ("complete ZIP SHA256 plus metadata and CRC/size inventory"
                                           if hash_archive else "exact JSON SHA256 plus ZIP CRC/size inventory; image bytes not SHA256 checked")},
            "settings": {k: v for k, v in settings.items() if k != "sensor_calibration"},
            "settings_sha256": digest(canonical(settings)),
            "field_preset_differences": {
                k: {"saved": settings.get(k), "recommended": expected}
                for k, expected in {"voxel_m": .01, "final_voxel_m": .005,
                                    "final_block_count": 10000, "final_weight": 2.0}.items()
                if settings.get(k) != expected
            },
            "manifest_reconstruction_settings_equal": settings == reconstruction.get("settings"),
            "calibration": {"present": calibration is not None,
                            "sha256": digest(canonical(calibration)),
                            "status": (calibration or {}).get("status"),
                            "date": (calibration or {}).get("date"),
                            "runtime_image_space": settings.get("camera", {}).get("image_space"),
                            "depth_encoding": manifest.get("depth_encoding"),
                            "ir_to_rgb_baseline_mm": (calibration or {}).get("ir_to_rgb", {}).get("baseline_mm")},
            "backend": reconstruction.get("backend"),
            "capture": {
                "selected_views": len(frames), **selected_capture,
                "reported_rgb_fps": counts(x.get("rgb_fps") for x in metadata),
                "png_headers": header_counts,
                "rgb_depth_delta_ms": distribution([x.get("rgb_depth_delta_ms") for x in metadata]),
                "rgb_depth_absolute_delta_ms": distribution([
                    abs(x["rgb_depth_delta_ms"]) for x in metadata if finite(x.get("rgb_depth_delta_ms"))]),
                "host_mapping_uncertainty_ms": distribution([
                    1000 * x["host_mapping_uncertainty_s"] for x in metadata
                    if finite(x.get("host_mapping_uncertainty_s"))]),
                "selection_age_ms": distribution([x.get("capture_selection", {}).get("age_ms") for x in metadata]),
                "selection_candidates": distribution([x.get("capture_selection", {}).get("candidates") for x in metadata]),
                "selection_sharpness": distribution([x.get("capture_selection", {}).get("sharpness") for x in metadata]),
                "visual_motion_valid_views": sum(x.get("valid") is True for x in visuals),
                "visual_reason_counts": counts(x.get("reason") for x in visuals),
                "visual_segment_count": len({x.get("segment") for x in visuals if x.get("segment")}),
                "visual_step_ms": distribution([x.get("elapsed_ms") for x in visuals]),
                "selected_motion_translation_m": distribution([x["translation_m"] for x in motion_rows]),
                "selected_motion_rotation_deg": distribution([x["rotation_deg"] for x in motion_rows]),
                "selected_motion_pairs": len(motion_rows),
                "selected_motion_exceeds_live_gate": [x for x in motion_rows
                    if x["translation_m"] > settings.get("max_translation_m", math.inf)
                    or x["rotation_deg"] > settings.get("max_rotation_deg", math.inf)],
                "accelerometer_valid_selected_views": sum(x.get("accelerometer", {}).get("valid") is True for x in metadata),
                "accelerometer_reason_counts": counts(x.get("accelerometer", {}).get("reason") for x in metadata),
                "continuous_rgb_depth_members": len(continuous_members),
                "continuous_rgb_depth_index_rows": sum(
                    s["logs"][kind].get("rows", 0) for s in sensors for kind in ("rgb", "depth")),
                "sensor_archive_manifest": {k: v for k, v in manifest.get("sensor_archive", {}).items()
                                            if k != "segments"},
                "sensor_segments": sensors,
            },
            "tracking": {
                "diagnostic_rows": len(rows),
                "live_success_reconstructed": sum(prior_success(x) for x in rows),
                "original_pose_count": (len(reconstruction["original_poses"])
                                        if reconstruction.get("original_poses") is not None else None),
                "live_accepted_indices": live_indices,
                "original_pose_indices": original_indices,
                "live_success_matches_original_pose_indices": (
                    live_indices == original_indices if original_indices is not None else None),
                "final_success": sum(x.get("success") is True for x in rows),
                "retained_pose_count": len(reconstruction.get("poses", [])),
                "retained_indices": retained_indices,
                "original_camera_path_extent": path_extents(reconstruction.get("original_poses")),
                "retained_camera_path_extent": path_extents(reconstruction.get("poses")),
                "recovered_indices": [x["index"] for x in rows if x.get("recovered_offline")],
                "excluded_live_indices": [x["index"] for x in rows if x.get("excluded_offline")],
                "live_failure_categories": counts(x["live_failure_category"] for x in frame_summary
                                                   if not x["live_success"]),
                "final_failure_categories": counts(failure_category(x.get("message")) for x in rows
                                                    if x.get("success") is not True),
                "final_method_counts": counts(x.get("method", "reference_or_failure") for x in rows),
                "server_frame_elapsed_ms": distribution([x.get("elapsed_ms") for x in rows]),
                "server_frame_elapsed_total_ms": live_elapsed_ms,
                "live_stage_ms_from_frames": dict(live_stage),
                "valid_depth_fraction": distribution([x.get("valid_depth_fraction") for x in rows]),
                "fitness": distribution([x.get("fitness") for x in rows]),
                "rmse_m": distribution([x.get("rmse_m") for x in rows]),
                "state": reconstruction.get("tracking"),
            },
            "finish": {
                "recorded_stage_ms": {k: stages.get(k) for k in FINISH_STAGES},
                "recorded_stage_sum_ms": finish_ms,
                "reconnection_fraction_of_recorded_finish_stages": (
                    stages.get("fragment_reconnection", 0) / finish_ms if finish_ms else None),
                "reconnection": {k: v for k, v in reconnection.items()
                                 if k not in ("fragments", "verified_bridges", "loop_closures")},
                "fragments": [{"id": x.get("id"), "connected": x.get("connected"),
                               "views": len(x.get("frame_indices", [])),
                               "frame_indices": x.get("frame_indices", []),
                               "context_views": len(x.get("context_frame_indices", []))} for x in fragments],
                "verified_bridges": len(reconnection.get("verified_bridges", [])),
                "loop_closures": len(reconnection.get("loop_closures", [])),
                "refinement": reconstruction.get("refinement"),
                "bundle_adjustment": reconstruction.get("bundle_adjustment"),
                "final_reconstruction": final,
                "requested_final_applied": final.get("applied") is True,
                "requested_final_block_limit": settings.get("final_block_count"),
                "required_final_blocks_exact": None,
                "required_final_blocks_lower_bound": (
                    settings.get("final_block_count", 0) + 1
                    if "Final volume exceeds" in final.get("reason", "") else None),
                "required_final_blocks_scope": (
                    "Exact total required 5 mm blocks are not recorded in the ZIP. "
                    "Reconnection fusion_required_blocks describes the 10 mm live volume, "
                    "not final volume requirements; no scaling estimate is substituted."),
                "fusion_failure": reconstruction.get("fusion_failure"),
                "input_failure": reconstruction.get("input_failure"),
                "mesh_geometry_present_in_zip": any(x.endswith((".ply", ".obj", ".glb")) for x in names),
            },
            "stage_totals_ms": stages,
            "diagnostic_frame_rows": frame_summary,
        }
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Archive changed during metadata inspection: {path}")
    row["archive"]["unchanged_during_inspection"] = True
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-archives", action="store_true",
                        help="Read complete ZIPs for SHA256; schedule outside hardware measurements")
    parser.add_argument("--skip-png-headers", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a saved field report; choose a fresh output")
    source_hashes = {name: file_digest(ROOT / name) for name in INTERPRETATION_SOURCES}
    started = time.perf_counter()
    sessions = [analyze(path, hash_archive=args.hash_archives,
                        inspect_headers=not args.skip_png_headers) for path in args.archives]
    source_after = {name: file_digest(ROOT / name) for name in INTERPRETATION_SOURCES}
    if source_hashes != source_after:
        raise RuntimeError("Interpretation source changed during field inspection")
    report = {
        "schema_version": 1, "kind": "offline-field-session-metadata",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script": {"path": str(Path(__file__).resolve()), "sha256": file_digest(Path(__file__))},
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "analysis_elapsed_s": time.perf_counter() - started,
        "interpretation_source_sha256": source_hashes,
        "interpretation_sources_unchanged": True,
        "capturing_server_source_sha256": None,
        "scope": [
            "Saved field observations, not a matched CPU/CUDA benchmark or independent ground truth.",
            "Selected capture cadence is not camera FPS; continuous frames require recorded sensor image streams.",
            "No image pixels or pose arrays are copied, and no archived poses seed or authorize live tracking.",
            "Live success is reconstructed using recovered_offline/excluded_offline flags because Finish mutates diagnostics.",
            "Engine live stage timers are exclusive; Finish stage totals are sequential recorded scopes, not total Finish wall time.",
            "Reported frame elapsed and stage sums can differ slightly because their clocks and synchronization scopes differ; no utilization/FPS is inferred.",
            "Failed final reintegration/extraction and uninstrumented mesh/export time can be absent from recorded Finish totals.",
            "Final applied status establishes saved server completion, not independent geometric correctness or full surface coverage.",
            "Current source hashes document interpretation only; the ZIP does not bind its capturing server to a core hash.",
        ],
        "sessions": sessions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for session in sessions:
        print(json.dumps({
            "archive": session["archive"]["name"],
            "views": session["capture"]["selected_views"],
            "live_accepted": session["tracking"]["live_success_reconstructed"],
            "retained": session["tracking"]["retained_pose_count"],
            "recorded_finish_stages_s": session["finish"]["recorded_stage_sum_ms"] / 1000,
            "final_applied": session["finish"]["requested_final_applied"],
        }))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
