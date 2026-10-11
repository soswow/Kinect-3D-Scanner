"""Portable projects, backward compatible with captured session ZIPs."""

import copy
import hashlib
import io
import json
import logging
import tempfile
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from shared.project_archive import (
    JPEG_QUALITY,
    copy_member,
    encode_image,
    pixel_digest,
    write_member,
)
from shared.settings import ScanSettings

logger = logging.getLogger(__name__)
PROJECT_ARCHIVE_LIMIT_BYTES = 64 * 1024**3
PROJECT_JSON_LIMIT_BYTES = 256 * 1024**2


def _member(archive, name, limit):
    if not isinstance(name, str) or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
        raise ValueError("Invalid project member path")
    info = archive.getinfo(name)
    if info.file_size > limit or info.flag_bits & 1:
        raise ValueError(f"Project member exceeds its size limit or is encrypted: {name}")
    return archive.read(info)


def _reuse_proofs(engine, reuse):
    if reuse is None:
        return {}
    if not isinstance(reuse, dict) or reuse.get("session_id") != engine.session_id:
        raise ValueError("Local capture reuse requires the current server session")
    rows = reuse.get("captures")
    if not isinstance(rows, list) or len(rows) > engine.MAX_FRAMES:
        raise ValueError("Invalid local capture reuse list")
    proofs = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid local capture proof")
        index = row.get("index")
        if type(index) is not int or index < 0 or index in proofs:
            raise ValueError("Invalid local capture index")
        for key in ("rgb_sha256", "depth_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid local capture pixel hash")
        proofs[index] = row
    return proofs


def compact_project_info(engine):
    """Describe a final-product copy without altering captures or reconstruction."""
    indices = sorted(i for i, _ in engine.poses)
    available = (engine.mesh is not None and bool(indices) and not engine.unprocessed_count
                 and not engine.fusion_failure and not engine.input_failure
                 and len(engine.diagnostics) == engine.stored_count
                 and set(indices) == {d["index"] for d in engine.diagnostics if d["success"]}
                 and len(indices) == len(set(indices))
                 and all(type(i) is int and 0 <= i < engine.stored_count for i in indices))
    descriptor = {
        "session_id": engine.session_id, "stored_count": engine.stored_count,
        "poses": [{"index": i, "pose": p.tolist()} for i, p in engine.poses],
        "settings": engine.settings.to_dict(),
        "mesh_vertices": len(engine.mesh.vertices) if engine.mesh is not None else 0,
        "mesh_triangles": len(engine.mesh.triangles) if engine.mesh is not None else 0,
    }
    revision = hashlib.sha256(json.dumps(descriptor, sort_keys=True, allow_nan=False,
                                         separators=(",", ":")).encode()).hexdigest()
    display_copies = 2 * sum(m.get("orientation", {}).get("rotation_cw_degrees", 0) in (90, 180, 270)
                             for m in engine.frame_metadata)
    return {"available": available, "session_id": engine.session_id, "revision": revision,
            "stored_count": engine.stored_count, "retained_count": len(indices),
            "excluded_count": engine.stored_count - len(indices),
            "retained_indices": indices, "display_copies_omitted": display_copies,
            "reason": "Ready" if available else "Finish the scan successfully before saving a compact project"}


def _compact_report(report, indices):
    """Preserve the final poses, with current indices separate from source history."""
    mapping = {old: new for new, old in enumerate(indices)}
    result = {key: copy.deepcopy(report[key]) for key in (
        "version", "session_id", "pose_convention", "length_unit", "settings", "backend",
        "fusion_failure", "input_failure", "stage_totals_ms", "final_reconstruction",
    ) if key in report}
    result["poses"] = [{**p, "index": mapping[p["index"]]} for p in report["poses"] if p["index"] in mapping]
    result["original_poses"] = None if report.get("original_poses") is None else [
        {**p, "index": mapping[p["index"]]} for p in report["original_poses"] if p["index"] in mapping]
    by_index = {d["index"]: d for d in report["frames"]}
    result["frames"] = [{"index": new, "source_index": old, "success": True,
                         "pose": next(p["camera_to_world"] for p in result["poses"] if p["index"] == new),
                         "method": "saved_pose", "message": "Preserved final-model pose",
                         "metadata": {**by_index[old].get("metadata", {}), "server_index": new},
                         "frame_count": new + 1, "last_tracked_index": new, "fusion_paused": False}
                        for new, old in enumerate(indices)]
    # Historical graph endpoints belong to the source index space. Retain them
    # in the explicit source report, never as active graph/refinement inputs.
    for key in ("refinement", "bundle_adjustment"):
        result[key] = {"applied": False, "reason": "Final poses preserved; source history is in provenance/source-reconstruction.json"}
    result["fragment_reconnection"] = {"applied": False, "algorithm": "preserved_final_poses",
                                       "reason": "Selected final poses retained without registration",
                                       "source_report": "provenance/source-reconstruction.json"}
    result["tracking_edges"] = []
    result["tracking"] = {"state": "tracking", "last_tracked_index": len(indices) - 1, "lost_at_index": None}
    return result


def export_session(engine, path, rgb_format="png", reuse=None, compact=False, compact_revision=None):
    started = time.monotonic()
    if rgb_format not in ("png", "jpeg"):
        raise ValueError("Project RGB format must be png or jpeg")
    if type(compact) is not bool:
        raise ValueError("Compact project option must be boolean")
    plan = compact_project_info(engine) if compact else None
    if compact and (not plan["available"] or compact_revision != plan["revision"]):
        raise ValueError("Compact project review is unavailable or stale; review the finished scan again")
    if compact and reuse is not None:
        raise ValueError("Compact projects require complete server captures")
    indices = plan["retained_indices"] if compact else list(range(len(engine.raw_frames)))
    proofs = _reuse_proofs(engine, reuse)
    reused = []
    manifest = {
        "version": 4 if rgb_format == "jpeg" else 3,
        "kind": "kinect-scanner-project",
        "depth_unit": "raw_11bit_disparity" if engine.settings.sensor_calibration else "millimetres",
        "depth_encoding": engine.settings.depth_encoding,
        "settings": engine.settings.to_dict(),
        "frames": [],
        "reconstruction": "reconstruction.json",
        "rgb_format": rgb_format,
    }
    if rgb_format == "jpeg":
        manifest["jpeg_quality"] = JPEG_QUALITY
    if compact:
        manifest["project_compaction"] = {"version": 1, "source_session_id": engine.session_id,
                                           "source_capture_count": engine.stored_count,
                                           "retained_source_indices": indices,
                                           "source_report": "provenance/source-reconstruction.json",
                                           "policy": "final_model_poses", "display_copies": False}
    source_path = getattr(engine, "_project_archive_path", None)
    if source_path and Path(path).resolve() == Path(source_path).resolve():
        raise ValueError("Save to a temporary path before replacing the opened project")
    with ExitStack() as stack:
        source = stack.enter_context(zipfile.ZipFile(source_path)) if source_path else None
        previous = json.loads(_member(source, "manifest.json", PROJECT_JSON_LIMIT_BYTES)) if source else {}
        archive = stack.enter_context(zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6))
        for index, source_index in enumerate(indices):
            rgb, depth = engine.raw_frames[source_index]
            metadata = engine.frame_metadata[source_index]
            extension = "jpg" if rgb_format == "jpeg" else "png"
            paths = {"rgb": f"rgb/{index:06d}.{extension}", "depth": f"depth/{index:06d}.png"}
            rotation = 0 if compact else metadata.get("orientation", {}).get("rotation_cw_degrees", 0)
            proof = proofs.get(index)
            use_local = proof is not None and all(pixel_digest(array) == proof[name + "_sha256"]
                                                  for name, array in (("rgb", rgb), ("depth", depth)))
            if use_local:
                reused.append(proof)
            for name, array in (("rgb", rgb), ("depth", depth)):
                if use_local:
                    if rotation in (90, 180, 270):
                        paths["display_" + name] = "display/" + paths[name]
                    continue
                if (source and source_index < len(previous["frames"])
                        and (name == "depth" or previous.get("rgb_format", "png") == rgb_format)):
                    # Raw observations form an immutable prefix after opening.
                    # Preserve matching encodings, including JPEGs, without generational loss.
                    original = previous["frames"][source_index]
                    for key in ((name,) if compact else (name, "display_" + name)):
                        if key not in original:
                            continue
                        destination = paths[name] if key == name else "display/" + paths[name]
                        paths[key] = destination
                        copy_member(source, archive, original[key], destination)
                    continue
                image_format = rgb_format if name == "rgb" else "png"
                write_member(archive, paths[name], encode_image(array, image_format))
                if rotation in (90, 180, 270):
                    paths["display_" + name] = "display/" + paths[name]
                    write_member(archive, paths["display_" + name], encode_image(np.rot90(array, k=-(rotation // 90)), image_format))
            saved_metadata = {**metadata, "server_index": index}
            if compact:
                saved_metadata.setdefault("source_server_index", source_index)
                saved_metadata.setdefault("source_capture", {"session_id": engine.session_id, "index": source_index})
            manifest["frames"].append({**paths, "timestamp_s": metadata.get("timestamp_s", source_index),
                                       "metadata": saved_metadata})
        # Retain independent sensor observations when an opened project is saved again.
        if source:
            if "sensor_archive" in previous:
                manifest["sensor_archive"] = previous["sensor_archive"]
            for item in source.infolist():
                if item.filename.startswith("sensors/"):
                    copy_member(source, archive, item)
                elif item.filename.startswith("provenance/"):
                    destination = item.filename.replace("provenance/", "provenance/previous/", 1) if compact else item.filename
                    copy_member(source, archive, item, destination)
            if not compact and "project_compaction" in previous:
                manifest["project_compaction"] = previous["project_compaction"]
            elif compact and "project_compaction" in previous:
                manifest["project_compaction"]["previous_provenance_prefix"] = "provenance/previous/"
                write_member(archive, "provenance/previous/compaction.json",
                             json.dumps(previous["project_compaction"], separators=(",", ":"), allow_nan=False))
        if getattr(engine, "mesh", None) is not None:
            import open3d as o3d

            with tempfile.TemporaryDirectory() as directory:
                mesh_path = Path(directory) / "mesh.ply"
                if not o3d.io.write_triangle_mesh(str(mesh_path), engine.mesh):
                    raise OSError("Could not save project mesh")
                manifest["mesh"] = "mesh.ply"
                archive.write(mesh_path, "mesh.ply")
        journal = getattr(engine,"motion_journal",{})
        if journal.get("segments"):
            manifest["motion_journal"] = "motion.json"
            write_member(archive, "motion.json",json.dumps(journal,allow_nan=False,separators=(",",":")))
        if reused:
            manifest["local_captures"] = {"session_id": engine.session_id, "captures": reused}
        write_member(archive, "manifest.json", json.dumps(manifest, separators=(",", ":"), allow_nan=False))
        report = engine.reconstruction_report()
        if compact:
            write_member(archive, "provenance/source-reconstruction.json", json.dumps(report, separators=(",", ":"), allow_nan=False))
            report = _compact_report(report, indices)
        write_member(archive, "reconstruction.json", json.dumps(report, separators=(",", ":"), allow_nan=False))
    logger.info("Project encoding: %d captures, %.1f MiB, %.2f s", len(manifest["frames"]),
                Path(path).stat().st_size / 1024**2, time.monotonic() - started)


def load_session(engine, path):
    """Populate a fresh candidate; callers commit it only after complete success.

    Restore the saved poses by fresh fusion, never replaying registration or
    changing their positions. Rebuild transient tracking caches for continuation.
    """
    from .fragments import _rigid

    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 100000 or sum(i.file_size for i in archive.infolist()) > PROJECT_ARCHIVE_LIMIT_BYTES:
            raise ValueError("Project archive exceeds the supported size")
        manifest = json.loads(_member(archive, "manifest.json", PROJECT_JSON_LIMIT_BYTES))
        if not isinstance(manifest, dict):
            raise ValueError("Project manifest must be an object")
        if manifest.get("version") not in (1, 2, 3, 4):
            raise ValueError("Unsupported project version")
        if "local_captures" in manifest:
            raise ValueError("Incomplete project transfer; local captures have not been assembled")
        settings = ScanSettings.from_dict(manifest["settings"])
        captures = manifest["frames"]
        if not isinstance(captures, list) or len(captures) > engine.MAX_FRAMES:
            raise ValueError(f"Project must contain at most {engine.MAX_FRAMES} captures")
        compaction = manifest.get("project_compaction")
        if compaction is not None:
            retained = compaction.get("retained_source_indices") if isinstance(compaction, dict) else None
            source_count = compaction.get("source_capture_count") if isinstance(compaction, dict) else None
            if (not isinstance(retained, list) or len(retained) > len(captures)
                    or type(source_count) is not int or source_count < len(retained)
                    or any(type(i) is not int or not 0 <= i < source_count for i in retained)
                    or retained != sorted(set(retained)) or compaction.get("version") != 1
                    or compaction.get("policy") != "final_model_poses"):
                raise ValueError("Invalid compact project provenance")
        engine.reset(settings=settings)
        if manifest.get("motion_journal"):
            from shared.motion_journal import MAX_JOURNAL_BYTES
            engine.store_motion_journal(json.loads(_member(archive,manifest["motion_journal"],MAX_JOURNAL_BYTES)))
        for index, frame in enumerate(captures):
            if not isinstance(frame, dict):
                raise ValueError("Invalid project capture")
            with Image.open(io.BytesIO(_member(archive, frame["rgb"], 16 * 1024**2))) as image:
                c = settings.rgb_camera
                if image.size != (c.width, c.height):
                    raise ValueError("Project RGB dimensions do not match calibration")
                rgb = np.array(image.convert("RGB"), dtype=np.uint8)
            with Image.open(io.BytesIO(_member(archive, frame["depth"], 2 * 1024**2))) as image:
                if image.size != (640, 480) or image.mode not in ("I", "I;16", "I;16B", "I;16L"):
                    raise ValueError("Project depth must be a 16-bit 640x480 image")
                raw = np.array(image)
                if np.any(raw < 0) or np.any(raw > 65535):
                    raise ValueError("Invalid project depth values")
                depth = raw.astype(np.uint16)
            metadata = dict(frame.get("metadata", {}))
            metadata["timestamp_s"] = frame.get("timestamp_s", index)
            result = engine.store_frame(rgb, depth, metadata)
            if not result["success"]:
                raise ValueError(f"Cannot load capture {index + 1}: {result['message']}")
        report = {}
        if manifest.get("reconstruction"):
            report = json.loads(_member(archive, manifest["reconstruction"], PROJECT_JSON_LIMIT_BYTES))
        if not isinstance(report, dict):
            raise ValueError("Project reconstruction must be an object")
        poses = []
        seen = set()
        diagnostics = report.get("frames", [])
        if not isinstance(diagnostics, list) or len(diagnostics) > len(captures):
            raise ValueError("Invalid reconstruction frame count")
        for index, diagnostic in enumerate(diagnostics):
            if not isinstance(diagnostic, dict) or diagnostic.get("index") != index or type(diagnostic.get("success")) is not bool:
                raise ValueError("Invalid reconstruction frame order")
        for item in report.get("poses", []):
            if not isinstance(item, dict):
                raise ValueError("Invalid saved camera pose")
            index, pose = item["index"], np.asarray(item["camera_to_world"], float)
            if type(index) is not int or not 0 <= index < len(diagnostics) or index in seen or not _rigid(pose):
                raise ValueError("Invalid saved camera pose")
            seen.add(index)
            poses.append((index, pose))
        if seen != {i for i, d in enumerate(diagnostics) if d["success"]}:
            raise ValueError("Saved poses disagree with reconstruction frames")
        if poses and report.get("pose_convention") != "camera_to_world":
            raise ValueError("Unsupported saved pose convention")
        engine.poses = sorted(poses, key=lambda item: item[0])
        # Fuse into a candidate volume before replacing the active project.
        for index, pose in engine.poses:
            rgb, depth = engine._prepare_input(*engine.raw_frames[index], settings)
            engine._integrate_vbg(rgb, depth, np.linalg.inv(pose))
        engine.frame_count = len(poses)
        engine._processed_count = len(diagnostics)
        engine.diagnostics = [{**d, "session_id": engine.session_id} for d in diagnostics]
        if poses:
            engine._extract_model_pcd()
            last_index, last_pose = engine.poses[-1]
            engine.cumulative_T = last_pose.copy()
            engine._last_rgbd = engine._make_rgbd(*engine._prepare_input(*engine.raw_frames[last_index], settings))
            engine._tracking_lost_frames = len(diagnostics) - last_index - 1
            engine._lost_at_index = last_index + 1 if engine._tracking_lost_frames else None
        for field in ("refinement", "bundle_adjustment", "fragment_reconnection", "final_reconstruction"):
            if isinstance(report.get(field), dict):
                setattr(engine, field, report[field])
        engine.tracking_edges = report.get("tracking_edges", [])
        if isinstance(report.get("stage_totals_ms"), dict):
            engine.stage_totals_ms = report["stage_totals_ms"]
        if report.get("original_poses") is not None:
            originals = []
            for item in report["original_poses"]:
                index, pose = item["index"], np.asarray(item["camera_to_world"], float)
                if type(index) is not int or not 0 <= index < len(captures) or not _rigid(pose):
                    raise ValueError("Invalid original saved camera pose")
                originals.append((index, pose))
            engine.original_poses = originals
        for count_field, report_field in (("_refined_count", "refinement"), ("_bundle_count", "bundle_adjustment"),
                                         ("_reconnection_count", "fragment_reconnection")):
            if report.get(report_field, {}).get("applied"):
                setattr(engine, count_field, len(captures))
        if manifest.get("mesh"):
            import open3d as o3d

            with tempfile.TemporaryDirectory() as directory:
                mesh_path = Path(directory) / "mesh.ply"
                mesh_path.write_bytes(_member(archive, manifest["mesh"], 256 * 1024**2))
                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if not len(mesh.vertices) or not len(mesh.triangles) or not np.isfinite(np.asarray(mesh.vertices)).all():
                raise ValueError("Invalid saved project mesh")
            mesh.compute_vertex_normals()
            engine.mesh = mesh
        if compaction is not None and len(compaction["retained_source_indices"]) == len(captures):
            if len(poses) != len(captures) or engine.mesh is None:
                raise ValueError("Compact final project requires a saved mesh and all retained poses")
            # This is a loaded final product, not a new registration proof. Its
            # validated saved poses were just restored by fresh fusion. Repeat
            # Finish preserves them; new captures invalidate these counts normally.
            engine._preserved_project_pose_count = engine._refined_count = engine._bundle_count = len(captures)
    return manifest
