"""Portable projects, backward compatible with captured session ZIPs."""

import io
import json
import logging
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from contextlib import ExitStack

import numpy as np
from PIL import Image

from shared.settings import ScanSettings

logger = logging.getLogger(__name__)


def _member(archive, name, limit):
    if not isinstance(name, str) or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
        raise ValueError("Invalid project member path")
    info = archive.getinfo(name)
    if info.file_size > limit or info.flag_bits & 1:
        raise ValueError(f"Project member exceeds its size limit or is encrypted: {name}")
    return archive.read(info)


def export_session(engine, path):
    started = time.monotonic()
    manifest = {
        "version": 3,
        "kind": "kinect-scanner-project",
        "depth_unit": "raw_11bit_disparity" if engine.settings.sensor_calibration else "millimetres",
        "depth_encoding": engine.settings.depth_encoding,
        "settings": engine.settings.to_dict(),
        "frames": [],
        "reconstruction": "reconstruction.json",
    }
    source_path = getattr(engine, "_project_archive_path", None)
    if source_path and Path(path).resolve() == Path(source_path).resolve():
        raise ValueError("Save to a temporary path before replacing the opened project")
    with ExitStack() as stack:
        source = stack.enter_context(zipfile.ZipFile(source_path)) if source_path else None
        previous = json.loads(_member(source, "manifest.json", 16 * 1024**2)) if source else {}
        archive = stack.enter_context(zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED))
        for index, ((rgb, depth), metadata) in enumerate(zip(engine.raw_frames, engine.frame_metadata)):
            paths = {"rgb": f"rgb/{index:06d}.png", "depth": f"depth/{index:06d}.png"}
            rotation = metadata.get("orientation", {}).get("rotation_cw_degrees", 0)
            for name, array in (("rgb", rgb), ("depth", depth)):
                if source and index < len(previous["frames"]):
                    # Raw observations form an immutable prefix after opening.
                    # Reuse their lossless PNGs instead of encoding them again.
                    original = previous["frames"][index]
                    for key in (name, "display_" + name):
                        if key not in original:
                            continue
                        destination = paths[name] if key == name else "display/" + paths[name]
                        paths[key] = destination
                        with source.open(original[key]) as data, archive.open(destination, "w", force_zip64=True) as output:
                            shutil.copyfileobj(data, output, length=1024 * 1024)
                    continue
                buffer = io.BytesIO()
                Image.fromarray(array).save(buffer, format="PNG")
                archive.writestr(paths[name], buffer.getvalue())
                if rotation in (90, 180, 270):
                    transpose = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
                                 270: Image.Transpose.ROTATE_90}[rotation]
                    display = io.BytesIO()
                    Image.fromarray(array).transpose(transpose).save(display, format="PNG")
                    paths["display_" + name] = "display/" + paths[name]
                    archive.writestr(paths["display_" + name], display.getvalue())
            manifest["frames"].append({**paths, "timestamp_s": metadata.get("timestamp_s", index),
                                       "metadata": {**metadata, "server_index": index}})
        # Retain independent sensor observations when an opened project is saved again.
        if source:
            if "sensor_archive" in previous:
                manifest["sensor_archive"] = previous["sensor_archive"]
            for item in source.infolist():
                if item.filename.startswith("sensors/"):
                    with source.open(item) as data, archive.open(item, "w", force_zip64=True) as output:
                        shutil.copyfileobj(data, output, length=1024 * 1024)
        if getattr(engine, "mesh", None) is not None:
            import open3d as o3d

            with tempfile.TemporaryDirectory() as directory:
                mesh_path = Path(directory) / "mesh.ply"
                if not o3d.io.write_triangle_mesh(str(mesh_path), engine.mesh):
                    raise OSError("Could not save project mesh")
                manifest["mesh"] = "mesh.ply"
                archive.write(mesh_path, "mesh.ply")
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
        archive.writestr("reconstruction.json", json.dumps(engine.reconstruction_report(), indent=2, allow_nan=False))
    logger.info("Project encoding: %d captures, %.1f MiB, %.2f s", len(manifest["frames"]),
                Path(path).stat().st_size / 1024**2, time.monotonic() - started)


def load_session(engine, path):
    """Populate a fresh candidate; callers commit it only after complete success.

    Restore the saved poses by fresh fusion, never replaying registration or
    changing their positions. Rebuild transient tracking caches for continuation.
    """
    from .fragments import _rigid

    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 100000 or sum(i.file_size for i in archive.infolist()) > 8 * 1024**3:
            raise ValueError("Project archive exceeds the supported size")
        manifest = json.loads(_member(archive, "manifest.json", 16 * 1024**2))
        if not isinstance(manifest, dict):
            raise ValueError("Project manifest must be an object")
        if manifest.get("version") not in (1, 2, 3):
            raise ValueError("Unsupported project version")
        settings = ScanSettings.from_dict(manifest["settings"])
        captures = manifest["frames"]
        if not isinstance(captures, list) or len(captures) > engine.MAX_FRAMES:
            raise ValueError(f"Project must contain at most {engine.MAX_FRAMES} captures")
        engine.reset(settings=settings)
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
            report = json.loads(_member(archive, manifest["reconstruction"], 32 * 1024**2))
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
    return manifest
