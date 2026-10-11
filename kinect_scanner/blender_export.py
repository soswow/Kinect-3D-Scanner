"""Prepare a saved RGB-D session and run a separate Blender scene writer.

The UI supplies an already textured OBJ ZIP. The standalone CLI can bake one
from the saved mesh without rerunning registration or contacting a server.
"""

import io
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile

logger = logging.getLogger(__name__)


def find_blender(path=None):
    """Resolve an executable or macOS .app; explicit selections never fall back."""
    def executable(candidate):
        candidate = Path(candidate).expanduser()
        if candidate.suffix.lower() == ".app":
            candidate /= "Contents/MacOS/Blender"
        return str(candidate.resolve()) if candidate.is_file() else None

    if path:
        resolved = executable(path) or shutil.which(str(path))
        if not resolved:
            raise ValueError("Blender was not found at the selected path. Choose its application or executable.")
        return resolved
    selected = os.environ.get("BLENDER_EXECUTABLE")
    if selected:
        return find_blender(selected)
    found = shutil.which("blender")
    if found:
        return found
    candidates = []
    if sys.platform == "darwin":
        for root in (Path("/Applications"), Path.home() / "Applications"):
            candidates.extend(sorted(root.glob("Blender*.app"), reverse=True))
    elif sys.platform == "win32":
        for variable in ("ProgramFiles", "ProgramFiles(x86)"):
            root = Path(os.environ.get(variable, "C:/Program Files"))
            candidates.extend(sorted((root / "Blender Foundation").glob("Blender */blender.exe"), reverse=True))
    for candidate in candidates:
        found = executable(candidate)
        if found:
            return found
    return None


def _read(archive, name, limit=32 * 1024**2):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Invalid session member path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Invalid session member path")
    info = archive.getinfo(name)
    if info.file_size > limit or info.flag_bits & 1:
        raise ValueError(f"Unsupported or oversized session member: {name}")
    return archive.read(name)


class SessionFrames:
    """Load observations on demand, retaining at most the caller's current frame."""
    def __init__(self, archive, manifest, settings):
        self.archive, self.frames, self.settings = archive, manifest["frames"], settings

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image

        frame = self.frames[index]
        arrays = []
        for key in ("rgb", "depth"):
            with Image.open(io.BytesIO(_read(self.archive, frame[key], 16 * 1024**2))) as image:
                camera = self.settings.rgb_camera if key == "rgb" else self.settings.camera
                if image.size != (camera.width, camera.height):
                    raise ValueError(f"Frame {index} {key} dimensions disagree with calibration")
                array = np.array(image)
            if key == "rgb" and (array.dtype != np.uint8 or array.shape != (camera.height, camera.width, 3)):
                raise ValueError("Session RGB must be an RGB uint8 image")
            if key == "depth":
                if (array.shape != (camera.height, camera.width) or array.dtype.kind not in "ui"
                        or np.any(array < 0) or np.any(array > (2047 if self.settings.sensor_calibration else 65535))):
                    raise ValueError("Session depth must be a uint16 image")
                array = array.astype(np.uint16)
            arrays.append(array)
        return tuple(arrays)


def snapshot_geometry(depth, settings, stride=2):
    """Triangulate one measured depth grid; never connect across holes or jumps."""
    import cv2
    import numpy as np
    from shared.calibration import pinhole_rays, prepare_metric_depth

    if stride not in (1, 2, 4, 8):
        raise ValueError("Snapshot sampling must be 1, 2, 4, or 8 pixels")
    metric = prepare_metric_depth(depth, settings).astype(np.float64) / 1000
    c = settings.camera
    sampled = metric[::stride, ::stride]
    points = (pinhole_rays(c)[::stride, ::stride] * sampled[..., None]).reshape(-1, 3)
    h, w = sampled.shape
    grid = np.arange(h * w).reshape(h, w)
    a, b, d, e = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    # All pixels in the sampled cell must support an edge. Sampling must not
    # skip an invalid pixel or connect foreground to background.
    kernel = np.ones((stride + 1, stride + 1), np.uint8)
    low = cv2.erode(metric, kernel, anchor=(0, 0))[:-stride:stride, :-stride:stride]
    high = cv2.dilate(metric, kernel, anchor=(0, 0))[:-stride:stride, :-stride:stride]
    valid = (low > 0) & (high - low <= np.maximum(0.03, low * 0.02))
    faces = np.concatenate((np.stack((a, d, b), axis=-1)[valid], np.stack((b, d, e), axis=-1)[valid]))
    used, remap = np.unique(faces, return_inverse=True)
    return points[used], remap.reshape(-1, 3).astype(np.int32)


def _snapshot(rgb, depth, settings, stride):
    import cv2
    import numpy as np
    from shared.calibration import camera_matrix, rectification_maps

    vertices, faces = snapshot_geometry(depth, settings, stride)
    camera = settings.rgb_camera
    calibration = settings.sensor_calibration
    extrinsic = np.eye(4)
    if calibration:
        extrinsic[:3, :3] = calibration.rotation
        extrinsic[:3, 3] = np.asarray(calibration.translation_mm) / 1000
    points = vertices @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    pixels = points @ camera_matrix(camera).T
    np.divide(pixels[:, :2], pixels[:, 2:3], out=pixels[:, :2], where=pixels[:, 2:3] != 0)
    visible = ((points[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width - 1)
               & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height - 1))
    px = np.clip(np.rint(pixels[:, 0]), 0, camera.width - 1).astype(int)
    py = np.clip(np.rint(pixels[:, 1]), 0, camera.height - 1).astype(int)
    nearest = np.full((camera.height, camera.width), np.inf)
    np.minimum.at(nearest, (py[visible], px[visible]), points[visible, 2])
    visible &= points[:, 2] <= nearest[py, px] + np.maximum(0.015, points[:, 2] * 0.01)
    uv = np.column_stack(((pixels[:, 0] + 0.5) / camera.width, 1 - (pixels[:, 1] + 0.5) / camera.height))
    if any(camera.distortion):
        maps = rectification_maps(camera)
        rgb = cv2.remap(rgb, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rgb, vertices, faces, uv, visible[faces].all(axis=1), extrinsic


def _scene_script():
    # Blender needs plain source, including in a frozen client where modules
    # otherwise live inside PyInstaller's PYZ archive.
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return root / "kinect_scanner/blender_scene.py"


def convert_session(session, output, *, blender=None, textured_model=None, stride=2,
                    texture_options=None, progress=None):
    """Write .blend plus its relative RGB sequence, preserving prior outputs on failure.

    Runs only a fresh, background Blender process. Does not touch an open scene,
    a scanner server, capture hardware, or the source archive.
    """
    import numpy as np
    from PIL import Image
    from shared.settings import ScanSettings

    notify = progress or (lambda message: logger.info("%s", message))
    executable = find_blender(blender)
    if not executable:
        raise ValueError("Install Blender 4.2 or newer, then choose its application or executable.")
    output = Path(output).resolve()
    if output.suffix.lower() != ".blend":
        raise ValueError("Choose an output filename ending in .blend")
    if output == Path(session).resolve() or (textured_model and output == Path(textured_model).resolve()):
        raise ValueError("The output must not replace an input archive")
    if stride not in (1, 2, 4, 8):
        raise ValueError("Snapshot sampling must be 1, 2, 4, or 8 pixels")
    output.parent.mkdir(parents=True, exist_ok=True)
    # A distinct relative directory keeps a previously exported scene and its
    # sequence usable until the replacement .blend is completely written.
    assets = output.parent / f"{output.stem}.images-{uuid.uuid4().hex[:8]}"
    committed = False
    try:
        with tempfile.TemporaryDirectory(prefix=".scanner-blend-", dir=output.parent) as temporary, zipfile.ZipFile(session) as archive:
            work = Path(temporary)
            infos = archive.infolist()
            if len(infos) > 100000 or len({i.filename for i in infos}) != len(infos) or sum(i.file_size for i in infos) > 64 * 1024**3:
                raise ValueError("Unsupported session archive size or duplicate members")
            manifest = json.loads(_read(archive, "manifest.json"))
            report = json.loads(_read(archive, manifest.get("reconstruction", "reconstruction.json")))
            if report.get("pose_convention") != "camera_to_world" or report.get("length_unit") != "metres":
                raise ValueError("The session needs accepted camera-to-world poses in metres")
            settings = ScanSettings.from_dict(manifest["settings"])
            expected = "raw_11bit_disparity" if settings.sensor_calibration else "millimetres"
            if manifest.get("depth_unit") != expected:
                raise ValueError("Session depth units disagree with its calibration")
            frames = SessionFrames(archive, manifest, settings)
            poses = []
            indices = set()
            for item in sorted(report["poses"], key=lambda p: p["index"]):
                index, pose = item["index"], np.asarray(item["camera_to_world"], float)
                if type(index) is not int or not 0 <= index < len(frames) or index in indices:
                    raise ValueError("Invalid or duplicate accepted frame index")
                if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
                        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
                    raise ValueError("Session contains a non-rigid camera pose")
                indices.add(index)
                poses.append((index, pose))
            if not poses:
                raise ValueError("The session has no accepted camera poses; finish reconstruction first")
            model = work / "textured-model.zip"
            if textured_model:
                shutil.copyfile(textured_model, model)
            else:
                notify("Baking the saved fused mesh textures…")
                from types import SimpleNamespace
                import open3d as o3d
                from scanner_server.texturing import export_texture

                mesh_path = work / "mesh.ply"
                mesh_path.write_bytes(_read(archive, manifest.get("mesh", "mesh.ply"), 2 * 1024**3))
                engine = SimpleNamespace(mesh=o3d.io.read_triangle_mesh(str(mesh_path)), settings=settings,
                                         poses=poses, raw_frames=frames,
                                         frame_metadata=[f.get("metadata", {}) for f in manifest["frames"]])
                export_texture(engine, model, "obj.zip", **(texture_options or {"max_triangles": 200000}))
            model_dir = work / "model"
            model_dir.mkdir()
            with zipfile.ZipFile(model) as bundle:
                for info in bundle.infolist():
                    if info.is_dir():
                        continue
                    name = PurePosixPath(info.filename)
                    if name.name != info.filename or info.file_size > 2 * 1024**3:
                        raise ValueError("Unsupported textured model member")
                    (model_dir / name.name).write_bytes(_read(bundle, info.filename, 2 * 1024**3))
            if not (model_dir / "scan.obj").is_file():
                raise ValueError("Textured model bundle must contain scan.obj")
            assets.mkdir()
            c = settings.rgb_camera
            scene = {"version": 1, "session_id": report.get("session_id"),
                     "camera": {"width": c.width, "height": c.height, "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy},
                     "settings": settings.to_dict(), "snapshot_stride": stride,
                     "snapshot_grid": {"width": len(range(0, settings.camera.width, stride)),
                                       "height": len(range(0, settings.camera.height, stride))},
                     "unregistered_indices": sorted(set(range(len(frames))) - indices),
                     "snapshots": [], "images": assets.name, "model": str(model_dir / "scan.obj")}
            for number, (index, pose) in enumerate(poses, 1):
                notify(f"Preparing snapshot {number}/{len(poses)}…")
                rgb, depth = frames[index]
                rgb, vertices, faces, uv, observed, extrinsic = _snapshot(rgb, depth, settings, stride)
                Image.fromarray(rgb).save(assets / f"{number:06d}.png")
                geometry = work / f"{number:06d}.npz"
                np.savez(geometry, vertices=vertices.astype(np.float32), faces=faces,
                         uv=uv.astype(np.float32), observed=observed)
                scene["snapshots"].append({"index": index, "animation_frame": number + 1,
                    "camera_to_world": pose.tolist(), "rgb_to_world": (pose @ np.linalg.inv(extrinsic)).tolist(),
                    "geometry": str(geometry), "image": f"{number:06d}.png",
                    "timestamp_s": manifest["frames"][index].get("timestamp_s"),
                    "metadata": manifest["frames"][index].get("metadata", {})})
            configuration = work / "scene.json"
            configuration.write_text(json.dumps(scene, allow_nan=False), encoding="utf-8")
            notify("Creating Blender scene and animation…")
            staged = work / output.name
            command = [executable, "--background", "--factory-startup", "--threads", "2",
                       "--python-exit-code", "1", "--python", str(_scene_script()), "--",
                       str(configuration), str(staged), str(output)]
            environment = os.environ.copy()
            # Frozen clients set these for their own libraries, not Blender's.
            for key in ("PYTHONHOME", "PYTHONPATH", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
                environment.pop(key, None)
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            with (work / "blender.log").open("w+b") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        env=environment, creationflags=creationflags, timeout=3600)
                log.seek(max(0, log.tell() - 12000))
                detail = log.read().decode("utf-8", errors="replace")
            if result.returncode or not staged.is_file() or staged.stat().st_size < 12:
                raise RuntimeError(f"Blender could not create the scene:\n{detail.strip()}")
            staged.replace(output)
            committed = True
            notify(f"Saved {output.name}: {len(poses)} snapshots; keep {assets.name} beside it")
            return {"snapshots": len(poses), "unregistered": len(scene["unregistered_indices"]), "images": str(assets)}
    finally:
        if not committed and assets.exists():
            shutil.rmtree(assets)
