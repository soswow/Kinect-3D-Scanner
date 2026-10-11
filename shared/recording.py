"""Opt-in lossless local recordings, with a portable calibration manifest."""

import json
import time
from pathlib import Path

import cv2

_PNG_OPTIONS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


class RecordingWriter:
    def __init__(self, path, settings, session_id=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "rgb").mkdir()
        (self.path / "depth").mkdir()
        self.manifest = {
            "version": 1,
            "depth_unit": "raw_11bit_disparity"
            if settings.get("sensor_calibration")
            else "millimetres",
            "depth_encoding": "raw_11bit"
            if settings.get("sensor_calibration")
            else "registered_mm",
            "settings": settings,
            "frames": [],
        }
        if session_id:
            self.manifest["session_id"] = session_id
        self._save_manifest()

    @classmethod
    def for_session(cls, root, session_id):
        """Recover a matching acknowledged recording after a client restart.

        Older recordings identify their session in the reconstruction sidecar.
        This performs no server calls and does not modify a candidate recording.
        """
        if not session_id:
            return None
        paths = sorted(Path(root).glob("scan-*/manifest.json"), reverse=True)
        for path in paths:
            try:
                if path.stat().st_size > 256 * 1024**2:
                    continue
                manifest = json.loads(path.read_text())
                identity = manifest.get("session_id")
                if identity is None and manifest.get("reconstruction"):
                    report = (path.parent / manifest["reconstruction"]).resolve()
                    if not report.is_relative_to(path.parent.resolve()) or report.stat().st_size > 256 * 1024**2:
                        continue
                    identity = json.loads(report.read_text()).get("session_id")
                if identity != session_id or not isinstance(manifest.get("frames"), list):
                    continue
                recording = cls.__new__(cls)
                recording.path = path.parent
                recording.manifest = manifest
                return recording
            except (OSError, ValueError, AttributeError, TypeError):
                continue
        return None

    def _save_manifest(self):
        temporary = self.path / "manifest.json.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2, allow_nan=False))
        temporary.replace(self.path / "manifest.json")

    def save_reconstruction(self, report):
        temporary = self.path / "reconstruction.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(self.path / "reconstruction.json")
        self.manifest["reconstruction"] = "reconstruction.json"
        self._save_manifest()

    def append(self, rgb, depth, metadata=None):
        index = len(self.manifest["frames"])
        color_path = f"rgb/{index:06d}.png"
        depth_path = f"depth/{index:06d}.png"
        if not cv2.imwrite(
            str(self.path / color_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), _PNG_OPTIONS
        ):
            raise OSError("Could not save RGB recording")
        if not cv2.imwrite(str(self.path / depth_path), depth, _PNG_OPTIONS):
            raise OSError("Could not save depth recording")
        metadata = dict(metadata or {})
        display_paths = {}
        rotation = metadata.get("orientation", {}).get("rotation_cw_degrees", 0)
        if rotation in (90, 180, 270):
            from .inertial import rotate_display
            for stream, array, relative in (("rgb", rgb, color_path), ("depth", depth, depth_path)):
                destination = self.path / "display" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shown = rotate_display(array, rotation)
                if stream == "rgb":
                    shown = cv2.cvtColor(shown, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(destination), shown, _PNG_OPTIONS):
                    raise OSError("Could not save portrait recording")
                display_paths["display_" + stream] = "display/" + relative
        self.manifest["frames"].append(
            {
                "rgb": color_path,
                "depth": depth_path,
                "timestamp_s": metadata.get("timestamp_s", time.time()),
                "metadata": metadata,
                **display_paths,
            }
        )
        self._save_manifest()
