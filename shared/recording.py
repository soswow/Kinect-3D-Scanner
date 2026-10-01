"""Opt-in lossless local recordings, with a portable calibration manifest."""

import json
import time
from pathlib import Path

import cv2


class RecordingWriter:
    def __init__(self, path, settings):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "rgb").mkdir()
        (self.path / "depth").mkdir()
        self.manifest = {
            "version": 1,
            "depth_unit": "millimetres",
            "settings": settings,
            "frames": [],
        }
        self._save_manifest()

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
            str(self.path / color_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ):
            raise OSError("Could not save RGB recording")
        if not cv2.imwrite(str(self.path / depth_path), depth):
            raise OSError("Could not save depth recording")
        metadata = dict(metadata or {})
        self.manifest["frames"].append(
            {
                "rgb": color_path,
                "depth": depth_path,
                "timestamp_s": metadata.get("timestamp_s", time.time()),
                "metadata": metadata,
            }
        )
        self._save_manifest()
