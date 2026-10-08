"""Writable client paths and process commands for source and bundled runs."""

import os
from pathlib import Path
import sys


def source_root():
    if not getattr(sys, "frozen", False):
        root = Path(__file__).resolve().parents[1]
        if (root / "pyproject.toml").is_file():
            return root
    return None


def data_root():
    source = source_root()
    if source is not None:
        return source
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/Kinect3DScanner"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Kinect3DScanner"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "Kinect3DScanner"


def log_path():
    if sys.platform == "darwin":
        root = Path.home() / "Library/Logs/Kinect3DScanner"
    else:
        root = data_root() / "logs"
    return root / "client.log"


def export_root():
    if source_root() is None:
        return Path.home() / "Documents/Kinect 3D Scanner/export"
    return data_root() / "export"


def viewer_command(arguments):
    if getattr(sys, "frozen", False):
        return [sys.executable, "--viewer", arguments]
    return [sys.executable, "-m", "kinect_scanner.viewer", arguments]
