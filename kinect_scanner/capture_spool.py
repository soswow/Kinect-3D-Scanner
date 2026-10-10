"""Lossless disk buffering for selected captures awaiting server acknowledgement."""

import json
import logging
import shutil
import threading
import uuid
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CaptureSpool:
    # Reserve space for the running app, recordings and final project saves.
    RESERVE_BYTES = 1024**3

    def __init__(self, root, session_id, settings):
        self.path = Path(root) / uuid.uuid4().hex
        self.session_id = session_id
        self.settings = settings
        self._pending = set()
        self._lock = threading.Lock()

    @property
    def pending_count(self):
        with self._lock:
            return len(self._pending)

    def append(self, rgb, depth, metadata):
        """Write without image compression; keep the submitting thread responsive."""
        self.path.mkdir(parents=True, exist_ok=True)
        if not (self.path / "session.json").exists():
            (self.path / "session.json").write_text(json.dumps({
                "session_id": self.session_id, "settings": self.settings,
                "format": "RGB and depth arrays plus JSON metadata in lossless NPZ files",
            }, allow_nan=False), encoding="utf-8")
        if shutil.disk_usage(self.path).free < self.RESERVE_BYTES + rgb.nbytes + depth.nbytes + 65536:
            raise OSError("insufficient disk space for pending uploads")
        destination = self.path / (uuid.uuid4().hex + ".npz")
        temporary = destination.with_suffix(".tmp")
        try:
            with temporary.open("wb") as output:
                np.savez(output, rgb=rgb, depth=depth,
                         metadata=json.dumps(metadata, allow_nan=False))
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        with self._lock:
            self._pending.add(destination)
        return destination

    @staticmethod
    def load(path):
        with np.load(path, allow_pickle=False) as frame:
            return frame["rgb"], frame["depth"], json.loads(str(frame["metadata"]))

    def acknowledge(self, path):
        # Failed or uncertain uploads remain on disk, including after app exit.
        path = Path(path)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Confirmed capture buffer could not be removed: %s", path)
        with self._lock:
            self._pending.discard(path)

    def cleanup(self):
        if not self.pending_count and self.path.exists():
            try:
                if not any(self.path.glob("*.npz")):
                    (self.path / "session.json").unlink(missing_ok=True)
                    self.path.rmdir()
            except OSError:
                logger.warning("Empty capture buffer could not be removed: %s", self.path)
