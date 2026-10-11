"""Durable, ordered lossless buffers for selected captures awaiting receipts."""

import json
import logging
import os
import re
import shutil
import threading
import uuid
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CaptureSpool:
    RESERVE_BYTES = 1024**3

    def __init__(self, root, session_id, settings, *, receipts=False, base_index=0):
        if type(base_index) is not int or base_index < 0:
            raise ValueError("Invalid capture stream base index")
        self.path = Path(root) / uuid.uuid4().hex
        self.session_id = session_id
        self.settings = settings
        self.stream_id = uuid.uuid4().hex
        self.receipts = bool(receipts)
        self.base_index = base_index
        self.next_sequence = 0
        self.version = 2
        self.ordering_error = ""
        self._pending = set()
        self._lock = threading.RLock()

    def _save_manifest(self):
        self.path.mkdir(parents=True, exist_ok=True)
        temporary = self.path / "session.json.tmp"
        with temporary.open("w", encoding="utf-8") as output:
            json.dump({"version": 2, "session_id": self.session_id, "settings": self.settings,
                       "stream_id": self.stream_id, "receipts": self.receipts,
                       "base_index": self.base_index, "next_sequence": self.next_sequence,
                       "format": "RGB and depth arrays plus JSON metadata in lossless NPZ files"},
                      output, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(self.path / "session.json")

    @classmethod
    def open(cls, path):
        path = Path(path)
        manifest = json.loads((path / "session.json").read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or not isinstance(manifest.get("session_id"), str)
                or not manifest["session_id"] or not isinstance(manifest.get("settings"), dict)):
            raise ValueError("Invalid capture buffer session manifest")
        spool = cls.__new__(cls)
        spool.path, spool.session_id, spool.settings = path, manifest["session_id"], manifest["settings"]
        spool.version = manifest.get("version", 1)
        spool.stream_id = manifest.get("stream_id")
        spool.receipts = manifest.get("receipts", False)
        spool.base_index = manifest.get("base_index", 0)
        spool.next_sequence = manifest.get("next_sequence", 0)
        spool.ordering_error = ""
        spool._pending = set(path.glob("*.npz"))
        spool._lock = threading.RLock()
        if spool.version == 2:
            if (not isinstance(spool.stream_id, str) or not re.fullmatch(r"[0-9a-f]{32}", spool.stream_id)
                    or type(spool.receipts) is not bool
                    or type(spool.base_index) is not int or spool.base_index < 0
                    or type(spool.next_sequence) is not int or spool.next_sequence < 0):
                raise ValueError("Invalid ordered capture buffer manifest")
            sequences = []
            for pending in spool._pending:
                if not re.fullmatch(r"[0-9]{20}\.npz", pending.name):
                    spool.ordering_error = "Invalid ordered capture buffer filename"
                    break
                sequence = int(pending.stem)
                sequences.append(sequence)
                try:
                    _, _, metadata = cls.load(pending)
                    if spool.receipts and metadata.get("upload_receipt") != {
                            "stream_id": spool.stream_id, "sequence": sequence,
                            "expected_index": spool.base_index + sequence}:
                        raise ValueError("Capture receipt does not match its buffer sequence")
                except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, EOFError) as exc:
                    spool.ordering_error = f"Unreadable retained capture {pending.name}: {exc}"
                    break
            if sequences:
                # File publication can precede the counter checkpoint at a crash.
                spool.next_sequence = max(spool.next_sequence, max(sequences) + 1)
        elif spool.version == 1:
            spool._legacy_order()
        else:
            spool.ordering_error = "Unsupported capture buffer version"
        if any(path.glob("*.npz.tmp")):
            spool.ordering_error = spool.ordering_error or "Interrupted capture write remains in this buffer"
        return spool

    @classmethod
    def discover(cls, root):
        result = []
        root = Path(root)
        for path in sorted(root.iterdir()) if root.exists() else ():
            if not path.is_dir():
                continue
            if not any(path.glob("*.npz")) and not any(path.glob("*.npz.tmp")):
                continue
            try:
                spool = cls.open(path)
            except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile, EOFError) as exc:
                spool = cls.__new__(cls)
                spool.path, spool.session_id, spool.settings = path, None, {}
                spool.version, spool.receipts = 0, False
                spool._pending = set(path.glob("*.npz"))
                spool._lock = threading.RLock()
                spool.ordering_error = f"Unreadable capture buffer manifest: {exc}"
            if any(path.glob("*.npz.tmp")):
                spool.ordering_error = spool.ordering_error or "Interrupted capture write remains in this buffer"
            result.append(spool)
        return result

    def _legacy_order(self):
        self._legacy_paths = ()
        try:
            rows = [(path, self.load(path)[2]) for path in self._pending]
            parsed = [re.fullmatch(r"(.+):([0-9]+)", str(metadata.get("frame_id", ""))) for _, metadata in rows]
            if rows and all(parsed) and len({m[1] for m in parsed}) == 1 and len({int(m[2]) for m in parsed}) == len(rows):
                ordered = sorted(zip(rows, parsed), key=lambda pair: int(pair[1][2]))
                self._legacy_paths = tuple(pair[0][0] for pair in ordered)
                return
            identifiers = [metadata.get("frame_id") for _, metadata in rows]
            if rows and all(type(identity) is int and identity >= 0 for identity in identifiers) and len(set(identifiers)) == len(rows):
                self._legacy_paths = tuple(path for path, _ in sorted(rows, key=lambda row: row[1]["frame_id"]))
                return
            raise ValueError("Legacy capture order cannot be established; preserve the buffer for manual recovery")
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, EOFError) as exc:
            self.ordering_error = str(exc)

    @property
    def retry_safe(self):
        return self.version == 2 and self.receipts and not self.ordering_error

    @property
    def pending_paths(self):
        with self._lock:
            if self.version == 1 and not self.ordering_error:
                return tuple(path for path in self._legacy_paths if path in self._pending)
            return tuple(sorted(self._pending))

    @property
    def pending_count(self):
        with self._lock:
            return len(self._pending)

    def append(self, rgb, depth, metadata):
        if self.version != 2 or self.ordering_error:
            raise ValueError(self.ordering_error or "Legacy capture buffers are read-only")
        with self._lock:
            self.path.mkdir(parents=True, exist_ok=True)
            if not (self.path / "session.json").exists():
                self._save_manifest()
            if shutil.disk_usage(self.path).free < self.RESERVE_BYTES + rgb.nbytes + depth.nbytes + 65536:
                raise OSError("insufficient disk space for pending uploads")
            sequence = self.next_sequence
            destination = self.path / f"{sequence:020d}.npz"
            if destination.exists():
                raise OSError("Capture sequence already exists; reopen the buffer before continuing")
            metadata = dict(metadata)
            if self.receipts:
                if "upload_receipt" in metadata:
                    raise ValueError("Capture already belongs to an upload receipt stream")
                metadata["upload_receipt"] = {"stream_id": self.stream_id, "sequence": sequence,
                                               "expected_index": self.base_index + sequence}
            temporary = destination.with_name(destination.name + ".tmp")
            try:
                with temporary.open("xb") as output:
                    np.savez(output, rgb=rgb, depth=depth, metadata=json.dumps(metadata, allow_nan=False))
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(destination)
            except Exception:
                # An incomplete write is evidence, not a confirmed local capture.
                if temporary.exists():
                    self.ordering_error = "Interrupted capture write remains in this buffer; preserve it for manual recovery"
                raise
            self._pending.add(destination)
            self.next_sequence = sequence + 1
            try:
                self._save_manifest()
            except Exception:
                self.ordering_error = "Capture was retained locally but its counter checkpoint failed; reopen this buffer for recovery before capturing again"
                raise
            return destination

    @staticmethod
    def load(path):
        with np.load(path, allow_pickle=False) as frame:
            rgb, depth = frame["rgb"], frame["depth"]
            metadata = json.loads(str(frame["metadata"]))
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or depth.dtype != np.uint16 or depth.ndim != 2 or not isinstance(metadata, dict)):
            raise ValueError("Invalid retained RGB/depth capture")
        return rgb, depth, metadata

    def acknowledge(self, path):
        path = Path(path)
        with self._lock:
            if path not in self._pending:
                raise ValueError("Capture does not belong to this pending buffer")
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Confirmed capture buffer could not be removed: %s", path)
                return False
            self._pending.discard(path)
            return True

    def cleanup(self):
        # Retain the durable counter even when every capture has been acknowledged.
        # A later reconnect can continue this stream without reusing its identities.
        return None
