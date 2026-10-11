"""Project image encoding, lossless archive compression and local capture reuse.

Only verified native pixels can replace images omitted from a server transfer.
Completion is transactional; its temporary archive is removed on failure.
"""

import copy
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

JPEG_QUALITY = 95


def member_compression(name):
    return zipfile.ZIP_DEFLATED if name.endswith((".json", ".jsonl", ".ply")) else zipfile.ZIP_STORED


def write_member(archive, name, data):
    archive.writestr(name, data, compress_type=member_compression(name), compresslevel=6)


def copy_member(source, target, item, destination=None):
    info = copy.copy(source.getinfo(item) if isinstance(item, str) else item)
    info.filename = destination or info.filename
    info.compress_type = member_compression(info.filename)
    with source.open(item) as incoming, target.open(info, "w", force_zip64=True) as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)


def encode_image(array, rgb_format="png"):
    if rgb_format not in ("png", "jpeg"):
        raise ValueError("Project RGB format must be png or jpeg")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG" if rgb_format == "jpeg" else "PNG",
                               **({"quality": JPEG_QUALITY} if rgb_format == "jpeg" else {}))
    return buffer.getvalue()


def pixel_digest(array):
    array = np.ascontiguousarray(array)
    # Include shape/type, and make uint16 byte order portable across hosts.
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    digest = hashlib.sha256(f"{array.shape}:{array.dtype.str}:".encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def read_capture(paths):
    with Image.open(paths["rgb"]) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    with Image.open(paths["depth"]) as image:
        if image.mode not in ("I", "I;16", "I;16B", "I;16L"):
            raise ValueError("Local project depth must be a uint16 image")
        depth = np.asarray(image)
        if np.any(depth < 0) or np.any(depth > 65535):
            raise ValueError("Invalid local project depth")
        depth = depth.astype(np.uint16)
    return rgb, depth


def recording_captures(recording):
    """Read acknowledged local files; missing or unreadable pairs use the server."""
    captures = {}
    root = recording.path.resolve()
    for frame in recording.manifest["frames"]:
        if not isinstance(frame, dict) or not isinstance(frame.get("metadata", {}), dict):
            continue
        index = frame.get("metadata", {}).get("server_index")
        if type(index) is not int or index < 0 or index in captures:
            continue
        try:
            paths = {key: (root / frame[key]).resolve() for key in ("rgb", "depth")}
        except (KeyError, TypeError, OSError):
            continue
        if not all(path.is_relative_to(root) for path in paths.values()):
            continue
        try:
            rgb, depth = read_capture(paths)
        except (OSError, ValueError):
            continue
        captures[index] = {**paths, "rgb_sha256": pixel_digest(rgb), "depth_sha256": pixel_digest(depth)}
    return captures


def complete_local_captures(path, captures, session_id):
    """Turn a partial transfer into a standalone project before publishing it."""
    path = Path(path)
    temporary = path.with_name(path.name + ".captures.tmp")
    try:
        with zipfile.ZipFile(path) as source:
            manifest = json.loads(source.read("manifest.json"))
            reuse = manifest.pop("local_captures", None)
            if reuse is None:
                return 0
            if reuse.get("session_id") != session_id:
                raise ValueError("Local capture transfer belongs to another session")
            rgb_format = manifest.get("rgb_format", "png")
            extension = "jpg" if rgb_format == "jpeg" else "png"
            seen = set()
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
                for item in source.infolist():
                    if item.filename != "manifest.json":
                        copy_member(source, target, item)
                for proof in reuse["captures"]:
                    index = proof["index"]
                    if type(index) is not int or not 0 <= index < len(manifest["frames"]) or index in seen:
                        raise ValueError("Invalid local capture transfer index")
                    seen.add(index)
                    local = captures[index]
                    rgb, depth = read_capture(local)
                    if any(pixel_digest(array) != proof[key + "_sha256"]
                           for key, array in (("rgb", rgb), ("depth", depth))):
                        raise ValueError("Local capture changed during save")
                    frame = manifest["frames"][index]
                    rotation = frame.get("metadata", {}).get("orientation", {}).get("rotation_cw_degrees", 0)
                    for key, array, suffix in (("rgb", rgb, extension), ("depth", depth, "png")):
                        name = f"{key}/{index:06d}.{suffix}"
                        if frame[key] != name or name in source.namelist():
                            raise ValueError("Invalid omitted project member")
                        write_member(target, name, encode_image(array, rgb_format if key == "rgb" else "png"))
                        display = frame.get("display_" + key)
                        if display:
                            if display != "display/" + name or rotation not in (90, 180, 270) or display in source.namelist():
                                raise ValueError("Invalid omitted project display member")
                            shown = np.rot90(array, k=-(rotation // 90))
                            write_member(target, display, encode_image(shown, rgb_format if key == "rgb" else "png"))
                write_member(target, "manifest.json", json.dumps(manifest, separators=(",", ":"), allow_nan=False))
        temporary.replace(path)
        return len(seen)
    finally:
        temporary.unlink(missing_ok=True)
