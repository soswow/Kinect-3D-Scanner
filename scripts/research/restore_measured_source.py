"""Check/recreate the measured Windows baseline's raw source newline bytes.

Normal Git blobs remain unchanged. --restore is explicit and accepts only the
bundled manifest; new production source/inventory cannot be relabelled as old.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = Path(__file__).with_name("measured-source-layout.json")
_MANIFEST_SHA256 = "9880174429eb1687fd64aee141d05dd4ea66c62d07121e7613f8deb8fa8ef370"
_COUNTS = (74, 48)
_SUFFIXES = {".py", ".cpp", ".cu", ".h", ".hpp", ".toml"}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_manifest():
    raw = _MANIFEST.read_bytes()
    require(digest(raw) == _MANIFEST_SHA256, "Bundled measured-source manifest changed")
    value = json.loads(raw)
    require(value["schema"] == 1 and len(value["files"]) == _COUNTS[0]
            and len(value["core_inventory"]) == _COUNTS[1], "Malformed baseline inventory")
    names = [row["path"] for row in value["files"]]
    require(len(set(names)) == len(names), "Duplicate measured source path")
    for name in names:
        require(isinstance(name, str) and name.split("/")[0] in ("scanner_server", "shared", "native", "scripts")
                and all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) and part not in (".", "..")
                        for part in name.split("/")), "Unsafe measured source path")
    require(set(value["core_inventory"]) == {name for name in names
            if name.split("/")[0] in ("scanner_server", "shared", "native")}, "Core inventory mismatch")
    return value


def reconstruct(normal, row):
    require(row["style"] in ("lf", "crlf", "mixed") and type(row["raw_bytes"]) is int
            and 0 <= row["raw_bytes"] <= 4*1024**2, "Malformed source layout")
    parts = normal.split(b"\n")
    count = len(parts)-1
    require(count == row["newlines"], "Normalized newline count changed")
    indices = []
    for start, end in row.get("lf_lines", []):
        require(type(start) is int and type(end) is int and 1 <= start <= end <= count,
                "Invalid mixed newline range")
        indices.extend(range(start, end+1))
    require(indices == sorted(set(indices)), "Overlapping/out-of-order newline ranges")
    require((0 < len(indices) < count) if row["style"] == "mixed" else not indices,
            "Mixed newline descriptor disagrees with style")
    lf = set(indices)
    result = b"".join(part + (b"\n" if row["style"] == "lf" or index in lf else b"\r\n")
                      for index, part in enumerate(parts[:-1], 1)) + parts[-1]
    require(len(result) == row["raw_bytes"] and digest(result) == row["raw_sha256"],
            "Reconstructed source does not match measured bytes")
    return result


def plan(root):
    manifest = load_manifest()
    root = root.resolve(strict=True)
    inventory = {p.relative_to(root).as_posix() for folder in ("scanner_server", "shared", "native")
                 for p in (root/folder).rglob("*") if p.suffix in _SUFFIXES
                 and "build" not in p.relative_to(root).parts}
    require(inventory == set(manifest["core_inventory"]), "Core inventory changed; use the preserved baseline commit")
    result = []
    for row in manifest["files"]:
        path = root/row["path"]
        require(not any(p.is_symlink() for p in (path, *path.parents))
                and path.resolve(strict=True).is_relative_to(root) and path.is_file()
                and path.stat().st_size <= 4*1024**2, "Unsafe/missing measured source target")
        before = path.read_bytes()
        normal = before.replace(b"\r\n", b"\n")
        require(b"\r" not in normal and digest(normal) == row["normalized_sha256"],
                "Normalized source changed: " + row["path"])
        after = reconstruct(normal, row)
        result.append((path, before, after, stat.S_IMODE(path.stat().st_mode)))
    return manifest, result


def atomic_write(path, raw, mode):
    descriptor, name = tempfile.mkstemp(prefix="."+path.name+".restore-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        primary = sys.exception()
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup:
            if primary is not None:
                primary.add_note(f"Temporary source cleanup also failed: {cleanup}")
                raise primary from cleanup
            raise


def restore(items):
    written = []
    try:
        for path, before, after, mode in items:
            require(path.read_bytes() == before, "Source changed after preflight: " + str(path))
            if before != after:
                written.append((path, before, after, mode))
                atomic_write(path, after, mode)
        require(all(path.read_bytes() == after for path, _, after, _ in items), "Post-restore source check failed")
    except BaseException as primary:
        for path, before, after, mode in reversed(written):
            try:
                current = path.read_bytes()
                if current != before:
                    require(current == after, "Concurrent source change prevents safe rollback")
                    atomic_write(path, before, mode)
            except BaseException as cleanup:
                primary.add_note(f"Rollback also failed for {path}: {cleanup}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true", help="Explicitly recreate only verified measured newline bytes")
    args = parser.parse_args()
    manifest, items = plan(ROOT)
    changed = [str(path.relative_to(ROOT)) for path, before, after, _ in items if before != after]
    if args.restore:
        restore(items)
    print(json.dumps({"measured_source_sha256": manifest["source_sha256"], "paths": len(items),
                      "ready": args.restore or not changed, "restored": args.restore,
                      "newline_changes": changed}, indent=2))
    return 0 if args.restore or not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
