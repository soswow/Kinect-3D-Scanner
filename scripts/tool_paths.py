"""Resolve catalogued source resources after the performance-tool reorganization.

Legacy names are accepted for shared artifact lists; this does not launch old
commands, change historical reports or authorize reuse of historical proofs.
"""

import json
from functools import lru_cache
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = Path(__file__).with_name("tool-catalog.json")


@lru_cache(maxsize=1)
def _relocations():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {entry["legacy_path"]: entry["path"] for entry in catalog["entries"]}


def tool_path(name):
    """Return a repository source path for a basename, old path or current path."""
    relative = PurePosixPath(str(name).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
        raise ValueError("Expected a relative script resource path")
    if not relative.parts or relative.parts[0] != "scripts":
        relative = PurePosixPath("scripts") / relative
    relative = PurePosixPath(_relocations().get(str(relative), str(relative)))
    result = ROOT.joinpath(*relative.parts)
    if not result.resolve().is_relative_to(ROOT):
        raise ValueError("Script resource escapes the repository")
    return result
