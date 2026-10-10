"""Find scanner tools and protocols without importing or running them.

Uses only the standard library. Directory groups describe intended use, not
validation status; research protocols record measured and unexecuted scopes.
"""

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".py", ".ps1", ".cu", ".cpp", ".md", ".json"}


def describe(path, roles):
    relative = path.relative_to(ROOT).as_posix()
    parts = path.relative_to(ROOT / "scripts").parts
    group = "archive" if "archive" in parts else (
        "research" if parts[0] == "research" else (
            "benchmarks" if parts[0] == "benchmarks" else "workflow"))
    source = path.read_text(encoding="utf-8-sig")
    purpose = roles.get(relative, "")
    if path.suffix == ".py":
        tree = ast.parse(source, filename=relative)
        purpose = (ast.get_docstring(tree) or purpose).split("\n")[0]
        # Some original validation scripts parse arguments at module level.
        cli = any(isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "parse_args" for node in ast.walk(tree))
        cli = cli or any(isinstance(node, ast.If)
                         and "__name__" in ast.unparse(node.test)
                         and "__main__" in ast.unparse(node.test) for node in tree.body)
        kind = "cli" if cli else "helper"
    elif path.suffix == ".ps1":
        kind = "cli"
        if not purpose:
            purpose = next((line.strip().strip("<#> ") for line in source.splitlines()
                            if line.startswith("<#") or line.startswith("# ")), "")
    elif path.suffix in {".cu", ".cpp"}:
        kind = "native"
    elif path.suffix == ".md":
        kind = "protocol"
        purpose = source.splitlines()[0].lstrip("# ") if source else purpose
    else:
        kind = "catalog"
        data = json.loads(source)
        purpose = data.get("scope", purpose) if isinstance(data, dict) else purpose
    return {"path": relative, "group": group, "kind": kind,
            "purpose": purpose or "Supporting source; see its family README/protocol."}


def inventory():
    roles = {}
    for catalog in (ROOT / "scripts/tool-catalog.json",
                    ROOT / "scripts/research/field-tool-catalog.json"):
        for entry in json.loads(catalog.read_text(encoding="utf-8"))["entries"]:
            if entry.get("role"):
                roles[entry["path"]] = entry["role"]
    return [describe(path, roles) for path in sorted((ROOT / "scripts").rglob("*"))
            if path.is_file() and path.suffix in SUFFIXES
            and "__pycache__" not in path.parts
            and path.name not in {"__init__.py", "README.md"}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", default="", help="Match path or purpose, ignoring case")
    parser.add_argument("--group", choices=("workflow", "benchmarks", "research", "archive"))
    parser.add_argument("--kind", choices=("cli", "helper", "native", "protocol", "catalog"))
    parser.add_argument("--json", action="store_true", help="Emit structured records for agents")
    args = parser.parse_args()
    needle = args.search.casefold()
    rows = [row for row in inventory()
            if (not args.group or row["group"] == args.group)
            and (not args.kind or row["kind"] == args.kind)
            and needle in (row["path"] + " " + row["purpose"]).casefold()]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=True))
    else:
        for row in rows:
            print(f"{row['path']} [{row['group']}/{row['kind']}]\n  {row['purpose']}")


if __name__ == "__main__":
    main()
