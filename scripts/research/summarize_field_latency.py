"""Summarize closed raw replay profiles without importing native libraries.

This is reporting only: it neither certifies mesh quality nor authorizes a
research backend. Selected archived views do not measure live camera FPS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if (value.get("schema_version") != 1
            or value.get("input_changed_during_profile") is not False
            or value.get("source_changed_during_profile") is not False
            or value.get("pose_seeds_used") is not False
            or value.get("finish_requested") is not True):
        raise ValueError(f"Not a closed unseeded replay profile: {path}")
    rows = value["live_diagnostics"]
    if len(rows) != value["frames"]:
        raise ValueError(f"Incomplete Live diagnostics: {path}")
    elapsed = [float(row["elapsed_ms"]) for row in rows]
    if not all(math.isfinite(item) and item >= 0 for item in elapsed):
        raise ValueError(f"Malformed Live latency: {path}")
    times = {name: value[name] for name in ("live_s", "finish_s", "processing_s")}
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool)
               and math.isfinite(item) and item >= 0 for item in times.values()):
        raise ValueError(f"Malformed replay times: {path}")
    return {
        "profile": path.name,
        "profile_sha256": hashlib.sha256(raw).hexdigest(),
        "session": value["session"],
        "input_sha256": value["input_sha256"],
        "source_sha256": value["source_sha256"],
        "settings_sha256": hashlib.sha256(json.dumps(value["settings"], sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "settings_overrides": value.get("settings_overrides", {}),
        "frames": len(rows),
        "accepted_before_finish": value["accepted_before_finish"],
        "accepted_after_finish": value["accepted"],
        "mesh_built": value["mesh_built"],
        "completed_scan_time_valid": value["mesh_built"] is True,
        **times,
        "live_latency_ms": {"p50": quantile(elapsed, .5), "p95": quantile(elapsed, .95),
            "maximum": max(elapsed), "quantile_method": "R7 linear interpolation"},
        "live_stages": value["live_stages"],
        "pipeline_options": value["pipeline_options"],
        "thread_policy": value["thread_policy"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve old summaries; require a fresh output")
    report = {"kind": "raw-field-replay-latency-summary-v1",
        "scope": "Recorded selected-view replay processing only; no camera FPS, statistical repeat confidence, mesh quality or research backend authority. Parent execution closure must be assessed separately. Failed mesh builds remain visible and are not completed-scan speed measurements.",
        "runs": [summarize(path) for path in args.profiles]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in report["runs"]:
        latency = row["live_latency_ms"]
        print(f"{row['profile']}: Live median {latency['p50']:.1f} ms, p95 {latency['p95']:.1f} ms, "
              f"Finish {row['finish_s']:.3f} s, mesh={row['mesh_built']}")


if __name__ == "__main__":
    main()
