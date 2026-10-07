"""Inspect an exported raw session without modifying it or a running server.

python scripts/inspect_session.py export/session.zip --output benchmark-output/session
Frame indices in JSON are zero based; contact sheets display one-based numbers.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def accepted_runs(frames):
    runs = []
    for frame in frames:
        if not frame.get("success"):
            continue
        index = frame["index"]
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def inspect(path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        report_name = manifest.get("reconstruction", "reconstruction.json")
        reconstruction = json.loads(archive.read(report_name))
        frames = reconstruction.get("frames", [])
        by_index = {f["index"]: f for f in frames}
        captures = manifest["frames"]
        timestamps = np.array([f["timestamp_s"] for f in captures])
        lags = [f.get("metadata", {}).get("rgb_depth_delta_ms") for f in captures]
        measured_lags = [x for x in lags if x is not None]
        runs = accepted_runs(frames)
        summary = {
            "source": str(Path(path).resolve()),
            "session_id": reconstruction.get("session_id"),
            "stored": len(captures),
            "processed": len(frames),
            "accepted": sum(bool(f.get("success")) for f in frames),
            "skipped_indices": [f["index"] for f in frames if not f.get("success")],
            "accepted_runs": runs,
            "note": "Runs are capture continuity segments, not independently registered pose graphs.",
            "capture_duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) else 0,
            "processing_duration_s": sum(f.get("elapsed_ms", 0) for f in frames) / 1000,
            "median_capture_interval_s": float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else None,
            "capture_gaps_over_2s": [
                {"before_index": i + 1, "gap_s": float(delta)}
                for i, delta in enumerate(np.diff(timestamps)) if delta > 2
            ],
            "rgb_depth_pair_count": len(measured_lags),
            "rgb_depth_pairs_within_20ms": sum(abs(x) <= 20 for x in measured_lags),
            "rgb_depth_pairs_over_20ms": sum(abs(x) > 20 for x in measured_lags),
            "color_tracking_requested": manifest["settings"].get("color_recovery", False),
            "appearance_recovery_requested": manifest["settings"].get("relocalize", False),
            "refinement": reconstruction.get("refinement"),
            "bundle_adjustment": reconstruction.get("bundle_adjustment"),
            "final_reconstruction": reconstruction.get("final_reconstruction"),
            "failures": [{"index": f["index"], "message": f.get("message")} for f in frames if not f.get("success")],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        sheet = Image.new("RGB", (1200, max(130, ((len(captures) + 4) // 5) * 130)), "#202329")
        draw = ImageDraw.Draw(sheet)
        for i, capture in enumerate(captures):
            with archive.open(capture["rgb"]) as source, Image.open(source) as original:
                image = original.convert("RGB")
                image.thumbnail((235, 105))
            x, y = (i % 5) * 240, (i // 5) * 130
            sheet.paste(image, (x, y + 23))
            frame = by_index.get(i)
            state = "pending" if frame is None else frame.get("method", "reference") if frame.get("success") else "SKIPPED"
            color = "#ffffff" if frame is None or frame.get("success") else "#ff8a83"
            draw.text((x + 3, y + 3), f"Frame {i + 1}: {state}", fill=color)
        sheet.save(output / "contact-sheet.jpg")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        inspect(args.session, args.output)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Cannot inspect session: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
