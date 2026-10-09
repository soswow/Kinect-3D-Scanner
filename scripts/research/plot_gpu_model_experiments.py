"""Plot scored synthetic GPU previews; these are not real-camera scan rates."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(directory, filename, period):
    report = json.loads((directory / filename).read_text())
    return next(row for row in report["runs"] if row["period"] == period)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("benchmark-output/cuda-pipeline"))
    args = parser.parse_args()
    selected = [
        (load(args.directory, "anchored-model-submaps.json", 0), "GPU map without corrections", "#8797ac"),
        (load(args.directory, "anchored-model-measured.json", 12), "Checked poses; original map", "#d4943d"),
        (load(args.directory, "anchored-model-submaps.json", 12), "Rebuild local map every 12 views", "#209582"),
        (load(args.directory, "anchored-model-submaps.json", 6), "Rebuild local map every 6 views", "#6e60a7"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for run, label, color in selected:
        rows = run["records"]
        frames = [row["index"] + 1 for row in rows]
        label += f" · RMS {run['preview_translation_rmse_m']*1000:.1f} mm"
        axes[0].plot(frames, [row["translation_error_m"]*1000 for row in rows], label=label, color=color)
        if run.get("reset_submaps", False) and run["period"]:
            axes[1].plot(frames[1:], [row["elapsed_ms"] for row in rows[1:]], color=color,
                         label=f"Every {run['period']} views · {run['steady_processing_fps']:.1f} prepared views/s")
    axes[0].set_ylabel("Translation error against known motion (mm)")
    axes[1].set_ylabel("Processing per prepared view (ms)")
    axes[1].set_xlabel("Synthetic camera view")
    axes[1].axhline(1000/30, color="#8797ac", linestyle="--", alpha=.7, label="33.3 ms per view")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("GPU preview: correcting the local map reduces drift", fontsize=15)
    fig.text(.5, .015, "60 noisy synthetic views · truth used only for scoring · uniform preview fusion · camera preparation/USB/network/final reconstruction excluded", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(args.directory / "gpu-model-motion.png", dpi=150)


if __name__ == "__main__":
    main()
