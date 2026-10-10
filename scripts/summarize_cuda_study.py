"""Reproduce the October 2026 fusion study from local closed CUDA profiles.

Contains the original study's fixed hardware/date and implementation labels.
For new studies, adapt that metadata before publishing selected evidence.
Generated summaries and charts stay under benchmark-output by default.
"""

import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("benchmark-output/cuda-study"))
    parser.add_argument("--summary", type=Path, default=Path("benchmark-output/cuda-session-performance.json"))
    args = parser.parse_args()
    import numpy as np

    aggregate = json.loads((args.directory / "backends.json").read_text())
    reports = [(run, json.loads(Path(run["report"]).read_text())) for run in aggregate["runs"]]
    rows = []
    for run, report in reports:
        diagnostics = report.get("live_diagnostics", report["diagnostics"])
        times = [d["elapsed_ms"] for d in diagnostics]
        accepted = report["accepted_indices_before_finish"]
        rows.append({"session": report["session"], "mode": run["mode"], "frames": report["frames"],
            "input_sha256": report["input_sha256"], "source_sha256": report["source_sha256"],
            "source_changed_during_profile": report["source_changed_during_profile"],
            "input_changed_during_profile": report["input_changed_during_profile"],
            "live_s": report["live_s"], "processing_fps": report["frames"] / report["live_s"],
            "frame_p50_s": float(np.percentile(times, 50) / 1000),
            "frame_p95_s": float(np.percentile(times, 95) / 1000),
            "first_frame_s": times[0] / 1000,
            "live_accepted": len(accepted), "live_accepted_indices": accepted,
            "finish_s": report["finish_s"], "final_accepted": report["accepted"] if report["finish_requested"] else None,
            "mesh_built": report["mesh_built"],
            "mesh_vertices": report["geometry"]["vertices"] if report["finish_requested"] else None,
            "mesh_triangles": report["geometry"]["triangles"] if report["finish_requested"] else None,
            "peak_rss_gib": report["peak_process_rss_bytes"] / 2**30,
            "stage_seconds": {name: stage["total_ms"] / 1000 for name, stage in report["live_stages"].items()},
            "backend": report["backend"], "final_reconstruction": report["final_reconstruction"]})
        if report["finish_requested"]:
            rows[-1]["finish_tested_pairs"] = report["fragment_reconnection"].get("tested_pairs")
            rows[-1]["finish_candidate_pairs"] = report["fragment_reconnection"].get("candidate_pairs")
            rows[-1]["finish_active_blocks"] = report["fragment_reconnection"].get("fusion_blocks")
    fusion = json.loads((args.directory / "fusion.json").read_text())
    speedups = {name: [sample["modes"][name]["median_ms"] /
                      sample["modes"]["cuda-fused"]["median_ms"] for sample in fusion["samples"]]
                for name in ("cpu-native", "cuda-tensor")}
    summary = {"date": "2026-10-07", "hardware": {"cpu": "Intel Core i7-12700F",
        "gpu": "RTX 3080 Ti", "vram_gib": 12, "system_ram_gib": 64, "os": "Windows 11"},
        "versions": {**reports[0][1]["versions"], "cupy": "13.6.0", "cuda_toolkit": "12.4", "nvidia_driver": "595.79"}, "omp_threads": 8,
        "cuda_kernel_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] /
            "scanner_server/weighted_fusion.cu").read_bytes()).hexdigest(),
        "post_measurement_change": "Optional CUDA compiler setup fallback was hardened; successful update path and CUDA kernel unchanged.",
        "full_replay_repeats": 1, "fusion_repeats_per_sample": 5,
        "measurement": "Sequential isolated processes; matched frames/settings. Live includes processing/storage. Finish separate. Excludes loading/imports, USB, network and textured exports. RSS excludes VRAM. No independent pose/geometry ground truth.",
        "runs": rows, "fusion_samples": fusion["samples"],
        "quality_comparisons": {path.stem: {key: value for key, value in json.loads(path.read_text()).items()
            if key != "baseline"} for path in args.directory.glob("*-quality-comparison.json")},
        "fusion_speedup_ranges": {name: [min(values), max(values)] for name, values in speedups.items()}}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.5, 1]})
    colors = {"cpu": "#78889c", "cuda-tensor": "#f0a44e", "cuda-fused": "#219c83", "cuda-legacy": "#5587c0"}
    mode_labels = {"cpu": "CPU", "cuda-tensor": "CUDA (existing)",
                   "cuda-fused": "CUDA (optimized)", "cuda-legacy": "CUDA + CPU tracking"}
    labels = [row["session"][:7] + " / " + mode_labels[row["mode"]] for row in rows]
    y = np.arange(len(rows))
    axes[0].barh(y, [row["live_s"] for row in rows], color=[colors[row["mode"]] for row in rows])
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Live processing time (seconds, lower is faster)")
    for i, row in enumerate(rows):
        axes[0].text(row["live_s"] + 3, i, f"{row['live_s']:.0f}s · {row['processing_fps']:.2f} fps", va="center", fontsize=9)
    axes[0].set_xlim(0, max(row["live_s"] for row in rows) * 1.42)
    modes = ("cpu-native", "cuda-tensor", "cuda-fused")
    medians = [np.median([sample["modes"][mode]["median_ms"] for sample in fusion["samples"]]) for mode in modes]
    axes[1].bar(np.arange(3), medians, color=[colors["cpu"], colors["cuda-tensor"], colors["cuda-fused"]])
    axes[1].set_xticks(np.arange(3), ["CPU native", "CUDA tensor", "CUDA fused"], rotation=15)
    axes[1].set_ylabel("Fusion per recorded frame (ms)")
    for i, value in enumerate(medians):
        axes[1].text(i, value + 3, f"{value:.1f} ms", ha="center")
    axes[1].set_ylim(0, max(medians) * 1.2)
    figure.suptitle("Recorded Kinect sessions: whole-pipeline speed and fusion speed", fontsize=15)
    figure.text(0.5, 0.01, "RTX 3080 Ti · i7-12700F · 5 mm voxels · unchanged confidence model · sequential runs", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.035, 1, 0.94))
    figure.savefig(args.directory / "performance.png", dpi=160)

    text = ["# Scanning performance on your recordings", "", "Measured on the i7-12700F / RTX 3080 Ti using all frames and the saved 5 mm scan settings.", "",
        "| Session | Backend | Live processing | Processing rate | Typical / p95 frame | Live accepted | Finish | Final accepted |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        finish = f"{row['finish_s']:.1f}s" if row["finish_s"] is not None else "—"
        final = str(row["final_accepted"]) if row["final_accepted"] is not None else "—"
        text.append(f"| {row['session'][:7]} | {mode_labels[row['mode']]} | {row['live_s']:.1f}s | {row['processing_fps']:.2f} fps | {row['frame_p50_s']:.2f}s / {row['frame_p95_s']:.2f}s | {row['live_accepted']}/{row['frames']} | {finish} | {final} |")
    text += ["", "![Performance](performance.png)", "", "Processing FPS measures how quickly already captured views are processed. USB, network transfer and texture export are excluded. Automatic capture pacing, camera motion and tracking recovery also affect scanning speed. A frame with lost tracking can take much longer than a typical frame.", "",
             "Full replays are single measurements per configuration; fusion has five warmed, synchronized measurements at each of six recorded views. CPU tracking/recovery timing varies. Finish can test different numbers of candidate links across backends; its timing difference cannot be attributed entirely to GPU fusion. These captures have no independent ground truth. Compare accepted frames and final geometry as well as speed.", "",
             "The fused confidence update is numerically checked against CPU and CUDA references. It preserves depth, confidence and truncation gates. Final graph reconnection and pose refinement still run on CPU."]
    text += ["", "## Final surface comparison", "",
             "Nearest-vertex distances compare two reconstructions of the same observations; they do not establish absolute accuracy or measure triangle interiors.", ""]
    for name, comparison in summary["quality_comparisons"].items():
        distance = comparison.get("symmetric_vertex_distance_m", {})
        accepted = "identical retained frame indices" if comparison["same_accepted_indices"] else "different retained frame indices"
        text.append(f"- {name[:7]}: {accepted}; vertex-distance RMS {distance.get('rms', 0)*1000:.3f} mm, p95 {distance.get('p95', 0)*1000:.3f} mm, maximum {distance.get('max', 0)*1000:.1f} mm. Maximum camera-pose difference {comparison['max_pose_translation_delta_m']*1000:.1f} mm / {comparison['max_pose_rotation_delta_deg']:.3f} degrees.")
        if surface := comparison.get("triangle_surface_metrics_vs_cpu"):
            text.append(f"  Area-weighted triangle-surface RMS {surface['surface_rmse_m']*1000:.3f} mm; p95 {surface['surface_p95_m']*1000:.3f} mm; precision/completeness within 5 mm {surface['precision']*100:.3f}% / {surface['completeness']*100:.3f}% (30,000 points per surface, fixed coordinates).")
    text += ["", "## Implementation and use", "",
             "The new fused confidence-weighted CUDA update borrows Open3D GPU buffers directly and processes bounded batches. Recorded fusion improved from a median 104.7 ms to 33.5 ms versus existing CUDA, with unchanged confidence equations and depth gates. CPU execution retains its existing implementation.", "",
             "Install the optional CUDA 12 dependency with `python -m pip install -r requirements-cuda-fusion.txt`. `KINECT_CUDA_FUSION=auto` uses it when available, `fused` requires it, and `tensor` selects the previous GPU implementation. Final validation passed 47 tests, including GPU parity, volume growth, rounding, compiler fallback and update-failure handling.", "",
             "The original study recorded a six-frame HTTP scan, PLY export and restoration of the test server settings. This formatter does not check a running server; these historical observations do not describe current server readiness."]
    (args.directory / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
