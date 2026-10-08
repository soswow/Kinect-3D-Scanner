"""Isolated matched CUDA pipeline experiments on unmodified session ZIPs.

Recipes isolate conservative optimizations, projective odometry and deferring
expensive live recovery to Finish. Every raw view is retained in every recipe.
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import file_hash, source_hash
from scripts.process_metrics import gpu_info

KEYS = ("KINECT_CUDA_REGISTRATION", "KINECT_CUDA_ODOMETRY", "KINECT_CUDA_MATCHING",
        "KINECT_MODEL_REFRESH", "KINECT_KEYFRAME_CACHE", "KINECT_LIVE_RECOVERY", "KINECT_VISUAL_FEATURES",
        "KINECT_VISUAL_REFINEMENT", "KINECT_FINAL_VISUAL_FIRST", "KINECT_FINAL_LOCAL_REFINEMENT",
        "KINECT_ADAPTIVE_EXPERIMENTAL", "KINECT_CUDA_INPUT", "KINECT_CUDA_CONFIDENCE")
RECIPES = {
    "baseline": ("cpu", "off", "cpu", "eager", "off", "full", "orb"),
    "cached": ("cpu", "off", "cuda", "lazy", "on", "full", "orb"),
    "tensor": ("tensor", "off", "cuda", "lazy", "on", "full", "orb"),
    "dense": ("cpu", "hybrid", "cuda", "lazy", "on", "full", "orb"),
    "dense-tensor": ("tensor", "hybrid", "cuda", "lazy", "on", "full", "orb"),
    "dense-plane": ("cpu", "point-to-plane", "cuda", "lazy", "on", "full", "orb"),
    "deferred": ("tensor", "hybrid", "cuda", "lazy", "on", "deferred", "orb"),
    "sift": ("tensor", "off", "cuda", "lazy", "on", "full", "sift"),
    "sift-dense": ("tensor", "hybrid", "cuda", "lazy", "on", "full", "sift"),
    "sift-deferred": ("tensor", "off", "cuda", "lazy", "on", "deferred", "sift"),
}
RECIPES = {name: options + ("icp", "off", "icp") for name, options in RECIPES.items()}
RECIPES.update({
    "sift-cpu": ("cpu", "off", "cuda", "lazy", "on", "full", "sift", "icp", "off", "icp"),
    "measured": ("cpu", "off", "cuda", "lazy", "on", "full", "sift", "measured", "off", "icp"),
    "measured-final": ("cpu", "off", "cuda", "lazy", "on", "full", "sift", "measured", "on", "measured"),
    "measured-visual-final": ("cpu", "off", "cuda", "lazy", "on", "full", "sift", "measured", "on", "icp"),
    "sift-visual-final": ("cpu", "off", "cuda", "lazy", "on", "full", "sift", "icp", "on", "icp"),
    "orb-fine": ("cpu", "off", "cuda", "lazy", "on", "full", "orb", "measured-fine", "off", "icp"),
    "sift-fine": ("cpu", "off", "cuda", "lazy", "on", "full", "sift", "measured-fine", "off", "icp"),
    "orb-fine-visual-final": ("cpu", "off", "cuda", "lazy", "on", "full", "orb", "measured-fine", "on", "icp"),
    "orb-visual-final": ("cpu", "off", "cuda", "lazy", "on", "full", "orb", "icp", "on", "icp"),
    "legacy-cached": ("cpu", "off", "cuda", "lazy", "on", "full", "orb", "icp", "off", "icp"),
    "legacy-preview-10mm": ("cpu", "off", "cuda", "lazy", "on", "full", "orb", "icp", "off", "icp"),
    "cpu-baseline": ("cpu", "off", "cpu", "eager", "off", "full", "orb", "icp", "off", "icp"),
    "cpu-cached": ("cpu", "off", "cpu", "lazy", "on", "full", "orb", "icp", "off", "icp"),
    "legacy-sift": ("cpu", "off", "cuda", "lazy", "on", "full", "sift", "icp", "off", "icp"),
    "legacy-sift-preview-10mm": ("cpu", "off", "cuda", "lazy", "on", "full", "sift", "icp", "off", "icp"),
    "cpu-sift-preview-10mm": ("cpu", "off", "cpu", "lazy", "on", "full", "sift", "icp", "off", "icp"),
    "legacy-adaptive-preview-10mm": ("cpu", "off", "cuda", "lazy", "on", "full", "adaptive", "icp", "off", "icp"),
    "legacy-adaptive": ("cpu", "off", "cuda", "lazy", "on", "full", "adaptive", "icp", "off", "icp"),
    "cpu-adaptive": ("cpu", "off", "cpu", "lazy", "on", "full", "adaptive", "icp", "off", "icp"),
    "cpu-adaptive-preview-10mm": ("cpu", "off", "cpu", "lazy", "on", "full", "adaptive", "icp", "off", "icp"),
})
# Rejected fine-resolution adaptive variants remain explicitly experimental.
# Every other recipe clears an override inherited from the shell.
RECIPES = {name: options + ("on" if name in ("legacy-adaptive", "cpu-adaptive") else "off",)
           for name, options in RECIPES.items()}
RECIPES = {name: options + ("off",) for name, options in RECIPES.items()}
RECIPES["legacy-cached-gpu-input"] = RECIPES["legacy-cached"][:-1] + ("on",)
RECIPES["legacy-adaptive-preview-10mm-gpu-input"] = RECIPES["legacy-adaptive-preview-10mm"][:-1] + ("on",)
RECIPES = {name: options + ("off",) for name, options in RECIPES.items()}
RECIPES["legacy-adaptive-preview-10mm-gpu-input-confidence"] = RECIPES["legacy-adaptive-preview-10mm-gpu-input"][:-1] + ("on",)
PREVIEW_VOXELS = {"legacy-preview-10mm": .01, "legacy-sift-preview-10mm": .01,
                  "cpu-sift-preview-10mm": .01, "legacy-adaptive-preview-10mm": .01,
                  "cpu-adaptive-preview-10mm": .01,
                  "legacy-adaptive-preview-10mm-gpu-input": .01,
                  "legacy-adaptive-preview-10mm-gpu-input-confidence": .01}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--runs", nargs="+", choices=RECIPES, default=("baseline", "dense", "dense-tensor", "deferred"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--quality-reference-directory", type=Path,
                        help="After each Finish, compare with <session-stem>-cpu-0.json in this directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tracking", choices=("legacy", "tensor"), default="tensor",
                        help="Tracking implementation; CUDA fusion/retrieval remain selected")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-output/cuda-pipeline/experiments.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Require positive repeats, threads and limit")
    if args.quality_reference_directory and (not args.finish or args.limit):
        parser.error("Quality references require complete Finish replays")
    if args.quality_reference_directory:
        for session in args.sessions:
            if not (args.quality_reference_directory / f"{session.stem}-cpu-0.json").exists():
                parser.error(f"Missing finished CPU reference for {session}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = source_hash()
    hardware = gpu_info()
    reports = []
    for session in args.sessions:
        input_hash = file_hash(session)
        with zipfile.ZipFile(session) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            count = len(manifest["frames"])
        selected = list(range(count))[:args.limit]
        for repeat in range(args.repeats):
            for mode in args.runs if repeat % 2 == 0 else reversed(args.runs):
                device = "cpu" if mode.startswith("cpu-") else "cuda"
                tracking = "legacy" if mode.startswith(("legacy-", "cpu-")) else args.tracking
                preview_voxel = PREVIEW_VOXELS.get(mode)
                adaptive = False  # Production recipes select the core adaptive policy.
                overrides = ({"voxel_m": preview_voxel, "final_voxel_m":
                    manifest["settings"].get("final_voxel_m") or manifest["settings"]["voxel_m"]}
                    if preview_voxel is not None else {})
                options = dict(zip(KEYS, RECIPES[mode]))
                env = dict(os.environ, **options, OMP_NUM_THREADS=str(args.threads), KINECT_NATIVE="on",
                           KINECT_BLOCK_COUNT="5000", KINECT_CUDA_FUSION="tensor" if device == "cpu" else "fused")
                destination = args.output.parent / f"{session.stem}-{mode}-{repeat}.json"
                quality_path = destination.with_name(destination.stem + "-quality.json")
                reusable = False
                if args.resume and destination.exists():
                    old = json.loads(destination.read_text())
                    reusable = (old["input_sha256"] == input_hash and old["source_sha256"] == fingerprint
                                and old.get("gpu_hardware") == hardware
                                and not old["source_changed_during_profile"] and not old["input_changed_during_profile"]
                                and old.get("pipeline_options") == options and old["selected_indices"] == selected
                                and old["finish_requested"] == args.finish and not old["pose_seeds_used"]
                                and old["settings_overrides"] == overrides and old["backend"]["device"] == ("CPU:0" if device == "cpu" else "CUDA:0")
                                and old.get("experimental_visual_fallback", False) == adaptive
                                and (not adaptive or old.get("experimental_policy_sha256") == file_hash(ROOT / "scripts/research/archive/adaptive_visual_experiment.py"))
                                and not old.get("experimental_policy_changed", False)
                                and old["backend"]["tracking"] == tracking and old["omp_threads"] == str(args.threads)
                                and old["native_mode"] == "on" and old["initial_blocks"] == "5000"
                                and (device == "cpu" or old["backend"].get("confidence_cuda", {}).get("implementation") == "fused"))
                print(f"{session.name}: {mode} {repeat+1}/{args.repeats}", flush=True)
                if not reusable:
                    # A replaced report must never inherit a previous surface check.
                    quality_path.unlink(missing_ok=True)
                    command = [sys.executable, str(ROOT / "scripts/profile_session.py"), str(session),
                               "--device", device, "--tracking", tracking, "--output", str(destination)]
                    if args.limit:
                        command.extend(("--limit", str(args.limit)))
                    if args.finish:
                        command.append("--finish")
                    if preview_voxel is not None:
                        command.extend(("--live-voxel", str(preview_voxel)))
                    if adaptive:
                        command.append("--visual-fallback-sift")
                    with destination.with_suffix(".log").open("w") as log:
                        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                                stderr=subprocess.STDOUT, timeout=7200)
                    if result.returncode:
                        raise RuntimeError(f"Replay failed: {destination.with_suffix('.log')}")
                report = json.loads(destination.read_text())
                if (source_hash() != fingerprint or report["source_sha256"] != fingerprint
                        or report["source_changed_during_profile"] or report["input_changed_during_profile"]
                        or report.get("experimental_policy_changed", False)
                        or report.get("gpu_hardware") != hardware or gpu_info() != hardware):
                    raise RuntimeError("Implementation, input or GPU driver changed during experiment; rerun fixed")
                reports.append({"session": session.name, "mode": mode, "repeat": repeat,
                                "report": str(destination.resolve()), **{key: report[key] for key in (
                                    "input_sha256", "source_sha256", "pipeline_options", "frames", "live_s", "finish_s",
                                    "accepted_before_finish", "accepted", "accepted_indices", "tracking_methods", "backend",
                                    "live_stages", "geometry", "fragment_reconnection", "peak_process_rss_bytes")}})
                args.output.write_text(json.dumps({"recipes": RECIPES, "runs": reports}, indent=2, allow_nan=False)+"\n")
                print(f"  live={report['live_s']:.2f}s finish={report['finish_s']} accepted={report['accepted_before_finish']}/{report['frames']} final={report['accepted']}", flush=True)
                if args.quality_reference_directory:
                    # A resumed report can also be compared with a changed reference.
                    quality_path.unlink(missing_ok=True)
                    comparison_command = [sys.executable, str(ROOT / "scripts/compare_session_profiles.py"),
                        str(args.quality_reference_directory / f"{session.stem}-cpu-0.json"),
                        str(destination), "--output", str(quality_path)]
                    if preview_voxel is not None:
                        comparison_command.append("--allow-preview-resolution-change")
                    with quality_path.with_suffix(".log").open("w") as log:
                        result = subprocess.run(comparison_command, cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, timeout=180)
                    if result.returncode:
                        raise RuntimeError(f"Quality comparison failed: {quality_path.with_suffix('.log')}")
                    quality = json.loads(quality_path.read_text())
                    surface = quality["triangle_surface_metrics_vs_cpu"]
                    print(f"  surface p95={surface['surface_p95_m']*1000:.3f}mm precision={surface['precision']:.4f} completeness={surface['completeness']:.4f} same_views={quality['same_accepted_indices']}", flush=True)


if __name__ == "__main__":
    main()
