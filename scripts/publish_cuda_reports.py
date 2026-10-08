"""Copy selected CUDA reports into docs without publishing captures or binaries.

Run after the summarizers. Copies existing results only; never runs a benchmark.
The local directories retain the complete execution proofs and raw artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path


PIPELINE_FILES = (
    "REPORT.md",
    "pipeline-current-release.png",
    "pipeline-confidence.png",
    "pipeline-performance.png",
    "pipeline-complete.png",
    "gpu-model-motion.png",
    "verification-profile/chest-3-summary.json",
    "optix-nearest/research-summary.json",
    "resident-icp/research-summary.json",
    "native-icp-signatures/research-summary.json",
    "uniform-grid-nearest/summary.json",
    "uniform-grid-nearest/research-summary.json",
    "selective-fragment-threads/research-summary.json",
    "uniform-grid-nearest/staged-pruned-v1/research-summary.json",
    "uniform-grid-nearest/flat-v1/research-summary.json",
    "uniform-grid-nearest/device-resident-v1/research-summary.json",
    "grid-lookup-ablation-v1/research-summary.json",
    "grid-lookup-ablation-order-v2/host-diagnostics-summary.json",
    "validation.json",
    "current-release-validation.json",
    "summary-provenance-validation.json",
)
STUDY_FILES = ("REPORT.md", "performance.png")


def publish(source, destination, names):
    source = source.resolve()
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate directory trees")
    # Check the complete explicit list before writing any destination files.
    payloads = {}
    source_records = {}
    for name in names:
        path = source / name
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"Artifact escapes the source directory: {name}")
        data = path.read_bytes()
        if path.suffix == ".json":
            json.loads(data)
        elif path.suffix == ".png":
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Invalid PNG: {name}")
        else:
            data.decode("utf-8")
        source_records[name] = {
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "source_bytes": len(data),
        }
        if path.suffix != ".png":
            data = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        payloads[name] = data
    manifest = {
        "scope": "Selected report snapshots with LF text; original hashes retained; raw captures, build outputs and full execution proofs remain local",
        "source_directory": source.name,
        "files": {},
    }
    for name, data in payloads.items():
        path = destination / name
        if not path.resolve().is_relative_to(destination):
            raise ValueError(f"Artifact escapes the destination directory: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest["files"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            **source_records[name],
        }
    (destination / "report-manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return len(payloads)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, default=root / "benchmark-output/cuda-pipeline")
    parser.add_argument("--study", type=Path, default=root / "benchmark-output/cuda-study")
    parser.add_argument("--destination", type=Path, default=root / "docs/benchmarks")
    args = parser.parse_args()
    pipeline_count = publish(args.pipeline, args.destination / "cuda-pipeline", PIPELINE_FILES)
    study_count = publish(args.study, args.destination / "cuda-study", STUDY_FILES)
    print(json.dumps({"pipeline_files": pipeline_count, "study_files": study_count}))


if __name__ == "__main__":
    main()
