"""Exercise native calibrated CUDA input through HTTP, then restore empty state.

This is an integration smoke check using three raw archived views. It does not
use archived poses or claim reconstruction accuracy/performance for that subset.
"""

import argparse
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from scripts.benchmark_native_kernels import read_frame
from scripts.profile_session import file_hash, source_hash
from scripts.http_check_safety import require_idle_empty, restore_settings
from shared.protocol import pack_frames
from shared.settings import ScanSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path,
        default=ROOT / "benchmark-output/cuda-pipeline/server-native-check.json")
    parser.add_argument("--require-cuda-confidence", action="store_true")
    args = parser.parse_args()
    source_before = source_hash()
    script_before = file_hash(Path(__file__))
    safety_path = Path(__file__).with_name("http_check_safety.py")
    safety_before = file_hash(safety_path)
    input_before = file_hash(args.session)
    with zipfile.ZipFile(args.session) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        settings = replace(ScanSettings.from_dict(manifest["settings"]),
            live_reconstruction=False, reconnect_fragments=False, refine_poses=False,
            bundle_adjustment=False, confidence_fusion=True,
            final_voxel_m=None, final_weight=.5)
        if settings.sensor_calibration is None:
            raise ValueError("Native CUDA smoke check requires a calibrated raw-disparity session")
        views = []
        for index, item in enumerate(manifest["frames"][:3]):
            rgb, depth = read_frame(archive, item)
            metadata = dict(item.get("metadata", {}))
            metadata.update(frame_id=index, timestamp_s=item["timestamp_s"])
            views.append((rgb, depth, metadata))
        if not views:
            raise ValueError("Native CUDA smoke check requires at least one raw archive view")
    with httpx.Client(base_url=args.url, timeout=180) as client:
        initial_response = client.get("/api/scan/status")
        initial_response.raise_for_status()
        initial = initial_response.json()
        require_idle_empty(initial)
        if initial["backend"]["cuda_input"]["requested"] not in ("auto", "on"):
            raise RuntimeError("Native CUDA smoke check requires a server started with -CudaInput auto or on")
        if args.require_cuda_confidence and initial["backend"].get("depth_confidence", {}).get("requested") not in ("auto", "on"):
            raise RuntimeError("This check requires -CudaConfidence auto or on")
        check_session_id = None
        try:
            reset = client.post("/api/scan/reset", json=settings.to_dict())
            reset.raise_for_status()
            check_session_id = reset.json()["session_id"]
            upload = client.post("/api/scan/frames", content=pack_frames(views))
            upload.raise_for_status()
            assert upload.json()["batch_size"] == len(views), upload.text
            built = client.post("/api/scan/build")
            built.raise_for_status()
            assert built.json()["success"], built.text
            status_response = client.get("/api/scan/status")
            status_response.raise_for_status()
            status = status_response.json()
            preparation = status["backend"]["cuda_input"]
            assert 0 < status["frame_count"] <= len(views), status
            assert status["stored_count"] == len(views), status
            assert status["backend"]["device"] == "CUDA:0", status
            assert status["backend"]["fusion"] == "confidence_weighted", status
            assert preparation["requested"] in ("auto", "on"), preparation
            assert preparation["implementation"] == "cuda", preparation
            assert preparation["device"] == status["backend"]["device"], preparation
            assert preparation["gpu_batches"] >= len(views), preparation
            assert preparation["fallback_batches"] == 0, preparation
            assert preparation["cpu_batches"] == 0, preparation
            assert preparation["probe_passed"], preparation
            confidence = status["backend"].get("depth_confidence", {})
            assert preparation["confidence_device"] == confidence.get("device", "CPU:0"), preparation
            if args.require_cuda_confidence:
                assert confidence["implementation"] == "cuda", confidence
                assert confidence["device"] == status["backend"]["device"], confidence
                assert confidence["gpu_calls"] >= max(1, status["frame_count"]), confidence
                assert confidence["cpu_calls"] == confidence["fallback_calls"] == 0, confidence
                assert confidence["probe_passed"], confidence
            assert status["backend"]["confidence_cuda"]["implementation"] == "fused"
            assert not status["volume_requires_reset"] and not status["input_requires_reset"]
            mesh = client.get("/api/scan/export/ply")
            mesh.raise_for_status()
            assert len(mesh.content) > 1000
            result = {"session": args.session.name, "raw_views": len(views),
                "archived_poses_used": False, "accepted": status["frame_count"],
                "mesh_built": True, "ply_bytes": len(mesh.content),
                "backend": status["backend"],
                "scope": "HTTP native-input/fusion/export integration; no subset quality/FPS claim"}
        finally:
            restore_settings(client, initial, check_session_id)
        result.update(stored_after_check=0, initial_settings_restored=True)
        source_after = source_hash()
        assert source_before == source_after, "Runtime source changed during HTTP check"
        assert script_before == file_hash(Path(__file__)), "HTTP check script changed during execution"
        assert safety_before == file_hash(safety_path), "HTTP check safety helper changed during execution"
        assert input_before == file_hash(args.session), "Raw archive changed during HTTP check"
        result["metadata"] = {"source_sha256": source_before, "source_sha256_after": source_after,
                              "script_sha256": script_before, "safety_helper_sha256": safety_before,
                              "input_sha256": input_before}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
