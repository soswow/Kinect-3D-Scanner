"""Verify the running optimized server, then restore its initial scan settings."""
import json
import sys
import argparse
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
from shared.settings import ScanSettings
from shared.protocol import pack_frames
from tests.test_quality import scene_frames
from scripts.process_metrics import finish_cuda_worker
from scripts.profile_session import file_hash, source_hash
from scripts.http_check_safety import require_idle_empty, restore_settings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--output", type=Path, default=ROOT / "benchmark-output/cuda-pipeline/server-check.json")
parser.add_argument("--require-cuda-confidence", action="store_true")
args = parser.parse_args()
source_before = source_hash()
script_before = file_hash(Path(__file__))
safety_path = Path(__file__).with_name("http_check_safety.py")
safety_before = file_hash(safety_path)
with httpx.Client(base_url=args.url, timeout=180) as client:
    initial_response = client.get("/api/scan/status")
    initial_response.raise_for_status()
    initial_status = initial_response.json()
    require_idle_empty(initial_status)
    if initial_status["backend"].get("cuda_input", {}).get("requested", "off") == "on":
        raise RuntimeError("Synthetic metric input requires KINECT_CUDA_INPUT=off or auto; "
                           "use check_cuda_native_server.py to check explicit native CUDA input")
    initial = initial_status["settings"]
    if args.require_cuda_confidence and initial_status["backend"].get("depth_confidence", {}).get("requested") not in ("auto", "on"):
        raise RuntimeError("This check requires -CudaConfidence auto or on")
    settings = replace(ScanSettings(), rgb_mode="rgb_low_res", confidence_fusion=True,
                       live_reconstruction=False, reconnect_fragments=False,
                       refine_poses=False, color_recovery=True, final_weight=0.5)
    check_session_id = None
    try:
        reset = client.post("/api/scan/reset", json=settings.to_dict())
        reset.raise_for_status()
        check_session_id = reset.json()["session_id"]
        frames = scene_frames(6, settings.camera)
        packet = pack_frames([(rgb, depth, {"frame_id": i, "timestamp_s": i * 0.2,
                                           "rgb_depth_delta_ms": 0})
                              for i, (rgb, depth, _) in enumerate(frames)])
        upload = client.post("/api/scan/frames", content=packet)
        upload.raise_for_status()
        assert upload.json()["batch_size"] == 6, upload.text
        built = client.post("/api/scan/build")
        built.raise_for_status()
        assert built.json()["success"], built.text
        status_response = client.get("/api/scan/status")
        status_response.raise_for_status()
        status = status_response.json()
        assert status["frame_count"] == 6, status
        assert status["backend"]["confidence_cuda"]["implementation"] == "fused", status
        assert status["backend"]["device"] == "CUDA:0"
        assert status["backend"]["tracking"] == "legacy", status
        matching = status["backend"]["descriptor_matching"]
        assert matching["implementation"] == "cuda" and matching["cuda_batches"] > 0, status
        assert status["backend"]["model_preparation"] == "lazy", status
        assert status["backend"]["keyframe_pyramid_cache"] == "on", status
        assert not status["volume_requires_reset"] and not status["input_requires_reset"], status
        assert status["backend"]["fusion"] == "confidence_weighted", status
        if args.require_cuda_confidence:
            confidence = status["backend"]["depth_confidence"]
            assert confidence["implementation"] == "cuda", confidence
            assert confidence["device"] == status["backend"]["device"], confidence
            assert confidence["gpu_calls"] >= 6, confidence
            assert confidence["cpu_calls"] == confidence["fallback_calls"] == 0, confidence
            assert confidence["probe_passed"], confidence
        mesh = client.get("/api/scan/export/ply")
        mesh.raise_for_status()
        assert len(mesh.content) > 1000
        result = {"accepted": status["frame_count"], "mesh_built": True,
                  "ply_bytes": len(mesh.content), "backend": status["backend"]}
    finally:
        restore_settings(client, initial_status, check_session_id)
    result.update(stored_after_check=0, initial_settings_restored=True)
    source_after = source_hash()
    assert source_before == source_after, "Runtime source changed during HTTP check"
    assert script_before == file_hash(Path(__file__)), "HTTP check script changed during execution"
    assert safety_before == file_hash(safety_path), "HTTP check safety helper changed during execution"
    result["metadata"] = {"source_sha256": source_before, "source_sha256_after": source_after,
                          "script_sha256": script_before, "safety_helper_sha256": safety_before,
                          "scope": "synthetic HTTP integration; no scan-speed claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
finish_cuda_worker()
