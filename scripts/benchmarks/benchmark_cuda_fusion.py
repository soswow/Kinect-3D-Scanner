"""Synchronized recorded-frame fusion timings and canonical voxel parity."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import argparse
import gc
import io
import json
import os
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import numpy as np
import open3d as o3d
from PIL import Image

from scanner_server.engine import ScanEngine
from shared.calibration import prepare_rgbd
from shared.settings import ScanSettings
from scripts.process_metrics import finish_cuda_worker


def synchronize():
    o3d.core.cuda.synchronize()


def snapshot(volume):
    h = volume.hashmap()
    ids = h.active_buf_indices().cpu().numpy().astype(np.int64)
    keys = h.key_tensor().cpu().numpy()[ids]
    order = np.lexsort(keys.T)
    return {"keys": keys[order], **{name: volume.attribute(name).cpu().numpy().reshape(
        -1, 4096, 3 if name == "color" else 1)[ids][order].copy()
        for name in ("tsdf", "weight", "color")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-output/cuda-study/fusion.json")
    args = parser.parse_args()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for session in args.sessions:
        with zipfile.ZipFile(session) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            settings = ScanSettings.from_dict(manifest["settings"])
            samples = [0, len(manifest["frames"]) // 2, len(manifest["frames"]) - 1]
            for index in samples:
                entry = manifest["frames"][index]
                rgb = np.array(Image.open(io.BytesIO(archive.read(entry["rgb"]))).convert("RGB"))
                depth = np.array(Image.open(io.BytesIO(archive.read(entry["depth"]))), dtype=np.uint16)
                rgb, depth = prepare_rgbd(rgb, depth, settings)
                row = {"session": session.name, "index": index, "modes": {}}
                reference = None
                for mode, device in (("cpu-native", "cpu"), ("cuda-tensor", "cuda"), ("cuda-fused", "cuda")):
                    os.environ["KINECT_CUDA_FUSION"] = "fused" if mode == "cuda-fused" else "tensor"
                    engine = ScanEngine(device=device, tracking="legacy")
                    engine.reset(settings=settings)
                    # Two observations exercise prior weights and warm all kernels.
                    for _ in range(2):
                        engine._integrate_vbg(rgb, depth, np.eye(4))
                    synchronize()
                    actual = snapshot(engine.vbg)
                    if reference is None:
                        reference = actual
                    parity = {"same_block_keys": np.array_equal(reference["keys"], actual["keys"])}
                    if parity["same_block_keys"]:
                        for name in ("tsdf", "weight", "color"):
                            parity[name + "_max_abs_delta"] = float(np.max(np.abs(reference[name] - actual[name])))
                            parity[name + "_allclose"] = bool(np.allclose(reference[name], actual[name], atol=2e-6, rtol=2e-6))
                    timings = []
                    for _ in range(args.repeats):
                        for name in ("tsdf", "weight", "color"):
                            engine.vbg.attribute(name)[:] = 0
                        synchronize()
                        start = time.perf_counter()
                        engine._integrate_vbg(rgb, depth, np.eye(4))
                        synchronize()
                        timings.append((time.perf_counter() - start) * 1000)
                    row["modes"][mode] = {"timings_ms": timings, "median_ms": float(np.median(timings)),
                        "parity_to_cpu_native": parity, "active_blocks": engine.vbg.hashmap().size(),
                        "backend": engine.backend}
                    del actual, engine
                    gc.collect()
                    print(f"{session.name} frame {index} {mode}: {np.median(timings):.2f}ms {parity}", flush=True)
                rows.append(row)
                args.output.write_text(json.dumps({"samples": rows}, indent=2, allow_nan=False))
    synchronize()
    gc.collect()
    if sys.platform == "win32":
        finish_cuda_worker()


if __name__ == "__main__":
    main()
