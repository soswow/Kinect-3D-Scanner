"""Investigate SIFT CUDA latency without changing the scanner implementation.

Compare square-root reduction and GEMM padding on identical quantized measured
descriptors. --tf32 enables an isolated process's CuPy tensor-core compute mode;
integer inputs fit TF32 exactly, and all positive dot products fit float32's
integer range. Every returned match is checked against CPU OpenCV.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import argparse
import json
import os
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
from PIL import Image

from scanner_server.appearance import correspondences, extract_features
from scanner_server.cuda_matching import CODE, Matcher
from scripts.process_metrics import finish_cuda_worker
from scripts.profile_session import file_hash, source_hash
from shared.calibration import prepare_rgbd
from shared.settings import ScanSettings

# Freeze the pre-optimization reduction independently of production defaults.
LEGACY_CODE = CODE.replace("__float_as_uint(d[query*m+j])", "__float_as_uint(sqrtf(d[query*m+j]))")
LEGACY_CODE = LEGACY_CODE.replace("__float_as_uint(d[i*m+j])", "__float_as_uint(sqrtf(d[i*m+j]))")
LEGACY_CODE = LEGACY_CODE.replace(
    "(double)sqrtf(__uint_as_float(first>>32))<0.75*(double)sqrtf(__uint_as_float(second>>32))",
    "(double)__uint_as_float(first>>32)<0.75*(double)__uint_as_float(second>>32)")


class ExperimentalMatcher(Matcher):
    def __init__(self, squared=False, padded=False):
        super().__init__()
        import cupy as cp
        self.padded = padded
        module = cp.RawModule(code=CODE if squared else LEGACY_CODE)
        self.kernels = self.kernels[:4] + tuple(module.get_function(name) for name in ("forward_l2", "backward_l2"))

    def match(self, source, targets):
        import cupy as cp
        a = source.descriptors
        assert a is not None and a.dtype == np.float32 and a.shape[1] == 128
        assert 40 <= len(a) <= 1200 and len(targets) <= 40
        assert all(f.descriptors is not None and 40 <= len(f.descriptors) <= 1200 for f in targets)
        offsets = np.cumsum([0] + [len(f.descriptors) for f in targets], dtype=np.int32)
        n, m, banks = len(a), int(offsets[-1]), len(targets)
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            for feature in targets:
                if id(feature) not in self.cache:
                    self.cache[id(feature)] = feature, cp.asarray(np.ascontiguousarray(feature.descriptors))
            a = cp.asarray(np.ascontiguousarray(a))
            b = cp.concatenate([self.cache[id(feature)][1] for feature in targets])
            if self.padded:
                extra = (-n) % 32
                matrix_a = cp.pad(a, ((0, extra), (0, 0))) if extra else a
                dot = (matrix_a @ b.T)[:n]
            else:
                dot = a @ b.T
            d = cp.maximum((a*a).sum(axis=1)[:, None] + (b*b).sum(axis=1)[None, :] - 2*dot, 0)
            off = cp.asarray(offsets)
            out, back = cp.empty((banks, n), cp.int32), cp.empty(m, cp.int32)
            args = lambda *v: tuple(np.int32(x) for x in v)
            self.kernels[4](((n*banks*32+255)//256,), (256,), (d, off, out, *args(n, m, banks)))
            self.kernels[5](((m+255)//256,), (256,), (d, back, *args(n, m)))
            self.kernels[3](((n*banks+255)//256,), (256,), (out, back, off, *args(n, banks)))
            indices = cp.asnumpy(out)
        return [np.column_stack((np.flatnonzero(row >= 0), row[row >= 0])) for row in indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Require positive repeats")
    os.environ["CUPY_TF32"] = "1" if args.tf32 else "0"
    import cupy as cp
    records = []
    for path in args.sessions:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            settings = ScanSettings.from_dict(manifest["settings"])
            count = len(manifest["frames"])
            indices = sorted(set(np.linspace(0, count-1, 40).astype(int)) | {count//2})
            bank = {}
            for index in indices:
                item = manifest["frames"][index]
                with archive.open(item["rgb"]) as f, Image.open(f) as image:
                    rgb = np.asarray(image.convert("RGB"))
                with archive.open(item["depth"]) as f, Image.open(f) as image:
                    depth = np.asarray(image, dtype=np.uint16)
                rgb, depth = prepare_rgbd(rgb, depth, settings)
                feature = bank[index] = extract_features(rgb, depth, settings.camera, method="sift")
                d = feature.descriptors
                assert d is not None and np.equal(d, np.rint(d)).all() and np.min(d) >= 0 and np.max(d) <= 255
        targets = [bank[i] for i in indices[:40]]
        matchers = {"production": ExperimentalMatcher(), "squared_reduction": ExperimentalMatcher(squared=True),
                    "padded_gemm": ExperimentalMatcher(padded=True),
                    "squared_padded": ExperimentalMatcher(squared=True, padded=True)}
        for index in (0, count//2, count-1):
            source = bank[index]
            expected = [correspondences(source, target) for target in targets]
            for matcher in matchers.values():
                matcher.match(source, targets)
            samples = {name: [] for name in matchers}
            for repeat in range(args.repeats):
                for name in list(matchers)[::1 if repeat % 2 else -1]:
                    cp.cuda.Device().synchronize()
                    started = time.perf_counter()
                    actual = matchers[name].match(source, targets)
                    cp.cuda.Device().synchronize()
                    samples[name].append((time.perf_counter()-started)*1000)
                    for a, b in zip(expected, actual):
                        np.testing.assert_array_equal(a, b)
            row = {"session": path.name, "source_index": index, "source_descriptors": len(source.points),
                   "samples_ms": samples, "median_ms": {name: float(np.median(v)) for name, v in samples.items()},
                   "exact_cpu_matches": True}
            records.append(row)
            print(json.dumps({k: row[k] for k in ("session", "source_index", "median_ms")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source_sha256": source_hash(), "tf32": args.tf32,
        "input_sha256": {path.name: file_hash(path) for path in args.sessions}, "repeats": args.repeats, "rows": records},
        indent=2, allow_nan=False)+"\n")
    cp.cuda.Device().synchronize()
    finish_cuda_worker()


if __name__ == "__main__":
    main()
