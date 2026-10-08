"""Measure bounded, exact CPU confidence reuse on real recorded depth images.

Research component only: no scanner engine, tracking, poses, fusion, or mesh.
Measured reports supply frame identities/order for live, reconnect and Final
confidence calls. Report poses are never read. Imports, ZIP decoding, depth
conversion, reference construction and parity checks are reported separately
from confidence/cache timing. Each lookup includes a depth snapshot, SHA-256,
LRU bookkeeping, and original CPU confidence calculation on a cache miss.

Example (matching session/profile arguments are positional pairs)::

    python scripts/research/archive/benchmark_confidence_cache.py --sessions scan.zip \
      --profiles measured.json --output benchmark-output/confidence-cache.json

An empty-at-Finish case measures reconnect/Final reuse. A live-warmed case
replays the original live accepted IDs before reconnect/Final. Both enforce a
256 MiB retained-cache budget and a session reset. They do not predict scanning
FPS or validate a complete reconstruction. No production cache is installed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import time
import zipfile
from collections import OrderedDict
from dataclasses import asdict



def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_files():
    """Explicit confidence and raw-depth preparation dependency fingerprint."""
    names = ("shared/confidence.py", "shared/calibration.py", "shared/depth.py",
             "shared/settings.py", "shared/sensor_calibration.py", "shared/native.py",
             "shared/capture.py")
    files = {name: file_hash(ROOT / name) for name in names}
    for path in sorted((ROOT / "native").rglob("*")):
        if path.suffix in (".cpp", ".h", ".hpp", ".py", ".toml") and "build" not in path.parts:
            files[str(path.relative_to(ROOT))] = file_hash(path)
    extension = importlib.util.find_spec("_kinect_native")
    if extension and extension.origin:
        files["installed_native_extension:" + extension.origin] = file_hash(extension.origin)
    return files


def depth_settings_signature(settings):
    """Depth/confidence inputs; fusion resolution and camera poses are absent."""
    calibration = settings.sensor_calibration
    return digest_json({"camera": asdict(settings.camera), "near_m": settings.near_m,
        "far_m": settings.far_m, "filter_depth": settings.filter_depth,
        "roi": settings.roi, "depth_encoding": settings.depth_encoding,
        "calibration": None if calibration is None else {
            "document": calibration.document_json, "a": calibration.a_per_code,
            "b": calibration.b, "scale": calibration.scale}})


class ConfidenceCache:
    """Session-owned LRU of immutable copies of original CPU confidence arrays.

    The budget includes retained array payload and Python cache objects. A
    conservative metadata reserve prevents insertion from temporarily exceeding
    that retained budget. Calculation scratch, returned temporary arrays and
    benchmark reference images are outside this cache budget and appear in RSS.
    """

    def __init__(self, cap_bytes, confidence):
        self.cap_bytes, self.confidence = cap_bytes, confidence
        self.entries = OrderedDict()
        self.generation = 0
        self.reset("uninitialized", None, {})

    def reset(self, session, settings, provenance):
        previous = {"entries": len(self.entries), "bytes": self.resident_bytes()}
        self.entries.clear()
        self.payload_bytes = self.peak_payload_bytes = self.peak_resident_bytes = 0
        self.hits = self.misses = self.evictions = self.bypasses = 0
        self.generation += 1
        self.settings = settings
        self.prefix = bytes.fromhex(digest_json({"session": session,
            "generation": self.generation, "provenance": provenance,
            "depth_settings": None if settings is None else depth_settings_signature(settings)}))
        # At 640x480 this leaves space for up to 217 entries within 256 MiB.
        self.metadata_reserve = min(1024 * 1024, self.cap_bytes // 8)
        return previous

    def resident_bytes(self):
        # getsizeof(owning ndarray) includes its owned pixel buffer.
        return (sys.getsizeof(self) + sys.getsizeof(self.__dict__)
                + sys.getsizeof(self.entries) + sys.getsizeof(getattr(self, "prefix", b""))
                + sum(sys.getsizeof(key) + sys.getsizeof(value)
                      for key, value in self.entries.items()))

    def snapshot(self):
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
            "bypasses": self.bypasses, "entries": len(self.entries),
            "payload_bytes": self.payload_bytes, "resident_bytes": self.resident_bytes(),
            "peak_payload_bytes": self.peak_payload_bytes,
            "peak_resident_bytes": self.peak_resident_bytes, "cap_bytes": self.cap_bytes,
            "generation": self.generation}

    def get(self, depth):
        import numpy as np

        camera = self.settings.camera
        if depth.dtype != np.uint16 or depth.shape != (camera.height, camera.width):
            raise ValueError("Require prepared uint16 depth with exact calibrated dimensions")
        # Snapshot first: content hashing never authorizes a later mutated buffer.
        snapshot = np.array(depth, dtype=np.uint16, copy=True, order="C")
        snapshot.flags.writeable = False
        digest = hashlib.sha256(self.prefix)
        digest.update(memoryview(snapshot).cast("B"))
        key = digest.digest()
        if key in self.entries:
            value = self.entries.pop(key)
            self.entries[key] = value
            self.hits += 1
            return value
        self.misses += 1
        value = np.array(self.confidence(snapshot, camera), dtype=np.float32,
                         copy=True, order="C")
        value.flags.writeable = False
        if value.nbytes > self.cap_bytes - self.metadata_reserve:
            self.bypasses += 1
            return value
        while self.entries and self.payload_bytes + value.nbytes > self.cap_bytes - self.metadata_reserve:
            _, old = self.entries.popitem(last=False)
            self.payload_bytes -= old.nbytes
            self.evictions += 1
        self.entries[key] = value
        self.payload_bytes += value.nbytes
        if self.resident_bytes() > self.cap_bytes:
            raise RuntimeError("Retained cache objects exceeded the explicit memory cap")
        self.peak_payload_bytes = max(self.peak_payload_bytes, self.payload_bytes)
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes())
        return value


def sequences(report):
    """Read frame membership only, never a transform or pose estimate."""
    settings = report["settings"]
    if (not report["finish_requested"] or not report["mesh_built"]
            or report["pose_seeds_used"] or report["input_changed_during_profile"]
            or report["source_changed_during_profile"]):
        raise ValueError("Require a completed unchanged live replay with no archived pose seeds")
    if (not settings["confidence_fusion"] or not settings["live_reconstruction"]
            or not settings["reconnect_fragments"]):
        raise ValueError("Require the measured live confidence-fusion workflow")
    if report["refinement"]["applied"] or report["bundle_adjustment"]["applied"]:
        raise ValueError("Intermediate refined fusion identities require a separate trace")
    reconnect = report["fragment_reconnection"]
    if not reconnect["applied"]:
        raise ValueError("Require measured successful fragment reconnect fusion")
    connected = sorted(index for fragment in reconnect["fragments"] if fragment["connected"]
                       for index in fragment["frame_indices"])
    live = report["accepted_indices_before_finish"]
    if live != [row["index"] for row in report["live_diagnostics"] if row["success"]]:
        raise ValueError("Live fusion identities do not match the original live diagnostics")
    final = report["accepted_indices"] if settings["final_voxel_m"] is not None else []
    if final and not report["final_reconstruction"]["applied"]:
        raise ValueError("Requested Final fusion did not complete")
    if connected != report["accepted_indices"] or len(set(connected)) != len(connected):
        raise ValueError("Report does not expose an unambiguous reconnect/Final identity sequence")
    for name, indices in (("live", live), ("reconnect", connected), ("final", final)):
        if indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < report["frames"] for i in indices):
            raise ValueError(f"Invalid measured {name} sequence")
    return [("live", live), ("reconnect", connected), ("final", final)]


def run_sequence(mode, phases, depths, reference, settings, context, cap_bytes):
    import numpy as np
    from shared.confidence import depth_confidence

    setup_started = time.perf_counter()
    cache = None if mode == "uncached" else ConfidenceCache(cap_bytes, depth_confidence)
    if cache is not None:
        cache.reset(context["session"], settings, context)
        if cache.entries or cache.payload_bytes:
            raise RuntimeError("Session reset did not clear the cache")
    setup_ms = (time.perf_counter() - setup_started) * 1000
    results, parity = [], {"changed_weight_bits": 0, "changed_zero_masks": 0,
                          "maximum_absolute_difference": 0.0}
    for name, indices in phases:
        if mode == "cold_finish" and name == "live":
            continue
        before = None if cache is None else cache.snapshot()
        calls, validation_ms = [], 0.0
        wall_started = time.perf_counter()
        for index in indices:
            started = time.perf_counter()
            actual = depth_confidence(depths[index], settings.camera) if cache is None else cache.get(depths[index])
            calls.append((time.perf_counter() - started) * 1000)
            check_started = time.perf_counter()
            expected = reference[index]
            changed = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
            zeros = int(np.count_nonzero((actual == 0) != (expected == 0)))
            parity["changed_weight_bits"] += changed
            parity["changed_zero_masks"] += zeros
            if changed:
                parity["maximum_absolute_difference"] = max(parity["maximum_absolute_difference"],
                                                            float(np.abs(actual - expected).max()))
            if cache is not None and (actual.flags.writeable or not actual.flags.owndata):
                raise RuntimeError("A cached weight image was not an immutable owning copy")
            validation_ms += (time.perf_counter() - check_started) * 1000
        after = None if cache is None else cache.snapshot()
        results.append({"phase": name, "calls": len(calls), "confidence_and_cache_ms": sum(calls),
            "call_p50_ms": statistics.median(calls) if calls else None,
            "call_p95_ms": float(np.percentile(calls, 95)) if calls else None,
            "wall_including_parity_ms": (time.perf_counter() - wall_started) * 1000,
            "parity_validation_ms": validation_ms,
            "cache_delta": None if cache is None else {key: after[key] - before[key]
                for key in ("hits", "misses", "evictions", "bypasses")}, "cache_after": after})
    snapshot = None if cache is None else cache.snapshot()
    reset = None if cache is None else cache.reset(context["session"] + ":reset-check", settings, context)
    if cache is not None and (cache.entries or cache.payload_bytes):
        raise RuntimeError("Final session reset retained derived evidence")
    return {"mode": mode, "setup_ms": setup_ms, "phases": results, "cache": snapshot,
        "reset_cleared": reset, "parity": parity,
        "finish_confidence_and_cache_ms": sum(row["confidence_and_cache_ms"] for row in results if row["phase"] != "live"),
        "total_component_ms": setup_ms + sum(row["confidence_and_cache_ms"] for row in results),
        "passed": not (parity["changed_weight_bits"] or parity["changed_zero_masks"])}


def session_experiment(path, profile_path, args, provenance):
    import numpy as np
    from PIL import Image
    from shared.calibration import prepare_metric_depth
    from shared.confidence import depth_confidence
    from shared.settings import ScanSettings

    archive_before, profile_before = file_hash(path), file_hash(profile_path)
    report = json.loads(profile_path.read_text(encoding="utf-8"))
    if report["input_sha256"] != archive_before:
        raise ValueError("Measured report and input archive SHA-256 differ")
    phases = sequences(report)
    settings = ScanSettings.from_dict(report["settings"])
    depth_contract = depth_settings_signature(settings)
    context = {"session": path.name + ":" + archive_before, "provenance": provenance,
               "depth_settings_signature": depth_contract}
    needed = sorted({i for _, indices in phases for i in indices})
    depths, reference, inputs = {}, {}, []
    prepare_started = time.perf_counter()
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        original_settings = ScanSettings.from_dict(manifest["settings"])
        original_sha = hashlib.sha256(json.dumps(original_settings.to_dict(),
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        if original_sha != report["original_settings_sha256"]:
            raise ValueError("Measured profile did not start from these exact archive settings")
        selected = report["selected_indices"]
        if len(selected) != report["frames"] or len(set(selected)) != len(selected):
            raise ValueError("Invalid measured frame selection")
        if any(type(i) is not int or not 0 <= i < len(manifest["frames"]) for i in selected):
            raise ValueError("Measured original frame index is outside the ZIP")
        for index in needed:
            item = manifest["frames"][selected[index]]
            with archive.open(item["depth"]) as handle, Image.open(handle) as image:
                raw = np.array(image, dtype=np.uint16, copy=True)
            raw.flags.writeable = False
            raw_sha = hashlib.sha256(memoryview(raw).cast("B")).hexdigest()
            depth = np.array(prepare_metric_depth(raw, settings), dtype=np.uint16, copy=True, order="C")
            depth.flags.writeable = False
            depths[index] = depth
            inputs.append({"stored_index": index, "original_index": selected[index],
                "depth_member": item["depth"], "raw_depth_sha256": raw_sha,
                "prepared_depth_sha256": hashlib.sha256(memoryview(depth).cast("B")).hexdigest(),
                "shape": list(depth.shape), "dtype": str(depth.dtype)})
    prepare_ms = (time.perf_counter() - prepare_started) * 1000
    reference_started = time.perf_counter()
    for index in needed:
        value = np.array(depth_confidence(depths[index], settings.camera), dtype=np.float32, copy=True, order="C")
        value.flags.writeable = False
        reference[index] = value
    reference_ms = (time.perf_counter() - reference_started) * 1000
    runs = []
    modes = ("uncached", "cold_finish", "live_warmed")
    for repeat in range(args.repeats):
        order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
        for mode in order:
            print(f"{path.name} repeat={repeat} mode={mode}", flush=True)
            result = run_sequence(mode, phases, depths, reference, settings, context, args.cap_bytes)
            runs.append({"repeat": repeat, **result})
        if dependency_files() != provenance["dependency_files"]:
            raise RuntimeError("Numerical dependencies changed during measurement")
    base = [row["finish_confidence_and_cache_ms"] for row in runs if row["mode"] == "uncached"]
    summary = []
    for mode in modes:
        rows = [row for row in runs if row["mode"] == mode]
        measured = statistics.median(row["finish_confidence_and_cache_ms"] for row in rows)
        saved_ms = statistics.median(base) - measured
        summary.append({"mode": mode, "repeats": len(rows), "median_finish_component_ms": measured,
            "median_complete_component_ms": statistics.median(row["total_component_ms"] for row in rows),
            "finish_component_speedup_vs_uncached": statistics.median(base) / measured,
            "finish_component_saved_ms": saved_ms,
            "report_only_estimated_finish_saved_fraction": saved_ms / (report["finish_s"] * 1000),
            "estimate_warning": "Measured confidence/cache component saving divided by an older full Finish wall time; no cached full replay, fusion or mesh validation."})
    if file_hash(path) != archive_before or file_hash(profile_path) != profile_before:
        raise RuntimeError("Archive or measured profile changed during experiment")
    return {"archive": str(path.resolve()), "input_sha256": archive_before,
        "measured_profile": str(profile_path.resolve()), "profile_sha256": profile_before,
        "profile_source_sha256": report["source_sha256"], "profile_finish_s": report["finish_s"],
        "settings": settings.to_dict(), "settings_sha256": digest_json(settings.to_dict()),
        "confidence_depth_settings_signature": depth_contract,
        "archive_original_settings_sha256": digest_json(original_settings.to_dict()),
        "profile_original_settings_sha256": original_sha,
        "sequences": {name: indices for name, indices in phases},
        "unique_prepared_views": len(needed), "immutable_inputs": inputs,
        "zip_decode_and_depth_preparation_ms": prepare_ms, "reference_construction_ms": reference_ms,
        "reference_image_bytes": sum(value.nbytes for value in reference.values()),
        "prepared_depth_image_bytes": sum(value.nbytes for value in depths.values()),
        "runs": runs, "summary": summary, "passed": all(row["passed"] for row in runs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, nargs="+", required=True)
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cap-mib", type=float, default=256)
    parser.add_argument("--opencv-threads", type=int, default=8)
    args = parser.parse_args()
    if (len(args.sessions) != len(args.profiles) or args.repeats < 1
            or not 1 <= args.cap_mib <= 256 or args.opencv_threads < 1):
        parser.error("Require paired sessions/profiles, positive repeats/threads, and 1..256 MiB cap")
    args.cap_bytes = int(args.cap_mib * 2**20)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("KINECT_NATIVE", "on")
    import cv2
    import numpy as np
    from scripts.process_metrics import peak_rss_bytes
    from scripts.profile_session import source_hash
    from shared.native import native_status

    cv2.setNumThreads(args.opencv_threads)
    before = source_hash()
    def library_info():
        return {"python": platform.python_version(), "numpy": np.__version__,
            "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
            "opencv_ipp_enabled": cv2.ipp.useIPP(), "opencv_ipp_version": cv2.ipp.getIppVersion(),
            "opencv_optimized": cv2.useOptimized(),
            "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()}
    libraries = library_info()
    provenance = {"dependency_files": dependency_files(), "libraries": libraries,
                  "native": native_status()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.with_suffix(".runtime.zip")
    with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ("scanner_server", "shared", "native"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.suffix in (".py", ".cpp", ".cu", ".h", ".hpp", ".toml") and "build" not in path.parts:
                    archive.write(path, str(path.relative_to(ROOT)))
        archive.write(Path(__file__), "scripts/research/archive/benchmark_confidence_cache.py")
    report = {"kind": "research-bounded-cpu-confidence-cache", "script_sha256": file_hash(__file__),
        "runtime_source_sha256": before, "runtime_snapshot": str(snapshot.resolve()),
        "runtime_snapshot_sha256": file_hash(snapshot), "component_dependencies_sha256": digest_json(provenance),
        "provenance": provenance, "memory_cap_bytes": args.cap_bytes,
        "scope": "Original CPU confidence on prepared depth, including per-call immutable snapshot, SHA256, cache lookups/copies/LRU. Setup recorded separately; parity validation and input preparation excluded from confidence timing.",
        "authority": "Measured frame IDs/order only; no poses, scanner, tracking decisions, fusion, or mesh. No production behavior changed.",
        "memory_scope": "256 MiB cap covers retained confidence owning arrays and Python cache objects. Temporary calculation/snapshot images, decoded depth inputs and parity reference arrays are outside cache cap and included in process RSS.",
        "cold_definition": "Empty derived-input cache; installed confidence code and OpenCV are warmed by independent reference construction.",
        "ignored_confidence_settings": ["voxel_m", "final_voxel_m", "truncation_m", "pose", "final_weight", "final_block_count"],
        "sessions": [session_experiment(session, profile, args, provenance)
                     for session, profile in zip(args.sessions, args.profiles)]}
    report.update(runtime_source_unchanged=source_hash() == before,
        dependencies_unchanged=dependency_files() == provenance["dependency_files"],
        script_unchanged=file_hash(__file__) == report["script_sha256"],
        libraries_unchanged=library_info() == libraries,
        peak_process_rss_bytes=peak_rss_bytes())
    if not all(report[key] for key in ("dependencies_unchanged", "script_unchanged", "libraries_unchanged")):
        raise RuntimeError("Numerical dependency/script/library fingerprint changed during experiment")
    report["passed"] = all(row["passed"] for row in report["sessions"])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "passed": report["passed"],
        "sessions": [{"archive": row["archive"], "summary": row["summary"]} for row in report["sessions"]]}, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
