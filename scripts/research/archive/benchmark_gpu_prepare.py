"""Check/timing research CUDA native RGB-D preparation on real archived input.

No scanner engine, volume, pose seeds, or mesh is used. This is an image-stage
experiment, and prepared-image capacity must not be reported as scanner FPS.
The resident result includes raw host uploads, calibrated GPU computation, and
synchronization. The host result additionally downloads color/depth/weights.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import argparse
import json
import os
import time
import zipfile
from dataclasses import replace

os.environ.setdefault("OMP_NUM_THREADS","8")
os.environ.setdefault("KINECT_NATIVE","on")

import cupy as cp
import cv2
import numpy as np

from scripts.benchmark_native_kernels import read_frame
from scripts.process_metrics import gpu_info
from scripts.profile_session import file_hash, source_hash
from scripts.research.archive.research_gpu_prepare import NativeGpuPrepare
from shared.calibration import prepare_rgbd
from shared.confidence import depth_confidence
from shared.native import native_status
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings


def host_result(result):
    return {name:cp.asnumpy(result[name]) for name in ("color","depth","confidence")
            if result[name] is not None}


def cpu_prepare(rgb,raw,settings,confidence):
    colors,depth = prepare_rgbd(rgb,raw,settings)
    return {"color":colors,"depth":depth,
            "confidence":depth_confidence(depth,settings.camera) if confidence else None}


def difference(reference,actual,tolerance):
    colors = np.abs(reference["color"].astype(np.int16)-actual["color"].astype(np.int16))
    depth = np.abs(reference["depth"].astype(np.int32)-actual["depth"].astype(np.int32))
    result = {"color_changed_channels":int(np.count_nonzero(colors)),
        "color_changed_pixels":int(np.count_nonzero(np.any(colors != 0,axis=2))),
        "color_max_difference":int(colors.max()),
        "depth_changed_pixels":int(np.count_nonzero(depth)),"depth_max_difference_mm":int(depth.max())}
    if reference["confidence"] is not None:
        delta = np.abs(reference["confidence"]-actual["confidence"])
        result.update(confidence_max_difference=float(delta.max()),
            confidence_rms_difference=float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
            confidence_zero_mask_changed=int(np.count_nonzero((reference["confidence"]==0)!=(actual["confidence"]==0))),
            confidence_above_tolerance=int(np.count_nonzero(delta>tolerance)))
    result["passed"] = not (result["color_changed_channels"] or result["depth_changed_pixels"] or
        result.get("confidence_zero_mask_changed",0) or result.get("confidence_above_tolerance",0))
    return result


def stats(values):
    values = np.asarray(values,np.float64)
    return {"samples":len(values),"mean_ms":float(values.mean()),
        "p50_ms":float(np.median(values)),"p95_ms":float(np.percentile(values,95)),
        "min_ms":float(values.min()),"max_ms":float(values.max())}


def timed(action):
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    result = action()
    cp.cuda.Stream.null.synchronize()
    return result,(time.perf_counter()-start)*1000


def check_session(path,args):
    before = file_hash(path)
    mismatches, totals, confidence_rms = [], {}, []
    sampled = {}
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        settings = ScanSettings.from_dict(manifest["settings"])
        count = len(manifest["frames"])
        sample_indices = set(np.linspace(0,count-1,min(args.samples,count),dtype=int).tolist())
        setup_started = time.perf_counter()
        prepare = NativeGpuPrepare(settings)
        cp.cuda.Stream.null.synchronize()
        setup_ms = (time.perf_counter()-setup_started)*1000
        cold_ms = None
        indices = range(count) if args.all_frames else sorted(sample_indices)
        for index in indices:
            rgb,raw = read_frame(archive,manifest["frames"][index])
            reference = cpu_prepare(rgb,raw,settings,args.confidence)
            result, elapsed = timed(lambda:prepare.prepare(rgb,raw,confidence=args.confidence))
            if cold_ms is None:
                cold_ms = elapsed
            actual = host_result(result)
            row = difference(reference,actual,args.confidence_tolerance)
            if "confidence_rms_difference" in row:
                confidence_rms.append(row["confidence_rms_difference"])
            for key,value in row.items():
                if isinstance(value,(int,float)) and key not in ("passed","confidence_rms_difference"):
                    totals[key] = max(totals.get(key,0),value) if "max_" in key else totals.get(key,0)+value
            if not row["passed"]:
                mismatches.append({"index":index,**row})
                if len(mismatches)<=3:
                    print(f"{path.stem} index={index} mismatch={json.dumps(row)}",flush=True)
            if index in sample_indices:
                sampled[index] = (rgb,raw)
            if (index+1)%20==0:
                print(f"parity {path.stem} {index+1}/{len(indices)}",flush=True)
    if before != file_hash(path):
        raise ValueError("Archive changed during measurement")
    if confidence_rms:
        totals["confidence_rms_difference"] = float(np.sqrt(np.mean(np.asarray(confidence_rms)**2)))
    # Warm both confidence branches even when the archive requests no weights.
    for rgb,raw in sampled.values():
        cpu_prepare(rgb,raw,settings,args.confidence)
        timed(lambda:prepare.prepare(rgb,raw,confidence=args.confidence))
    rows, durations = [], {"cpu_host":[],"cpu_then_gpu":[],"gpu_resident":[],"gpu_then_host":[]}
    rng = np.random.default_rng(0)
    for repeat in range(args.repeats):
        for index in rng.permutation(sorted(sampled)):
            rgb,raw = sampled[index]
            def cpu_gpu():
                reference = cpu_prepare(rgb,raw,settings,args.confidence)
                return {key:cp.asarray(value) for key,value in reference.items() if value is not None}
            actions = {
                "cpu_host":lambda:cpu_prepare(rgb,raw,settings,args.confidence),
                "cpu_then_gpu":cpu_gpu,
                "gpu_resident":lambda:prepare.prepare(rgb,raw,confidence=args.confidence),
                "gpu_then_host":lambda:host_result(prepare.prepare(rgb,raw,confidence=args.confidence)),
            }
            order = rng.permutation(list(actions))
            row = {"repeat":repeat,"index":int(index)}
            for mode in order:
                _,elapsed = timed(actions[mode])
                row[mode+"_ms"] = elapsed
                durations[mode].append(elapsed)
            rows.append(row)
    timing = {mode:stats(values) for mode,values in durations.items()}
    return {"archive":path.name,"input_sha256":before,"frames":count,
        "parity_frames":len(indices),"all_frame_parity":args.all_frames,
        "settings":settings.to_dict(),"confidence":args.confidence,
        "color_sampler":"continuous_float32_fma" if prepare.interpolation else "opencv_fixed_5bit",
        "confidence_absolute_tolerance":args.confidence_tolerance,
        "passed":not mismatches,"aggregate_differences":totals,"mismatches":mismatches,
        "setup_including_raw_kernel_compile_ms":setup_ms,
        "cold_first_frame_including_confidence_compile_ms":cold_ms,"timing":timing,
        "gpu_resident_speedup_over_cpu_then_gpu":timing["cpu_then_gpu"]["mean_ms"]/timing["gpu_resident"]["mean_ms"],
        "gpu_host_speedup_over_cpu_host":timing["cpu_host"]["mean_ms"]/timing["gpu_then_host"]["mean_ms"],
        "timing_rows":rows}


def check_edge_inputs(path,args):
    """Use actual calibration for disparity sentinel, ROI, and filtering checks."""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        settings = ScanSettings.from_dict(manifest["settings"])
        rgb,raw = read_frame(archive,manifest["frames"][0])
    values = (0,1,500,1000,1500,2046,2047,2048,65535)
    adversarial = np.array(raw,copy=True)
    for index,value in enumerate(values):
        adversarial[index::len(values),:] = value
    cases = [("sentinels",settings,rgb,adversarial),
        ("roi",replace(settings,roi=(73,51,519,421)),rgb,raw),
        ("filter_disabled",replace(settings,filter_depth=False),rgb,raw),
        ("near_far",replace(settings,near_m=.8,far_m=1.5),rgb,raw),
        ("noncontiguous_input",settings,rgb[::-1],raw[::-1])]
    # Reuse the calibrated sample that caught a previous scalar-vs-BLAS color
    # regression. Whole-session equality must not conceal that numeric boundary.
    calibration = load_calibration()
    calibration = replace(calibration,rgb_low_res=replace(calibration.rgb_low_res,cx=306.32208255633054))
    boundary_settings = ScanSettings(sensor_calibration=calibration,rgb_mode="rgb_low_res",filter_depth=False)
    boundary_raw = np.random.default_rng(718).integers(650,851,(480,640),dtype=np.uint16)
    _,x = np.indices((480,640))
    stripes = np.repeat(((x%2)*32)[...,None],3,axis=2).astype(np.uint8)
    cases.append(("blas_color_interpolation_boundary",boundary_settings,stripes,boundary_raw))
    rows = []
    for name,s,color,depth in cases:
        prepare = NativeGpuPrepare(s)
        actual = host_result(prepare.prepare(color,depth,confidence=args.confidence))
        reference = cpu_prepare(color,depth,s,args.confidence)
        rows.append({"case":name,**difference(reference,actual,args.confidence_tolerance)})
    for name,color,depth in (("signed_raw",rgb,raw.astype(np.int16)),
                             ("float_raw",rgb,raw.astype(np.float32)),
                             ("wrong_raw_shape",rgb,raw[:-1]),
                             ("wrong_rgb_shape",rgb[:-1],raw),
                             ("wrong_rgb_dtype",rgb.astype(np.float32),raw)):
        try:
            NativeGpuPrepare(settings).prepare(color,depth)
        except ValueError:
            rows.append({"case":name,"passed":True,"rejected_invalid_input":True})
        else:
            rows.append({"case":name,"passed":False,"rejected_invalid_input":False})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--all-frames",action="store_true")
    parser.add_argument("--confidence",action="store_true")
    parser.add_argument("--confidence-tolerance",type=float,default=2e-5)
    parser.add_argument("--samples",type=int,default=12)
    parser.add_argument("--repeats",type=int,default=3)
    args = parser.parse_args()
    if args.samples<1 or args.repeats<1 or args.confidence_tolerance<=0:
        parser.error("Require positive samples, repeats, and confidence tolerance")
    core_before = source_hash()
    report = {"kind":"standalone-native-rgbd-gpu-preparation",
        "source_sha256":core_before,"script_sha256":file_hash(Path(__file__)),
        "kernel_script_sha256":file_hash(ROOT/"scripts/research/archive/research_gpu_prepare.py"),
        "gpu":gpu_info(),"native":native_status(),"cuda_runtime":cp.cuda.runtime.runtimeGetVersion(),
        "cupy":cp.__version__,"numpy":np.__version__,"opencv":cv2.__version__,
        "opencv_ipp_enabled":cv2.ipp.useIPP(),"opencv_ipp_version":cv2.ipp.getIppVersion(),
        "timing_scope":"Calibrated input preparation only. Raw host-to-device transfer and synchronization included in GPU times; GPU resident outputs avoid downloads. CPU-then-GPU includes prepared-image upload. Host modes return CPU color/depth/confidence. Cached calibration and LUT setup, first JIT compilation, ZIP decoding, tracking, fusion, mesh work and camera/transport excluded.",
        "limitations":"Image parity alone does not validate a mesh or scanning FPS; confidence filters permit reported floating-point tolerance; no server integration.",
        "sessions":[check_session(path,args) for path in args.sessions],
        "edge_cases":check_edge_inputs(args.sessions[0],args)}
    if source_hash()!=core_before:
        raise ValueError("Core source changed during the image-stage experiment")
    report["passed"] = all(row["passed"] for row in report["sessions"]+report["edge_cases"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    for row in report["sessions"]:
        print(json.dumps({"archive":row["archive"],"passed":row["passed"],
            "differences":row["aggregate_differences"],"timing":row["timing"]},indent=2),flush=True)
    print(f"Report {args.output}; passed={report['passed']}",flush=True)
    return 0 if report["passed"] else 1


if __name__=="__main__":
    sys.exit(main())
