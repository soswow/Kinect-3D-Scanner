"""Fresh original-brute-shadowed comparison of exact CPU/CUDA corner minima.

Allocated offline component experiment only. Native marker extraction/projection
matches analyze_native_markers. CUDA runs its unchanged full original-brute
shadow exactly once per view; the efficient CPU path must then match every ID
and metric bit. No pose, graph, fusion or complete-scan speed authority.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
FROZEN_CORE="9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a"
FROZEN_CUDA="932c381820fe9884a903ca76da823da48a1b4182849c3474885cdbd838bf7b07"
ARTIFACTS=("scripts/research/benchmark_native_corner_lookups.py",
    "scripts/research/native_corner_cpu.py","scripts/research/native_corner_cuda.py",
    "scripts/research/analyze_native_markers.py","shared/calibration.py",
    "shared/settings.py","shared/visual_tracking.py","shared/capture.py","scripts/process_metrics.py")


def file_hash(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024**2),b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def native_binding(np,cv2,cpu,cuda):
    module=sys.modules.get("cv2.cv2")
    if module is None:
        module=sys.modules.get("cv2")
    candidates=[Path(getattr(module,"__file__","")).resolve()]
    package=Path(cv2.__file__).resolve().parent
    candidates += list(package.glob("*.pyd")) + list(package.glob("*.so"))
    cv_binaries={str(path):{"sha256":file_hash(path),"bytes":path.stat().st_size}
        for path in candidates if path.is_file() and path.suffix in (".pyd",".so")}
    if not cv_binaries:
        raise RuntimeError("Cannot bind actual OpenCV native module")
    numpy_binaries={}
    for name,module in list(sys.modules.items()):
        if name.startswith("numpy.") and name.endswith("._multiarray_umath"):
            path=Path(module.__file__).resolve()
            numpy_binaries[str(path)]={"sha256":file_hash(path),"bytes":path.stat().st_size}
    if not numpy_binaries:
        raise RuntimeError("Cannot bind actual NumPy metric binary")
    cpu_metadata={key:value for key,value in cpu.provenance.items() if key!="setup_and_import_s"}
    path=Path(cpu_metadata["ckdtree_binary"]["path"])
    cpu_metadata["ckdtree_binary"]={"path":str(path),"sha256":file_hash(path)}
    if cpu_metadata["ckdtree_binary"]!=cpu.provenance["ckdtree_binary"]:
        raise RuntimeError("Loaded SciPy candidate binary path/bytes changed")
    cp=cuda.cp
    with cp.cuda.Device(cuda.device),cp.cuda.Stream.null:
        props=cp.cuda.runtime.getDeviceProperties(cuda.device)
        cuda_metadata={key:value for key,value in cuda.provenance.items() if key!="setup_and_compile_s"}
        cuda_metadata.update(device=cuda.device,cupy=cp.__version__,
            driver_version=cp.cuda.runtime.driverGetVersion(),runtime_version=cp.cuda.runtime.runtimeGetVersion(),
            device_name=props["name"].decode() if isinstance(props["name"],bytes) else props["name"])
    return {"python":sys.version,"python_executable":sys.executable,"numpy":np.__version__,
        "opencv":cv2.__version__,"opencv_threads":cv2.getNumThreads(),
        "thread_environment":{name:os.environ.get(name) for name in
            ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
        "opencv_binaries":cv_binaries,"numpy_binaries":numpy_binaries,
        "cpu":cpu_metadata,"cuda":cuda_metadata}


def synthetic_cases(np):
    # Identical six input/order cases to frozen NativeCornerCuda.self_check.
    cases=[
        (np.asarray([[0.,-0.],[-0.,0.],[1.,0.],[-1.,0.]],np.float64),
         np.asarray([[0.,0.],[.5,0.],[np.nextafter(.5,1.),0.],[2048.,2048.]],np.float64)),
        (np.asarray([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],np.float64),
         np.asarray([[0.,0.],[np.nextafter(0.,1.),0.],[np.nextafter(0.,-1.),0.]],np.float64)),
        (np.column_stack((np.arange(307200)%640,np.arange(307200)//640)).astype(np.float64),
         np.asarray([[1.,1.],[1.5,1.5],[639.5,479.5]],np.float64))]
    for targets,queries in cases:
        for order in (np.arange(len(targets)),np.arange(len(targets))[::-1]):
            yield queries,np.ascontiguousarray(targets[order])


def compare_lookup(np,cpu,cuda,queries,targets,reverse=False):
    # Reverse individual views to avoid a fixed helper-order advantage; each
    # helper's own timer includes its allocations and copies. The full brute
    # shadow inside CUDA remains measured separately and is never subtracted
    # from this actual audited experiment's total process wall.
    if reverse:
        cpu_i,cpu_d,cpu_record=cpu.query(queries,targets)
        gpu_i,gpu_d,gpu_record=cuda.query(queries,targets)
    else:
        gpu_i,gpu_d,gpu_record=cuda.query(queries,targets)
        cpu_i,cpu_d,cpu_record=cpu.query(queries,targets)
    if (not gpu_record.get("all_queries_shadowed") or gpu_record.get("id_mismatches")!=0
            or gpu_record.get("squared_distance_bit_mismatches")!=0
            or cpu_record.get("all_queries_shadowed") is not False):
        raise RuntimeError("Fresh full original-brute shadow or explicit CPU direct-audit policy missing")
    ids=int(np.count_nonzero(cpu_i!=gpu_i))
    bits=int(np.count_nonzero(cpu_d.view(np.uint64)!=gpu_d.view(np.uint64)))
    if ids or bits:
        raise RuntimeError("Efficient CPU versus original-brute-shadowed CUDA disagreement: IDs=%d bits=%d"%(ids,bits))
    return gpu_i,gpu_d,{"query_rows":len(queries),"target_rows":len(targets),
        "order":["cpu","cuda"] if reverse else ["cuda","cpu"],"cpu_fast":cpu_record,"cuda_original_shadow":gpu_record,
        "cpu_fast_direct_audit":False,"fresh_original_brute_authority_via_cuda":True,
        "all_ids_exact":True,"all_squared_bits_exact":True,"id_mismatches":ids,"squared_bit_mismatches":bits}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions",nargs=3,type=Path,help="chest-5, chest-6, chest-7 original raw ZIPs in that order")
    parser.add_argument("--run-allocated",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not args.run_allocated:
        parser.error("Require root's exclusive allocated hardware slot")
    args.output=args.output.resolve()
    try:
        args.output.relative_to(ROOT/"benchmark-output")
    except ValueError:
        parser.error("Keep private runtime outputs in ignored benchmark-output")
    if args.output.exists():
        parser.error("Preserve previous reports; require a fresh output")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1",8000))==0:
            parser.error("Stop idle field server before allocated component research")
    os.environ.setdefault("OMP_NUM_THREADS","8")
    if os.environ["OMP_NUM_THREADS"]!="8":
        parser.error("Require declared OMP8/OpenCV8/CPU-candidate-workers1 policy")
    from scripts.research.validate_device_flat_grid_proof import current_core_hash
    core=current_core_hash()
    artifacts={name:file_hash(ROOT/name) for name in ARTIFACTS}
    if core!=FROZEN_CORE or artifacts["scripts/research/native_corner_cuda.py"]!=FROZEN_CUDA:
        raise RuntimeError("Require unchanged original frozen core/CUDA brute-shadow helper")
    report={"kind":"exact-native-corner-efficient-cpu-vs-cuda-v1","status":"running",
        "start_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"source_sha256":core,
        "artifacts_sha256":artifacts,"thread_policy":{"opencv":8,"omp":"8","cpu_candidate_workers":1},
        "cpu_fast_direct_audit":False,"actual_gold":"Fresh original full NumPy brute shadow inside unchanged CUDA helper, then every efficient CPU ID and metric bit compared before association",
        "pose_authority":False,"full_scanning_speed_claim":False,"synthetic":[],"sessions":[],
        "scope":"Per-view exact nearest minima only, including out-of-radius IDs. Component helper timers include their documented per-query/per-view work; constructors/setup separately recorded. Decoding/projection/depth support/brute shadows/report work remain outside helper comparison timers and inside actual experiment wall. Alternating helper order; one complete CPU brute shadow per actual view."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    save()
    failure=None
    try:
        import cv2
        import numpy as np
        from PIL import Image
        from shared.settings import ScanSettings
        from shared.calibration import _rectified_depth,_color_maps_numpy,prepare_metric_depth
        from shared.visual_tracking import sampled_points
        from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS
        from scripts.research.native_corner_cpu import NativeCornerCPU
        from scripts.research.native_corner_cuda import NativeCornerCuda
        cv2.setNumThreads(8)
        cv2.setRNGSeed(0)
        if cv2.getNumThreads()!=8:
            raise RuntimeError("OpenCV8 thread policy not applied")
        cpu,cuda=NativeCornerCPU(workers=1,audit=False),NativeCornerCuda(device=0)
        report["constructors"]={"cpu":cpu.provenance,"cuda":cuda.provenance}
        runtime=native_binding(np,cv2,cpu,cuda)
        report["runtime_binding"]=runtime
        for i,(queries,targets) in enumerate(synthetic_cases(np)):
            _,_,row=compare_lookup(np,cpu,cuda,queries,targets,reverse=bool(i%2))
            report["synthetic"].append(row)
        if sum(row["query_rows"] for row in report["synthetic"])!=20:
            raise RuntimeError("Original focused synthetic coverage changed")
        detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
        for session_position,session in enumerate(args.sessions):
            session=session.resolve(strict=True)
            before=file_hash(session)
            item={"archive":str(session),"archive_sha256":before,"complete":False,"views":[]}
            report["sessions"].append(item)
            with zipfile.ZipFile(session) as archive:
                manifest_bytes=archive.read("manifest.json")
                manifest=json.loads(manifest_bytes)
                settings=ScanSettings.from_dict(manifest["settings"])
                if settings.sensor_calibration is None or len(manifest["frames"])!=(27,38,115)[session_position]:
                    raise ValueError("Require original full chest-5/6/7 calibration/view scope")
                item.update(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                    settings_sha256=canonical_hash(settings.to_dict()),
                    calibration_sha256=canonical_hash(manifest["settings"]["sensor_calibration"]))
                for index,frame in enumerate(manifest["frames"]):
                    with archive.open(frame["rgb"]) as stream,Image.open(stream) as image:
                        rgb=np.array(image.convert("RGB"),np.uint8)
                    with archive.open(frame["depth"]) as stream,Image.open(stream) as image:
                        raw=np.array(image,np.uint16)
                    corners,ids,_=detector.detectMarkers(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
                    values=[] if ids is None else ids.ravel().tolist()
                    duplicates={value for value in values if values.count(value)>1}
                    metric=_rectified_depth(raw,settings)
                    depth=prepare_metric_depth(raw,settings)
                    map_x,map_y,visible=_color_maps_numpy(metric,depth,settings,rgb.shape)
                    ys,xs=np.nonzero(visible)
                    projections=np.column_stack((map_x[ys,xs],map_y[ys,xs])).astype(np.float64)
                    query_keys,queries=[],[]
                    for marker,polygon in zip(values,corners):
                        if marker in duplicates:
                            continue
                        for corner,uv in enumerate(np.asarray(polygon).reshape(4,2)):
                            query_keys.append((marker,corner))
                            queries.append(uv)
                    queries=np.asarray(queries,np.float64).reshape(-1,2)
                    row={"index":index,"decoded_ids":values,"duplicate_ids_rejected":sorted(duplicates),
                        "query_keys_sha256":canonical_hash(query_keys),"raw_rgb_sha256":hashlib.sha256(rgb.tobytes()).hexdigest(),
                        "raw_depth_sha256":hashlib.sha256(raw.tobytes()).hexdigest(),
                        "query_rows":len(queries),"target_rows":len(projections),"lookup":None}
                    if len(queries) and len(projections):
                        winners,squared,row["lookup"]=compare_lookup(np,cpu,cuda,queries,projections,reverse=bool(index%2))
                        keys,pixels=[],[]
                        for key,winner,distance in zip(query_keys,winners,squared):
                            if distance<=9.:
                                keys.append(key);pixels.append((int(xs[winner]),int(ys[winner])))
                        reused={pixel for pixel in pixels if pixels.count(pixel)>1}
                        pixels_array=np.asarray(pixels,np.float64).reshape(-1,2)
                        points,valid=sampled_points(depth,pixels_array,settings.camera)
                        for offset,pixel in enumerate(pixels):
                            if pixel in reused:
                                valid[offset]=False
                        lag=frame.get("metadata",{}).get("rgb_depth_delta_ms")
                        if lag is not None and abs(lag)>RGB_DEPTH_ASSISTANCE_LIMIT_MS:
                            valid[:]=False
                        row.update(mapped_corners=len(keys),supported_corners=int(np.count_nonzero(valid)),
                            association_sha256=canonical_hash({"keys":keys,"pixels":pixels,"support":valid.tolist()}),
                            reused_depth_pixels_rejected=len(reused))
                    item["views"].append(row)
                item["archive_sha256_after"]=file_hash(session)
                if item["archive_sha256_after"]!=before:
                    raise RuntimeError("Original field archive changed during component experiment")
            rows=[row["lookup"] for row in item["views"] if row["lookup"] is not None]
            item["summary"]={"actual_queries":sum(row["query_rows"] for row in rows),
                "cpu_fast_all_in_s":sum(row["cpu_fast"]["cpu_map_all_in_s"] for row in rows),
                "gpu_component_s":sum(row["cuda_original_shadow"]["gpu_upload_allocate_lookup_download_s"] for row in rows),
                "original_brute_shadow_s":sum(row["cuda_original_shadow"]["cpu_brute_shadow_s"] for row in rows),
                "cpu_full_brute_fallback_queries":sum(row["cpu_fast"]["full_brute_fallback_queries"] for row in rows),
                "all_ids_and_squared_bits_exact":all(row["all_ids_exact"] and row["all_squared_bits_exact"] for row in rows)}
            item["complete"]=True
            save()
            print(json.dumps({"session":session.name,**item["summary"]}),flush=True)
        if [item["summary"]["actual_queries"] for item in report["sessions"]]!=[1988,984,16308]:
            raise RuntimeError("Fresh full original field corner query scope changed")
        report["runtime_binding_after"]=native_binding(np,cv2,cpu,cuda)
        if report["runtime_binding_after"]!=runtime:
            raise RuntimeError("Actual native libraries/device/thread/config changed")
        report["source_sha256_after"]=current_core_hash()
        report["artifacts_sha256_after"]={name:file_hash(ROOT/name) for name in ARTIFACTS}
        if report["source_sha256_after"]!=core or report["artifacts_sha256_after"]!=artifacts:
            raise RuntimeError("Original math/helper/driver source changed")
        from scripts.process_metrics import peak_rss_bytes
        report["peak_process_rss_bytes"]=peak_rss_bytes()
        report["rss_scope"]="Whole process high-water RSS including imports, both helpers, decoders, native tree and original-brute shadows; not isolated CPU-index memory or a retained byte cap"
        report.update(status="passed",actual_queries=19280,synthetic_queries=20,
            all_actual_queries_original_brute_shadowed=True,all_cpu_fast_ids_and_squared_bits_exact=True)
    except BaseException as error:
        failure=error
        report.update(status="failed",failure={"type":type(error).__name__,"message":str(error)})
        raise
    finally:
        report["end_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            save()
        except BaseException as write_error:
            if failure is not None:
                failure.add_note("Final component diagnostics write also failed: "+str(write_error))
            else:
                raise


if __name__=="__main__":
    main()
