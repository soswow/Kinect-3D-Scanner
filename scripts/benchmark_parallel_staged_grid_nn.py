"""Separate staged/pruned grid and original nine-proposal bridge proof gates.

GPU EXECUTION: only after an exclusive slot is allocated. No production imports
these helpers. Complete audited missing results are correctness experiments;
audit timings cannot support acceleration claims. Default timing keeps original
CPU missing-query fallback. Native CPU bridge is the performance reference.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ["KINECT_CUDA_REGISTRATION"] = "cpu"
EXPECTED_SOURCE = "9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host_enclosure_checks():
    import itertools
    import numpy as np
    from scripts.cuda_parallel_staged_grid_registration import dyadic_shift, host_cells, packed_keys
    rows = []
    for radius in (.0075, .015, .03, .06, .12, .125, .03125, .25, .5, 1., 2.**-20):
        shift = dyadic_shift(radius)
        width = np.ldexp(1., -shift)
        assert width >= radius and width/2 < radius
        tested = accepted = 0
        for cell in (-1048576, -17, -1, 0, 1, 19, 1048575):
            for fraction in (0., .25, .5, np.nextafter(1., 0.)):
                base = (cell+fraction)*width
                target = np.array([[base, 0., 0.]])
                tc = host_cells(target, shift)
                if tc is None:
                    continue
                for sign in (-1., 1.):
                    edge = base+sign*radius
                    for q in (np.nextafter(edge, base), edge, np.nextafter(edge, sign*np.inf)):
                        query = np.array([[q, 0., 0.]])
                        qc = host_cells(query, shift)
                        if qc is None:
                            continue
                        tested += 1
                        squared = np.sum((query-target)*(query-target))
                        if squared < radius*radius:
                            accepted += 1
                            assert np.max(np.abs(tc-qc)) <= 1
        rows.append({"radius": radius, "cell_width": float(width), "supported_boundary_pairs": tested,
                     "strict_hits_inside27cells": accepted, "passed": True})
    corners = np.array(list(itertools.product((-1048576, -1, 0, 1048575), repeat=3)), np.int64)
    keys = packed_keys(corners)
    assert len(np.unique(keys)) == len(keys) and int(keys.min()) == 0 and int(keys.max()) == 2**63-1
    assert host_cells(np.array([[np.inf, 0., 0.]]), 5) is None
    assert host_cells(np.array([[1e7, 0., 0.]]), 5) is None
    assert dyadic_shift(3.) is None and dyadic_shift(2.**-21) is None
    return {"domain_boundary_cases": rows, "packing_unique_cases": len(keys),
            "unsupported_guard_passed": True, "claim": "Finite sample gate plus declared dyadic enclosure argument; complete actual hit/miss CPU audits still required"}


def extra_device_cases(solver):
    import numpy as np
    import open3d as o3d
    rows = []
    for radius in (.03, .06, .12, .125, 2.**-20, 1.):
        from scripts.cuda_parallel_staged_grid_registration import dyadic_shift
        width = np.ldexp(1., -dyadic_shift(radius))
        for cell in (-1048576, -17, -1, 0, 1, 19, 1048575):
            base = cell*width
            targets = np.array([[base, 0., 0.], [base+width*.25, 0., 0.]])
            edge = base+radius
            queries = np.array([[base, 0., 0.], [np.nextafter(base, -np.inf), 0., 0.],
                [np.nextafter(base, np.inf), 0., 0.], [np.nextafter(edge, base), 0., 0.],
                [edge, 0., 0.], [np.nextafter(edge, np.inf), 0., 0.]])
            target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(targets))
            item = solver._dataset(target, radius)
            ids = solver.nearest_indices(queries, item, radius)
            expected = []
            for query in queries:
                count, found, _ = item["cpu"].search_hybrid_vector_3d(query, radius, 1)
                expected.append(found[0] if count else -1)
            mismatch = int(np.count_nonzero(ids != expected))
            rows.append({"case": "dyadic-and-signed-domain-boundary", "radius": radius, "cell": cell,
                         "queries": len(queries), "index_mismatches": mismatch})
            if mismatch:
                raise RuntimeError("Signed-domain/cell-boundary original CPU IDs differ")
    tiny = np.nextafter(np.float64(0.), np.float64(1.))
    targets = np.array([[-tiny, 0., 0.], [tiny, 0., 0.], [np.finfo(np.float64).tiny, 0., 0.]])
    queries = np.array([[0., 0., 0.], [-tiny, 0., 0.], [tiny, 0., 0.], [np.finfo(np.float64).tiny, 0., 0.]])
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(targets))
    item = solver._dataset(target, 2.**-20)
    ids = solver.nearest_indices(queries, item, 2.**-20)
    expected = []
    for query in queries:
        count, found, _ = item["cpu"].search_hybrid_vector_3d(query, 2.**-20, 1)
        expected.append(found[0] if count else -1)
    mismatch = int(np.count_nonzero(ids != expected))
    rows.append({"case": "original-double-subnormal-and-minimum-radius", "queries": len(queries), "index_mismatches": mismatch})
    if mismatch:
        raise RuntimeError("Original subnormal CPU tie/underflow evidence changed")
    return rows


def interval_guard_cases(solver):
    import numpy as np
    import open3d as o3d
    tiny = float(np.finfo(np.float32).tiny)
    quarter = .12*.25
    edge = np.float64(np.float32(1.234567))+quarter
    cases = [
        ("float-normal-minimum-and-adjacent-doubles", np.array([[0.,0.,0.],[tiny,0.,0.],[-tiny,0.,0.]]),
            np.array([[np.nextafter(tiny,0.),0.,0.],[tiny,0.,0.],[np.nextafter(tiny,np.inf),0.,0.]]),2.**-20),
        ("inner-stage-boundary-full-radius-margin",np.array([[edge-quarter,0.,0.]]),
            np.array([[np.nextafter(edge,-np.inf),0.,0.],[edge,0.,0.],[np.nextafter(edge,np.inf),0.,0.]]),.12),
        ("distinct-tie-at-inner-stage-boundary",np.array([[-quarter,0.,0.],[quarter,0.,0.]]),
            np.array([[0.,0.,0.],[1e-16,0.,0.],[-1e-16,0.,0.]]),.12),
        ("negative-large-coordinate-outward-float-bounds",np.array([[-32768.,0.,0.],[-32768.+.02,0.,0.]]),
            np.array([[-32768.+.01,0.,0.],[-32768.+.12,0.,0.],[-32768.+.5,0.,0.]]),.12),
        ("mixed-original-coordinate-below-FLT_MIN-exact-path",np.array([[1e-45,.3,0.],[2e-45,.31,0.]]),
            np.array([[1e-45,.3,0.],[1e-45,.42,0.],[1e-45,3.4,0.]]),.12),
        ("unsupported-full-radius-supported-inner-stages-forced-CPU",np.array([[0.,0.,0.],[1.,0.,0.]]),
            np.array([[0.,0.,0.],[2.,0.,0.],[5.,0.,0.]]),3.),
    ]
    rows=[]
    for name,points,queries,radius in cases:
        target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        item=solver._dataset(target,radius)
        ids=solver.nearest_indices(queries,item,radius)
        expected=[]
        for query in queries:
            count,found,_=item["cpu"].search_hybrid_vector_3d(query,radius,1)
            expected.append(found[0] if count else -1)
        mismatches=int(np.count_nonzero(ids!=expected))
        rows.append({"case":name,"queries":len(queries),"index_mismatches":mismatches})
        if mismatches:
            raise RuntimeError("Staged interval/inner-boundary guard changed original CPU IDs")
    # Inner target table exists, but each query just exceeds its signed domain.
    # This exercises the shared unsupported-stage branch and wider-stage reuse.
    base=np.nextafter(np.float64(32768.),np.float64(0.))
    points=np.array([[base,0.,0.],[base-.01,0.,0.]])
    target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    item=solver._dataset(target,.12)
    from scripts.cuda_parallel_staged_grid_registration import dyadic_shift
    if item["grids"].get(dyadic_shift(.03)) is None:
        raise RuntimeError("Focused unsupported-query guard lacks a present inner target grid")
    queries=np.tile(np.array([[base+offset,0.,0.] for offset in (.01,.04,.1,.2)]),(257,1))
    before=dict(solver.statistics)
    ids=solver.nearest_indices(queries,item,.12)
    expected=[]
    for query in queries:
        count,found,_=item["cpu"].search_hybrid_vector_3d(query,.12,1)
        expected.append(found[0] if count else -1)
    mismatches=int(np.count_nonzero(ids!=expected))
    exercised=(solver.statistics["stage0_rows"]==before["stage0_rows"]
        and solver.statistics["stage1_rows"]>before["stage1_rows"]
        and solver.statistics["stage2_rows"]>before["stage2_rows"])
    rows.append({"case":"inner-query-unsupported-full-stage-supported-uniform-barrier", "queries":len(queries),
                 "index_mismatches":mismatches,"inner_query_unsupported_full_supported":exercised})
    if mismatches or not exercised:
        raise RuntimeError("Unsupported inner-query/wider-stage synchronization guard failed")
    return rows


def close_synthetic_solver(solver):
    primary=sys.exception()
    try:
        solver.close()
    except BaseException as error:
        if primary is None:
            raise
        primary.add_note(f"Secondary synthetic cache cleanup failed: {type(error).__name__}: {error}")


def budget_fallback_case():
    import numpy as np
    import open3d as o3d
    from scripts.cuda_parallel_staged_grid_registration import ParallelStagedUniformGridICP
    solver=ParallelStagedUniformGridICP(max_clouds=1,max_cache_bytes=256,audit_nearest=True,audit_misses=True,
                              miss_policy="direct-miss-research-v1")
    try:
        points=np.random.default_rng(73).normal(size=(32,3))
        target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        item=solver._dataset(target,.12)
        queries=np.concatenate((points[:3],[[10.,10.,10.]]))
        ids=solver.nearest_indices(queries,item,.12)
        expected=[]
        for query in queries:
            count,found,_=item["cpu"].search_hybrid_vector_3d(query,.12,1)
            expected.append(found[0] if count else -1)
        passed=(np.array_equal(ids,expected) and item["cpu_only"] and item["tables"].get(.12) is None
                and solver.statistics["declared_gpu_misses"]==0 and solver.statistics["exact_cpu_queries"]==len(queries)
                and solver.cache_bytes<=256)
        if not passed:
            raise RuntimeError("Incomplete/budget-bypassed stage grids did not use full original CPU radius")
        return {"passed":True,"queries":len(queries),"index_mismatches":0,"retained_bytes":solver.cache_bytes,
                "complete_original_CPU_fallback_queries":solver.statistics["exact_cpu_queries"]}
    finally:
        close_synthetic_solver(solver)


def ownership_guard_cases():
    import numpy as np
    import open3d as o3d
    from scripts.cuda_parallel_staged_grid_registration import ParallelStagedUniformGridICP
    solver=ParallelStagedUniformGridICP(max_clouds=1,max_cache_bytes=4*1024**2,audit_nearest=True,audit_misses=True,
                              miss_policy="direct-miss-research-v1")
    try:
        points=np.array([[0.,0.,0.],[.001,0.,0.],[.002,.001,0.]])
        target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        queries=np.array([[0.,0.,0.],[.002,.001,0.],[.1,.1,0.],[3.,3.,3.]])
        def check(item,radius):
            ids=solver.nearest_indices(queries,item,radius)
            expected=[]
            for query in queries:
                count,found,_=item["cpu"].search_hybrid_vector_3d(query,radius,1)
                expected.append(found[0] if count else -1)
            if not np.array_equal(ids,expected):
                raise RuntimeError("Grid/table eviction/rebuild changed original full-radius CPU evidence")
        for radius in (1.,.5,.25,.125,.0625,.03125,.015625,1.,.015625):
            item=solver._dataset(target,radius)
            check(item,radius)
            if len(item["grids"])>5 or len(item["tables"])>4 or solver.cache_bytes>4*1024**2:
                raise RuntimeError("Per-cloud grid/table or retained-byte bound exceeded")
        # Five distinct radius keys with the same three dyadic shifts exercise
        # pointer-table-cap eviction without any accompanying grid eviction.
        solver._dataset(target,.1)
        grid_evictions=solver.statistics["grid_slot_evictions"]
        table_evictions=solver.statistics["pointer_table_evictions"]
        for radius in (.105,.11,.115,.12):
            item=solver._dataset(target,radius)
            check(item,radius)
        table_only=(solver.statistics["pointer_table_evictions"]>table_evictions
                    and solver.statistics["grid_slot_evictions"]==grid_evictions)
        if not table_only or solver.statistics["peak_tables_per_cloud"]!=4:
            raise RuntimeError("Four-pointer-table cap eviction was not independently exercised")
        item=solver._dataset(target,.12)
        saved=item["tables"][.12]
        cp=solver.cp
        before=dict(solver.statistics)
        with cp.cuda.Device(solver.device_id),cp.cuda.Stream.null:
            host=cp.asnumpy(saved["buffers"][0])
            host[2,0]=0
            bad=cp.asarray(host)
            item["tables"][.12]={**saved,"buffers":(bad,saved["buffers"][1])}
        try:
            check(item,.12)
        finally:
            primary=sys.exception()
            try:
                with cp.cuda.Device(solver.device_id),cp.cuda.Stream.null:
                    cp.cuda.Stream.null.synchronize()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f"Secondary absent-final synchronization failed: {type(error).__name__}: {error}")
            finally:
                item["tables"][.12]=saved
        absent_final=(solver.statistics["exact_cpu_queries"]-before["exact_cpu_queries"]==len(queries)
                      and solver.statistics["declared_gpu_misses"]==before["declared_gpu_misses"])
        if not absent_final:
            raise RuntimeError("Absent original-radius pointer was consumed as an incomplete direct miss")
        check(item,.12)
        other=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points+.01))
        solver._dataset(other,.12)
        # Previously retained item now has no grids/tables; CPU tree still owns
        # its original unchanged points, and must settle all original-radius queries.
        check(item,.12)
        passed=(not item["grids"] and not item["tables"] and solver.statistics["grid_slot_evictions"]>0
                and solver.statistics["pointer_table_evictions"]>0 and solver.statistics["cache_evictions"]>0)
        if not passed:
            raise RuntimeError("Five-grid/table/global-cache ownership paths were not exercised")
        return {"passed":True,"queries_per_radius":len(queries),"radii_tested":9,"absent_final_kernel_guard_passed":absent_final,
                "grid_slot_evictions":solver.statistics["grid_slot_evictions"],
                "pointer_table_evictions":solver.statistics["pointer_table_evictions"],
                "global_cache_evictions":solver.statistics["cache_evictions"],
                "four_table_cap_exercised":table_only,
                "peak_grids_per_cloud":solver.statistics["peak_grids_per_cloud"],
                "peak_tables_per_cloud":solver.statistics["peak_tables_per_cloud"]}
    finally:
        close_synthetic_solver(solver)


def restore_matches(fixture, fragments, module):
    for fragment in fragments.values():
        for view in fragment.keys:
            view.match_cache.clear()
    for index, data in fixture["fragments"].items():
        for original, view in zip(data["keys"], fragments[index].keys):
            for other_fragment in fragments.values():
                for other in other_fragment.keys:
                    matches = original["matches"].get(other.index)
                    if matches is not None:
                        module._cache_matches(view, other, matches.copy())


def load_fixture_scope(args, component_hash):
    fixture_sha, reference_sha = file_hash(args.fixture), file_hash(args.reference)
    with args.fixture.open("rb") as stream:
        fixture = pickle.load(stream)
    reference = json.loads(args.reference.read_text())
    if (fixture["metadata"].get("local_pose_source") != "measured Finish fragment report; no archived ZIP poses"
            or fixture["metadata"]["component_source_sha256"] != component_hash
            or reference["metadata"]["fixture_sha256"] != fixture_sha):
        raise RuntimeError("Require original-raw measured fragment fixture and exact reference/math provenance")
    raw_path = Path(fixture["metadata"]["session"])
    raw_sha = file_hash(raw_path)
    if raw_sha != fixture["metadata"]["input_sha256"]:
        raise RuntimeError("Original raw archive changed")
    tasks = [task for task in fixture["tasks"] if task["position"] in (0,4)]
    if [task["pair"] for task in tasks] != [[8,12],[0,2]] or sum(len(task["proposals"]) for task in tasks) != 9:
        raise RuntimeError("Expected actual accepted/rejected nine-proposal membership")
    binding = {"fixture_sha256": fixture_sha, "reference_sha256": reference_sha,
        "original_raw_input_sha256": raw_sha, "local_pose_source": fixture["metadata"]["local_pose_source"],
        "tasks": [{"position": task["position"], "pair": task["pair"],
                   "proposal_sha256": [hashlib.sha256(pose.tobytes()).hexdigest() for pose in task["proposals"]]} for task in tasks]}
    return fixture, reference, raw_path, tasks, binding


def current_runtime_binding(report, cp, np, o3d):
    from scripts.validate_parallel_staged_grid_proof import canonical_hash
    backend = sys.modules.get(o3d.geometry.PointCloud.__module__.rsplit(".",1)[0])
    numpy_core = sys.modules.get("numpy._core._multiarray_umath")
    if backend is None or not getattr(backend,"__file__",None) or numpy_core is None or not getattr(numpy_core,"__file__",None):
        raise RuntimeError("Installed CPU binary layout cannot be bound to an exact grid proof")
    return {"source_sha256": report["source_sha256"], "component_source_sha256": report["component_source_sha256"],
        "artifacts_sha256": report["artifacts_sha256"], "gpu": report["gpu"], "domain": report["domain"],
        "domain_sha256": canonical_hash(report["domain"]), "versions": report["versions"],
        "thread_policy": report["thread_policy"], "cache_policy": report["cache_policy"],
        "policy": "direct-miss-research-v1", "stage_policy": "parallel-lookup-staged-v1: fractions(.25,.5,1); original-radius uncertainty; restricted directed-FP32 screen; original explicit FP64 final distances",
        "kernel_options": ["--std=c++11","--fmad=false"],
        "cupy_injected_compiler_options": ["-ftz=true; single precision only; original geometry is explicit FP64"],
        "cuda_driver_version": cp.cuda.runtime.driverGetVersion(), "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "nvrtc_version": list(cp.cuda.nvrtc.getVersion()), "cuda_compute_capability": cp.cuda.Device(0).compute_capability,
        "open3d_backend_binary_sha256": file_hash(Path(backend.__file__)),
        "numpy_core_binary_sha256": file_hash(Path(numpy_core.__file__)),
        "numpy_build_configuration_sha256": canonical_hash(getattr(np.__config__,"CONFIG",{})),
        "numpy_cpu_features": {key:bool(value) for key,value in getattr(numpy_core,"__cpu_features__",{}).items()}}


def host_diagnostic_equivalence():
    """Fresh GPU proof of the HOST-only toggle; raw output includes bit payloads."""
    import numpy as np
    import open3d as o3d
    from scripts.cuda_parallel_staged_grid_registration import ParallelStagedUniformGridICP
    rng=np.random.default_rng(738215)
    points=rng.uniform(-.4,.4,(193,3)).astype(np.float64)
    queries=np.r_[points[:47]+rng.uniform(-.02,.02,(47,3)),rng.uniform(-2.,2.,(241,3))]
    quarter=.12*.25
    cases=[("mixed-actual-work-and-misses",points,queries,.12),
        ("inner-boundary-distinct-tie",np.array([[-quarter,0.,0.],[quarter,0.,0.]]),
         np.array([[0.,0.,0.],[1e-16,0.,0.],[-1e-16,0.,0.],[quarter,0.,0.],[1.,0.,0.]]),.12),
        ("inner-unsupported-wider-supported",np.array([[np.nextafter(32768.,0.),0.,0.]]),
         np.tile(np.array([[32768.+.01,0.,0.],[32768.+.04,0.,0.],[32768.+.1,0.,0.],[32768.+.2,0.,0.]]),(17,1)),.12),
        ("signed-normal-and-subnormal-coordinates",np.array([[0.,0.,0.],[1e-45,.3,0.],[-1e-45,.31,0.]]),
         np.array([[0.,0.,0.],[1e-45,.3,0.],[1e-45,.42,0.],[1e-45,3.4,0.]]),2.**-20)]
    solvers=[]
    rows=[]
    primary=None
    try:
        for value in (True,False):
            solvers.append(ParallelStagedUniformGridICP(record_stage_diagnostics=value))
        for name,target_points,query_points,radius in cases:
            target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
            before=(target_points.tobytes(),query_points.tobytes(),np.asarray(target.points).tobytes())
            outputs=[]
            for solver in solvers:
                item=solver._dataset(target,radius)
                with solver.cp.cuda.Device(solver.device_id),solver.cp.cuda.Stream.null:
                    device=solver.cp.asarray(query_points)
                    raw=solver.cp.asnumpy(solver._raw_device(device,item,radius))
                    host=solver.nearest_indices(query_points,item,radius)
                    resident_ids,resident_squared=solver.nearest_device(device,item,radius)
                    device_ids,device_squared=solver.cp.asnumpy(resident_ids),solver.cp.asnumpy(resident_squared)
                expected=[]
                for query in query_points:
                    count,ids,_=item["cpu"].search_hybrid_vector_3d(query,radius,1)
                    expected.append(ids[0] if count else -1)
                if not np.array_equal(host,expected) or not np.array_equal(device_ids,host):
                    raise RuntimeError("Diagnostic toggle changed original CPU host/device IDs")
                outputs.append((raw.copy().view(np.uint64),host,device_ids,device_squared))
            if not all(np.array_equal(a,b) for a,b in zip(outputs[0],outputs[1])):
                raise RuntimeError("HOST diagnostic toggle changed raw output bits or resolved results")
            unchanged=before==(target_points.tobytes(),query_points.tobytes(),np.asarray(target.points).tobytes())
            if not unchanged:
                raise RuntimeError("Diagnostic toggle mutated original arrays")
            rows.append({"case":name,"queries":len(query_points),"raw_output_bits_equal":True,
                "original_cpu_ids_equal":True,"host_device_ids_and_squared_distances_equal":True,
                "arrays_unchanged":True})
        uncollected=[key for key in solvers[1].statistics if key.startswith("stage") or key in ("pruned_candidates","double_evaluations")]
        if uncollected:
            raise RuntimeError("Disabled diagnostics reported uncollected work as numeric counters")
        return {"passed":True,"cases":rows,"both_modes_fresh_gpu_checked":True,
            "kernel_and_seven_column_output_unchanged":True,"disabled_stage_work":"uncollected",
            "collected_statistics":dict(solvers[0].statistics),
            "disabled_statistics":dict(solvers[1].statistics)}
    except BaseException as error:
        primary=error
        raise
    finally:
        cleanup=[]
        for solver in solvers:
            try:
                solver.close()
            except BaseException as error:
                cleanup.append(error)
        if cleanup:
            reason="Diagnostic toggle secondary solver cleanup failed: "+"; ".join(str(error) for error in cleanup)
            if primary is not None:
                primary.add_note(reason)
            else:
                raise RuntimeError(reason) from cleanup[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.fixture.pickle")
    parser.add_argument("--reference", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--audit-nearest", action="store_true")
    parser.add_argument("--audit-misses", action="store_true")
    parser.add_argument("--miss-policy", choices=("cpu-fallback", "direct-miss-research-v1"), default="cpu-fallback")
    parser.add_argument("--check-device-adapter", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cache-mib", type=int, default=256)
    parser.add_argument("--max-clouds", type=int, default=64)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE)
    parser.add_argument("--proof-synthetic", type=Path)
    parser.add_argument("--proof-bridge", type=Path)
    parser.add_argument("--host-stage-diagnostics", choices=("collect","skip"), default="collect")
    args = parser.parse_args()
    if args.repeats < 1 or args.cache_mib < 1 or args.max_clouds < 1:
        parser.error("Require positive repeat/cache bounds")
    if bool(args.proof_synthetic) != bool(args.proof_bridge):
        parser.error("Both separate proof artifacts are required")
    if args.proof_synthetic and (args.synthetic_only or args.miss_policy != "direct-miss-research-v1"):
        parser.error("Proof authority applies to the real fixed-bridge direct-miss experiment")
    if args.miss_policy != "cpu-fallback" and not (args.audit_nearest and args.audit_misses) and not args.proof_synthetic:
        parser.error("Direct misses require both full audits or separately validated current proof artifacts")
    if args.host_stage_diagnostics == "skip" and (args.synthetic_only or args.audit_nearest or args.audit_misses or not args.proof_synthetic):
        parser.error("Skipping host stage decoding requires successful fresh dual-mode proofs and nonaudit real timing")
    if args.output.exists():
        parser.error("Require a fresh output path; preserve existing proof/timing files before rerunning")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from scripts.profile_session import source_hash
    from scripts import benchmark_parallel_fragments as baseline
    source_before, component_before = source_hash(), baseline.numerical_source_hash()
    if source_before != args.expected_source_sha256:
        raise RuntimeError("Require frozen production source before numerical imports")
    artifact_paths = [Path(__file__), ROOT/"scripts/cuda_parallel_staged_grid_registration.py", ROOT/"scripts/research_parallel_staged_grid_nn.cu",
        ROOT/"scripts/PARALLEL_STAGED_GRID_RESEARCH.md", ROOT/"scripts/benchmark_optix_nn.py", ROOT/"scripts/benchmark_parallel_fragments.py",
        ROOT/"scripts/benchmark_selective_fragment_threads.py", ROOT/"scripts/profile_fragment_verification.py",
        ROOT/"scanner_server/cuda_nn_registration.py", ROOT/"scripts/validate_parallel_staged_grid_proof.py"]
    artifacts = {str(path.relative_to(ROOT)): file_hash(path) for path in artifact_paths}
    import cupy as cp
    import cv2
    import numpy as np
    import open3d as o3d
    from scanner_server import fragments as module
    from scripts.cuda_parallel_staged_grid_registration import ParallelStagedUniformGridICP, DOMAIN
    from scripts.benchmark_optix_nn import check_indices, evidence_agreement, canonical_evidence, pair_verdict, gpu_memory_snapshot
    from scripts.benchmark_selective_fragment_threads import fresh_fragments
    from scripts.profile_fragment_verification import array_digest, cloud_digest
    from scripts.process_metrics import finish_cuda_worker, gpu_info, peak_rss_bytes
    from scripts.validate_parallel_staged_grid_proof import validate_grid_proof, canonical_hash
    o3d.utility.set_max_threads(20)
    cv2.setNumThreads(20)
    report = {"kind": "standalone-parallel-staged-pruned-original-double-uniform-grid", "status": "running", "source_sha256": source_before,
        "expected_source_sha256": args.expected_source_sha256, "component_source_sha256": component_before,
        "artifacts_sha256": artifacts, "domain": DOMAIN, "gpu": gpu_info(),
        "performance_attribution_valid": not (args.audit_nearest or args.audit_misses), "miss_policy": args.miss_policy,
        "cpu_hit_audit": args.audit_nearest, "cpu_miss_audit": args.audit_misses,
        "host_diagnostics":{"mode":args.host_stage_diagnostics,"stage_work":"collected" if args.host_stage_diagnostics=="collect" else "uncollected",
            "changed_scope":"Host packed per-stage decode only; original kernel, seven-column output, query/total-visit/search/copy/fallback and NN math retained"},
        "synthetic_only": args.synthetic_only, "synthetic_device_adapter_checked": args.check_device_adapter,
        "thread_policy": {"open3d": 20, "opencv": 20, "omp": os.environ["OMP_NUM_THREADS"]},
        "cache_policy": {"max_clouds": args.max_clouds, "retained_gpu_bytes": args.cache_mib*1024**2,
            "max_grids_per_cloud":5,"max_pointer_tables_per_cloud":4,
            "limit": "Retained grid arrays only; transient launch/transfer buffers and CuPy pools separately observed"},
        "versions": {"numpy": np.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__, "cupy": cp.__version__},
        "authority": "Original raw fixed component with measured local poses only; all nine original proposals/order/gates, no archived live seeds or full Finish/mesh throughput claim",
        "real_runs": []}
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    original_match, solver, failure, fixture, raw_path = module._match, None, None, None, None
    save()
    try:
        report["host_enclosure"] = host_enclosure_checks()
        bindings = current_runtime_binding(report, cp, np, o3d)
        report["proof_bindings"], report["proof_bindings_sha256"] = bindings, canonical_hash(bindings)
        proof_authority = None
        if not args.synthetic_only:
            fixture, reference, raw_path, tasks, fixture_binding = load_fixture_scope(args, component_before)
            report.update(fixture_sha256=fixture_binding["fixture_sha256"], reference_sha256=fixture_binding["reference_sha256"],
                original_raw_input_sha256=fixture_binding["original_raw_input_sha256"], fixture_provenance=fixture["metadata"],
                fixture_binding=fixture_binding, fixture_binding_sha256=canonical_hash(fixture_binding))
            if args.proof_synthetic:
                proof_authority = validate_grid_proof(args.proof_synthetic, args.proof_bridge, bindings, fixture_binding)
                report["validated_proof_authority"] = {"bindings_sha256": proof_authority.bindings_sha256,
                    "fixture_binding_sha256": proof_authority.fixture_binding_sha256,
                    "synthetic_report_sha256": proof_authority.synthetic_report_sha256,
                    "bridge_report_sha256": proof_authority.bridge_report_sha256,
                    "target_membership_count": len(proof_authority.target_digests)}
        started = time.perf_counter()
        solver = ParallelStagedUniformGridICP(max_clouds=args.max_clouds, max_cache_bytes=args.cache_mib*1024**2,
            audit_nearest=args.audit_nearest, audit_misses=args.audit_misses, miss_policy=args.miss_policy,
            proof_authority=proof_authority,record_stage_diagnostics=args.host_stage_diagnostics=="collect")
        report["setup_s"] = time.perf_counter()-started
        report["device"] = str(solver.device)
        report["synthetic_indices"] = check_indices(solver, args.check_device_adapter)
        report["synthetic_grid_boundaries"] = extra_device_cases(solver)
        if proof_authority is None:
            report["synthetic_interval_guards"] = interval_guard_cases(solver)
            report["synthetic_budget_fallback"] = budget_fallback_case()
            report["synthetic_ownership_guards"] = ownership_guard_cases()
            report["synthetic_host_diagnostics_equivalence"] = host_diagnostic_equivalence()
        else:
            report["synthetic_interval_guards"] = []
            report["synthetic_budget_fallback"] = report["synthetic_ownership_guards"] = None
            report["synthetic_host_diagnostics_equivalence"] = None
            report["synthetic_extras_scope"] = "Dedicated unchanged audited proof supplies all staged synthetic coverage; no repeated synthetic GPU work in timing"
        report["synthetic_statistics"] = dict(solver.statistics)
        report["synthetic_gpu_memory"] = gpu_memory_snapshot(solver.device_id)
        solver.clear_cache()
        solver.target_membership_sha256.clear()
        save()
        print("Original CPU synthetic host/device/cell boundary gates passed", flush=True)
        if not args.synthetic_only:
            fixture_arrays_before = array_digest(fixture)
            cpu = fresh_fragments(fixture, baseline, module)
            gpu = fresh_fragments(fixture, baseline, module)
            cloud_before = (cloud_digest(cpu), cloud_digest(gpu))
            reference_proposals = {}
            reference_verdicts = {}
            for mode, repeat in [("native_cpu", 0)]+[("grid", i) for i in range(args.repeats)]:
                packed = cpu if mode == "native_cpu" else gpu
                restore_matches(fixture, packed, module)
                module._match = original_match if mode == "native_cpu" else solver.match
                before = dict(solver.statistics)
                run = {"mode": mode, "repeat": repeat, "pairs": [], "complete": False}
                report["real_runs"].append(run)
                started = time.perf_counter()
                for task in tasks:
                    a,b = task["pair"]
                    values, proposals = [], []
                    pair_started = time.perf_counter()
                    for index, pose in enumerate(task["proposals"]):
                        proposal_started = time.perf_counter()
                        value = module._verify_bridge(packed[a], packed[b], pose, fixture["camera"])
                        values.append(value)
                        quality = None if mode == "native_cpu" else evidence_agreement(reference_proposals[(task["position"],index)],value)
                        proposals.append({"proposal_index": index, "input_sha256": hashlib.sha256(pose.tobytes()).hexdigest(),
                            "elapsed_s": time.perf_counter()-proposal_started, "evidence": canonical_evidence(value), "quality": quality})
                        if mode == "native_cpu":
                            reference_proposals[(task["position"],index)] = value
                        print(f"{mode} repeat{repeat} pair{a}/{b} proposal{index}: {proposals[-1]['elapsed_s']:.3f}s", flush=True)
                    verdict = pair_verdict(values)
                    if mode == "native_cpu":
                        reference_verdicts[task["position"]] = verdict
                    same_verdict = all(verdict[key] == reference_verdicts[task["position"]][key] for key in
                        ("accepted", "ambiguous", "verified_proposals"))
                    run["pairs"].append({"position": task["position"], "pair": task["pair"], "proposal_results": proposals,
                                         "verdict": verdict, "pair_verdict_same": same_verdict, "elapsed_s": time.perf_counter()-pair_started})
                    save()
                run.update(complete=True, elapsed_s=time.perf_counter()-started,
                    statistics_delta={key: solver.statistics[key]-before[key] for key in solver.statistics},
                    gpu_memory_after=gpu_memory_snapshot(solver.device_id))
                save()
            report["fixture_arrays_unchanged"] = fixture_arrays_before == array_digest(fixture)
            report["cloud_arrays_unchanged"] = cloud_before == (cloud_digest(cpu), cloud_digest(gpu))
            original_rows = [row for result in reference["results"] if result["configuration"] == "1x20"
                             for row in result["rows"] if row["position"] in (0,4)]
            native_rows = [{"pair": pair["pair"], "proposal_count": len(pair["proposal_results"]),
                **pair["verdict"]} for pair in report["real_runs"][0]["pairs"]]
            report["original_fixture_authority"] = baseline.compare_rows(original_rows, native_rows)
            report["real_target_membership_sha256"] = sorted(solver.target_membership_sha256)
            report["fixture_sha256_after"], report["reference_sha256_after"] = file_hash(args.fixture), file_hash(args.reference)
            report["original_raw_input_sha256_after"] = file_hash(raw_path)
            if (report["fixture_sha256_after"] != report["fixture_sha256"] or report["reference_sha256_after"] != report["reference_sha256"]
                    or report["original_raw_input_sha256_after"] != report["original_raw_input_sha256"]):
                raise RuntimeError("Fixture/reference bytes changed during measurement")
    except BaseException as exc:
        failure = exc
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        module._match = original_match
        cleanup_failures = []
        def cleanup(name, action):
            try:
                return action()
            except BaseException as exc:
                cleanup_failures.append({"action": name, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
                return None
        report["source_sha256_after"] = cleanup("core source hash", source_hash)
        report["component_source_sha256_after"] = cleanup("math source hash", baseline.numerical_source_hash)
        report["artifacts_sha256_after"] = cleanup("helper artifact hashes", lambda: {str(path.relative_to(ROOT)):file_hash(path) for path in artifact_paths})
        report["gpu_after"] = cleanup("GPU identity", gpu_info)
        if solver is not None:
            report["statistics"] = dict(solver.statistics)
            report["gpu_memory_before_close"] = cleanup("GPU memory snapshot", lambda:gpu_memory_snapshot(solver.device_id))
            cleanup("grid owned cache cleanup", solver.close)
        report["peak_process_rss_bytes"] = cleanup("process RSS", peak_rss_bytes)
        report["cleanup_failures"], report["cleanup_passed"] = cleanup_failures, not cleanup_failures
        passed = (failure is None and not cleanup_failures and report["gpu"] is not None and report["gpu_after"] == report["gpu"]
            and report["source_sha256_after"] == source_before and
            report["component_source_sha256_after"] == component_before and report["artifacts_sha256_after"] == artifacts and
            (args.synthetic_only or (report["fixture_arrays_unchanged"] and report["cloud_arrays_unchanged"] and
                report["original_fixture_authority"]["same_decisions_and_support"] and
                all(run["complete"] and all(pair["pair_verdict_same"] for pair in run["pairs"])
                    and all(p["quality"]["passed"] for pair in run["pairs"] for p in pair["proposal_results"])
                    for run in report["real_runs"] if run["mode"] == "grid"))))
        report["status"] = "passed" if passed else "failed"
        save()
    print(json.dumps({"output": str(args.output), "status": report["status"],
        "runs": [{key: run.get(key) for key in ("mode", "repeat", "elapsed_s")} for run in report["real_runs"]]}), flush=True)
    if not passed:
        raise RuntimeError("Uniform-grid exactness/provenance gates failed; diagnostic report preserved") from failure
    finish_cuda_worker()


if __name__ == "__main__":
    main()
