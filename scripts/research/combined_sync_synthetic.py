"""Focused new guard/equation tests, executed only in an allocated GPU slot.

This module imports stdlib only. Each test owns its complete retrieval/cache and
resident scratch; deliberate malformed/fallback cases never affect the main
real-trajectory solver. None of these tests invokes the CPU 6x6 solve.
"""

import hashlib
import sys


REQUIRED_CHECKS = (
    "common_hit_equation_bits", "common_miss_equation_bits", "partial_block_equation_bits",
    "tie_cpu_resolution_before_equations", "strict_boundary_cpu_resolution_before_equations",
    "unsupported_cpu_resolution_before_equations", "malformed_hard_fault_before_cpu",
    "query_cap_before_gpu_launch", "configuration_mutation_before_gpu_launch",
    "owned_scratch_released", "inputs_unchanged")


def run_combined_checks():
    import cupy as cp
    import numpy as np
    import open3d as o3d
    from scripts.research.cuda_device_flat_grid_registration import DeviceFlatGridICP
    from scripts.research.research_combined_sync_icp import CombinedSyncResidentICP, ResidentFallback

    result = {"passed": False, "required_checks": list(REQUIRED_CHECKS), "cases": []}
    for name in REQUIRED_CHECKS:
        result[name] = False

    def forbidden_solve(*_):
        raise RuntimeError("Focused equation proof must not call the CPU solve")

    def digest(value):
        return hashlib.sha256(value.tobytes(order="C")).hexdigest()

    def exercise(name, targets, queries, radius, *, branch, corrupt=False):
        retrieval = None
        resident = None
        primary = None
        row = {"case": name, "queries": len(queries), "complete": False, "expected_branch": branch}
        result["cases"].append(row)
        try:
            retrieval = DeviceFlatGridICP(audit_nearest=True, audit_misses=True,
                miss_policy="direct-miss-research-v1")
            resident = CombinedSyncResidentICP(retrieval, forbidden_solve, gpu_timing=False)
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(targets)
            normals_host = np.tile(np.asarray([0., 0., 1.], dtype=np.float64), (len(targets), 1))
            target.normals = o3d.utility.Vector3dVector(normals_host)
            before = {"target": digest(np.asarray(target.points)), "normals": digest(np.asarray(target.normals)),
                "queries": digest(queries)}
            with cp.cuda.Device(retrieval.device_id), cp.cuda.Stream.null:
                item = retrieval._dataset(target, radius)
                if item["data"] is None:
                    raise RuntimeError("Focused target unexpectedly failed resident cache preparation")
                moving = cp.asarray(queries)
                normals = cp.asarray(normals_host)
                filtered = cp.empty(len(queries), cp.int32)
                partials = cp.empty(((len(queries) + 127) // 128, 30), cp.float64)
                totals = cp.empty(30, cp.float64)
                if corrupt:
                    original_raw = retrieval._raw_device
                    def malformed(*args):
                        values = original_raw(*args)
                        values[0, 0] = cp.nan
                        return values
                    retrieval._raw_device = malformed
                    original_auditor = retrieval.nearest_device
                    def forbidden_auditor(*_):
                        raise RuntimeError("Malformed provisional classification reached CPU audit")
                    retrieval.nearest_device = forbidden_auditor
                try:
                    data = resident._iteration_data(moving, item, normals, filtered, radius, partials, totals)
                    if corrupt:
                        raise RuntimeError("Malformed provisional NN unexpectedly passed")
                    row["strict_correspondence_count"] = data[-1]
                except RuntimeError as error:
                    if not corrupt or "CPU recovery is prohibited" not in str(error):
                        raise
                    row["expected_hard_fault"] = str(error)
                    result["malformed_hard_fault_before_cpu"] = True
                finally:
                    if corrupt:
                        retrieval._raw_device = original_raw
                        retrieval.nearest_device = original_auditor
                cp.cuda.Stream.null.synchronize()
                if digest(cp.asnumpy(moving)) != before["queries"]:
                    raise RuntimeError("Focused equations changed original query coordinates")
            stats = resident.combined_statistics
            row["statistics"] = dict(stats)
            if branch == "common":
                if not (stats["common_calls"] == stats["term_bit_comparison_calls"] == 1
                        and stats["id_bit_comparison_rows"] == stats["metric_bit_comparison_rows"]
                        == stats["filtered_bit_comparison_rows"] == len(queries)
                        and stats["equation_bit_mismatches"] == stats["nearest_bit_mismatches"] == 0):
                    raise RuntimeError("Focused common nearest/equation byte proof incomplete")
                result[name] = True
            elif branch == "flagged":
                if not (stats["flagged_calls"] == stats["discarded_placeholder_calls"] == 1
                        and stats["full_query_audited_rows"] == len(queries)
                        and stats["common_calls"] == stats["term_bit_comparison_calls"] == 0):
                    raise RuntimeError("Focused original CPU resolution/placeholder discard incomplete")
                result[name] = True
            if (digest(np.asarray(target.points)) != before["target"]
                    or digest(np.asarray(target.normals)) != before["normals"] or digest(queries) != before["queries"]):
                raise RuntimeError("Focused original host inputs changed")
            row["inputs_unchanged"] = True
            resident._scratch = None  # Selected stream synchronized above; dispose only these owned buffers.
            row["owned_scratch_released"] = resident._scratch is None
            row["complete"] = True
        except BaseException as error:
            primary = error
            row["failure"] = {"type": type(error).__name__, "message": str(error)}
            error.add_note("Focused combined-sync partial proof: " + repr(result))
            raise
        finally:
            errors = []
            for action in ((resident.close if resident is not None else None),
                           (retrieval.close if retrieval is not None else None)):
                if action is not None:
                    try:
                        action()
                    except BaseException as error:
                        errors.append(error)
            if errors:
                row["cleanup_failures"] = [{"type": type(error).__name__, "message": str(error)} for error in errors]
                if primary is not None:
                    primary.add_note("Focused combined cleanup also failed: " + repr(errors))
                else:
                    raise errors[0]

    try:
        # Original exact doubles, ordinary normals, supported cells, away from ties/radius boundary.
        targets = np.asarray([[0., 0., 0.], [0.5, 0., 0.], [0., .5, 0.]], np.float64)
        exercise("common_hit_equation_bits", targets, np.asarray([[0.001, 0., 0.]], np.float64), .125, branch="common")
        exercise("common_miss_equation_bits", targets, np.asarray([[3., 4., 5.]], np.float64), .125, branch="common")
        exercise("partial_block_equation_bits", targets,
            np.tile(np.asarray([[.001, .002, .003]], np.float64), (129, 1)), .125, branch="common")
        exercise("tie_cpu_resolution_before_equations", np.asarray([[0., 0., 0.], [0., 0., 0.]], np.float64),
            np.asarray([[.001, 0., 0.]], np.float64), .125, branch="flagged")
        exercise("strict_boundary_cpu_resolution_before_equations", np.asarray([[.125, 0., 0.]], np.float64),
            np.asarray([[0., 0., 0.]], np.float64), .125, branch="flagged")
        exercise("unsupported_cpu_resolution_before_equations", targets,
            np.asarray([[1_000_000., 0., 0.]], np.float64), .125, branch="flagged")
        exercise("malformed_hard_fault_before_cpu", targets,
            np.asarray([[.001, 0., 0.]], np.float64), .125, branch="malformed", corrupt=True)

        # Positive source-only flags above are numeric; these two control guards
        # also run against a real constructed selected-device adapter in this job.
        retrieval = DeviceFlatGridICP(audit_nearest=True, audit_misses=True,
            miss_policy="direct-miss-research-v1", max_query_bytes=1)
        resident = None
        try:
            resident = CombinedSyncResidentICP(retrieval, forbidden_solve, gpu_timing=False)
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(np.asarray([[0., 0., 0.]], np.float64))
            try:
                resident._match_resident(cloud, cloud, np.eye(4))
            except ResidentFallback as error:
                if "byte cap exceeded" not in str(error):
                    raise
                result["query_cap_before_gpu_launch"] = resident.combined_statistics["iteration_calls"] == 0
            resident.retrieval.audit_nearest = False
            try:
                resident._check_configuration()
            except RuntimeError as error:
                if "policy changed" not in str(error):
                    raise
                result["configuration_mutation_before_gpu_launch"] = True
            finally:
                resident.retrieval.audit_nearest = True
        finally:
            primary = sys.exception()
            errors = []
            for action in ((resident.close if resident is not None else None), retrieval.close):
                if action is not None:
                    try:
                        action()
                    except BaseException as error:
                        errors.append(error)
            if errors:
                if primary is not None:
                    primary.add_note("Focused limit cleanup also failed: " + repr(errors))
                else:
                    raise errors[0]
        result["owned_scratch_released"] = all(row["owned_scratch_released"] for row in result["cases"])
        result["inputs_unchanged"] = all(row["inputs_unchanged"] for row in result["cases"])
        result["passed"] = all(result[name] is True for name in REQUIRED_CHECKS)
        if not result["passed"]:
            raise RuntimeError("Focused combined-sync proof coverage incomplete")
        return result
    except BaseException as error:
        error.add_note("Combined-sync focused result: " + repr(result))
        raise

