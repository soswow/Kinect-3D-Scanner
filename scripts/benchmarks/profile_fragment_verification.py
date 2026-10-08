"""Profile unchanged CPU bridge verification on an existing original-raw fixture.

This is a standalone component experiment. The fixed fixture contains eight
accepted/rejected fragment pairs and their original ordered proposals, using
local poses from a measured Finish replay solely to reproduce component inputs.
No archived pose is used to seed scanning, and no server policy is modified.

Python wrappers time the original functions and native calls. They do not
replace ICP, nearest-neighbor search, gates, proposals, or witnesses. Optional
proposal/index diagnostics run AFTER fixed verification and do not authorize
any result. Native Open3D wall time includes opaque tree construction, searches,
linear solves and convergence checks; these costs cannot be separated here.
Configured iteration limits are not actual iteration counts.

Example (requires an exclusive hardware slot):
  python scripts/benchmarks/profile_fragment_verification.py --threads 20 \
    --fixture benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.fixture.pickle \
    --reference benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.json \
    --proposal-diagnostic --target-index-diagnostic \
    --output benchmark-output/cuda-pipeline/verification-profile/chest-3.json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import pickle
import time
from collections import defaultdict
from dataclasses import fields, is_dataclass

FROZEN_SOURCE_SHA256 = "9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a"
FAILURE_STATE = {}
QUALITY_GATES = ("same_decisions_and_support", "same_pair_order", "same_positions",
                 "fixture_arrays_unchanged", "verification_clouds_unchanged", "source_unchanged",
                 "component_source_unchanged", "profiler_script_unchanged", "fixture_loader_script_unchanged",
                 "fixture_file_unchanged", "reference_file_unchanged")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ["KINECT_CUDA_REGISTRATION"] = "cpu"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Recorder:
    """Nested inclusive/self wall times; self rows partition instrumented work."""

    def __init__(self, max_events=100000):
        self.events = []
        self.stack = []
        self.clouds = {}
        self.phase = "setup"
        self.pair_position = self.proposal_index = None
        self.max_events = max_events
        self.next_id = 0
        self.observed_targets = {}

    def cloud(self, cloud):
        return self.clouds.get(id(cloud), {"id": "unregistered", "points": len(cloud.points)})

    def register(self, cloud, label, **details):
        self.clouds[id(cloud)] = {"id": label, "points": len(cloud.points), **details}

    @contextlib.contextmanager
    def span(self, name, **details):
        row = {"event_id": self.next_id, "parent_id": self.stack[-1]["event_id"] if self.stack else None,
               "name": name, "phase": self.phase, "pair_position": self.pair_position,
               "proposal_index": self.proposal_index, **details}
        self.next_id += 1
        self.stack.append(row)
        children_s = 0.0
        row["_children_s"] = children_s
        started = time.perf_counter()
        try:
            yield row
        except BaseException as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            elapsed = time.perf_counter() - started
            popped = self.stack.pop()
            if popped is not row:
                raise RuntimeError("Profiler scope stack changed")
            row["inclusive_s"] = elapsed
            row["self_s"] = max(0.0, elapsed - row.pop("_children_s"))
            if self.stack:
                self.stack[-1]["_children_s"] += elapsed
            if len(self.events) >= self.max_events:
                raise RuntimeError("Profiler event bound exceeded; measurement is incomplete")
            self.events.append(row)

    def scope(self):
        return "/".join(row["name"] for row in self.stack)


def result_details(result):
    """Inspect only exposed result fields, never derive iteration counts."""
    details = {}
    for key in ("fitness", "inlier_rmse"):
        if hasattr(result, key):
            details[key] = float(getattr(result, key))
    if hasattr(result, "correspondence_set"):
        details["correspondences"] = len(result.correspondence_set)
    exposed = {}
    for key in ("num_iterations", "iteration_count", "iterations"):
        value = getattr(result, key, None)
        if isinstance(value, int):
            exposed[key] = value
    details["exposed_iteration_fields"] = exposed
    return details


class RegistrationProxy:
    NAMES = {"registration_icp", "evaluate_registration", "compute_fpfh_feature",
             "get_information_matrix_from_point_clouds", "registration_ransac_based_on_feature_matching"}

    def __init__(self, original, recorder):
        self.original, self.recorder = original, recorder
        self.wrappers = {}

    def __getattr__(self, name):
        original = getattr(self.original, name)
        if name not in self.NAMES:
            return original
        if name not in self.wrappers:
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                rec = self.recorder
                details = {"branch_path": rec.scope()}
                if args:
                    details["source"] = rec.cloud(args[0])
                if name == "compute_fpfh_feature":
                    param = args[1]
                    details.update(radius_m=param.radius, max_neighbors=param.max_nn)
                elif len(args) >= 2:
                    details["target"] = rec.cloud(args[1])
                    rec.observed_targets[id(args[1])] = args[1]
                if name in ("registration_icp", "evaluate_registration", "get_information_matrix_from_point_clouds"):
                    details["max_correspondence_distance_m"] = float(args[2])
                if name == "registration_icp":
                    criteria = args[5] if len(args) > 5 else kwargs.get("criteria")
                    if criteria is not None:
                        details["configured_max_iterations"] = int(criteria.max_iteration)
                        details["relative_fitness"] = float(criteria.relative_fitness)
                        details["relative_rmse"] = float(criteria.relative_rmse)
                    details["direction"] = "unspecified"
                    pair = next((item for item in reversed(rec.stack) if item["name"] == "pair"), None)
                    if pair is not None:
                        details["direction"] = ("forward" if details["source"]["id"] == pair["source"]["id"] else "reverse")
                    details["normal_estimator"] = "original point-to-plane with HuberLoss(0.01)"
                if name == "registration_ransac_based_on_feature_matching":
                    details["max_correspondence_distance_m"] = float(args[5])
                    details["configured_max_iterations"] = int(args[9].max_iteration)
                    details["configured_confidence"] = float(args[9].confidence)
                with rec.span("native." + name, **details) as row:
                    result = original(*args, **kwargs)
                # Reading exposed attributes is outside the native wall timer.
                row.update(result_details(result))
                return result
            self.wrappers[name] = wrapped
        return self.wrappers[name]


@contextlib.contextmanager
def instrumentation(recorder, fragments, refinement, appearance, cv2):
    """Temporarily redirect module globals; all numerical functions stay original."""
    replacements = []

    def replace(module, name, value):
        replacements.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    proxy = RegistrationProxy(fragments.REG, recorder)
    replace(fragments, "REG", proxy)
    replace(refinement, "REG", proxy)
    names = {"_verify_bridge": "bridge", "_verify_partial_bridge": "partial_bridge",
             "_verify_visual_bridge": "visual_bridge", "_pair": "pair", "_match": "match",
             "_heldout": "heldout", "_visual_witness": "visual_witness", "_matches": "descriptor_matches",
             "_strong": "strong_gate", "_global_seed": "fpfh_ransac_proposal"}
    for function_name, label in names.items():
        original = getattr(fragments, function_name)

        def make_wrapper(original, function_name, label):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                details = {}
                if function_name in ("_pair", "_match", "_heldout"):
                    details.update(source=recorder.cloud(args[0]), target=recorder.cloud(args[1]))
                    details["cloud_scope"] = ("fragment_union" if ":union:" in details["source"]["id"] else "camera")
                    if function_name in ("_pair", "_heldout"):
                        details["minimum_overlap"] = float(args[3] if len(args) > 3 else kwargs.get("minimum", .5))
                elif function_name in ("_matches", "_visual_witness"):
                    details.update(source_view_index=args[0].index, target_view_index=args[1].index)
                elif function_name == "_global_seed":
                    details.update(seed=int(args[2]), source_fragment=args[0].index, target_fragment=args[1].index)
                with recorder.span(label, **details) as row:
                    result = original(*args, **kwargs)
                if function_name in ("_heldout", "_visual_witness"):
                    row["gate_passed"] = bool(result[0])
                elif function_name == "_strong":
                    row["gate_passed"] = bool(result)
                elif function_name == "_matches":
                    row["match_count"] = len(result)
                else:
                    row["result_present"] = result is not None
                if isinstance(result, dict) and "validation_scope" in result:
                    row["validation_scope"] = result["validation_scope"]
                return result
            return wrapped
        replace(fragments, function_name, make_wrapper(original, function_name, label))

    original_propose = appearance.propose_transform
    @functools.wraps(original_propose)
    def propose(*args, **kwargs):
        with recorder.span("rgb_pnp_proposal", source_features=len(args[0].points),
                           target_features=len(args[1].points)) as row:
            result = original_propose(*args, **kwargs)
        row["result_present"] = result is not None
        return result
    replace(appearance, "propose_transform", propose)
    replace(fragments, "propose_transform", propose)

    original_pnp = cv2.solvePnPRansac
    @functools.wraps(original_pnp)
    def pnp(*args, **kwargs):
        with recorder.span("native.solvePnPRansac", matches=len(args[0]),
                           configured_max_iterations=kwargs.get("iterationsCount"),
                           reprojection_error_px=kwargs.get("reprojectionError"),
                           configured_confidence=kwargs.get("confidence")) as row:
            result = original_pnp(*args, **kwargs)
        row["returned_success"] = bool(result[0])
        row["inliers"] = 0 if result[3] is None else len(result[3])
        row["actual_iterations"] = None
        return result
    replace(cv2, "solvePnPRansac", pnp)
    try:
        yield
    finally:
        for module, name, original in reversed(replacements):
            setattr(module, name, original)


def summaries(events):
    groups = defaultdict(lambda: {"calls": 0, "inclusive_s": 0., "self_s": 0., "configured_iteration_budget_sum": 0})
    for row in events:
        group = groups[(row["phase"], row["name"])]
        group["calls"] += 1
        group["inclusive_s"] += row["inclusive_s"]
        group["self_s"] += row["self_s"]
        group["configured_iteration_budget_sum"] += row.get("configured_max_iterations") or 0
    return [{"phase": phase, "name": name, **value} for (phase, name), value in sorted(groups.items())]


def timer_accounting(events):
    """Inclusive times overlap; only self times can be added across levels."""
    phases = defaultdict(lambda: {"root_inclusive_s": 0., "instrumented_self_s": 0., "events": 0})
    for row in events:
        phase = phases[row["phase"]]
        phase["instrumented_self_s"] += row["self_s"]
        phase["events"] += 1
        if row["parent_id"] is None:
            phase["root_inclusive_s"] += row["inclusive_s"]
    return {"rule": "Nested inclusive_s overlaps and must not be added. self_s subtracts timed children and partitions instrumented work; native self time still includes opaque C++ substeps. External loop/report/checksum overhead is outside root timers.",
            "by_phase": [{"phase": name, **values,
                          "partition_residual_s": values["root_inclusive_s"] - values["instrumented_self_s"]}
                         for name, values in sorted(phases.items())]}


def native_breakdown(events):
    groups = defaultdict(lambda: {"calls": 0, "inclusive_s": 0., "self_s": 0.,
                                   "configured_iteration_budget_sum": 0})
    for row in events:
        if row["phase"] != "fixed_verification" or not row["name"].startswith("native."):
            continue
        source = row.get("source", {})
        scope = "fragment_union" if ":union:" in source.get("id", "") else "camera"
        branch = ("partial_overlap" if "/partial_bridge/" in row.get("branch_path", "") else
                  "visual" if "/visual_bridge/" in row.get("branch_path", "") else "direct")
        key = (row["name"], scope, source.get("kind"), row.get("direction"), branch)
        group = groups[key]
        group["calls"] += 1
        group["inclusive_s"] += row["inclusive_s"]
        group["self_s"] += row["self_s"]
        group["configured_iteration_budget_sum"] += row.get("configured_max_iterations") or 0
    return [{"name": key[0], "cloud_scope": key[1], "cloud_kind": key[2],
             "direction": key[3], "branch": key[4], **value}
            for key, value in sorted(groups.items(), key=lambda item: str(item[0]))]


def write_compact_summary(report, output):
    """Tracked summary excludes event traces and all pose/information matrices."""
    compact = {"status": report.get("status", "failed"), "metadata": report["metadata"],
               "quality": report.get("quality"), "verification_s": report.get("verification_s"),
               "total_wall_s": report.get("total_wall_s"), "summaries": report.get("summaries", []),
               "timer_accounting": report.get("timer_accounting"),
               "verification_native_call_breakdown": native_breakdown(report.get("events", [])),
               "immutable_target_native_icp_uses": report.get("immutable_target_native_icp_uses", []),
               "target_index_diagnostic": report.get("target_index_diagnostic", []),
               "proposal_diagnostic": report.get("proposal_diagnostic", []),
               "pairs": [{key: row[key] for key in ("position", "pair", "proposal_count", "verified_proposals",
                                                       "ambiguous", "accepted", "elapsed_s")}
                         for row in report.get("rows", [])],
               "failure": report.get("failure")}
    summary_path = output.with_name(output.stem + "-summary.json")
    summary_path.write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n")
    return summary_path


def preserve_failure(error):
    """Keep completed rows/events even if source, native work or authority fails."""
    import traceback
    path = FAILURE_STATE.get("output")
    if path is None:
        return
    recorder = FAILURE_STATE.get("recorder")
    events = [] if recorder is None else sorted(recorder.events, key=lambda row: row["event_id"])
    report = {**FAILURE_STATE.get("completed_report", {}),
              "status": "failed", "metadata": FAILURE_STATE.get("metadata", {}),
              "failure": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
              "rows": FAILURE_STATE.get("rows", []), "quality": FAILURE_STATE.get("quality"),
              "proposal_diagnostic": FAILURE_STATE.get("proposal_diagnostic", []),
              "target_index_diagnostic": FAILURE_STATE.get("target_index_diagnostic", []),
              "events": events, "summaries": summaries(events), "timer_accounting": timer_accounting(events)}
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [finite(item) for item in value]
        return value
    failure_path = path.with_suffix(".failure.json")
    safe_report = finite(report)
    failure_path.write_text(json.dumps(safe_report, indent=2, allow_nan=False) + "\n")
    write_compact_summary(safe_report, path)
    print(f"Failure diagnostics preserved: {failure_path}", file=sys.stderr, flush=True)


def array_digest(fixture):
    """Guard original fixture arrays; hashing is outside timed verification."""
    import numpy as np
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif is_dataclass(value):
            for field in fields(value):
                digest.update(field.name.encode())
                visit(getattr(value, field.name))
    visit(fixture)
    return digest.hexdigest()


def cloud_digest(fragments):
    import numpy as np
    digest = hashlib.sha256()
    for index, fragment in sorted(fragments.items()):
        clouds = [fragment.train, fragment.heldout]
        clouds.extend(cloud for view in fragment.keys for cloud in (view.train, view.heldout))
        for cloud in clouds:
            for name in ("points", "normals", "colors"):
                array = np.asarray(getattr(cloud, name))
                digest.update(str((index, name, array.shape, array.dtype.str)).encode())
                digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--positions", type=int, nargs="+")
    parser.add_argument("--proposal-diagnostic", action="store_true")
    parser.add_argument("--target-index-diagnostic", action="store_true")
    parser.add_argument("--diagnostic-pnp-seed", type=int, default=20261008)
    parser.add_argument("--max-events", type=int, default=100000)
    parser.add_argument("--expected-source-sha256", default=FROZEN_SOURCE_SHA256,
                        help="Require this frozen production source before importing numerical libraries")
    args = parser.parse_args()
    if args.threads < 1 or args.max_events < 1:
        parser.error("Require positive thread/event bounds")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    from scripts.research import benchmark_parallel_fragments as baseline
    from scripts.process_metrics import finish_cuda_worker, peak_rss_bytes
    from scripts.profile_session import source_hash
    source_before = source_hash()
    component_before = baseline.numerical_source_hash()
    script_before = file_hash(Path(__file__))
    loader_before = file_hash(Path(baseline.__file__))
    reference_before = file_hash(args.reference)
    FAILURE_STATE.update(output=args.output, metadata={"expected_source_sha256": args.expected_source_sha256,
        "source_sha256": source_before, "component_source_sha256": component_before,
        "script_sha256": script_before, "fixture_loader_script_sha256": loader_before,
        "reference_sha256": reference_before, "fixture": str(args.fixture.resolve()),
        "reference": str(args.reference.resolve())})
    if source_before != args.expected_source_sha256:
        raise RuntimeError("Production source does not match the requested frozen implementation")
    import cv2
    import numpy as np
    import open3d as o3d
    from scanner_server import appearance, fragments as module, refinement
    o3d.utility.set_max_threads(args.threads)
    cv2.setNumThreads(args.threads)
    rec = Recorder(args.max_events)
    FAILURE_STATE["recorder"] = rec
    fixture_sha = file_hash(args.fixture)
    FAILURE_STATE["metadata"]["fixture_sha256"] = fixture_sha
    reference = json.loads(args.reference.read_text())
    expected_sha = reference["metadata"].get("fixture_sha256")
    if expected_sha and fixture_sha != expected_sha:
        raise RuntimeError("Reference fixture hash differs")
    with rec.span("fixture_load"):
        with args.fixture.open("rb") as stream:
            fixture = pickle.load(stream)
    if fixture["metadata"]["component_source_sha256"] != component_before:
        raise RuntimeError("Original numerical verification dependencies changed")
    fixture_arrays_before = array_digest(fixture)
    reference_result = next(result for result in reference["results"] if result["configuration"] == "1x20")
    if args.positions is not None:
        if len(set(args.positions)) != len(args.positions) or not set(args.positions).issubset(
                {task["position"] for task in fixture["tasks"]}):
            raise ValueError("Requested positions are duplicated or absent")
    tasks = [task for task in fixture["tasks"] if args.positions is None or task["position"] in args.positions]
    wanted = {task["position"] for task in tasks}
    reference_rows = [row for row in reference_result["rows"] if row["position"] in wanted]
    with rec.span("cloud_unpack_and_match_cache"):
        fragments = {index: baseline.unpack_fragment(data) for index, data in fixture["fragments"].items()}
        for index, data in fixture["fragments"].items():
            fragment = fragments[index]
            for name in ("train", "heldout"):
                rec.register(getattr(fragment, name), f"fragment:{index}:union:{name}", kind=name)
            for original, view in zip(data["keys"], fragment.keys):
                for name in ("train", "heldout"):
                    rec.register(getattr(view, name), f"fragment:{index}:view:{view.index}:{name}", kind=name)
                for other_fragment in fragments.values():
                    for other in other_fragment.keys:
                        matches = original["matches"].get(other.index)
                        if matches is not None:
                            module._cache_matches(view, other, matches.copy())
    clouds_before = cloud_digest(fragments)

    rows, proposal_diagnostics, target_indexes = [], [], []
    FAILURE_STATE.update(rows=rows, proposal_diagnostic=proposal_diagnostics,
                         target_index_diagnostic=target_indexes)
    verification_started = time.perf_counter()
    with instrumentation(rec, module, refinement, appearance, cv2):
        rec.phase = "fixed_verification"
        for task in tasks:
            rec.pair_position, rec.proposal_index = task["position"], None
            a, b = task["pair"]
            verified, proposals = [], []
            with rec.span("fixture_pair", pair=task["pair"]) as pair_event:
                for position, pose in enumerate(task["proposals"]):
                    rec.proposal_index = position
                    with rec.span("fixture_proposal") as event:
                        result = module._verify_bridge(fragments[a], fragments[b], pose, fixture["camera"])
                    proposals.append({"proposal_index": position,
                                      "initial_sha256": hashlib.sha256(pose.tobytes()).hexdigest(),
                                      "result": baseline.json_value(result), "elapsed_s": event["inclusive_s"]})
                    if result is not None:
                        verified.append(result)
                ambiguous = bool(verified) and any(module._disagrees(verified[0]["transform"], item["transform"])
                                                   for item in verified[1:])
                best = verified[0] if verified and not ambiguous else None
            rows.append({"position": task["position"], "pair": task["pair"],
                         "proposal_count": len(task["proposals"]), "verified_proposals": len(verified),
                         "ambiguous": ambiguous, "accepted": best is not None,
                         "best": baseline.json_value(best), "elapsed_s": pair_event["inclusive_s"],
                         "proposal_results": proposals})
            print(f"Fixed pair {a}/{b}: {pair_event['inclusive_s']:.3f}s, accepted={best is not None}", flush=True)
        verification_s = time.perf_counter() - verification_started
        quality = baseline.compare_rows(reference_rows, rows)
        FAILURE_STATE["quality"] = quality
        quality["same_pair_order"] = [row["pair"] for row in rows] == [row["pair"] for row in reference_rows]
        quality["same_positions"] = [row["position"] for row in rows] == [row["position"] for row in reference_rows]

        if args.target_index_diagnostic:
            # This constructor benchmark is separate from the opaque native calls.
            targets = list(rec.observed_targets.values())
            rec.phase, rec.pair_position, rec.proposal_index = "target_index_diagnostic", None, None
            for target in targets:
                with rec.span("standalone.KDTreeFlann", target=rec.cloud(target)) as event:
                    tree = o3d.geometry.KDTreeFlann(target)
                target_indexes.append({"target": rec.cloud(target), "construction_s": event["inclusive_s"]})
                del tree

        if args.proposal_diagnostic:
            rec.phase, rec.pair_position, rec.proposal_index = "proposal_preparation_diagnostic", None, None
            used = sorted({index for task in tasks for index in task["pair"]})
            for index in used:
                fragment = fragments[index]
                with rec.span("coarse_and_fpfh", fragment=index):
                    with rec.span("coarse.voxel_down_sample", voxel_m=.04):
                        fragment.coarse = fragment.train.voxel_down_sample(.04)
                    rec.register(fragment.coarse, f"fragment:{index}:coarse", kind="coarse")
                    with rec.span("coarse.estimate_normals", radius_m=.08, max_neighbors=30):
                        fragment.coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.08, max_nn=30))
                    fragment.fpfh = module.REG.compute_fpfh_feature(
                        fragment.coarse, o3d.geometry.KDTreeSearchParamHybrid(radius=.2, max_nn=100))
            cv2.setRNGSeed(args.diagnostic_pnp_seed)
            for task in tasks:
                rec.pair_position = task["position"]
                a, b = task["pair"]
                source, target = fragments[a], fragments[b]
                rgb_proposals = 0
                with rec.span("diagnostic_pair_proposals", pair=task["pair"]) as event:
                    for aa in source.keys:
                        for bb in target.keys:
                            matches = module._matches(aa, bb)
                            proposal = appearance.propose_transform(aa.features, bb.features, fixture["camera"], matches)
                            rgb_proposals += proposal is not None
                    geometric = []
                    for seed in (a * 100 + b, a * 100 + b + 10000):
                        geometric.append(module._global_seed(source, target, seed) is not None)
                proposal_diagnostics.append({"pair": task["pair"], "rgb_proposals_present": rgb_proposals,
                                             "fpfh_proposals_present": geometric, "elapsed_s": event["inclusive_s"]})
                print(f"Diagnostic proposals {a}/{b}: {event['inclusive_s']:.3f}s", flush=True)

    source_after = source_hash()
    component_after = baseline.numerical_source_hash()
    script_after = file_hash(Path(__file__))
    loader_after = file_hash(Path(baseline.__file__))
    reference_after = file_hash(args.reference)
    fixture_after = file_hash(args.fixture)
    fixture_arrays_after = array_digest(fixture)
    quality["fixture_arrays_unchanged"] = fixture_arrays_before == fixture_arrays_after
    clouds_after = cloud_digest(fragments)
    quality["verification_clouds_unchanged"] = clouds_before == clouds_after
    quality.update(source_unchanged=source_before == source_after,
                   component_source_unchanged=component_before == component_after,
                   profiler_script_unchanged=script_before == script_after,
                   fixture_loader_script_unchanged=loader_before == loader_after,
                   fixture_file_unchanged=fixture_sha == fixture_after,
                   reference_file_unchanged=reference_before == reference_after)
    events = sorted(rec.events, key=lambda row: row["event_id"])
    native_icp = [row for row in events if row["phase"] == "fixed_verification" and row["name"] == "native.registration_icp"]
    target_uses = defaultdict(lambda: {"calls": 0, "inclusive_s": 0., "point_query_iteration_budget": 0})
    for row in native_icp:
        target = target_uses[row["target"]["id"]]
        target["calls"] += 1
        target["inclusive_s"] += row["inclusive_s"]
        target["point_query_iteration_budget"] += row["source"]["points"] * row["configured_max_iterations"]
    report = {"status": "passed" if all(quality[key] for key in QUALITY_GATES) else "failed",
        "metadata": {"scope": "standalone fixed bridge component; not full Finish, live FPS or graph timing",
        "fixture": str(args.fixture.resolve()), "fixture_sha256": fixture_sha,
        "fixture_metadata": fixture["metadata"], "reference": str(args.reference.resolve()),
        "reference_sha256": reference_before, "reference_sha256_after": reference_after,
        "source_sha256": source_before, "expected_source_sha256": args.expected_source_sha256,
        "script_sha256": script_before, "script_sha256_after": script_after,
        "fixture_loader_script_sha256": loader_before, "fixture_loader_script_sha256_after": loader_after,
        "fixture_sha256_after": fixture_after, "component_source_sha256_after": component_after,
        "source_sha256_after": source_after, "component_source_sha256": component_before,
        "threads": o3d.utility.get_max_threads(), "opencv_threads": cv2.getNumThreads(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"), "cuda_registration": os.environ["KINECT_CUDA_REGISTRATION"],
        "versions": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__, "open3d": o3d.__version__},
        "profiler": "nested original-function/native-call wall-time wrappers; no cProfile",
        "iteration_counter_limits": "Only native result fields, if exposed. Configured max_iteration sums and point budgets are ceilings, never actual iterations or queries.",
        "native_cost_limits": "Legacy native ICP/evaluation/info include tree setup, NN, linear solves and convergence work. Individual costs are opaque; standalone KDTreeFlann construction is a separate diagnostic, not their internal tree cost.",
        "proposal_limits": "Fixed recorded proposals authorize verification. Separate original PnP/FPFH diagnostic functions run afterwards; PnP has an explicit diagnostic RNG seed and does not reproduce original global RNG history/ranking.",
        "preparation_limits": "Existing exact prepared clouds and saved normals/cache loaded once. Coarse normals/FPFH rebuilt only for optional proposal diagnostics; raw RGB/depth and local fragment estimation are excluded.",
        "local_pose_source": "measured Finish fragment-local poses for component reproduction only; no archived live seeds",
        "fixture_arrays_sha256_before": fixture_arrays_before, "fixture_arrays_sha256_after": fixture_arrays_after,
        "cloud_arrays_sha256_before": clouds_before, "cloud_arrays_sha256_after": clouds_after,
        "diagnostic_pnp_seed": args.diagnostic_pnp_seed, "peak_process_rss_bytes": peak_rss_bytes()},
        "verification_s": verification_s, "total_wall_s": time.perf_counter() - started,
        "quality": quality, "rows": rows, "summaries": summaries(events), "timer_accounting": timer_accounting(events),
        "immutable_target_native_icp_uses": [{"target_id": name, **value} for name, value in sorted(target_uses.items())],
        "target_index_diagnostic": target_indexes, "proposal_diagnostic": proposal_diagnostics, "events": events}
    FAILURE_STATE["completed_report"] = report
    FAILURE_STATE["metadata"] = report["metadata"]
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary_path = write_compact_summary(report, args.output)
    print(f"Compact summary: {summary_path}", flush=True)
    print(json.dumps({"output": str(args.output), "verification_s": verification_s, "quality": quality}), flush=True)
    if not all(quality[key] for key in QUALITY_GATES):
        raise RuntimeError("Fixed component authority/provenance gate failed; inspect saved report")
    finish_cuda_worker()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        try:
            preserve_failure(error)
        except BaseException as saving_error:
            print(f"Could not save failure diagnostics: {saving_error}", file=sys.stderr, flush=True)
        raise
