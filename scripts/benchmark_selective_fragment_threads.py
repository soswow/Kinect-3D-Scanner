"""Compare unchanged native bridge ICP with selective per-camera TBB limits.

Standalone research only. The existing fixture's measured local poses reproduce
component inputs, never archived live scanning seeds. Every original competing
proposal, reciprocal/held-out gate and witness remains authoritative and ordered.
Only the TBB maximum around the original three-scale _match changes, for two
camera-training clouds no larger than --small-max-points. Union ICP, evaluation,
descriptors and information calls retain 20 TBB threads. Both setter/restoration
cost and native timing are measured; actual iterations remain unexposed.

Quick accepted/rejected pilot, after an exclusive slot is allocated:
  python scripts/benchmark_selective_fragment_threads.py --positions 2 4 \
    --fixture benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.fixture.pickle \
    --reference benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.json \
    --output benchmark-output/cuda-pipeline/selective-fragment-threads/pilot.json
"""

import argparse
import contextlib
import functools
import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ["KINECT_CUDA_REGISTRATION"] = "cpu"
EXPECTED_SOURCE = "9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SelectiveThreads:
    def __init__(self, o3d, module, refinement, fragments, small_threads, limit):
        self.o3d, self.module, self.refinement = o3d, module, refinement
        self.small_threads, self.limit = small_threads, limit
        self.camera_cloud_ids = {id(view.train) for fragment in fragments.values() for view in fragment.keys}
        self.current_threads = 20
        self.match_calls = self.eligible_matches = self.thread_reduced_matches = self.thread_switch_calls = 0
        self.match_s = self.eligible_match_s = self.thread_switch_s = 0.
        self.native_groups = defaultdict(lambda: {"calls": 0, "native_s": 0., "configured_iteration_budget_sum": 0})

    def switch(self, threads):
        start = time.perf_counter()
        self.o3d.utility.set_max_threads(threads)
        self.thread_switch_s += time.perf_counter() - start
        self.thread_switch_calls += 1
        self.current_threads = threads

    @contextlib.contextmanager
    def install(self):
        original_match = self.module._match
        original_reg = self.refinement.REG
        selector = self

        class NativeProxy:
            def __getattr__(self, name):
                return getattr(original_reg, name)

            def registration_icp(self, *args, **kwargs):
                start = time.perf_counter()
                result = original_reg.registration_icp(*args, **kwargs)
                elapsed = time.perf_counter() - start
                key = (selector.current_threads, float(args[2]))
                group = selector.native_groups[key]
                group["calls"] += 1
                group["native_s"] += elapsed
                group["configured_iteration_budget_sum"] += int(args[5].max_iteration)
                return result

        @functools.wraps(original_match)
        def match(source, target, initial):
            started = time.perf_counter()
            eligible = (id(source) in selector.camera_cloud_ids and id(target) in selector.camera_cloud_ids
                        and max(len(source.points), len(target.points)) <= selector.limit)
            threads = selector.small_threads if eligible else 20
            selector.match_calls += 1
            selector.eligible_matches += eligible
            selector.thread_reduced_matches += eligible and threads != 20
            try:
                if threads != selector.current_threads:
                    selector.switch(threads)
                return original_match(source, target, initial)
            finally:
                if selector.current_threads != 20:
                    selector.switch(20)
                elapsed = time.perf_counter() - started
                selector.match_s += elapsed
                if eligible:
                    selector.eligible_match_s += elapsed

        self.module._match, self.refinement.REG = match, NativeProxy()
        try:
            yield
        finally:
            self.module._match, self.refinement.REG = original_match, original_reg
            if self.current_threads != 20:
                self.switch(20)

    def diagnostics(self):
        return {"match_calls": self.match_calls, "eligible_matches": self.eligible_matches,
                "thread_reduced_matches": self.thread_reduced_matches, "match_s_including_switch": self.match_s,
                "eligible_match_s_including_switch": self.eligible_match_s,
                "thread_switch_calls": self.thread_switch_calls, "thread_switch_s": self.thread_switch_s,
                "thread_limit_after": self.o3d.utility.get_max_threads(),
                "native_icp": [{"threads": key[0], "distance_m": key[1], **value}
                               for key, value in sorted(self.native_groups.items())]}


def fresh_fragments(fixture, baseline, module):
    fragments = {index: baseline.unpack_fragment(data) for index, data in fixture["fragments"].items()}
    # Each policy gets exactly the worker's original bounded cache installation.
    for index, data in fixture["fragments"].items():
        for original, view in zip(data["keys"], fragments[index].keys):
            for other_fragment in fragments.values():
                for other in other_fragment.keys:
                    matches = original["matches"].get(other.index)
                    if matches is not None:
                        module._cache_matches(view, other, matches.copy())
    return fragments


def compare_every_proposal(reference, candidate, baseline):
    """Check each outcome/witness/pose/info, including competing accepted seeds."""
    differences, comparisons = [], []
    for old, new in zip(reference, candidate):
        if len(old["proposal_results"]) != len(new["proposal_results"]):
            differences.append({"pair": old["pair"], "reason": "changed proposal count"})
            continue
        for a, b in zip(old["proposal_results"], new["proposal_results"]):
            if (a["proposal_index"], a["initial_sha256"]) != (b["proposal_index"], b["initial_sha256"]):
                differences.append({"pair": old["pair"], "reason": "changed proposal order/input"})
                continue
            def row(proposal):
                return {"pair": old["pair"], "accepted": proposal["result"] is not None,
                        "ambiguous": False, "verified_proposals": int(proposal["result"] is not None),
                        "proposal_count": 1, "best": proposal["result"]}
            quality = baseline.compare_rows([row(a)], [row(b)])
            comparisons.append({"pair": old["pair"], "proposal_index": a["proposal_index"], **quality})
            if not quality["same_decisions_and_support"]:
                differences.extend(quality["differences"])
    return {"every_proposal_same_decisions_witnesses_order_pose_information": not differences,
            "differences": differences, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, nargs="+")
    parser.add_argument("--small-threads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--small-max-points", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE)
    args = parser.parse_args()
    if args.repeats < 1 or args.small_max_points < 1 or any(t < 1 or t >= 20 for t in args.small_threads):
        parser.error("Positive point/repeat bounds and selective limits below20 required")
    if len(set(args.small_threads)) != len(args.small_threads):
        parser.error("Duplicate thread policies")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from scripts import benchmark_parallel_fragments as baseline
    from scripts import profile_fragment_verification as invariance_helper
    from scripts.profile_fragment_verification import array_digest, cloud_digest
    from scripts.profile_session import source_hash
    from scripts.process_metrics import finish_cuda_worker, peak_rss_bytes
    source_before, component_before = source_hash(), baseline.numerical_source_hash()
    script_before, loader_before = file_hash(Path(__file__)), file_hash(Path(baseline.__file__))
    invariance_helper_before = file_hash(Path(invariance_helper.__file__))
    if source_before != args.expected_source_sha256:
        raise RuntimeError("Require the expected frozen production source before numerical imports")
    import cv2
    import numpy as np
    import open3d as o3d
    from scanner_server import fragments as module, refinement
    o3d.utility.set_max_threads(20)
    cv2.setNumThreads(20)
    fixture_before, reference_before = file_hash(args.fixture), file_hash(args.reference)
    with args.fixture.open("rb") as stream:
        fixture = pickle.load(stream)
    reference = json.loads(args.reference.read_text())
    if reference["metadata"]["fixture_sha256"] != fixture_before or fixture["metadata"]["component_source_sha256"] != component_before:
        raise RuntimeError("Fixture/reference numerical dependency or byte hash mismatch")
    if args.positions is not None and (len(set(args.positions)) != len(args.positions) or not set(args.positions).issubset(
            {task["position"] for task in fixture["tasks"]})):
        raise ValueError("Requested positions are absent or duplicated")
    tasks = [task for task in fixture["tasks"] if args.positions is None or task["position"] in args.positions]
    positions = {task["position"] for task in tasks}
    reference_rows = [row for result in reference["results"] if result["configuration"] == "1x20"
                      for row in result["rows"] if row["position"] in positions]
    input_before = array_digest(fixture)
    report = {"status": "running", "metadata": {"scope": "unchanged original-raw fixed verification component; no full Finish or live throughput claim",
        "source_sha256": source_before, "expected_source_sha256": args.expected_source_sha256,
        "component_source_sha256": component_before, "script_sha256": script_before,
        "fixture_loader_script_sha256": loader_before, "fixture_sha256": fixture_before,
        "invariance_helper_script_sha256": invariance_helper_before,
        "reference_sha256": reference_before, "fixture_metadata": fixture["metadata"],
        "baseline_threads": 20, "opencv_threads": cv2.getNumThreads(), "omp_threads": os.environ["OMP_NUM_THREADS"],
        "registration": "original legacy CPU point-to-plane, HuberLoss0.01, three native distances/criteria unchanged",
        "selection": "Both original camera-training clouds have at most small_max_points; union ICP and all other work retain20 threads",
        "small_max_points": args.small_max_points, "requested_small_threads": args.small_threads,
        "cache_policy": "Fresh unpacked clouds and original bounded descriptor-cache installation before every policy; no result/index cache",
        "iteration_limits": "Recorded native call count and configured budgets are not actual unexposed iterations",
        "authority_tolerances": "Original decisions, independent witness IDs/scopes and proposal order identical; transform translation<=1e-6m/angle<=1e-4deg and information absolute<=1e-5 as original fixture benchmark",
        "versions": {"python": sys.version, "numpy": np.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__}}, "results": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    failure = None
    try:
        for repeat in range(args.repeats):
            policies = [20] + args.small_threads
            if repeat % 2:
                policies.reverse()
            current_rows = {}
            for small_threads in policies:
                fragments = fresh_fragments(fixture, baseline, module)
                clouds_before = cloud_digest(fragments)
                selector = SelectiveThreads(o3d, module, refinement, fragments, small_threads, args.small_max_points)
                rows = []
                result_row = {"repeat": repeat, "small_threads": small_threads, "rows": rows, "complete": False}
                report["results"].append(result_row)
                started = time.perf_counter()
                with selector.install():
                    for task in tasks:
                        pair_started = time.perf_counter()
                        a, b = task["pair"]
                        verified, proposals = [], []
                        for proposal_index, pose in enumerate(task["proposals"]):
                            proposal_started = time.perf_counter()
                            bridge = module._verify_bridge(fragments[a], fragments[b], pose, fixture["camera"])
                            proposals.append({"proposal_index": proposal_index,
                                "initial_sha256": hashlib.sha256(pose.tobytes()).hexdigest(),
                                "result": baseline.json_value(bridge), "elapsed_s": time.perf_counter() - proposal_started})
                            if bridge is not None:
                                verified.append(bridge)
                        ambiguous = bool(verified) and any(module._disagrees(verified[0]["transform"], item["transform"])
                                                           for item in verified[1:])
                        best = verified[0] if verified and not ambiguous else None
                        rows.append({"position": task["position"], "pair": task["pair"], "proposal_count": len(task["proposals"]),
                            "verified_proposals": len(verified), "ambiguous": ambiguous, "accepted": best is not None,
                            "best": baseline.json_value(best), "proposal_results": proposals,
                            "elapsed_s": time.perf_counter() - pair_started})
                        print(f"Repeat{repeat}, small{small_threads}, pair{a}/{b}: {rows[-1]['elapsed_s']:.3f}s, accepted={best is not None}", flush=True)
                result_row.update(complete=True, verification_s=time.perf_counter() - started,
                                  diagnostics=selector.diagnostics(), quality=baseline.compare_rows(reference_rows, rows),
                                  cloud_arrays_unchanged=clouds_before == cloud_digest(fragments))
                result_row["quality"].update(pair_order_unchanged=[row["pair"] for row in rows] == [task["pair"] for task in tasks],
                    positions_unchanged=[row["position"] for row in rows] == [task["position"] for task in tasks])
                current_rows[small_threads] = rows
                save()
                del fragments
            # Baseline is first on even repetitions and last on odd repetitions.
            for result_row in report["results"]:
                if result_row["repeat"] == repeat:
                    result_row["every_proposal_quality"] = compare_every_proposal(current_rows[20], result_row["rows"], baseline)
            save()
    except BaseException as exc:
        import traceback
        failure = exc
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        o3d.utility.set_max_threads(20)
        after = {"source_sha256_after": source_hash(), "component_source_sha256_after": baseline.numerical_source_hash(),
                 "script_sha256_after": file_hash(Path(__file__)), "fixture_loader_script_sha256_after": file_hash(Path(baseline.__file__)),
                 "invariance_helper_script_sha256_after": file_hash(Path(invariance_helper.__file__)),
                 "fixture_sha256_after": file_hash(args.fixture), "reference_sha256_after": file_hash(args.reference)}
        report["metadata"].update(after)
        report["provenance"] = {"all_source_file_hashes_unchanged": all(report["metadata"][key[:-6]] == value for key, value in after.items()),
            "fixture_arrays_unchanged": input_before == array_digest(fixture), "thread_limit_restored": o3d.utility.get_max_threads() == 20,
            "peak_process_rss_bytes": peak_rss_bytes()}
        passed = (failure is None and all(report["provenance"][key] for key in
            ("all_source_file_hashes_unchanged", "fixture_arrays_unchanged", "thread_limit_restored"))
            and all(row["complete"] and row["quality"]["same_decisions_and_support"] and row["quality"]["pair_order_unchanged"]
                    and row["quality"]["positions_unchanged"] and row["cloud_arrays_unchanged"]
                    and row["every_proposal_quality"]["every_proposal_same_decisions_witnesses_order_pose_information"]
                    for row in report["results"]))
        report["status"] = "passed" if passed else "failed"
        save()
    print(json.dumps({"output": str(args.output), "status": report["status"],
        "policies": [{key: row.get(key) for key in ("repeat", "small_threads", "verification_s")} for row in report["results"]]}), flush=True)
    if not passed:
        raise RuntimeError("Selective threading authority/provenance test failed; saved diagnostic report") from failure
    finish_cuda_worker()


if __name__ == "__main__":
    main()
