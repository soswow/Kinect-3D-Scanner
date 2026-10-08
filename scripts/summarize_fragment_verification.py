"""Derive compact verifier evidence without importing numerical libraries.

The measured raw report and its profiler snapshot remain immutable. Branch
classification uses path components, including an immediate partial/visual
parent. Timings and numerical authority are copied, never recomputed.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_breakdown(events):
    groups = defaultdict(lambda: {"calls": 0, "inclusive_s": 0., "self_s": 0.,
                                   "configured_iteration_budget_sum": 0})
    for row in events:
        if row["phase"] != "fixed_verification" or not row["name"].startswith("native."):
            continue
        source = row.get("source", {})
        scope = "fragment_union" if ":union:" in source.get("id", "") else "camera"
        path = row.get("branch_path", "").split("/")
        branch = "partial_overlap" if "partial_bridge" in path else "visual" if "visual_bridge" in path else "direct"
        key = (row["name"], scope, source.get("kind"), row.get("direction"), branch)
        group = groups[key]
        group["calls"] += 1
        group["inclusive_s"] += row["inclusive_s"]
        group["self_s"] += row["self_s"]
        group["configured_iteration_budget_sum"] += row.get("configured_max_iterations") or 0
    return [{"name": key[0], "cloud_scope": key[1], "cloud_kind": key[2],
             "direction": key[3], "branch": key[4], **value}
            for key, value in sorted(groups.items(), key=lambda item: str(item[0]))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-profiler-script", type=Path, required=True)
    args = parser.parse_args()
    report_before = file_hash(args.report)
    script_before = file_hash(Path(__file__))
    snapshot_sha = file_hash(args.measured_profiler_script)
    report = json.loads(args.report.read_text())
    if snapshot_sha != report["metadata"]["script_sha256"]:
        raise RuntimeError("Measured profiler snapshot hash differs from immutable report")
    summary = {key: report[key] for key in ("status", "metadata", "quality", "verification_s", "total_wall_s",
        "summaries", "timer_accounting", "immutable_target_native_icp_uses", "target_index_diagnostic", "proposal_diagnostic")}
    summary["verification_native_call_breakdown"] = native_breakdown(report["events"])
    summary["pairs"] = [{key: row[key] for key in ("position", "pair", "proposal_count", "verified_proposals",
        "ambiguous", "accepted", "elapsed_s")} for row in report["rows"]]
    totals = {row["name"]: row["self_s"] for row in report["summaries"] if row["phase"] == "fixed_verification"}
    native = sum(value for name, value in totals.items() if name.startswith("native."))
    summary["disjoint_verification_self_s"] = {"native_icp": totals["native.registration_icp"],
        "native_evaluate": totals["native.evaluate_registration"],
        "native_information": totals["native.get_information_matrix_from_point_clouds"],
        "descriptor_lookup_matching": totals["descriptor_matches"],
        "other_python_gates_and_wrappers": sum(totals.values()) - native - totals["descriptor_matches"]}
    report_after = file_hash(args.report)
    script_after = file_hash(Path(__file__))
    if report_before != report_after or script_before != script_after:
        raise RuntimeError("Report or postprocessor changed before saving derived summary")
    summary["summary_postprocessing"] = {"source_report": str(args.report.resolve()),
        "source_report_sha256": report_before, "source_report_sha256_after": report_after,
        "postprocessor": str(Path(__file__).resolve()), "postprocessor_sha256": script_before,
        "postprocessor_sha256_after": script_after, "measured_profiler_script": str(args.measured_profiler_script.resolve()),
        "measured_profiler_script_sha256": snapshot_sha,
        "change": "Classify immediate partial/visual parents using path components; raw events, timers and numerical results are unchanged."}
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "disjoint_verification_self_s": summary["disjoint_verification_self_s"],
                      "verification_native_call_breakdown": summary["verification_native_call_breakdown"]}, indent=2))


if __name__ == "__main__":
    main()
