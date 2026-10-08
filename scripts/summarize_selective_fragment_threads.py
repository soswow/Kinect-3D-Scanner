"""Pure-stdlib compact summary; do not alter the measured pilot report."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root/"benchmark-output/cuda-pipeline/selective-fragment-threads"
    path = folder/"pilot.json"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = json.loads(path.read_text())
    assert report["status"] == "passed" and not report.get("failure")
    assert all(report["provenance"][key] for key in
               ("all_source_file_hashes_unchanged","fixture_arrays_unchanged","thread_limit_restored"))
    baseline = report["results"][0]
    assert baseline["small_threads"] == 20
    rows = []
    for result in report["results"]:
        assert result["complete"] and result["cloud_arrays_unchanged"]
        assert result["quality"]["same_decisions_and_support"]
        assert result["quality"]["pair_order_unchanged"] and result["quality"]["positions_unchanged"]
        assert result["every_proposal_quality"]["every_proposal_same_decisions_witnesses_order_pose_information"]
        rows.append({"small_threads":result["small_threads"],"verification_s":result["verification_s"],
            "ratio_to_baseline":result["verification_s"]/baseline["verification_s"],
            "pairs":[{"pair":row["pair"],"accepted":row["accepted"],"elapsed_s":row["elapsed_s"],
                "proposal_count":row["proposal_count"]} for row in result["rows"]],
            "diagnostics":result["diagnostics"],"all_original_proposal_and_pair_gates_passed":True})
    assert before == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {"kind":"selective-native-thread-component-summary","status":"passed",
        "result":"Selective small-camera thread reductions do not improve this pilot; keep original20, research only",
        "raw_report_sha256":before,"metadata":report["metadata"],"provenance":report["provenance"],
        "policies":rows,"summary_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits":"One repeat of accepted[8,14]/rejected[0,2] original nine-proposal component; no full Finish/camera speed claim. Native timings are inclusive within match totals, configured budgets are not actual iterations. Raw ZIP identity is carried by measured fixture metadata; this pilot itself rehashes fixture/reference/source, not ZIP at completion."}
    output = folder/"research-summary.json"
    output.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output.relative_to(root)),"status":summary["status"],
                      "ratios":[row["ratio_to_baseline"] for row in rows]}))


if __name__ == "__main__":
    main()
