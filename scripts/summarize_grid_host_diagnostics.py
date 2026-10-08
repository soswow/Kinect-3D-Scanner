"""Stdlib compact derivation of preserved sampled host diagnostic evidence."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/"benchmark-output/cuda-pipeline/grid-lookup-ablation-order-v2"
    raw=folder/"host-diagnostics.json"
    before=digest(raw)
    report=json.loads(raw.read_text())
    execution=json.loads((folder/"host-diagnostics-execution.json").read_text())
    assert execution["exit_code"]==0 and execution["report_status"]=="passed" and execution["report_sha256"]==before
    assert report["status"]=="passed" and report["all_provenance_guards_passed"] and report["input_arrays_unchanged"]
    assert report["host_diagnostic_helper"]["unchanged"] and not report["solver_cleanup_failures"]
    rows=report["host_diagnostic_measurements"]
    assert len(rows)==48 and all(row["same_kernel_raw_output_bits_equal"] and len(row["samples"])==20
        and all(sample["original_cpu_ids_equal"] for sample in row["samples"]) for row in rows)
    modes={row["variant"] for row in rows}
    assert modes=={family+mode for family in ("serial_staged","parallel_staged") for mode in ("_collect","_skip")}
    families=[]
    for family in ("serial_staged","parallel_staged"):
        collected={row["trace_batch"]:row for row in rows if row["variant"]==family+"_collect"}
        skipped={row["trace_batch"]:row for row in rows if row["variant"]==family+"_skip"}
        assert set(collected)==set(skipped)==set(range(12))
        for index,row in collected.items():
            for a,b in zip(row["samples"],skipped[index]["samples"]):
                assert a["stage_work"]=="collected" and b["stage_work"]=="uncollected" and "stage_counters" not in b
                assert b["host_stage_decode_s"]==0 and a["host_stage_decode_s"]>0
                assert all(a[key]==b[key] for key in ("repeat","query_rows","candidate_visits","cpu_fallback_queries"))
        a=report["host_diagnostic_totals"][family+"_collect"]
        b=report["host_diagnostic_totals"][family+"_skip"]
        families.append({"kernel_family":family,"sampled_batch_count":12,"repetitions":20,
            "adapter_per_12_batches_s":{"collect":a["adapter_wall_s"]/20,"skip":b["adapter_wall_s"]/20},
            "wall_saved_per_12_batches_s":(a["adapter_wall_s"]-b["adapter_wall_s"])/20,
            "collect_to_skip_wall_ratio":a["adapter_wall_s"]/b["adapter_wall_s"],
            "host_decode_per_12_batches_s":a["host_stage_decode_s"]/20,
            "mean_host_decode_per_sampled_call_s":a["host_stage_decode_s"]/240,
            "nested_search_transfer_per_12_batches_s":{"collect":a["gpu_search_transfer_s"]/20,"skip":b["gpu_search_transfer_s"]/20},
            "nested_cpu_fallback_per_12_batches_s":{"collect":a["cpu_fallback_s"]/20,"skip":b["cpu_fallback_s"]/20}})
    summary={"kind":"derived-bounded-host-stage-diagnostics-summary","status":"passed",
        "result":"Measurable approximately3percent sampled adapter saving; insufficient evidence to justify full parallel-staged audit",
        "production_integration":False,"new_kernel_full_bridge_authority":False,
        "source_sha256":report["source_sha256"],"raw_report_sha256":before,
        "execution_report_sha256":digest(folder/"host-diagnostics-execution.json"),
        "actual_process_exit_code":0,"process_elapsed_s":execution["elapsed_s"],
        "helper":report["host_diagnostic_helper"],"fixture_binding":report["fixture_binding"],
        "proof_report_hashes":report["proof_report_hashes"],"runtime_binding":report["full_radius_proof_binding"],
        "trace_sha256":report["trace_sha256"],"trace_manifest_sha256":report["trace_manifest_sha256"],
        "families":families,"sampled_queries_per_batch_max":2048,
        "original_full_batch_sizes":sorted({row["original_full_query_count"] for row in rows}),
        "quality":{"sampled_cpu_ids_equal":True,"same_kernel_all_raw_output_bits_equal":True,
            "original_arrays_source_raw_runtime_helpers_unchanged":True,"cleanup_passed":True,
            "disabled_stage_work":"uncollected, counters omitted"},
        "limits":"Bounded closed actual-query trace only. Original CPU missing/tie/support fallback is active and dominates these sampled adapter calls. Nested decode/search/fallback timers overlap adapter wall. Alternating order, no linear extrapolation to larger full batches or claim of full bridge/Finish/scanning gain. Parallel-staged full-proof files remain prepared but unexecuted.",
        "summary_script_sha256":digest(Path(__file__))}
    assert digest(raw)==before
    output=folder/"host-diagnostics-summary.json"
    output.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output.relative_to(ROOT)),"status":"passed","families":families}))


if __name__=="__main__":main()
