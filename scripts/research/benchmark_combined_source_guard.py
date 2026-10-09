"""Stdlib-only causal check of repeated AST versus full-file source hashing.

Measures only source guard primitives, not CUDA or registration throughput.
The source contract is checked once, and every hash test requires the exact
same complete original file bytes. No numerical module may be imported.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
KIND="stdlib-original-combined-source-guard-microbenchmark-v1"


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_no_numerics():
    loaded=sorted(name for name in sys.modules if name.split(".")[0] in ("numpy","open3d","cupy","cv2"))
    if loaded:raise RuntimeError("Source-only benchmark imported numerical libraries: "+str(loaded))


def run(args):
    guard_no_numerics()
    from scripts.research import research_combined_sync_icp as original
    guard_no_numerics()
    source=original.ORIGINAL_RESIDENT
    raw_before=file_hash(args.reference);reference=json.loads(args.reference.read_text(encoding="utf-8"))
    expected=reference["combined_runtime_binding"]["artifacts_sha256"][str(source.relative_to(ROOT))]
    if reference["status"]!="passed" or file_hash(source)!=expected:
        raise RuntimeError("Require current original source and closed unchanged component reference")
    contract=original.source_contract()
    if contract["original_math_bytes_unchanged"] is not True:raise RuntimeError("Original math contract failed")
    body=source.read_text(encoding="utf-8")
    report={"kind":KIND,"status":"running","reference":{"path":str(args.reference.resolve()),"sha256":raw_before},
        "source":{"path":str(source.relative_to(ROOT)),"sha256":expected,"bytes":source.stat().st_size,
            "ast_nodes":sum(1 for _ in ast.walk(ast.parse(body)))},"contract":contract,"calls_per_component":236,
        "iterations":args.iterations,"repeats":args.repeats,"results":[],"numerical_imports":False,
        "scope":"Only isolated stdlib source checks; concurrent parent field research may affect host wall. No registration-speed or timer-subtraction claim."}
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save()
    try:
        for repeat in range(args.repeats):
            order=("original_ast_contract","whole_file_sha") if repeat%2==0 else ("whole_file_sha","original_ast_contract")
            for mode in order:
                started=time.perf_counter()
                for _ in range(args.iterations):
                    if mode=="original_ast_contract":
                        if original.source_contract()!=contract:raise RuntimeError("Original source contract changed")
                    elif file_hash(source)!=expected:raise RuntimeError("Complete original source bytes changed")
                elapsed=time.perf_counter()-started
                report["results"].append({"mode":mode,"repeat":repeat,"elapsed_s":elapsed,
                    "wall_us_per_check":elapsed/args.iterations*1e6})
                save()
        report["median_s"]={mode:statistics.median(row["elapsed_s"] for row in report["results"] if row["mode"]==mode)
            for mode in ("original_ast_contract","whole_file_sha")}
        report["source_guard_ratio"]=report["median_s"]["original_ast_contract"]/report["median_s"]["whole_file_sha"]
        guard_no_numerics()
        if file_hash(source)!=expected or original.source_contract()!=contract or file_hash(args.reference)!=raw_before:
            raise RuntimeError("Original source/contract/component reference changed during source-only check")
        report["source_and_reference_unchanged"]=True
        report["script_sha256"]=file_hash(__file__)
        report["status"]="passed"
        save()
    except BaseException as error:
        report.update(status="failed",failure={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()})
        try:save()
        except BaseException as write_error:error.add_note(f"Failed diagnostic write: {write_error}")
        raise
    print(json.dumps({"output":str(args.output),"status":report["status"],"median_s":report["median_s"]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=ROOT/"benchmark-output/field-cuda-study/combined-sync-timing-v1/report.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--iterations",type=int,default=236)
    parser.add_argument("--repeats",type=int,default=3)
    args=parser.parse_args()
    if args.output.exists():parser.error("Require fresh output; preserve prior evidence")
    if not 1<=args.iterations<=236 or not 1<=args.repeats<=3:parser.error("Require bounded source-only checks")
    run(args)


if __name__=="__main__":main()
