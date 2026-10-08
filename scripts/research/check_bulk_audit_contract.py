"""Stdlib fault contracts for the separate bulk CPU-auditor authority.

These are deliberately fabricated, tiny contract fixtures. Already independently
tested inherited device/source/math guards are mocked at their entry points;
the new API closure, complete dual accounting, input closure and token consumer
run normally. This produces no numerical, native or GPU correctness evidence.
"""

import argparse
import ast
import contextlib
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import validate_bulk_resident_audit as guard


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")


class Fixture:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.dual_path = self.folder / "dual.json"
        self.api_path = self.folder / "api.json"
        self.artifacts = {name: digest(ROOT / name) for name in guard.BULK_ARTIFACTS}
        self.created = []

        def closed(name):
            path = self.folder / name
            path.write_bytes(("non-numerical-contract-fixture:" + name).encode())
            self.created.append(path)
            return {"path": str(path), "sha256": digest(path)}

        native = closed("native-not-a-dll.bin")
        numpy = closed("numpy-not-a-dll.bin")
        self.runtime = {
            "python": "contract-python", "python_executable": "contract-executable",
            "numpy": "contract-numpy", "numpy_file": "contract-numpy-file", "open3d": "contract-open3d",
            "selected_modules": ["contract-module"],
            "native_binaries": {native["path"]: {"sha256": native["sha256"], "bytes": Path(native["path"]).stat().st_size}},
            "numpy_native_binaries": {numpy["path"]: {"sha256": numpy["sha256"], "bytes": Path(numpy["path"]).stat().st_size}},
            "thread_environment": {"OMP_NUM_THREADS": "8"},
        }
        trace = {}
        for name in ("trace", "manifest", "capture"):
            value = closed(name + ".fixture")
            trace.update({name + "_path": value["path"], name + "_sha256": value["sha256"]})
        reference = closed("official-reference.fixture")
        sources = [dict(snapshot=value["path"], sha256=value["sha256"])
                   for value in (closed("source-%d.fixture" % i) for i in range(4))]

        def case(rows, chunks):
            return {"query_rows": rows, "target_rows": 3, "all_original_rows_checked": True,
                "arrays_unchanged": True, "scalar_wall_s": .01,
                "bulk_runs": [{"query_rows": rows, "target_rows": 3, "chunk_rows": chunk,
                    "id_errors": 0, "diagnostic_metric_bit_errors": 0, "evaluate_calls": 1,
                    "all_in_wall_s": .01} for chunk in chunks]}

        api_runtime = copy.deepcopy(self.runtime)
        api_runtime["thread_environment"]["OMP_NUM_THREADS"] = None
        self.api = {"kind": "scalar-vs-original-legacy-bulk-nearest-api-crosscheck", "status": "passed",
            "script_sha256": self.artifacts[guard.BULK_ARTIFACTS[0]],
            "script_sha256_after": self.artifacts[guard.BULK_ARTIFACTS[0]],
            "runtime": api_runtime, "runtime_after": copy.deepcopy(api_runtime),
            "new_field_bulk_only_shadow_authorized": False, "gpu_queries_run": False,
            "trace_binding": trace, "trace_binding_after": copy.deepcopy(trace),
            "source_reference": {"manifest": reference["path"], "sha256": reference["sha256"], "sources": sources},
            "normalization_contracts": {"passed": True},
            "synthetic_summary": {"query_rows": 5924, "all_ids_and_metric_bits_exact": True},
            "real_summary": {"query_rows": 24576, "all_retained_rows_scalar_checked": True,
                "stored_original_gold_equal": True, "all_ids_and_metric_bits_exact": True},
            "synthetic_cases": [case(6, [65536]) for _ in range(931)] + [case(338, [65536])],
            "real_cases": [case(2048, [257, 4096, 65536]) for _ in range(12)]}
        inputs = {name: closed(name + "-input.fixture") for name in ("fixture", "reference", "raw")}
        self.scope = {"fixture_sha256": inputs["fixture"]["sha256"],
            "reference_sha256": inputs["reference"]["sha256"], "original_raw_input_sha256": inputs["raw"]["sha256"],
            "tasks": [{"position": position, "pair": pair,
                "proposal_sha256": [hashlib.sha256((str(position)+":"+str(i)).encode()).hexdigest() for i in range(n)]}
                for position, pair, n in ((0, [8, 12], 5), (4, [0, 2], 4))]}
        self.binding = {"artifacts_sha256": {name.replace("/", "\\"): value for name, value in self.artifacts.items()},
            "bulk_audit_configuration": {"policy": guard.POLICY, "audit_mode": "dual", "chunk_rows": 65536},
            "bulk_cpu_runtime": copy.deepcopy(self.runtime), "resident_configuration": {"device": "CUDA:0"}}
        target = "a" * 64
        records = []
        left = guard.EXPECTED_QUERIES
        while left:
            rows = min(left, 1_000_000)
            records.append({"complete": True, "bulk_domain_supported": True, "target_unchanged": True,
                "audit_query_descriptor": {"dtype": "<f8", "shape": [rows, 3], "bytes": 24*rows, "sha256": "b"*64},
                "target_digest": target, "scalar_ids_sha256": "c"*64, "bulk_ids_sha256": "c"*64,
                "scalar_squared_sha256": "d"*64, "bulk_squared_sha256": "d"*64,
                "id_mismatches": 0, "metric_bit_mismatches": 0, "audit_hit_rows": rows-1,
                "audit_miss_rows": 1, "radius_m": .06, "evaluate_calls": 1, "all_in_wall_s": .01})
            left -= rows
        hits = guard.EXPECTED_QUERIES-len(records)
        stats = self.coverage(guard.EXPECTED_QUERIES, hits, len(records))
        stats.update({"dual_scalar_bulk_queries": guard.EXPECTED_QUERIES, "bulk_shadow_queries": guard.EXPECTED_QUERIES,
            "bulk_shadow_hits": hits, "bulk_shadow_misses": len(records), "bulk_evaluate_calls": len(records),
            "dual_id_mismatches": 0, "dual_metric_bit_mismatches": 0, "bulk_domain_scalar_queries": 0,
            "exact_cpu_queries": 0, "resident_xyz_uploads": 1})

        def run(mode):
            pairs = [{"position": task["position"], "pair": task["pair"],
                "proposal_results": [{"proposal_index": i, "input_sha256": value, "quality": {"passed": True}}
                    for i, value in enumerate(task["proposal_sha256"])],
                "verdict": {"accepted": task["position"] == 0}, "pair_verdict_same": True}
                for task in self.scope["tasks"]]
            return {"mode": mode, "repeat": 0, "complete": True, "pairs": pairs,
                "statistics_delta": copy.deepcopy(stats), "bulk_records": copy.deepcopy(records),
                "bulk_records_sha256": guard.canonical_hash(records)}

        checks = {name: True for name in guard.device_proof.DEVICE_CHECKS}
        checks.update({"passed": True, "required_checks": list(guard.device_proof.DEVICE_CHECKS),
            "classifier_cases": 32, "classifier_rows": 5128, "packet_rows": 1,
            "positive_mask_coverage": {name: 1 for name in ("unsupported", "uncertain", "audited_hits", "audited_misses")}})
        self.report = {"kind": guard.KIND, "traversal": guard.TRAVERSAL, "status": "passed",
            "proof_bindings": self.binding, "proof_bindings_after": self.binding,
            "bulk_audit": dict(self.binding["bulk_audit_configuration"], records=records,
                records_sha256=guard.canonical_hash(records), api_native_identity_match=True,
                new_field_bulk_shadow_authorized_by_report_alone=False),
            "synthetic_only": False, "synthetic_device_adapter_checked": True,
            "synthetic_indices": [{"queries": 2, "index_mismatches": 0, "device_adapter_index_mismatches": 0,
                "device_adapter_max_squared_distance_delta": 0.} for _ in range(16)],
            "synthetic_grid_boundaries": [{"queries": 2, "index_mismatches": 0,
                "case": "original-double-subnormal-and-minimum-radius" if i==0 else "contract-boundary"} for i in range(43)],
            "synthetic_statistics": self.coverage(2, 1, 1), "device_flat_synthetic": checks,
            "flat_cache_ownership": {"passed": True, "cloud_evictions": 1, "bounded_retained_bytes": True,
                "clear_releases_all_owned_arrays": True, "one_byte_budget_original_cpu_fallback": True},
            "fixture_binding": self.scope, "fixture_binding_sha256": guard.canonical_hash(self.scope),
            "fixture_inputs": inputs, "fixture_inputs_after": copy.deepcopy(inputs),
            "fixture_arrays_unchanged": True, "cloud_arrays_unchanged": True,
            "original_fixture_authority": {"same_decisions_and_support": True},
            "fixture_provenance": {"local_pose_source": "measured Finish fragment report; no archived ZIP poses"},
            "real_runs": [run("native_cpu"), run("grid")], "real_target_membership_sha256": [target],
            "solve_metadata": {"mocked_inherited_solve": True}}
        for name in ("fixture_sha256", "reference_sha256", "original_raw_input_sha256"):
            self.report[name] = self.report[name+"_after"] = self.scope[name]

    @staticmethod
    def coverage(rows, hits, misses):
        return {"query_rows": rows, "direct_gpu_hits": hits, "declared_gpu_misses": misses,
            "audited_hits": hits, "audited_misses": misses, "audit_index_mismatches": 0, "audit_false_misses": 0,
            "device_calls": 1, "device_malformed_results": 0, "device_flagged_rows": rows,
            "device_query_download_rows": rows, "device_query_download_bytes": 64*rows}

    def save(self):
        write_json(self.api_path, self.api)
        value = {"path": str(self.api_path), "sha256": digest(self.api_path)}
        self.binding["bulk_api_proof"] = value
        self.report["bulk_api_proof"] = self.report["bulk_api_proof_after"] = value
        write_json(self.dual_path, self.report)

    def validate(self):
        self.save()
        return guard.validate_bulk_resident_audit(self.dual_path, self.binding, self.scope)


def contracts():
    results = []
    with tempfile.TemporaryDirectory(prefix="bulk-audit-stdlib-contract-") as folder:
        fixture = Fixture(folder)
        pristine = copy.deepcopy((fixture.api, fixture.report, fixture.binding, fixture.scope))
        def restore():
            fixture.api, fixture.report, fixture.binding, fixture.scope = copy.deepcopy(pristine)
            fixture.report["proof_bindings"] = fixture.report["proof_bindings_after"] = fixture.binding
        def common(report, binding, *_):
            guard.require(report.get("status") == "passed" and report.get("proof_bindings") == binding
                and report.get("proof_bindings_after") == binding, "Mocked inherited common guard propagation")
        with patch.object(guard.device_proof, "validate_configuration", side_effect=lambda b: guard.device_proof.normalized_artifacts(b["artifacts_sha256"])) as config, \
             patch.object(guard.device_proof, "validate_current_artifacts", return_value={}) as current, \
             patch.object(guard.device_proof, "common_checks", side_effect=common) as inherited, \
             patch.object(guard.device_proof, "resident_checks", side_effect=lambda *_: fixture.report["solve_metadata"]) as resident:
            def check(name, mutation=None, action=None, rejected=True):
                restore()
                if mutation:
                    mutation(fixture)
                try:
                    token = fixture.validate()
                    if action:
                        action(fixture, token)
                except (guard.GridProofError, FrozenInstanceError):
                    if not rejected:
                        raise
                else:
                    if rejected:
                        raise AssertionError("Accepted invalid contract: " + name)
                results.append({"case": name, "passed": True, "expected_rejection": rejected})
            check("positive full dual scope, API environment distinct, Windows normalized artifact keys", rejected=False)
            check("old component kind", lambda f: f.report.update(kind=guard.device_proof.KIND))
            check("initial API alone not dual", lambda f: f.report.update(traversal="API-only"))
            check("inherited failed status propagates", lambda f: f.report.update(status="failed"))
            check("common source/runtime binding guard called", lambda f: f.report.update(proof_bindings_after={}))
            check("missing new adapter artifact", lambda f: f.binding["artifacts_sha256"].pop(guard.BULK_ARTIFACTS[1].replace("/", "\\")))
            check("wrong auditor mode", lambda f: f.binding["bulk_audit_configuration"].update(audit_mode="bulk-shadow"))
            check("wrong chunk budget", lambda f: f.binding["bulk_audit_configuration"].update(chunk_rows=65537))
            check("CPU audit does not authorize field pose", lambda f: f.report["bulk_audit"].update(new_field_bulk_shadow_authorized_by_report_alone=True))
            check("API does not authorize GPU queries", lambda f: f.api.update(gpu_queries_run=True))
            check("wrong loaded native version", lambda f: f.api["runtime"].update(open3d="other"))
            check("API runtime changed", lambda f: f.api["runtime_after"].update(python="other"))
            check("API helper source changed", lambda f: f.api.update(script_sha256_after="f"*64))
            check("API normalization failure", lambda f: f.api["normalization_contracts"].update(passed=False))
            check("API missing boundary query", lambda f: f.api["synthetic_cases"].pop())
            check("API scalar ID mismatch", lambda f: f.api["real_cases"][0]["bulk_runs"][1].update(id_errors=1))
            check("API diagnostic metric bit mismatch", lambda f: f.api["real_cases"][0]["bulk_runs"][2].update(diagnostic_metric_bit_errors=1))
            check("API missing chunk crosscheck", lambda f: f.api["real_cases"][0]["bulk_runs"].pop())
            check("API identity array mutation", lambda f: f.api["real_cases"][0].update(arrays_unchanged=False))
            check("API trace mutation", lambda f: f.api["trace_binding_after"].update(trace_sha256="f"*64))
            check("API missing official source", lambda f: f.api["source_reference"]["sources"].pop())
            check("API source bytes changed", lambda f: f.api["source_reference"]["sources"][0].update(sha256="f"*64))
            check("noncanonical full thread policy", lambda f: f.binding["bulk_cpu_runtime"]["thread_environment"].update(OMP_NUM_THREADS="20"))
            check("binary extent differs", lambda f: next(iter(f.binding["bulk_cpu_runtime"]["native_binaries"].values())).update(bytes=1))
            check("synthetic only", lambda f: f.report.update(synthetic_only=True))
            check("actual device adapter absent", lambda f: f.report.update(synthetic_device_adapter_checked=False))
            check("missing new boundary proof", lambda f: f.report["synthetic_grid_boundaries"].pop())
            check("classifier fault uncovered", lambda f: f.report["device_flat_synthetic"].update(malformed_raw_hard_failure=False))
            check("zero positive miss classifier branch", lambda f: f.report["device_flat_synthetic"]["positive_mask_coverage"].update(audited_misses=0))
            check("cache ownership release absent", lambda f: f.report["flat_cache_ownership"].update(clear_releases_all_owned_arrays=False))
            check("fixture current path mismatch", lambda f: f.report["fixture_inputs_after"]["raw"].update(sha256="f"*64))
            check("raw bytes mismatch", lambda f: f.report.update(original_raw_input_sha256_after="f"*64))
            check("archived poses forbidden", lambda f: f.report["fixture_provenance"].update(local_pose_source="archived ZIP poses"))
            check("native run missing", lambda f: f.report["real_runs"].pop(0))
            check("proposal order changed", lambda f: f.report["real_runs"][1]["pairs"][0]["proposal_results"].reverse())
            check("accepted gate changed", lambda f: f.report["real_runs"][1]["pairs"][0]["verdict"].update(accepted=False))
            check("original witness failed", lambda f: f.report["real_runs"][1]["pairs"][0]["proposal_results"][0]["quality"].update(passed=False))
            check("not all 104 million queries", lambda f: f.report["real_runs"][1]["statistics_delta"].update(query_rows=guard.EXPECTED_QUERIES-1))
            check("miss CPU shadow absent", lambda f: f.report["real_runs"][1]["statistics_delta"].update(audited_misses=0))
            check("proof packet not copied once", lambda f: f.report["real_runs"][1]["statistics_delta"].update(device_query_download_bytes=1))
            check("scalar ambiguity fallback hidden", lambda f: f.report["real_runs"][1]["statistics_delta"].update(exact_cpu_queries=1))
            check("unsupported bulk domain fallback", lambda f: f.report["real_runs"][1]["statistics_delta"].update(bulk_domain_scalar_queries=1))
            check("dual ID errors", lambda f: f.report["real_runs"][1]["statistics_delta"].update(dual_id_mismatches=1))
            check("dual metric errors", lambda f: f.report["real_runs"][1]["statistics_delta"].update(dual_metric_bit_mismatches=1))
            check("record hash mutation", lambda f: f.report["real_runs"][1]["bulk_records"][0].update(bulk_ids_sha256="f"*64))
            check("missing cumulative records", lambda f: f.report["bulk_audit"]["records"].pop())
            check("distinct token consumer positive and fresh full rebuild", action=lambda f,t: guard.validate_authority_for_bulk_shadow(t, device="CUDA:0", chunk_rows=65536, native_runtime=f.runtime), rejected=False)
            check("old token type", action=lambda f,t: guard.validate_authority_for_bulk_shadow(object(), device="CUDA:0", chunk_rows=65536, native_runtime=f.runtime))
            check("consumer device mismatch", action=lambda f,t: guard.validate_authority_for_bulk_shadow(t, device="CUDA:1", chunk_rows=65536, native_runtime=f.runtime))
            check("consumer chunk bool", action=lambda f,t: guard.validate_authority_for_bulk_shadow(t, device="CUDA:0", chunk_rows=True, native_runtime=f.runtime))
            check("consumer native identity mismatch", action=lambda f,t: guard.validate_authority_for_bulk_shadow(t, device="CUDA:0", chunk_rows=65536, native_runtime=dict(f.runtime, open3d="other")))
            check("immutable token", action=lambda f,t: setattr(t,"dual_report_sha256","f"*64))
            def isolated(f,t):
                value=t.runtime_binding; value["resident_configuration"]["device"]="other"
                assert t.runtime_binding["resident_configuration"]["device"]=="CUDA:0"
            check("runtime defensive copy", action=isolated, rejected=False)
            check("constructed wrong fixture token", action=lambda f,t: guard.validate_authority_for_bulk_shadow(replace(t,fixture_binding_sha256="f"*64), device="CUDA:0", chunk_rows=65536,native_runtime=f.runtime))
            def report_changed(f,t):
                f.dual_path.write_text("{}",encoding="utf-8")
                guard.validate_authority_for_bulk_shadow(t,device="CUDA:0",chunk_rows=65536,native_runtime=f.runtime)
            check("closed proof bytes changed after token", action=report_changed)
            def reconstructed_failed_report(f,t):
                f.report["real_runs"][1]["pairs"][0]["proposal_results"][0]["quality"]["passed"]=False
                f.save()
                forged=replace(t,dual_report_sha256=digest(f.dual_path))
                guard.validate_authority_for_bulk_shadow(forged,device="CUDA:0",chunk_rows=65536,native_runtime=f.runtime)
            check("constructed token cannot bypass failed gates", action=reconstructed_failed_report)
            assert config.call_count and current.call_count and inherited.call_count and resident.call_count

        # Direct unit mutations recompute enclosing hashes, so a stale outer
        # fingerprint cannot be the sole reason for rejecting malformed rows.
        restore()
        stats=fixture.report["real_runs"][1]["statistics_delta"]
        base=fixture.report["real_runs"][1]["bulk_records"]
        for name, mutate in (
            ("unknown original target",lambda r:r.update(target_digest="f"*64)),
            ("non-FP64 query",lambda r:r["audit_query_descriptor"].update(dtype="<f4")),
            ("boolean query shape",lambda r:r["audit_query_descriptor"].update(shape=[True,3])),
            ("query exceeds bound",lambda r:r["audit_query_descriptor"].update(shape=[1000001,3],bytes=24000024)),
            ("query byte extent wrong",lambda r:r["audit_query_descriptor"].update(bytes=1)),
            ("scalar/bulk ID hash mismatch",lambda r:r.update(bulk_ids_sha256="f"*64)),
            ("scalar/bulk metric hash mismatch",lambda r:r.update(bulk_squared_sha256="f"*64)),
            ("target replaced",lambda r:r.update(target_unchanged=False)),
            ("wrong strict radius",lambda r:r.update(radius_m=.031)),
            ("boolean hit count",lambda r:r.update(audit_hit_rows=True)),
            ("negative audit timer",lambda r:r.update(all_in_wall_s=-1)),
            ("nonfinite audit timer",lambda r:r.update(all_in_wall_s=float("nan"))),
            ("missing native bulk invocation",lambda r:r.update(evaluate_calls=0))):
            records=copy.deepcopy(base); mutate(records[0])
            try:
                guard.validate_bulk_records(records,stats,set(fixture.report["real_target_membership_sha256"]))
            except guard.GridProofError:
                pass
            else:
                raise AssertionError("Accepted invalid record: "+name)
            results.append({"case":name,"passed":True,"expected_rejection":True})

    # Execute only the extracted stdlib source-scope check, never the adapter
    # module with NumPy/Open3D imports.
    path=ROOT/"scripts/research/cuda_bulk_audit_grid_registration.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="unchanged_transport_scope")
    namespace={"ast":ast,"ROOT":ROOT,"Path":Path,"__file__":str(path)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),namespace)
    assert namespace["unchanged_transport_scope"]() is True
    results.append({"case":"AST exact copied transport/classifier/audit/scatter outside CPU block","passed":True})

    # Extract only the producer's argparse/preflight prefix. Stop before its
    # first directory creation/source/native imports, so actual CLI restrictions
    # are exercised without invoking numerical modules or touching old reports.
    producer=ROOT/"scripts/research/benchmark_bulk_resident_audit.py"
    tree=ast.parse(producer.read_text(encoding="utf-8"))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run")
    stop=next(i for i,node in enumerate(fn.body) if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
        and node.value.func.attr=="mkdir")
    fn.body=fn.body[:stop]+[ast.Return(value=ast.Name(id="args",ctx=ast.Load()))]
    fn=ast.fix_missing_locations(fn)
    namespace={"argparse":argparse,"Path":Path,"ROOT":ROOT,"os":os,"__doc__":"contract prefix",
        "EXPECTED_SOURCE":guard.device_proof.FROZEN}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(producer),"exec"),namespace)
    with tempfile.TemporaryDirectory(prefix="bulk-cli-stdlib-contract-") as folder:
        output=Path(folder)/"fresh.json"
        valid=["producer", "--run-allocated", "--api-proof", str(Path(folder)/"api.json"),
            "--audit-nearest", "--audit-misses", "--check-device-adapter", "--miss-policy",
            "direct-miss-research-v1", "--repeats", "1", "--output", str(output)]
        def cli_case(name, argv, rejected=True, env="8"):
            with patch.object(sys,"argv",argv), patch.dict(os.environ,{"OMP_NUM_THREADS":env}), \
                 contextlib.redirect_stderr(io.StringIO()):
                try:
                    args=namespace["run"]()
                except SystemExit as error:
                    if not rejected or error.code != 2:
                        raise
                else:
                    if rejected:
                        raise AssertionError("Accepted invalid producer CLI: "+name)
                    assert args.repeats==1 and args.audit_nearest and args.audit_misses
            results.append({"case":"producer CLI: "+name,"passed":True,"expected_rejection":rejected})
        cli_case("canonical exact dual command",valid,rejected=False)
        for flag in ("--run-allocated","--audit-nearest","--audit-misses","--check-device-adapter"):
            cli_case("missing "+flag,[x for x in valid if x!=flag])
        cli_case("synthetic-only cannot authorize complete dual",valid+["--synthetic-only"])
        cli_case("old proof pair cannot shortcut dual",valid+["--proof-synthetic","old-synthetic.json","--proof-bridge","old-bridge.json"])
        for flag,value in (("--repeats","2"),("--cache-mib","257"),("--max-clouds","0"),
                ("--scratch-mib","257"),("--query-mib","137"),("--expected-source-sha256","f"*64),
                ("--miss-policy","cpu-fallback")):
            cli_case("invalid "+flag+"="+value,valid+[flag,value])
        cli_case("wrong inherited OMP policy",valid,env="20")
        output.write_text("preserve existing report",encoding="utf-8")
        cli_case("existing output refused",valid)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refuse replacing an earlier guard contract report")
    results=contracts()
    report={"kind":"bulk-audit-stdlib-fault-contracts","status":"passed","cases":results,
        "passed_cases":len(results),"script_sha256":digest(__file__),
        "validator_sha256":digest(guard.__file__),
        "scope":"Fabricated contract fixtures; inherited independently tested device/source/math guards mocked. No native, GPU, numeric or performance proof.",
        "mocked_inherited_guards":["validate_configuration","validate_current_artifacts","common_checks","resident_checks"],
        "numerical_imports_run":False,"hardware_calls_run":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    print("PASS: %d pure-stdlib bulk authority contracts" % len(results))


if __name__=="__main__":
    main()
