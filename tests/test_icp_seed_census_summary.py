"""Small synthetic metadata census; does not create numerical evidence."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.research import summarize_icp_seed_reuse_census as summary


def fixture():
    source=summary.census;calls=[];captures=[];proposal_global=0
    for ci,counts in enumerate(((4,),(4,3,4,3,4,3,1,4))):
        capture={"capture":f"private{ci}","capture_sha256":"a"*64,"tasks":[]};captures.append(capture)
        for position,count in enumerate(counts):
            pair={"position":position,"pair":[position,position+1],"proposals":[],"original_pair_verdict":{"proposal_count":count},"input_bytes_unchanged":True,"seed_bytes_unchanged":True}
            capture["tasks"].append(pair)
            for pi in range(count):
                row={"proposal_index":pi,"complete":True,"result":None,"gates":[{"complete":True}],"calls":[]};pair["proposals"].append(row)
                for call_index in range(24 if proposal_global<24 else 23):
                    index=len(calls);matrix=[[1.,0.,0.,index*1e-13],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
                    seed_sha=hashlib.sha256(struct.pack("<16d",*(v for r in matrix for v in r))).hexdigest()
                    call={"context":{"capture_index":ci,"task":position,"pair":pair["pair"],"proposal_index":pi},"call_index":call_index,
                        "complete":True,"input_bytes_unchanged":True,"seed_values":matrix,
                        "input_binding":{"source":{"observed_original_cloud":index%292},"target":{"observed_original_cloud":0},"seed":{"sha256":seed_sha}},
                        "result":{"transformation":copy.deepcopy(matrix),"fitness":.5,"inlier_rmse":.01,"correspondence_mapping":{"sha256":"b"*64},"raw_correspondences":{"sha256":"b"*64}}}
                    row["calls"].append(call);calls.append(call)
                proposal_global+=1
    artifacts={n:summary.sha(summary.ROOT/n) for n in source.OWN_FILES+tuple(source.HELD_FILES)}
    binding={"source_sha256":source.CURRENT,"artifacts_sha256":artifacts,"fixed_files":{"private0":"a"*64,"private1":"a"*64,"binary":"b"*64},
        "verification_dependencies":{"original":"recorded"},"runtime":{"registration":"original legacy CPU","thread_policy":{"opencv":20,"open3d":20,"omp":"8"},
            "environment":dict(source.ENVIRONMENT),"binary_sha256":{"binary":"b"*64}}}
    return {"kind":source.KIND,"status":"passed","failure":None,"cleanup_passed":True,"cleanup_failures":[],"original_cpu_results_returned_unchanged":True,
        "skipped_icp_calls":0,"timing_authority":False,"whole_finish_authority":False,"binding":binding,"binding_after":copy.deepcopy(binding),"captures":captures,"call_count":714,
        "ordered_calls_sha256":hashlib.sha256(source.scope.canonical(calls).encode()).hexdigest(),"clustering":[source.cluster_rows(calls,t) for t in source.THRESHOLDS]}


class CensusSummaryTests(unittest.TestCase):
    def test_complete_scalar_census_has_no_private_groups_seeds_clouds_or_paths(self):
        result=summary.compact(fixture())
        self.assertEqual((result["pairs"],result["genuine_proposals"],result["original_cpu_calls"],result["exact_directed_cloud_value_buckets"]),(9,30,714,292))
        self.assertEqual(len(result["thresholds"]),5)
        self.assertFalse(result["speed_authority"]);self.assertFalse(result["safe_reuse_established"])
        encoded=json.dumps(result)
        for private in ('"groups"','"seed_values"','"captures"','private0','observed_original_cloud','"input_binding"'):self.assertNotIn(private,encoded)

    def test_failure_cleanup_skipping_or_authority_claim_refused(self):
        base=fixture()
        for key,value in (("status","running"),("failure","fault"),("cleanup_failures",["fault"]),("cleanup_passed",False),
            ("original_cpu_results_returned_unchanged",False),("skipped_icp_calls",1),("timing_authority",True),("whole_finish_authority",True)):
            report=copy.deepcopy(base);report[key]=value
            with self.assertRaises(ValueError):summary.compact(report)

    def test_missing_call_proposal_or_cloud_bucket_scope_refused(self):
        base=fixture()
        for mutate in (lambda r:r["captures"][0]["tasks"][0]["proposals"].pop(),
            lambda r:r["captures"][0]["tasks"][0]["proposals"][0]["calls"].pop(),
            lambda r:r["clustering"][0].update(exact_cloud_value_buckets=291)):
            report=copy.deepcopy(base);mutate(report)
            with self.assertRaises(ValueError):summary.compact(report)

    def test_ordered_input_result_hash_and_recomputed_groups_refuse_tampering(self):
        base=fixture()
        for mutate in (lambda r:r["captures"][0]["tasks"][0]["proposals"][0]["calls"][0]["result"].update(fitness=.7),
            lambda r:r["captures"][0]["tasks"][0]["proposals"][0]["calls"][0].update(call_index=4),
            lambda r:r["clustering"][1]["max_result_deltas"].update(transform_max_abs_delta=99.),
            lambda r:r["clustering"][1].update(safe_reuse_established=True)):
            report=copy.deepcopy(base);mutate(report)
            with self.assertRaises(ValueError):summary.compact(report)

    def test_source_runtime_receipt_and_loaded_helper_mutation_refused(self):
        base=fixture()
        report=copy.deepcopy(base);report["binding_after"]["source_sha256"]="f"*64
        with self.assertRaises(ValueError):summary.compact(report)
        report=copy.deepcopy(base);report["binding"]["runtime"]["thread_policy"]["open3d"]=8;report["binding_after"]=copy.deepcopy(report["binding"])
        with self.assertRaises(ValueError):summary.compact(report)
        for fn in (summary.census.cluster_rows,summary.census.scope.canonical):
            code=fn.__code__
            try:
                fn.__code__=(lambda *args:None).__code__
                with self.assertRaises(ValueError):summary.check_helpers()
            finally:fn.__code__=code

    def test_fresh_output_and_failed_diagnostics_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=summary.ROOT/"benchmark-output") as folder:
            folder=Path(folder);source=folder/"input.json";output=folder/"summary.json"
            report=fixture();source.write_text(json.dumps(report),encoding="utf-8")
            args=SimpleNamespace(report=source,output=output);summary.run(args)
            record=json.loads(output.read_text(encoding="utf-8"));self.assertEqual(record["status"],"passed")
            self.assertFalse(record["fixed_private_resources_rescanned"]);self.assertFalse(record["os_child_exit_independently_verified"])
            previous=output.read_bytes()
            with self.assertRaises(ValueError):summary.run(args)
            self.assertEqual(output.read_bytes(),previous)
            report["skipped_icp_calls"]=1;source.write_text(json.dumps(report),encoding="utf-8")
            failed=folder/"failed.json"
            with self.assertRaises(ValueError):summary.run(SimpleNamespace(report=source,output=failed))
            self.assertEqual(json.loads(failed.read_text(encoding="utf-8"))["status"],"failed")


if __name__=="__main__":unittest.main()
