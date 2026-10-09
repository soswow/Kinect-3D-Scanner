"""Explicit RANSAC-only protocol evidence, using artificial stdlib JSON."""

import copy
import unittest

from scripts.research import validate_checkpoint_finish_proof as proof


class ProposalPolicyContracts(unittest.TestCase):
    def setUp(self):
        self.gates = [{"gate_index":0,"function":"fragments._global_seed","result":None},
                      {"gate_index":1,"function":"fragments._global_seed","result":{"matrix":"unaltered"}}]
        rows = [{"invocation":index,"source":0,"target":1,"original_seed":8123456789+index,
                 "requested_threads":1,"complete":True,"restored_threads":20,
                 "wall_s":.1,"proposal_present":index == 1} for index in range(2)]
        self.report = {"scope_binding":{"pipeline_options":{"research_final_preplan":False,
                        "research_ransac_threads":1},"thread_policy":dict(proof.original.THREAD_POLICY),
                        "selected_indices":[0,1]},"research_ransac_threads":1,
                        "ransac_proposals":{"policy":proof.RANSAC_POLICY,"requested_threads":1,
                            "outside_threads":20,"calls":rows,"call_count":2,"failure":None,"hooks_restored":True}}

    def reject(self, report=None, gates=None):
        with self.assertRaises(proof.GridProofError):
            proof.validate_proposal_report(report or self.report, self.gates if gates is None else gates)

    def test_both_declared_budgets_preserve_original_icp_policy(self):
        for threads in (1,20):
            report = copy.deepcopy(self.report)
            report["scope_binding"]["pipeline_options"]["research_ransac_threads"] = threads
            report["research_ransac_threads"] = threads
            report["ransac_proposals"]["requested_threads"] = threads
            for row in report["ransac_proposals"]["calls"]:
                row["requested_threads"] = threads
            proof.validate_proposal_report(report, self.gates)
            self.assertEqual(report["scope_binding"]["thread_policy"], {"open3d":20,"opencv":20,"omp":"8"})
            self.assertEqual(report["ransac_proposals"]["calls"][0]["original_seed"], 8123456789)

    def test_absent_bool_float_and_unsupported_thread_budgets_are_rejected(self):
        for value in (None,True,1.0,"1",0,2,19,21):
            with self.subTest(value=value):
                self.report["scope_binding"]["pipeline_options"]["research_ransac_threads"] = value
                self.reject()

    def test_original_icp_cv_and_omp_cannot_be_changed_by_proposal_option(self):
        for key,value in (("open3d",1),("opencv",1),("omp","1")):
            report = copy.deepcopy(self.report)
            report["scope_binding"]["thread_policy"][key] = value
            self.reject(report)
        report = copy.deepcopy(self.report)
        report["scope_binding"]["pipeline_options"]["research_final_preplan"] = 1
        self.reject(report)

    def test_top_scope_row_and_runtime_report_policies_must_agree(self):
        for mutate in (lambda r:r.update(research_ransac_threads=20),
                       lambda r:r["ransac_proposals"].update(requested_threads=20),
                       lambda r:r["ransac_proposals"].update(policy="old-unscoped-thread-policy"),
                       lambda r:r["ransac_proposals"]["calls"][0].update(requested_threads=20)):
            report = copy.deepcopy(self.report)
            mutate(report)
            self.reject(report)

    def test_missing_suffix_duplicate_invocation_and_unobserved_gate_rejected(self):
        report = copy.deepcopy(self.report)
        report["ransac_proposals"]["calls"].pop()
        report["ransac_proposals"]["call_count"] = 1
        self.reject(report)
        report = copy.deepcopy(self.report)
        report["ransac_proposals"]["calls"][1]["invocation"] = 0
        self.reject(report)
        self.reject(gates=self.gates+[self.gates[0]])

    def test_thread_restoration_and_evidence_failures_never_authorize_quality(self):
        for mutate in (lambda r:r["ransac_proposals"].update(hooks_restored=False),
                       lambda r:r["ransac_proposals"].update(failure={"message":"restore failed"}),
                       lambda r:r["ransac_proposals"].update(outside_threads=1),
                       lambda r:r["ransac_proposals"]["calls"][0].update(restored_threads=1),
                       lambda r:r["ransac_proposals"]["calls"][0].update(complete=False),
                       lambda r:r["ransac_proposals"]["calls"][0].update(original_exception={"message":"failure"})):
            report = copy.deepcopy(self.report)
            mutate(report)
            self.reject(report)

    def test_source_target_seed_and_original_proposal_result_are_not_rounded_or_faked(self):
        for change in ({"source":2},{"target":True},{"original_seed":8123456789.0},
                       {"proposal_present":True},{"wall_s":float("nan")},{"wall_s":-1}):
            report = copy.deepcopy(self.report)
            report["ransac_proposals"]["calls"][0].update(change)
            self.reject(report)

    def test_empty_proposal_scope_is_valid_only_when_original_trace_also_has_none(self):
        self.report["ransac_proposals"].update(calls=[],call_count=0)
        proof.validate_proposal_report(self.report, [{"gate_index":0,"function":"other","result":True}])
        self.reject()


if __name__ == "__main__":
    unittest.main()
