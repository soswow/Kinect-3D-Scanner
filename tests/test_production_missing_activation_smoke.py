"""Stdlib-only schema and failure gates; physical bits require allocated CUDA."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.research import benchmark_production_missing_activation as smoke


def complete_pair():
    def frames(capacities):return [dict(frustum_rows=rows,active_keys=active,capacity=capacity)
        for rows,active,capacity in zip((3,3,4),(3,4,4),capacities)]
    return dict(inputs_unchanged=True,nonzero_weight_voxels=3,fractional_confidence_pixels=9,
        fractional_weight_voxels=2,ordinary=dict(initial_capacity=4,final_capacity=8,final_blocks=4,frames=frames((4,8,8))),
        candidate=dict(initial_capacity=4,final_capacity=4,final_blocks=4,frames=frames((4,4,4))),
        comparison={name:dict(elements=4,bit_mismatches=0) for name in ("tsdf","weight","color")})


class SmokeContracts(unittest.TestCase):
    def test_complete_positive_physical_schema_is_required(self):
        pair=complete_pair();smoke.validate_pair(pair)
        self.assertTrue(pair["complete"])
    def test_any_single_backend_attribute_bit_fault_fails(self):
        for name in ("tsdf","weight","color"):
            pair=complete_pair();pair["comparison"][name]["bit_mismatches"]=1
            with self.assertRaises(RuntimeError):smoke.validate_pair(pair)
            self.assertNotIn("complete",pair)
    def test_inputs_nonzero_or_fractional_evidence_cannot_be_missing(self):
        for name in ("inputs_unchanged","nonzero_weight_voxels","fractional_confidence_pixels","fractional_weight_voxels"):
            pair=complete_pair();pair[name]=False if name=="inputs_unchanged" else 0
            with self.assertRaises(RuntimeError):smoke.validate_pair(pair)
    def test_capacity_claim_is_exact_for_both_installed_routes(self):
        for mode,value in (("ordinary",4),("ordinary",16),("candidate",8)):
            pair=complete_pair();pair[mode]["final_capacity"]=value
            with self.assertRaises(RuntimeError):smoke.validate_pair(pair)
    def test_incomplete_frames_and_missing_keys_fail(self):
        for mode in ("ordinary","candidate"):
            pair=complete_pair();pair[mode]["frames"].pop()
            with self.assertRaises(RuntimeError):smoke.validate_pair(pair)
            pair=complete_pair();pair[mode]["final_blocks"]=3
            with self.assertRaises(RuntimeError):smoke.validate_pair(pair)
    def test_private_flag_is_the_only_engine_policy_difference(self):
        ordinary=smoke.policy_engine("CUDA:0",False);candidate=smoke.policy_engine("CUDA:0",True)
        for key in ordinary.__dict__:
            if key=="_final_missing_only_activation":continue
            left,right=getattr(ordinary,key),getattr(candidate,key)
            self.assertEqual(left,right)
        self.assertIs(ordinary._final_missing_only_activation,False)
        self.assertIs(candidate._final_missing_only_activation,True)
        self.assertIs(candidate.settings.confidence_fusion,True)
    def test_allocation_is_rejected_before_native_imports(self):
        with self.assertRaisesRegex(ValueError,"allocated"):smoke.run(SimpleNamespace(run_allocated=False))
    def test_immutable_helper_drift_stops_before_numerical_imports(self):
        args=SimpleNamespace(run_allocated=True,output=smoke.ROOT/"benchmark-output/never-existing-smoke-contract.json")
        with patch.object(smoke,"file_hash",return_value="changed"),self.assertRaisesRegex(RuntimeError,"helper changed"):
            smoke.run(args)
    def test_current_production_helper_must_be_installed_before_native_imports(self):
        args=SimpleNamespace(run_allocated=True,output=smoke.ROOT/"benchmark-output/never-existing-smoke-contract.json")
        with (patch.object(smoke,"file_hash",return_value=smoke.HELPER_SHA256),
                patch.object(Path,"is_file",return_value=False),self.assertRaisesRegex(RuntimeError,"not installed")):
            smoke.run(args)
    def test_fresh_output_guard_preserves_existing_report(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"report.json";path.write_text("original bytes",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"fresh"):
                smoke.run(SimpleNamespace(run_allocated=True,output=path))
            self.assertEqual(path.read_text(encoding="utf-8"),"original bytes")
    def fault_run(self,path,**options):
        return (patch.object(smoke,"file_hash",return_value=smoke.HELPER_SHA256),
            patch.object(Path,"is_file",return_value=True),
            patch.object(smoke,"source_hash",return_value="0"*64),
            patch.object(smoke,"pins",**options),
            patch.object(smoke,"peak_rss_bytes",return_value=0),
            patch.object(smoke.os.environ,"update",side_effect=RuntimeError("original setup failure")))
    def test_primary_setup_and_secondary_cleanup_failures_are_both_preserved(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=smoke.ROOT/"benchmark-output") as folder:
            path=Path(folder)/"report.json"
            contexts=self.fault_run(path,side_effect=[{"source":"0"*64},RuntimeError("source cleanup failure")])
            from contextlib import ExitStack
            with ExitStack() as stack:
                for context in contexts:stack.enter_context(context)
                with self.assertRaisesRegex(RuntimeError,"physical smoke failed"):
                    smoke.run(SimpleNamespace(run_allocated=True,output=path))
            value=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["status"],"failed")
            self.assertEqual(value["failure"]["message"],"original setup failure")
            self.assertEqual(value["cleanup_failures"][0]["message"],"source cleanup failure")
    def test_final_diagnostic_failure_does_not_mask_primary(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=smoke.ROOT/"benchmark-output") as folder:
            path=Path(folder)/"report.json";actual_write=Path.write_text;count=0
            def write(target,*args,**kwargs):
                nonlocal count
                count+=1
                if count==2:raise OSError("final diagnostics failure")
                return actual_write(target,*args,**kwargs)
            from contextlib import ExitStack
            with ExitStack() as stack:
                for context in self.fault_run(path,return_value={"source":"0"*64}):stack.enter_context(context)
                stack.enter_context(patch.object(Path,"write_text",write))
                with self.assertRaisesRegex(RuntimeError,"original setup failure") as failure:
                    smoke.run(SimpleNamespace(run_allocated=True,output=path))
            self.assertIsInstance(failure.exception.__cause__,OSError)


if __name__=="__main__":unittest.main()
