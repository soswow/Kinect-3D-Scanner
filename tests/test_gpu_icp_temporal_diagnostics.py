"""Mocked metadata contracts; no geometry/numerical authority or imports."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.research import compare_gpu_icp_temporal_diagnostics as supplement


def fixture():
    binding = {"source_sha256": "a"*64, "runtime": {"device": "CUDA:0", "threads": 20}}
    fragment = {"fragments": [{"id": 0}, {"id": 1}], "candidate_pairs": 1,
        "verified_bridges": [{"source": 0, "target": 1, "validation_scope": "temporal camera pair",
                              "temporal_constraint": True}],
        "temporal_bridges": 1, "ambiguous_temporal_pairs": []}
    profile = {"source_sha256": "a"*64, "input_sha256": "b"*64, "selected_indices": [0, 1],
        "seed": 0, "settings": {"confidence_fusion": True}, "pipeline_options": {},
        "thread_policy": {"open3d": 20}, "accepted_indices": [0, 1], "mesh_built": True,
        "finish_requested": True, "pose_seeds_used": False, "input_changed_during_profile": False,
        "source_changed_during_profile": False, "fragment_reconnection": fragment,
        "geometry": {"artifact": "/private/native.npz", "sha256": "c"*64}}
    def report(mode):
        return {"kind": "gpu-icp-whole-finish-candidate-"+mode+"-v1", "mode": mode,
            "status": "passed", "failure": None, "cleanup_passed": True, "cleanup_failures": [],
            "binding": copy.deepcopy(binding), "binding_after": copy.deepcopy(binding),
            "scope_binding": {"checkpoint_sha256": "d"*64}, "profile": copy.deepcopy(profile),
            "registration": {"mode": mode, "complete": True, "closed": True, "restored": True,
                "failure": None, "cleanup_failures": [], "original_build_calls": 1,
                "successful_builds": 1, "default_evidence": True},
            "candidate_source": {"candidate_method": {"policy": "actual-current"}},
            "candidate_source_after": {"candidate_method": {"policy": "actual-current"}}}
    native, candidate = report("native"), report("audit")
    candidate["profile"]["geometry"] = {"artifact": "/private/candidate.npz", "sha256": "e"*64}
    refs = {"native_ref": {"path": "/private/native.json", "sha256": "1"*64},
            "candidate_ref": {"path": "/private/candidate.json", "sha256": "2"*64}, "current_source": "a"*64}
    prior = {"kind": supplement.PRIOR_KIND, "status": "passed", "failure": None,
        "comparator_sha256": supplement.COMPARATOR_SHA256,
        "native_report": copy.deepcopy(refs["native_ref"]), "candidate_report": copy.deepcopy(refs["candidate_ref"]),
        "checks": {key: True for key in ("same_checkpoint_raw_settings_and_accepted_ids",
            "same_published_bridge_witness_frontier_memberships", "pose_bounds", "fixed_coordinate_surface_bounds")},
        "metrics": {"pose_translation_max_m": .0001, "pose_rotation_max_deg": .001},
        "surface": {"surface_p95_m": 1e-7, "precision": 1., "completeness": 1.},
        "reference_geometry": {"path": "/private/native.npz", "sha256": "c"*64},
        "candidate_geometry": {"path": "/private/candidate.npz", "sha256": "e"*64}}
    return native, candidate, prior, refs


class TemporalContracts(unittest.TestCase):
    def test_positive_retains_presence_and_authority_limits(self):
        a, b, q, refs = fixture()
        result = supplement.compare_documents(a, b, q, **refs)
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["diagnostics"]["temporal_bridges"],
            {"presence": "present", "availability": "reported", "value": 1})
        self.assertEqual(result["diagnostics"]["rejected_fallback_boundaries"]["presence"], "missing")
        self.assertFalse(any(result["limits"].values()))

    def test_every_requested_diagnostic_difference_refused(self):
        for field in supplement.FIELDS:
            with self.subTest(field=field):
                a, b, q, refs = fixture()
                b["profile"]["fragment_reconnection"][field] = 2 if field == "temporal_bridges" else [[0, 1]]
                with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)

    def test_missing_optional_conflict_is_not_an_explicit_empty_list(self):
        a, b, q, refs = fixture()
        b["profile"]["fragment_reconnection"]["rejected_optimized_boundaries"] = []
        with self.assertRaisesRegex(ValueError, "diagnostics differ"):
            supplement.compare_documents(a, b, q, **refs)

    def test_unpublished_early_temporal_pass_has_explicit_availability(self):
        a, b, q, refs = fixture()
        for report in (a, b):
            report["profile"]["fragment_reconnection"] = {"fragments": [], "verified_bridges": [],
                "reason": "No skipped views or trajectory to revalidate"}
        result = supplement.compare_documents(a, b, q, **refs)
        self.assertEqual(result["diagnostics"]["temporal_bridges"],
            {"presence": "missing", "availability": "temporal_pass_not_published"})

    def test_required_pass_field_cannot_disappear_after_graph_receipts(self):
        for field in supplement.FIELDS[:2]:
            a, b, q, refs = fixture()
            for report in (a, b): del report["profile"]["fragment_reconnection"][field]
            with self.assertRaisesRegex(ValueError, "Missing required"):
                supplement.compare_documents(a, b, q, **refs)

    def test_temporal_camera_edge_requires_actual_true_flag(self):
        for value in (False, 1, None):
            a, b, q, refs = fixture()
            b["profile"]["fragment_reconnection"]["verified_bridges"][0]["temporal_constraint"] = value
            with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)

    def test_optional_false_flag_and_missing_flag_are_distinct(self):
        a, b, q, refs = fixture()
        for report in (a, b):
            edge = report["profile"]["fragment_reconnection"]["verified_bridges"][0]
            edge["validation_scope"] = "independent camera pairs"
            del edge["temporal_constraint"]
        b["profile"]["fragment_reconnection"]["verified_bridges"][0]["temporal_constraint"] = False
        with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)

    def test_malformed_count_pairs_and_ids_refused(self):
        for field, value in (("temporal_bridges", True), ("ambiguous_temporal_pairs", [[0, 2]]),
                             ("rejected_fallback_boundaries", [[0, 1], [0, 1]])):
            a, b, q, refs = fixture()
            b["profile"]["fragment_reconnection"][field] = value
            with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)

    def test_closed_envelopes_required(self):
        for key, value in (("status", "failed"), ("failure", {}), ("cleanup_passed", False), ("cleanup_failures", ["late"])):
            a, b, q, refs = fixture(); b[key] = value
            with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)
        a, b, q, refs = fixture(); b["registration"]["restored"] = False
        with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)

    def test_missing_envelope_is_unavailable_not_successful_zero_evidence(self):
        for key, value in (("mode", None), ("registration", None), ("profile", None)):
            a, b, q, refs = fixture(); b[key] = value
            with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)
        a, b, q, refs = fixture(); a["scope_binding"] = {}; b["scope_binding"] = {}
        with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)

    def test_source_runtime_checkpoint_and_current_version_exact(self):
        for key in ("runtime", "source_sha256"):
            a, b, q, refs = fixture(); b["binding"][key] = "changed"; b["binding_after"] = copy.deepcopy(b["binding"])
            with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)
        a, b, q, refs = fixture(); b["scope_binding"] = {"checkpoint_sha256": "0"*64}
        with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)
        a, b, q, refs = fixture(); refs["current_source"] = "0"*64
        with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)

    def test_prior_comparison_cannot_be_bypassed_or_relabelled(self):
        for key, value in (("status", "failed"), ("kind", "old-quality"), ("comparator_sha256", "0"*64)):
            a, b, q, refs = fixture(); q[key] = value
            with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)
        a, b, q, refs = fixture(); q["checks"]["pose_bounds"] = False
        with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)

    def test_prior_report_and_geometry_references_must_bind_exact_bytes(self):
        for key in ("native_report", "candidate_report", "reference_geometry", "candidate_geometry"):
            a, b, q, refs = fixture(); q[key]["sha256"] = "0"*64
            with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)

    def test_matching_malformed_digests_are_not_exact_byte_evidence(self):
        for value in ("", "not-a-digest", "F"*64):
            a, b, q, refs = fixture()
            refs["native_ref"]["sha256"] = q["native_report"]["sha256"] = value
            with self.assertRaisesRegex(ValueError, "digest"):
                supplement.compare_documents(a, b, q, **refs)

    def test_prior_numeric_bounds_and_finiteness_not_just_pass_flags(self):
        for mapping, key, value in (("metrics", "pose_rotation_max_deg", .101),
                ("surface", "precision", .998), ("surface", "surface_p95_m", float("nan"))):
            a, b, q, refs = fixture(); q[mapping][key] = value
            with self.assertRaises(ValueError): supplement.compare_documents(a, b, q, **refs)

    def test_fresh_import_has_no_numerical_or_authority_modules(self):
        code = "from scripts.research import compare_gpu_icp_temporal_diagnostics; import sys; assert not any(n in sys.modules for n in ('numpy','cupy','open3d','cv2','scripts.research.compare_gpu_icp_candidate_finishes'))"
        run = subprocess.run([sys.executable, "-S", "-c", code], cwd=supplement.ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_cli_failure_is_saved_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths = [root/name for name in ("native.json", "candidate.json", "quality.json")]
            for path in paths: path.write_text("{}", encoding="utf-8")
            output = root/"benchmark-output/result.json"
            with patch.object(supplement, "ROOT", root), patch.object(supplement, "sha", side_effect=RuntimeError("primary read")):
                with self.assertRaisesRegex(RuntimeError, "primary read"):
                    supplement.main([*map(str, paths), "--output", str(output)])
            saved = output.read_bytes(); self.assertEqual(json.loads(saved)["status"], "failed")
            with patch.object(supplement, "ROOT", root):
                with self.assertRaises(ValueError): supplement.main([*map(str, paths), "--output", str(output)])
            self.assertEqual(output.read_bytes(), saved)

    def test_failed_report_write_does_not_replace_primary_read_fault(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths = [root/name for name in ("native.json", "candidate.json", "quality.json")]
            for path in paths: path.write_text("{}", encoding="utf-8")
            primary = RuntimeError("primary read")
            with patch.object(supplement, "ROOT", root), patch.object(supplement, "sha", side_effect=primary), \
                 patch.object(Path, "open", side_effect=OSError("secondary write")):
                with self.assertRaises(RuntimeError) as caught:
                    supplement.main([*map(str, paths), "--output", str(root/"benchmark-output/result.json")])
            self.assertIs(caught.exception, primary)
            self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_late_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths = [root/name for name in ("native.json", "candidate.json", "quality.json")]
            for path in paths: path.write_text("{}", encoding="utf-8")
            output = root/"benchmark-output/result.json"
            primary = RuntimeError("read fault after another writer")
            def late_writer(path):
                output.write_text("preserved existing evidence", encoding="utf-8")
                raise primary
            with patch.object(supplement, "ROOT", root), patch.object(supplement, "sha", side_effect=late_writer):
                with self.assertRaises(RuntimeError) as caught:
                    supplement.main([*map(str, paths), "--output", str(output)])
            self.assertIs(caught.exception, primary)
            self.assertIsInstance(caught.exception.__cause__, FileExistsError)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserved existing evidence")


if __name__ == "__main__":
    unittest.main()
