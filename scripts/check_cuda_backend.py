"""Backend regression checks in the measured server environment (no Qt GUI)."""
import json
import argparse
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
from scripts.process_metrics import finish_cuda_worker, gpu_info
from scripts.profile_session import source_hash

MODULES = [
    "tests.test_cuda_fusion", "tests.test_cuda_tracking", "tests.test_icp_source_cache",
    "tests.test_features", "tests.test_quality", "tests.test_fragments", "tests.test_final_budgets",
    "tests.test_native_fusion", "tests.test_depth_confidence", "tests.test_session_confidence",
    "tests.test_appearance", "tests.test_replay", "tests.test_api", "tests.test_live_api",
    "tests.test_live_tracking_speed", "tests.test_photometric", "tests.test_profile_comparison", "tests.test_adaptive_visual",
    "tests.test_fusion_failure",
    "tests.test_input_failure",
    "tests.test_cuda_input",
    "tests.test_cuda_confidence",
]
EXCLUDED = {
    "tests.test_features.FeatureTests.test_partial_batch_recording_uses_individual_acknowledgements",
    "tests.test_quality.QualityTests.test_frame_batches_preserve_command_barriers",
}

def flat(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flat(test)
        else:
            yield test

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "benchmark-output/cuda-pipeline/validation.json")
args = parser.parse_args()
fingerprint = source_hash()
suite = unittest.TestSuite(test for test in flat(unittest.defaultTestLoader.loadTestsFromNames(MODULES))
                           if test.id() not in EXCLUDED)
result = unittest.TextTestRunner(verbosity=2).run(suite)
unchanged = source_hash() == fingerprint
report = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
          "skipped": result.skipped, "success": result.wasSuccessful() and unchanged, "modules": MODULES,
          "excluded": sorted(EXCLUDED), "gpu_hardware": gpu_info(),
          "source_sha256": fingerprint, "source_changed_during_validation": not unchanged,
          "note": "Qt client tests excluded; these are backend checks for the CUDA server environment."}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2)+"\n")
if not report["success"]:
    raise SystemExit(1)
import open3d as o3d
if o3d.core.cuda.is_available():
    o3d.core.cuda.synchronize()
    finish_cuda_worker()
