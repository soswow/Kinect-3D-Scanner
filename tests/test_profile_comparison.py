"""Preview experiments must not hide different export detail or camera inputs."""

import copy
import unittest

from scripts.profile_session import comparison_settings


class PreviewComparisonTests(unittest.TestCase):
    def report(self):
        return {"settings": {"voxel_m": .005, "final_voxel_m": None,
            "final_block_count": 10000, "camera": {"fx": 525}, "min_fitness": .35},
            "finish_requested": True, "mesh_built": True, "final_reconstruction": {"voxel_m": .005}}

    def test_only_live_resolution_and_implicit_final_resolution_are_normalized(self):
        reference = self.report()
        candidate = copy.deepcopy(reference)
        candidate["settings"].update(voxel_m=.01, final_voxel_m=.005)
        self.assertNotEqual(comparison_settings(candidate), comparison_settings(reference))
        self.assertEqual(comparison_settings(candidate, preview_resolution=True),
                         comparison_settings(reference, preview_resolution=True))
        candidate["settings"]["camera"]["fx"] = 526
        self.assertNotEqual(comparison_settings(candidate, preview_resolution=True),
                            comparison_settings(reference, preview_resolution=True))

    def test_different_final_detail_and_volume_budget_remain_different(self):
        reference = self.report()
        for key, value in (("final_block_count", 5000), ("min_fitness", .25),
                           ("final_voxel_m", .01)):
            candidate = copy.deepcopy(reference)
            candidate["settings"][key] = value
            if key == "final_voxel_m":
                candidate["final_reconstruction"]["voxel_m"] = value
            self.assertNotEqual(comparison_settings(candidate, preview_resolution=True),
                                comparison_settings(reference, preview_resolution=True))

    def test_unfinished_or_wrong_actual_export_voxel_cannot_pass(self):
        for key, value in (("finish_requested", False),
                           ("mesh_built", False),
                           ("final_reconstruction", {"voxel_m": .01})):
            candidate = self.report()
            candidate[key] = value
            with self.assertRaisesRegex(ValueError, "completed mesh"):
                comparison_settings(candidate, preview_resolution=True)
