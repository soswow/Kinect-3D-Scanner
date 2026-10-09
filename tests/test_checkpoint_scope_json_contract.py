"""Pure-stdlib JSON scope normalization; no native/image/GPU imports."""

import json
import math
import struct
import unittest

from scripts.research.profile_checkpoint_resident_finish import canonical_json_scope


class CheckpointScopeJsonContract(unittest.TestCase):
    def test_calibration_tuple_schema_becomes_exact_json_lists(self):
        original = {"settings": {"camera": {"distortion": (0., -0., .125, -.125, 0.)},
            "roi": (0, 1, 640, 480), "sensor_calibration": {
                "rotation": ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                "translation_mm": (12.345678901234567, 0., -0.)}},
            "seed": 0, "selected_indices": [0, 1], "native": True, "unsupported": None}
        value = canonical_json_scope(original)
        self.assertEqual(value, json.loads(json.dumps(original, allow_nan=False)))
        self.assertIsInstance(value["settings"]["camera"]["distortion"], list)
        self.assertIsInstance(value["settings"]["sensor_calibration"]["rotation"][0], list)
        self.assertEqual(value, json.loads(json.dumps(value, allow_nan=False)))
        self.assertIsInstance(original["settings"]["camera"]["distortion"], tuple)

    def test_finite_double_bits_and_integers_preserved(self):
        values = [0., -0., math.nextafter(1., 0.), math.nextafter(1., 2.),
            math.ldexp(1., -1074), math.ldexp(1., 1023)]
        result = canonical_json_scope({"values": tuple(values), "counter": 2**63-1})
        self.assertEqual(result["counter"], 2**63-1)
        for before, after in zip(values, result["values"]):
            self.assertEqual(struct.pack("<d", before), struct.pack("<d", after))

    def test_nonfinite_and_nonjson_objects_fail(self):
        for value in (math.nan, math.inf, -math.inf, object()):
            with self.subTest(value=repr(value)):
                with self.assertRaises((ValueError, TypeError)):
                    canonical_json_scope({"value": value})

    def test_normalization_detaches_mutable_scope(self):
        original = {"settings": {"values": [1, 2]}}
        normalized = canonical_json_scope(original)
        normalized["settings"]["values"].append(3)
        self.assertEqual(original, {"settings": {"values": [1, 2]}})


if __name__ == "__main__":
    unittest.main()
