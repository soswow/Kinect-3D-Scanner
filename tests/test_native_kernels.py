"""Native image kernels preserve the portable reconstruction reference.

The numerical tests skip when the optional extension is not installed. Loader
tests always run, including the explicit mode that must never time a fallback.
"""

import importlib
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import shared.native as native_loader
from shared.calibration import (
    _color_maps_numpy,
    pinhole_rays,
    prepare_native_rgbd,
)
from shared.depth import _prepare_depth_numpy, prepare_depth
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings


class NativeLoaderTests(unittest.TestCase):
    def setUp(self):
        native_loader._extension.cache_clear()

    def tearDown(self):
        native_loader._extension.cache_clear()

    def test_off_never_imports_extension(self):
        with (
            patch.dict(os.environ, {"KINECT_NATIVE": "off"}),
            patch.object(
                native_loader.importlib,
                "import_module",
                side_effect=AssertionError("unexpected import"),
            ),
        ):
            self.assertIsNone(native_loader.kernels())
            self.assertEqual(
                native_loader.native_status(),
                {
                    "requested": "off",
                    "active": False,
                    "api_version": None,
                    "fallback_reason": None,
                },
            )

    def test_auto_falls_back_for_missing_or_unloadable_extension(self):
        for failure in (ImportError("not installed"), OSError("wrong architecture")):
            with (
                self.subTest(failure=failure),
                patch.dict(os.environ, {"KINECT_NATIVE": "auto"}),
                patch.object(
                    native_loader.importlib, "import_module", side_effect=failure
                ) as imported,
            ):
                native_loader._extension.cache_clear()
                self.assertIsNone(native_loader.kernels())
                status = native_loader.native_status()
                self.assertFalse(status["active"])
                self.assertEqual(status["fallback_reason"], str(failure))
                imported.assert_called_once_with("_kinect_native")

    def test_on_requires_extension_instead_of_silently_falling_back(self):
        for failure in (ImportError("not installed"), OSError("wrong architecture")):
            with (
                self.subTest(failure=failure),
                patch.dict(os.environ, {"KINECT_NATIVE": "on"}),
                patch.object(
                    native_loader.importlib, "import_module", side_effect=failure
                ),
            ):
                native_loader._extension.cache_clear()
                with self.assertRaisesRegex(
                    RuntimeError, "Native kernels requested but unavailable"
                ):
                    native_loader.kernels()

    def test_incompatible_api_is_fallback_in_auto_and_error_in_on(self):
        for api in (None, 0, 1, 3):
            module = SimpleNamespace(API_VERSION=api)
            for mode in ("auto", "on"):
                with (
                    self.subTest(api=api, mode=mode),
                    patch.dict(os.environ, {"KINECT_NATIVE": mode}),
                    patch.object(
                        native_loader.importlib, "import_module", return_value=module
                    ),
                ):
                    native_loader._extension.cache_clear()
                    if mode == "on":
                        with self.assertRaisesRegex(RuntimeError, "incompatible API"):
                            native_loader.kernels()
                    else:
                        self.assertIsNone(native_loader.kernels())
                        self.assertIn(
                            "incompatible API",
                            native_loader.native_status()["fallback_reason"],
                        )

    def test_compatible_extension_is_cached_and_reports_active(self):
        module = SimpleNamespace(API_VERSION=2)
        with (
            patch.dict(os.environ, {"KINECT_NATIVE": "ON"}),
            patch.object(
                native_loader.importlib, "import_module", return_value=module
            ) as imported,
        ):
            self.assertIs(native_loader.kernels(), module)
            self.assertEqual(
                native_loader.native_status(),
                {
                    "requested": "on",
                    "active": True,
                    "api_version": 2,
                    "fallback_reason": None,
                },
            )
            imported.assert_called_once_with("_kinect_native")

    def test_default_mode_is_auto_and_invalid_modes_are_rejected(self):
        with patch.dict(os.environ, clear=False):
            os.environ.pop("KINECT_NATIVE", None)
            self.assertEqual(native_loader.native_mode(), "auto")
        for mode in ("yes", "", "native"):
            with (
                self.subTest(mode=mode),
                patch.dict(os.environ, {"KINECT_NATIVE": mode}),
                self.assertRaisesRegex(ValueError, "auto, on, or off"),
            ):
                native_loader.kernels()


class NativeKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.native = importlib.import_module("_kinect_native")
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest(
                f"Optional native extension unavailable: {exc}"
            ) from exc
        if getattr(cls.native, "API_VERSION", None) != 2:
            raise AssertionError(
                "Rebuild the installed native extension for API_VERSION=2"
            )
        cls.calibration = load_calibration()

    def native_depth(self, depth, settings):
        h, w = depth.shape
        x0, y0, x1, y1 = settings.roi or (0, 0, w, h)
        x0, x1, _ = slice(x0, x1).indices(w)
        y0, y1, _ = slice(y0, y1).indices(h)
        return self.native.prepare_depth(
            depth,
            settings.near_m * 1000,
            settings.far_m * 1000,
            settings.filter_depth,
            x0,
            y0,
            x1,
            y1,
        )

    def assert_depth_parity(self, depth, settings):
        before = depth.copy()
        actual = self.native_depth(depth, settings)
        np.testing.assert_array_equal(actual, _prepare_depth_numpy(depth, settings))
        np.testing.assert_array_equal(depth, before)
        self.assertEqual(actual.dtype, np.uint16)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertFalse(np.shares_memory(actual, depth))
        return actual

    def projection(self, metric, depth, settings, rgb_shape):
        c = settings.rgb_camera
        points_ir = pinhole_rays(settings.camera) * metric[..., None]
        points_rgb = (
            points_ir.reshape(-1, 3)
            @ np.asarray(settings.sensor_calibration.rotation).T
            + settings.sensor_calibration.translation_mm
        ).reshape(*metric.shape, 3)
        return self.native.project_native(
            points_rgb,
            depth,
            np.array([c.fx, c.fy, c.cx, c.cy, *c.distortion], dtype=np.float64),
            rgb_shape[0],
            rgb_shape[1],
        )

    def test_depth_fractional_and_inclusive_bounds(self):
        depth = np.array([[0, 499, 500, 501, 999, 1000, 1001, 65535]], np.uint16)
        for near, far in ((0.5, 1), (0.5004, 1.0004), (0.4996, 0.9996), (0, 8)):
            settings = ScanSettings(near_m=near, far_m=far, filter_depth=False)
            with self.subTest(near=near, far=far):
                self.assert_depth_parity(depth, settings)
        np.testing.assert_array_equal(
            self.native_depth(
                depth, ScanSettings(near_m=0.5004, far_m=1.0004, filter_depth=False)
            ),
            [[0, 0, 0, 501, 999, 1000, 0, 0]],
        )

    def test_filter_preserves_holes_thin_surfaces_and_image_boundaries(self):
        thin = np.zeros((7, 7), dtype=np.uint16)
        thin[:, 3] = 1000
        hole = np.full((5, 5), 1000, dtype=np.uint16)
        hole[2, 2] = 0
        settings = ScanSettings()
        cases = (
            thin,
            hole,
            np.full((2, 2), 1000, np.uint16),
            np.full((1, 5), 1000, np.uint16),
        )
        for depth in cases:
            with self.subTest(shape=depth.shape):
                self.assert_depth_parity(depth, settings)
        result = self.native_depth(thin, settings)
        np.testing.assert_array_equal(result[1:-1, 3], np.full(5, 1000))
        self.assertEqual(result[0, 3], 0)
        self.assertEqual(result[-1, 3], 0)
        self.assertEqual(self.native_depth(hole, settings)[2, 2], 0)

    def test_filter_reads_all_clipped_neighbours_before_removing_any(self):
        depth = np.array([[1000, 1000, 1000]], dtype=np.uint16)
        actual = self.assert_depth_parity(depth, ScanSettings())
        np.testing.assert_array_equal(actual, [[0, 1000, 0]])
        cropped = ScanSettings(roi=(1, 0, 2, 1))
        np.testing.assert_array_equal(
            self.assert_depth_parity(depth, cropped), [[0, 0, 0]]
        )
        near = ScanSettings(near_m=1, far_m=2)
        clipped = np.array([[999, 1000, 1000]], dtype=np.uint16)
        np.testing.assert_array_equal(
            self.assert_depth_parity(clipped, near), [[0, 0, 0]]
        )

    def test_filter_tolerance_uses_center_depth_and_includes_exact_limit(self):
        for depth in (
            np.array([[985, 1000, 1015]], np.uint16),
            np.array([[984, 1000, 1015]], np.uint16),
            np.array([[1980, 2000, 2020]], np.uint16),
            np.array([[1979, 2000, 2021]], np.uint16),
        ):
            with self.subTest(depth=depth.tolist()):
                self.assert_depth_parity(depth, ScanSettings())
        self.assertEqual(
            self.native_depth(np.array([[985, 1000, 1015]], np.uint16), ScanSettings())[
                0, 1
            ],
            1000,
        )
        self.assertEqual(
            self.native_depth(np.array([[984, 1000, 1015]], np.uint16), ScanSettings())[
                0, 1
            ],
            0,
        )

    def test_depth_randomized_roi_and_range_parity(self):
        rng = np.random.default_rng(93)
        depth = rng.integers(0, 6000, (31, 37), dtype=np.uint16)
        depth[4:16, 5:19] = rng.integers(1980, 2021, (12, 14), dtype=np.uint16)
        depth[9:11, 12:14] = 0
        for filtered in (False, True):
            for roi in (None, (0, 0, 37, 31), (5, 4, 19, 16), (8, 8, 9, 9)):
                settings = ScanSettings(
                    near_m=0.5001, far_m=4.9999, filter_depth=filtered, roi=roi
                )
                with self.subTest(filtered=filtered, roi=roi):
                    self.assert_depth_parity(depth, settings)

    def test_empty_images_and_resolved_empty_rois_match_numpy(self):
        for shape in ((0, 0), (0, 5), (5, 0)):
            for filtered in (False, True):
                with self.subTest(shape=shape, filtered=filtered):
                    self.assert_depth_parity(
                        np.empty(shape, np.uint16), ScanSettings(filter_depth=filtered)
                    )
        depth = np.full((5, 7), 1000, np.uint16)
        for roi in ((4, 0, 2, 5), (0, 4, 7, 2), (-3, -4, -1, -1), (1, 1, 20, 20)):
            # The wrapper retains NumPy slice semantics even when passed a
            # settings-like object whose ROI exceeds this small test image.
            settings = SimpleNamespace(near_m=0.5, far_m=4, filter_depth=True, roi=roi)
            with (
                self.subTest(roi=roi),
                patch("shared.depth.kernels", return_value=self.native),
            ):
                np.testing.assert_array_equal(
                    prepare_depth(depth, settings),
                    _prepare_depth_numpy(depth, settings),
                )

    def test_python_depth_wrapper_keeps_view_and_dtype_compatibility(self):
        source = np.arange(64, dtype=np.float64).reshape(8, 8) * 30 + 500
        source = source[::-1, ::2]
        before = source.copy()
        settings = ScanSettings(filter_depth=False, roi=(1, 1, 4, 7))
        with patch("shared.depth.kernels", return_value=self.native):
            actual = prepare_depth(source, settings)
        np.testing.assert_array_equal(actual, _prepare_depth_numpy(source, settings))
        np.testing.assert_array_equal(source, before)
        unaligned = np.ndarray((3, 4), dtype=np.uint16, buffer=bytearray(25), offset=1)
        unaligned[:] = 1000
        with patch("shared.depth.kernels", return_value=self.native):
            np.testing.assert_array_equal(
                prepare_depth(unaligned, ScanSettings()),
                _prepare_depth_numpy(unaligned, ScanSettings()),
            )

    def test_calibrated_projection_matches_reference_in_both_rgb_modes(self):
        y, x = np.indices((480, 640))
        metric = np.ascontiguousarray(900 + 0.7 * x + 0.3 * y, dtype=np.float64)
        metric[170:200, 280:315] = 0
        for mode in ("rgb_high_res", "rgb_low_res"):
            settings = ScanSettings(
                sensor_calibration=self.calibration, rgb_mode=mode, filter_depth=False
            )
            depth = np.rint(metric).astype(np.uint16)
            shape = (settings.rgb_camera.height, settings.rgb_camera.width, 3)
            with self.subTest(mode=mode):
                expected = _color_maps_numpy(metric, depth, settings, shape)
                actual = self.projection(metric, depth, settings, shape)
                for computed, reference in zip(actual[:2], expected[:2]):
                    np.testing.assert_array_equal(computed, reference)
                    self.assertEqual(computed.dtype, np.float32)
                    self.assertTrue(computed.flags.c_contiguous)
                np.testing.assert_array_equal(actual[2], expected[2])
                self.assertEqual(actual[2].dtype, np.bool_)
                self.assertTrue(actual[2].flags.c_contiguous)

    def test_full_rgbd_native_and_reference_are_identical_without_input_mutation(self):
        y, x = np.indices((480, 640))
        raw = np.asarray(680 + ((x // 8 + y // 6) % 100), dtype=np.uint16)
        raw[140:170, 240:290] = 2047
        raw[220:230, 350:365] = 65535
        raw_before = raw.copy()
        for mode in ("rgb_high_res", "rgb_low_res"):
            settings = ScanSettings(
                sensor_calibration=self.calibration,
                rgb_mode=mode,
                roi=(30, 25, 600, 450),
            )
            ry, rx = np.indices((settings.rgb_camera.height, settings.rgb_camera.width))
            rgb = np.stack(
                (rx % 256, ry % 256, (rx // 3 + ry // 2) % 256), axis=-1
            ).astype(np.uint8)
            before = rgb.copy()
            with self.subTest(mode=mode):
                with (
                    patch("shared.depth.kernels", return_value=None),
                    patch("shared.calibration.kernels", return_value=None),
                ):
                    expected = prepare_native_rgbd(rgb, raw, settings)
                with (
                    patch("shared.depth.kernels", return_value=self.native),
                    patch("shared.calibration.kernels", return_value=self.native),
                ):
                    actual = prepare_native_rgbd(rgb, raw, settings)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                np.testing.assert_array_equal(raw, raw_before)
                np.testing.assert_array_equal(rgb, before)

    def test_projection_preserves_blas_rounding_at_color_interpolation_boundary(self):
        # Scalar evaluation of R @ point used to change this valid calibrated
        # sample by one float32 ULP and cross OpenCV's 1/32-pixel remap boundary.
        calibration = replace(
            self.calibration,
            rgb_low_res=replace(self.calibration.rgb_low_res, cx=306.32208255633054),
        )
        settings = ScanSettings(
            sensor_calibration=calibration,
            rgb_mode="rgb_low_res",
            filter_depth=False,
        )
        raw = np.random.default_rng(718).integers(650, 851, (480, 640), dtype=np.uint16)
        _, x = np.indices((480, 640))
        rgb = np.repeat(((x % 2) * 32)[..., None], 3, axis=2).astype(np.uint8)
        with (
            patch("shared.depth.kernels", return_value=None),
            patch("shared.calibration.kernels", return_value=None),
        ):
            expected = prepare_native_rgbd(rgb, raw, settings)
        with (
            patch("shared.depth.kernels", return_value=self.native),
            patch("shared.calibration.kernels", return_value=self.native),
        ):
            actual = prepare_native_rgbd(rgb, raw, settings)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(expected[0][176, 539], [14, 14, 14])
        np.testing.assert_array_equal(actual[0][176, 539], expected[0][176, 539])

    def handcrafted_projection(self, u, v, z, depth=None, rgb_shape=(5, 6)):
        metric = np.asarray(z, dtype=np.float64).reshape(1, -1)
        depth = (
            np.ones(metric.shape, dtype=np.uint16)
            if depth is None
            else np.asarray(depth, np.uint16).reshape(metric.shape)
        )
        rays = (
            np.stack((np.asarray(u), np.asarray(v), np.ones(metric.size)), axis=-1)
            .reshape(1, -1, 3)
            .astype(np.float64)
        )
        lens = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], np.float64)
        points = rays * metric[..., None]
        return self.native.project_native(points, depth, lens, *rgb_shape)

    def test_projection_occlusion_uses_complete_nearest_buffer_and_depth_tolerance(
        self,
    ):
        # The foreground occurs after the background in scan order. The last
        # three depths also distinguish the 15 mm floor and the 1% tolerance.
        u = [2.1, 2.2, 2.3, 2.4, 2.1, 3.1, 3.2, 3.3]
        v = [1.1] * len(u)
        z = [1040, 1015, 1000, 1016, 1040, 2000, 2020, 2021]
        _, _, actual = self.handcrafted_projection(u, v, z)
        nearest = {}
        for px, py, distance in zip(u, v, z):
            cell = (int(np.rint(px)), int(np.rint(py)))
            nearest[cell] = min(nearest.get(cell, float("inf")), distance)
        expected = [
            [
                distance
                <= nearest[(int(np.rint(px)), int(np.rint(py)))]
                + max(15, distance * 0.01)
                for px, py, distance in zip(u, v, z)
            ]
        ]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            actual, [[False, True, True, False, False, True, True, False]]
        )

    def test_projection_rounds_half_pixel_occlusion_bins_to_even(self):
        # 0.5 -> 0 and 2.5 -> 2, while 1.5 and 3.5 -> 2 and 4. An
        # implementation that always rounds halves up changes these masks.
        u = [0.5, 0.4, 1.5, 2.4, 2.5, 3.5, 4.4]
        v = [0.5, 0.4, 1.5, 2.4, 2.5, 1.5, 2.4]
        z = [1000, 1100, 1200, 1000, 1100, 1000, 1100]
        _, _, visible = self.handcrafted_projection(u, v, z, rgb_shape=(6, 6))
        np.testing.assert_array_equal(
            visible, [[True, False, False, True, False, True, False]]
        )

    def test_projection_visibility_excludes_invalid_depth_behind_camera_and_last_border(
        self,
    ):
        u = [0, -0.01, 4.999, 5, 1, 1, 1, 1]
        v = [0, 1, 3.999, 1, 4, 1, 1, 1]
        z = [1000, 1000, 1000, 1000, 1000, 0, -1000, 1000]
        depth = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 0]
        _, _, visible = self.handcrafted_projection(u, v, z, depth=depth)
        np.testing.assert_array_equal(
            visible, [[True, False, True, False, False, False, False, False]]
        )

    def test_projection_does_not_modify_any_input(self):
        settings = ScanSettings(sensor_calibration=self.calibration)
        metric = np.full((480, 640), 1200.0, np.float64)
        depth = np.full(metric.shape, 1200, np.uint16)
        points_ir = pinhole_rays(settings.camera) * metric[..., None]
        points_rgb = (
            points_ir.reshape(-1, 3) @ np.asarray(self.calibration.rotation).T
            + self.calibration.translation_mm
        ).reshape(*metric.shape, 3)
        inputs = (
            points_rgb,
            depth,
            np.array([600, 600, 320, 240, 0, 0, 0, 0, 0], np.float64),
        )
        snapshots = [value.copy() for value in inputs]
        for value in inputs:
            value.flags.writeable = False
        self.native.project_native(*inputs, 480, 640)
        for value, snapshot in zip(inputs, snapshots):
            np.testing.assert_array_equal(value, snapshot)

    def test_projection_empty_images_have_correct_output_shapes_and_dtypes(self):
        for shape in ((0, 0), (0, 5), (5, 0)):
            with self.subTest(shape=shape):
                result = self.native.project_native(
                    np.empty((*shape, 3), np.float64),
                    np.empty(shape, np.uint16),
                    np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], np.float64),
                    5,
                    6,
                )
                for output, dtype in zip(result, (np.float32, np.float32, np.bool_)):
                    self.assertEqual(output.shape, shape)
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(output.flags.c_contiguous)

    def test_direct_depth_kernel_rejects_wrong_dtype_rank_and_strides(self):
        for source in (
            np.ones((3, 4), np.float64),
            np.ones((3, 4), np.int16),
            np.ones(12, np.uint16),
            np.ones((3, 4, 1), np.uint16),
            np.ones((3, 8), np.uint16)[:, ::2],
            np.ones((3, 4), np.uint16).T,
            np.ones((3, 4), dtype=np.dtype(np.uint16).newbyteorder("S")),
            np.ndarray((3, 4), dtype=np.uint16, buffer=bytearray(25), offset=1),
        ):
            with (
                self.subTest(
                    dtype=source.dtype, shape=source.shape, strides=source.strides
                ),
                self.assertRaises((ValueError, TypeError)),
            ):
                self.native.prepare_depth(source, 500, 4000, True)
        with self.assertRaises(TypeError):
            self.native.prepare_depth([[1000]], 500, 4000, True)

    def test_direct_depth_kernel_rejects_invalid_bounds(self):
        depth = np.ones((3, 4), np.uint16)
        for near, far in ((float("nan"), 4000), (500, float("inf")), (4000, 500)):
            with self.subTest(near=near, far=far), self.assertRaises(ValueError):
                self.native.prepare_depth(depth, near, far, True)
        for roi in (
            (-1, 0, 4, 3),
            (0, -1, 4, 3),
            (0, 0, 5, 3),
            (0, 0, 4, 4),
            (5, 0, 2, 3),
        ):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                self.native.prepare_depth(depth, 500, 4000, True, *roi)

    def test_direct_projection_rejects_array_dtype_shape_and_strides(self):
        inputs = [
            np.ones((3, 4, 3), np.float64),
            np.full((3, 4), 1000, np.uint16),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], np.float64),
        ]
        wrong_shapes = [(3, 4, 2), (3, 3), (8,)]
        for index, source in enumerate(inputs):
            noncontiguous = np.empty(
                tuple(n * 2 for n in source.shape), dtype=source.dtype
            )
            noncontiguous = noncontiguous[
                tuple(slice(None, None, 2) for _ in source.shape)
            ]
            unaligned = np.ndarray(
                source.shape,
                dtype=source.dtype,
                buffer=bytearray(source.nbytes + 1),
                offset=1,
            )
            for replacement in (
                source.astype(np.float32),
                np.zeros(wrong_shapes[index], dtype=source.dtype),
                noncontiguous,
                unaligned,
                source.astype(source.dtype.newbyteorder("S")),
            ):
                arguments = inputs.copy()
                arguments[index] = replacement
                with (
                    self.subTest(
                        argument=index,
                        dtype=replacement.dtype,
                        shape=replacement.shape,
                        strides=replacement.strides,
                    ),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    self.native.project_native(*arguments, 5, 6)
        for height, width in ((0, 5), (5, 0), (-1, 5)):
            with (
                self.subTest(height=height, width=width),
                self.assertRaises(ValueError),
            ):
                self.native.project_native(*inputs, height, width)


if __name__ == "__main__":
    unittest.main()
