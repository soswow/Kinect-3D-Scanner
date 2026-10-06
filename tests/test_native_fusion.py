"""Shared-buffer and float32 parity checks for optional native TSDF fusion."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import _kinect_native as native
except ImportError:
    native = None


def reference_integrate(
    xyz,
    indices,
    depth,
    confidence,
    rgb,
    tsdf,
    weight,
    color,
    fx,
    fy,
    cx,
    cy,
    max_depth,
    truncation,
):
    z = xyz[:, 2]
    safe = np.maximum(z, np.float32(1e-6))
    projected = xyz[:, :2] * [np.float32(fx), np.float32(fy)]
    projected /= safe[:, None]
    projected += [np.float32(cx), np.float32(cy)]
    with np.errstate(invalid="ignore"):
        pixels = np.copysign(np.floor(np.abs(projected) + 0.5), projected).astype(
            np.int64
        )
    u, v = pixels.T
    inside = (z > 0) & (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
    u, v, z = u[inside], v[inside], z[inside]
    ids = indices[inside]
    observed, incoming = depth[v, u], confidence[v, u]
    sdf = observed - z
    valid = (
        (observed > 0) & (observed <= max_depth) & (incoming > 0) & (sdf >= -truncation)
    )
    ids = ids[valid]
    distances = np.minimum(sdf[valid, None] / truncation, 1)
    contribution = incoming[valid, None]
    old = weight[ids]
    total = old + contribution
    tsdf[ids] = (tsdf[ids] * old + distances * contribution) / total
    color[ids] = (color[ids] * old + rgb[v[valid], u[valid]] * contribution) / total
    weight[ids] = total
    return len(ids)


@unittest.skipUnless(
    native is not None and hasattr(native, "integrate_weighted_cpu"),
    "Install optional extension with python -m pip install ./native",
)
class NativeFusionTests(unittest.TestCase):
    def make_inputs(self, count=1000):
        rng = np.random.default_rng(51)
        return {
            "xyz": rng.uniform([-1, -1, -0.2], [1, 1, 2], (count, 3)).astype(
                np.float32
            ),
            "indices": rng.permutation(count).astype(np.int64),
            "depth": rng.uniform(0.5, 2, (12, 16)).astype(np.float32),
            "confidence": rng.uniform(0.05, 1, (12, 16)).astype(np.float32),
            "rgb": rng.integers(0, 256, (12, 16, 3), np.uint8),
            "tsdf": rng.uniform(-1, 1, (count, 1)).astype(np.float32),
            "weight": rng.uniform(0, 5, (count, 1)).astype(np.float32),
            "color": rng.uniform(0, 255, (count, 3)).astype(np.float32),
            "fx": 8.3,
            "fy": 7.6,
            "cx": 7.5,
            "cy": 5.5,
            "max_depth": 1.7,
            "truncation": 0.08,
        }

    def assert_parity(self, inputs):
        expected = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in inputs.items()
        }
        count = reference_integrate(**expected)
        self.assertEqual(count, native.integrate_weighted_cpu(**inputs))
        for key in ("tsdf", "weight", "color"):
            np.testing.assert_array_equal(expected[key], inputs[key], err_msg=key)
        return count

    def test_parity_after_numpy_camera_transforms(self):
        for angle in (0, 0.013, -0.273):
            with self.subTest(angle=angle):
                inputs = self.make_inputs(10000)
                c, s = np.cos(angle), np.sin(angle)
                transform = np.array(
                    [
                        [c, 0, s, 0.043],
                        [0, 1, 0, -0.012],
                        [-s, 0, c, 0.21],
                        [0, 0, 0, 1],
                    ],
                    np.float32,
                )
                inputs["xyz"] = inputs["xyz"] @ transform[:3, :3].T + transform[:3, 3]
                inputs["depth"][0] = 0
                inputs["confidence"][1] = 0
                inputs["xyz"][0] = np.nan
                inputs["xyz"][1] = [np.inf, 0, 1]
                self.assertGreater(self.assert_parity(inputs), 0)

    def test_shared_storage_and_readonly_observations(self):
        inputs = self.make_inputs(1000)
        backing = {}
        for key in ("tsdf", "weight", "color"):
            backing[key] = inputs[key].reshape(10, 10, 10, -1).copy()
            inputs[key] = backing[key].reshape(inputs[key].shape)
        for key in ("xyz", "indices", "depth", "confidence", "rgb"):
            inputs[key].flags.writeable = False
        before = {key: value.copy() for key, value in backing.items()}
        self.assertGreater(self.assert_parity(inputs), 0)
        for key, value in backing.items():
            self.assertFalse(np.array_equal(before[key], value), key)

    def test_half_away_rounding_float32_boundaries_and_duplicate_indices(self):
        inputs = self.make_inputs(10)
        x = np.array(
            [
                0.5,
                1.5,
                2.5,
                -0.5,
                -0.49,
                np.nextafter(np.float32(1.5), np.float32(0)),
                0,
                0,
                0,
                0,
            ],
            np.float32,
        )
        inputs.update(
            xyz=np.column_stack((x, np.zeros(10, np.float32), np.ones(10, np.float32))),
            indices=np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8], np.int64),
            fx=1,
            fy=1,
            cx=0,
            cy=0,
        )
        inputs["depth"][:] = 1
        inputs["confidence"][:] = 0.5
        inputs["depth"][0, 0] = 0
        self.assertEqual(4, self.assert_parity(inputs))

    def test_observation_depth_confidence_and_truncation_gates(self):
        inputs = self.make_inputs(9)
        z = np.array([1, 0, -1, 1, 1, 1.09, 0.91, 1, 1], np.float32)
        inputs.update(
            xyz=np.column_stack(
                (np.arange(9, dtype=np.float32) * z, np.zeros(9, np.float32), z)
            ),
            indices=np.arange(9, dtype=np.int64),
            fx=1,
            fy=1,
            cx=0,
            cy=0,
        )
        inputs["depth"][:] = 1
        inputs["depth"][0, 3] = 0
        inputs["depth"][0, 4] = 2
        inputs["confidence"][:] = 0.5
        inputs["confidence"][0, 7] = 0
        inputs["confidence"][0, 8] = np.nan
        for key in ("tsdf", "weight", "color"):
            inputs[key][:] = 0
        self.assertEqual(2, self.assert_parity(inputs))
        np.testing.assert_array_equal(
            inputs["weight"].reshape(-1), [0.5, 0, 0, 0, 0, 0, 0.5, 0, 0]
        )
        self.assertEqual(0, inputs["tsdf"][0, 0])
        self.assertEqual(1, inputs["tsdf"][6, 0])

    def test_invalid_inputs_do_not_mutate_shared_attributes(self):
        variants = []
        inputs = self.make_inputs(100)
        for key, value in (
            ("xyz", inputs["xyz"].astype(np.float64)),
            ("confidence", inputs["confidence"].astype(np.float64)),
            ("indices", inputs["indices"].astype(np.int32)),
            ("depth", inputs["depth"].astype(">f4")),
            ("rgb", inputs["rgb"][:, :, :2].copy()),
            ("tsdf", inputs["tsdf"].reshape(-1)),
            ("color", np.zeros((100, 6), np.float32)[:, ::2]),
            ("truncation", 0),
            ("max_depth", np.nan),
            ("fx", np.inf),
        ):
            variants.append((key, value))
        readonly = inputs["weight"].copy()
        readonly.flags.writeable = False
        variants.append(("weight", readonly))
        unaligned = np.ndarray(
            (100, 1), dtype=np.float32, buffer=bytearray(401), offset=1
        )
        variants.append(("tsdf", unaligned))
        for key, value in variants:
            with self.subTest(key=key, dtype=getattr(value, "dtype", None)):
                trial = self.make_inputs(100)
                trial[key] = value
                before = {
                    name: trial[name].copy() for name in ("tsdf", "weight", "color")
                }
                with self.assertRaises((TypeError, ValueError)):
                    native.integrate_weighted_cpu(**trial)
                for name, value in before.items():
                    np.testing.assert_array_equal(value, trial[name])
        for invalid in (-1, 100):
            trial = self.make_inputs(100)
            trial["indices"][-1] = invalid
            before = {name: trial[name].copy() for name in ("tsdf", "weight", "color")}
            with self.assertRaises(IndexError):
                native.integrate_weighted_cpu(**trial)
            for name, value in before.items():
                np.testing.assert_array_equal(value, trial[name])

    def test_aliases_are_rejected_before_mutation(self):
        for alias_input in (True, False):
            inputs = self.make_inputs(100)
            if alias_input:
                inputs["xyz"] = inputs["color"]
            else:
                inputs["weight"] = inputs["tsdf"]
            before = {key: inputs[key].copy() for key in ("tsdf", "weight", "color")}
            with self.assertRaises(ValueError):
                native.integrate_weighted_cpu(**inputs)
            for key, value in before.items():
                np.testing.assert_array_equal(value, inputs[key])

    def test_empty_batch_leaves_attributes_unchanged(self):
        inputs = self.make_inputs(100)
        inputs["xyz"] = np.empty((0, 3), np.float32)
        inputs["indices"] = np.empty(0, np.int64)
        self.assertEqual(0, self.assert_parity(inputs))

    def test_numpy_and_native_paths_match_shared_open3d_volume(self):
        import open3d as o3d

        from scanner_server.weighted_fusion import integrate_weighted

        core = o3d.core
        camera = SimpleNamespace(width=16, height=12, fx=8.3, fy=7.6, cx=7.5, cy=5.5)
        engine = SimpleNamespace(
            settings=SimpleNamespace(camera=camera),
            device=core.Device("CPU:0"),
            max_depth_m=1.5,
            sdf_trunc=0.08,
        )
        # More than 128 blocks also exercise the chunk boundary. Small blocks
        # keep this real Open3D shared-buffer regression lightweight.
        coordinates = np.stack(
            np.meshgrid(
                np.arange(-2, 2), np.arange(-3, 3), np.arange(3, 9), indexing="ij"
            ),
            axis=-1,
        )
        blocks = core.Tensor(coordinates.reshape(-1, 3).astype(np.int32))
        y, x = np.indices((12, 16))
        rgb = np.stack((x * 13, y * 17, (x + y) * 7), axis=-1).astype(np.uint8)
        first = (1000 + x * 3 + y * 5).astype(np.uint16)
        first[4:6, 7:9] = 0
        first[2, 3] = 2000
        second = (first + 12).astype(np.uint16)
        second[first == 0] = 0
        transform = np.eye(4)
        angle = 0.023
        transform[:3, :3] = [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
        transform[:3, 3] = [0.017, -0.012, 0.004]
        results = []
        statistics = []
        for mode in ("off", "on"):
            volume = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"),
                attr_dtypes=(core.float32,) * 3,
                attr_channels=((1,), (1,), (3,)),
                voxel_size=0.05,
                block_resolution=4,
                block_count=256,
                device=engine.device,
            )
            with patch.dict(os.environ, {"KINECT_NATIVE": mode}):
                for depth, pose in ((first, np.eye(4)), (second, transform)):
                    statistics.append(
                        integrate_weighted(engine, volume, blocks, rgb, depth, pose)
                    )
            hashmap = volume.hashmap()
            ids = hashmap.active_buf_indices().numpy().astype(np.int64)
            keys = hashmap.key_tensor().numpy()[ids]
            order = np.lexsort(keys.T)
            attributes = {
                name: volume.attribute(name)
                .numpy()
                .reshape(-1, 64, 3 if name == "color" else 1)[ids][order]
                .copy()
                for name in ("tsdf", "weight", "color")
            }
            results.append((keys[order].copy(), attributes))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        for name in results[0][1]:
            np.testing.assert_array_equal(
                results[0][1][name], results[1][1][name], err_msg=name
            )
        self.assertGreater(np.count_nonzero(results[1][1]["weight"]), 0)
        self.assertEqual(statistics[:2], statistics[2:])


if __name__ == "__main__":
    unittest.main()
