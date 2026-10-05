"""Complete Kinect v1 calibration, preserving the published measurement record."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .settings import CameraCalibration

DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] / "calibration/default.json"
)


def _matrix(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} requires finite values with shape {shape}")
    return result


def _profile(value, mode, size):
    if value["image_size"] != list(size):
        raise ValueError(f"{mode} calibration must describe {size}")
    if value["distortion_model"] != "OpenCV Brown-Conrady" or value[
        "distortion_coefficient_order"
    ] != ["k1", "k2", "p1", "p2", "k3"]:
        raise ValueError("Use the OpenCV Brown-Conrady five-coefficient model")
    k = _matrix(value["camera_matrix"], (3, 3), mode)
    if not np.array_equal(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError("Camera matrices require zero skew and a [0,0,1] last row")
    return CameraCalibration(
        name=mode,
        width=size[0],
        height=size[1],
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        distortion=value["distortion_coefficients"],
        image_space="native_depth"
        if mode == "depth"
        else "native_rgb"
        if mode.startswith("rgb")
        else "native_ir",
    )


@dataclass(frozen=True)
class SensorCalibration:
    # An immutable JSON snapshot keeps all accuracy/provenance data portable and
    # prevents caller mutation from changing an active reconstruction.
    document_json: str
    rgb_low_res: CameraCalibration
    rgb_high_res: CameraCalibration
    ir: CameraCalibration
    depth: CameraCalibration
    rotation: tuple
    translation_mm: tuple
    shift_px: tuple
    a_per_code: float
    b: float
    scale: float

    @classmethod
    def from_dict(cls, document):
        try:
            if document["schema_version"] != 1:
                raise ValueError("Unsupported complete calibration schema version")
            profiles = document["intrinsics"]
            low = _profile(profiles["rgb_low_res"], "rgb_low_res", (640, 480))
            high = _profile(profiles["rgb_high_res"], "rgb_high_res", (1280, 1024))
            ir = _profile(profiles["ir"], "ir", (640, 488))
            depth = _profile(profiles["depth"], "depth", (640, 480))
            grid = document["ir_to_depth"]
            if grid["selected_runtime_model"] != "translation":
                raise ValueError(
                    "Only the measured IR/depth translation model is supported"
                )
            if grid["ir_image_size"] != [640, 488] or grid["depth_image_size"] != [
                640,
                480,
            ]:
                raise ValueError(
                    "IR/depth mapping requires the native IR and depth grids"
                )
            shift = _matrix(grid["ir_to_depth_shift_px"], (2,), "IR/depth shift")
            expected = [ir.fx, ir.fy, ir.cx + shift[0], ir.cy + shift[1]]
            if (
                not np.allclose(
                    [depth.fx, depth.fy, depth.cx, depth.cy],
                    expected,
                    atol=1e-8,
                    rtol=0,
                )
                or depth.distortion != ir.distortion
            ):
                raise ValueError(
                    "Depth intrinsics must equal IR intrinsics with the grid shift applied once"
                )
            if (
                not np.allclose(
                    _matrix(
                        grid["depth_camera_matrix"], (3, 3), "Derived depth matrix"
                    ),
                    _matrix(profiles["depth"]["camera_matrix"], (3, 3), "Depth matrix"),
                    atol=1e-8,
                    rtol=0,
                )
                or tuple(grid["distortion_coefficients"]) != depth.distortion
            ):
                raise ValueError("IR/depth mapping and depth profile disagree")
            pose = document["ir_to_rgb"]
            rotation = _matrix(pose["R"], (3, 3), "IR-to-RGB rotation")
            translation = _matrix(pose["T_mm"], (3,), "IR-to-RGB translation in mm")
            if not np.allclose(
                rotation.T @ rotation, np.eye(3), atol=1e-6
            ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
                raise ValueError(
                    "IR-to-RGB rotation must be a proper orthonormal rotation"
                )
            conversion = document["raw_depth_to_mm"]
            if (
                conversion["selected_model"] != "factory_scale"
                or conversion["input"]["mode"] != "FREENECT_DEPTH_11BIT"
                or conversion["input"]["invalid_value"] != 2047
                or conversion["output"]["units"] != "millimetres"
                or conversion["output"]["invalid_value"] != 0
            ):
                raise ValueError(
                    "Require raw 11-bit input and factory-scale axial millimetre conversion"
                )
            if (
                conversion["input"]["storage"] != "uint16"
                or conversion["input"]["encoded_code_range"] != [0, 2047]
                or document["depth_encoding"]["invalid_value"] != 2047
                or document["depth_encoding"]["size"] != [640, 480]
            ):
                raise ValueError("Require native uint16 11-bit disparity encoding")
            a = float(conversion["factory_inverse_metres"]["a_per_code"])
            b = float(conversion["factory_inverse_metres"]["b"])
            scale = float(conversion["scale"])
            if not all(math.isfinite(v) for v in (a, b, scale)) or not (
                a < 0 < b and 0.8 <= scale <= 1.2
            ):
                raise ValueError(
                    "Invalid inverse-disparity conversion or metric correction"
                )
            snapshot = json.dumps(document, allow_nan=False)
            return cls(
                snapshot,
                low,
                high,
                ir,
                depth,
                tuple(tuple(float(v) for v in row) for row in rotation),
                tuple(map(float, translation)),
                tuple(map(float, shift)),
                a,
                b,
                scale,
            )
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"Incomplete sensor calibration: {exc}") from exc

    def to_dict(self):
        return json.loads(self.document_json)

    @property
    def name(self):
        return (
            f"Kinect {self.to_dict().get('camera_serial', '')} · measured RGB/IR/depth"
        )


def load_calibration(path=DEFAULT_CALIBRATION_PATH):
    """Load and validate the complete Kinect calibration publication."""
    return SensorCalibration.from_dict(json.loads(Path(path).read_text()))
