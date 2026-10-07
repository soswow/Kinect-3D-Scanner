"""Validated settings for metric datasets and calibrated native Kinect streams."""

import math
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING

from .capture import validate_rgb_exposure

if TYPE_CHECKING:
    from .sensor_calibration import SensorCalibration


@dataclass(frozen=True)
class CameraCalibration:
    name: str = "Registered RGB (approximate)"
    width: int = 640
    height: int = 480
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5
    image_space: str = "registered_rgb"
    distortion: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    depth_scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "distortion", tuple(self.distortion))
        if len(self.distortion) != 5 or not all(
            math.isfinite(v) and abs(v) <= 2 for v in self.distortion
        ):
            raise ValueError("Use five finite OpenCV distortion coefficients")
        if not math.isfinite(self.depth_scale) or not 0.8 <= self.depth_scale <= 1.2:
            raise ValueError("Measured depth-scale correction must be 0.8–1.2")
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("Image dimensions must be integers")
        sizes = {
            "registered_rgb": ((640, 480),),
            "native_depth": ((640, 480),),
            "native_rgb": ((640, 480), (1280, 1024)),
            "native_ir": ((640, 488),),
        }
        if (
            self.image_space not in sizes
            or (self.width, self.height) not in sizes[self.image_space]
        ):
            raise ValueError("Unsupported camera image space or dimensions")
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("Camera parameters must be finite")
        if not (100 < self.fx < 2000 and 100 < self.fy < 2000):
            raise ValueError("Invalid focal length")
        if not (0 <= self.cx < self.width and 0 <= self.cy < self.height):
            raise ValueError("Principal point must lie in the image")
        radius = math.hypot(
            max(self.cx, self.width - self.cx) / self.fx,
            max(self.cy, self.height - self.cy) / self.fy,
        )
        k1, k2, _, _, k3 = self.distortion
        if any(
            1 + 3 * k1 * (r := radius * i / 64) ** 2 + 5 * k2 * r**4 + 7 * k3 * r**6
            <= 0
            for i in range(65)
        ):
            raise ValueError("Lens calibration folds within the image; recalibrate")


@dataclass(frozen=True)
class ScanSettings:
    camera: CameraCalibration = field(default_factory=CameraCalibration)
    near_m: float = 0.5
    far_m: float = 4.0
    voxel_m: float = 0.005
    truncation_m: float = 0.04
    min_fitness: float = 0.35
    max_rmse_m: float = 0.02
    max_translation_m: float = 0.3
    max_rotation_deg: float = 30.0
    final_weight: float = 2.0
    min_component_triangles: int = 30
    # Pixel rectangle [left, top, right, bottom], exclusive right/bottom.
    roi: tuple[int, int, int, int] | None = None
    filter_depth: bool = True
    color_recovery: bool = False
    live_reconstruction: bool = False
    refine_poses: bool = False
    bundle_adjustment: bool = False
    reconnect_fragments: bool = False
    relocalize: bool = False
    confidence_fusion: bool = False
    final_voxel_m: float | None = None
    final_block_count: int = 5000

    sensor_calibration: "SensorCalibration | None" = None
    rgb_mode: str = "rgb_high_res"
    rgb_exposure_mode: str = "auto"
    # Reciprocal seconds: 125 means a fixed 1/125 s exposure in manual mode.
    rgb_shutter_speed: int = 125
    rgb_gain: int = 1

    def __post_init__(self):
        if not isinstance(self.camera, CameraCalibration):
            raise TypeError("Invalid camera calibration")
        if self.rgb_mode not in ("rgb_high_res", "rgb_low_res"):
            raise ValueError("RGB mode must be rgb_high_res or rgb_low_res")
        if self.sensor_calibration is not None:
            from .sensor_calibration import SensorCalibration

            if not isinstance(self.sensor_calibration, SensorCalibration):
                raise ValueError("Invalid complete sensor calibration")
            object.__setattr__(self, "camera", self.sensor_calibration.depth)
        elif self.camera.image_space != "registered_rgb":
            raise ValueError(
                "Native camera profiles require a complete sensor calibration"
            )
        else:
            object.__setattr__(self, "rgb_mode", "rgb_low_res")
        validate_rgb_exposure(self.rgb_exposure_mode, self.rgb_shutter_speed, self.rgb_mode, self.rgb_gain)
        if (
            type(self.filter_depth) is not bool
            or type(self.color_recovery) is not bool
            or type(self.live_reconstruction) is not bool
            or type(self.refine_poses) is not bool
            or type(self.bundle_adjustment) is not bool
            or type(self.reconnect_fragments) is not bool
            or type(self.relocalize) is not bool
            or type(self.confidence_fusion) is not bool
            or type(self.min_component_triangles) is not int
        ):
            raise ValueError("Invalid filter or component setting type")
        floats = (
            self.near_m,
            self.far_m,
            self.voxel_m,
            self.truncation_m,
            self.min_fitness,
            self.max_rmse_m,
            self.max_translation_m,
            self.max_rotation_deg,
            self.final_weight,
        )
        if not all(math.isfinite(v) for v in floats):
            raise ValueError("Scan settings must be finite")
        if not (0 <= self.near_m < self.far_m <= 8):
            raise ValueError("Require 0 <= near < far <= 8 metres")
        if not (0.002 <= self.voxel_m <= 0.03):
            raise ValueError("Voxel size must be 2–30 mm")
        if not (2 * self.voxel_m <= self.truncation_m <= 0.2):
            raise ValueError("Truncation must be at least two voxels and <= 0.2 m")
        if not (0.1 <= self.min_fitness <= 1 and 0.001 <= self.max_rmse_m <= 0.1):
            raise ValueError("Invalid tracking thresholds")
        if not (0 < self.max_translation_m <= 2 and 0 < self.max_rotation_deg <= 90):
            raise ValueError("Invalid motion limits")
        if not (
            0.5 <= self.final_weight <= 20
            and 0 <= self.min_component_triangles <= 10000
        ):
            raise ValueError("Invalid mesh confidence settings")
        if self.final_voxel_m is not None and not (
            math.isfinite(self.final_voxel_m)
            and 0.002 <= self.final_voxel_m <= self.voxel_m
        ):
            raise ValueError("Final voxel must be 2 mm up to the live voxel size")
        if (
            type(self.final_block_count) is not int
            or not 1 <= self.final_block_count <= 50000
        ):
            raise ValueError("Final volume budget must be 1–50000 blocks")
        if self.roi is not None:
            if len(self.roi) != 4 or any(type(v) is not int for v in self.roi):
                raise ValueError("ROI requires four integer pixel coordinates")
            x0, y0, x1, y1 = self.roi
            if not (
                0 <= x0 < x1 <= self.camera.width and 0 <= y0 < y1 <= self.camera.height
            ):
                raise ValueError("ROI must be inside the image")

    def to_dict(self):
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        result["camera"] = asdict(self.camera)
        result["sensor_calibration"] = (
            self.sensor_calibration.to_dict() if self.sensor_calibration else None
        )
        return result

    @property
    def rgb_camera(self):
        return (
            getattr(self.sensor_calibration, self.rgb_mode)
            if self.sensor_calibration
            else self.camera
        )

    @property
    def depth_encoding(self):
        return "raw_11bit" if self.sensor_calibration else "registered_mm"

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.get("sensor_calibration") is not None:
            from .sensor_calibration import SensorCalibration

            value["sensor_calibration"] = SensorCalibration.from_dict(
                value["sensor_calibration"]
            )
        if "camera" in value:
            value["camera"] = CameraCalibration(**value["camera"])
        if value.get("roi") is not None:
            value["roi"] = tuple(value["roi"])
        return cls(**value)
