# Measured Kinect default

`default.json` is an exact copy of the complete final calibration publication
from `../../calibration/final/kinect-calibration.json`, accepted 5 October 2026
for camera **A00363W00948202A**. It retains numerical precision, validation,
physical board measurements, nuisance terms, and provenance. `raw-to-mm.bin`
is the matching 2,048-entry little-endian uint16 lookup table. Runtime computes
conversion directly from the JSON formula; it needs no external files. Source
paths and report/LUT references inside the publication are archival metadata.

The GUI loads this file on startup and selects **1280 × 1024 RGB at 10 fps**.
The RGB selector also offers **640 × 480 at 30 fps**, using its independently
measured profile. **Load calibration JSON** requires the complete publication;
loading it restarts capture. Settings freeze when scanning starts. Single-camera
calibration JSON files are rejected. Capture always uses native raw depth.

**Auto every** selects every N fresh RGB/depth pairs. The seconds field rounds to
whole RGB-frame periods: **0.1, 0.2, 0.3… seconds** at 10 fps, and **1/30,
2/30, 3/30… seconds** at 30 fps. The minimum is one frame, so the automatic
capture rate cannot exceed the selected camera mode. For example, 0.5 seconds
means every five pairs in high-resolution mode or every fifteen in VGA mode.
Changing RGB mode updates the interval while preserving its nominal duration
as closely as whole frames allow.

Capture runs when a new pair arrives. Changing the interval takes effect
immediately, and manual capture restarts the automatic cadence. Preview and
reconstruction backpressure pause capture; resuming selects one fresh pair and
starts a new cadence, without a catch-up burst. USB delays or dropped pairs can
make the actual interval longer than its nominal value. Cached frames are never
captured repeatedly.

## Complete JSON structure

The accepted format is the published `schema_version: 1` document. Required
runtime fields are:

- `intrinsics.rgb_low_res`, `intrinsics.rgb_high_res`, `intrinsics.ir`, and
  `intrinsics.depth`: native `image_size`, `camera_matrix`,
  `distortion_model: "OpenCV Brown-Conrady"`,
  `distortion_coefficient_order: ["k1", "k2", "p1", "p2", "k3"]`, and
  `distortion_coefficients`.
- `ir_to_depth`: `selected_runtime_model: "translation"`, native IR/depth sizes,
  `ir_to_depth_shift_px`, `depth_camera_matrix`, and `distortion_coefficients`.
  The depth matrix must retain IR focal lengths and translate its principal
  point by this shift. Redundant values are checked for agreement.
- `ir_to_rgb`: proper orthonormal `R` and `T_mm`, defining
  `X_rgb_mm = R @ X_ir_mm + T_mm` in x-right, y-down, z-forward coordinates.
- `raw_depth_to_mm`: `selected_model: "factory_scale"`, `input` specifying
  `FREENECT_DEPTH_11BIT`, uint16 storage, range `[0, 2047]`, invalid `2047`;
  `output` in millimetres with invalid `0`; `factory_inverse_metres` containing
  `a_per_code` and `b`; and the multiplicative `scale`.
- `depth_encoding`: native `size: [640, 480]` and `invalid_value: 2047`.

All remaining publication fields are retained in session settings and exports.
The loader validates dimensions, distortion model/order, camera matrices,
IR/depth consistency, rotation, and conversion parameters before use. It never
uses boundary expansion or board thickness as optical corrections.

## Runtime geometry and colour

Capture preserves unmirrored native RGB and **640 × 480 raw disparity**. Raw
zero is a valid code; 2047 is invalid. The guarded conversion is
`Z_mm = scale * 1000 / (a_per_code * raw + b)`, requiring a positive denominator
and `0 < Z_mm < 10000`. The published scale is applied once. This is IR-camera
axial Z; measured accuracy covers approximately **0.8–1.6 m**.

Reconstruction rectifies depth with nearest-neighbour sampling and reconstructs
rays using the derived depth matrix. That matrix already includes the grid
translation, which is not applied a second time. Geometry and tracking stay
on the 640 × 480 depth grid. Colour is sampled from the native RGB image through
measured R/T and the selected RGB K/D, with a nearest-Z collision check. Texture
exports project mesh points directly into the original full-resolution RGB,
using rectified depth for visibility. They retain RGB detail beyond the depth
sampling resolution.

Local recordings and server session ZIP exports retain original RGB/raw-depth
PNGs plus the complete calibration in `settings.sensor_calibration`.
`depth_unit` is `raw_11bit_disparity` for this path. Replay applies the same
conversion and registration. Public dataset replay supplies explicit camera
intrinsics and metric depth for benchmark evaluation.

API users can supply the same complete JSON object in `sensor_calibration`
and choose `rgb_mode: "rgb_high_res"` or `"rgb_low_res"` when resetting a session.
The derived `camera` is set automatically. An empty API reset uses the complete
measured Kinect default and high-resolution RGB. High-resolution uploads use bounded
`RGB3` packets; original VGA and `RGB2` packets remain supported. Frame metadata
includes `depth_encoding`; mismatches with session settings are rejected.

```python
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings

settings = ScanSettings(sensor_calibration=load_calibration())
# POST settings.to_dict() to /api/scan/reset before uploading native frames.
```
