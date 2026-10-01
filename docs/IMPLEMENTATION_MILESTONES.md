# Ordered implementation milestones

Continue from `50d448c`, following RESEARCH_ROADMAP.md. Each milestone is a
separate tested commit; experimental options stay opt-in. Existing uncommitted
UI dock sizing edits are preserved outside these commits.

1. **Device evidence and calibration.** Checkerboard fitting with held-out
   reprojection, coverage and tilt checks; common RGB/depth lens rectification;
   measured depth-scale correction; plane precision and pairing reports.
   Implemented and verified with known-camera fixtures and the regression suite.
   Physical accuracy acceptance needs a measured
   checkerboard/plane and repeated recordings; no fabricated calibration.
2. **Compute profiling.** Matched replay reports and backend/stage attribution.
   Actual NVIDIA throughput remains unverified until run on an NVIDIA server.
3. **Global tracking.** Appearance retrieval outside estimated pose radius,
   reciprocal geometric verification, held-out checks and fresh fusion.
4. **Sensor-aware fusion.** Explicit range/angle/edge confidence, controlled
   surface tests, observed-space preservation; compare error and completeness.
5. **Final appearance.** Measured-view exposure consistency and sharp view
   selection, with coverage reports and an observed-color fallback.
6. **Live/final budgets.** Separate final reintegration and memory limits;
   show actionable tracking/backlog feedback.

The older SCAN_QUALITY.md roadmap also requests turntable capture and adaptive
capture. Turntable support remains a separate subsequent milestone: foreground
masking alone cannot give the object's rotation. It needs measured angles and
an axis, or a validated object-pose tracker. Surface evaluation and capture
feedback belong alongside milestones 4 and 6.

## Calibration workflow

Save a full RGB-D session and unzip it into ignored `recordings/`. For intrinsics,
collect at least 16 sharp views of a flat checkerboard spanning the image and
several tilts. Columns/rows count **inner corners**; measure printed square size.

```bash
python scripts/calibrate_camera.py recordings/board \
  --output benchmark-output/measured-camera.json intrinsics \
  --columns 9 --rows 6 --square-mm 25
python scripts/calibrate_camera.py recordings/plane \
  --output benchmark-output/plane.json evidence \
  --plane-roi 100 100 540 380 --expected-z-m 1.0
```

Only accepted fits write a loadable camera JSON; a separate report records
held-out errors and rejection reasons. Load that camera through the GUI's
calibration control. Depth scale is **not** estimated from RGB checkerboards:
use several known front-facing plane depths to validate any scale correction.
Plane residuals describe precision, not absolute accuracy. Recordings remain
raw; correction is applied consistently before tracking, fusion and texturing.
Lens rectification keeps K unchanged and uses nearest-neighbour depth sampling
so missing depth stays missing. Recordings replay their pairing metadata.

Calibration follows [OpenCV's calibrated camera model](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

## Compute profiling milestone

Implemented isolated, sequential replays with per-stage total/p50/p95, stage
placement, acceptance/error, source hash and peak process RSS. It detects source
changes during a run. CUDA requests fail explicitly rather than falling back.

```bash
OMP_NUM_THREADS=4 python scripts/profile_backends.py --dataset redwood --repeats 3
# On an NVIDIA server, use the same recorded input and settings:
python scripts/profile_backends.py --dataset recording --path recordings/session \
  --repeats 3 --runs cpu-legacy cuda-tensor
```

Two sequential five-frame Redwood repeats on this Mac accepted 5/5 in both CPU
modes: median processing/final-extraction 3.37 s legacy, 4.06 s tensor. These short
runs are a profiling smoke check, not a general throughput result. CUDA was
unavailable. Peak RSS includes CPU allocations and imports, not GPU VRAM.
[Aggregate evidence](benchmarks/backend-profile.json) contains no captured images.
NVIDIA throughput validation remains pending target hardware. The current hybrid
stage placement is explicit; no new GPU optimization is justified by a CPU-only
profile.
