# Repeated depth measurements and surface confidence

Research and controlled evaluation: 7 October 2026. This increment improves the
existing optional confidence-weighted TSDF. Ordinary Open3D fusion remains the
default. Sensor coefficients are still engineering assumptions, not a calibrated
probability model for this particular Kinect.

## Candidates

| Candidate | Why it fits this scanner | Status / prerequisite |
| --- | --- | --- |
| Stable surface orientation plus range-aware observation weights | Noisy one-pixel normals discarded useful depth measurements before averaging. More reliable normals help the existing fusion retain independent observations. [Nguyen et al., 2012](https://users.cecs.anu.edu.au/~nguyen/papers/conferences/Nguyen2012-ModelingKinectSensorNoise.pdf) measures Kinect noise versus range and angle. | Implemented local inverse-depth plane fits for confidence only. |
| Calibrated precision / static Kalman fusion | The existing incremental weighted mean already has the same mean update as a static scalar Kalman filter. Physically meaningful precision would permit confidence-based extraction. [Curless and Levoy, 1996](https://lightfield.stanford.edu/papers/volrange/) establishes weighted distance fusion and its least-squares interpretation under stated assumptions. | Existing averaging retained. Measure this device's axial noise, systematic bias, and correlation before reporting millimetre uncertainty. |
| Probabilistic TSDF with depth and pose uncertainty | Confidence should reflect uncertain correspondence and camera motion as well as sensor noise. [Rosinol et al., WACV 2023](https://arxiv.org/abs/2210.01276) derives depth uncertainty from bundle adjustment for volumetric fusion. Its depth source is monocular SLAM; its uncertainty-aware integration principle is relevant here. | Next research candidate after calibrated depth noise and dependable pose uncertainty. Requires extra voxel attributes and extraction semantics. |
| Confidence-maintaining surfel fusion | A surfel stores a small surface patch rather than a full volumetric neighbourhood. [Keller et al., 2013](https://reality.tf.fau.de/projects/kinect/keller13realtime.html) accumulates confidence and removes unstable/outlier points over time. | Larger alternative architecture; benchmark memory, fine detail, and final meshing before replacing TSDF. |
| Joint appearance, geometry, pose and lighting refinement | Repeated depth reduces random noise but cannot recover details absent from depth alone. [Intrinsic3D, ICCV 2017](https://research.nvidia.com/publication/2017-10_intrinsic3d-high-quality-3d-reconstruction-joint-appearance-and-geometry) uses image shading to refine geometry along with pose and appearance. | Later Finish candidate; needs reliable poses, calibrated RGB, illumination/material assumptions, and independent geometry validation. |

## What accumulating measurements already does

For a voxel's distance estimate `m`, accumulated weight `W`, new distance `d`,
and observation weight `w`, our update is:

```
gain = w / (W + w)
m_new = m + gain * (d - m)
W_new = W + w
```

This is also the mean update of a scalar Kalman filter for a stationary surface
with zero process noise. If `W` and `w` were actual inverse variances of
independent Gaussian measurements, the posterior variance would be
`1 / (W + w)`. Equal-quality independent samples then reduce standard deviation
approximately as `1 / sqrt(number of samples)`.

Our weights include clipping, a heuristic range law, viewing angle, and
observation rejection. They are **relative evidence scores**, not calibrated
inverse variances. The accumulated weight must not be displayed as independent
sample count or converted into a physical error bar. Repeating a biased or
correlated measurement increases this score without establishing increased
accuracy. Calibration errors, pose drift and repeated speckle patterns need
separate treatment; Kalman filtering does not remove them by itself.

## Implemented change

`shared/confidence.py` estimates orientation from a seven-pixel square patch.
Inverse depth on a perspective-viewed plane is affine in image coordinates,
so its least-squares slopes give an accurate plane normal without constructing
a noisy three-dimensional cross product from neighbouring pixels. A smaller
three-pixel patch handles narrow surfaces where the larger patch cannot fit.

A fit requires complete valid measurements and cannot cross a detected depth
discontinuity. Pixels without a supported fit retain the existing conservative
angle contribution. Missing depths stay missing; depth steps and isolated large
outliers retain their rejection gates. Only the confidence image changes:
measured depths, holes, fine surface details, TSDF equations and final extraction
thresholds are untouched. The native kernel interface and volume attributes are
unchanged.

## Controlled results

The reproducible benchmark compares the frozen preceding estimator with the
local plane estimator using identical observations and exact fixed poses.
Synthetic axial noise is independent Gaussian with standard deviation
`0.001 + 0.002 * depth_m²`, followed by integer-millimetre quantization. This
tests the algorithm under a known model; it does not validate that model on
the physical camera.

Actual CPU TSDF runs use 5 mm voxels, 40 mm truncation, and unchanged final
weight threshold **2.0**. Completeness is the fraction of known surface samples
within 10 mm of extracted geometry; precision uses the same measured interior
domain. RMSE below measures reconstructed axial depth against known geometry,
not registration residual against data used to estimate the camera poses.

| Known geometry / views | Axial RMSE, old → new | Completeness, old → new |
| --- | --- | --- |
| Front plane / 4 | 1.366 → 0.985 mm | 90.7% → 100% |
| Front plane / 12 | 0.662 → 0.585 mm | 100% → 100% |
| 55° plane / 12 | 1.167 → 1.057 mm | 92.6% → 83.7% |
| 55° plane / 32 | 1.093 → 1.029 mm | 100% → 100% |
| 60 mm step and missing patch / 12 | 0.702 → 0.616 mm | 100% → 100% |
| Sinusoidal 10 mm detail / 16 | 0.892 → 0.754 mm | 100% → 100% |

All compared cases have 100% precision within 10 mm in the measured interior.
Neither reconstruction projects points into the missing patch. The known
10 mm detail amplitude is reconstructed as 9.951 / 9.943 mm respectively;
lower noise does not come from flattening the measured detail.

The sparse oblique case is a real tradeoff: more accurate orientation stops
some accidentally large noisy weights and can require more observations to
reach the same final threshold. This evidence supports improving the optional
estimator, not enabling it globally or lowering final confidence.

For weighted depth means before voxelization, independent front-plane samples
reduce the new estimator's RMSE from 1.498 mm at four views to 0.533 mm at 32
views. Confidence-qualified pixel completeness rises from 96.3% to 100%.
The four-view preceding estimator gives 1.632 mm RMSE on only 22.6% of pixels.
The report includes a missing-aware capped error so sparse output cannot appear
better solely by omitting difficult measurements.

At 640×480, alternating warm CPU runs measure median confidence preparation
**13.60 → 6.56 ms**. A separate alternating benchmark on the same preallocated
volume measures **53.25 → 48.56 ms** for block discovery, confidence, voxel
coordinates/transforms and fusion. These exclude RGB/depth preparation, tracking,
network work, first allocation and extraction; they are not end-to-end scanner
or GPU speedups. Concurrent machine work can affect timings.

## Reproduction and checks

The report preserves source hashes, platform, runtime, native-kernel status,
error, completeness, detail amplitude and timings:
[`benchmarks/confidence-fusion.json`](benchmarks/confidence-fusion.json).

```
OMP_NUM_THREADS=4 python scripts/benchmark_confidence_fusion.py \
  --output docs/benchmarks/confidence-fusion.json
OMP_NUM_THREADS=4 python -m unittest \
  tests.test_depth_confidence tests.test_sensor_fusion tests.test_native_fusion -v
```

All 21 focused tests pass. They exercise exact and quantized oblique planes,
independent measurement accumulation, step/outlier/hole handling, narrow
surfaces, actual TSDF error/completeness and fine detail amplitude, existing
tracking/fusion, native shared-buffer safety, and NumPy/native/tensor numerical
parity on CPU. CUDA execution and real-device surface accuracy remain unmeasured.
