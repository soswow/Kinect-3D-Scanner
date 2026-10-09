# Recorded-session CUDA performance

This is the initial fusion-only study. The subsequent tracking, retrieval,
model-preparation and architectural experiments are documented in
[CUDA pipeline experiments](CUDA_EXPERIMENTS.md); use that follow-up for the
current pipeline's behavior and measurements.

Measured on 7 October 2026 using the exported chest-3 and chest-4 sessions, on
Windows 11 with an Intel Core i7-12700F, 64 GB RAM and an RTX 3080 Ti (12 GB).
The environment uses CUDA-enabled Open3D 0.20.0, Python 3.12.7, NumPy 2.5.3,
eight OpenMP threads for native kernels and the installed API-v2 native CPU
extension. A later runtime check established that Open3D 0.20 uses TBB with
20 threads here; the OpenMP setting did not cap Open3D registration.

## Results and scanning expectations

| Session | Frames | CPU live | Existing CUDA live | Optimized CUDA live | CPU Finish | Optimized CUDA Finish |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chest-3 | 126 | 211.6 s | 197.0 s | 195.5 s | 424.2 s | 357.9 s |
| chest-4 | 164 | 219.8 s | 212.5 s | 208.2 s | 267.8 s | 248.8 s |

Optimized CUDA reduced live processing time by **5.3–7.6% versus CPU**, and
**0.8–2.0% versus the existing CUDA path** in these single full replays. The
whole-pipeline differences are small relative to tracking/recovery variation;
they are observations, not established repeat-run averages. At the preserved
capture quality, measured processing capacity is **0.64–0.79 recorded views/s**.
Typical optimized frame time is **0.88–1.05 s**; the p95 is **3.55–5.02 s**.
Processing plus network/capture overhead determines the actual preview cadence.
Good continuous tracking may be quicker than these recovery-heavy recordings.

Confidence fusion is a stronger, repeated result: the median across the six
sample medians is **142.5 ms on native CPU, 104.7 ms on existing CUDA, and
33.5 ms on fused CUDA**. Depending on the view, the new implementation is
**2.59–3.65× faster than existing CUDA** and **2.79–5.50× faster than native CPU**.
Each sampled frame has five synchronized, warmed measurements. Two repeated
observations from an identity camera pose have identical block keys and
bit-identical TSDF, weight and color attributes across all three implementations.
Nonidentity transforms are checked separately against the CUDA tensor reference.

Tracking and model refresh dominate live time, and fragment registration and
verification dominate Finish. Finish remains a wait of several minutes. Its
timing difference includes different graph-search paths: chest-3 tested 46
candidate pairs on CPU and 40 on optimized CUDA. It cannot be attributed entirely
to the faster fusion kernel. No tracking thresholds or voxel sizes were lowered.

Both CPU and optimized CUDA retain **121/126** views after Finish for chest-3,
and **162/164** for chest-4, with identical final frame indices. Live acceptance
is 106/126 on all chest-3 backends; chest-4 accepts 133 on CPU, 130 on existing
CUDA, and 132 on optimized CUDA. Track loss still requires recovery.

Actual triangle-surface comparison uses 30,000 area-weighted points per mesh,
fixed coordinates and no pose/scale fitting. Optimized-versus-CPU surface p95 is
**0.159 mm for chest-3** and **0.020 mm for chest-4**. Precision/completeness within
5 mm is 99.993%/99.970% and 100%/100%, respectively. These are reconstruction
agreement checks, not absolute accuracy against independent ground truth.
Nearest-vertex outliers reach 20.8 mm and 7.4 mm, so results are not bit-identical.

The compact reproducible record is
[cuda-session-performance.json](benchmarks/cuda-session-performance.json).
The [saved initial report](benchmarks/cuda-study/REPORT.md) and
[performance chart](benchmarks/cuda-study/performance.png) are historical results.
The [follow-up study](CUDA_EXPERIMENTS.md) contains the current recommendation. Validation passed **47 tests**, including real CUDA parity,
hash-map growth, half-pixel rounding, compiler fallback and no-retry-on-update-failure
checks, plus CPU fusion, ICP, replay, budgets and server API regressions.

## What the measurements include

Complete replays preserve every archived frame, timestamp, calibration and scan
setting. Live time includes storage, depth preparation, registration, tracking,
fusion and model refresh. Finish is timed separately, including reconnection,
refinement and mesh extraction. Imports, ZIP decoding, capture hardware, network
transport, texturing and writing geometry reports are outside those timings.
These are processing rates for already recorded views, not Kinect capture FPS.

Detailed reports, geometry and contact sheets are under the ignored
`benchmark-output/cuda-study/` directory. Raw captures are never committed.
The initial exploratory CUDA replay is excluded from controlled comparisons:
source files were edited while it ran, and Open3D reported a driver-shutdown
error during interpreter teardown after saving its report.

The controlled CUDA tensor baseline also saved a complete synchronized report
before that Windows wheel teardown error. Its measurements are retained.
Isolated profiling CLIs now flush closed reports and exit before the problematic
Windows CUDA DLL finalizers. Runtime errors still fail the subprocess normally;
this workaround is outside measured work and is not applied to the server.
All six controlled replays report unchanged archive and implementation checksums,
with a common implementation SHA-256 of
`6720d48cf532ea65e659b694e1c7c6c15f72a251a5dc2c8ebb9d116e1f5a7989`.
After measurement, optional compiler-error fallback was hardened and validated;
the CUDA kernel and successful update path were unchanged.

## Implemented improvement

Confidence-weighted fusion previously used many small tensor operations for
each batch of 128 blocks. The optional fused CUDA path combines projection,
confidence/depth/truncation gates and weighted TSDF/color updates in a CUDA
kernel. DLPack borrows Open3D's GPU attribute buffers directly. It processes at
most 1,024 blocks per batch and synchronizes at library boundaries to preserve
buffer lifetime and accurate timing.

The confidence model, observations, truncation and accumulated weights are
unchanged. Compile options disable fused multiply-add and fast math. Tests
exercise repeated observations, holes, rejected depths, zero/nonfinite
confidence, nonidentity camera transforms and volume growth. Real recorded
samples additionally compare canonical block keys and all voxel attributes
against the native CPU implementation.

CPU execution is unchanged. Without CuPy, `auto` uses the existing tensor CUDA
path and reports its fallback reason in reconstruction diagnostics. An explicit
`fused` request raises when setup is unavailable. Kernel execution failures are
never retried against a volume that may have already been updated.

## Reproduce

In an existing environment containing CUDA-enabled Open3D and a CUDA 12 toolkit:

```powershell
python -m pip install -r requirements-cuda-fusion.txt
$env:OMP_NUM_THREADS = '8'
$env:KINECT_NATIVE = 'on'
python scripts/benchmarks/benchmark_cuda_fusion.py export/chest-3-scan-session_20261006_215208.zip export/chest-4-scan-session_20261007_173028.zip
python scripts/profile_session_backends.py export/chest-3-scan-session_20261006_215208.zip export/chest-4-scan-session_20261007_173028.zip --finish --finish-runs cpu cuda-fused --repeats 1 --runs cpu cuda-tensor cuda-fused
```

`scripts/compare_session_profiles.py` adds actual triangle-surface comparison to
two matching finished profiles. `scripts/summarize_cuda_study.py` generates the
compact JSON, local report and PNG; its plotting step needs Matplotlib. This
measurement used CuPy 13.6.0 with the installed CUDA 12.4 toolkit and NVIDIA
driver 595.79. The server environment gained only the optional CuPy dependency
and its lock dependency, fastrlock.

`KINECT_CUDA_FUSION=auto` is the default. Use `tensor` to reproduce the previous
CUDA integration and `fused` to require the new implementation. CUDA tracking
still uses Open3D; visual tracking, recovery, final pose refinement and texturing
retain their existing CPU work. GPU fusion speedup does not imply that those
stages are accelerated.

The profiling scripts now support Windows peak working-set memory and use a
high-resolution timer for stage measurements. Host RAM measurements do not
include GPU VRAM.

Technical references: [Open3D tensor interoperability](https://www.open3d.org/docs/latest/tutorial/core/tensor.html),
[CuPy RawKernel](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html),
[CuPy installation](https://docs.cupy.dev/en/v13.4.1/install.html).
