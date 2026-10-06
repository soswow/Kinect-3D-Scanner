# Native backend performance

Measured on 7 October 2026, Apple M1 Pro (10 cores, 32 GiB), macOS, Python 3.13.7, Open3D 0.20.0,
NumPy 2.5.3, and OpenCV 5.0.0. Implementation: `603c432`. The optional extension
is `kinect-scanner-native` 0.2.0, API version 2, built with Apple Clang 21,
C++17, `-O3`, and `-ffp-contract=off`.
The final measurement harness is `93baae6`; aggregate evidence is in
[native-performance.json](benchmarks/native-performance.json).

## What moved into C++

- Depth clipping, ROI masking, and the existing four-neighbour filter. Filtering
  reads an immutable clipped snapshot, so pixel order cannot change support.
- Calibrated RGB lens projection, visibility, and nearest-surface occlusion.
  OpenCV still performs bilinear color sampling.
- CPU confidence-weighted TSDF projection, observation gates, weighted averages,
  and attribute updates. Updates operate on Open3D's shared CPU buffers, retaining
  the existing 128-block batch limit and NumPy's repeated-index behavior.

The camera transforms stay in NumPy to preserve BLAS accumulation order. A
regression fixture showed that changing that order can move a float32 projection
across an OpenCV interpolation bin and change one delivered color pixel. The
native path now preserves that fixture exactly. Confidence calculation, tracking,
cloud preparation, normals, TSDF allocation/extraction, pose refinement, fragment
verification, and CUDA fusion retain their existing implementations.

All native loops release Python's GIL. The extension checks dtype, rank, layout,
alignment, shape relationships, and mutable attribute storage before computation.
Fusion additionally checks overlap and every voxel index before changing shared
attributes. No fast math or implicit copying of mutable buffers is enabled.

## Paired helper measurements

All **376 recorded frames** produce byte-identical prepared RGB and depth in
native and NumPy modes. Each session's preparation timing uses five spread frames
and five warmed, alternating-order pairs per frame. Image decoding, environment
switching, and garbage-collection changes are outside the timed region.

| Recording | Frames checked | NumPy preparation median | Native preparation median | Median paired speedup |
|---|---:|---:|---:|---:|
| chest-3 | 126 | 22.47 ms | 11.41 ms | 1.98× |
| chest-2 | 200 | 22.13 ms | 11.32 ms | 1.96× |
| chest | 50 | 22.18 ms | 11.42 ms | 1.95× |

Fusion uses three spread frames per session and five warmed, alternating pairs
per frame. Timed calls include confidence calculation, block discovery, voxel
coordinates, the NumPy transform, and native/reference voxel updates. They reuse
preactivated blocks and zero attributes outside timing; first block allocation,
RGB-D preparation, tracking, and final extraction are excluded. All active TSDF,
weight, and color bytes, plus observation statistics, match exactly after two
observations of each sampled frame.

| Recording | NumPy fusion median | Native fusion median | Median paired speedup |
|---|---:|---:|---:|
| chest-3 | 295.63 ms | 94.75 ms | 3.08× |
| chest-2 | 333.35 ms | 106.95 ms | 3.14× |
| chest | 456.81 ms | 143.35 ms | 3.18× |

These are helper speedups, not whole-scanner speedups. In the diagnostic chest-3
live replay, tracking and model refresh together accounted for about 69% of stage
time; weighted fusion accounted for 18%. Those large tracking/model operations
already run in Open3D's native backend. A complete language port using the same
calls would leave those costs substantially intact.

The useful boundary is the dense array work that previously built many NumPy
temporaries. The extension fuses those passes and updates existing buffers.
Rewriting the Python HTTP layer or orchestration would not remove the observed
Open3D tracking and fragment-verification costs. Those remaining costs need
separate algorithm or data-reuse changes, with their own accuracy validation.

## Complete recorded-session replays

Timings below include live storage, tracking, fusion, and Finish. They exclude
startup, image decoding, comparison, and export. Runs are sequential, with four
OpenMP threads, the same fixed seed, and the same settings within each comparison.
The 50-frame session has three runs per mode; chest-3 and chest-2 each have one pair.

| Recording | NumPy live + Finish | Native live + Finish | Processing time reduction | Final accepted frames |
|---|---:|---:|---:|---:|
| chest, median of three runs | 51.22 s | 45.77 s | 10.6% | 14 / 50 |
| chest-3 | 432.44 s | 367.88 s | 14.9% | 120 / 126 |
| chest-2, common 10,000-block final budget | 1,096.23 s | 978.01 s | 10.8% | 174 / 200 |

The first 50-frame pair was slower in native mode (53.11 s versus 51.65 s) because
tracking took longer; the two additional native runs took 45.04 s and 45.77 s.
All three NumPy runs took 50.27–51.65 s. This variation is why the complete-session
result uses the median and preserves every measured run.

Chest-3 live processing fell from 223.93 s to 190.83 s; Finish fell from 208.51 s
to 177.04 s. All 104 live acceptance decisions and all 120 final accepted indices
matched. The original 50-frame recording disables fragment reconnection, and
that choice is retained rather than changing which views are reconstructed.

An additional accumulated-fusion check replays all **308 final accepted views**
at the identical baseline poses (14 / 120 / 174 views by recording). For every
view, the NumPy update and native update start from the same accumulated voxel
state. All touched TSDF, weight, and color bytes and returned statistics match
exactly; 305 views update voxels with nonzero prior weights. Final volumes contain
2,481 / 7,112 / 6,551 blocks. Block IDs, capacity, inputs, settings, sources, and
extension remain unchanged. This check performs no tracking or timing.

The rewritten kernels are exact on the tested arrays. Whole reconstruction is
not bit deterministic: Open3D's parallel tracking, graph optimization, and
extraction can produce different poses and mesh topology between runs. For
chest-3, symmetric nearest-vertex distance was 1.30 mm RMS and 1.68 mm at the 95th
percentile; the largest pose translation difference was 6.12 mm. Meshes contained
200,138 versus 199,694 vertices. The largest individual vertex distance was
100.89 mm, so proximity of most vertices does not guarantee every boundary or
component is identical.

For the 50-frame native runs, the 95th percentile of nearest-vertex distances
was 0.32–0.41 mm, with RMS 0.47–1.29 mm and isolated maxima of 25–69 mm. Repeating
the NumPy path itself changed mesh topology, with RMS about 2.78 mm and maxima
about 140 mm against its first run. All six runs accepted the same 14 indices.
These comparisons are regression evidence, not measurements against an
independent reference surface.

Chest-2 live processing fell from 239.03 s to 185.03 s; Finish fell from 857.20 s
to 792.98 s. All 53 live and 174 final accepted indices matched. Its mesh had
244,837 versus 243,130 vertices; symmetric nearest-vertex distance was 2.63 mm RMS,
5.00 mm at the 95th percentile, and 76.30 mm maximum. The largest pose translation
difference was 14.57 mm, with a largest rotation difference of 0.25 degrees.
These differences remain material when assessing individual views or boundaries;
this single pair does not prove whole-pipeline accuracy equivalence.
Its unchanged Open3D live tracking also varied from 176.67 s to 142.16 s, so the
entire measured runtime difference cannot be attributed solely to the C++ loops.
The alternating paired helper measurements isolate those loops more directly.

Peak process memory did not improve: chest-3 measured 1.59 GiB in NumPy mode and
1.79 GiB in native mode. RSS includes loaded recordings and library allocators;
these measurements support a processing-time claim, not a memory-saving claim.
Chest-2 likewise measured 1.67 GiB versus 1.92 GiB.

## Installation and selection

From the repository root, using the scanner's Python environment:

```bash
python -m pip install ./native
```

This requires a C++17 compiler and Python development headers. Build dependencies
are declared in `native/pyproject.toml`; there is no runtime compiler requirement.
Reinstall after native source changes or moving to a different Python environment.

`KINECT_NATIVE=auto` uses the installed compatible extension and falls back to
NumPy when absent. `on` requires native kernels and reports an unavailable or
incompatible extension as an error. `off` runs the reference implementation.
Health/status and reconstruction exports expose `backend.native_kernels`.

## Reproduction

The session ZIPs are under the primary checkout's ignored `export/` directory.
Run from that checkout with the same environment as the server:

```bash
python scripts/benchmark_native_kernels.py \
  export/chest-3-scan-session_20261006_215208.zip \
  export/chest-2scan-session_20261006_090302.zip \
  export/chest_20261005_230320.zip --all-frames

python scripts/benchmark_native_fusion.py \
  export/chest-3-scan-session_20261006_215208.zip \
  export/chest-2scan-session_20261006_090302.zip \
  export/chest_20261005_230320.zip

KINECT_NATIVE=off python scripts/profile_session.py \
  export/chest-3-scan-session_20261006_215208.zip --finish \
  --output benchmark-output/chest3-python.json
KINECT_NATIVE=on python scripts/profile_session.py \
  export/chest-3-scan-session_20261006_215208.zip --finish \
  --output benchmark-output/chest3-native.json \
  --compare benchmark-output/chest3-python.json
```

Run timing commands sequentially. Scripts default to `OMP_NUM_THREADS=4` and
`KINECT_BLOCK_COUNT=5000`, matching the launcher configuration. Original session
settings, including confidence, thresholds, voxel size, and reconnection choices,
are retained. Pose seeds are optional proposals that Finish revalidates; they
cannot skip fusion for sessions without fragment reconnection.

Chest-2's archived final budget is 5,000 blocks. The original-budget native replay
estimated 6,557 required blocks at 5 mm, returned the existing capacity error,
and retained the live reconstruction; it did not build a final mesh. That run is
excluded from speed comparisons. Successful chest-2
comparisons use `--final-block-count 10000` for **both** modes. The profiler records
this explicit override and the original settings hash. No recording is edited.
The successful pair required 6,551 and 6,548 blocks respectively.
The earlier [NumPy algorithm review](benchmarks/algorithm-review.json) also
reported 6,561 required blocks for this recording, exceeding its saved limit.

```bash
KINECT_NATIVE=off python scripts/profile_session.py \
  export/chest-2scan-session_20261006_090302.zip --finish \
  --final-block-count 10000 --output benchmark-output/chest2-python.json
KINECT_NATIVE=on python scripts/profile_session.py \
  export/chest-2scan-session_20261006_090302.zip --finish \
  --final-block-count 10000 --output benchmark-output/chest2-native.json \
  --compare benchmark-output/chest2-python.json
```

`--cprofile` helps identify call sites but is unsuitable for speed comparisons.
Open3D calls on this Python build cause incomplete/inconsistent cProfile
accounting. Phase and stage wall clocks are authoritative; cProfile self time
also includes opaque compiled calls and is not a measurement of Python overhead.

## Validation and limits

The native-enabled full suite ran **317 tests: 315 passed and two CUDA tests were
skipped** because CUDA is unavailable. Coverage includes exact array parity, boundary
rounding, independent occlusion fixtures, immutable observations, shared Open3D
buffers across multiple batches and camera poses, input rejection before mutation,
and the existing HTTP/WebSocket/client reconstruction workflow.

The recorded sessions have no independent reference trajectory or ground-truth
surface. Accepted indices, pose differences, and surface differences can detect
a regression against the existing reconstruction, but do not establish absolute
scan accuracy. Original inputs are hashed before and after measurement and are
never modified. Raw sessions, geometry, per-frame diagnostics, and detailed logs
remain in ignored local benchmark output.
The preserved detailed reports, meshes as NumPy geometry arrays, and test logs
are under `benchmark-output/native-performance-20261007/` in the primary checkout.
Original report bytes are preserved for provenance hashes; copies under its
`portable/` directory relocate geometry and baseline paths to that checkout.
The ignored `check_accumulated_fusion.py` and `accumulated-fusion-parity.json`
preserve the complete fixed-trajectory validation method and per-view evidence.

The C++17 sources include platform-specific compiler flags for Windows, but this
installation and these performance numbers are verified only on Apple Silicon
macOS. CUDA speed and new physical Kinect capture are not measured by these replays.
