# CUDA pipeline experiments

Follow-up to [the initial fusion study](CUDA_PERFORMANCE.md), measured on
8 October 2026 with the same RTX 3080 Ti 12 GB, i7-12700F, 64 GB RAM, Windows,
CUDA-enabled Open3D 0.20, eight OpenMP threads for native kernels, and the
runtime's default 20 TBB/OpenCV threads. Open3D 0.20 uses TBB: setting
`OMP_NUM_THREADS` does not cap its registration threads. The small initial improvement
was real: accelerating fusion alone left most tracking and recovery work intact.

Windows Update restarted this machine at 06:18 and 06:26 Sydney time on
8 October, interrupting an unfinished replay and replacing NVIDIA driver
595.79 with 610.88. The completed pre-update studies remain historical results.
The final hybrid comparison reruns its control on 610.88, and new profiling
reports record GPU identity/driver; resume rejects a different driver.

The source observations are the original chest-3 (126 views) and chest-4 (164
views) ZIP exports. Calibration, measured depth, every captured view, pose
evidence and final reconstruction budgets remain intact. The primary comparison
preserves 5 mm live and final fusion. A separately labelled preview tradeoff
uses 10 mm live fusion and retains the same 5 mm final detail and memory budget. Archived
poses never initialize these experiments. The exports contain selected captures
several seconds apart, not the complete moving camera stream. Processing rates
below describe these views, not USB capture or continuous camera tracking FPS.

Per-view latency includes rejected observations and recovery attempts. Stage
diagnostics are not all disjoint: tracking verification is part of tracking.
Host peak resident memory is reported separately from GPU memory; it is not a
VRAM measurement. Finish timing includes graph-search paths, so the CPU/CUDA
difference cannot automatically be credited entirely to kernel acceleration.

## Current release and expected scanning performance

The [script index](../scripts/README.md) distinguishes maintained tools, active
research and archived experiments. Current reproduction commands below use the
organized paths. Saved reports retain their original measured paths and hashes;
moving helpers changes their source fingerprints and requires fresh research
proofs before timing a changed implementation.

The [saved detailed report](benchmarks/cuda-pipeline/REPORT.md) and
[processing chart](benchmarks/cuda-pipeline/pipeline-current-release.png) are
available in a fresh checkout. [Saved CUDA results](benchmarks/CUDA_REPORTS.md)
describes the published summaries and the local proof artifacts.

The frozen `9331dee7…` release has twelve complete comparisons: one clean run
per CPU mode and archive, and two alternating-order runs per CUDA mode and
archive. All use driver 610.88, the same thread resources and original raw
observations. Every reconstruction passed the same final-view and surface
agreement gates. The CUDA rows below enable exact native RGB/depth preparation;
confidence calculation remains on CPU. The full report includes both CUDA
repetitions and the separate optional-confidence comparison.

| Archive | Workflow | Live processing | Finish | Combined processing | Live retained / final retained |
| --- | --- | ---: | ---: | ---: | ---: |
| chest-3 | Original CPU, 5 mm preview | 211.66 s | 450.06 s | 661.73 s | 106 / 121 |
| chest-3 | CPU adaptive, 10 mm preview | 86.13 s | 348.01 s | 434.14 s | 99 / 121 |
| chest-3 | CUDA adaptive + input, 10 mm preview | 70.12 s | 301.44 s | 371.56 s | 99 / 121 |
| chest-4 | Original CPU, 5 mm preview | 226.13 s | 270.84 s | 496.97 s | 130 / 162 |
| chest-4 | CPU adaptive, 10 mm preview | 88.07 s | 283.40 s | 371.47 s | 144 / 162 |
| chest-4 | CUDA adaptive + input, 10 mm preview | 64.99 s | 230.59 s | 295.58 s | 144 / 162 |

CUDA values are medians of two complete runs. Combined processing ranged from
365.55–377.57 s for chest-3 and 294.46–296.71 s for chest-4. These ranges are
observed repetition ranges, not statistical confidence intervals. The original
chest-3 CPU control was repeated after a brief read-only header search overlapped
its Finish phase; the affected run is preserved and excluded from this table.
Exact report hashes and the activity interval are retained in
`benchmark-output/cuda-pipeline/current-release-activity.json`.

The recommended workflow raises live processing capacity by **3.02× / 3.48×**
and reduces combined processing by **43.9% / 40.5%** against the original CPU
workflow. That includes caching, adaptive tracking and the coarser preview.
Comparing the same adaptive 10 mm policy instead, the CUDA backend reduces
live time by **18.6% / 26.2%** and combined time by **14.4% / 20.4%**. Finish
includes graph search and recovery; these workflow differences cannot all be
assigned to individual CUDA kernels. A single CPU repetition limits precision
of the percentage estimates.

Expect a typical selected-view processing time of **203 / 172 ms**, with
p95 at **2.02 / 1.72 s** when recovery is needed. Average capacity is
**1.80 / 2.52 selected views/s**. Finish takes roughly **5.0 / 3.8 minutes**.
These are sparse archived-view rates, not camera capture FPS. Capture and
processing can overlap, so the summed times are not a prediction of scan
wall-clock duration. Final reconstruction stays at 5 mm with the original
10,000-block budget and confidence threshold 2. All final view indices match
the CPU reference; worst surface p95 is below 0.2 mm and agreement within
5 mm exceeds 99.9%. This measures agreement with that reference, not absolute
accuracy against an independent measurement.

Use `./scripts/start_cuda_server.ps1 -Recipe adaptive -CudaInput auto`
and select Live voxel 10 mm / Final voxel 5 mm in the client. CUDA confidence
is optional and defaults off because its complete-processing results are mixed.

## Earlier matched-source matrix

The initial controlled comparison ran four recipes twice on each archive, in isolated
processes, reversing recipe order for the second repetition. The final comparison
runs fresh CPU and CUDA controls, a conservative CPU-tracking/CUDA-fusion hybrid,
and the explicit 10 mm preview tradeoff through complete Finish. Imports, decoding
the ZIP, writing reports, USB capture and network transport are outside measured
processing. Source and archive checksums must remain fixed during each matrix.
Separate Finish replays measure reconnection, pose refinement, reintegration
and meshing. Final comparisons use triangle surfaces in the original coordinates
without pose/scale fitting; the CPU reconstruction is a reference, not ground truth.

Generated results are in
[cuda-pipeline-experiments.json](benchmarks/cuda-pipeline-experiments.json).
The detailed local report, chart, per-view diagnostics and geometry stay under
the ignored `benchmark-output/cuda-pipeline/` directory.

The matched-source comparison contains a CPU control, CPU-only caching/model
preparation, the CUDA hybrid, and CPU/CUDA versions of the adaptive preview.
The matched-backend tables isolate CUDA's additional contribution for the same
tracking policy and resolution. The larger gain against the original CPU
pipeline includes algorithm and preview-resolution changes. Matched-source
controls are single complete replays; earlier repeated CUDA/prototype runs are
retained separately with their source fingerprints. Graph-search work varies
even when the final surface agrees, particularly on chest-3.

| Archive | Profile | Live | Finish | Complete | Live capacity | Live retained | Typical / p95 view | Final surface p95 | Reference gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| chest-3 | CPU control, 5 mm | 210.5 s | 431.8 s | 642.3 s | 0.60 views/s | 106/126 | 1281 / 5003 ms | 0.159 mm | PASS |
| chest-3 | CPU cached, 5 mm | 190.1 s | 346.4 s | 536.6 s | 0.66 views/s | 106/126 | 750 / 5250 ms | 0.165 mm | PASS |
| chest-3 | CUDA hybrid, 5 mm | 176.1 s | 392.1 s | 568.2 s | 0.72 views/s | 106/126 | 500 / 5437 ms | 0.000 mm | PASS |
| chest-3 | CUDA adaptive, 5 mm | 133.4 s | 348.7 s | 482.1 s | 0.94 views/s | 113/126 | 453 / 4937 ms | 3.693 mm | FAIL |
| chest-3 | CPU adaptive preview, 10 mm | 85.8 s | 358.8 s | 444.6 s | 1.47 views/s | 99/126 | 320 / 2316 ms | 0.179 mm | PASS |
| chest-3 | CUDA adaptive preview, 10 mm | 78.2 s | 304.2 s | 382.4 s | 1.61 views/s | 99/126 | 250 / 2230 ms | 0.191 mm | PASS |
| chest-4 | CPU control, 5 mm | 216.3 s | 263.8 s | 480.1 s | 0.76 views/s | 133/164 | 985 / 3593 ms | 0.018 mm | PASS |
| chest-4 | CPU cached, 5 mm | 199.5 s | 265.3 s | 464.8 s | 0.82 views/s | 133/164 | 625 / 3721 ms | 0.000 mm | PASS |
| chest-4 | CUDA hybrid, 5 mm | 173.6 s | 229.5 s | 403.1 s | 0.94 views/s | 132/164 | 422 / 3626 ms | 0.000 mm | PASS |
| chest-4 | CUDA adaptive, 5 mm | 155.9 s | 247.0 s | 402.9 s | 1.05 views/s | 143/164 | 422 / 3608 ms | 3.259 mm | FAIL |
| chest-4 | CPU adaptive preview, 10 mm | 88.1 s | 274.7 s | 362.8 s | 1.86 views/s | 144/164 | 328 / 1842 ms | 0.003 mm | PASS |
| chest-4 | CUDA adaptive preview, 10 mm | 74.2 s | 240.0 s | 314.2 s | 2.21 views/s | 144/164 | 219 / 1772 ms | 0.022 mm | PASS |

Every profile uses 5 mm final reconstruction and the same 10,000-block final budget. The 10 mm rows change live preview resolution. Times include rejected observations. A PASS means the same final view indices and the declared surface agreement gate, not independently established absolute accuracy.

For these exported views, the CUDA adaptive preview processes a typical view
in 219–250 ms; difficult views and recovery raise p95 to 1.77–2.23 s. Average
capacity is 1.61–2.21 selected views/s, and Finish takes 4.0–5.1 minutes. The
coarser preview retained fewer live chest-3 views (99 rather than 106), while
Finish recovered the same 121 final views; chest-4 retained 144 live views and
162 final views. Every rejected raw observation remains available to Finish.
Live processing and camera capture can overlap, so summed replay processing
time is not a prediction of wall-clock scan duration.

Against the original CPU workflow, the validated fast preview increases live
processing capacity by 2.69× and 2.91× and reduces complete processing time by
40.5% and 34.6%. The following matched-policy comparison isolates CUDA's
additional benefit from the changes that also accelerate a CPU server.

| Matched policy | Archive | CUDA live time reduction | CUDA complete time reduction |
| --- | --- | ---: | ---: |
| ORB / cached 5 mm | chest-3 | 7.4% | -5.9% |
| ORB / cached 5 mm | chest-4 | 13.0% | 13.3% |
| Adaptive / 10 mm preview | chest-3 | 8.8% | 14.0% |
| Adaptive / 10 mm preview | chest-4 | 15.7% | 13.4% |

Negative reduction means CUDA was slower in that complete replay. Finish includes different graph-search paths: record candidate/tested-pair counts when interpreting these single runs. The preceding table compares complete matched workflows; it does not assign every timing change to a CUDA kernel.

The primary matrix uses runtime fingerprint `aeafd9929e47a75f9c5aaaa58c098de2f018d254c6e2a2cb7299784c4c5994aa` with NVIDIA driver 610.88. Its source snapshot is `benchmark-output/cuda-pipeline/runtime-measured.zip`; 153 backend tests passed. The adaptive guard, exact GPU input and failure handling use fingerprint `67d115f112ce279d9b569ca6e2422b5469c43176549d2599415f7a26715be1da`, saved in `runtime-input-measured.zip`, with all 191 backend tests passing. The paired GPU-input replays use that fingerprint separately. The final exact-confidence release is `9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a`, saved in `runtime-final.zip`; all 210 backend tests passed, including hardware CUDA checks, with no source changes or skips. Supported pose verification and fusion mathematics remain unchanged. Two Qt-dependent client checks were excluded from this server environment.

## Changes being measured

Default pose authorization remains multiscale tracking with full CPU geometric
verification and immediate recovery. ORB remains the default live feature type. The
faster pose solvers, SIFT live proposals, visual-first Finish and deferred
recovery are opt-in experiments, with the failed full-session outcomes below.
CUDA matching, immutable keyframe caching and lazy model preparation are
explicit options in the measured hybrid recipe. Plain server startup retains
CPU matching, eager model preparation and uncached raw targets; faster pose
solvers remain disabled. The CUDA launcher selects backend options for a measured
recipe; the client selects fusion resolutions, budgets and scan settings.

The `adaptive` recipe keeps the original verified ORB decision whenever it
succeeds, then tries SIFT through the same pose gates after ORB rejects a view.
Its secondary descriptor bank is bounded by the same observed keyframe set.
The fallback reuses immutable raw clouds, restores the ORB bank even on errors,
and releases both banks on a new scan. It does not cache pose authorization or
weaken depth, feature, motion or Finish checks. This differs from switching
every live decision to SIFT, which failed chest-4's finished surface comparison.

The same adaptive policy at 5 mm live resolution failed both full-session
surface gates despite recovering the same final view indices. Chest-3 retained
96.98% precision and 97.52% completeness within 5 mm; chest-4 retained 97.59%
and 97.13%. Its lower processing time is a rejected result. The public adaptive
policy therefore activates only for 10 mm live / 5 mm effective final
reconstruction. Other settings fall back to the original ORB policy and report
the reason. Research replays require an explicit experimental override to
exercise the rejected setting.

1. **CUDA descriptor retrieval.** ORB Hamming distances and mutual ratio matches
   run in a batched CUDA kernel. Quantized SIFT descriptors use exact float32
   squared-distance algebra followed by mutual two-neighbour ratio filtering.
   The bounded bank contains at most 40 views, 1,200 ORB descriptors or 1,280
   quantized SIFT descriptors per view. SIFT retains equal-response ties beyond
   its requested feature count. Chest-4 view 70 had 1,201 features: this one extra
   feature excluded 2,854 bank pairs in the exhaustive comparison study under the
   earlier limit. This is a test count, not the number of runtime fallback calls.
   The allowance keeps
   every feature and caps the largest SIFT distance matrix at 250 MiB. All
   2,854 newly supported comparisons match CPU exactly, joining the original
   39,628 checks: all 42,482 archive view/type pairs were checked with zero
   mismatches. Unsupported/nonquantized descriptors retain the CPU matcher.
2. **Immutable keyframe preparation.** Registration point-cloud levels,
   normals and GPU uploads are reused for unchanged observed keyframes. Mutable
   model clouds remain separate; source levels live only for a registration
   decision. Eviction and scan reset release caches.
3. **Prepare the model only when needed.** Live surface snapshots still refresh.
   Full CPU normals, registration pyramids and their GPU uploads are deferred
   while verified visual tracking suffices, then prepared from the already
   extracted snapshot before model-based recovery.
4. **Stronger live SIFT retrieval.** More reliable scale-tolerant appearance
   proposals avoid expensive failed recovery paths. Extraction and PnP remain
   on CPU; matching runs on CUDA. Raw depth support, distributed feature
   identities, motion bounds and reciprocal geometric checks still authorize
   each integrated view.
5. **CUDA geometric verification.** Existing full-cloud robust point-to-plane
   registration can run with Open3D CUDA tensors. Distances and maximum iteration
   counts remain 120/60/30 mm and 40/30/20; Huber loss remains 10 mm. Float32 GPU
   clouds are a numerical implementation change, so final geometry must be checked.
6. **Optional deferred recovery.** A view lacking verified visual alignment is
   stored for Finish instead of repeatedly attempting expensive live recovery.
   It contributes no geometry until independently connected. The live preview
   can have fewer views; this option trades immediate recovery for lower capture
   latency. Finish work still exists and must be included in total-time comparisons.
7. **Measured-feature pose refinement.** Fit rigid motion to fixed RGB-D feature
   identities, then jointly refine their 3D and image reprojection errors. This
   small CPU solve avoids a full ICP proposal when distributed visual support,
   forward/reverse raw-depth overlap, residuals and motion bounds all pass.
   Failed proposals retain the original ICP fallback. Raw-cloud normals are
   prepared only if a geometric fallback needs them.
8. **Finish descriptor batching and verification order.** Batch exact SIFT
   retrieval on CUDA for immutable camera views. Optional visual-first bridge
   verification uses the existing held-out depth checks and two independently
   moving camera witnesses on both sides. All competing graph proposals and
   ambiguity checks still run. Optional measured local fragment poses retain
   visual, reciprocal held-out depth and per-step motion checks.

The earlier fused confidence kernel remains in use. Confidence weights, TSDF
truncation, primary fusion resolution and final budgets are unchanged. The
optional preview comparison normalizes only live voxel size and the equivalent
explicit final voxel size. It rejects unfinished meshes, a changed actual final
resolution, changed budgets or other changed settings; its three regression
checks cover those restrictions.

## Experiments that did not justify a production replacement

**Direct CUDA RGB-D odometry:** on sampled pairs, projective GPU solves took
roughly 5–10 ms, compared with about 75–650 ms for CPU full-cloud registration.
This attractive component number did not survive complete sparse-session replay.
On chest-3, an early incorrect chain lost a critical turn; only 16/126 views were
accepted and processing took about 393 seconds. Falling back after failed feature
and overlap checks did not resolve cases whose biased poses still passed those
checks. This route remains explicitly experimental and disabled.

**GPU float64 ICP:** double precision was substantially slower on this GeForce
GPU. It is not the recommended route to faster verification. A second split
kept Open3D's original CPU Huber pose solve and convergence rules, caching only
CUDA float64 neighbour indexes. It reproduced the sampled CPU results closely
but was roughly 2.5–5 times slower. A float32 candidate search with exact CPU
reranking and CPU fallback for uncertain candidates reproduced all ten sampled
poses to numerical precision, yet its median slowdown was 8.7 times. These dense
33,000–95,000-point clouds favour the existing CPU search. Neither research
implementation is connected to the production backend.

A custom pruned CUDA KD-tree preserved double-precision coordinates and the
original CPU Huber solve. It passed 3,095 synthetic/boundary/duplicate queries
and reproduced all ten real correspondence arrays and poses to numerical
precision. Nevertheless, median registration was 1.50 times slower than CPU.
One difficult pair sped up, but that does not establish a pipeline improvement.

**Full-cloud GPU float32 verification:** the complete Finish runs became slower
   despite testing similar numbers of graph proposals. More seriously, chest-4's
   reconstructed triangle surface differed from the CPU reference by about
   28 mm at the 95th percentile; only about 46% of either surface lay within
   5 mm of the other. The same 162 retained views did not establish equivalent
   geometry. This route is now explicitly opt-in; the default geometric verifier
   remains the original CPU implementation.

**Coarse-to-fine full-cloud verification:** 40/20 mm coarse levels followed by
   refinement on the complete original cloud shortened sampled registrations
   by roughly 2–7 times. Some difficult proposals moved 17–48 mm relative to
   the original solver or failed visual gates. A faster component alone does
   not justify changing graph geometry; this remains a research script.

**Thread policy and descriptor matrix padding:** changing OpenMP from eight
   threads to one/four or using passive waiting barely affected the matched
   28-view pilots. OpenCV still reported 20 threads, so these are not evidence
   for OpenCV thread tuning or Open3D TBB tuning. A separate experiment using
   Open3D's actual `set_max_threads` API found median registration slowdowns of
   1.30/2.18/7.76 times for 8/4/1 threads compared with the default 20. Actual
   OpenCV thread control preserved all extracted features: ORB stayed about
   5.8 ms at every tested policy; SIFT improved from 31.5 to 26.2 ms with eight
   threads. This small SIFT-only saving does not justify a global default change.
   GEMM padding did not offer a reliable additional
   benefit. Removing square roots from the full SIFT distance reduction did;
   only the selected two distances now pay for square roots, preserving exact
   tested CPU correspondence identities. TF32 was investigated in isolated
   processes and is not enabled by the scanner.

**Deferring recovery with ORB alone:** the 28-view chest-3 pilot finished very
quickly but accepted only its reference view. Throughput obtained by losing the
scan is not an improvement. Stronger visual proposals were necessary before
deferring recovery became useful.

**Deferring all recovery with SIFT:** chest-3 improves substantially, but chest-4
loses an early connection and accepts only 4/164 live views. The approximately
21-second chest-4 processing time is therefore a failure, not a usable speedup.
This is not a suitable general live-scanning default, even if Finish can later
recover the observations. Bounded recovery must still preserve critical turns.

**GPU frame-to-model raycasting:** a research prototype using Open3D's dense
SLAM model had moving synthetic processing medians around 10–12 ms. However,
point-to-plane tracking drifted about 26.5 mm RMS; hybrid tracking drifted about
9.2 mm RMS. Even repeatedly processing one stationary recorded view produced
about 17.3 mm RMS drift. These controlled processing rates demonstrate GPU
headroom, not verified scanner performance. The prototype uses uniform fusion
and lacks this scanner's independent pose gates, recovery and global correction.

**Feature-only live and Finish poses:** a small measured-feature solve looked
promising on controlled synthetic scenes, but complete recorded scans failed
the geometry check. With measured live poses and measured/visual-first Finish,
chest-3 retained 122 views but its surface p95 difference reached 7.4 mm;
chest-4 retained only 46 views and reached 44.4 mm. Keeping CPU ICP while
switching live retrieval to SIFT retained all 162 final chest-4 views, yet its
surface p95 difference was 9.2 mm. Different live poses can change which graph
bridges survive even when Finish uses the same solver. These options remain
experimental and disabled by default.

**Combined cached/lazy pipeline with CUDA Finish matching:** the first complete
chest-3 run retained the same 121 views but differed by 39.7 mm at surface p95
(85.0% precision and 89.8% completeness within 5 mm). Live time fell to 168.8 s,
but that combined recipe cannot be recommended on speed alone. A feature-seeded
fine ICP run with the same retrieval/preparation options retained 121 views and
matched the reference within 0.16 mm p95. Further runs isolate the live policy
and Finish retrieval; component correspondence equality alone is insufficient
to establish a stable recovered graph.

The feature-seeded fine ICP repeat retained the same 121 chest-3 views but
differed by 5.8 mm p95, with only 94.2%/91.3% precision/completeness within
5 mm. Its 162 chest-4 views remained close to the reference. Trying visual
bridge verification first retained only 65/126 and 79/164 final views. These
failed complete runs override the attractive initial component timings; neither
option is recommended as a production replacement.

The post-update repeat with the original CPU live tracker also tested SIFT
retrieval at both 5 mm and 10 mm live voxels, preserving 5 mm final reconstruction.
Chest-3 passed both final checks (121 views, approximately 0.19 mm surface p95),
with live processing of 135.9 and 69.7 seconds. Chest-4 retained 162 final views
but failed both surface checks at approximately 8.48 mm p95 and only 88% agreement
within 5 mm. The faster 82.5-second chest-4 preview is therefore rejected. Simply
retaining all final views, using CPU tracking or increasing feature strength does
not establish equivalent recovered geometry.

A regression check also exposed cached normal orientation becoming stale when
an experimental proposal prepared raw normals after creating its cached voxel
levels. Both CPU levels and GPU uploads now invalidate when raw normal presence
changes, and the full multiscale fallback prepares raw target normals first.
The test failed before the fix and passes afterwards. This fixes derivative
state consistency; it does not prove that the bug caused the earlier graph
differences. Results taken before this fix retain their source fingerprints.

**GPU normal preparation:** generating normals in float64 on CUDA while keeping
CPU voxel positions offered only modest, inconsistent sampled savings. It was
not promoted. Feature-seeded finest-level ICP is measured separately below;
it retains the original multiscale fallback and still requires full-session
geometry checks.

Scripts and full diagnostics for these experiments are retained. No faster
experimental pose is silently promoted to the default pipeline.

## Where substantially higher frame rates can come from

There are two different workloads: reconstructing selected views separated by
seconds, and tracking a moving camera through many small successive motions.
These archives test the first. A continuous GPU tracker needs the intermediate
RGB/depth pairs to assess drift, occlusion and loss of tracking honestly.

A promising architectural route is a fast GPU projective tracker against a
raycast local map, paired with a slower independent keyframe verifier and global
repair. Tracking and raw capture can proceed independently of mesh preparation;
only verified poses authorize final fusion. Local submaps bound map size and
make corrections possible without continually rebuilding the full model.
The raycasting prototype establishes compute headroom, but its measured drift
means that verification, loss detection and repair are essential work.

The next prototype corrects the GPU camera pose using recent independent raw
keyframes, then rebuilds a bounded local preview map from those checked poses.
Keeping a map that contains observations placed with earlier biased poses lets
the tracker drift back toward them. Rebuilding the local map reduced that effect:

| Noisy synthetic moving input | Prepared-view processing rate | p95 processing | Translation RMS / maximum | CPU anchors accepted |
| --- | ---: | ---: | ---: | ---: |
| GPU hybrid map without corrections | 83.4 views/s | 14.3 ms | 9.59 / 15.10 mm | None |
| Checked camera poses every 12 views, retaining the original map | 54.5 views/s | 75.9 ms | 8.61 / 14.16 mm | 5/5 |
| Rebuild local map every 12 views, up to three checked references | 55.3 views/s | 79.6 ms | 4.57 / 9.96 mm | 5/5 |
| Rebuild local map every six views, up to three checked references | 43.0 views/s | 87.4 ms | 3.89 / 8.26 mm | 10/10 |

All four listed runs processed 60 views. Truth enters only scoring after each
decision; it never initializes a tracker or authorizes a keyframe. Measured
feature proposals still need the existing independent held-out depth/visual
checks, with the original ICP fallback. The older six-view correction run using
one reference failed its last anchor; searching recent checked references
recovered that case in the local-map experiment.

The local-map repeat also processed all 60 views without a failed anchor.
Six-view corrections measured 44.6 prepared views/s with 3.89 mm translation
RMS and 80.6 ms p95 processing; twelve-view corrections measured 56.6 prepared
views/s with 4.58 mm RMS and 77.6 ms p95. Both repeats retained every attempted
independent anchor (10/10 and 5/5). These repeat ranges describe this synthetic
sequence and remain separate from the archive CPU/CUDA comparison.

These are warmed synthetic processing measurements, using uniform preview-only
fusion, and exclude camera preparation, USB/network and final reconstruction.
The input generator has 2 mm depth noise and 3% missing pixels. Synthetic metric
VGA preparation averaged 1.77 ms separately; it omits native Kinect color/depth
projection. The archived native-camera replays spend roughly 22 ms per view in
input preparation, already included in their live times. Neither 43 nor 55
prepared views/s establishes a real-camera scanning rate. CPU corrections also
cause the visible processing spikes; moving verification to a bounded background
worker and preparing the GPU preview input on-device remain architectural work.

The client already has continuous visual tracking and full camera-stream
recording options. A new recording with both enabled can measure that route;
the two existing ZIPs have neither intermediate images nor visual tracking
metadata. Merely replacing TSDF integration with another CUDA mapper would
leave the current tracking bottleneck in place.

For the next continuous experiment, select **Motion detail · 640 × 480, 30 fps**,
then enable **Color-assisted tracking**, **Live
fused point cloud feedback**, and **Record all camera frames (large files)**
before starting a new scan. Record a short pass with translation, a wide turn,
temporary occlusion and a return to the starting view. The existing recorder
keeps raw intermediate frames and reports completeness; missing frames should
not be replaced by invented motion. Replay can use `--sensor-streams
--recompute-motion` to score the intermediate path. See
[continuous tracking](CONTINUOUS_VISUAL_TRACKING.md) and the
[recording controls](../README.md).

The next implementation would keep a local GPU map, verify selected keyframes
independently, and treat preview poses separately from verified final fusion.
Submaps would make loop corrections local rather than repeatedly rebuilding a
growing global model. A continuous recording is the missing evidence for that
architecture; none of the current results establishes its real-camera drift
or a validated 30 FPS scanning rate.

For the sparse-view workload, measured feature identities also provide a
possible cheaper pose-refinement route than repeated anonymous nearest-neighbour
ICP. That experiment must compare both pose error and reconstructed surfaces,
including wide turns, before it can replace the current refinement.

An additional CPU Finish experiment rebuilt real raw fragment inputs and tested
eight fixed candidate pairs, four verified and four rejected. The original
bridge verifier retained every competing proposal, independent witness and
acceptance gate. One worker with 20 Open3D threads took 71.34 s; two workers with
10 threads each took 67.24 s; four workers with five threads each took 71.80 s.
These include process imports and the 209 MB fixture loading. Decisions and
witness sets agreed, with numerical pose differences below 0.001 mm. Static
pair distribution produced little benefit because expensive rejected candidates
were unevenly assigned. This is a component experiment using measured Finish
fragment-local poses to reproduce inputs; it does not initialize live tracking
or establish a faster full reconstruction. The graph frontier depends on earlier
verified links, so parallel results must be consumed in its original order.

Dynamic distribution improved that fixed workload: serial verification took
72.06 s, two ten-thread workers took 58.36 s (1.24×), and two twenty-thread
workers took 59.88 s. Parallelizing only the competing proposals in the current
pair preserved the fixture's pair and proposal order and took 65.01 s with two ten-thread workers or
60.04 s with two twenty-thread workers. Every acceptance, ambiguity decision
and witness set agreed. The latter uses 40 Open3D threads on this 20-thread CPU
for a 16.7% component time saving; worker resident memory totals about 1.4 GB.
These variants remain research tools pending full Finish scaling and surface
validation; they are not selected by the public launcher.

Capture timestamps suggest a different way to reduce waiting: the two scans
span 563.92 and 523.06 s, leaving roughly 485.71 and 448.81 s after subtracting
the fast profile's live compute, compared with 304.19 and 239.98 s of Finish.
That is potential overlap headroom, not a measured reduction in Finish time.
Online fragments are provisional because the final median capture gap changes
reference eligibility, and recent fragments can gain more owned views. Safe
background preparation must key immutable image geometry by input/calibration
hashes; cached bridge evidence also needs exact owned/context view IDs, local
poses, seeds, proposals, gate settings and source versions. Finish must retain
its original ranking/frontier and optimized-graph validation. A streaming replay
and bounded scheduler are needed before claiming this latency improvement.

Native-camera CUDA preprocessing has now passed exact RGB/depth comparison on
all 290 archived views, together with clipping, ROI, sentinel, layout and
interpolation-boundary cases. Calibrated projection reproduces the installed
BLAS and OpenCV/Intel IPP accumulation and rounding. CPU output preparation
averaged 21.68–21.74 ms; CUDA preparation returning CPU arrays averaged
5.34–5.64 ms. Keeping output on the GPU averaged 4.87–4.92 ms, including raw
uploads and synchronization, versus 22.48–22.61 ms for CPU preparation followed
by upload. First compilation and calibration setup are recorded separately.
These are input-stage measurements, not scanning FPS.

The original separate GPU confidence experiment failed: four pixels changed the original
zero-weight mask, and the maximum weight difference reached about 0.0625 at
the angular gate. It remains a rejected research implementation. The opt-in
server input path accelerates RGB/depth and keeps the original CPU confidence
calculation for final weighted fusion unless the separate confidence option is
selected. Compatibility probes and full-session mesh checks protect the paths.

A later exact-confidence implementation fixes two independent numerical causes.
It reproduces OpenCV's filter reduction order, including explicit fused
multiply/add where the CPU uses it, and uses multiplication for squares instead
of GPU `powf`. All 290 real views and 11 boundary/invalid-input cases now match
every confidence float32 bit and every zero mask, for both separate and fused
filter variants. On the original failure views, the corrected reductions remove
the mask flips; corrected squares remove another 33,207/43,088 differing bits.
The original rejected report is retained.

The all-view exact component matrix measured CPU RGB/depth-plus-confidence
preparation at 48.2–49.1 ms, GPU host output at 13.7–14.0 ms and GPU-resident
output at 12.3–12.4 ms, including transfers and synchronization. Fusing the
filter passes offered little additional benefit. These component measurements
do not establish the complete scanner gain; that requires the production
confidence mode's separate matched Finish comparison.

The production confidence helper independently matched every float32 bit on all
290 archive views and six additional native-input cases under the final runtime
fingerprint. Including uploads and host outputs, its paired warmed component
times were 26.31/26.17 ms on CPU and 9.46/9.50 ms on CUDA. Full-size first-input
and boundary probes check the installed OpenCV/NumPy reduction behavior before
this implementation can process a fusion frame; unsupported inputs retain CPU
processing in `auto` mode.

The final confidence comparison uses source `9331dee7…`, exact CUDA RGB/depth
input in both modes, and two complete repetitions per mode and archive. The
second repetition reverses the mode order. Every repetition retained the same
121/162 final indices as the CPU reference and passed its surface gate:

| Archive | Confidence processor | Live median (range) | Finish median (range) | Complete median (range) | Typical / p95 view |
| --- | --- | ---: | ---: | ---: | ---: |
| chest-3 | CPU | 70.12 s (70.09–70.14) | 301.44 s (295.40–307.48) | 371.56 s (365.55–377.57) | 203 / 2019 ms |
| chest-3 | CUDA | 68.45 s (67.75–69.15) | 317.08 s (316.35–317.80) | 385.53 s (384.09–386.96) | 172 / 1996 ms |
| chest-4 | CPU | 64.99 s (64.87–65.11) | 230.59 s (229.58–231.59) | 295.58 s (294.46–296.71) | 172 / 1720 ms |
| chest-4 | CUDA | 61.92 s (61.71–62.13) | 225.18 s (219.94–230.42) | 287.10 s (281.65–292.55) | 156 / 1659 ms |

CUDA confidence reduced median live processing by 2.4%/4.7%. Complete processing
was 3.8% slower for chest-3 and 2.9% faster for chest-4. These two repetitions
give a range, not a statistical confidence interval or a universal speedup.
The implementation remains a separate default-off option.

The first chest-3 pair locates its extra Finish time in fragment reconnection:
25.58 s more graph work, while final reintegration saved 4.56 s. Both modes made
341 confidence calls; candidates/tested pairs were 60/40 in both. The CPU mode
had one extra bridge (5→8) that was pruned and never authorized final fusion;
retained bridge identities, witnesses, loop closures and final indices agreed.
Final matrix coefficients differed by at most 2.08e-10. Chest-4 used 468 equal
confidence calls and 98/23 equal candidate/tested counts. Graph processing
occupied 96–98% of these Finish totals. The saved reports do not expose internal
proposal branches or native ICP iteration counts, so CPU runtime variance and
different inner verification paths cannot be separated from these records.

An alternative bounded CPU confidence cache also passed all bit/zero-mask
comparisons on the measured live/reconnect/Final view sequences. Including
depth copies, SHA hashing and LRU work, chest-3 confidence work fell from 6.369 s
to 0.716 s with a live-warmed cache; chest-4 fell from 8.460 s to 0.600 s.
The retained cache used 144.18/189.89 MiB under a 256 MiB limit and reset cleanly.
Those savings are only about 1.9%/3.3% of the earlier complete Finish timings.
This remains a research component, without a cached full-reconstruction gate;
exact CUDA confidence can reduce the repetition cost with less session state.

## Additional exact GPU input preparation

Four complete replays on release fingerprint `67d115...` compared the same
CUDA adaptive 10 mm preview / 5 mm Finish with native GPU input preparation
disabled and enabled. CPU confidence, all raw views, calibration, pose gates,
settings, thread resources and driver stayed fixed. Startup compatibility probes
and setup are included. These are single matched replays, with ordinary Finish
timing variability; they are separate from the earlier AE-source CPU/CUDA matrix.

| Archive | GPU input | Live | Finish | Complete | Final views | Surface p95 | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| chest-3 | off | 78.94 s | 336.24 s | 415.18 s | 121/126 | 0.184 mm | PASS |
| chest-3 | on | 71.37 s | 317.00 s | 388.37 s | 121/126 | 0.187 mm | PASS |
| chest-4 | off | 75.21 s | 242.98 s | 318.19 s | 162/164 | 0.022 mm | PASS |
| chest-4 | on | 65.19 s | 234.27 s | 299.47 s | 162/164 | 0.022 mm | PASS |

GPU input reduced live processing by 9.6%/13.3% and complete processing by
6.5%/5.9%. Chest-3 exercised 870 GPU preparation calls, with no CPU fallback;
both archives passed the same final-index, p95 and 99% surface agreement gates.
Chest-3's paired runs both tested 40 graph pairs from 60 candidates. The faster
input stage is useful, while geometric verification and graph repair remain
the dominant costs. Its isolated roughly 4× speedup is not a 4× scanner speedup.

Reproduce this separate comparison with:

```powershell
python scripts/profile_cuda_pipeline.py @sessions --runs legacy-adaptive-preview-10mm legacy-adaptive-preview-10mm-gpu-input --finish --quality-reference-directory benchmark-output/cuda-study --output benchmark-output/cuda-pipeline/gpu-input-integration/experiments.json
```

The results generator requires immutable input/source/settings, matching
selections and seeds, successful finished meshes, finite surface metrics with
the declared 5 mm/30,000-sample gate, and matching thread resources for the input
comparison. It checks the executed CUDA backend/probes/counters as well as the
requested option. Missing quality cannot be published as a validated gain.

## Controls and reproduction

All options appear in reconstruction backend diagnostics. Explicit unavailable
CUDA requests fail; automatic CUDA descriptor matching can fall back during
setup and records the reason. Runtime update errors are not retried against a
possibly partially updated fusion volume.

A fused update can fail after an earlier chunk has already changed voxel
weights. That failure now quarantines the live volume instead of treating it as
an ordinary rejected frame. Processing, Build and Preview stop with an
actionable error; raw capture and Save Session remain available. Save the session,
then reset and replay it. Synchronization cleanup preserves this error even if
CUDA synchronization also fails. A failed fresh Finish candidate is discarded
without quarantining the original live volume. Fault-injection tests cover
both API behavior and a real CUDA chunk that writes before a later failure.
An error in model preparation after successful live fusion also quarantines
the volume, including the first reference frame and lazy preview refresh;
it cannot leave a fused image recorded as an ordinary rejected capture.

A hard CUDA input error before fusion instead pauses input processing at the
failed raw view. It does not claim that the existing volume was modified.
Status and HTTP 409 errors explain how to save, reset and replay; recording and
session export remain available. Both failure paths prevent automatic worker
restart from silently consuming the failed view.

Native GPU RGB/depth preparation is optional and defaults to `off`. `auto`
checks calibration, shapes, dtypes and CPU-library compatibility, then compares
the first real prepared image against the CPU implementation. Compatible inputs
use CUDA; unsupported inputs fall back before fusion and expose the reason.
`on` requires this path and fails clearly when it cannot be used. Device and
stream ownership are explicit. Calibration/ROI/filter changes invalidate the
cache. Confidence defaults to CPU unless its separate CUDA option is enabled,
and diagnostics report actual GPU,
CPU and fallback batch counts, probes and setup time. These checks are local
compatibility evidence; they do not establish equivalence for every possible
camera input or software version.

| Environment option | Values | Purpose |
| --- | --- | --- |
| `KINECT_CUDA_FUSION` | `auto`, `tensor`, `fused` | Confidence-weighted volume update |
| `KINECT_CUDA_INPUT` | `off`, `auto`, `on` | Optional exact native RGB/depth preparation; confidence controlled separately |
| `KINECT_CUDA_CONFIDENCE` | `off`, `auto`, `on` | Separate exact CUDA confidence calculation with CPU compatibility probes; default `off` |
| `KINECT_TRACKING` | `auto`, `legacy`, `tensor` | Original CPU or CUDA tensor live tracking |
| `KINECT_CUDA_REGISTRATION` | `auto`, `cpu`, `tensor` | Full-cloud geometric verification |
| `KINECT_CUDA_MATCHING` | `auto`, `cpu`, `cuda` | Bounded visual descriptor retrieval |
| `KINECT_KEYFRAME_CACHE` | `on`, `off` | Immutable keyframe registration levels |
| `KINECT_MODEL_REFRESH` | `eager`, `lazy` | Prepare model pyramids before they are needed |
| `KINECT_VISUAL_FEATURES` | `orb`, `sift`, `adaptive` | Original ORB, SIFT, or verified ORB followed by SIFT fallback |
| `KINECT_ADAPTIVE_EXPERIMENTAL` | `off`, `on` | Allow adaptive research outside validated 10 mm live / 5 mm final; launcher forces `off` |
| `KINECT_LIVE_RECOVERY` | `full`, `deferred` | Immediate recovery or retain for Finish |
| `KINECT_CUDA_ODOMETRY` | `off`, `hybrid`, `point-to-plane` | Experimental projective refinement; keep `off` |
| `KINECT_VISUAL_REFINEMENT` | `icp`, `measured`, `measured-fine` | Visual proposal solver, with independent pose gates |
| `KINECT_FINAL_VISUAL_FIRST` | `off`, `on` | Try existing visual/depth bridge evidence before full ICP |
| `KINECT_FINAL_LOCAL_REFINEMENT` | `icp`, `measured` | Local fragment proposal solver |

Use the existing CUDA-enabled server environment, CUDA 12 toolkit and optional
CuPy dependency in `requirements-cuda-fusion.txt`:

```powershell
$env:OMP_NUM_THREADS = '8'
$env:KINECT_NATIVE = 'on'
$sessions = @('export/chest-3-scan-session_20261006_215208.zip',
              'export/chest-4-scan-session_20261007_173028.zip')
python scripts/profile_cuda_pipeline.py @sessions --runs baseline cached sift sift-deferred --repeats 2 --output benchmark-output/cuda-pipeline/controlled-live/experiments.json
python scripts/profile_cuda_pipeline.py @sessions --runs sift sift-deferred --finish --output benchmark-output/cuda-pipeline/finish/experiments.json
python scripts/benchmarks/benchmark_cuda_matching.py @sessions --method sift --output benchmark-output/cuda-pipeline/sift-matching.json
python scripts/benchmarks/benchmark_registration_options.py @sessions --output benchmark-output/cuda-pipeline/registration-options.json
python scripts/benchmarks/benchmark_visual_refinement.py @sessions --output benchmark-output/cuda-pipeline/visual-refinement.json
python scripts/benchmarks/benchmark_registration_options.py @sessions --include-nearest --include-threads --methods cpu_icp cuda_reranked_icp cpu_threads_8 cpu_threads_4 cpu_threads_1 --repeats 3 --output benchmark-output/cuda-pipeline/reranked-registration.json
python scripts/benchmarks/benchmark_opencv_threads.py @sessions --output benchmark-output/cuda-pipeline/opencv-threads.json
python scripts/summarize_cuda_pipeline.py
```

The final matched CUDA recipes can be reproduced with:

```powershell
python scripts/profile_cuda_pipeline.py @sessions --runs baseline legacy-cached legacy-preview-10mm --finish --quality-reference-directory benchmark-output/cuda-study --output benchmark-output/cuda-pipeline/legacy-repeat/experiments.json
python scripts/profile_cuda_pipeline.py @sessions --runs legacy-cached --finish --quality-reference-directory benchmark-output/cuda-study --output benchmark-output/cuda-pipeline/production-control/experiments.json
python scripts/profile_cuda_pipeline.py @sessions --runs legacy-adaptive-preview-10mm --finish --quality-reference-directory benchmark-output/cuda-study --output benchmark-output/cuda-pipeline/adaptive-repeat/experiments.json
python scripts/profile_cuda_pipeline.py @sessions --runs cpu-baseline cpu-cached cpu-adaptive-preview-10mm --finish --quality-reference-directory benchmark-output/cuda-study --output benchmark-output/cuda-pipeline/cpu-control/experiments.json
python scripts/benchmarks/benchmark_matching_exhaustive.py @sessions --output benchmark-output/cuda-pipeline/exhaustive-matching.json
python scripts/research/benchmark_anchored_gpu_model.py --frames 60 --periods 0 6 12 --anchor-solver measured --references 3 --reset-submaps --output benchmark-output/cuda-pipeline/anchored-model-submaps.json
python scripts/check_cuda_backend.py
./scripts/start_cuda_server.ps1 -Recipe hybrid
python scripts/check_cuda_server.py
```

The final release comparison adds original and adaptive-preview CPU controls,
then repeats the two CUDA preparation recipes in reverse order:

```powershell
python scripts/profile_cuda_pipeline.py @sessions --runs cpu-baseline cpu-adaptive-preview-10mm --finish --quality-reference-directory benchmark-output/cuda-study --output benchmark-output/cuda-pipeline/current-release-control/experiments.json
python scripts/profile_cuda_pipeline.py @sessions --runs legacy-adaptive-preview-10mm-gpu-input legacy-adaptive-preview-10mm-gpu-input-confidence --repeats 2 --finish --quality-reference-directory benchmark-output/cuda-study --output benchmark-output/cuda-pipeline/gpu-confidence-integration/experiments.json
```

New quality results record the exact candidate and reference report hashes.
Replay removes old quality sidecars before replacing reports; comparisons abort
if either report changes while they are being scored. The current release
summary requires complete, distinct repetitions, matching source/driver/thread
resources, and quality results bound to the measured report bytes.

The launcher sets every recipe option explicitly and starts the server hidden;
stdout/stderr and the process ID are saved under `logs/`. Its default is the
hybrid recipe. Plain `python -m scanner_server` retains conservative feature
matching/cache/model-preparation defaults. The HTTP check uploads six synthetic
views, exercises confidence fusion and color recovery, builds/exports a PLY,
then restores the initial scan settings and empty state.

Both checks passed against the frozen release through a dedicated loopback
server on 8 October. Six synthetic metric-depth views produced a 2,638,474-byte
PLY using CUDA matching, exact CUDA confidence and fused weighted fusion;
native RGB/depth preparation correctly used its documented CPU fallback for
that unsupported metric stream. Three original calibrated raw views from
chest-3 produced a 2,023,247-byte PLY with CUDA input preparation and confidence,
positive compatibility probes and zero CPU/fallback preparation or confidence
calls. Both restored the exact initial settings and an idle, healthy, empty
session. These are application integration checks; subset mesh success does
not establish scan accuracy or frame rate. The helpers refuse occupied or
faulted servers before writing and refuse cleanup of a session replaced by
another client. The launcher verifies listener ancestry and process creation
identity rather than accepting an unrelated health responder.

For the measured fast-preview tradeoff, stop the existing server, start
`./scripts/start_cuda_server.ps1 -Recipe adaptive -CudaInput auto`, and select **Live voxel
10 mm** and **Final voxel 5 mm** in the client's advanced scan settings. Keep
**Final memory budget 10,000 blocks** and **Final surface confidence 2** to
match these archives. Keep **Color-assisted tracking**, **Use sensor confidence**,
**Recover lost tracking**, **Refine final camera poses**, and **Reconnect separated
views at Finish** enabled; leave bundle adjustment disabled, as in these archives.
The launcher selects the tracking policy; the client selects fusion
resolution. A 10 mm preview reduces live detail, while Finish still processes
the original recorded views at 5 mm. These timings do not apply to arbitrary
client settings. The default launcher recipe remains the validated 5 mm hybrid.
The adaptive launcher remains ORB-only until the client selects the validated
resolution pair. Full-resolution adaptive is available only through the
explicit research override and is rejected for production by these results.

Use `scripts/compare_session_profiles.py` to compare matched successful finished
profiles, including actual triangle surfaces. Plots require Matplotlib. CUDA
workers synchronize before measured results and close their artifacts before
the Windows profiling-only DLL teardown workaround; the server does not use it.

## Where unchanged bridge verification spends time

The standalone [verification profiler](../scripts/benchmarks/profile_fragment_verification.py)
replayed eight accepted/rejected Chest 3 fragment pairs from the original-raw
fixture, with every recorded competing proposal in its original order. The
fixture's measured fragment-local poses reproduce this component only; they
are not archived live scanning seeds. Production source stayed frozen at
`9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a`.
Open3D and OpenCV each used 20 threads, with `OMP_NUM_THREADS=8` and CPU
registration. All decisions, proposal counts, independent witness sets and
ordering matched the original serial reference. Maximum transform translation
change was `5.20e-16 m`, rotation change was zero, and information-matrix change
was `7.28e-12`. Runtime, component, fixture, reference, script and input/cloud
hashes passed their before/after invariance checks.

Fixed verification took **69.524 seconds**, excluding the separate proposal
and target-index diagnostics. These disjoint self times account for its
69.524 seconds of instrumented work:

| Original work | Calls | Self time |
| --- | ---: | ---: |
| Native ICP | 4,116 | 60.777 s |
| Native registration evaluation | 1,868 | 6.383 s |
| Descriptor cache lookup or matching | 1,804 | 1.425 s |
| Native information matrices | 20 | 0.060 s |
| Other Python gates, wrappers and control | — | 0.879 s |

Camera-pair ICP consumed **58.600 seconds**; fragment-union ICP consumed
2.177 seconds. Within camera matching, the original partial-overlap fallback
consumed 45.682 seconds of native ICP, including 34.932 seconds forward and
10.750 seconds reciprocal matching. The costly rejected pair `0/3` alone took
26.901 seconds. This points the next exact acceleration experiment toward
repeated camera matching. It does not show that all native ICP time can be
removed by replacing nearest-neighbor retrieval.

The 4,116 native ICP calls reused **76 immutable target clouds**, with one
target used 255 times. Separately constructing the original CPU indexes for
all 116 observed ICP/evaluation/information targets took **0.273 seconds**.
That diagnostic is outside verification; it is not the internal index-build
cost of the native calls and cannot establish a cache speedup. Separate
proposal preparation took **3.955 seconds**: descriptor lookup/matching used
3.203 seconds, 16 original seeded FPFH/RANSAC calls used 0.416 seconds, and 368
native PnP calls used 0.119 seconds. Diagnostic PnP uses a declared RNG seed and
does not reproduce the original global RNG history or replace fixed proposals.

Inclusive branch timers overlap their children and must not be added. Self
timers subtract timed children; their measured partition residual is zero.
Open3D's native calls do not expose actual iteration counts or separate tree
construction, neighbor search, solving and convergence timings here. Recorded
40/30/20 ICP and 12,000 RANSAC limits are configured ceilings. Raw events and
the original profiler snapshot remain immutable under
`benchmark-output/cuda-pipeline/verification-profile/`; the compact
`chest-3-summary.json` is derived by
[summarize_fragment_verification.py](../scripts/research/archive/summarize_fragment_verification.py),
with raw-report, measured-script and postprocessing hashes. These are component
measurements, not complete Finish timing or real-camera throughput.

Technical references: [Open3D CUDA RGB-D odometry](https://www.open3d.org/html/python_api/open3d.t.pipelines.odometry.rgbd_odometry_multi_scale.html),
[Open3D raycasting](https://www.open3d.org/html/tutorial/t_reconstruction_system/ray_casting.html),
[Open3D dense SLAM example](https://github.com/isl-org/Open3D/blob/main/examples/python/t_reconstruction_system/dense_slam.py),
[CuPy RawKernel](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html),
[Open3D 0.20 thread controls](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/pybind/utility/parallel.cpp).

## RTX nearest-neighbour research

The standalone [OptiX experiment](../scripts/research/archive/OPTIX_RESEARCH.md) uses RTX AABB
traversal to retrieve candidates, then original FP64 coordinates and distance
arithmetic to choose neighbours. The original CPU Huber point-to-plane
estimator, pose updates, radius stages, convergence and independent bridge
gates remain. Float32 AABBs are outward enclosures; tied or near-boundary
results and unsupported inputs use the original CPU tree. Production source
stayed frozen at `9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a`.

Public [NVIDIA optix-dev headers](https://github.com/NVIDIA/optix-dev) enabled the
standalone Windows prototype without an account download. The
[v9.0.0 release](https://github.com/NVIDIA/optix-dev/releases/tag/v9.0.0) requires
an R570 or newer driver; the machine has RTX 3080 Ti, driver 610.88 and toolkit
12.4. NVIDIA's [SDK license](https://github.com/NVIDIA/optix-dev/blob/v9.0.0/LICENSE.txt)
is retained with the fixed-tag local dependency. This is not a public scanner
setting.

Both completed policies reproduced all nine competing proposals for accepted
pair `[8,12]` and rejected pair `[0,2]`, preserving witness sets, support gates
and ordered pair decisions. Each real audit covered **104,123,989 query rows**;
these include repeated ICP searches rather than unique points. The original
conservative version CPU-shadowed 58,809,404 direct RTX hits with zero index
mismatches and resolved 45,314,585 complete-radius misses with the CPU, finding
no omitted neighbours. The separately proven `direct-miss-research-v1` with
radius fractions `[0.25,0.5,1]` CPU-shadowed every 58,809,403 direct hit and
every 45,314,585 declared miss, with zero changed indices or false misses. One
small-bin uncertainty retained CPU search. Synthetic host and device adapters
also matched original CPU indices and finite squared distances.

The narrower bins reduced candidate visits from **6.229 billion to 909.288
million**, but the complete bridge remained slower. Separate unaudited calls
include target hashing, uploads, newly needed index builds, CPU pose solving
and all original gates; setup and fixture loading are reported separately:

| Research policy and pair | Native CPU | RTX first pass | RTX cached repeat |
| --- | ---: | ---: | ---: |
| Conservative, accepted `[8,12]` | 2.979 s | 30.539 s | Not run |
| Conservative, rejected `[0,2]` | 8.163 s | 142.836 s | Not run |
| Staged direct misses, accepted `[8,12]` | 2.850 s | 6.950 s | 8.205 s |
| Staged direct misses, rejected `[0,2]` | 8.391 s | 23.858 s | 23.506 s |
| Single-launch staged RTX, accepted `[8,12]` | 2.806 s | 5.023 s | 5.782 s |
| Single-launch staged RTX, rejected `[0,2]` | 8.264 s | 16.524 s | 15.904 s |

The staged/direct policy took **30.809/31.711 seconds versus 11.240 seconds
natively**, a 2.74–2.82 times slowdown. Its module/context setup was 0.165 s.
Search and transfer/orchestration used 20.618/22.046 s; other correspondence
array and metric work used about 5.9 s. Original CPU estimator, transformation
and Python-loop residual was 1.3–1.8 s. These timers describe the research
bridge and do not expose the native C++ ICP's internal split. CPU-shadow audit
times are excluded from speed comparisons, and the slower cached accepted
repeat is preserved.

The retained target cache is bounded at 64 clouds and 512 MiB. Staged/direct
retained cache peaked at 94.0 MiB, BVH buffers at 53.3 MiB and process RSS at
936 MiB. Query buffers, temporary build workspace and CuPy allocator pools are
outside the retained cache cap. Original conservative sources, binaries and
raw reports are preserved in `optix-nearest/conservative-v1/manifest.json`.
The canonical [research summary](benchmarks/cuda-pipeline/optix-nearest/research-summary.json)
includes source, component, script, binary, SDK, fixture, raw archive and raw
report hashes; it is generated by
[summarize_optix_research.py](../scripts/research/archive/summarize_optix_research.py).

The separate `staged-v1` version moves all radius-bin traversal inside one
OptiX raygen launch with register payload state. Its fresh synthetic host/device
parity and all-nine-proposal real hit/miss audits passed under new source and
binary hashes: the same 104,123,989 query rows, 58,809,403 fully audited direct
hits, 45,314,585 fully audited misses and zero mismatches/false misses. It used
11,569 launches per real pass, preserving the same 909,287,750 candidate visits.
Separate first/cached full-bridge timings were **21.547/21.686 seconds versus
11.070 seconds natively**, still 1.95–1.96 times slower. RTX search used
14.438/15.247 s, other correspondence work about 2.81 s and original CPU
estimator/transform/loop residual about 1.72–1.79 s. Setup was 0.150 s, retained
cache/BVH peaks remained 94.0/53.3 MiB and process RSS peaked at 934 MiB.
Fusion improves the preceding RTX implementation but establishes no native
speedup. Its raw reports are preserved under `optix-nearest/staged-v1/`.
These fixed fragment-local poses only reproduce component inputs; they never
seed live tracking. No RTX variant has completed a whole-session Finish/fusion
mesh proof, and none is promoted to production or a default.

## GPU-resident FP64 ICP research

The standalone [resident ICP prototype](../scripts/research/archive/research_resident_icp.py)
keeps original points, target normals, transformed queries, neighbour IDs and
Huber point-to-plane equation reductions on the GPU. Each iteration copies 30
FP64 terms and metrics to a separately compiled Eigen LDLT/Euler bridge;
final correspondences return to the original independent CPU pose gates.
The bridge uses the exact Eigen archive pinned by Open3D 0.20.0, with all 366
header/license files verified. It reimplements the solve rather than calling
the original Open3D binary. GPU reduction order and compiler choices change
roundoff, so source resemblance is not a claim of numerical equivalence.

The completed resident hit/miss audit preserved all nine proposals, witnesses,
support and ordered decisions for accepted pair `[8,12]` and rejected pair
`[0,2]`. Its 104,123,989 repeated query rows included 58,809,403 direct hits
and 45,314,585 declared misses; every hit and miss was CPU-shadowed, with zero
changed indices or false misses. One uncertain query retained CPU search.
All 236 resident ICP calls completed without a full-call CPU fallback.
Production source stayed frozen at `9331`; runtime, mathematics, artifacts,
fixture/raw data and GPU invariance checks passed. Both the audit and timing
processes exited cleanly.

Separate unaudited timing retained the same proofs, staged fractions
`[0.25,0.5,1]`, `direct-miss-research-v1`, solve DLL and original CPU gates:

| Fixed bridge work | Contemporary native CPU | Resident first pass | Resident cached repeat |
| --- | ---: | ---: | ---: |
| Accepted `[8,12]` | 2.904 s | 15.679 s | 17.492 s |
| Rejected `[0,2]` | 8.286 s | 55.138 s | 53.788 s |
| Both pairs, all nine proposals | **11.190 s** | **70.817 s** | **71.280 s** |

Resident iteration was **6.33–6.37 times slower** than contemporary native
bridge work. Across its two repeats, the disjoint wall partition was 129.597 s
in nearest-neighbour calls, 8.693 s elsewhere inside resident ICP, and 3.807 s
outside resident ICP, totaling 142.098 s of bridge work. Nearest-neighbour
calls consumed 93.7% of resident ICP wall time. Their inclusive RTX search
timer was 106.180 s. Equation/system copies took 2.538 s and the standalone
CPU solve 0.365 s; GPU equation and transform event times were 2.188 s and
0.927 s. GPU events overlap host enqueue/copy timers and must not be added to
wall totals. These measurements do not expose native C++ ICP's internal split.

Removing iteration transfers did not improve this implementation. The timer
comparison and adapter structure point toward device-side CuPy pending-row
scans, masks and diagnostic synchronization as costs to investigate; this run
does not isolate each of them. Nearest-neighbour orchestration needs attention
before further equation optimization.
It does not establish that every GPU-resident ICP algorithm would be slower.
The [resident research summary](benchmarks/cuda-pipeline/resident-icp/research-summary.json)
binds raw audit/timing reports, script/kernel/solver/build/dependency hashes,
retrieval proofs, original fixture/raw archive and separate timer scopes.
This is a fixed-pair component result; it has no whole-session Finish/fusion
mesh proof and adds no public scanner setting or production default.

## Exact native ICP result-cache observation

The [signature observer](../scripts/research/archive/research_native_icp_signatures.py) ran the
original full eight-pair fixed fixture with native ICP on frozen source `9331`,
OMP 8 and Open3D/OpenCV 20 threads. It always called the original native
function. Each signature included freshly hashed original FP64 point and normal
bytes, the unrounded FP64 initial matrix, radius, exposed Huber/kernel and
convergence settings, thread policy and native binary/runtime identity. All
4,116 native events were covered, with identical before/after input hashes,
unchanged raw ZIP/fixture/reference/producer/binary/source hashes, all original
eight-pair gates passing and process exit zero.

All **4,116 complete signatures were different**. The observed exact result
cache therefore has **zero reusable calls and a zero-second native-time savings
ceiling**, before lookup, hashing or result-copy costs. There were 496 recurring
point/normal cloud pairs and five repeated seed-byte groups, but their complete
combined inputs never repeated. Dropping or rounding initial matrices would
change that contract; target reuse alone does not authorize a result cache.
No cache pilot or production cache was added.

Native ICP consumed 62.571 seconds inside 76.355 seconds of fixed verification.
Per-call observation added 5.145 seconds outside the original native wrapper,
including 2.388 seconds of input capture and 2.621 seconds of after-input/result
hashing. Those capture fields are subsets of observer overhead. Verification
includes that overhead; later report serialization and initial raw hashing have
separate scopes. This is an observation of reuse, not a speed benchmark.

Result fingerprints include transformation, fitness/RMSE and correspondence
bytes. Because no full input repeated, bitwise repeated-result agreement is
unmeasured, not established. Native C++ search, reduction and solve internals
remain opaque; exposed input state does not prove hidden-state determinism or
whole-session equivalence. The
[compact signature summary](benchmarks/cuda-pipeline/native-icp-signatures/research-summary.json)
binds the immutable observation/native reports, exact dependency and raw-input
hashes, CPU/runtime/thread policy, coverage, overhead and this negative result.

## Original-double dyadic uniform-grid retrieval

The separate full-radius grid experiment preserves original double coordinates,
strict CPU radius filtering and CPU tie/boundary fallback. Dyadic cell widths
and reversible scaling bound every possible strict-radius hit to 27 injectively
packed signed cells. Its synthetic host/device and signed-boundary checks passed.
The actual accepted `[8,12]` and rejected `[0,2]` bridge fixtures then passed all
nine original competing proposals, witnesses, held-out gates, poses and
information matrices. Every one of 58,809,404 direct hits and 45,314,585 direct
misses was shadow-checked against the original CPU tree: zero changed IDs or
false misses in 104,123,989 real queries. Raw inputs, fixture arrays, source,
binary/configuration and GPU guards passed. Auditing took 393.919 seconds,
including 342.739 seconds of CPU shadow work; this is not a speed measurement.

A separately proof-authorized run measured **10.907 seconds native CPU versus
17.456 seconds first grid use and 16.832 seconds warm grid reuse**, making this
grid 1.54–1.60 times slower. Each grid pass visited 14,097,651,238 candidates
(135.4 per query on average); warm reuse required no new indexes. Retained grid
arrays peaked at 27,087,708 bytes, with transient/allocator memory reported
separately. The warm disjoint wall breakdown was 8.710 seconds GPU search and
host transfers, 4.187 seconds other correspondence work, 0.170 seconds empty
fallback/audit-loop setup, 0.098 seconds dataset work, 1.550 seconds CPU
estimator/transforms/ICP-loop work combined, and 2.117 seconds outside ICP.
These sum to 16.832 seconds. Nested inclusive timers must not be added again;
native C++ search/solve internals and kernel-only device time are not exposed.

The [compact grid summary](benchmarks/cuda-pipeline/uniform-grid-nearest/summary.json)
binds separate immutable synthetic/audit/timing reports and their measured
source snapshots. Three early failed reports are preserved: duplicate NVRTC
FTZ option, unavailable host math headers, and a redundant synthetic mutation
check on proof-excluded CPU-only targets. The corrected narrow timing driver
omits that synthetic setup only after both unchanged proofs validate, records
its own hash and runs the original nine-proposal loop and every final gate.
It claims no fresh timing-phase synthetic GPU coverage. This result remains
research-only; no whole-Finish, mesh or real-camera speed gain is established.
With its other costs fixed, warm search would need to improve from 8.710 to
about 2.785 seconds (3.13 times) just to match contemporary native bridge time.

## Selective thread counts for small native camera ICP

A separate original-raw fixture pilot tested whether small per-camera clouds
benefit from fewer native TBB threads. Only matches where both camera-training
clouds had at most 10,000 points used the selected limit; union ICP, evaluation,
information and other work retained 20 threads. Accepted `[8,14]` and rejected
`[0,2]` retained all nine proposals, original witness/pose/information gates,
input arrays and source/fixture/reference guards for every policy.

Contemporary original-20 verification took **11.279 seconds**, compared with
**33.659 seconds at one small-cloud thread, 13.915 at four, and 11.571 at eight**.
The selective policies were 2.98, 1.23 and 1.03 times slower. The 72 eligible
matches out of 256 cost 2.392 seconds at 20 threads and 24.708, 5.064 and 3.065
seconds at one, four and eight, including restoration. Their 144 thread switches
cost only 0.0009–0.0015 seconds, so thread-setting overhead does not explain the
regression. Actual native iteration counts remain unexposed. This one-repeat
pilot supports retaining the existing 20-thread setting and does not justify a
larger selective-thread experiment or a production change.
The [compact pilot summary](benchmarks/cuda-pipeline/selective-fragment-threads/research-summary.json)
records exact configuration, original fixture scope, measured-source hashes,
quality gates and limits; it makes no whole-Finish or real-camera claim.

## Staged grids with conservative float screening

The separate staged grid scans quarter, half and full radii, using directed float32 interval bounds to discard candidates that cannot enter that stage's radius. Every remaining candidate retains the original double coordinates, original point ID and double distance calculation. Early acceptance uses the full-radius ambiguity margin; ties, boundaries and unsupported inputs still use the original CPU search. These experiments do not change the released scanner.

The dedicated synthetic proof passed host/device, signed-domain, interval, unsupported-inner-stage, cache eviction, budget and absent-final-table checks. The original nine accepted/rejected bridge proposals then audited all 104,123,989 actual nearest-neighbour rows against the CPU: 58,809,404 hits and 45,314,585 misses, with zero changed IDs or false misses. Original decisions, witnesses, proposal order, source/input arrays and provenance checks passed; the maximum pose coefficient difference was 1.84e-15 and information-matrix difference 7.28e-12.

This reduced candidate visits from 14.098 billion to 3.191 billion and original double distance evaluations to 391.189 million, but increased complete bridge time. Contemporary native CPU took 11.1360 seconds; first and warm staged runs took 23.2338 and 22.1924 seconds. The warm disjoint costs were 11.7622 seconds for GPU search/transfers, 6.2536 for other correspondence work, 1.7798 for CPU estimation/transforms/ICP-loop work, 2.1271 outside ICP, and 0.2697 for dataset and empty fallback/audit bookkeeping. Inclusive timers overlap; the disjoint costs subtract nested timers. Native C++ search/solve internals remain unexposed, and these are component timings, not scanning FPS or full-Finish gains.

The negative result rules out distance-count reduction alone as a sufficient optimization. The staged shader still serially searches all 27 cell keys on thread zero, and host decoding of the per-stage packed diagnostics lies within other correspondence work. A separate lookup ablation will distribute those searches across 27 lanes and measure the host diagnostic cost independently before another complete audit. The measured reports and source snapshots are preserved at `benchmark-output/cuda-pipeline/uniform-grid-nearest/staged-pruned-v1/`; its canonical compact result is `research-summary.json`.

## Bounded real-query grid lookup ablation

The original proved bridge captured 12 batches of 2,048 exact, unrounded real query rows from accepted pair `[8,12]` and rejected pair `[0,2]`, covering camera and fragment-union clouds at the original 0.12, 0.06 and 0.03 metre ICP radii. The complete original nine proposals retained their decisions. Each trace occupies 7.18 MB; each comparison variant retains at most 32 MiB of GPU cache. Fresh original CPU nearest-neighbour checks, focused ties/boundaries/subnormal/unsupported cases, and every-repeat byte equality of all raw output columns passed. Equality includes second minima, candidate counts and packed staged diagnostics, so the parallel lookup and flat iterator comparisons preserve the complete corresponding serial output on these sampled rows.

After preserving the initial source and measurements, a fresh trace was captured under the order-only harness revision. Forward and reverse comparisons use this same exact trace and five measured repeats after one warmup. Both processes exited successfully, closed their source/input/hardware guards and released their caches. The initial and new captures remain separate evidence: some later query bytes differ after the original native ICP iterations, while their decisions pass the original authority checks.

| Kernel | Raw lookup event, forward / reverse (ms) | Lookup and output transfer wall, forward / reverse (ms) |
|---|---:|---:|
| Serial full grid | 10.775 / 10.639 | 16.297 / 15.863 |
| Parallel full grid | 7.585 / 7.557 | 12.429 / 12.700 |
| Serial staged grid | 19.397 / 18.698 | 24.721 / 23.401 |
| Parallel staged grid | 9.572 / 8.736 | 15.571 / 13.335 |
| Flat full grid | 6.658 / 6.259 | 12.071 / 11.027 |

These totals cover the bounded real batches, not the original 104 million query trajectory. The flat iterator leads the aggregate in both orders: 1.62–1.70 times the serial full-grid event speed and 1.35–1.44 times its lookup-plus-transfer wall speed. Its event lead over parallel full-grid lookup is smaller, 1.14–1.21 times, and it wins 10 of 12 batches in each order. Parallel staged lookup improves its serial event time by 2.03–2.14 times, but remains slower than flat full-grid lookup on the aggregate.

CUDA events enclose the raw lookup call, including output allocation and enqueue gaps, and overlap the lookup wall timer. The output transfer follows that timer. Dataset hashing/tree construction/sorting/upload, query upload, replacement compilation and CPU resolution are recorded separately; event and wall times must not be added. The sampled original CPU resolution alone costs roughly 0.116–0.119 seconds per variant across the measured repeats, illustrating why a faster shader does not establish a faster bridge. The prior complete full-grid bridge would need roughly a 3.13-fold search improvement to match its native CPU control; this small-batch ablation supplies no such full-component or whole-Finish conclusion.

The flat shader is the most plausible next kernel candidate from this sample. It still needs fresh complete accepted/rejected hit-and-miss audits and original pose/witness gates before any full bridge or production claim. Core source remains frozen at `9331dee7…`; the initial harness snapshot is `a630cb0a…` and the forward/reverse revision is `d5c946fe…`. The [canonical summary](benchmarks/cuda-pipeline/grid-lookup-ablation-v1/research-summary.json) binds both source snapshots, both closed traces, exact producer/fixture/raw/proof/hardware hashes, actual process exits and per-batch measurements. Original files remain in `grid-lookup-ablation-v1/`; the matched order runs are in `grid-lookup-ablation-order-v2/`.


## Host stage-diagnostic cost on the same queries

A separate collect/skip experiment used the closed original-query trace with 12 batches and 20 alternating repetitions, preserving each staged shader and all seven raw output columns. Every sampled nearest ID matched fresh original CPU search; raw output bits, original arrays, source/runtime/raw bindings and cleanup passed. The actual worker exited zero in 6.378 seconds. CPU missing/tie/support fallback remained active; the skipped per-stage counters are labelled uncollected.

Across 12 sampled calls, serial staged wall time fell from 31.009 to 30.175 milliseconds, and parallel staged from 28.855 to 27.898 milliseconds: approximately 3% savings. The nested decoder itself cost 1.016 and 1.008 milliseconds respectively, about 84 microseconds per sampled call. CPU fallback took approximately 23.3 milliseconds per group and dominates this sampled adapter timing. These batches contain at most 2,048 retained rows, while their original full batches contained 7,846–10,286 rows; the result cannot be extrapolated linearly to a complete bridge. The limited gain does not justify another full parallel-staged audit now. Its separate proof/timing files remain prepared and unexecuted, with no production selection.

The compact result is [host-diagnostics-summary.json](benchmarks/cuda-pipeline/grid-lookup-ablation-order-v2/host-diagnostics-summary.json); raw output, execution metadata, logs and measured helper snapshots are preserved alongside it. A true device flat-grid prototype is the next distinct research option: keep original double queries/target data/results resident, and copy only ambiguous or unsupported rows for original CPU resolution. It still requires fresh device and complete bridge proofs before any performance conclusion.

## Complete flat full-radius host bridge

The sampled flat iterator was tested in a separate complete component experiment.
Its adapter overrides construction only; original cache, double geometry,
CPU ties/boundaries/unsupported resolver, estimator and convergence methods are
inherited unchanged. Fresh flat-specific synthetic host/device, signed-domain,
subnormal, cloud-eviction, reset and one-byte-budget CPU fallback checks passed.
Original serial or staged proof authority cannot authorize this new shader.

The real proof audited all **104,123,989** queries: 58,809,404 CPU-shadowed hits
and 45,314,585 CPU-shadowed misses, with zero changed IDs, false misses or
CPU fallback queries. All five accepted `[8,12]` and four rejected `[0,2]`
proposals retained their order, witnesses and original decisions. Source,
installed runtime, raw/fixture/reference arrays, hardware and cleanup guards
passed. All 14.098 billion original full-radius candidates are still visited;
the flat iterator changes lane utilization, not candidate completeness.

Contemporary native CPU took **11.280 seconds**, while first and warm flat host
bridge runs took **17.515 and 16.486 seconds**, or **1.55 and 1.46 times slower**.
The warm disjoint costs were 8.463 seconds for search/upload/download/output
extraction, 4.071 for other correspondence work, 1.528 for CPU
estimation/transforms/ICP-loop work, 2.157 outside ICP, 0.096 for dataset work
and 0.170 for empty fallback/audit bookkeeping. These sum to 16.486 seconds;
inclusive timers overlap and cannot be added again. The search timer is not
kernel-only GPU time. Both timed runs executed all original quality gates
without CPU shadows after the separate unchanged proofs validated.

The warm non-search floor is 8.022 seconds. With that floor fixed, search would
need to fall below 3.257 seconds merely to match native CPU. This measured
host bridge supplies no whole-archive, Finish, scanning FPS or mesh speedup
and is not promoted. A distinct resident experiment can test whether keeping
queries/results/ICP data on the GPU removes sufficient host work; its new
device classification, ownership and trajectory require fresh proofs.

The retained index peak was **25.83 MiB** under a 256 MiB cap; process RSS peaked
at approximately **983 MiB**. Transient query/output buffers and CuPy pools are
separate from the retained-index cap. The first timed run built 90 indexes;
the warm run built none. The 385.15-second correctness run included 347.41
seconds of CPU shadowing and cannot be used as acceleration evidence.

The [canonical flat summary](benchmarks/cuda-pipeline/uniform-grid-nearest/flat-v1/research-summary.json)
binds the immutable synthetic, audit and timing reports, exact original/new
artifact and installed runtime hashes, proposal/input membership, disjoint
timing partition and limits. Measured source snapshots and an honest worker
execution/cleanup record are preserved in the same folder. Reproduction is in
[FLAT_GRID_RESEARCH.md](../scripts/research/archive/FLAT_GRID_RESEARCH.md); its preparation-time
status remains frozen as part of the measured artifact rather than being edited
after the experiment.

## True-device flat grid with resident ICP

Keeping the original query coordinates, target XYZ and nearest IDs on the GPU
produced a useful **component** improvement. This separate research adapter
inherits the existing resident transform, normal-equation reduction,
convergence/update order and pinned Eigen solve. Its classifier applies the
original tie/boundary policy and copies only rows requiring CPU resolution or
an explicit audit. The released scanner still uses its validated native CPU
registration path.

Fresh synthetic proofs passed 16 actual host/device cases, 43 signed-domain and
subnormal cases, 32 partial-warp/block classifier cases, and all 14 focused
transport, malformed-output, policy, memory-budget and XYZ-lifetime checks.
The new resident trajectory then CPU-audited **104,123,989** queries:
58,809,404 hits and 45,314,585 misses, with zero changed IDs, false misses,
malformed results or CPU fallback. All five competing proposals for accepted
pair `[8,12]` and four for rejected pair `[0,2]` retained their original order,
witnesses and decisions. The maximum pose coefficient difference was
6.94e-16 and information-matrix difference 4.37e-11; all existing gates passed.
The 422.67-second fully shadowed run is correctness evidence only.

| Complete fixed bridge | Wall time (seconds) | Native / resident ratio |
|---|---:|---:|
| Contemporary native CPU | 11.209 | 1.000 |
| Device resident, first | 8.339 | 1.344 |
| Device resident, warm | 7.450 | 1.505 |

Both unaudited runs repeated all nine original proposal checks after distinct
fresh device-resident proof authority validated. They visit the same 14.098
billion candidates as the full grid. Each executes 236 resident ICP calls,
10,861 pose iterations and 11,569 device nearest calls. Neither downloads any
full query packet or invokes CPU nearest fallback; the classifier synchronizes
an 80-byte counter per nearest call, totaling 925,520 bytes per run. Normal,
pose, small equation-system and final-correspondence transfers remain.

Warm host wall splits into 3.016 seconds of inclusive nearest work, 2.225 of
other resident work and 2.210 outside resident ICP, totaling 7.450 seconds.
Nearest raw lookup/classification/counter waiting contributes 2.657 seconds;
its remaining host control contributes 0.359. CUDA event timing is disabled
and explicitly uncollected. Queued GPU work can finish during a later blocking
copy, so these are disjoint **host-wall scopes**, not isolated kernel durations.
Initial solver setup costs 0.036 seconds outside the bridge timer; the first
run builds 90 indexes and the warm run builds none.

Retained sorted grids plus original-order XYZ peak at **32.63 MiB** under the
shared 256 MiB cap. Observed query scratch peaks at 1.90 MiB and combined
resident/query scratch at 2.89 MiB; their separate configured caps are 136 and
256 MiB. Process RSS peaks at approximately 984 MiB. External query storage,
host packets and CuPy pools remain outside those array caps.

This is a two-pair component experiment with two timing repeats and a single
contemporary CPU control. It establishes no complete archive Finish, live
tracking or mesh throughput/quality result. Resident reductions can change
roundoff, and production promotion would require separate complete-session
validation. No such trial or promotion is included in the field-ready release.
The [canonical resident summary](benchmarks/cuda-pipeline/uniform-grid-nearest/device-resident-v1/research-summary.json)
binds closed raw proofs/timings, exact source/runtime/solve/input/hardware
identity, ordered proposals, actual settings, memory and timing limits.
Measured source snapshots and execution logs are preserved alongside it.
[DEVICE_FLAT_GRID_RESEARCH.md](../scripts/research/DEVICE_FLAT_GRID_RESEARCH.md)
contains reproduction commands; its preparation-time status remains frozen
as a measured artifact.
