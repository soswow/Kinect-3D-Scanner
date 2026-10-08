# Field CUDA research: chest-5, chest-6 and chest-7

The field option ready to try is `scripts/start_cuda_server.ps1 -Recipe adaptive
-CudaInput auto -LiveRecovery deferred`. On these three raw selected-view
replays, deferred recovery reduced Live processing by 46.5–78.3% and total
Live+Finish processing by 20.4–30.8%. All three retained the same Final views
and passed fixed-coordinate surface comparisons. The Live preview is sparser;
Finish still performs the original recovery and geometric verification. The
launcher defaults to `full`, and the production ICP label remains `legacy`.
Chest-7's 5 mm Final requires a budget of at least 13,302 unique blocks;
the successful experiments used a common 20,000-block budget.

The new field archives make fragment verification the main speed target, but
they also expose two quality/completion issues that a speed comparison must
preserve. Chest-5 retained only the anchored seven-view fragment. Chest-7 spent
225.891 seconds reconnecting fragments and then failed to build the requested
5 mm final volume within its 10,000-block budget. Neither outcome should be
treated as a successful full-scan performance result.

These observations come from the original ZIP manifests, reconstruction
diagnostics, image headers and sensor indexes. They are saved field results,
not matched CPU/CUDA replays or independent geometric ground truth. The
[offline analyzer](../scripts/research/analyze_field_sessions.py) imports only
the Python standard library and never starts a scanner, initializes CUDA,
decodes image pixels or uses archived poses as tracking seeds.

| Saved observation | chest-5 | chest-6 | chest-7 |
|---|---:|---:|---:|
| Selected RGB-D views | 27 | 38 | 115 |
| First-to-last selected capture span, seconds | 98.327 | 90.527 | 339.021 |
| Median selected-view interval, seconds | 3.528 | 2.432 | 3.040 |
| Originally accepted live views | 21 | 33 | 82 |
| Retained poses after Finish | 7 | 38 | 108 |
| Recovered / excluded live views | 1 / 15 | 5 / 0 | 26 / 0 |
| Median server frame processing, milliseconds | 250 | 234 | 313 |
| p95 server frame processing, milliseconds | 1,444 | 1,149 | 1,929 |
| Requested 5 mm final volume applied | Yes | Yes | No: block budget exceeded |
| Recorded final blocks after successful build | 3,328 | 2,860 | Not recorded |
| Mesh geometry stored in ZIP | No | No | No |

Live acceptance is reconstructed from `recovered_offline` and
`excluded_offline`: Finish rewrites frame success and messages. The reconstructed
accepted indices exactly match `original_poses` in all three archives. Retained
counts satisfy live accepted minus excluded plus recovered.

| Recorded Finish scope, seconds | chest-5 | chest-6 | chest-7 |
|---|---:|---:|---:|
| Fragment reconnection | 9.125 | 8.265 | 225.891 |
| Final pose refinement | 0.625 | 6.032 | 4.406 |
| Bundle adjustment | 0.563 | 1.234 | 1.297 |
| Completed final reintegration | 0.562 | 2.109 | Not recorded |
| Sum of these recorded scopes | 10.875 | 17.640 | 231.594 |
| Reconnection share of recorded scopes | 83.9% | 46.9% | 97.5% |

These sums are not total Finish wall times. Mesh extraction, export and failed
final fusion can be absent from the timers. Live stage timers are exclusive,
while frame elapsed and stage sums use different clock/synchronization scopes
and differ slightly. The numbers establish where recorded time goes; they do
not establish camera FPS, CPU utilization or matched speedups.

Chest-5's anchored fragment owns indices 0–6. The remaining fragments own
`[7]`, `[8,9,10]`, `[11..23]` and `[24,25,26]`; all four were disconnected.
No bridge was verified. Three candidate pairs were tested and three further
candidates were unreachable from the anchored graph. Neither the pair nor
fragment budget was exhausted. Finish excluded live indices
`[7,11,12,13,14,15,16,17,18,19,20,21,22,23,24]`, recovered index 5, and committed
`21 - 15 + 1 = 7` poses. This is a connection/coverage result, not evidence that
acceptance should be relaxed.

Chest-6 has three connected fragments, two measured sequential bridges and one
tested graph candidate. All 38 views were retained. Its 25 validated refinement
loops reduced the saved independent-depth objective from 0.0007563 to 0.0005873
square metres. Bundle adjustment was not applied because connected multi-view
tracks were insufficient. The archive has no triangle surface for an external
mesh comparison.

Chest-7 has 14 fragments, 50 graph candidates and 49 tested pairs. Its report
contains 12 verified bridges and two loop closures; nine fragments connected
and five small fragments remained disconnected. It recovered 26 views and
corrected 80 poses. Optimization fell back to revalidated measured bridge poses
after two optimized bridges were rejected. No search budget was exhausted.
Its 10 mm reconnection volume needed 2,468 blocks. That number does not describe
the requested 5 mm volume: the exact final requirement is absent from the ZIP.
The failure establishes only that more than 10,000 blocks were needed, hence a
lower bound of 10,001. A voxel-ratio extrapolation is not an exact allocation
measurement.

The retained camera-path XYZ spans are about `(0.300, 0.089, 0.080)` metres for
chest-5, `(0.496, 0.386, 0.284)` for chest-6 and `(2.101, 1.223, 2.224)` for
chest-7. These are pose diagnostics, not object dimensions. A broader path may
legitimately increase the volume, or may include drift; this evidence cannot
distinguish those explanations.

All three archives use 10 mm Live / 5 mm Final, a 10,000-block limit, calibrated
native 640×480 raw 11-bit depth, 1280×1024 RGB at a reported 10 FPS, 0.5–3 metre
depth bounds and 80 mm truncation. Depth filtering, color recovery, confidence
fusion, pose refinement, bundle adjustment, reconnection and relocalization are
enabled. RGB exposure is manual, shutter control 120 (reported 8,294 µs), gain 4.
The complete calibration is identical across the three sessions; the compact
report preserves its hash and the saved settings hash.

Chest-5 uses final surface confidence 2, matching the recommended field recipe.
Chest-6 and chest-7 use 0.5. A future comparison must either retain each archive's
actual settings or explicitly report a separate controlled-settings experiment.
Surfaces at 0.5 and 2 are not interchangeable quality baselines.

The saved backend reports show CUDA input `auto` selecting CUDA with successful
calibration probes, 152 / 333 / 799 GPU preparation batches, zero CPU or fallback
batches, and identical calibration signatures. Descriptor retrieval and fused
confidence-weighted integration ran on CUDA. Depth confidence ran on CPU because
CUDA confidence was off; confidence weighting itself remained enabled.
Tracking and geometric verification used the original CPU path. No hard CUDA
input/fusion fault was recorded. The archives do not record the capturing
server's source hash, so these observations do not bind an exact software build.

There are no continuous raw camera streams in these ZIPs. All three set
`record_full_camera_streams=false`; sensor RGB/depth indexes have zero rows and
no NPY image members. Selected camera-side visual motion is marked valid for
27 / 38 / 111 views, but its intermediate raw images cannot be recovered.
Comparable consecutive selected views in the same valid visual segment produce
25 / 37 / 102 motion pairs; none exceeds the existing 0.3 metre / 30 degree seed
limits. The current server already treats this motion as a bounded seed and
requires independent alignment verification. It is not replacement pose
authority or sufficient data for a continuous dense-model replay.

All 180 selected views reject accelerometer assistance with
`Image-to-host timing uncertain`. Median mapping uncertainty is about 70 ms;
median absolute RGB/depth separation is 7.5 / 7.6 / 8.3 ms. Chest-5/6 contain
1,531 / 1,397 valid accelerometer read rows, but gravity-axis calibration is
unverified. Chest-7 contains no accelerometer rows and an event stating that
the accelerometer was disabled after read errors or excessive latency. Reducing
the uncertainty gate without validated timing/axis evidence would change
authority rather than demonstrate a speed improvement.

The unchanged native field replays and allocation diagnostics below establish
the next baseline. A separately audited device-resident nearest-query/ICP
variant must use the same inputs and settings.
Chest-7 provides substantial graph workload; chest-5 checks that acceleration
preserves legitimate exclusions, and chest-6 checks a connected scan with
successful refinement. The earlier resident result was only a two-pair component
gain ([1.34× first / 1.50× warm](CUDA_EXPERIMENTS.md#true-device-flat-grid-with-resident-icp));
it does not predict these full-session outcomes.

Reducing fragmentation is a separate promising target. The observed sparse
selection intervals and live recovery failures suggest testing overlap-aware
selection and reference retention with bounded continuous raw recordings.
Those intermediate frames are missing here, so this proposal cannot yet be
validated by replaying these ZIPs. Existing measured-motion hints must remain
independently checked.

Capture also offers possible preparation overlap: chest-7 spans 339 seconds
while its selected live frame processing totals about 76 seconds. That is an
opportunity to profile background preparation of immutable closed fragment
inputs; it is not 263 seconds of proven idle hardware. Future graph frontier,
ranking and acceptance remain authoritative at Finish. Any reusable data must
bind exact raw image, calibration, settings and local-pose hashes and invalidate
on reset or pose changes. This architecture needs separate measurements before
integration.

## Matched raw replay evidence

Fresh raw-selected-view Live plus Finish replays used production source
`9331dee7…`, the original CPU tracking/registration, identical archive settings
and the original thread policy. CUDA enabled its existing input, descriptor
matching and fused integration paths; resident ICP and marker proposals were
not enabled. Each row is one run, not a repeated timing estimate. Processing
times exclude imports, archive decoding, hashing, geometry export and report
writes; they are not acquisition wall time or camera FPS.

| Successful replay | CPU Live / Finish, seconds | CUDA Live / Finish, seconds | CPU / CUDA total, seconds | Less processing time | Retained views |
|---|---:|---:|---:|---:|---:|
| chest-5, original 10,000-block Final cap | 14.895 / 14.045 | 12.465 / 11.636 | 28.939 / 24.101 | 16.72% | 7 / 7 |
| chest-6, original 10,000-block Final cap | 16.919 / 26.810 | 12.471 / 18.559 | 43.729 / 31.030 | 29.04% | 38 / 38 |
| chest-7, common 20,000-block Final cap | 93.154 / 293.090 | 76.480 / 250.730 | 386.244 / 327.210 | 15.28% | 108 / 108 |

All six rows built meshes. Chest-5 still preserves its original disconnected
coverage result: accelerating the seven retained views does not recover the
other fragments. The first chest-7 pair with its original 10,000-block cap
retained 108 views on both paths but failed Final on both; its 393.224 / 335.250
second totals are preserved as failed-completion evidence, not used in this
successful-scan table. The common larger cap is a declared experiment setting.

Chest-7's completed 5 mm final volume contains exactly 13,302 blocks. A separate
original-input allocation diagnostic also planned 13,302 blocks in 0.800 seconds.
This is a measured requirement for that replay and settings, rather than the
earlier ZIP's lower bound or a voxel-ratio estimate. Both successful chest-7
replays used the same 20,000-block allowance.

The CPU reference is a matched reconstruction, not physical ground truth.
Thirty thousand triangle-surface samples per mesh, at a 5 mm threshold in the
fixed input coordinate frame, gave precision/completeness 1.0 for all three
CPU/CUDA pairs. Surface RMSE was 0.0000000935 m for chest-5,
0.0000066101 m for chest-6 and 0.0000028875 m for chest-7; chest-7's p95 was
0.0000001201 m. Mesh vertex/triangle counts need not be identical. No fitting,
trajectory alignment or scale adjustment was used.

Local evidence is preserved in `raw-baselines-v1/execution.json`,
`raw-baselines-20k-v1/` and `raw-baselines-20k-cuda-v1/`, under
`benchmark-output/field-cuda-study/`, with per-run source/input/runtime hashes
and separate surface comparisons. The successful chest-7 CPU child closed with
exit 0 and its complete hashed report; its parent driver then failed while
printing a Unicode arrow to the Windows CP1252 console. That parent execution
remains marked failed, with the exact measured driver snapshot preserved.
The CUDA run used a fresh parent whose sole change replaced that display arrow
with ASCII. The numerical child/profile source, inputs, settings and thread
policy match; the failed parent record has not been relabelled complete.

## Where the next speed experiment applies

A separate instrumented original-CPU-registration chest-7 replay recorded
2,842 original registration calls and 13,715 unchanged gate observations,
retained the same 108 views and built a mesh. Registration accounted for
172.683 of its 253.722 seconds of Finish: 249 local calls took 6.973 seconds,
500 union-bridge calls 24.784 seconds, 2,044 partial-bridge calls 140.470 seconds,
and 49 refinement calls 0.457 seconds. Instrumentation and pipeline settings
make this a workload diagnostic, not a replacement for the paired raw timings.
It targets repeated camera-cloud verification more strongly than local
tracking or proposal generation.

Fresh bulk original-native CPU query auditing is being tested to make complete
GPU trajectory proofs practical. The initial API crosscheck matched every
original scalar nearest ID and squared-distance bit for 5,924 synthetic and
24,576 retained real queries, including ties, duplicate points and strict
radius boundaries. Its small real sample took 0.09195 seconds with scalar
queries and 0.04118 seconds all-in with 65,536-row bulk chunks. The separate
full scalar/bulk dual-shadow proof then passed all 104,123,989 original queries:
58,809,404 hits and 45,314,585 misses, every ID and squared-distance hash exact,
zero domain/fallback/malformed rows, and all nine original ordered proposal
gates preserved. Independent stdlib proof validation passed. New field audits
may use this separately validated CPU auditor while preserving original scalar
ambiguity handling and fresh complete field shadows; this is audit
infrastructure, not a scanning speedup.
See [the bounded bulk experiment](../scripts/research/BULK_LEGACY_NN_AUDIT.md).

Marker identity provides an independent proposal experiment. On the prepared
640×480 views, chest-7 decoded markers in 103 of 115 views; 77 views had at least
40 depth-supported corners. Of 840 pairs with at least 40 identity matches,
460 passed the original PnP and original feature-support checks. The median
detector time was 12.68 ms. Chest-5 and chest-6 each provided only one such
passing pair in this prepared-view diagnostic.

For all 111 eligible nearby chest-7 pairs (selected index gap at most three),
93 marker-seeded proposals passed unchanged reciprocal point-to-plane ICP,
and 22 additionally passed both the original held-out geometry check and a
separate original SIFT witness. These are pair diagnostics, not graph or live
tracking acceptance, improved fragment connectivity, mesh quality or Finish
speed. No archived pose or physical board dimensions authorized these seeds.
Full native-resolution RGB offers additional detected identity matches.
A separate bounded FP64 CUDA corner-association audit completed with every
original `numpy.argmin` identity and squared-distance bit exact for 19,280 actual
corners plus 20 synthetic queries, including duplicate/tied/adjacent-ULP and
maximum-grid cases. It compares all visible original projected depth pixels,
retains measured depth support and rejects reused pixel assignments. Marker
minima use NumPy's first-row tie rule, distinct from the native ICP KDTree's
tie policy.

| Native-RGB corner association component | Actual corner queries | GPU timed upload/allocation/lookup/download, seconds | Original CPU brute shadow, seconds |
|---|---:|---:|---:|
| chest-5 | 1,988 | 0.2039 | 10.1617 |
| chest-6 | 984 | 0.1626 | 5.1785 |
| chest-7 | 16,308 | 1.9385 | 88.8803 |

The chest-7 association component is 45.9× faster than that original brute
reference on those timed scopes. A fresh comparison against an exact, improved
CPU lookup gives the more useful CPU/CUDA comparison below. The CPU tree query
only supplies an upper bound; exact original-double reranking preserves every
original row identity and squared-distance bit, including ties.

| Fresh corner lookup comparison | Improved CPU all-in lookup, seconds | CUDA all-in lookup, seconds | Original brute CPU shadow, seconds | CUDA gain over improved CPU |
|---|---:|---:|---:|---:|
| chest-5 | 1.7473 | 0.2156 | 10.1898 | 8.10× |
| chest-6 | 1.0716 | 0.1604 | 5.1935 | 6.68× |
| chest-7 | 9.0336 | 1.9100 | 88.8170 | 4.73× |

This fresh comparison passed all 1,988/984/16,308 actual corner queries plus
20 synthetic queries against the unchanged CUDA helper's complete original
brute CPU shadow. Efficient CPU IDs and squared-distance bits were exact, with
zero wide-search fallback. Per-view setup, hashing and lookup are included in
the all-in component scopes; detector, geometry support, pair verification and
whole scanning remain outside them. The producer explicitly records that the
CPU helper's direct audit is disabled and that the driver obtains its gold
result through the freshly brute-shadowed CUDA lookup. Raw report:
`benchmark-output/field-cuda-study/native-corner-cpu-cuda-v2/report.json`.

Kernel setup/compilation took a separate 0.606 seconds; CPU shadows, detector,
depth support, pair verification and full scanning are outside the GPU lookup
timer. The diagnostic deliberately runs both paths, so its overall wall time
does not represent an unaudited production speedup. Native-RGB chest-7 decoded
markers in 109 of 115 views; 3,736 pairs had at least 40 matched identities and
1,243 passed original PnP/feature support. This new bounded proposal policy is
still component research: it has not passed full tracking/graph/mesh quality
or demonstrated reduced whole-scan latency. Raw report:
`benchmark-output/field-cuda-study/native-marker-cuda-audit-v4/report.json`.

The successful raw CUDA replays improve typical Live latency more than the
slowest views. The stdlib reporting tool
[`summarize_field_latency.py`](../scripts/research/summarize_field_latency.py)
recomputes R7 linear quantiles from the original Live diagnostic rows.

| Successful raw replay | CPU median / p95, ms | CUDA median / p95, ms |
|---|---:|---:|
| chest-5 | 359 / 1,506 | 250 / 1,375 |
| chest-6 | 344 / 1,254 | 242 / 1,123 |
| chest-7, common 20,000 Final budget | 468 / 2,102 | 312 / 1,947 |

These are one matched selected-view replay per mode, not continuous-camera
FPS or statistical confidence intervals. In the successful chest-7 CUDA run,
tracking consumed 63.649 seconds of the 76.480-second Live processing scope.
Reducing expensive unsuccessful tracking/recovery searches therefore remains
an important target. The separate
[`benchmark_field_policy.py`](../scripts/research/benchmark_field_policy.py)
can measure the existing `full` and `deferred` recovery workflows from raw
inputs; altered Live acceptance or Finish coverage requires separate assessment.

Full resident-ICP field audits passed complete actual-input CPU query and CPU
ICP-result shadows for chest-5 and chest-6. Their original strict
native-versus-resident ordered-history comparisons **failed** and do not
authorize old full-Finish timing. Chest-5 took a different global-proposal
history. Chest-6 kept the ordered calls but changed some intermediate
information-matrix memberships and candidate results. A complete trace
diagnostic found a one-correspondence information count difference and an
intermediate candidate transform difference of about 166 micrometres.
The accepted chest-6 graph/witness memberships and all 38 Final views remained
the same; Final translations differed by at most 2.05e-15 metres. A separate
fixed-coordinate, 30,000-sample triangle comparison passed (p95 1.19e-7 metres,
precision/completeness 1.0). That observable comparison alone is not backend
or whole-pipeline equivalence authority.

A captured chest-5 input diagnostic identified a concrete proposal stability
problem: coarse points differing by at most 3.11e-15 metres produced FPFH
differences up to 29.78, including one normal orientation flip. The new
[`proposal-only canonical experiment`](../scripts/research/CANONICAL_FPFH_RESEARCH.md)
quantizes private coarse proposal points at 1 micrometre and estimates fresh
normals/features. All six paired captured invocations produced identical
points, normals and FPFH bytes across two repeats. Original training,
held-out, ICP and fusion points were unchanged. RANSAC, accepted graph quality
and scanning speed were not exercised by that preprocessing diagnostic.

The separate device LDLT micro-prototype passed 93 synthetic systems and
10,861 observed original ICP systems against a freshly reproduced original
Eigen DLL reference. Maximum actual incremental-transform difference was
2.97e-16; no original intermediate factorization or whole-trajectory bit
equivalence is claimed. A preuploaded one-system GPU launch/synchronization
averaged 53.7 microseconds over 100 repeats, while the original DLL loop took
49.65 milliseconds for all 10,861 systems (about 4.57 microseconds per solve).
These micro-scopes exclude different overheads and do not measure a complete
ICP iteration. They argue against moving only the small solver onto CUDA;
combined device control or independent candidate batches need separate tests.
See [`benchmark_device_ldlt.py`](../scripts/research/benchmark_device_ldlt.py).

The first exact-unique Final allocation experiment also exposed backend growth:
chest-5 planned 3,328 blocks, allocated capacity 3,328, then the original
integration automatically grew it to 6,656. The experiment rejected that growth
and preserved the Live owners and poses. A revised research plan accounts for
the maximum previous unique keys plus the full incoming frustum row count,
matching the pessimistic reserve rule in
[Open3D HashMap activation](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/core/hashmap/HashMap.cpp).
Physical capacity, configured unique-block limit and attribute bytes are separate
quantities. The revised chest-5 reserve-headroom trial completed with capacity
4,619 before and after fusion, 3,328 unique blocks, identical unrounded Final
poses and a passing fixed-coordinate surface comparison (p95 6.01e-8 metres,
precision/completeness 1.0). Its measured attribute allocation was 360.86 MiB,
against the original 781.25 MiB. This is the TSDF/weight/colour allocation floor,
not total process or GPU memory. Fresh missing-key activation subsequently
passed all three weighted CPU/tensor/fused backends with identical key-mapped
TSDF, weight and colour bits. Whole-Final trials kept actual capacity at 3,328
for chest-5 (260 MiB), and 13,302 for chest-7 (1,039.22 MiB), against native
capacities of 10,000 and 20,000 respectively. Both retained the same Final views
and passed independent fixed-coordinate surface comparisons. Chest-5's
unrounded Final poses were identical; chest-7's strict pose-bit comparison
failed and remains recorded, while its separate bounded comparison passed
(maximum translation difference 5.08e-15 m). These are attribute allocations,
not peak process/GPU memory or demonstrated Finish-time improvements.

Appending native marker proposals did not help the tested whole Finish workflow.
With the same physical Live checkpoint and original CPU verification, chest-5
took 13.511 seconds originally and 25.175 seconds with five additional marker
seeds; chest-7 took 279.850 and 433.712 seconds with 75 additional seeds. Both
kept the same Final view counts (7 and 108) and passed separate fixed-coordinate
surface comparisons. The CUDA mapping trial included complete CPU mapping
shadows inside Finish, so its wall time is not an unaudited production speed
measurement. Nevertheless, the proposal policy increased ICP calls from 163
to 176 and from 2,720 to 4,266 without extending coverage. It is not enabled.
Exact marker lookup remains useful as a component; adding more candidates needs
a different measured scheduling policy.

The new combined-synchronization component passed all 104,123,989 original
query rows, 11,569 iterations and 236 complete original CPU ICP-result shadows.
Every common nearest ID, squared-distance, strict-radius filter and all 30
normal-equation totals matched their original references bit for bit. Each
iteration used one 320-byte counter/equation host transfer. Focused ambiguous,
boundary and malformed cases exercised the guarded CPU-resolution path.
The fully audited 139.762-second run is validation overhead, not a speed result.
A separately authorized three-round component comparison passed all nine
original gates in every run: median original CPU was 11.418 seconds, existing
resident GPU was 7.679 seconds, and combined synchronization was 9.486 seconds.
The combined path was 23.5% slower than the better resident GPU path. It is not
promoted. A stdlib diagnostic found repeated full-source AST validation costs
about 1.864 seconds for its 236 checks, versus 0.020 seconds for a full-file
digest check; a separate causal component experiment is needed to determine
how much this changes actual registration wall time. That fresh causal
comparison passed all nine original gates in each of three rounds per mode.
Median combined time fell from 9.515 to 7.173 seconds, 24.6% less wall time.
Its contemporary existing resident GPU control took 7.615 seconds, so the
remaining gain was 5.8%; original CPU took 11.130 seconds (1.55× component
speedup). Successive runs may include hardware drift. No complete field Finish
or production backend authority is established by these component results.

A separate actual-input field-conformance protocol retains the failed strict
history reports rather than changing their meaning. Its original CPU shadows,
actual consumed optimizer graph, accepted independent witnesses, stage decisions,
view coverage, unrounded Final pose limits and fixed-coordinate mesh criteria
all passed for chest-5. Native Finish took 13.333 seconds; the first authorized
GPU timing took 19.743 seconds, including 6.444 seconds of offline authority
validation charged inside Finish. This complete measured scope did not improve
Finish speed. Removing that recorded setup time arithmetically does not prove
a faster production workflow. See the
[distinct field protocol](../scripts/research/FIELD_FINISH_CONFORMANCE.md).

Fresh session-6 native/audit runs under version 2 also passed actual Final
graph, witness, coverage, pose and mesh criteria. Its timing run stopped before
GPU call 228: all preceding complete registration inputs and transforms matched,
but the next seed differed in its unrounded bytes. The original CPU information
matrices had drifted by around 1e-12, and post-optimizer fragment poses by
3.33e-16. The next refinement seed came from those poses. Current exact-input
authority correctly refused that new seed; a passing surface comparison does
not authorize this failed timing or a production ICP switch.

The existing `deferred` recovery workflow is a promising field option. It keeps
raw unsuccessful views for Finish instead of attempting every Live geometric
fallback. Fresh CUDA replays used all raw views and original settings:

| Selected raw replay | Full Live / Finish, s | Deferred Live / Finish, s | Final views |
|---|---:|---:|---:|
| chest-5 | 12.465 / 11.636 | 5.493 / 11.184 | 7 in both |
| chest-6 | 12.471 / 18.559 | 6.667 / 17.833 | 38 in both |
| chest-7, common 20,000 Final budget | 76.480 / 250.730 | 16.582 / 243.784 | 108 in both |

All three completed meshes passed 30,000-sample fixed-coordinate surface
comparisons with precision/completeness 1.0. Chest-7's surface p95 was 7.43e-6
metres. Chest-6 accepted 31 Live views instead of 33, then recovered the same
38 Final views. Chest-7 accepted 38 Live views instead of 82 and recovered the
same 108 Final views, so its Live preview is substantially sparser.

| Selected raw replay | Full median / p95, ms | Deferred median / p95, ms |
|---|---:|---:|
| chest-5 | 250 / 1,375 | 203 / 347 |
| chest-6 | 242 / 1,123 | 172 / 264 |
| chest-7, common 20,000 Final budget | 312 / 1,947 | 109 / 250 |

These latencies include unsuccessful stored views, which deferred mode rejects
quickly. Among the successful Live views alone, chest-7's median/p95 changed
from 234/693 to 196/411 milliseconds. The total selected-view Live+Finish scope
dropped by 30.8%, 21.0% and 20.4% respectively. These are one replay per workflow;
they do not establish continuous-camera FPS or guarantee other scans retain
coverage. The useful field tradeoff is a responsive, sparser Live preview while
keeping raw views for the original Finish recovery and geometric verification.

Reproduce the metadata inspection from the repository root with base Python:

```powershell
python -s scripts/research/analyze_field_sessions.py `
  F:/Projects/coding/Kinect-3D-Scanner/export/chest-5-scan-session.zip `
  F:/Projects/coding/Kinect-3D-Scanner/export/chest-6-scan-session_20261009_002019.zip `
  F:/Projects/coding/Kinect-3D-Scanner/export/chest-7-scan-session_20261009_003259.zip `
  --output benchmark-output/field-cuda-study/field-session-summary-new.json
```

The saved current report is local under
`benchmark-output/field-cuda-study/field-session-summary-v2.json`; it contains
settings/backend hashes, sensor-index summaries, pose index accounting and
small per-frame timing records, not image pixels or pose arrays. Default input
identity combines exact JSON SHA256 and ZIP member CRC/size inventories.
`--hash-archives` additionally reads complete ZIPs for SHA256 and should run
outside comparative hardware measurements. Existing outputs are never
overwritten. Original ZIPs, private images and derived runtime outputs remain
ignored. The metadata analyzer itself does not change production code or a server.
