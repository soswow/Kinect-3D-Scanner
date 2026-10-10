# AprilTag scan performance on 10 October 2026

The chest scan reconstructed all 98 selected captures successfully, but Finish
spent 537.05 seconds on the server. Repeated fitting of a large camera and
ordinary feature graph dominated the wait. The AprilTag pose calculation itself
is already fast. A compact solve using consistent tag identities provides a
promising initializer for subsequent depth refinement and selective recovery.

The measured archive is `chest-with-markers-scan-session_20261010_213124.zip`.
The running server snapshot has the same session ID and selected images. Raw
captures and annotated diagnostic images remain in the ignored `export` folder.
The machine-readable summary is in
[`benchmarks/chest-markers-apriltag-performance.json`](benchmarks/chest-markers-apriltag-performance.json).
Production reconstruction behavior was not changed by this analysis.

## Where Finish spent its time

The server log supplies these phase boundaries. The client also waited about
55 seconds for remaining capture uploads before starting reconstruction; that
transport wait is outside the 537 seconds below.

| Phase | Seconds |
| --- | ---: |
| Prepare depth, RGB features and tag observations | 17.51 |
| AprilTag pair stage | 60.90 |
| Local depth registration | 31.79 |
| Initial graph optimization and rejection | 19.45 |
| Retrieve and check RGB loops | 26.29 |
| Joint refinement, repeated revalidation and interpolation | 330.56 |
| Accumulated map recovery and final geometry audits | 33.61 |
| Storage planning and preview fusion | 8.18 |
| Final fusion and mesh completion | 8.77 |

The long joint phase made **23 refinement calls**. Their logged wall times total
271.35 seconds; the remaining 59.21 seconds covered checks, graph work and
interpolation. The complete build made 13 graph rejection/recheck passes and
recorded 279 rejected links. The saved report retains only the final two joint
refinement reports, so summing their times would miss most of the work.

The final large-component refinement contained 17,378 ordinary feature
landmarks, 78,295 feature observations and 312 intermediate cameras in addition
to 87 selected cameras. The 10-capture component separately expanded to 389
intermediate cameras. This evidence is expensive to assemble and fit repeatedly.

## What the AprilTag stage actually did

Every selected view had usable RGB-D tags: 657 detections, 527 usable tag
observations, 17 distinct IDs, and 2 to 11 usable tags per capture. Reproducing
the same pair bank and acceptance outcomes in an isolated research process gave:

| Work | Seconds |
| --- | ---: |
| Detect tags in all 98 prepared images | 0.52 |
| Fit 2,757 shared-label pair hypotheses | 2.48 |
| Compute general RGB pose seeds that the tag branch discards | 17.95 |
| Independently validate 844 fitted poses against pair depth | 30.13 |
| Validate candidate connections against accumulated views | 11.83 |

The median tag pair fit took **0.77 milliseconds**. These measurements come from
one CPU run with the resident server idle; the instrumented phase timings should
not be substituted for the original build's exact wall time.

In `recover_depth_graph.match`, `motion.seeds(a, b)` runs before dispatching the
AprilTag branch. It computes general RGB feature correspondences and poses even
though that branch uses only `tag_motion`. Tags are already searched first, but
their poses do not initialize the subsequent local registration calls.

Later, `landmark_observations` builds ordinary tracked/SIFT landmarks rather than
AprilTag corner landmarks. Tags remain pairwise edges and a post-solve rejection
test. Consequently the joint fit does not minimize the same tag corner errors
that the revalidation later requires it to satisfy. This is a plausible source
of repeated rejection; the available logs do not partition its causal cost from
other inconsistent depth and RGB measurements.

## Repeated identities in this recording

Original 1280 by 1024 RGB images independently show two physical markers
decoding as ID 8 in capture 1, ID 11 in capture 25, ID 9 in capture 88, and ID 10
in capture 34. IDs 8, 9, 10 and 11 also occupy separated positions under the
saved camera map. The same-image duplicate check correctly excludes copies
visible together, but an individual copy can pass when its duplicate is hidden.

Repeated IDs add contradictory cross-view evidence. They explain why a solve
that simply merges every occurrence of each ID into one landmark fails. They do
not establish how much of the original runtime would disappear after correcting
the printouts: that requires another full reconstruction with distinct labels.
The pipeline's unnecessary work and repeated large solves remain independent
optimization targets.

## Compact tag and depth experiment

The research script redetects tags from raw selected images. It excludes IDs
8, 9, 10 and 11 explicitly, builds a maximum-evidence camera tree from the
remaining tag measurements, and jointly fits camera poses and shared corners.
Saved poses are used only for comparison. Intermediate camera images, ordinary
RGB feature landmarks and archived camera poses do not initialize this solve.

| Proposal | Captures | Shared corner landmarks | Solve wall time | Empty-space conflict |
| --- | ---: | ---: | ---: | ---: |
| Tag tree before joint fitting | 95 | 52 | initializer | 9.72% |
| Joint tag corners | 95 | 52 | 0.20 s | 7.65% |
| Then selected depth surfaces and shared planes | 95 | 52 | 1.70 s | 2.41% |
| Original finished reconstruction | 98 | ordinary RGB-D graph | 537.05 s total Finish | 1.91% |

The compact fit contains 1,540 tag corner observations. The depth refinement
adds 40,436 surface constraints and 12,480 plane observations. Its median camera
translation difference from the saved result is 18.9 mm after gauge alignment;
the 95th percentile is 38.8 mm. This is a comparison with another reconstruction,
not ground-truth accuracy. Captures 72, 73 and 87 lack a connected usable unique
tag pose and still need recovery.

The same all-view depth audit is retained for the research proposals. Its 8%
acceptance threshold is an engineering consistency bound, not a correctness
probability. Tag-plus-depth results of 2.40% and 2.41% were obtained in two runs.
The existing strict 3-pixel tag agreement is still not satisfied by every pair
after depth refinement. Handling noisy corners and preserving whole-marker
consistency must be validated before promoting this experiment to production.

The 1.90 seconds covers the two solves only. The measured run also spent 3.90
seconds decoding selected images, 7.91 seconds preparing views, 0.50 seconds on
tag detection and 1.46 seconds fitting tag pairs. Complete depth audits, missing
view recovery and final mesh fusion are additional work. This is evidence for a
faster initialization strategy, not a demonstrated 1.90-second replacement for
the complete 98-view reconstruction.

## Recommended reconstruction order

Choose the route per capture and connected component. Availability of some tags
does not imply that every camera can be positioned from tags. A useful gate
considers unique shared identities, measured depth and synchronization, corner
extent and pose conditioning, reprojection residuals and agreement across views.
Tag count alone is insufficient. A small reliable set can seed or constrain an
ordinary RGB/depth search even when it cannot determine a pose independently.
Missing, occluded, ambiguous or poorly conditioned tags leave the ordinary
recovery methods available; this must also hold for a completely unmarked scan.

1. Detect and validate distinct tag identities, then construct a compact shared
   corner map. Preserve raw observations and explicitly flag repeated physical
   IDs rather than allowing a hidden copy to become a false identity match.
2. Fit cameras against that map, retaining tag pixel and depth residuals in the
   optimizer. Use consistent tag connections to establish poses before ordinary
   feature extraction, local depth searches or intermediate camera expansion.
3. Refine those poses with measured selected-view depth and perform the final
   geometry audit. A tag constraint should not require large anonymous depth
   overlap to serve as a pose proposal; fusion still requires validation.
4. Retrieve ordinary RGB features and geometry only for uncovered captures,
   disconnected components or proposals that fail measurement checks. Supply
   tag-derived poses as initializers to those methods.
5. Make graph rejection incremental. Reuse validated components, shared
   landmarks and intermediate observation expansion instead of refitting every
   unaffected component after rejecting one link. Record every refinement call
   and phase duration, including tag detection, seeds, fitting and validation.

Removing the unused general seed computation is the smallest independent change
suggested by this profile. The larger gain requires integrating persistent tag
measurements into the joint fit and making the broad recovery pipeline a
conditional fallback. Hard-coding this recording's excluded IDs is not a
general solution.

With measured Kinect depth, shared corners already provide metric 3D points;
relative rigid motion can be fitted directly. Calibrated 2D projections alone
do not supply metric scale. Known marker dimensions or measured depth can supply
that scale, and [OpenCV PnP](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)
supports estimating poses from known 3D points and corresponding image corners.
The current code detects on the prepared 640 by 480 depth grid. Retaining native
RGB corner precision with the calibrated RGB intrinsics and RGB/depth extrinsics
is a later accuracy opportunity after resolving the larger scheduling costs.

## Reproduce the research

```powershell
.\.venv\Scripts\python.exe scripts/research/analyze_apriltag_session.py `
  export/chest-with-markers-scan-session_20261010_213124.zip `
  --output export/chest-markers-analysis/full-pair-analysis.json

.\.venv\Scripts\python.exe scripts/research/analyze_apriltag_session.py `
  export/chest-with-markers-scan-session_20261010_213124.zip `
  --output export/chest-markers-analysis/compact-tag-depth.json `
  --exclude-id 8 9 10 11 --compact-only --depth-refinement
```

These commands read the archive and write research reports. They neither load a
project into the running server nor replace its mesh. No client or server
runtime changes were installed for this analysis.

## Implemented tag priority

The subsequent implementation uses a shared corner map first in depth-mode
Finish. It fits measured tag landmarks, refines with depth surfaces, and extracts
ordinary RGB features on demand for cameras whose tags are missing or weak.
Complete markers rejected by the initial fit do not pull those recovered cameras
away from their RGB-D measurements. Recovery is bounded; insufficient coverage,
disconnected final measurements or failed validation invokes the existing broad
pipeline with tag poses retained as proposals.

Final camera authority requires a connected graph of supported shared markers or
revalidated RGB/depth connections, followed by the unchanged complete raw-depth
audit. The raw corner measurements also participate in the general joint fit
when broad recovery is needed. Repeated family/IDs found in any image are excluded
throughout server processing; native RGB catches duplicates hidden on the depth
grid. No IDs are hard-coded.

Cold CUDA Finish replay of the same 98 selected captures took **51.9 seconds**
and built a mesh from **98/98 cameras**, compared with the recorded server's
**537.05 seconds**. The compact map/recovery/audit took **19.1 seconds**;
17 cameras used measured recovery and 44 views required ordinary RGB features.
The final measurement graph connected all 98 cameras. Sampled free-space
conflicts were **2.42%**, passing the existing 8% bound (saved result: 1.91%).
Earlier complete replays took 49.8 and 51.5 seconds. These are single-session
measurements, not a guarantee for sparse or conflicting tag scenes.

The replay used raw frames and the saved settings, never archived camera poses
as initializers. Timing excludes imports, ZIP/image decoding, hashing and artifact
export, as does the recorded server build's already-loaded input timing. The
summary and provenance are in
[chest-markers-apriltag-priority.json](benchmarks/chest-markers-apriltag-priority.json).
The saved mesh is a comparison reference, not ground truth; its median symmetric
vertex distance from an earlier completed replay was 3.4 mm and p95 was 10 mm.

Live tracking already tried tags before ordinary features and ICP. Successful
tag tracking now defers source normals and model registration levels, while
extracting the current preview. Geometry fallback prepares the pending model
before use. Live measurements on this recording also include the added native
duplicate checks and changed rejection decisions, so they do not isolate the
speed benefit of deferred geometry preparation on a clean unique-tag scan.

Validation exercised 85 tests: 84 passed. The existing ordinary RGB tracker test
`test_continuous_motion_has_metric_accuracy_and_seeds_sparse_fusion` fails its
10 mm RMS assertion at 59.88 mm, identically on the unchanged `762a6da` baseline
and this implementation. Tag-map, missing-tag recovery, native duplicate
quarantine, independent depth rejection, live fallback/preview and transactional
fusion checks passed. Full Finish and live replays used CUDA.

```powershell
.\.venv\Scripts\python.exe scripts/profile_session.py `
  export/chest-with-markers-scan-session_20261010_213124.zip `
  --finish --finish-only --device cuda `
  --output benchmark-output/apriltag-priority.json
```

This replay runs in an isolated process. It does not replace the loaded server
project. The registration algorithm version is now `offline_depth_graph_v3`,
so a saved result from the previous algorithm is recomputed on the next Finish.
