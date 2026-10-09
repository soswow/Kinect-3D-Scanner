# Tracking and reconstruction algorithm review

This records the 6–7 October implementation and its historical decisions.
The 10 October depth-only experiment reconsiders camera-count, live-pose, RGB,
motion, fragment-size and anchoring restrictions; see
[experimental depth registration](FRAGMENT_RECONNECTION.md#experimental-depth-only-final-registration)
for the current alternative and its recorded-session qualification. The
historical recommendation below to retain multiple camera witnesses is not a
requirement of that new mode.

Review dates: 6–7 October 2026. Starting code: `e31c03e`. Scope: the active
camera/client pipeline, calibrated RGB-D preparation, live registration, TSDF
fusion, offline fragment registration, pose refinement, mesh extraction, and
texture projection. The old `kinect_scanner/engine.py` is not the GUI's active
reconstruction engine.

The largest recorded improvement is chest-3: 120 of 126 captures now contribute
to a verified reconstruction, compared with 16 under the starting code. The
combined result includes the continuous visual tracking work from the concurrent
review. Chest-2 retains its previous 174 verified views; the oldest recording
still yields only a partial 16-view reconstruction. Resolution and final surface
confidence were held at 5 mm and 2.0.

## Findings and implemented changes

### 1. Artificial fragment boundaries discarded measured continuity

`scanner_server/fragments.py` previously treated its 16-view storage limit like
a camera pause or failed registration. It never tested the raw frame pair at
that boundary. Independently useful local maps then required a global bridge,
including two different camera witnesses on each side, to enter the final scan.
Failing that more demanding search could discard a continuous part of a scan.

The boundary is now tested with exactly the local registration used inside a
fragment: measured feature initialization when available, reciprocal geometric
registration, motion bounds, and separate held-out points. A successful match
creates a sequential graph edge. A pause or failed match still creates a
disconnected fragment. The optimized graph must pass the boundary's held-out
measurements again before fresh fusion. Recorded live poses remain initial
guesses, never evidence authorizing an edge.

Each bounded fragment also keeps at most two preceding measured camera views
as overlap witnesses. Their poses are converted into the new fragment's local
coordinates. They do not become owned captures and are never fused twice. This
lets a short tail fragment participate in a later reconnection without removing
the requirement for multiple genuinely different camera observations. Pauses
and failed tracking do not carry context across the break.

This separates storage boundaries from measurement failures, as is usual for
[fragment reconstruction with sequential odometry](https://www.open3d.org/docs/latest/tutorial/reconstruction_system/register_fragments.html).
It does not remove the fragment-size or memory limits.

For chest-3, captures 16 and 17 have approximately 74%/77% held-out overlap,
10.2/10.1 mm held-out residual, and a 0.57 mm / 0.10 degree reciprocal cycle.
The verified motion is 298 mm / 21.3 degrees. The old artificial split discarded
this measurement without trying it. The movement is close to the configured
300 mm limit; faster capture tracking remains important.

Regression coverage uses noisy raycast geometry with evaluation-only camera
poses. It checks continuous boundaries, real tracking failures, and corruption
of optimized boundary poses. A two-fragment continuous fixture retains all eight
views without global matching and stays within 15 mm of the known positions.

### 2. Global search spent time on components unable to reach the scan

The original search ranked all eligible fragment pairs and could spend its
budget verifying links entirely within a disconnected component. Those links
do not add a surface to the anchored reconstruction.

Search now prioritizes pairs crossing from the connected map to an unconnected
fragment. Each verified bridge expands that frontier. Once all crossing pairs
have been exhausted, a match between two remaining disconnected fragments
cannot create a path to the anchor. Those pairs are skipped and counted in
`unreachable_candidate_pairs`. Remaining pairs within the connected component
can still provide loop constraints. Existing ambiguity checks, competing
proposals, reciprocal tests, held-out thresholds, and the total pair budget
remain in force.

Evaluation order is kept separate from graph authority: measured sequential
edges come first, then the original evidence ranking initializes the graph.
Early completion cannot skip higher-ranked remaining loop candidates. Changing
the order of expensive work must not arbitrarily choose a different spanning
tree or discard the better-ranked constraints.

Tests cover both an unreachable component and a chain that becomes reachable
through a newly connected intermediate fragment. Sequential bridges are not
duplicated as additional global graph edges.

### 3. Depth-only work unnecessarily projected every RGB image

Final block-allocation planning ran the full native RGB registration pipeline
only to throw its color output away. Depth evaluation and calibration evidence
had the same unnecessary work: lens projection, an RGB visibility buffer, and
bilinear sampling of the large image.

`shared.calibration.prepare_metric_depth` now shares disparity conversion,
nearest-neighbour rectification, scale correction, clipping, ROI, and filtering
with the full RGB-D path. Planning and depth-only evaluation use this helper.
The resulting depth is bit-for-bit equal; reconstruction resolution, confidence,
color integration, and clipping are unchanged. Tests cover native and registered
inputs, missing pixels, distortion, scale, ROI, and filter settings.

### 4. Identical visual matching was repeated for each geometric proposal

Prepared camera views now cache exact mutual ORB correspondences, including the
correctly ordered reverse match. Overlap views share the cache because their
measured features are identical; pose-dependent validation is always rerun.
The cache keeps at most 16 target entries per view and checks feature-object
identity, so replacing a view's features cannot reuse stale matches.
Both source and target identities are checked, including when a dataclass copy
retains the original context cache but replaces its features.

This particularly helps the concurrently added visual bridge checks, which can
test the same camera pair under several geometric proposals. Regression tests
compare forward and reverse results to fresh matching, check context reuse and
replacement, and exercise eviction. This optimization changes neither matching
thresholds nor the evidence used by PnP or geometry verification.

## Review of the rest of the pipeline

| Area | Assessment | Consequence / next action |
| --- | --- | --- |
| Capture cadence | Upload/fusion pacing previously also limited the observations available to tracking. Chest-3's median stored interval is 4.65 s despite faster camera acquisition. | Continuous tracking at acquisition time is the highest-impact live improvement. Changes from “Investigate Kinect tracking loss” have been merged and validated together with this review's code. |
| RGB/depth timing | Native stream pairing is bounded and retains timestamps; larger known lag disables visual assistance. Packet timestamps are not measured exposure centers. | Preserve timing evidence and gate visual constraints; exact exposure synchronization cannot be inferred from these recordings. |
| Calibration | Native raw disparity, depth rectification, RGB extrinsics/distortion, and metric conversion are explicitly represented. | Preserve calibration and units. The recorded calibration's metric validation spans roughly 0.8–1.6 m, while these scans admit farther points; accuracy at greater range remains unmeasured. |
| Live geometric alignment | Robust multiscale point-to-plane ICP against the accumulated model is stronger than raw previous-frame alignment alone, but low residual and high overlap are not unique-pose guarantees. | The combined implementation retains measured feature constraints and checks multiple saved camera views. Continuous motion estimates remain seeds rather than fusion authority. |
| Observability | The normal-covariance test detects flat translational ambiguity. It does not test the full six-dimensional pose Jacobian. A sphere can have diverse normals while its rotation remains ambiguous. | A complete observability check needs scaled rotational/translational conditioning and a visual-evidence escape path. Do not simply relax normal-diversity thresholds. |
| Offline local registration | Raw observations are re-estimated rather than trusting the saved trajectory. A single geometric chain can still accumulate drift. | The combined implementation retains visual identities and independent depth checks, including across sequential boundaries. The boundary fix preserves evidence; it does not make dead reckoning drift-free. |
| Fragment bridges | Reciprocal ICP, held-out geometry, multiple camera witnesses, and rejection of competing transforms are valuable protections against repeated surfaces. | Retain these requirements for genuinely disconnected geometry. More accepted views or more triangles alone are not an accuracy measure. |
| Graph refinement | Correction bounds and held-out validation prevent some bad updates. At most 32 selected keyframes are optimized; intermediate corrections are interpolated. | The combined fragment graph can fall back to independently revalidated measured bridges if optimization fails. Long trajectories still need validation beyond selected refinement keyframes. Held-out points used for proposal acceptance are not an independent final accuracy benchmark. |
| TSDF fusion | Poses are committed through fresh fusion; truncation, observed-space behavior, and final memory budgets are explicit. | Preserve the transactional path. Finer voxels increase work and do not recover absent sensor detail. |
| Confidence weights | Range and angle weights are engineering heuristics. Local normals are computed from short-baseline, noisy depth differences. | A controlled noisy-plane probe shows substantial extra weight loss from noisy normals. A calibrated noise model and robust normal estimation need separate surface-error/completeness evaluation before changing production weights. |
| Final extraction | Final weight 2 differs substantially from low-confidence live display. Isolated components are also filtered. | Face count differences can be expected even with identical poses. Report pose exclusions separately from confidence filtering; never silently lower final confidence to inflate coverage. |
| Color / textures | Final textures use measured RGB, depth visibility, viewing angle, and optional bounded exposure correction. | Black RGB placeholders at invalid native color projections are not a distinct validity channel for TSDF color. Propagating explicit color validity would require separate color weights; it must not erase valid depth. |
| Compute backend | This machine exercises the CPU path. CUDA availability and stage placement are explicit. | No GPU throughput or equivalence claim is made without GPU execution. Larger search budgets are not a speed improvement. |

The reason color evidence matters is geometric: point-to-plane residuals do not
constrain sliding along a broad plane. A joint objective can retain information
that a geometric refinement loses after using RGB only for initialization; see
the [Open3D colored registration example](https://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html?highlight=point+cloud).

## Recorded-session evaluation

The three supplied ZIPs contain 376 raw captures. Their archived accepted poses
are revalidated as initial guesses, not treated as reference trajectories.
Input hashes and aggregate results belong in
[`benchmarks/algorithm-review.json`](benchmarks/algorithm-review.json).
Raw observations, detailed poses, logs, and rebuilt meshes stay outside Git.

| Recording | Raw views | Archived accepted views | Median stored interval |
| --- | ---: | ---: | ---: |
| chest-3 | 126 | 68 | 4.65 s |
| chest-2 | 200 | 123 | 1.18 s |
| chest | 50 | 42 | 1.07 s |

These recordings contain no independent reference trajectory or reference mesh.
Coverage and algorithmic work counts are useful, but they cannot establish
absolute pose accuracy. Replays overlapped other work on this machine; their
elapsed times must not be interpreted as a controlled end-to-end speedup.

### Results with identical archived pose seeds

The baseline runs measure registration only at `e31c03e`; the combined runs also
perform fresh fusion and mesh extraction. Every saved pose is revalidated.
The combined replay implementation is `32fd3ad`, with documentation at `41132fe`.
The final cache guard does not change results for these immutable prepared views.

| Recording | Verified views before → after | Tested fragment pairs before → after | Final triangles | Required blocks |
| --- | ---: | ---: | ---: | ---: |
| chest-3 | 16 → 120 | 91 → 54 | 405,164 | 7,361 |
| chest-2 | 174 → 174 | 153 → 148 | 475,294 | 6,561 |
| chest | 16 → 16 | 10 → 7 | 82,058 | 2,975 |

All builds succeeded with a **10,000-block final budget**, preserving the saved
5 mm voxel size and confidence 2.0. The archived 5,000-block budget is too small
for the two larger reconstructions. No source archive was changed; SHA-256
hashes were checked again after rebuilding. Registration thresholds were not
relaxed to increase coverage.

Chest-3's cumulative development trials isolate some of the effect: preserving
the boundary motion alone retained 17 views; adding frontier search kept those
17 while reducing tested pairs from 90 to 13; retaining the two overlap witnesses
in the geometry-only path retained 96. The combined visual verification reaches
120. These stages alter the available candidates, so their pair counts are not
identical workloads or independent estimates of each feature's contribution.

The [continuous tracking review](CONTINUOUS_VISUAL_TRACKING.md) separately replayed
the stored chest-3 observations: accepted live views increased from 68 to 104,
and loss episodes fell from 15 to 9. Initializing this review's combined build
from those new live poses also retains 120 views, producing 389,980 triangles and
requiring 7,112 blocks. It tests 26 of 75 candidate pairs. Six raw views remain
unconnected; local registration and graph verification exclude them.

Both combined chest-3 builds reject some optimized bridge poses and explicitly
fall back to revalidated measured graph poses. The extra final refinement finds
no trustworthy loop constraints in any of the three recordings. Preserving
verified connectivity does not establish that accumulated drift was corrected.

### Speed and geometry checks

Depth preparation was measured on eight evenly sampled frames per recording,
with 24 alternating measurements of each path. Median full RGB-D versus
depth-only preparation was 34.94 / 6.83 ms, 30.08 / 5.52 ms, and 25.95 / 4.60 ms
for chest-3, chest-2, and chest respectively: **5.1–5.6× faster for that helper**,
with identical depth arrays. A repeated matching workload of 16 requests took
99.61 ms fresh versus 6.06 ms cached, including its cold fill, with exact forward
and reverse results. These are operation-level measurements, not whole-scanner
speedups.

As a consistency check, 20 common chest-3 views were raycast at each mesh's own
estimated poses and compared with raw calibrated depth, sampled every four
pixels. The fraction of measured pixels explained within 15 mm was 66.09% for
the other review's build, 67.28% for this combined build with the same new live
seeds, and 69.11% with archived seeds. Corresponding 95th-percentile errors over
hits were 26.07, 25.62, and 23.82 mm. Uncapped RMSE over hits was slightly worse
(56.47, 58.69, and 59.35 mm), so the evidence is mixed, not a uniform error
improvement. Missing-aware capped RMSE was 48.67, 48.62, and 47.33 mm. Background
and occlusion mismatches affect these values. This is **in-sample consistency**;
neither the observations nor estimated poses supply independent ground truth.

Oblique and overhead renders show much more of the chest-3 lid and one chest in
chest-2 without the earlier displaced duplicate. Some side-wall holes and noisy
floor boundaries remain. The oldest recording still has a large missing lid
region; it is not a successful full reconstruction.

Detailed reports, PLY meshes, logs, and previews are preserved locally under
`/Users/sasha/hobby/xbox360/algorithm-review-20261006/`. The `chest-3`, `chest-2`,
and `chest` folders use archived seeds; `chest-3-tracked` uses the new live seeds.
`chest-3-final-comparison.jpg` shows the earlier result alongside the combined
new-live-seed build at identical camera views. `remaining-sessions.jpg` shows the
other two final meshes. `evidence/` preserves registration ablations and logs.

### Validation and remaining measurements

The combined full suite ran **285 tests: 283 passed and two CUDA tests skipped**.
The separate real HTTP/WebSocket plus offscreen Qt synthetic workflow passed
capture, Finish, rebuilding, and exports. Tests cover measured boundary motion,
real gaps, independent overlap witnesses, invalid optimized poses, disconnected
components, bounded exact-match caching, depth equivalence, and transactional
fusion failure. The final cache-identity guard was included in a fresh full-suite
run before integration.

Physical camera motion with continuous acquisition tracking still needs a new
scan: intermediate camera images are absent from these ZIPs. CUDA performance,
absolute dimensions, and long-sequence drift remain unmeasured. The next useful
experiments are controlled surface/trajectory measurements for full pose
observability, robust depth normals, and confidence weights; this review does
not change those heuristics based only on larger output meshes.

Reproduction uses the existing read-only session loader and reconstruction tool:

```bash
OMP_NUM_THREADS=4 KINECT_BLOCK_COUNT=5000 python scripts/reconnect_session.py \
  /path/to/session.zip --use-pose-seeds --block-budget 10000 \
  --output-dir benchmark-output/review
OMP_NUM_THREADS=4 KINECT_BLOCK_COUNT=5000 QT_QPA_PLATFORM=offscreen \
  python -m unittest discover -s tests -v
```

For strict timing comparisons, run the same input and settings sequentially on
an otherwise idle machine, record source versions, and compare final accepted
indices as well as elapsed time. Lower search time achieved by discarding useful
geometry is not a successful optimization.
