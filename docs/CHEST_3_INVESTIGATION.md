# Chest 3 tracking and final surface investigation

The follow-up [continuous visual tracking trial](CONTINUOUS_VISUAL_TRACKING.md)
implements and tests changes motivated by this investigation. The description
below records how the original session was captured.

Investigated 6 October 2026 on local revision `030486c`, using
`export/chest-3-scan-session_20261006_215208.zip`. The archive is a capture-time
snapshot: fragment reconnection, pose refinement, and final reconstruction all
say **Awaiting final build**. Frame numbers here are one based; JSON indices are
zero based. Detailed observations and meshes stay outside Git.

## Measured capture behavior

The scan ran from 21:42:07 to 21:51:31 Sydney time, lasting 563.92 seconds.
Of 126 observations, **68 were accepted and 58 rejected**. The final three
minutes contain 32 observations, only 14 accepted. There were 15 loss episodes:
two began with a pose jump and thirteen with failed raw-camera verification.
Another 43 observations failed while trying to recover.

| Measurement | Result |
| --- | ---: |
| Median stored capture interval | 4.65 s |
| First 25 intervals, mean | 2.08 s |
| Intervals beginning at captures 76–100, mean | 6.28 s |
| Capture interval range | 1.07–8.57 s |
| Median processing time | 2.47 s |
| 90th percentile processing time | 4.82 s |
| All stored pairs eligible under the 20 ms RGB/depth guard | 126 / 126 |
| RGB/depth timestamp offset range | −16.51 to +16.59 ms |
| Accepted methods | 52 RGB-D + ICP, 7 anchor recovery, 8 appearance recovery, 1 reference |

Unlike the earlier chest session, timestamp eligibility did not disable color
assistance. These are packet-end timestamps, not an independent measurement of
exposure synchronization. In selected frames, 93–97% of the depth image remains
valid after calibration and clipping. This is not a shortage of depth pixels.

The original recorded processing stages total about 341 seconds: tracking 57.2%,
model extraction/normal pyramids 20.8%, transition verification 9.6%, fusion 8.0%,
and input/cloud preparation 4.4%. The camera supplies high-resolution RGB at
10 fps, but the tracker consumes only the stored observations. Automatic pacing
includes recent processing and capture-to-feedback completion time, with 15%
headroom and slow recovery after a slowdown. No continuous high-rate trajectory
was recorded between those observations.

The client log between the first capture upload and session export also contains
70 invalid-magic, 27 inconsistent-flag, and 19 packet-resynchronization warnings.
They warrant a separate USB/acquisition check, but the archive cannot associate
each warning with a stored observation. They are not established as the cause
of the reproducible registration disagreements below.

## What the tracker actually does

`scanner_server/engine.py` tracks each calibrated observation against an
**accumulated TSDF surface cloud in world coordinates**, not exclusively against
the immediately previous frame. The source cloud uses 7.5 mm downsampling for
this scan. Point-to-plane ICP runs at 30, 15, and 7.5 mm scales, with respective
correspondence radii of 90, 45, and 22.5 mm. It minimizes geometric residuals;
RGB does not participate in that final ICP objective.

With color assistance enabled, hybrid RGB-D odometry first estimates motion
against the **last accepted raw RGB-D observation**. That is an initializer for
model ICP, not a retained photometric constraint. RGB is projected onto the
640 × 480 depth grid for this path; original 1280 × 1024 images remain available
for texture export. ORB recovery is separate and bounded at 1,200 features,
32 sampled historical views, and the three best descriptor candidates. It only
starts after at least two rejected observations and requires strong measured
depth/geometric verification.

The model's nominal refresh interval is three integrations, but color assistance
refreshes it whenever an integration has occurred since the last extraction.
For this setting, model preparation therefore occurs before almost every next
ordinary tracked observation. The increasingly large room-wide cloud, its
normals, and multiple ICP passes make tracking expensive; slower automatic
capture then increases the baseline that the tracker must solve.

Ordinary model registration accepts overlap ≥0.35, RMSE ≤20 mm, motion ≤30 cm /
30°, and a normal-diversity check. A step over **10 cm or 8°**, or an unusually
long capture gap, needs another check against the last accepted raw observation:
reciprocal overlap ≥0.60, RMSE ≤15 mm, non-planar normals, cycle error <10 mm /
2°, and agreement with the model-derived pose within 30 mm / 3°.

After rejection, ordinary model ICP and global FPFH alone cannot resume fusion.
Return-to-anchor recovery starts raw-camera ICP at identity and requires motion
within 15 cm / 15°, reciprocal verification, and model agreement. Verified
appearance relocalization is the other recovery path. Thus the normal tracker
uses the full model, but its color motion, transition evidence, and first recovery
route remain dependent on the last accepted observation. Continuing around the
object after loss takes the camera farther from that recovery anchor.

The graph linking is **offline at Finish**. Fragment reconnection re-estimates
raw sequences, verifies inter-fragment bridges, optimizes a graph, and fuses only
the connected component anchored to the first retained fragment. It can recover
rejected views and exclude previously accepted views. Optional pose refinement
then uses up to 32 accepted keyframes and bounded loop proposals. Neither graph
currently maintains the live trajectory while capturing.

## Why many pixels still permit the wrong pose

Point-to-plane ICP matches nearest geometry, rather than following persistent
pixel identities. Many measurements on the same floor or broad box face repeat
the same constraints. Motion along a plane can leave the residual small. Wicker
and floorboards also provide repeated appearance, while the captured depth may
resolve their fine texture poorly. This scan includes the room and floor, with
no ROI and a 0.5–3 m depth range, so global overlap is not object-specific.

The [Open3D colored registration example](https://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html)
demonstrates planar slipping under geometric ICP and a joint color/geometric
objective that constrains tangent motion. The scanner's color initializer is
not that joint objective. This explains a plausible mechanism; the measurements
below establish disagreement between solvers, not an absolute ground-truth pose.

The isolated diagnostic replay integrated the **archived accepted pose estimates**
and re-evaluated the first rejection of every loss episode. All thirteen original
transition rejections reproduced, as did the two pose-jump errors. None of those
thirteen gaps exceeded the adaptive timestamp-gap limit. Their estimated motion
triggered verification. Across those checks, ten failed model/raw-pair agreement,
two failed reciprocal overlap, and two failed cycle translation; categories can
overlap. None failed the normal-diversity threshold in this replay.

| Rejected capture | Model overlap | Model RMSE | Difference from raw-camera alignment |
| --- | ---: | ---: | ---: |
| 63 | 96.3% | 7.8 mm | 46 mm / 3.18° |
| 74 | 95.0% | 6.6 mm | 217 mm / 13.43° |
| 87 | 98.9% | 8.5 mm | 211 mm / 10.23° |
| 89 | 96.4% | 9.1 mm | 272 mm / 14.75° |
| 118 | 98.9% | 6.6 mm | 133 mm / 9.19° |

For capture 89 the raw forward/reverse cycle error is only 1.3 mm. A high model
overlap therefore does not establish that the model registration chose the right
pose. The gates are rejecting substantial disagreements, not simply running out
of usable pixels. Several smaller disagreements are close to current limits,
but enlarging limits would also accept the much larger examples.

These are recomputed candidates, not recorded original rejected poses: current
failure diagnostics omit the candidate pose and individual failed checks. The
saved prefix may itself contain drift, and no reference camera trajectory exists.

## Why the final mesh has fewer faces

Live confidence fusion extracts a **point cloud** at effective weight **0.01**,
bounded to 30,000 displayed points. It is not a preview of final mesh faces.
Inspect Scan's temporary mesh uses weight 0.5. This session's final mesh uses
weight **2.0**, then removes connected components under 30 triangles.
Confidence weights are fractional range/angle/edge contributions, not frame
counts. The median per-frame mean contribution was 0.272, ranging 0.135–0.440.
Weight 2 therefore often needs roughly eight useful observations of the same
surface; the actual requirement varies per pixel and view.

An isolated fusion of the same 68 saved poses gives:

| Extraction weight | Surface points before display cap | Mesh triangles after cleanup | Surface area |
| --- | ---: | ---: | ---: |
| 0.01, live-cloud threshold | 1,248,792 | 2,141,548 | 17.29 m² |
| 0.2 | 508,511 | 847,410 | 7.08 m² |
| 0.5, temporary mesh threshold | 326,069 | 589,936 | 4.99 m² |
| 1.0 | 251,779 | 463,897 | 3.95 m² |
| 2.0, final threshold | 183,437 | 332,064 | 2.84 m² |

These counts include surrounding surfaces and drift artifacts; more faces are
not evidence of greater accuracy. At weight 2 the small-component filter removes
only 5,088 triangles, so confidence filtering accounts for most of this sweep's
reduction. There is no mesh decimation in ordinary PLY final building.

The actual final PLY downloaded at **21:59:57** was retained in the client's
temporary directory. It contains **52,722 vertices and 96,845 triangles** and was
copied to the analysis folder before it could be cleaned up. The additional
reduction from 332,064 to 96,845 cannot be explained by applying weight 2 to the
unchanged live volume. Finish can change both the retained observations and
their poses. The original ZIP contains no report of that later build.

The Finish replay now **reproduces the actual final mesh's 52,722 vertices and
96,845 triangles**, with exactly matching vertex positions. It retains
**only captures 1–16**, corrects thirteen of those
poses, excludes **52 previously accepted captures**, and recovers **zero** rejected
captures. It prepares 21 local fragments and tests all 91 eligible fragment
pairs, without reaching a search or memory cap. There are 37 verified bridges
among later fragments, but **none connects to fragment 0**. Twenty fragments
remain outside the anchored model. Final pose refinement finds no additional
trustworthy loop constraints. Fresh anchored fusion requires 2,333 blocks,
well within the configured 10,000-block budget.

Thus the graph did run, but it reduced the reconstruction to the first sixteen
views. Those views do not supply enough weight over every lid/side voxel at the
final threshold. This is the main explanation for the additional disappearance
of surfaces at Finish. The retained raw observations and later local fragments
are still present in the session; exclusion does not delete them. The replay
report records their connection status. A successful build currently means a
nonempty mesh was built, not that every captured view was connected.

## Recommended tracking work

1. **Separate tracking cadence from expensive fusion.** Track fresh small RGB-D
   observations at a short baseline; send selected posed keyframes to the fusion
   worker. Bound latency and measure throughput before choosing a target rate.
   Increasing the upload queue alone would add delayed work without fixing the
   changing reference baseline.
2. **Keep visual evidence in the solve and verification.** Evaluate joint
   photometric/geometric tracking or persistent measured 3D feature tracks.
   Reject disagreement between the color proposal and ICP instead of silently
   letting ICP erase a useful constraint. Check observability and object/region
   support in addition to aggregate nearest-neighbor overlap.
3. **Use a small visible local map and several nearby keyframes.** Avoid preparing
   the full accumulated room model for every observation. Retain independently
   measured keyframe geometry and appearance for confirmation and recovery.
4. **Connect local maps during capture.** Continue a local short-baseline tracker
   through loss of the global model, then attach it only after independently
   verified multi-view bridges. Bounded background graph work needs versioned
   state and safe fusion reintegration when poses change.
5. **Make the preview predict export coverage.** Show a final-confidence surface
  layer and explain the views removed by Finish. Record candidate poses, color
   proposal corrections, reciprocal metrics, failed gate names, and solver
  conditioning on rejected frames. This session's generic loss message hides
  which mechanism failed.

The Finish interface should also report this scan's result prominently: **16 of
126 views retained; 52 live views excluded; 20 fragments unconnected**. That is a
material loss of scan coverage even though native mesh extraction succeeds.

Priority is the capture/track cadence and preserving visual constraints. The
current protective gates already prevent several convincing geometric matches
from corrupting the live volume, but the architecture gives them increasingly
difficult, widely spaced observations.

## Reproduction and retained evidence

Use the existing scanner environment; both commands run isolated engines and
leave the source ZIP and running server alone:

```bash
OMP_NUM_THREADS=4 KINECT_BLOCK_COUNT=5000 python scripts/diagnose_session.py \
  export/chest-3-scan-session_20261006_215208.zip \
  --output-dir benchmark-output/chest-3-investigation --export-meshes

OMP_NUM_THREADS=4 KINECT_BLOCK_COUNT=5000 python scripts/reconnect_session.py \
  export/chest-3-scan-session_20261006_215208.zip --use-pose-seeds \
  --output-dir benchmark-output/chest-3-investigation/final-replay
```

The diagnostic meshes use unvalidated saved live poses and are comparison
artifacts. The Finish replay independently verifies archived poses as seeds.
Detailed reports, contact sheet, surface comparison, and copied final model are
preserved in `/Users/sasha/hobby/xbox360/chest-3-session-analysis/`.
The [aggregate measurements](benchmarks/chest-3-summary.json) contain no images
or raw geometry. Source ZIP SHA-256:
`08b8f2615348dfd6ceb9a14b8c9b964ea05efbc3ca0351b8f3d94783f9c4a3e4`.

Validation: the full 126-frame diagnostic replay reproduced all fifteen loss
starts; the Finish replay reproduced the actual final surface counts; a separate
two-frame synthetic ZIP audit verified report generation and unchanged source
bytes. The diagnostic tool passes Ruff and Python compilation. No production
tracking or fusion policy was changed by this investigation.
