# Tracking loss and the chest session

The session `chest_20261005_230320.zip` retains the raw observations needed for another reconstruction attempt. It contains 50 captures, 42 accepted poses, and 8 rejected observations. Frame numbers below are one based; reconstruction JSON indices are zero based.

## What happened

| Capture | Result | Gap before capture | Last accepted recovery reference |
| --- | --- | --- | --- |
| Frames 15–19 | Rejected; the first alignment asked for about 0.62 m and 41° of motion | 10.37 s before Frame 15 | Frame 14 |
| Frame 36 | Rejected; about 0.89 m and 55° | 15.55 s | Frame 35 |
| Frames 49–50 | Rejected; about 0.62 m and 28° initially | 29.93 s before Frame 49 | Frame 48 |

Accepted runs are Frames 1–14, 20–35, and 37–48. These are runs in capture continuity, not separate pose-graph coordinate systems. The engine always registers against the existing fused model; it does not start an independent submap when tracking fails. A low-error match to the wrong side of a box or to the floor can still pass the ordinary overlap threshold. After the first gap, Frame 20 resumed ordinary ICP with overlap 0.590. A direct reciprocal match between Frames 35 and 37 disagreed by about 0.75 m / 49°, which is evidence against trusting that transition.

The object, floor, and surrounding room are all included: no crop was configured, and the depth range was 0.5–3.0 m. Cuboid faces, woven repeating texture, and a broad planar floor make geometric matches ambiguous.

Color-assisted tracking and appearance recovery were enabled, but **45 of the 50 stored RGB/depth pairs exceeded their 20 ms synchronization limit**. Their offsets range from about 14 to 49 ms. Only 5 pairs were eligible individually, and adjacent RGB-D odometry requires both observations to pass. These settings therefore supplied much less assistance than their enabled checkboxes suggest. The guards should not be weakened merely to hide this problem. The live panel now shows a persistent warning for a requested color recovery path whose current observation exceeds this timing limit.

Typical capture spacing was 1.07 s; reconstruction took roughly 1–4.4 s per accepted frame. Total recorded capture duration was 105.25 s and summed processing time was 106.45 s. The three long gaps precede the abrupt viewpoint changes; they do not establish whether movement was paused deliberately or by capture backpressure.

## Offline experiments and limits

The archive's final refinement and fresh final reconstruction were still awaiting a final build. Running the existing conservative pose refinement against all 42 saved poses found **zero trustworthy loop constraints**, including zero appearance candidates. Pressing Finish can attempt refinement, but it does not guarantee that this session becomes correct.

An independent SIFT feature experiment, using measured depth support and the existing PnP/depth-consistency gates, found plausible overlap from early views to Frames 49–50. It did not find a verified bridge between the three main view segments. Direct geometric matches across the large changes had asymmetric overlap; the existing reciprocal confidence checks rejected them. These experiments used the retained pairs with known timestamp mismatch to search for candidates, so their appearance proposals are not authority to fuse data.

A partial colored point cloud and mesh were reconstructed from **Frames 1–14 only**, with their saved accepted poses. The mesh uses the captured calibration and confidence fusion, with an extraction threshold of 0.2 effective observation weight. It includes the nearby environment, has 176,598 vertices and 313,775 triangles, and does not recover the unseen sides. This is a partial salvage, not a verified repair of the complete chest. The original ZIP is unchanged.

Local outputs are in `/Users/sasha/hobby/xbox360/chest-session-analysis/`: `summary.json`, `contact-sheet.jpg`, `prefix-frames-0-13.ply`, `prefix-frames-0-13-mesh.ply`, recovery UI screenshots, and the experiment reports.

## Live recovery behavior

Tracking loss now freezes fusion and preserves the last accepted camera pose. Automatic capture keeps supplying bounded recovery probes, with only one new probe queued after the existing processing queue drains. Rejected probes remain available in the raw session; they do not seed a new model. Manual Pause still stops capture completely.

The screen shows a large red **STOP — TRACKING LOST / Model paused** notice. An inset shows an overhead camera trajectory, accepted camera orientations, the last good camera in red, and its RGB reference image. Edges are omitted across skipped frames. The overhead direction is estimated from a dominant reference-frame plane; if that estimate is unavailable, the inset explicitly uses initial camera up. Neither view claims to know the current camera pose while tracking is lost.

To recover, stop moving forward around the subject, retrace to the highlighted camera, and match the reference image's framing, distance, and angle. Include a corner and stable background texture. Move slowly around that viewpoint until verification passes. A close return (at most 15 cm / 15°, or tighter configured motion limits) must match the actual last accepted raw observation in both directions, pass geometric confidence and cycle consistency, and agree with model refinement. Ordinary model ICP and FPFH matching alone cannot resume a lost track. Existing appearance relocalization remains available only with its stronger independent checks. A successful verified observation automatically resumes fusion.

## Algorithms that can help

Existing implementations already include coarse-to-fine ICP, optional hybrid RGB-D odometry, FPFH/Fast Global Registration, ORB/PnP appearance relocalization, and bounded pose-graph refinement followed by fresh TSDF fusion. See the primary Open3D documentation on [global registration](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html), [multiway pose-graph registration](https://www.open3d.org/docs/latest/tutorial/pipelines/multiway_registration.html), and [fragment registration](https://www.open3d.org/docs/latest/tutorial/reconstruction_system/register_fragments.html).

Useful next improvements are better sensor pairing for the selected RGB mode; a persistent warning when requested color recovery is unavailable; separate fragment storage with explicit connection status; independent fragment-to-fragment feature retrieval with reciprocal geometry verification; and robust global optimization followed by complete reintegration. A background recovery worker would need bounded work and session-version checks before committing any proposal. It cannot create overlap that was never captured, and a single ambiguous cube/floor match should not connect entire fragments.

For this particular scan, manually selected distinct correspondences across the gaps, or a new bridging scan that overlaps both neighboring segments, may provide the missing constraints. Merely increasing motion or overlap tolerance risks authorizing more wrong-side matches.

## Implementation and validation

Developed in the isolated `codex/chest-tracking-recovery` worktree and combined with the other chat's capture sound and automatically saved settings. Added engine recovery gates and tracking snapshots, a Qt trajectory/reference inset, a large loss notice, recovery probe backpressure, synchronization warnings, and `scripts/inspect_session.py` for reproducible read-only archive inspection. The UI continues to use the accepted camera orientation for Follow and supports Orbit separately.

Validation includes real Open3D tests for rejected depth leaving fusion and pose unchanged, return-to-anchor recovery, planar ambiguity rejection, and preventing ordinary model ICP from bypassing recovery. Qt checks cover estimated-up projection, loss/recovery visibility, and recovery probe queue limits. Screenshots use the actual session images and geometry at 960×600 and 1280×800. Physical Kinect recovery and CUDA remain unverified.

Combined verification: 177 tests passed, 2 CUDA tests skipped; the synthetic HTTP/WebSocket and Qt scan pipeline passed. After the synchronization warning was added, 56 affected tests passed and 1 CUDA test was skipped. Lint passed for the changed UI, API, inspection script, and test files; the engine retains its four pre-existing broad-exception lint findings.
