# Chest 4 tracking recovery

The saved Chest 4 recording contains 164 raw RGB-D pairs. Its original live
scan accepted 114, but Finish retained only four after the disconnected
trajectory failed verification. The updated Finish reconstructs **162 of 164
captures** from those same four archived pose proposals, with **five independent
cycles in the surviving verified fragment graph**. Captures 52 and 110 remain
disconnected. Capture numbers in this document are one based.

## Changes

- The camera motion tracker retains five recent good references. A bad image
  or one badly synchronized RGB-D pair no longer erases them. A gap exceeding
  0.75 seconds still starts a new camera coordinate system.
- Automatic capture chooses a sharp, fresh pair from the last five arrivals
  within 300 ms, preferring a valid measured camera motion estimate. RGB,
  depth and motion metadata remain paired. The score compares candidates;
  it cannot guarantee a sharp image when all candidates are blurred.
- While tracking is lost, automatic capture sends a fresh recovery probe as
  soon as the previous probe finishes, with a 100 ms minimum spacing and only
  one outstanding probe. Normal pacing resumes after recovery. Unused camera
  candidates are discarded; uploaded observations remain available to Finish.
- Live registration ranks up to forty accepted views, including the initial
  five, recent eight and spaced landmarks. The five strongest descriptor
  candidates undergo geometry checks. Raw geometric recovery also tries the
  last five accepted observations.
- Finish tries the previous five prepared views and retains more fragment
  boundary witnesses. Scale-tolerant SIFT features assist wider recovery and
  Finish retrieval. Reciprocal geometry, feature identity, independent depth
  samples, and multiple independent camera witnesses still verify connections.
- Nonadjacent local links use the existing per-capture motion allowance over
  at most three capture steps. Depth residual, overlap and cycle-consistency
  thresholds remain unchanged. Final guidance reports verified loop constraints.

## Rebuilding the saved final session

| Result | Original | Updated Finish |
| --- | ---: | ---: |
| Retained captures | 4 | 162 |
| Vertices | 1,843 | 301,668 |
| Triangles | 2,615 | 581,767 |
| Surface area | 0.0229 m² | 4.8427 m² |
| Voxel size | 5 mm | 5 mm |
| Final confidence | 2.0 | 2.0 |

The new pass connects fifteen of seventeen fragments, reusing nine links to
recent views earlier than the immediate predecessor. It tests 28 of 98
candidate fragment pairs; neither the pair nor fragment limit is reached.
Four surviving returning-view bridges connect captures around 22–27 to
captures around 142–147. They preserve multiple independent color/depth
witnesses, providing actual evidence from the return around the object.

Global optimization moves several bridges beyond their raw-data checks, so
the existing fallback restores measured bridge poses, revalidates surviving
constraints and recomputes connectivity. The five reported loops survive that
fallback. A separate refinement pass finds no additional trustworthy loops.
This result demonstrates verified return-view connections and much greater
coverage; it does not establish that all accumulated drift has been corrected.

Matched oblique and overhead renders show a single box/table reconstruction
with much more coverage. The lower woven basket still has substantial holes.
Additional floor and wall coverage also contributes to the triangle count;
voxel resolution and confidence settings are unchanged.

## Full live and Finish replay

Replaying all raw captures through live tracking before Finish accepts **130
captures**, versus 114 in the original recording. Consecutive rejected-capture
runs drop from sixteen to fourteen. This server-side replay still loses track;
it cannot exercise the new camera-reference retention or sharp selection on
intermediate camera frames missing from the archive.

Finish recovers the other 32 connected observations, corrects 129 accepted
poses and excludes no live-accepted captures. It reaches the same 162 captures,
five graph cycles, 28 tested fragment pairs, and two missing captures as the
saved-pose rebuild. The mesh has 301,667 vertices, 581,764 triangles and
4.8427 m² of surface, using 9,120 of 10,000 blocks. Both runs use unchanged
calibration, 5 mm voxels and confidence 2.0; their small mesh-count difference
does not establish a quality difference.

## Validation and artifacts

The final implementation ran 326 tests: 324 passed and two CUDA tests were
skipped. Tests exercise actual optical flow across a bad image, measured
scale-changing feature matches, sharp/blurred frame selection, recovery pacing
and bounded probes, reconnection across invalid depth, and a synthetic
returning loop against evaluation-only camera truth. Lint passes for changed
code, with existing `BLE001` warnings in `engine.py` and `RUF007` in
`refinement.py` excluded.

The original ZIP checksum remains
`ab0465d56d05c172074075b056c22f10a3337c83b2e564c148d4f10ea64ce49f`.
The measured replay reports confirm unchanged input and implementation sources
throughout their runs. CPU runs overlapped, so their timings are not a
controlled performance comparison. The saved session contains selected
captures and old motion summaries, not intermediate camera images. Sharp
candidate selection, camera-reference preservation and faster physical recovery
need a fresh Kinect capture. Absolute geometry accuracy remains unmeasured.

Complete reports, meshes and matched renders are under the ignored directory
`/Users/sasha/hobby/xbox360/Kinect-3D-Scanner/benchmark-output/chest-4-improvement/`.
The saved-session rebuild is `sift-finish.json`, `sift-finish.ply`, and
`sift-finish-comparison.jpg`; the full live replay is `final-after.json`,
`final-after.ply`, and `final-after-comparison.jpg`. The compact checked-in record is
[chest-4-tracking-recovery.json](benchmarks/chest-4-tracking-recovery.json).
Restart the client and server to use the changes in a new scan.
