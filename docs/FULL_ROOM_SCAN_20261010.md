# Fast room capture: normal-upload replay, 10 October 2026

The reported scan originally retained 42/130 selected captures after roughly
33 minutes. Some accepted poses were inverted relative to reliable gravity;
the resulting model was unsuitable. The client did acquire acceleration, but
RGB delivery delay was being included in the shared receipt-spread estimate,
making its image associations unreliable. The offline depth graph also did not
consume the continuous visual and acceleration measurements, and spent most of
its time on geometric recovery without correcting room-wide drift.

The optional full camera archive was used **only as a virtual client input**.
`scripts/benchmarks/replay_normal_capture.py` runs normal camera tracking,
capture selection, packet encoding/decoding and Finish journal construction.
The reconstruction engine then opens only the resulting ordinary archive:
selected RGB-D images, bounded intermediate feature measurements, motion
metadata and `motion.json`. It reads no intermediate images or archived poses.
The new capture policy selected 294 images from the same physical recording,
including additional overlapping images during fast or unverified movement.
Consequently this is not a comparison using the same selected image count.

The cold server run started without poses, pair caches or prepared-view caches.
It produced 256 connected/fused captures, 1,751,004 vertices and 3,407,456
triangles at the original final 5 mm resolution in 1,620.34 seconds (27 minutes).
The remaining 38 views were preserved in seven independent fragments, including
a 30-view fragment whose tested room placements contradicted measured depth.
The final selected map's sampled empty-space diagnostic was 4.145%, over
48,738,004 supported surface projections. These are consistency measurements,
not ground-truth pose or dimensional-accuracy measurements.

The ordinary upload carried 1,401 intermediate visual observations, 2,087
intervening acceleration reads and a 2,292-read Finish journal. All 294 capture
intervals were complete. The source journal's dropped read remained explicit;
its completeness was not upgraded. Intermediate feature/motion metadata totalled
about 12 MB. Optional intermediate images were absent from the server input.

Sparse camera/landmark fitting, measured selected surfaces and shared planes
correct internal drift before accumulated maps are rigidly connected. Selected
raw images supply their own depth observations; compact intermediate pixel/depth
measurements never authorize fusion without selected raw-depth checks. Gravity
supplies uncertain roll/pitch evidence. Kinect v1 has no gyro, and acceleration
is not integrated into position. Full-motion transport requires the updated
client and server; old clients cannot retrospectively supply missing feature
history in an ordinary server export.

The final depth audit allows isolated moving surfaces with an 8% aggregate
empty-space limit and retains severe individual pair conflicts in the report.
This is a stricter mean limit than the earlier 12% policy, while removing its
single-pair veto. Pair overlap, range residuals, feature identities, reciprocal
fits, conditioning and competing hypotheses remain checked. The user confirmed
minor object movement. The reconstructed window, walls, floor, green board and
furniture form one recognizable room, but holes, local surface distortion and
unconnected captures remain. The preview uses 25 mm vertex clustering for its
3D display; the exported mesh retains the original 5 mm reconstruction.

Finish is still expensive. Replaying this recording does not validate a new
physical capture's USB timing, client throughput or absolute geometric accuracy.
The improvements are implemented as ordinary next-scan behavior, with their
engineering weights and incomplete evidence reported explicitly.

Validation covered motion history/journals, high-resolution packet bounds,
clock pairing, buffering, Finish ordering, API/project round trips, camera
tracking, sparse derivatives, known-pose recovery, raw surface constraints,
non-Manhattan measured planes, map coordinate changes and moving-object versus
camera-displacement rejection. Two existing capture-performance tests failed
also on unchanged master: CPU/tensor fusion parity and mocked stage timing.
No claim is made that the entire repository suite passes on this Windows host.
After integration with the current marker-tracking and texture-export changes,
307 focused tests passed in 57.78 seconds.
