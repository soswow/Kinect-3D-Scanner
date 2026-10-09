# Measured loop refinement after fragment reconnection

The optional pose-graph refinement stage independently estimates camera-level
constraints after fragments have been connected. A fragment end/start match is
useful evidence for retrieval, but does not automatically become an accepted
camera-level loop or certify distributed drift correction.

The pass selects up to 64 keyframes. Half the capacity reserves uniform temporal
coverage; the remainder includes exact support views from retained, committed
fragment bridges, prioritizing protected temporal/storage boundaries. A saved
fragment transform is never used as a measured camera constraint. Only camera
pair identities are reused; raw RGB/depth must establish their transforms again.

Clouds use 7.5 mm voxel sampling, capped at 48,000 points per selected view.
Alternating points train ICP; the remaining points split between constraint
witnesses and final independent geometry validation. The earlier 20 mm sampling
and subsequent split could produce sampling error comparable to the 15 mm
alignment threshold. Sampling is now denser without relaxing that threshold.

A loop requires finite transforms, at least 60% training overlap, 100
correspondences, at most 15 mm RMSE, and varied surface normals. Forward and
reverse alignment must agree within 1 cm and 2 degrees. Maximum correction from
the current estimate remains 20 cm/15 degrees for purely geometric proposals,
or 75 cm/30 degrees for measured RGB-D seeds. Candidate count is bounded at 40;
the cooperative proposal time budget is 90 seconds, including validation.

Sequential graph edges retain reliable ICP when measured visual identities
confirm it; otherwise they prefer distributed measured RGB-D motion when
separate witnesses confirm it. Retained temporal/storage ties
are re-estimated with distributed measured RGB-D
features and separate geometric witnesses. They are certain graph constraints,
with information weighted 20 times the geometric edge information to discourage
ICP sliding over established visual identities. This is engineering weighting,
not calibrated uncertainty. A post-optimization raw check remains mandatory;
weighting cannot authorize violating a measured boundary. If keyframe capacity
cannot cover those boundaries, the proposal is refused explicitly.

The robust optimizer must retain measured loop edges. Independent keyframe
geometry must improve by at least 2%, with no constraint worsening by more than
10% plus the existing saturated-loss allowance. Corrections interpolated in the
anchored first-camera world frame preserve the first camera and do not depend
on the world origin or switch local axes as intermediate cameras turn.
Every resulting view, including omitted cameras between keyframes, is then
checked with on-demand independent measured depth outside feature patches. Its
mean loss must improve and every comparison must retain overlap and avoid
worsening beyond the same per-pair allowance. Initially low-overlap comparisons
must retain their initial overlap within the existing five-percentage-point tolerance,
rather than failing solely because the unchanged baseline is below the normal
35% overlap floor. The cloud cache is bounded at eight views, as in bundle adjustment.
Fresh TSDF fusion must succeed before poses and geometry are committed together.

`candidate_results` records the stored camera indices, origin, measured overlap,
alignment error, reciprocal disagreement and correction magnitude. Rejection
counts distinguish missing RGB-D seeds, insufficient overlap, high error,
degeneracy and correction bounds. Optimizer pruning, independent-validation
failures, protected-boundary violations and time-budget exhaustion have separate
messages. A refusal means this bounded pass lacked an acceptable proposal; it
does not imply that physical overlap is absent.

To replay both refinement passes without fusion or a running server:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:KINECT_DEVICE = 'cpu'
$env:KINECT_NATIVE = 'off'
$env:KINECT_CUDA_INPUT = 'off'
$env:KINECT_CUDA_REGISTRATION = 'cpu'
python scripts/benchmark_pose_refinement.py --session path/to/original.zip `
  --poses path/to/connected/reconstruction.json --output path/to/proposals.json
```

The producer records source and input checksums, preserves raw timing/calibration
and verifies that its inputs are unchanged. It records proposed poses separately
from applied reconstruction results. Without independent camera or surface
truth, lower held-out error demonstrates consistency, not absolute accuracy.
