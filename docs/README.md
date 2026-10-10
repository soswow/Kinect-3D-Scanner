# Documentation

Start with the [scanner setup and workflow](../README.md). Run tools from the
repository root in the scanner environment; the [script index](../scripts/README.md)
separates maintained commands, component benchmarks and source-bound research.

Local investigation reports and agent review ledgers belong in ignored
`docs/investigations/<YYYY-MM-DD>-<topic>/` (create it as needed). Find prior
reports with `rg --files --no-ignore docs/investigations`. They stay in the local
checkout; promote durable findings into the relevant maintained guide so they
are available to other contributors and fresh clones.

## Using the scanner

| Need | Guide |
| --- | --- |
| Measured Kinect calibration and native depth units | [Calibration](../calibration/README.md) |
| Capture without waiting for reconstruction or uploads | [Offline capture and buffering](OFFLINE_CAPTURE_UPLOADS.md) |
| RGB/depth timing warnings | [Pairing and color assistance](RGB_DEPTH_TIMING.md) |
| Tracking loss, reference view and recovery | [Tracking recovery](TRACKING_RECOVERY.md) |
| Camera motion, feature flow and recording for replay | [Continuous visual tracking](CONTINUOUS_VISUAL_TRACKING.md) |
| Accelerometer calibration, gravity and portrait display | [Kinect accelerometer](KINECT_ACCELEROMETER.md) |
| CPU/native/CUDA selection and textures | [Live pipeline and exports](LIVE_RECONSTRUCTION.md), [native kernels](NATIVE_PERFORMANCE.md) |

## Current architecture and evaluation

- [Final registration](FRAGMENT_RECONNECTION.md): experimental depth/color/motion
  mode and the existing bounded fragment mode, with different acceptance policies.
- [Camera loop refinement](POSE_REFINEMENT.md) and [joint RGB-D refinement](JOINT_RGBD_REFINEMENT.md):
  optional legacy Finish proposals, independent depth checks and transactional fusion.
- [Depth confidence](STATISTICAL_DEPTH_FUSION.md): relative weights, controlled
  surface tests and why weight is not a physical uncertainty estimate.
- [Quality benchmarks](QUALITY_BENCHMARKS.md): withheld views, missing-surface
  penalties and reproducible comparisons.
- [Research roadmap](RESEARCH_ROADMAP.md): candidates and unresolved evidence;
  [development history](DEVELOPMENT_HISTORY.md) records completed checkpoints.

## Historical evidence and specialist references

These results describe their recorded sources and inputs, not today's defaults
or independent scan accuracy. Raw recordings, meshes and detailed local reports
are intentionally absent from a public checkout. Follow their compact benchmark
links for checked-in evidence; use your own session paths for reproduction.

| Evidence | Scope |
| --- | --- |
| [Scan quality baseline](SCAN_QUALITY.md) | Public RGB-D normalization and early trajectory measurements |
| [Algorithm review](ALGORITHM_REVIEW.md) | Boundary continuity, frontier search and exact preparation/matching reuse |
| [Chest 3](CHEST_3_INVESTIGATION.md), [Chest 4](CHEST_4_TRACKING_RECOVERY.md), [Chest 8](CHEST_8_INVESTIGATION.md) | Coverage loss, recovery and half-turn graph failures |
| [Recorded RGB-D evaluation](RECORDED_RGBD_EVALUATION.md) | Cache variation, confidence coverage and refused bundle proposals |
| [Capture performance](CAPTURE_PERFORMANCE.md) | Compression, CPU fusion and bounded packed live feedback |
| [Initial CUDA study](CUDA_PERFORMANCE.md) | Fusion-only measurements |
| [CUDA experiment reference](CUDA_EXPERIMENTS.md) | Full matrices, rejected approaches, exact-query proofs and reproduction |
| [Field CUDA research](FIELD_CUDA_RESEARCH.md) | Chest 5–7 allocation, recovery tradeoffs and source-bound experiments |
| [Room replay](FULL_ROOM_SCAN_20261010.md) | Normal-upload motion evidence and unresolved room coverage |
| [AprilTag priority](FRAGMENT_RECONNECTION.md#apriltag-priority) | Conditional shared-corner initialization, recovery and recorded timing limits |
| [Saved CUDA reports](benchmarks/CUDA_REPORTS.md) | Published summaries, charts and provenance |
| [Linux/Python API reference](KINECT_V1_LINUX_PYTHON_REFERENCE.md) | Educational standalone examples; use production guides for the app |

Read the focused guide first. Long research references retain negative results,
exact source fingerprints and evidence limits that are needed to interpret old
proofs; their component timings must not be presented as complete scanner gains.
