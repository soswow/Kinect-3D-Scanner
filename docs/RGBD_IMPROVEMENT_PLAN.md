# RGB-D tracking, statistical fusion, and joint final refinement

Research checked 7 October 2026. This plan extends the existing scanner rather
than replacing Kinect's measured depth with an RGB-only surface. Rankings are
engineering judgments for this CPU-capable Python/Open3D application; published
performance is not a benchmark of this implementation.

## Candidates and priorities

| Area | Candidate | What it contributes | Decision |
| --- | --- | --- | --- |
| Live speed | Reuse measured reference features and source cloud pyramids; avoid LK work for features that can never pass measured-depth checks | Removes repeated computation while retaining the same measured associations and acceptance gates | Implemented; identical-observation benchmarks and scoped caches validated |
| Live speed | [KinectFusion](https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-dense-surface-mapping-tracking/) / [Open3D dense RGB-D SLAM](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/dense_slam.html): projective correspondence against a raycast model | Uses image projection for dense association and keeps reconstruction/tracking data on the GPU, reducing cloud extraction and nearest-neighbour work | Next backend experiment; compare CPU and NVIDIA end-to-end, including raycasting and transfers |
| Live robustness | [RTAB-Map](https://introlab.github.io/rtabmap/): bounded appearance retrieval and RGB-D odometry | Mature keyframe/loop retrieval and alternative pose provider | Compare offline through an adapter before adding its native runtime |
| Live tracking | [DVO](https://jsturm.de/publications/data/kerl13icra.pdf): robust dense RGB-D alignment | Combines image brightness and measured geometry with robust multiscale fitting on CPU | Current hybrid RGB-D fallback already belongs to this family; improve its uncertainty and calibration rather than duplicate it |
| Live tracking / global correction | [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3): keyframes, local bundle adjustment and map reuse | Mature feature-based RGB-D pose and loop baseline | External comparison candidate; C++ dependencies/vocabulary and GPLv3 differ from current Python/Open3D stack |
| Live tracking | [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) / [DPVO](https://github.com/princeton-vl/DPVO): learned dense or sparse patch optimization | Learned matching and iterative pose/depth updates can improve difficult image sequences | Later NVIDIA experiment; model weights, CUDA/custom kernels and RGB-D adaptation need evaluation on Kinect v1 |
| Fusion quality | [Curless–Levoy weighted signed-distance fusion](https://graphics.stanford.edu/papers/volrange/) with [Kinect range/angle uncertainty](https://users.cecs.anu.edu.au/~nguyen/papers/conferences/Nguyen2012-ModelingKinectSensorNoise.pdf) | Repeated reliable measurements refine a common surface; unreliable observations contribute less | Existing foundation; improve noisy angular weights without altering raw depths |
| Fusion quality | Wider inverse-depth plane fits for observation normals | Stabilizes incidence-angle estimates, reducing unjustified confidence loss on noisy planes while keeping discontinuities and holes | Implemented; opt-in confidence path tested for error, coverage and detail |
| Fusion confidence | Calibrated precision accumulation and robust innovation testing | Gives a statistical surface estimate and rejects measurements inconsistent with their uncertainty | Subsequent experiment; needs per-device noise evidence, pose uncertainty, and correlated-observation handling |
| Finish quality | [BAD SLAM](https://openaccess.thecvf.com/content_CVPR_2019/html/Schops_BAD_SLAM_Bundle_Adjusted_Direct_RGB-D_SLAM_CVPR_2019_paper.html)-inspired joint RGB-D bundle adjustment | Jointly adjusts camera positions and shared 3D features with image and metric-depth residuals | Implemented as an optional Finish pass with bounded sparse CPU optimization, separate depth validation, and fresh fusion |
| Finish quality | [BundleFusion](https://graphics.stanford.edu/projects/bundlefusion/): historical sparse/dense constraints and reintegration | Corrects trajectory drift and rebuilds the surface after changing poses | Retain this architectural principle; full online dense global optimization is a larger GPU project |
| Finish correspondence | [COLMAP](https://colmap.github.io/faq.html): multi-image feature tracks and global bundle adjustment | Broader RGB matching and jointly consistent camera/landmark estimates | Adopt multi-view tracks locally; external COLMAP is a later comparison backend |

The first implementation increment targets three bounded changes: cheaper live
motion estimation, more stable observation confidence, and actual joint
camera/landmark refinement at Finish. It does not import the named systems as
new scanning backends. Larger replacements require matched-recording evidence
that justifies their dependency and integration cost.

## What a Kalman filter would change

For a stationary scalar surface quantity, independent unbiased Gaussian
measurements with variances `variance_i` can be accumulated as:

```
precision_i = 1 / variance_i
mean_new = (precision_old * mean_old + precision_i * measurement_i)
           / (precision_old + precision_i)
variance_new = 1 / (precision_old + precision_i)
```

This is the static Kalman measurement update written as a weighted average.
Weighted TSDF fusion already has that averaging structure: a voxel retains its
weighted signed-distance estimate and accumulated observation weight. The
connection to a calibrated Gaussian posterior is an interpretation that holds
only when the measurement/noise assumptions and units match. TSDF truncation,
view-dependent projective distances and engineering confidence weights mean
the current weight is not a certified uncertainty in millimetres.

Independent repeated observations can reduce random noise approximately as
`1 / sqrt(N)`. They do not remove calibration bias, wrong camera positions,
correlated depth errors, or missing detail. Replaying the same image is not
additional independent evidence. A pose-aware posterior would need observation
covariance and camera uncertainty as well as a voxel mean; simply displaying
`1 / weight` as physical variance would be misleading.

The immediate confidence change estimates normals more reliably from measured
depth neighbourhoods. It changes observation weights, not the observed depth
map. It must reject depth discontinuities and preserve unknown pixels. Further
calibrated variance/innovation experiments remain separate so they cannot
silently change reconstruction confidence semantics.

## What joint Finish processing changes

The existing fragment and final refinement passes optimize camera relationships
in a pose graph. A joint bundle pass also gives each shared feature an explicit
3D landmark variable. Every observation connects one camera to one landmark;
features observed in three or more views form multi-view tracks. The optimizer
adjusts cameras and landmarks together to reduce image reprojection and measured
depth residuals. The first camera fixes the coordinate system and calibration
stays fixed.

This is a sparse RGB-D bundle adjustment, not BAD SLAM's dense surfel/direct
implementation. BAD SLAM also demonstrates how synchronization, rolling shutter,
and calibration affect direct RGB-D methods; those issues cannot be repaired by
allowing arbitrary camera/lens changes. Existing RGB/depth timing limits remain.

Proposals must preserve the accepted connected frame set, pass spatial support
and correction bounds, and agree with raw depth excluded from feature fitting.
All output poses, including interpolated captures, require validation. New
positions are committed only after allocation, fresh fusion and model extraction
succeed; the final block budget remains a hard limit.

## Evaluation requirements

- Live: compare median/tail processing time on identical images, exact feature
  associations where applicable, accepted indices, and known-pose synthetic
  trajectory error. Intermediate physical camera frames are absent from stored
  sessions, so archive replay alone cannot validate camera-rate motion.
- Surface: use known planar/oblique geometry with independent noise across
  measurements. Score surface error, completeness, holes, steps and thin details
  together. Keep resolution, extraction threshold, and measurement count equal.
- Finish: measure camera/landmark error on a known synthetic scene, verify
  corruption/degeneracy rejection, and test fusion failure preservation. Saved
  Kinect observations can measure raw-depth consistency and coverage, but do not
  provide independent absolute pose or surface ground truth.

Physical moving-camera validation and NVIDIA execution remain separate acceptance
work. No speed or accuracy claim is inferred from triangle count alone.

## Implemented results and usage

The first three increments are implemented and integrated locally. The speed
changes apply automatically. Enable **Use sensor confidence** for the improved
observation weights, and **Joint RGB-D refinement at Finish** for the new joint
camera/landmark pass. Both quality options remain off by default. Replay tools
accept `--bundle-adjustment`; `profile_session.py` also requires `--finish`.
Install the updated server requirements for SciPy.

| Change | Controlled evidence | Practical limit |
| --- | --- | --- |
| Camera tracking cache and cheaper LK bookkeeping | Whole-update medians 9.54 to 8.91 ms for a moving textured plane, 9.32 to 7.93 ms with depth holes; native stationary Kinect input 18.66 to 17.59 ms. Poses/support/reference choices unchanged. | Sparse archived captures cannot establish camera-rate movement; no physical moving-camera throughput claim. |
| Per-registration ICP source cache | Five model proposals 344.57 to 309.08 ms; five raw-observation proposals 816.19 to 698.13 ms; downsample calls 15 to 3. Correspondences/support identical; legacy pose differences below 4e-16. | Benefit requires repeated proposals; timings exclude acquisition and the rest of fusion. CUDA execution unavailable here. |
| Measured inverse-depth normal confidence | Four noisy front-plane views: axial TSDF RMSE 1.366 to 0.985 mm, coverage 90.7% to 100%. Sixteen-view 10 mm ridges: error 0.892 to 0.754 mm, retained amplitude 9.943 mm. | At 55 degrees with 12 views, surviving-point error improves but coverage falls 92.6% to 83.7%; missing-aware error worsens. Keep opt-in. |
| Joint RGB-D Finish refinement | Actual SIFT on six raycast views: camera translation RMSE 31.61 to 2.63 mm, held-out depth loss down 7.8%. An actual solver-to-Finish regression checks known mesh error falls by more than half without a large coverage loss. | The 120-view chest recording safely declines because selected views lack connected three-view tracks. No demonstrated improvement on that recording. |

Reproducible benchmark reports: [camera tracking](benchmarks/live-tracking-depth-cache.json),
[ICP preparation](benchmarks/icp-source-cache.json),
[confidence fusion](benchmarks/confidence-fusion.json), and
[joint RGB-D refinement](benchmarks/rgbd-bundle-adjustment.json).
See [statistical depth fusion](STATISTICAL_DEPTH_FUSION.md) and
[joint refinement](JOINT_RGBD_REFINEMENT.md) for implementation details and limits.

The strongest next experiments are overlapping local bundle windows for longer
scans, projective model tracking to reduce dense correspondence cost, and
calibrated robust depth accumulation with pose uncertainty. RTAB-Map/COLMAP are
comparison candidates; learned CUDA systems are a later hardware experiment.

Integration validation: the full suite completed 371 tests successfully with
three CUDA cases skipped on this host. The standalone synthetic HTTP/WebSocket/Qt
workflow passed capture, preview, cancellation, Finish, and mesh/session exports.
Concurrent tracking diagnostics were retained and checked with the speed changes.

## Recorded archive evaluation

All four supplied sessions were tested. [Recorded results and reproduction](RECORDED_RGBD_EVALUATION.md) document mixed source-cache timing gains, improved held-out depth coverage from local-plane confidence, saved-budget refusals and separately labelled expanded-budget comparisons, and conservative joint Finish declines. These sensor-consistency measurements use estimated archived poses; they do not establish independent absolute surface accuracy.
