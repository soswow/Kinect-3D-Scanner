# Joint RGB-D camera and feature refinement

`bundle_adjustment` enables an optional Finish step after existing fragment
reconnection and pose-graph refinement. It simultaneously adjusts camera poses
and common 3D feature positions. It is disabled by default.

A SIFT feature must be supported by measured depth and match consistently in at
least three selected views. Mutual descriptor matches are checked with metric
RGB-D pose estimates. Identity cycles that contain different features from the
same image are discarded. Each selected camera needs at least 24 spatially
distributed observations and connection to the anchored first camera.

The sparse solver minimizes robust image reprojection and measured axial-depth
errors. Calibration and the first accepted camera stay fixed. Depth residuals
use `0.002 + 0.002*z*z` metres as an engineering range-dependent weighting; this
is not a calibrated sensor uncertainty or confidence estimate. Optimized sparse
landmarks constrain camera positions; they never replace recorded depth, fill
holes, or become synthetic fusion observations.

Every proposed camera position, including positions interpolated between
selected cameras, must pass independent measured-depth validation. Its mean
loss must improve by 2%, with no comparison worsening by more than the existing
10% plus saturated-loss allowance. Validation
samples avoid every extracted feature's depth patch, use symmetric nearest-point
checks, count lost overlap with saturated distance, and reject worsening pairs.
Only then does the server allocate a fresh volume within the configured final
block budget and re-fuse recorded observations. A failure preserves poses and
geometry; a successful fusion commits them together.

The solver bounds are 24 selected cameras, 800 landmarks, 96 matching pairs,
60 solver evaluations, and a 45-second proposal budget including depth
validation. There is no total accepted-view cutoff. Validation loads at most
eight sampled clouds into its cache, with at most 2,000 points per cloud, and
processes all adjacent camera pairs plus verified keyframe pairs. Full depth
images are prepared on demand rather than retained for every view. Losing a
late view's depth, exceeding the budget, or failing any validation gate rejects
the entire proposal; validation is never truncated to fit a recording limit.
The report records coverage, peak cache residency, stage, and failed pairs.

Keyframe selection reserves temporal coverage and includes exact camera pairs
from committed fragment bridges when capacity permits. These are retrieval
hints; saved transforms never become camera/landmark constraints. Every feature
identity is remeasured. Unsupported cameras are reported by both keyframe
position and stored capture index, with track-count, spatial-support, or
connectivity failure distinguished. Later corrections must also preserve fresh
raw visual and held-out checks of retained temporal/storage boundaries.

Camera corrections are limited to 25 cm and 20 degrees. Timed RGB-D
pairs must satisfy the existing 20 ms assistance limit; legacy recordings with
missing timing metadata retain the existing compatibility convention. Missing
or insufficient evidence produces a reported refusal.

## Research candidates and choice

[BAD SLAM](https://openaccess.thecvf.com/content_CVPR_2019/papers/Schops_BAD_SLAM_Bundle_Adjusted_Direct_RGB-D_SLAM_CVPR_2019_paper.pdf)
provides direct joint RGB-D map/trajectory optimization and highlights sensitivity
to synchronization and calibration. Its published dense GPU implementation is
not incorporated here; the [official implementation](https://github.com/ETH3D/badslam)
requires CUDA and targets Linux/Windows.

[BundleFusion](https://graphics.stanford.edu/projects/bundlefusion/) combines
historical sparse and dense correspondence with global camera refinement and
surface reintegration. We use its architectural lesson: camera corrections
require rebuilding the fused surface from observations.

[COLMAP's official guidance](https://colmap.github.io/faq.html) describes feature
tracks, retaining known calibration, and final bundle adjustment after model
merging. Our sparse solver follows those principles and adds Kinect's measured
metric depth. It is a genuine camera/landmark solve rather than a renamed pose
graph. [RTAB-Map's parameter reference](https://introlab.github.io/rtabmap/api/latest/parameters.html)
exposes bounded local bundle adjustment, a useful next candidate for larger scans.

## Validation and limits

`tests/test_bundle_adjustment.py` covers noisy camera/landmark recovery, corrupt
depth, nonidentity world anchors, ambiguous track identities, repeated/blank RGB,
missing depth, timing gates, omitted-camera depth rejection, world-origin
invariance, landmark budgeting, and timeout during validation. Separate Finish
integration tests exercise allocation budgets, rollback, caching, and new captures.

The reproducible read-only benchmark is:

```sh
python scripts/benchmark_bundle_adjustment.py \
  --session path/to/recording.zip \
  --poses path/to/connected/reconstruction.json \
  --output docs/benchmarks/rgbd-bundle-adjustment.json
```

Omit `--session` and `--poses` to run only synthetic cases. The numerical test
simultaneously recovers noisy cameras and landmarks; the raycast case performs
actual SIFT extraction, matching, optimization, and held-out depth validation.
Known truth creates deliberately perturbed synthetic inputs and scores output;
archived camera estimates initialize the optional recorded test. Benchmark JSON
records solver/source versions and checksums.

On the six-view raycast case, camera translation RMSE fell from 31.61 mm to
2.63 mm and held-out depth loss fell 7.8%. This establishes recovery in that
controlled scene, not universal surface-detail gains. The established 120-view
chest reconstruction produced 61 verified feature pairs but insufficient
three-view tracks for eight selected cameras, so the proposal was safely refused
in about 2.24 seconds. That recording has no independent camera/surface truth.

The next useful improvement is overlapping local windows or chunked bundle
adjustment, keeping denser temporal features while connecting windows globally.
Other candidates are independently calibrated depth-noise weighting and dense
direct residuals once synchronization/calibration support them. Sparse depth
validation is deliberately conservative: sample spacing can dominate millimetre
corrections on distant or flat surfaces. No fine-detail improvement is claimed
without recorded/device validation.
