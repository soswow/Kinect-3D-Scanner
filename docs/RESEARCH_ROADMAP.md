# Research and open-source roadmap for Kinect v1 scanning

The practical route is to improve the existing RGB-D reconstruction pipeline,
preserve raw observations, and evaluate changes on identical recordings. GPU
compute increases throughput; calibration, reliable poses, and sensor-aware
fusion determine geometric fidelity. Texture quality also needs accurate RGB
projection and photometric consistency. Millions of triangles alone do not
establish accuracy.

This is a targeted review of the main relevant research families and integration
candidates, not an exhaustive survey of every 3D-scanning paper. Sources were
checked in October 2026. Recommendations below are engineering assessments for
this project; paper results are not measurements of this implementation.

## Implementation checkpoint, October 2026

This roadmap preserves the original research assessment. Completed work and
later experiments are indexed in [docs/README.md](README.md); priorities here
are engineering judgments, not promises that these integrations exist.

The six ordered software milestones below are implemented in separate commits.
[development history](DEVELOPMENT_HISTORY.md) records completed checkpoints,
acceptance checks and limitations. Hardware-dependent acceptance remains open:
measured per-device calibration, repeated physical accuracy/completeness tests,
and broader matched hardware runs. Later [CUDA](CUDA_EXPERIMENTS.md) and
[field](FIELD_CUDA_RESEARCH.md) studies provide NVIDIA evidence. Appearance loops, confidence fusion, exposure matching
and finer final fusion are available experiments, not established quality gains
for arbitrary Kinect subjects. The difficult desk sequence still lacks a trusted
loop. Turntable capture and patch seam optimization remain subsequent work.

## Papers with direct application

| Work | Useful idea | Application and priority |
|---|---|---|
| [Curless & Levoy, 1996](https://graphics.stanford.edu/papers/volrange/) | Weighted signed-distance fusion; uncertainty and observed/unknown space | Foundation already present through TSDF. Extend confidence weighting before chasing smaller voxels. |
| [KinectFusion, 2011](https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-dense-surface-mapping-tracking/) | Coarse-to-fine tracking against the growing model; GPU reconstruction | Keep model-based tracking. Next GPU optimization should keep cloud construction, normals, and matching on device and consider raycast feedback. |
| [Nguyen, Izadi & Lovell, 2012](https://users.cecs.anu.edu.au/~nguyen/papers/conferences/Nguyen2012-ModelingKinectSensorNoise.pdf) | Kinect axial/lateral noise varies with distance and surface angle | Highest-value fusion experiment: measured uncertainty weights and depth-edge rejection. Calibrate the actual sensor; do not assume published coefficients fit every device. |
| [Khoshelham & Oude Elberink, 2012](https://research.utwente.nl/en/publications/accuracy-and-resolution-of-kinect-depth-data-for-indoor-mapping-a/) | Kinect depth accuracy/resolution and calibration limits | Establish range-dependent error using planar targets and known dimensions. A 2 mm voxel setting is not evidence of 2 mm sensor accuracy. |
| [Zhou & Koltun, 2014: color-map optimization](https://vladlen.info/publications/color-map-optimization-for-3d-reconstruction-with-consumer-depth-cameras/) | Joint photometric pose/image correction | Improve sharpness after geometry is finalized. Preserve original RGB, calibration, and poses; measure reprojection consistency before applying warps. |
| [Choi, Zhou & Koltun, 2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Choi_Robust_Reconstruction_of_2015_CVPR_paper.html) | Fragment registration and robust global optimization that rejects bad constraints | [Bounded offline fragment reconnection](FRAGMENT_RECONNECTION.md) now searches independently of estimated live positions, verifies bridges, optimizes the anchored graph, and freshly fuses connected views. Broader real-scan evaluation and background recovery remain future work. |
| [Park, Zhou & Koltun, 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf) | Joint color/geometric registration constrains motion along surfaces | Test as a recovery or verification signal. Color-assisted RGB-D initialization already exists; it is not a full implementation of this paper. Require synchronization, texture, and geometric checks. |
| [BundleFusion, 2017](https://arxiv.org/abs/1604.01093) | Global pose optimization plus surface reintegration | The implemented final refinement follows the architectural principle of rebuilding fusion after pose correction. It is a bounded offline graph, not BundleFusion's online global system. |
| [Waechter, Moehrle & Goesele, 2014](https://download.hrz.tu-darmstadt.de/pub/FB20/GCC/paper/Waechter-2014-LTB.pdf) | View selection and seam correction for texture patches | Next texture stage after depth-tested projection: exposure consistency, best-view patches, and seam leveling. |
| [BAD SLAM, 2019](https://github.com/ETH3D/badslam) | Direct RGB-D bundle adjustment and calibrated camera/depth modeling | Useful reference for pose/calibration architecture. Its authors caution that lower-quality RGB-D can perform poorly, so replacement requires Kinect v1 evaluation. |

The current implementation uses Open3D algorithms and independently implemented
bounded validation/projection logic. No code from a noncommercial research
release has been copied into the project.

Additional candidates from the later RGB-D plan remain useful comparisons:
[DVO](https://jsturm.de/publications/data/kerl13icra.pdf) for robust dense CPU
alignment, [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) for keyframe/local
bundle/map reuse (native dependencies, vocabulary and GPLv3), and
[DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) or
[DPVO](https://github.com/princeton-vl/DPVO) for later learned CUDA experiments.
They are not imported scanner backends; model/operator requirements and Kinect
RGB-D adaptation need separate evaluation. Current joint sparse refinement and
confidence semantics are in [joint refinement](JOINT_RGBD_REFINEMENT.md) and
[depth confidence](STATISTICAL_DEPTH_FUSION.md).

## Open-source integration candidates

Licenses below describe the main project, not every dependency or dataset. Pin
the revision and inspect the complete dependency graph before importing code.

| Project | Main license | Best role here | Integration cost |
|---|---|---|---|
| [Open3D](https://github.com/isl-org/Open3D), [license](https://raw.githubusercontent.com/isl-org/Open3D/main/LICENSE) | MIT | Existing TSDF, tensor ICP, pose graph, UV/baking substrate | Low; use current dependency first |
| [RTAB-Map](https://github.com/introlab/rtabmap), [license](https://raw.githubusercontent.com/introlab/rtabmap/master/LICENSE) | BSD 3-clause | Verified RGB-D relocalization/loop closure, keyframe management; optional offline pose provider | Medium/high; adapter and build/runtime dependencies |
| [mvs-texturing](https://github.com/nmoehrle/mvs-texturing) | BSD 3-clause | Final mesh texture view selection and seam leveling | Medium; export calibrated camera/image/mesh inputs and build C++ tools |
| [Cupoch](https://github.com/neka-nat/cupoch) | MIT | Alternative GPU registration/features/cloud processing | Medium; another geometry runtime and conversion costs |
| [nvblox](https://github.com/nvidia-isaac/nvblox), [license](https://raw.githubusercontent.com/nvidia-isaac/nvblox/public/LICENSE.md) | Apache 2.0; some BSD components | Specialized NVIDIA TSDF backend after profiling establishes a need | High; native dependency and integration interface. Mapping is not a complete pose tracker. |
| [BAD SLAM](https://github.com/ETH3D/badslam) | BSD 3-clause | Alternative calibrated RGB-D pose/scene backend for comparison | High; NVIDIA platform assumptions and demanding capture quality |
| [COLMAP](https://github.com/colmap/colmap) | BSD 3-clause | Optional offline RGB photogrammetry/pose refinement experiment | High; not a Kinect RGB-D scanner replacement without adapters |
| [AliceVision](https://github.com/alicevision/AliceVision) | MPL 2.0; some MIT components | Offline photogrammetry and texture-pipeline reference | High; substantial native toolchain |
| [Redwood reconstruction release](https://www.redwood-data.org/indoor/pipeline.html) | MIT | Fragment registration/global optimization reference and benchmarks | Medium/high; Open3D provides a more convenient current substrate |
| [OpenMVS](https://github.com/cdcseacave/openMVS), [license](https://raw.githubusercontent.com/cdcseacave/openMVS/master/LICENSE) | AGPL v3 | Offline dense reconstruction/texturing option | High; copyleft is different from the permissive candidates |

[BundleFusion's released code](https://github.com/niessner/BundleFusion/blob/master/LICENSE.txt)
uses CC BY-NC-SA 4.0. It is useful research material but does not satisfy the
request for fully open-source integration without a noncommercial restriction.
The same distinction must be checked for other downloadable academic scanners:
public source availability alone is insufficient. The BSD mvs-texturing project
is a better first external texturing candidate.

Neural implicit mapping and Gaussian splatting are a separate experiment. They
can improve an appearance representation while adding training/runtime costs,
specialized GPU dependencies, and sometimes restrictive subcomponent licenses.
A visually convincing rendering does not establish a dimensionally accurate,
watertight mesh. Keep them outside the primary Kinect scanning path until they
beat the conventional pipeline on held-out geometry, not just novel-view images.

## Plan with measurable acceptance criteria

1. **Establish per-device evidence.** Record the same stationary, textured object
   and camera motion. Calibrate RGB intrinsics in registered-depth image space,
   verify depth units/scale, and measure pairing lag. Compare known dimensions,
   plane residuals, edge preservation, completeness, and repeated-scan variation.
   Retain raw frames and rejection diagnostics. This requires physical hardware.
2. **Profile the NVIDIA path.** Run identical CPU and CUDA recordings and compare
   accepted poses, surface metrics, stage times, memory, and upload/backlog age.
   Move cloud construction/normals/downsampling onto tensors only if profiling
   shows it improves end-to-end throughput. Current code still performs these
   stages on CPU; CUDA fusion alone cannot remove that bottleneck.
3. **Strengthen global tracking.** Use appearance/feature retrieval to propose
   distant loops and relocalization after lost tracking. Geometrically verify
   candidates in both directions, use robust graph constraints, validate on
   separate observations, and fuse corrected frames into a new volume. Compare
   RTAB-Map as a pose provider before writing an entire SLAM system. Current final
   refinement is an initial bounded implementation with a safe retain-original
   outcome, and does not fix the difficult desk replay.
4. **Improve sensor-aware fusion.** Introduce depth-edge, range, and angle-aware
   confidence through a custom integration kernel or compatible backend. Test
   thin surfaces and corners with controlled noise. Preserve unknown regions;
   indiscriminate smoothing or hole filling can improve appearance while losing
   measured detail. Compare mesh-to-reference distances and completeness together.
5. **Improve final appearance.** With geometry/poses fixed, compare current
   measured-image projection, Open3D color-map optimization, and mvs-texturing.
   Measure reprojection error, seam contrast, sharpness, observed-texture coverage,
   and export interoperability. Keep an explicit fused-color fallback for areas
   with no trustworthy view. Use real measured images for scan fidelity.
6. **Separate live and final budgets.** Keep a bounded live cloud or raycast view
   for responsive feedback. Use offline refinement, higher confidence, optional
   finer fusion, and texture generation for the final asset. Show backlog and
   reconstruction confidence so a user can slow down or revisit missing areas.

The first three priorities address the dominant risks: uncalibrated geometry,
tracking drift, and processing latency. Seamless texture is valuable once the
surface and camera poses agree; it cannot repair inconsistent geometry.
