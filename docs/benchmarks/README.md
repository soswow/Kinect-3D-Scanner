# Published benchmark evidence

These are historical observations at recorded source/input/environment
fingerprints. Read a focused guide first, then the relevant evidence; loading
every JSON file is unnecessary. Fixed-coordinate agreement, held-out sensor
consistency and synthetic truth have different scopes. They do not all measure
physical accuracy, camera FPS or whole-session speed. Private captures and full
proof artifacts remain local.

| Question | Evidence and interpretation |
|---|---|
| What happened in a recorded scan? | `chest-3-*`, `chest-4-*`, `chest8-loop-refinement.json`, `chest-markers-apriltag-*.json`; see the [documentation index](../README.md) for their investigation notes |
| Did capture or tracking work improve? | `capture-profile.json`, `backend-profile.json`, `adaptive-feature-tracking.json`, `archived-adaptive-feature-tracking.json`, `live-tracking-depth-cache.json`, `recorded-tracking-{speed,repeatability}.json`, `icp-source-cache.json`, `twenty-frame-flow-trails.json`; read [capture performance](../CAPTURE_PERFORMANCE.md) |
| What supports the original feature/quality decisions? | `feature-summary.json`, `rgbd-summary.json`, `{desk-supported,xyz}-heldout-summary.json`, `algorithm-review.json`, `milestone-validation.json`; see [quality benchmarks](../QUALITY_BENCHMARKS.md) and [development history](../DEVELOPMENT_HISTORY.md) |
| What was tested for confidence and pose proposals? | `confidence-fusion.json`, `recorded-confidence-fusion*.json`, `rgbd-bundle-adjustment.json`, `recorded-bundle-proposals.json`; read [depth fusion](../STATISTICAL_DEPTH_FUSION.md) and [joint refinement](../JOINT_RGBD_REFINEMENT.md) |
| What supports sensor and native preparation behavior? | `accelerometer-{probe,implementation}.json`, `native-performance.json`; read [accelerometer](../KINECT_ACCELEROMETER.md) and [native performance](../NATIVE_PERFORMANCE.md) |
| What was measured for CUDA? | [CUDA result index](CUDA_REPORTS.md), `cuda-session-performance.json`, [initial fusion report](cuda-study/REPORT.md), [pipeline report](cuda-pipeline/REPORT.md) and component summaries under `cuda-pipeline/` |
| What changed in the field studies? | [Field summary](field-study-v1/research-summary.json), [Final allocation validation](field-study-production-v1/validation-summary.json); read [field interpretation](../FIELD_CUDA_RESEARCH.md) |
| What did GPU ICP/reuse experiments establish? | Scalar receipts under `field-study-gpu-icp-v1/`; [whole-Finish limits](../../scripts/research/GPU_ICP_CURRENT_FINISH_RESULTS.md) and [experiment family index](../../scripts/research/README.md) distinguish component timing from failed whole-Finish quality |
| What supports depth-only reconstruction? | [Offline depth qualification](offline-depth-registration.json); read [full-room findings](../FULL_ROOM_SCAN_20261010.md) |

## Full historical run matrix

The original `cuda-pipeline-experiments.json` contained 123,553 lines and about
4.2 MB of repeated run/component/quality data. Its full contents remain available
at [the reviewed historical commit](https://github.com/soswow/Kinect-3D-Scanner/blob/762a6dac3b5a48b5865d382bbf18cd1746c37999/docs/benchmarks/cuda-pipeline-experiments.json).
The pipeline report, charts, scoped component summaries and validation receipts
remain in this checkout. Original LF-content SHA-256:
`eaaa64cb85fac3460f96f8a7249f244755d759029a73c72f814975b5fb20a151`.
Use that Git record when exact old detail is needed; do not recreate the full
matrix in `docs/` as a default output.

## Reproduction and publication

Use [script workflows](../../scripts/README.md) with explicit local inputs.
Full generated profiles/matrices go to ignored `benchmark-output/`. Publish only
reviewed bounded summaries/charts with the source, inputs, environment, timing
scope and failures needed to assess their claims. Existing historical report
manifests and fingerprints stay unchanged; they do not authorize a new adapter
or describe a currently running server. See [research protocols](../../scripts/research/README.md)
for locally rebuilt fixtures/native libraries and fresh proof requirements.
