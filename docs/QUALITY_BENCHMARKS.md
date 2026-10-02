# Repeated quality comparisons without new hardware

This benchmark uses downloaded RGB-D recordings to check reconstructed surfaces
against **withheld sensor views**. It complements camera-trajectory scores and
synthetic exact-surface tests. It does not replace an independent laser scan or
measured calibration for an individual Kinect.

## Procedure

```bash
OMP_NUM_THREADS=4 KINECT_BLOCK_COUNT=5000 python scripts/benchmark_quality.py \
  --dataset tum --path datasets/rgbd_dataset_freiburg1_xyz \
  --stride 10 --repeats 3 --variants uniform confidence \
  --output benchmark-output/xyz-heldout.json
```

Training uses associated captures 0,10,20,…; scoring uses 5,15,25,… . Reader
association happens before selection, with each depth image used at most once.
The held-out RGB/depth frames are never stored in the reconstruction engine.
Reference camera poses are used only by the evaluator and trajectory scorer.
The first accepted camera with a reference pose defines one rigid coordinate
anchor; no trajectory-wide alignment, scale fitting or ICP against references
is allowed. If reconstruction produces no surface, missing coverage is scored
explicitly rather than excluding that run.

Each variant runs in a separate process. Variants are interleaved by seed and
run sequentially. The report records seeds, versions, source/input hashes,
settings, acceptance, failed builds, rejection messages and per-view counts.
Min/median/max across repeats make instability visible. Seeds do not guarantee
bitwise deterministic Open3D/hash-map results. Completed runs are saved before
starting the next, so interruption does not discard evidence.

## Metrics and interpretation

The evaluator raycasts the finalized mesh from held-out reference cameras and
compares axial depth to measured pixels (default every fourth pixel, 10 mm
tolerance). Reference samples retain sensor noise and missing depth; they use
lens/scale correction and the same near/far/ROI limits, without the reconstruction
filter. Scores are pixel-weighted, not area-weighted.

- **Hit fraction:** fraction of valid measured pixels with any rendered surface.
- **Depth RMSE/p95 over hits:** residuals where both depths exist. These must be
  read with coverage: deleting geometry can reduce these residuals.
- **Precision within tolerance:** agreement among rendered hits at valid measured
  pixels; this is not whole-mesh precision.
- **Completeness within tolerance:** agreement divided by all valid measured
  pixels, including missing rendered geometry.
- **Capped RMSE including missing:** residuals capped at 100 mm, with each
  missing rendered surface contributing the full 100 mm penalty. Uncapped
  overlap RMSE is retained to expose large errors.

These metrics combine pose drift, fusion error, calibration and reference-pose
association error. Held-out depth shares the same sensor's systematic errors.
TUM inputs use its recommended approximate registered-grid intrinsics; reference
poses are nearest associations within 20 ms. Do not call the result absolute
mesh accuracy, independent ground truth, or whole-object completeness. Geometry
projected into pixels without valid reference depth cannot be judged here.

A Redwood three-training/two-withheld-view smoke run exercises both successful
uniform extraction and absent confidence-weighted extraction at the default
final weight. Fractional weights require additional observations; changing that
threshold is a separate setting, not an automatic quality improvement.

The source dataset and registered-depth conventions are documented by
[TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).
Independent exact rendered geometry is available in
[Augmented ICL-NUIM](https://www.open3d.org/docs/release/python_api/open3d.data.RedwoodIndoorLivingRoom1.html),
which is synthetic and should be labeled separately from real Kinect recordings.
