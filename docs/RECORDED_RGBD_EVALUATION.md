# Recorded RGB-D evaluation

The four original session ZIPs in `export/sessions` were evaluated locally on
7 October 2026. Tracking/proposal measurements used the algorithm snapshot at
`fdf144b`, with exact source hashes in the reports. Later concurrent camera
feature-replenishment changes are outside these measurements; rerunning the
commands uses the currently checked-out code. The originals remain inputs; compact reports contain their SHA-256
checksums. Detailed camera/geometry reports and meshes are stored separately under
`/Users/sasha/hobby/xbox360/algorithm-review-20261007/session-evaluation`.

| Recording | Stored captures | Archived accepted views | RGB/depth pairs within 20 ms |
| --- | ---: | ---: | ---: |
| chest_20261005_230320 | 50 | 42 | 5 |
| chest-2scan-session_20261006_090302 | 200 | 123 | 17 |
| chest-3-scan-session_20261006_215208 | 126 | 68 | 126 |
| chest-4-scan-session_20261007_173028 | 164 | 4 | 164 |

Archived accepted poses are estimated camera positions. They provide no
independent absolute camera or surface truth. Missing intermediate camera frames
also prevent these ZIPs from recreating the original continuous live stream.
Timing metadata, calibration, depth units and correspondence gates are retained.
Tracking replay replaces upload frame identity with its local index and removes
any reference-pose scoring field. Historical camera code uses current preparation
helpers; it is not a complete historical application benchmark.

## Comparisons

Camera tracking compares the historical visual tracker at `26fc9ed` with the
current tracker on every saved capture. Measurements exclude image decoding,
the origin frame, and rejected RGB/depth timing pairs. Reports distinguish a
seeded origin from verified motion and compare support, references and poses.

Server tracking compares the current reconstruction pipeline with and without
the per-registration ICP source cache. Other algorithms, settings and recorded
camera hints stay identical. Each session uses the first 24 captures, three
interleaved repeats, short warmups, legacy CPU ICP, and four configured OpenCV/
OpenMP threads. Measurements cover computation and tracking; acquisition,
networking, display and Finish are excluded. Embedded reconstruction poses never initialize this tracking replay; recorded
visual motion hints remain available to both variants. See [tracking report](benchmarks/recorded-tracking-speed.json).

Confidence fusion holds archived accepted poses fixed for both estimators.
Every fifth accepted view is withheld; the four-view recording withholds its
last accepted view. Every remaining accepted observation is fused. Resolution,
final weight threshold, block budget and component filtering match within each
comparison. Held-out raw depth is calibrated without smoothing and selected by a
common complete, stable 3×3 support test independent of confidence weights.
Actual triangle distances and camera-depth rays score both agreement and
coverage; missing geometry contributes to capped losses. This measures sensor
consistency conditional on estimated poses, including their errors.

## Joint Finish proposals

All four bounded proposals retained the original timing and validation gates:

| Recording | Result |
| --- | --- |
| chest / chest-2 | Declined: selected RGB/depth measurements exceed the 20 ms timing limit. |
| chest-3 | Declined: nine selected cameras lack sufficient multi-view tracks, despite 800 landmarks and 4,276 observations. |
| chest-4 | Declined: 131 landmarks and 421 observations fit consistently, but separate depth loss improves only 0.57%, below the required 2%. |

These proposal tests perform no reintegration or mesh replacement. They have
demonstrated no Finish quality improvement on these archived camera estimates.
Overlapping local bundle windows remain the next correspondence candidate for
longer scans; lowering the evidence threshold solely to accept a recording is
not justified. See [proposal report](benchmarks/recorded-bundle-proposals.json).

## Reproduction

```sh
python scripts/evaluate_session_tracking.py export/sessions \
  --baseline-revision 26fc9ed --server-frames 24 --repeats 3 \
  --output docs/benchmarks/recorded-tracking-speed.json

python scripts/benchmark_bundle_adjustment.py \
  --session export/sessions/chest-4-scan-session_20261007_173028.zip \
  --archived-poses --output benchmark-output/chest-4-bundle.json

python scripts/evaluate_session_confidence.py \
  export/sessions/chest-4-scan-session_20261007_173028.zip \
  --output benchmark-output/chest-4-confidence-detail.json \
  --summary-output docs/benchmarks/session-confidence-chest-4.json \
  --mesh-dir benchmark-output/confidence-meshes
```

The confidence evaluator defaults to each archive's resolution, threshold and
block budget. The two older scans exceeded their saved 5,000-block budgets. Those
refusals are retained; separate 10,000-block experiments give both estimators the
same allowance without changing production settings. An insufficient hard allocation budget produces a reported refusal
before candidate fusion. Report destinations must differ from original inputs.


## Measured results

The source-cache server comparison preserved accepted indices in all three
repeats on every prefix. It did not preserve camera matrices bit-for-bit.
Same-seed, same-version controls on Chest 2's first 16 captures already varied by
0.38 mm / 0.018 degrees uncached and 0.26 mm / 0.011 degrees cached. This establishes
run variation; it does not attribute every before/after difference to that cause.
See [repeatability controls](benchmarks/recorded-tracking-repeatability.json).
The 5,000-block live allocation is initial capacity, not a hard memory budget.

| Prefix | Accepted views, both variants | Median processing, uncached → cached |
| --- | ---: | ---: |
| chest | 14 | 1,216.3 → 1,073.3 ms |
| chest-2 | 15 | 1,086.4 → 1,047.4 ms |
| chest-3 | 20 | 830.6 → 848.5 ms |
| chest-4 | 19 | 840.3 → 844.3 ms |

These medians exclude image decoding and the origin frame. Cache savings are
workload-specific: about 12% and 4% in two prefixes, with no improvement in the
other two. The slow recovery tail on Chest 2 was around five seconds. The stored
camera sequences produced no verified motion reports; camera timing/rejection
paths agree, but these sparse images do not establish moving-camera speed.

Held-out ray coverage below is the fraction of stable measured pixels with
reconstructed depth within 10 mm. Capped RMS includes every sampled pixel,
including missing rays, at a maximum penalty of 100 mm. Large losses reflect
missing geometry, occlusion, and archived pose/depth disagreement; they are not
millimetre estimates of absolute scanner accuracy.

| Recording / fusion budget | Training / withheld views | Ray coverage, legacy → local planes | Capped RMS, legacy → local planes |
| --- | ---: | ---: | ---: |
| chest / 10,000 blocks | 34 / 8 | 40.9% → 42.8% | 68.66 → 67.28 mm |
| chest-2 / 10,000 blocks | 99 / 24 | 45.5% → 47.9% | 64.15 → 61.77 mm |
| chest-3 / saved 10,000 blocks | 55 / 13 | 53.7% → 55.7% | 57.46 → 55.20 mm |
| chest-4 / saved 10,000 blocks | 3 / 1 | 0.1% → 3.1% | 99.94 → 98.44 mm |

Triangle-distance coverage and capped loss also improve in these comparisons.
Chest 4 remains severely sparse; its relative increase is not evidence of a
complete final surface. The results support the optional confidence change's
sensor consistency on these observations, while the earlier oblique synthetic
coverage tradeoff remains. No independent surface-detail or calibrated
probabilistic-confidence claim follows from these ZIPs.

Reports: [saved-budget fusion](benchmarks/recorded-confidence-fusion.json) and
[separate expanded-budget fusion](benchmarks/recorded-confidence-fusion-10000-blocks.json).
Reproduce an expanded-budget experiment with `--block-budget 10000`; use a
separate report destination to preserve the saved-budget refusal.

Evaluation checks: 21 focused tests passed, covering raw units/timing, frame
identity, physical pose metrics, true triangle distances, missing-geometry
penalties, disjoint fusion inputs, hard block-budget refusal, and output alias
protection. Original ZIP checksums remained unchanged in all evaluations.

The recorded priorities are overlapping local bundle windows, bounded recovery
search, and projective model tracking. The joint Finish pass has not demonstrated
an improvement on these archived estimates, and source caching is only a small
part of the real recovery cost.
