This is a standalone nearest-neighbour experiment. The scanner does not select
it through any public setting. Offline compilation, the CPU enclosure check,
synthetic host/device index parity and one accepted/rejected real-pair audit
have passed for the conservative version. Its complete CPU-loop bridge is
slower than native CPU ICP; no production speed or quality claim follows.

NVIDIA publishes the driver API headers in [optix-dev](https://github.com/NVIDIA/optix-dev).
The [v9.0.0 release](https://github.com/NVIDIA/optix-dev/releases/tag/v9.0.0)
requires an R570 or newer driver. The experiment uses installed driver 610.88,
CUDA toolkit 12.4, and the original
[NVIDIA SDK license](https://github.com/NVIDIA/optix-dev/blob/v9.0.0/LICENSE.txt).
Public fixed-tag headers were retrieved without an account or interactive
download step; the license remains with the ignored local dependency.

Build and check the float enclosure without creating a GPU context:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/research/archive/build_optix_nn.ps1 -FetchHeaders
python scripts/research/archive/check_optix_enclosure.py
```

The build helper downloads the fixed v9.0.0 ZIP and checks SHA-256
`461664fda99a901d364996bf048a3dae399ce1162c725e5c1855c336c6c9d0a8`.
It uses the local Visual Studio 2022 C++ tools and compiles PTX for compute_86
with fused multiply-add disabled. `-CudaRoot` and `-VcVars` override tool paths.

After hardware is available, first test individual original CPU nearest indices:

```powershell
python scripts/research/archive/benchmark_optix_nn.py --synthetic-only --check-device-adapter --bins 1 --output benchmark-output/cuda-pipeline/optix-nearest/synthetic.json
```

Then compare all competing proposals for one accepted and one rejected real
archive fixture pair, including reciprocal, independent-camera, held-out,
normal-diversity and information gates. The audit checks every direct RTX hit
against the original CPU tree. Its CPU shadow queries confound performance:

```powershell
python scripts/research/archive/benchmark_optix_nn.py --pairs 2 --repeats 1 --bins 1 --audit-nearest --check-device-adapter --output benchmark-output/cuda-pipeline/optix-nearest/real-pairs-audit.json
```

A separate timing run omits the audit, retaining identical fallback rules:

```powershell
python scripts/research/archive/benchmark_optix_nn.py --pairs 2 --repeats 1 --bins 1 --output benchmark-output/cuda-pipeline/optix-nearest/real-pairs.json
```

`--bins .25 .5 1` prepares smaller-to-larger radius indexes. A nearest point
strictly inside a smaller sphere is also globally nearest within the complete
original radius. Ties, near-boundary results, missing candidates and unsupported
coordinate regimes use the original complete-radius CPU tree. All original
double coordinates, CPU Huber point-to-plane estimator, pose updates, radius
stages and convergence remain. Conservative float AABBs only retrieve candidates.
The separately tested `nearest_device` adapter returns resolved device indices
and squared distances for future resident-iteration research. Full bridge timing
continues to use the original CPU iteration loop.

The original conservative artifacts and reports are preserved under
`benchmark-output/cuda-pipeline/optix-nearest/conservative-v1/manifest.json`.
Its real audit shadowed 58,809,404 direct RTX hits with zero index mismatches;
45,314,585 complete-radius misses were resolved by the original CPU tree,
which found no omitted neighbours. All nine competing proposal witnesses and
the accepted/rejected pair decisions agreed. These are fixed-component results.

Separate unaudited timing measured accepted pair `[8,12]` at 2.979 s native CPU
versus 30.539 s RTX bridge, and rejected pair `[0,2]` at 8.163 s versus
142.836 s. The RTX bridge spent 125.630 s in Python CPU miss fallback and
36.155 s in RTX search across the two pairs. The hit-shadow audit is excluded
from those timings. No second slow repeat was performed.

An experimental `--miss-policy direct-miss-research-v1` skips the CPU loop only
for complete-radius empty results within the explicitly recorded enclosure
domain. It retains CPU tie, boundary and unsupported-domain fallback. Every
new source/policy/bin combination needs fresh synthetic host/device parity and
real accepted/rejected **hit and miss** CPU shadow proofs before unaudited
timing. The new source hashes cannot reuse the conservative proof:

```powershell
python scripts/research/archive/benchmark_optix_nn.py --synthetic-only --check-device-adapter --bins .25 .5 1 --miss-policy direct-miss-research-v1 --audit-nearest --audit-misses --output benchmark-output/cuda-pipeline/optix-nearest/synthetic-direct-staged.json
python scripts/research/archive/benchmark_optix_nn.py --pairs 2 --repeats 1 --check-device-adapter --bins .25 .5 1 --miss-policy direct-miss-research-v1 --audit-nearest --audit-misses --output benchmark-output/cuda-pipeline/optix-nearest/real-pairs-direct-staged-audit.json
```

The direct-miss/staged policy passed those fresh proofs: 58,809,403 direct hits
and 45,314,585 declared misses were each CPU shadowed, with zero changed
indices or false misses. One small-bin uncertainty retained CPU search. All
nine proposal witnesses and ordered pair decisions agreed. Candidate visits
fell from 6.229 billion to 909.288 million for the same 104,123,989 query rows.

Separate two-repeat timings still favour original native ICP:

| Fixed pair | Native CPU | RTX first pass | RTX cached repeat |
| --- | ---: | ---: | ---: |
| Accepted `[8,12]` | 2.850 s | 6.950 s | 8.205 s |
| Rejected `[0,2]` | 8.391 s | 23.858 s | 23.506 s |
| Combined | 11.240 s | 30.809 s | 31.711 s |

The RTX bridge remains 2.74–2.82 times slower. Setup was 0.165 s, separate
from those calls; the first pass includes newly needed target indexes.
Retained cache peaked at 94.0 MiB and BVH buffers at 53.3 MiB. First/cached
RTX search accounted for 20.618/22.046 s; other CPU correspondence-array and
metric work accounted for about 5.9 s. The remaining original CPU estimator,
transform and Python-loop residual was 1.3–1.8 s. The slower cached accepted
repeat is retained; index reuse alone did not produce a timing improvement.
These timers describe this bridge, not the opaque original native ICP internals.

Raw proofs/timings stay immutable, and `research-summary.json` records all three
negative results with report hashes. Regenerate that summary without CUDA:

```powershell
python scripts/research/archive/summarize_optix_research.py
```

CPU shadowing explicitly invalidates speed attribution. Retained domain metadata and every source,
PTX/DLL, original CPU adapter, fixture and raw archive hash must agree before
the separate resident-iteration experiment can consume a proof.

The `staged-v1` prototype is built and tested in separate files. It moves all
radius-bin traversal into one OptiX raygen launch and stores best/second FP64
distances, distinct IDs and a 64-bit visit counter in eight payload registers.
It retains the same double-distance, CPU ambiguity and experimental missing-query
policies. NVIDIA documents payload writes as visible to subsequent tracing
programs in the [OptiX programming guide](https://raytracing-docs.nvidia.com/optix9/guide/optix_guide.250130.A4.pdf).
Fresh synthetic host/device parity and real accepted/rejected every-hit and
every-miss CPU audits passed, preserving the same 104,123,989 real query rows,
58,809,403 direct hits, 45,314,585 declared misses, 909,287,750 candidate visits
and all nine proposal witnesses/decisions. Separate first/cached full-bridge
calls took 21.547/21.686 s versus native CPU 11.070 s, still 1.95–1.96 times
slower. Setup was 0.150 s. Search used 14.438/15.247 s, other correspondence
work about 2.81 s and original CPU estimator/transform/Python-loop residual
about 1.72–1.79 s. It used 11,569 staged launches per real pass. No production
promotion follows this improvement over the earlier research implementation.
The preceding proven sources/DLL/PTX stay unchanged for resident research:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/research/archive/build_optix_staged.ps1
python scripts/research/archive/benchmark_optix_staged.py --synthetic-only --check-device-adapter --bins .25 .5 1 --miss-policy direct-miss-research-v1 --audit-nearest --audit-misses --output benchmark-output/cuda-pipeline/optix-nearest/staged-v1/synthetic.json
python scripts/research/archive/benchmark_optix_staged.py --pairs 2 --repeats 1 --check-device-adapter --bins .25 .5 1 --miss-policy direct-miss-research-v1 --audit-nearest --audit-misses --output benchmark-output/cuda-pipeline/optix-nearest/staged-v1/real-pairs-audit.json
python scripts/research/archive/benchmark_optix_staged.py --pairs 2 --repeats 2 --bins .25 .5 1 --miss-policy direct-miss-research-v1 --output benchmark-output/cuda-pipeline/optix-nearest/staged-v1/real-pairs.json
```

Build and all numerical execution still require the coordinator's exclusive
hardware slot; do not run these commands during matched scanner measurements.

Reports retain exact per-proposal evidence and ordered pair acceptance/ambiguity
decisions. They include module/driver setup, fixture loading, first and warm full
bridge times, hashing/index/CPU-fallback costs, candidate visits, raw RTX misses
that the CPU says have neighbours, and numerical/source/script/binary/input
hash checks. Additive ICP timers separate dataset and correspondence costs from
the remaining CPU estimator, point transformation and Python loop overhead;
they do not expose the original native C++ ICP's internal timing.

The cache retains at most 64 target clouds and 512 MiB of point, AABB and BVH
buffers. An oversized target falls back to CPU. One new scene, temporary build
workspace, query buffers and CuPy allocator pools can exceed that retained cap;
reported BVH and cache peaks do not represent total device usage.

The fixture contains original raw ZIP/calibration-derived point data and measured
Finish fragment-local poses. Those poses only reproduce the component's inputs.
The experiment does not run live tracking, discover the current graph frontier,
perform optimized-graph revalidation, fuse a fresh mesh, or establish full Finish
speed or mesh quality. Component success cannot promote a production default.
