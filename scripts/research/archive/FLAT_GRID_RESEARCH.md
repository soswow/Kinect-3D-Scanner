# Separate flat full-radius grid experiment

This is standalone research. No production module selects it. The shader is
the frozen `research_flat_grid_nn.cu`: 27 parallel cell lookups followed by a
shared exclusive prefix over disjoint ranges. Each lane visits global ordinal
`tid + 128*k`; upper_bound skips empty ranges. Every candidate retains its
original double coordinates, original index and explicit non-FMA FP64 distance.
The original distinct two minima and five-column output remain unchanged.

The bounded real-query ablation found a promising kernel comparison. That
sampled result does not establish full-bridge, scanner FPS or Finish speed.
The inherited Python/CPU ICP work and transfers remain measurable costs.

`FlatUniformGridICP` overrides construction only and inherits the immutable
original grid cache, raw argument/output contract, host/device adapters,
complete CPU tie/boundary/unsupported resolver, ICP estimator and convergence.
Its construction includes the inherited serial kernel setup and replacement
flat compilation; both are recorded. Retained index arrays are bounded by
64 clouds/256 MiB. Transient arrays, CuPy pools and process RSS are reported
separately; this cap is not a total process/device memory cap. Focused proof
also checks cloud eviction, cache clear, and original CPU fallback under a
one-byte index budget. Cache ownership ends at solver close.

Old serial/staged/OptiX proofs cannot authorize flat direct misses. A distinct
`FlatGridProofAuthority` requires separate fresh flat synthetic host/device
proof and the accepted/rejected original nine-proposal CPU hit-and-miss audit.
Bindings include original inherited source/validator, flat shader/adapter/
producer/validator/timing driver/document, frozen core and numerical source,
raw archive/fixture/reference/proposal bytes, domain, installed CPU/CUDA library
binaries/configuration, actual CUDA device/driver, thread and cache policies.
Ambiguous/tied/boundary/unsupported queries always use the original CPU tree.
Direct misses are a scoped research policy. Proof files are immutable outputs.

Run only after the parent allocates exclusive hardware. Commands use the
original Open3D20/OpenCV20/OMP8 policy. All output paths must be fresh.

```powershell
$env:OMP_NUM_THREADS='8'
$py='python' # Use the active CUDA-enabled Python environment.
$out='benchmark-output/cuda-pipeline/uniform-grid-nearest/flat-v1'
& $py scripts/research/archive/benchmark_flat_grid_nn.py --run-allocated --synthetic-only --check-device-adapter --audit-nearest --audit-misses --miss-policy direct-miss-research-v1 --repeats 1 --output "$out/synthetic-proof.json"
& $py scripts/research/archive/benchmark_flat_grid_nn.py --run-allocated --check-device-adapter --audit-nearest --audit-misses --miss-policy direct-miss-research-v1 --repeats 1 --output "$out/bridge-audit.json"
& $py scripts/research/archive/benchmark_flat_grid_timing.py --run-allocated --miss-policy direct-miss-research-v1 --proof-synthetic "$out/synthetic-proof.json" --proof-bridge "$out/bridge-audit.json" --repeats 2 --output "$out/timing.json"
```

The timing driver validates both proofs and current source/runtime/input before
omitting redundant synthetic setup. Original native versus flat all-nine
proposal execution, evidence/witness/pose/information comparisons, final
array/input/core/artifact guards and cache cleanup still execute. CPU audits
are absent only in this separately authorized timing phase. Setup/index builds
and first/warm full-bridge wall times are retained. Inclusive NN/dataset/ICP
timers overlap and must be subtracted when constructing a disjoint partition.

A future GPU-resident adapter changes query classification, ownership and
trajectory. This initial roundtrip host/device proof cannot authorize those
new methods. That experiment requires separate source binding, original CPU
hit/miss shadow audit on the new trajectory, all-nine gate proof and timing.
No complete archive replay or production promotion follows from a kernel-only
comparison. At preparation time the flat full-bridge experiment is unexecuted.
