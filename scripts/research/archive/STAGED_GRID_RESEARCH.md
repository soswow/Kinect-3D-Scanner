# Separate staged grid with conservative FP32 screening

This experiment is source-only until an exclusive hardware slot and independent
reviews permit execution. Production core `9331…` and the measured full-radius
grid files/proofs stay unchanged. Canonical outputs belong under
`benchmark-output/cuda-pipeline/uniform-grid-nearest/staged-pruned-v1/`:
`synthetic-proof.json`, `bridge-audit.json`, `timing.json`, and the derived
`research-summary.json`. No production selector imports this prototype.

The original CPU estimator, transformed double coordinates, target normals,
three ICP radii/iteration limits, updates and all original bridge authorizers
remain authoritative. Retrieval uses fractions `[0.25, 0.5, 1]` of each original
radius and one kernel launch/one result transfer per nearest-neighbor call.
An inner stage is skipped if below the original `2^-20` radius domain or if its
signed-cell domain is unsupported. The final stage uses the actual original
radius. Missing final-grid ownership or budget bypass yields `-3`, forcing the
original full-radius CPU tree; an incomplete narrow-stage scan cannot declare
a complete miss. Unsupported-query branches have a uniform block barrier before
the next stage changes shared flags/ranges.

Each stage uses the existing dyadic-width enclosure: width is the next power of
two at least as large as the stage radius; original finite coordinates undergo
reversible `ldexp`, floors lie in signed 21-bit cells, and injective packed keys
cover every possible strict-radius hit within 27 adjacent cells. Actual stage
enclosure checks include radii `0.0075` and `0.015`, the original radii, dyadic
boundaries and minimum/maximum support. Early exit requires the first exact
double distance to be below stage-radius squared by the **full original-radius**
`64*eps64` margin, and separated from the distinct second under that full margin.
Any potential equal/near-equal competitor to such a winner is still inside the
stage sphere. Ambiguous, boundary or unresolved queries continue wider or use
the original full-radius CPU fallback.

The float screen is a conservative lower bound, never a float winner estimate.
Target intervals are outward float endpoints derived from original doubles and
checked for containment. Query endpoints use directed double-to-float conversion.
For each coordinate the nonnegative interval gap is
`max(round_down(qlo-phi), round_down(plo-qhi), 0)`. Directed-down squares and two
directed-down additions produce `L <= exact real squared distance`.
[NVIDIA documents these directed intrinsics](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
and [directed conversions](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html).

Because CuPy injects `-ftz=true`, pruning is allowed only when every original
coordinate is zero or has magnitude at least `FLT_MIN`; endpoints are then zero
or normal floats. Unsupported target lanes/query coordinates keep the original
explicit double work. Positive subnormal float intermediate results may flush
to zero, which remains a lower bound after the nonnegative gap operation.
Within the bounded original-coordinate domain, each nonzero double difference
is at least `2^-178` and its square at least `2^-356`, so original CPU distance
terms are normal binary64. With `u64=2^-53`, original rounded CPU distance obeys
`dCPU >= real_squared_distance*(1-u64)^5`; contraction cannot increase the
rounding count. The prune threshold is upward-float conversion of
`RN64(stage_radius_squared*(1+64*eps64))`, exceeding the threshold needed to retain
every possible CPU strict-radius hit. A candidate is skipped only when `L` is
strictly larger; all retained candidates use original explicit FP64 RN distance,
IDs and first/distinct-second reduction. The early full-margin guard prevents a
pruned outside-stage second from silently changing the original tie policy.
These are declared numerical premises, to be tested against the exact installed
CPU/CUDA configuration, not a universal version-independent compatibility claim.

Five dyadic grids/cloud cover the common union of original ICP radii and three
fractions without artificial four-slot thrashing. Four pointer tables/cloud,
64 clouds and a 256 MiB total retained-byte cap bound ownership. Cached bytes
include original sorted doubles/IDs/keys/offsets, six float endpoints, validity
flags and pointer/radius tables. Eviction synchronizes the selected CUDA null
stream and invalidates affected pointer tables before grid release. Oversized
targets remain CPU-only; transient query/result arrays and allocator pools are
reported separately. Target mutation invalidates its content-hashed indexes.

One seven-column host result includes two bit-only payload columns. Per-stage
candidate counts use 20 bits each, plus a three-bit execution mask; one stage
visits at most one million unique IDs, so the mask/count pack fits 63 bits.
Payloads are copied unchanged and decoded through `uint64` views before any
arithmetic or JSON. Reports distinguish candidates visited, candidates pruned
and actual double evaluations (`visited-pruned`), plus stage rows/certificates
and index/table/global-cache evictions. FP32 screening reduces double work but
does not imply less candidate memory scanning. No kernel-only timing is claimed.

Fresh separate synthetic host/device and all-nine actual accepted/rejected
bridge proofs are mandatory. The validator binds current core/math, adapter,
kernel/harness/validator, CPU binaries, NumPy/OpenCV/CUDA configuration, GPU,
raw/fixture/reference, original task/proposal order, stage/domain/FTZ policy and
unchanged arrays. Both inner certificate branches and pruning/original-FP64
branches need positive zero-error CPU hit/miss coverage. Tests also exercise
inner-query unsupported/full-stage supported synchronization, original radius
outside support, subnormal exact paths, missing-final-pointer CPU fallback,
five-grid/table eviction/rebuild and total-cache eviction. Only validated
audited target digests can permit unaudited research direct misses. The timing
wrapper requires a fresh output and preserves primary plus cleanup errors;
synthetic timing setup is explicitly omitted in favor of the separate proofs.

The measured full-radius grid was 16.832 s warm versus 10.907 s native, including
8.710 s search/transfers. Its search must improve by at least 3.13 times merely
to match native with other costs fixed. Staging adds cell lookups for unresolved
queries, and float bounds add storage, scanning and diagnostics; fewer double
evaluations alone cannot establish a speed gain. NVIDIA lists the RTX 3080 Ti as
[compute capability 8.6](https://developer.nvidia.com/cuda/gpus), whose
[documented non-Tensor FP32:FP64 arithmetic throughput is 64:1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
That motivates this experiment; FP64 dominance in the measured kernel is an
inference without a per-kernel profile. No whole-Finish, mesh or real-camera
throughput improvement is established by a component proof.
