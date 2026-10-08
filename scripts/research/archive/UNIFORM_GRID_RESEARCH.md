# Exact dyadic uniform-grid retrieval experiment

Source-only until a hardware slot is allocated. Production source `9331…` does
not import these files. This changes neighbor retrieval for the existing
standalone CPU ICP adapter; it does not change points, normals, the CPU
Huber estimator, update order, convergence limits or bridge authorizers.

For each original radius (`0.12`, `0.06`, `0.03`), the cell width is the next
power of two at least as large (`0.125`, `0.0625`, `0.03125`). Target and query
coordinates must be finite and their `ldexp` scaling reversible. Flooring
must lie in the signed 21-bit range `[-1048576, 1048575]`. Biasing each axis by
`1048576` and packing `x<<42 | y<<21 | z` into uint64 is injective on that
domain. Out-of-range adjacent cells are skipped before packing, so no wrapped
cell can collide with a supported target. Unsupported cases use the original
CPU KDTreeFlann, preserving its errors for inputs it does not accept.

The enclosure argument is about the original strict floating-point predicate,
not approximate projective geometry. A nonnegative squared-distance sum cannot
be below the original radius squared if one exact coordinate separation is at
least the dyadic cell width: correctly rounded subtraction is monotone, the
dyadic width and its square are representable, and the declared exponent
range avoids overflow/underflow of the width squared. With every accepted
coordinate separation strictly smaller than a cell width, exact scaled floors
differ by at most one, so all possible accepted candidates lie in 27 cells.
The chosen radius domain is `2^-20 <= r <= 1`. The arithmetic/rounding premise
must be checked against the installed CPU configuration, and complete hit/miss
audits remain mandatory before any unaudited direct-miss experiment.

Cell entries retain original IDs and original double coordinates, sorted by
key. A 128-thread block scans each query's 27 cells, with contiguous candidate
loads. Explicit RN double subtraction/products/left-associated sums and
`--fmad=false` avoids contraction. CuPy injects `-ftz=true`; this option affects
single-precision denormals, while these distances use explicit double precision.
NVIDIA documents the [NVRTC option](https://docs.nvidia.com/cuda/nvrtc/index.html)
and its [lack of effect on double precision](https://docs.nvidia.com/cuda/archive/11.5.2/floating-point/index.html).
Reduction
tracks first and second minima with distinct original IDs. GPU equal-distance
ties use ID ordering only for retrieval; the original CPU tree decides the
final tie. A conservative `64 * eps` metric margin also sends near ties and
strict-radius boundaries to CPU. The margin is an experimental IEEE arithmetic
assumption to be tested, not a universal software-version compatibility claim.

Default misses are CPU queries. Fresh `direct-miss-research-v1` proof runs check
every direct hit and direct miss against the original CPU tree before its
result is consumed; such runs cannot support performance claims. The separate
pure-stdlib `validate_uniform_grid_proof.py` requires distinct successful
synthetic host/device and all-nine-proposal bridge artifacts, complete positive
hit/miss coverage with zero errors, unchanged raw/fixture/cloud arrays and exact
current core/math/helper/kernel/harness/CPU-binary/GPU/driver/configuration/
domain/policy/task bindings. Only that scoped authority enables an unaudited
timing run. Target point bytes outside audited membership remain CPU-only.
Passing an arbitrary filename cannot certify itself.
No geometry or miss policy is promoted to production by this experiment.

Cache contents are keyed by original target identity and a content hash, with
bounded cloud count and retained device bytes. CPU index construction, content
hashing, sorting/uploads, GPU searches/transfers and fallback/audit timings are
recorded. CUDA device and null stream are explicit. One result buffer holds
IDs, both distances and visit count for one host transfer; bounded IDs/counts
are exactly representable as doubles. Transient CuPy buffers and allocator
pools are outside the retained-cache cap and must be reported separately.

The first device adapter prioritizes correctness: it downloads query
coordinates for CPU fallback, calls the host adapter, then uploads IDs and
distances. It makes no resident-throughput claim. The existing scanning path
already supplies CPU-transformed queries, so host transfer costs are included
in its bridge component timings.

Required gates: reversible host scaling and packed-key uniqueness; original
cell/radius boundary, duplicates and distinct ties; unsupported domain and
target mutation/cache invalidation; CPU parity of every direct hit and miss;
then accepted `[8,12]` and rejected `[0,2]` fixture bridges with all nine
original competing proposals, complete original witness/validation evidence,
poses and information. Measured fragment-local poses only reproduce component
inputs; no archived pose seeds live scanning. Full bridge performance compares
the original native CPU implementation, first grid index use and warm reuse.
Index/search wins do not imply whole-Finish or camera throughput improvements.
