# Bulk original nearest-query CPU audit

Status: the bounded native API crosscheck and complete old dual-shadow
trajectory both passed. Existing scalar auditors,
frozen historical proofs and the production scanner are unchanged. No scanning
speedup has been established.

The original v0.20.0 `EvaluateRegistration` copies the query point cloud and,
with exact identity supplied, skips `Transform`. Its registration reduction
calls `KDTreeFlann.SearchHybrid(point, radius, 1)` for each point. The same
reduction is used by original ICP. Bulk correspondence order is not fixed, so
the helper maps results by source index rather than relying on order.
See the [official registration implementation](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/pipelines/registration/Registration.cpp).

`SearchHybrid` uses nanoflann's nearest search, then excludes squared distances
greater than or equal to the original squared radius. It builds from the
original target point order. Equal-distance selection remains the library's
policy; this experiment does not substitute a lowest-index rule. Rebuilding a
tree must therefore be checked against the retained original scalar tree.
See [KDTreeFlann](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/geometry/KDTreeFlann.cpp)
and the [scalar Python binding](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/pybind/geometry/kdtreeflann.cpp).

This suggests removing Python's per-row call overhead without changing the
search function. Bulk evaluation also rebuilds the target tree and copies query
points, so those costs must be included. Installed binaries may differ from an
official source tag: source inspection alone is not authority. The proof binds
the actual loaded Open3D/NumPy modules and shared libraries and compares outputs.

The new [helper](bulk_legacy_nn_audit.py) exposes:

```python
result = bulk_legacy_ids(
    np, o3d, original_queries, original_target_cloud, radius_m,
    expected_target_digest=item["digest"], chunk_rows=65536,
)
original_ids = result.ids
```

IDs are immutable `int32`, with `-1` for misses. Queries may be strided original
float64 packet views; the helper makes byte-preserving copies. The initial
domain is finite float64 `(N,3)`, at most one million target/query rows, absolute
coordinates at most `2^20` and radii `2^-20 .. 1` metres. Unsupported input raises
`UnsupportedBulkDomain`; an integrating experiment must perform the complete
original scalar audit for those rows, never skip them.

Native IDs alone decide acceptance. The returned squared metric is a separate
original-double subtraction/multiply/add diagnostic. The first crosscheck
requires both ID and scalar distance-bit agreement; its metric must never be
used to expand or shrink native correspondence membership. Malformed,
duplicate/out-of-range correspondence rows, changed identity, transport or
query/target mutation are hard failures. Target ownership must remain immutable
through the call, and the cached-tree target digest must match.

For a future device audit, retain original scalar ambiguity/unsupported handling
for packet `reason & 7`. Only direct-hit/direct-miss shadows (`reason & 24`) are
eligible for the bulk path after distinct proof validation. Audited counts and
comparison before correction/scatter must remain unchanged. Successful bulk
evaluation is still an original CPU shadow, not GPU registration authority.

The prepared CLI exhausts 720 target orders of six exact tied axes, 24 orders of
duplicated/signed-zero points, 180 strict-radius/adjacent-ULP cases and eight
finite random cases: 932 cases and 5,924 synthetic query rows. It additionally
scalar-checks every retained old real trace row: 12 batches and 24,576 queries,
compares saved original gold IDs and repeats bulk queries with chunk sizes
257, 4,096 and 65,536. Artificial output/lifetime contracts are separate from
actual native evidence.

After a hardware slot is explicitly allocated, the initial command is:

```powershell
./.venv/Scripts/python.exe -s scripts/research/bulk_legacy_nn_audit.py `
  --run-allocated `
  --trace benchmark-output/cuda-pipeline/grid-lookup-ablation-order-v2/real.npz `
  --output benchmark-output/field-cuda-study/bulk-nn-audit/scalar-bulk-proof.json
```

The local official source snapshots and `source-reference.json` are already
prepared under `benchmark-output/field-cuda-study/bulk-nn-audit/reference/`.
They record exact URL, bytes and SHA256 for the registration implementation,
KDTree implementation and both Python binding sources. A separate checkout
must fetch those same official files and construct the same manifest, or pass
its explicit `--references` path; private trace arrays and native DLLs remain
local and are not distributed. The helper refuses existing output reports.

The allocated API check closed with exact native IDs and diagnostic distance
bits for all 5,924 synthetic queries and all 24,576 retained real rows. On that
small retained sample, original scalar queries took 0.09195 seconds; bulk all-in
queries took 0.04153 seconds with 4,096-row chunks and 0.04118 seconds with
65,536-row chunks. Those timings include bulk copies, target-tree rebuilding,
normalization and hashes. They exclude native import startup and are not a full
trajectory audit throughput or Finish speed result. See the local ignored
`benchmark-output/field-cuda-study/bulk-nn-audit/scalar-bulk-proof.json` report.

A passed bounded API crosscheck explicitly sets
`new_field_bulk_only_shadow_authorized=false`. It checks every retained trace
row, not all 104 million old resident iterations. Before using bulk-only
shadows in new field proofs, retain a distinct current-source complete old
resident trajectory with dual scalar/bulk IDs, followed by complete new-field
CPU hit/miss shadows and all original proposal/pose/witness/mesh gates. No old
report fingerprints or guard conditions should be edited to authorize the new
auditor. First measure the audit's all-in wall time; claims about final scanning
speed remain a separate experiment.

The separate [dual adapter](cuda_bulk_audit_grid_registration.py) and
[complete-old producer](benchmark_bulk_resident_audit.py) preserve the original
GPU/raw/classifier/packet/audit/scatter method outside its CPU-resolution block.
A source AST guard checks that boundary. The original all-nine ordered proposal,
pose and witness task loop also remains unchanged. Dual mode retains every
original scalar lookup, compares native bulk IDs and diagnostic metric bits
before correction/scatter, and records original query/target descriptors and
both result hashes. The real proof requires exactly 104,123,989 queries with
zero bulk-domain diversion, original CPU fallback or mismatch.

The complete dual proof was generated with:

```powershell
./.venv/Scripts/python.exe -s scripts/research/benchmark_bulk_resident_audit.py `
  --run-allocated `
  --api-proof benchmark-output/field-cuda-study/bulk-nn-audit/scalar-bulk-proof.json `
  --audit-nearest --audit-misses --check-device-adapter `
  --miss-policy direct-miss-research-v1 --repeats 1 `
  --output benchmark-output/field-cuda-study/bulk-nn-audit/old-trajectory-dual-proof.json
```

It needs an exclusive allocated slot and the separate new bulk authority
validator; use a fresh output filename to reproduce it. The
initial API process had its recorded thread environment; the complete dual
proof separately binds and checks the actual original OMP8/Open3D20/OpenCV20
policy and the same loaded native binaries. Native bulk temporary allocator
memory remains opaque; visible additional host arrays are bounded by the point
and chunk limits and are reported as estimates. Raw arrays are not added to
the proof JSON.

Only a validated distinct complete dual token can enable `audit_mode="bulk-shadow"`
for a subsequent field audit. That path still performs original scalar handling
for `reason & 7`, and scalar-shadows every affected row when the bulk domain is
unsupported. Nonaudited field timing uses the original device adapter; this new
class is deliberately restricted to complete hit/miss auditing.

The closed complete old proof covers all 104,123,989 original resident-trajectory
queries: 58,809,404 hits and 45,314,585 misses across 11,569 query batches and
236 resident registration calls. Scalar and bulk IDs and squared-distance hashes
agree for every batch, with zero mismatches, unsupported-domain diversions,
scalar ambiguity fallbacks or malformed device results. Both original pairs,
all nine ordered proposals and their original pose/witness/information gates
passed. Current raw/fixture/source/library/runtime bindings and cleanup passed.
An independent stdlib validation also passed; the closed report SHA256 is
`61126101d2c50bea67fa1a8151182b0c4b8bcdda549cd74d66d169d49602d62b`.

Bulk helper all-in time totalled 53.7475 seconds, including copies, tree builds,
normalization and hashes. Original scalar per-query call timers totalled
263.3799 seconds in the same simultaneous proof. Those timer scopes differ,
and the dual run deliberately performs both auditors plus additional comparisons;
it is not a scanning or full-Finish speed measurement. New field queries still
need complete original CPU hit/miss shadows, original CPU ICP result shadows
and independent whole-Finish quality authority. The old target set authorizes
only that component proof, not new field targets or poses.

Run the pure-stdlib fault/CLI contracts separately from numerical benchmarks:

```powershell
python -s scripts/research/check_bulk_audit_contract.py `
  --output benchmark-output/field-cuda-study/bulk-nn-audit/contracts-new.json
```

The current 86 cases include deliberately fabricated contract fixtures and
explicit mocks of the independently tested inherited math/device guards.
They test rejection and closure behavior, not numerical or GPU correctness.
