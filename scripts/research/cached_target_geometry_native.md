# Immutable target index: source preparation

This additive research helper is **unbuilt and unqualified**. It does not change
the scanner, accept old proof tokens, cache poses/results, implement registration
metrics, or establish a speed gain. Importing its Python module or running its
recipe CLI loads no numerical/native package and executes no compiler.

`CachedTargets.query(target_bytes, target_rows, transformed_query_bytes,
query_rows, radius)` caches only an immutable owned target and its index. Both
arguments contain exact original little-endian FP64 XYZ values in original row
order. A future caller must obtain queries from the original Open3D source-copy
and transform path, then pass those actual bytes. The helper applies no transform,
rounding, normal approximation, or nearest-answer reuse. It returns fresh `<i8`
IDs (`-1` for misses) and `<f8` squared distances (`+inf` for misses), in original
query-row order. Original source/target colors/normals, gates and reductions stay
with the original evaluator. The helper itself cannot produce a faithful
`RegistrationResult` merely by summing these distances in Python.

## Source and build route

The installed wheel has `Open3D.dll` and `tbb12.dll`, but no Open3D headers or
Windows import library was found. Calling original C++ classes from that DLL
would first need an exact matching SDK/export/ABI route. This prototype instead
uses standalone pinned headers and a small owned C ABI:

- `ctgn_create` copies the target into an Eigen column-major 3-by-N matrix and
  builds `KDTreeEigenMatrixAdaptor<const Eigen::MatrixXd,-1,metric_L2,false>`
  with leaf size 15, including Open3D's explicit `buildIndex` call.
- `ctgn_query` calls `knnSearch(...,1,...)` on each fresh query, then the original
  `lower_bound` radius-squared test. It preserves traversal tie choice and
  forbids `NANOFLANN_FIRST_MATCH` rather than substituting lowest target ID.
- `ctgn_destroy` releases the owned matrix/index. Integer handles live in a
  serialized registry; no raw object pointer crosses the ABI. Query workers
  share only immutable tree data and write disjoint row ranges. All started
  workers join before the registry can destroy a target.

Exact dependencies are recorded by `source_contract()` and the verified recipe:

| Asset | Pin | License |
| --- | --- | --- |
| Open3D reference | v0.20.0 KDTreeFlann.cpp SHA `13a726ef9e7a2174dc560e663a334555824e42ef4d9ab7b8969735aa9e5d3185` | MIT |
| Eigen | commit `da7909592376c893dabbc4b6453a8ffe46b1eb8e`; archive SHA `37f71e1d7c408e2cc29ef90dcda265e2de0ad5ed6249d7552f6c33897eafd674` | MPL-2.0 |
| nanoflann | v1.5.0; archive SHA `89aecfef1a956ccba7e40f24561846d064f309bc547cc184af7f4426e42f8e65` | BSD |
| TBB reference only | v2021.12.0; archive SHA `c7bb7aa69c254d91b8f0041a71c5bcc3936acb64408a1719aec0b2b7639dd84f` | Apache-2.0; not linked here |

The existing ignored Eigen assets under
`benchmark-output/cuda-pipeline/resident-icp/vendor/` are reusable only after the
new recipe verifies every Eigen header and `COPYING.MPL2` against the pinned
archive. Nanoflann's `include/nanoflann.hpp` and `COPYING` must similarly equal the
exact pinned archive. Do not replace these with an installed latest package.
Retain their notices/licenses with any distributed research binary. Official
pins: [Eigen CMake](https://github.com/isl-org/Open3D/blob/v0.20.0/3rdparty/eigen/eigen.cmake),
[nanoflann CMake](https://github.com/isl-org/Open3D/blob/v0.20.0/3rdparty/nanoflann/nanoflann.cmake),
[TBB CMake](https://github.com/isl-org/Open3D/blob/v0.20.0/3rdparty/mkl/tbb.cmake).

Root alone may allocate compilation/crosschecks. The CLI emits a verified JSON
recipe and performs no download/extraction/build:

```powershell
python -S scripts/research/cached_target_geometry_native.py --eigen-root <verified-Eigen-root> --eigen-archive <pinned-Eigen.tar.gz> --nanoflann-root <verified-nanoflann-root> --nanoflann-archive <pinned-nanoflann.tar.gz> --compiler <absolute-cl.exe> --output <fresh-private-output.dll>
```

Its explicit MSVC x64 C++17 command uses `/O2 /EHsc /LD /fp:strict` and
`EIGEN_DONT_PARALLELIZE`. Run it only in the proper x64 developer environment.
Record exact command/compiler, stdout/stderr, actual exit, recipe/dependency
receipts, current source/helper hashes and resulting DLL hash in a fresh
`.build.json`. Change `stage` to `built`, add integer `actual_exit_code:0` and
`library_sha256` only after actual success. `NativeLibrary` checks that receipt
before an explicit DLL load; this is build provenance, not numerical authority.

## Bounded scope and lifecycle

Targets/queries are finite XYZ with absolute coordinates at most 2^20, at most
1,000,000 rows, and radius in [2^-20, 1]. Empty query batches are allowed; empty
targets and other domains are refused. No successful full-call fallback exists.
The native registry is capped at 16 targets / 512 MiB reserved. The default
Python LRU caps four targets / 256 MiB of target-byte plus native reservations.
Changed target bytes, order or signed-zero bits produce a new index. This is
content ownership, not reliance on an address or stale PointCloud identity.

The native `320*N+65536` reservation is a conservative counted matrix/index/
pool-layout allowance checked against the built index. It excludes allocator
metadata, tree-build stack, worker stacks, CRT/DLL overhead and process RSS.
Current query input/output copies also lie outside retained-cache accounting:
Python receives immutable bytes and makes explicit C-compatible copies for the
ABI. Report and charge those copies; do not call the cache cap a total-process
cap. Thread count is explicitly 1, 2, 4, 8 or 20. Creation/join costs are charged
per query; there is no hidden thread pool.

Every query validates supplied bytes, output lengths, ID bounds, positive/finite
hit metrics and strict radius. Native failures and malformed output latch the
cache. Failed destruction retains that handle/target and independently attempts
other cleanup; primary errors survive. Actual worker floating rounding/FTZ/DAZ
control must equal the creating thread. The DLL checks these controls rather
than altering them. A mismatch rejects the call.

## Required fresh semantic checks before any use

The installed original binary is not attested by a matching source tag. Strict
compiler flags and matching headers alone do not prove equal winners/distances.
Fresh root-exclusive tests must compare this helper against the actual original
`KDTreeFlann.search_hybrid_vector_3d(..., radius, 1)` on identical query bytes:

1. Duplicate/equidistant targets, permutations, axis/pivot ties and signed zero;
   canonical source-to-target IDs must agree exactly, without rewriting ties.
2. Strict radius equality and adjacent FP64 values, subnormal differences,
   cancellation/large supported coordinates, zero hits and partial query ranges.
   Accepted squared-distance bits must agree exactly; misses must agree.
3. All declared thread counts against serial and original outputs, target reuse,
   eviction/clear, byte mutation/replacement, native failure and cleanup paths.
4. Fresh actual held-out/evaluation query bytes produced by the original
   `PointCloud.Transform` on current inputs and transforms. Bind actual loaded
   Open3D backend/Open3D.dll, compiler, Eigen/nanoflann and floating controls.

Original `EvaluateRegistration` skips transformation when Eigen `isIdentity()`
accepts it and otherwise uses a homogeneous 4-vector multiply and division. An
affine NumPy shortcut or exact-identity-only decision would change query bytes.
Keep the original transform outside this helper. Source references:
[registration](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/pipelines/registration/Registration.cpp),
[geometry transform](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/geometry/Geometry3D.cpp),
[KDTree](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/geometry/KDTreeFlann.cpp),
[nanoflann metric/ties](https://github.com/jlblancoc/nanoflann/blob/v1.5.0/include/nanoflann.hpp).

Original TBB evaluation joins correspondence vectors and partial FP64 error
sums. Source-ordered helper output does not reproduce its raw pair order or
reduction schedule. Canonical IDs can be tested exactly, while original fitness,
RMSE, information matrices and full gate outcomes require a separately declared
native reduction/evaluator study. Retain original reductions now. Sequential
or standard-thread bulk search may be slower than original TBB20 despite index
reuse; measure cold build, warm lookup, transformation, copies, thread setup,
reduction and closure before making any performance inference.
