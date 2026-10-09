# Cached target geometry: measured lookup pilots

The native cached-target helper and its prepared-target sibling passed fresh
original Open3D scalar nearest-neighbor checks in two closed pilots. Preparing
an immutable target once removed repeated Python target scans and output-row
scans from warm queries. Cold setup remains material, and neither pilot measures
full registration, actual Finish queries, scanner throughput or FPS.

This note records later measurements without changing the frozen
[source-preparation note](cached_target_geometry_native.md), helper sources or
original reports. No scanner integration or numerical qualification token was
created.

The reusable [Windows builder](build_cached_target_geometry_native.py) has also
run successfully. It captures an allowlisted MSVC environment, verifies the
pinned source archives and staged headers/licenses, compiles with `/fp:strict`
and writes a success or failure receipt in a fresh output directory. The first
public-builder attempt failed before compilation because Windows command-line
quoting escaped its batch-call path incorrectly. Its receipt and exact sources
remain private; invoking the owned script basename in its working directory
fixes that issue without changing the native numerical source.

The corrected public route completed with actual compiler and parent exit 0,
then its new DLL passed another fresh 66-case prepared parity run at all five
thread counts. Source/dependency closure and selected cleanup passed. This
validates that build route on the installed Windows toolchain and finite NN
fixtures; it adds no scanner integration or performance claim. Local evidence:
`benchmark-output/field-cuda-study/cached-target-geometry-v1/build-public-d/`
and `build-public-d-prepared-parity.json`.

## Exact scope and coverage

Each report contains **66 parity cases**, each exercised with **1, 2, 4, 8 and
20 native query threads**. Every case used fresh original
`KDTreeFlann.search_hybrid_vector_3d(query, radius, 1)` results. Original IDs and
FP64 squared-distance bits agreed exactly for the raw ABI and full Python cache;
the prepared pilot also required the prepared lease to agree. Coverage includes
duplicate/equidistant targets and permutations, ties across the leaf size 15
boundary, signed zero, strict-radius equality and neighboring FP64 values,
subnormal differences, a supported finite-coordinate extreme, empty queries,
and original Open3D transforms. The reports respectively cover 5,450 and 6,318
query rows per thread count; these are fixture-query counts, not Finish coverage.

The timing inputs are three geometries: a synthetic lattice, selected native
mesh vertices with identity, and the same selected vertices transformed through
the original Open3D path. The baseline uses the C5 native mesh: 6,427 targets and
1,607 queries. The prepared pilot uses a different C6 native mesh: 163,268 original
vertices, deterministically selected down to 40,817 targets and 2,041 queries.
The lattice has 16,384 targets and 2,048 queries. Field vertices are data inputs;
they are not captured ICP/evaluation queries or an independent geometry gate.
Both field timing cases use a 0.03 m radius.

Source, loaded native binaries, runtime and input records close before/after in
both reports. Dependency/header inventories and hashes are recorded build
provenance; staged header assets are not rehashed at runtime. Every retained
owner closes, original thread and environment settings restore, and
failure/cleanup lists are empty. The
original evaluator runs with Open3D 20 threads; helper thread counts are explicit
and independently tested. Runtime records identify Python 3.12.7, NumPy 2.5.3 and
Open3D 0.20.0.

## What each clock includes

- **Original evaluation:** `EvaluateRegistration`, including its tree rebuild,
  source copy/transform, TBB search and registration metric reduction. Canonical
  IDs are checked outside that call timer.
- **Raw cached native:** fresh search over a retained immutable target index,
  ABI query/output copies and worker creation/join. It does not implement the
  original registration reducer.
- **Full Python cache:** raw search plus target scans/hashes, cache lookup,
  per-row output guards and metadata on each call.
- **Prepared lease:** one cold target validation/hash/native copy; warm calls
  retain native query-domain, strict-radius, ID and distance checks while omitting
  redundant Python target/output-row scans. No nearest result or pose is cached.
- **Cold:** constructor/preparation plus the first query. **Whole four-query
  trial:** cold first query, three warm queries, exact output checks and close.
  Whole-trial values are single observed trials, not medians.

Original source-copy/transform and query-byte preparation are recorded
separately, outside warm helper clocks. Field preparation took 0.0433/0.0488 ms
for baseline identity/transform and 0.1064/0.1126 ms for prepared identity/transform.
DLL loading was also separate: 3.6825 ms baseline and 3.6382 ms prepared. These costs
must be charged by a caller using this route.

Both pilots use fixed order: original evaluation, raw native, then full Python
cache, with threads ascending 1/2/4/8/20. The prepared lease follows those methods
in its report. This is not a balanced-order speed experiment.

## Baseline pilot

Milliseconds; warm entries are medians of three calls, all helper entries below
use one thread. Cold and whole columns describe the full Python cache.

| Geometry | Original evaluation | Raw native warm | Full Python warm | Full Python cold | Full Python whole4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic lattice |2.1121|0.3102|5.3350|7.0485|24.3275|
| C5 mesh, identity |1.4364|0.3435|2.8144|4.0252|13.4076|
| C5 mesh, original transform |1.3967|0.4005|2.7163|4.1613|13.2070|

The raw index lookup is small, but the guarded full Python path is slower than
the original evaluator on these inputs. Reusing an index alone does not remove
the wrapper's per-call scans and ownership work. More helper threads did not
consistently help; one-thread raw medians were below 20-thread medians on all
three baseline geometries. No occupancy or parallel scalability inference is
made from this small pilot.

## Prepared-target pilot

Milliseconds; warm entries are medians of three calls, all helper entries below
use one thread. Cold and whole columns describe the prepared lease.

| Geometry | Original evaluation | Raw native warm | Full Python warm | Prepared warm | Prepared cold | Prepared whole4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic lattice |2.1971|0.3165|5.3309|0.3307|6.8742|9.2214|
| C6 mesh, identity |10.7345|0.6299|11.6454|0.6218|22.6987|25.7763|
| C6 mesh, original transform |10.5847|0.6980|11.4248|0.7377|22.2682|25.6641|

The C6 prepared warm query is close to the raw native path, while one cold query
costs more than one original evaluation. Any setup break-even therefore requires
reusing the unchanged target. The four-query totals include that setup and exact
checks; the warm median alone cannot represent a fresh target's cost. A full
evaluator study must additionally preserve and measure original transformation,
metric reduction and actual caller lifetimes before claiming a break-even for
registration. No baseline-to-prepared ratio is computed across the different
C5/C6 geometries.

All five helper thread counts passed. In this prepared pilot the C6 identity
prepared warm medians ranged from 0.4778 ms at two threads to 2.0134 ms at 20 threads;
transformed medians ranged from 0.6095 ms at two threads to 2.1383 ms at 20 threads.
Native worker creation/join is charged per query, and the fixture is small.
These values do not establish a preferred production thread policy.

## Build and reproducibility records

The first `build-a` attempt failed with actual exit 1: quoting of the Visual
Studio developer batch path was rejected before a DLL was produced. Its recipe,
failed build receipt and log remain preserved. `build-b` records actual exit 0
and a compiled 268,800-byte DLL with SHA256
`a26be2b1088f1d700d548b6228ce53ffb6474df1c56c7ec131b7c40eeae286c0`.
It uses MSVC 14.44.35207, C++17, `/O2 /EHsc /LD /fp:strict` and
`EIGEN_DONT_PARALLELIZE`, with the pinned Eigen and nanoflann headers/licenses
described in the source-preparation note. Matching headers/compiler policy are
not themselves an equivalence proof; the fresh original scalar checks are
required evidence.

Private local records under
`benchmark-output/field-cuda-study/cached-target-geometry-v1/`:

| Record | SHA256 |
| --- | --- |
| `parity-benchmark-b.json` | `b9391e1c51a8d82d1f6616b4515ebe4aeebcb76f70cb8116158cb71f9054fde9` |
| `prepared-parity-benchmark.json` | `ad8ca39c44ae6f4f807df9136a488a0639b6f824db19106b24b0e6034a91cfce` |

The raw helper keeps only immutable owned target data/index state. Reservation
caps cover counted retained target/index storage, not total process RSS, native
thread stacks, allocator metadata or all query temporaries. Selected owners
remain live through native completion and close. The prepared sibling changes
Python validation frequency, not the original transformed query bytes, native
nearest winner, strict radius or accepted distance bits.

The next useful experiment is a bounded actual verifier-call capture that tests
target reuse with the original evaluator's metric reductions and gate outcomes.
Until that exists, these reports establish finite-fixture NN parity and scoped
lookup/setup measurements only; they establish no full registration, Finish,
trajectory or FPS gain.
