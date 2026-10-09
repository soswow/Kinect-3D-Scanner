# Proposal-only canonical FPFH experiment

This is a new, unmeasured research policy. It does not authorize the old
checkpoint, component, surface-quality or timing routes. Production sources,
original measured reports and the current checkpoint protocol stay unchanged.

The C5 paired input diagnostic observed identical coarse point counts and row
order with point differences of about 3e-15 m. The corresponding FPFH arrays
differed by as much as 29.78, and one coarse normal changed sign. The hypothesis
is that tiny coordinate differences and inherited coarse normal orientation
can amplify in proposal preparation. Those observations do not prove a cause.

`canonical_fpfh_proposals.py` builds a private proposal cloud using exactly
FP64 `rint(points / 1e-6) * 1e-6`. It preserves row order and duplicates, turns
both signs of zero into positive zero, and rejects nonfinite, oversized or
unsupported inputs. Its coordinate bound is half a micrometre plus the small
FP64 multiplication error; the three-dimensional displacement bound is about
0.866 micrometres. Coordinates within 64 FP64 ULPs of a scaled half-integer
are rejected before native work because tiny perturbations there can choose
different quantization cells. The scalar reference tests exact ties to even,
both signs and nearby representable values; it does not quietly move ties.

The new cloud has no inherited normals. It calls the original normal search
at 80 mm/30 neighbours and FPFH search at 200 mm/100 neighbours. Both searches
use Open3D20 and restore the caller's declared RANSAC1 or RANSAC20 thread
context, including failure paths. `--normal-policy fresh` is the primary
hypothesis. The separate `largest-component-positive` option additionally
chooses a deterministic sign: the largest absolute normal component is
positive, with exact axis ties resolved X, Y, Z. It changes sign only, without
renormalization, and is explicitly a different proposal policy. It cannot
resolve genuinely ambiguous or degenerate normal directions.

Run the paired preprocessing proof only in an allocated CPU/native slot:

```powershell
python scripts/research/canonical_fpfh_proposals.py `
  --left benchmark-output/field-cuda-study/ransac-input-diagnostic-v2/chest-5-native-inputs.json `
  --right benchmark-output/field-cuda-study/ransac-input-diagnostic-v2/chest-5-audit-inputs.json `
  --output benchmark-output/field-cuda-study/canonical-fpfh-v1/chest-5-fresh.json `
  --normal-policy fresh --repeats 2 --run-allocated
```

Use a different fresh output for the explicit sign variant. The command
validates the closed input JSON and NPZ bytes, every array descriptor/payload,
ordered fragment pairs, seeds and shapes. It then records canonical point,
normal and FPFH hashes for both sides and repetitions. Success requires exact
bytes in all three arrays and unchanged original captures/source files. It
never runs RANSAC, CUDA, fusion or geometry acceptance. Preparation wall time
is diagnostic, not a scanning speed claim. Python `-S` is appropriate for
the contract tests and `--help`, but the native command needs the configured
Open3D/NumPy environment.

For a future separately declared Finish producer, enter
`CanonicalGlobalSeedScope(fragments, FreshCanonicalFeatures(np, o3d))` before
the original output-only gate observers and before `GlobalSeedThreadScope`.
The scope source-checks the entire original `_global_seed` function and its
loaded code. It passes the unchanged original function shallow fragment
proxies with private coarse clouds/FPFH, retaining original train and heldout
owners. The original seed, mutual feature filtering, 80 mm correspondence
radius, three-point estimator, edge/distance checkers, 12000 iterations,
0.999 confidence and original returned result remain unchanged. Private
features use a content-invalidated 32-fragment cache capped at 12000 points
per cloud (at most 119808000 logical numeric bytes). Native trees, temporary
workspaces and allocator pools are additional memory. Original and private
input bytes are checked after every original call. A preprocessing, native,
evidence or hook-restoration fault hard-latches the research scope; it cannot
silently become an ordinary rejected frame.

This only removes one candidate source of proposal instability. It does not
guarantee reproducible voxel enumeration, row order in other sessions,
parallel RANSAC scheduling, normal eigenspaces, floating-point ICP results or
later branch decisions. It never sorts/deduplicates clouds or quantizes train,
heldout, ICP, fusion, poses or accepted-transform checks. Even exact paired
FPFH bytes are only a preprocessing result. A new fixed-source native/audit
Finish comparison, every original gate, unchanged full query/CPU result
shadows and independent geometry quality proof remain necessary before any
performance or production claim.
