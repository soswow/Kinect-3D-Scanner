# Fresh Live checkpoint Finish experiment

This is an offline research route. It leaves the production server and original
tracking, graph proposals, geometric witnesses, information matrices, fusion and
mesh gates unchanged. It has no default or whole-session promotion. Every native
job needs an exclusive hardware allocation and the field server stopped.

The first independent raw-Live native/resident experiment exposed real Live pose
drift between runs, despite the same accepted indices. Its report remains closed
and cannot authorize timing. Version 2 captures one actual raw-Live engine, then
materializes it for both complete Finish controls. Exported reconstruction poses
are never used as Live seeds, and the Live volume is never reconstructed by
re-fusion.

The private JSON graph plus contained numeric NPY files retain raw and actual
prepared images, unrounded poses, raw diagnostics, semantic decisions, shared
object/backing-array aliases, original CPU model/preview/feature point order,
preparation and descriptor caches, transaction state, and every active logical
VBG key/attribute bit with the original capacity. Native inactive buffers,
frustum scratch, physical VBG buffer indices and allocator/kernel pools are not
serialized; their reservations are observed separately. Unknown state, native
tensor layouts outside the declared scope and unhealthy/transient sessions fail
closed. Materialization must re-prove the entire graph, capacity/voxel bits and
every actual prepared image before Finish.

Python and NumPy global RNG state is retained. Opaque native RNG state is not
exposed: both Finish controls explicitly apply the shared seed before
materialization. Original fragment RANSAC reseeds each proposal; official
OpenCV 5 legacy EPNP RANSAC source uses a fixed local RNG per call. Source
semantics do not substitute for actual ordered call/gate and native-binary proof.

From the repository root, use the Python environment containing the pinned
Open3D CUDA wheel, CuPy, native extension and original resident Eigen DLL:

```powershell
python scripts/research/profile_checkpoint_resident_finish.py <raw.zip> --mode capture --checkpoint benchmark-output/<fresh-checkpoint-dir> --output benchmark-output/<capture.json> --component-synthetic <current-synthetic-proof.json> --component-bridge <current-bridge-audit.json> --final-preplan --run-allocated
python scripts/research/profile_checkpoint_resident_finish.py <raw.zip> --mode native --checkpoint benchmark-output/<fresh-checkpoint-dir>/checkpoint.json --output benchmark-output/<native.json> --component-synthetic <current-synthetic-proof.json> --component-bridge <current-bridge-audit.json> --final-preplan --run-allocated
python scripts/research/profile_checkpoint_resident_finish.py <raw.zip> --mode audit --checkpoint benchmark-output/<fresh-checkpoint-dir>/checkpoint.json --output benchmark-output/<audit.json> --component-synthetic <current-synthetic-proof.json> --component-bridge <current-bridge-audit.json> --final-preplan --run-allocated
python scripts/research/compare_checkpoint_resident_finishes.py benchmark-output/<native.resident.json> benchmark-output/<audit.resident.json> --output benchmark-output/<quality.json> --run-allocated
python scripts/research/profile_checkpoint_resident_finish.py <raw.zip> --mode timing --checkpoint benchmark-output/<fresh-checkpoint-dir>/checkpoint.json --output benchmark-output/<timing.json> --component-synthetic <current-synthetic-proof.json> --component-bridge <current-bridge-audit.json> --finish-audit benchmark-output/<audit.resident.json> --quality-proof benchmark-output/<quality.json> --final-preplan --run-allocated
```

Use the same seed, Final budget and preplanner option in capture and every
control. A larger Final budget must be declared at capture with
`--final-block-count` and matched in all phases. Every destination is fresh;
failed artifacts remain available for diagnosis. The checkpoint, all producer
dependencies, calibration, effective policy, runtime/native libraries and raw
archive bytes stay frozen throughout this sequence.

Audit runs full original CPU nearest hit/miss shadows and complete original CPU
registration shadows on exactly the resident call's arrays and seed. Exact
canonical source-to-target correspondence IDs and strict numeric tolerances are
required; raw correspondence order remains diagnostic. Original ordered graph
witnesses/gates and an independent native triangle surface must also pass.
Unsupported complete resident calls are hard failures. Counted per-query
original CPU ambiguity/domain handling is retained. A latched research fault
cannot become a rejected frame or a successful fallback mesh.

An optional `--bulk-audit-proof <closed-dual-report.json>` enables an independently
validated original native bulk CPU query auditor during audit only. It requires
the separate complete old nine-proposal scalar/bulk identity and metric-bit
proof under current native libraries. Domain/ambiguity rows remain scalar and
are counted. Timing may consume that token solely to validate the audited proof;
its device query adapter and resident math remain the original research route.
Native and capture controls consume no bulk token. All controls pin the same
fifteen source dependencies even when the option is unused.

Capture timing includes observer copies and cannot replace independent Live
throughput measurements. Checkpoint loading/parity checks, imports and exports
are outside `finish_s`. Resident setup, exact signature/gate recording, optional
block planning and full-Finish authority validation are inside it. Audit timing
includes its shadows and is never a speed result. Timing requires the new
checkpoint/quality authority and rejects an altered, missing or extra ordered
call before accepting the mesh.
