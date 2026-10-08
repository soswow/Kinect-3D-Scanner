# True-device flat grid with inherited resident ICP

This separate component experiment changes query/result transport and data
ownership. It is not selected by production and has no complete archive
performance or quality authority at preparation time. Old flat host, staged
grid and OptiX proofs cannot authorize this device-resident trajectory.

The frozen flat shader still visits every original full-radius candidate using
original double coordinates and IDs. The new classifier applies the same
64-epsilon tie/boundary policy, retains direct IDs/metrics on the GPU, and
downloads only CPU resolution or explicitly audited rows. A bounded uint64
counter packet synchronizes classification. Exceptions and malformed output
propagate. CPU ties, boundaries, unsupported inputs and conservative misses
retain the original CPU tree. Direct misses require both CPU audits or the
new distinct `DeviceFlatGridProofAuthority`.

Original-order target XYZ and sorted indexes share the 256 MiB/64-cloud cache
cap and synchronized release. Queries have a separate worst-case owned array
bound of `136*N+80` bytes, capped at 136 MiB. The resident wrapper bounds its
original math arrays plus query temporaries together at 256 MiB. External
query storage, CPU packets, transient arrays outside that scope and CuPy pools
are not total-memory capped by these index/query budgets and are observed
separately. Only flagged packets leave the device in the supported unaudited
path; a fully audited proof deliberately copies every direct hit and miss.

The producer composite delegates matches to `DeviceGridResidentICP` and the
unchanged `research_resident_icp.ResidentICP` transform, equation reduction,
Huber loss, convergence, update order and Eigen solver. GPU reductions can
change roundoff relative to native CPU, so fresh original proposal witnesses,
pose/information and accepted/rejected decisions remain mandatory. The pinned
existing DLL, manifest, CPU source and GPU-source hashes are bound alongside
all inherited/new source, configuration, installed binary/runtime/driver,
raw/fixture/reference/proposal and cache/thread provenance. No NumPy solve or
quality threshold replacement is used. Full-call CPU fallback resets to the
original seed; a fallback on the measured real trajectory invalidates this
experiment's resident timing authority.

Run only after explicit parent allocation, with fresh output files and frozen
source. Default solver is the already built pinned
`benchmark-output/cuda-pipeline/resident-icp/resident_solve.dll`.

```powershell
$env:OMP_NUM_THREADS='8'
$py='python' # Use the active CUDA-enabled Python environment.
$out='benchmark-output/cuda-pipeline/uniform-grid-nearest/device-resident-v1'
& $py scripts/benchmark_device_grid_resident.py --run-allocated --synthetic-only --check-device-adapter --audit-nearest --audit-misses --miss-policy direct-miss-research-v1 --repeats 1 --output "$out/synthetic-proof.json"
& $py scripts/benchmark_device_grid_resident.py --run-allocated --check-device-adapter --audit-nearest --audit-misses --miss-policy direct-miss-research-v1 --repeats 1 --output "$out/bridge-audit.json"
& $py scripts/benchmark_device_grid_resident_timing.py --run-allocated --miss-policy direct-miss-research-v1 --proof-synthetic "$out/synthetic-proof.json" --proof-bridge "$out/bridge-audit.json" --repeats 2 --output "$out/timing.json"
```

Focused synthetic tests cover exact masks, packet/scatter metrics, CPU audits
before correction, malformed/nonfinite hard failures, policy mutation/input/
scratch guards, original XYZ mutation/eviction/clear, cache-budget CPU paths
and zero full-query downloads for unambiguous unaudited supported hits.
Deliberate faults use separate owned instances and do not contaminate main
proof statistics. Generic host/device and signed-domain checks supply positive
fully audited hits/misses. All nine actual original competing proposals then
run on the new resident trajectory with both CPU shadows before timing.

The separate timing driver omits redundant synthetic setup only after both
new unchanged proofs and actual configuration/source/runtime/input validate.
It still runs the contemporary native baseline, all original proposals and
every final independent quality/provenance/cleanup gate. NN host wall includes
stream synchronization and may absorb earlier queued transforms/uploads;
correction enqueue is not synchronized elapsed time. Canonical synthetic,
trajectory proof and timing all bind `gpu_timing=False`, so per-iteration
CUDA events are not allocated or recorded. Resident event fields are omitted
and explicitly listed as uncollected; host timers and counters supply the
observations. Inclusive timers cannot be summed as total wall or shader-only speed.
Only completed full-bridge wall time compared with its own native baseline can
support a component performance conclusion. No result is measured yet.
