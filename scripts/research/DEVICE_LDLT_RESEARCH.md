# Original Eigen equations on the device

This is an unexecuted, isolated FP64 solve prototype. It changes no scanner,
resident ICP, nearest-neighbor, convergence, graph or geometry gate. Passing a
micro-experiment would authorize only the systems and tolerance it measures.

`research_device_ldlt.cu` ports the pinned Eigen Lower LDLT pivot/permutation,
factor sign, diagonal conditioning guard and diagonal pseudo-inverse semantics.
The original bridge solves `A*x=-gradient`; the incremental pose uses
`Rz(x[2])*Ry(x[1])*Rx(x[0])` via Eigen's generic quaternion association and
translation `x[3:6]`. Explicit binary64 RN arithmetic and `--fmad=false` are used.
The scalar accumulation and libdevice trigonometry differ from the CPU evaluator,
so neither bit equivalence nor whole-ICP equivalence is claimed. The derived
Eigen portions retain the MPL-2.0 notice.

The source references are Eigen commit
[`da7909592376c893dabbc4b6453a8ffe46b1eb8e` LDLT.h](https://gitlab.com/libeigen/eigen/-/blob/da7909592376c893dabbc4b6453a8ffe46b1eb8e/Eigen/src/Cholesky/LDLT.h)
and its
[Quaternion.h](https://gitlab.com/libeigen/eigen/-/blob/da7909592376c893dabbc4b6453a8ffe46b1eb8e/Eigen/src/Geometry/Quaternion.h).
The locally preserved official header archive/tree is the same one checked by
the unchanged resident solve DLL build manifest. This is a comparison with that
standalone bridge; it does not invoke the Open3D binary's utility solve.

The adapter accepts only disjoint, contiguous, selected-device buffers under a
4096-system batch cap. Each system owns 844 bytes of numeric inputs and outputs.
One scalar thread solves each system. Device damage propagates; rejected status
codes are never silently accepted, retried or applied to a trajectory. Previous
pose input and composition-overflow statuses 8/9 are additional prototype guards,
distinct from the original bridge's equation rejection codes.

`benchmark_device_ldlt.py` first shadows 93 synthetic systems with the current
original Eigen DLL: regular SPD, pivot ties, threshold conditioning, zero/rank,
subnormal and nonfinite equations, signed zero, angle stress, rotated previous
poses and finite-overflow cases. CPU status and incremental update are compared;
composition is checked against original NumPy `update @ previous`. Absolute
update/composition tolerance is `1e-8`. Rejected outputs remain diagnostic.
Step, pivots and diagonal values are saved but not certified against unavailable
original DLL intermediates. Nonfinite differences remain explicit JSON failures,
with the raw outputs preserved.

After source review and explicit exclusive hardware allocation:

```powershell
python scripts/research/benchmark_device_ldlt.py --output benchmark-output/field-cuda-study/device-ldlt-v1/synthetic.json --repeats 100 --singleton-repeats 100 --run-allocated
```

The repeated batch timer includes Python shape/alias guards, launch and one
terminal synchronization on preuploaded data. The optional singleton timer waits
after every one-system launch. Neither includes original NN, normal-equation
reduction, input transfers, convergence or graph work. Batch throughput is not
sequential ICP latency.

The separately source-bound `capture_resident_equations.py` observer invokes the
original resident Eigen callable exactly once per actual original all-nine
component iteration and returns its identical original object. It records only
private equation/update copies, original proposal/match contexts and status;
observation overhead makes that component wall diagnostic. No exported or
recorded poses become Live seeds. All current component/source/DLL/raw fixture
and final gate closures must pass before a capture is usable.

The optional `--equations <closed manifest>` micro-comparison revalidates those
current proofs, the closed original component output and every numeric row hash.
The bounded NPZ must contain exactly four ordinary numeric arrays. Every record
must belong to its unchanged original match and proposal/seed context. A fresh
call to the original DLL must reproduce recorded status and incremental-update
bits before GPU comparison. Captured previous accumulated poses are unavailable;
their GPU composition operand is explicitly identity, not an actual trajectory
composition witness. Fresh loader/version/device/driver and before/after source
bindings remain required.

An earlier measured `system_copy_s` includes waiting for queued GPU reductions;
it is not a measured PCIe-only cost. Moving a solve alone cannot eliminate the
original per-query ambiguity/counter synchronization or CPU convergence control.
A combined synchronization or CUDA graph path needs its own equation, result,
query/failure and whole-geometry proof before any speed claim or promotion.
