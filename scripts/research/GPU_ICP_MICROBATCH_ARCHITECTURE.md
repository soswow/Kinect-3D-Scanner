# Next GPU registration experiment: bounded proposal batches and device iterations

This is an unbound architecture note, not a measured implementation, proof token,
or production recommendation. It proposes no change to the installed scanner.
All cited measurements retain their original source, protocol, and failure scope.

The next useful experiment is a small batch of independent registration seeds
for one already ranked fragment pair, with shared immutable target indexes and
separate device iteration state. Porting the 6x6 solve alone is not useful. A
device solve becomes interesting when it removes the host round trip for a
complete iteration, including pose update and convergence.

## Evidence and limits

The closed `combined-sync-source-guard-v1/whole-file-sha.json` component measured
median native / existing device / combined registration walls of 11.1301 /
7.6149 / 7.1726 seconds over three rotating rounds. Every round retained all nine
original proposal gates. The combined path was 1.55 times as fast as native and
5.8% less wall time than its contemporary device control. This is a component result on
the old 9331 source, not a whole Finish or current production speed claim.

The corresponding AST guard experiment measured combined wall 9.5153 seconds;
replacing repeated source parsing with the separately proved immutable source
guard reduced nested source-check wall from 1.6496 to 0.0661 seconds. Host work
was material. The sequential order of the two experiments prevents attributing
every difference between their controls to the guard alone.

The closed `device-ldlt-v4/actual-and-synthetic.json` experiment checked 10,861
actual normal-equation systems against a fresh replay of the original Eigen
bridge and 93 synthetic systems. Maximum actual GPU update difference was
2.97e-16. Original CPU solves averaged about 4.57 microseconds each; one GPU
launch plus synchronization averaged about 53.67 microseconds. Batch throughput
does not predict the latency of dependent ICP iterations. Captured equations
did not contain actual previous composed poses.

For the existing component trajectory, 104,123,989 query rows across 11,569
query/equation steps give roughly 9,000 rows per step. A 128-row normal-equation
kernel then launches roughly 71 blocks. Small transform/equation work may leave
some GPU capacity unused, so concurrent independent seeds are plausible. The
nearest-neighbour shader uses one block per query and already launches many
blocks. Underutilization of the entire pipeline is not established by these
counts; that would require kernel profiling.

The old C7 native 20,000-block Finish recorded 2,842 CPU registrations and
172.68 seconds inside them out of 253.72 seconds of Finish. Partial-bridge
verification contributed 2,044 calls / 140.47 seconds. That motivates studying
registration scheduling, but it does not establish the same distribution for a
new policy, current core, or another scan.

## Safe batching seam

`scanner_server/fragments.py:propose_fragment_poses` already constructs up to
four ranked appearance proposals and two original global proposals for a
selected fragment pair. Deduplicate with the original thresholds first. Initial
aggregate forward registrations for the remaining seeds share source and
target inputs and can be evaluated in a bounded batch. Start with two or four
seeds, not an unbounded list.

Return results to the original consumer in its existing order. Preserve ranking,
proposal budgets, duplicate tests, held-out validation, independent-camera
witnesses, ambiguity checks, graph frontier selection, and stopping rules. Do
not replace accepted poses with a proposal or accept a bridge from aggregate
ICP alone.

There are important dependencies. `_pair` computes the reverse seed from the
forward result. Camera verification seeds depend on the aggregate result. The
next graph frontier can depend on an accepted bridge. Those calls cannot be
precomputed as independent seeds without changing the policy. A first batch
may do work that an early failure would have avoided; record that speculative
work and charge it to the measured component or Finish.

Each lane needs private moving points, pose, normal-equation partials/totals,
metrics, convergence flags, and result ownership. Share only immutable target
XYZ/normals/index buffers and source inputs. Account the combined cache,
per-lane scratch, temporary query packets, and graph storage under explicit
byte caps. Retain owners until selected-device completion before eviction or
reuse. A budget fallback must occur before executing the declared batch and
must be an explicit policy outcome, not a swallowed device fault.

## Device iteration seam

Retain the existing three scales (.12 / .06 / .03 metres), maximum updates
(40 / 30 / 20), Huber terms, strict radius test, original double distance metric,
pose convention, and fitness/RMSE convergence tests in the first experiment.
There is an initial query/equation evaluation at each scale before updates;
omitting it changes the algorithm.

A useful graph chunk would contain transforms, exact neighbour retrieval,
classification, normal equations, LDLT/update, and device convergence control.
Keep a lane active until its original stopping condition is met; subsequent
nodes for a finished lane must perform no further numerical update. Download
terminal results or a bounded exception packet rather than equation matrices
after every ordinary iteration. Charge graph construction, parameter updates,
target preparation, captures, and synchronization honestly.

Nearest-neighbour ambiguity remains a required CPU boundary. Unsupported rows,
ties, radius-boundary cases, malformed results, or numerical solve uncertainty
must prevent that lane's equation/pose update. Resolve legitimate ambiguous
rows using the original CPU search, then resume from the first blocked
iteration. Placeholder correspondences must never reach a solve. Damaged
execution is a hard error; it cannot silently retry a complete call or produce
a successful reconstruction.

This needs a new owned execution path. The current synchronous host copies,
CPU solver, null-stream orchestration, and mutable cache lifetimes cannot simply
be wrapped in a graph. Use explicit stream/event ownership, stable buffers,
bounded cached plans, and reviewed capture boundaries. A persistent kernel
with a global synchronization scheme is a larger change and should follow a
successful small graph/batch experiment rather than precede it.

## Quality and authority for a new experiment

Keep all old exact-history failures failed. C6 timing diverged only in the seed
at call 228 after original CPU information reductions changed by about 4.55e-13,
graph inputs by about 3.64e-12, and optimized poses by 3.33e-16. Canonical FPFH
cannot by itself remove that seed dependency. Neither rounding an expected seed
nor enlarging the old tolerance creates legitimate timing authority.

A separately named method/protocol should first audit every actual GPU query
against the original CPU on the same actual input, and every actual registration
against an original CPU result shadow with identical arrays and seed. Preserve
original source-bound acceptance gates and evaluate actual retained graph
edges, connectivity, witness membership, coverage, ambiguity and budget
outcomes. Require unrounded Final view poses within 0.5 mm / 0.1 degrees of the
matching native control and fixed-coordinate 30,000-point physical surface
checks: p95 at most 0.5 mm and at least .999 precision/completeness within 5 mm.
Mesh similarity alone does not establish graph authority.

If device updates or reductions change the method, name that change explicitly,
provide a CPU implementation/shadow of the declared method, and retain the
original reconstruction gates as independent acceptance checks. New timing
must consume a new source/runtime/domain/ownership proof. It must also close
its actual decisions and Final graph/pose/surface quality; an old component or
failed exact-history token cannot authorize the new path.

## Executed measurements and remaining integration

Independent two/four-seed batching, complete device iterations, short graph
chunks, persistent setup and owned immutable proof references have all been
implemented and physically measured. The [executed results](GPU_ICP_MICROBATCH_RESULTS.md)
retain each distinct source, fresh actual-input CPU/query/gate audit and
separate charged timing scope. The first easy complete-pair cases were close
to CPU performance. The same complete GPU method reduced time by 38.9% on the
expensive accepted session-7 pair 10,11, by 26.0% on rejected pair 0,3, and by
24.9% on rejected pair 0,6. Their fresh audits covered all 442 GPU calls and
266,311,538 nearest rows, with unchanged original query/result/gate checks.
Independent CPU near-seed reuse reduced the nine-pair complete phase by 7.0%.
These results establish useful component behaviour, not a production backend.

The smallest whole-Finish study can keep the original frontier, proposal order,
ambiguity and other stages unchanged while routing only calls in the original
bridge-verification subtree through the complete GPU method. Enter after a
verified current raw-Live checkpoint. Retain one pristine workspace across
Finish and one immutable aggregate/camera cache across all competing proposals
for the same directed fragment pair. Complete every selected stream, lane and
lease before replacing the pair cache. Non-bridge calls remain original CPU.

A new scoped dispatcher and ContextVar bypass can preserve imported original
`_match` aliases. Capture the original verifier and use a wrapper only to
establish pair context, returning its original tuple/object unchanged. CPU
shadows call the captured original implementation under bypass. Original
ownership guards reject these new wrappers, so this requires a separately
reviewed explicit wrapper guard rather than disabling an old check.

The new experiment must bind the current core, archive, settings, checkpoint,
loaded native libraries, thread policy, method sources and complete actual
trajectory. Audit every actual query and result, then compare an independent
original Finish from the same checkpoint using all graph/witness/pose/surface
criteria above. Include preparation, exact hashing, per-call capture, copying,
validation and cleanup inside Finish wall time. Keep the 256 MiB pair cache,
512 MiB retrieval/lane ceiling and 4,096-job workspace limit explicit, while
observing native fusion allocation separately.

Existing pair permits and older checkpoint/component authorities cannot unlock
this whole-Finish route. CPU information reductions and graph optimization may
still change later seed bytes between its own audit and timing. Preserve a
timing refusal even if Final quality passes; the fixed-pair CPU near-seed
comparison does not authorize changed clouds or an entire GPU Finish history.
Production integration remains unmeasured.

## Next experiment: share immutable geometry search indexes

The whole-Finish strict study and direct control are now executed for sessions
5 and 6; see [the measured results](GPU_ICP_MICROBATCH_RESULTS.md#complete-finish-strict-audit-and-observer-overhead).
The distinct current-core lower-overhead candidate and reproduction commands
are in the [research README](README.md#current-core-whole-finish-gpu-candidate).
Its physical quality and complete Finish timings remain pending an exclusive
hardware window. Historical strict failures remain failed.

A separate useful target is repeated construction of target search indexes.
Open3D 0.20 constructs a target KD-tree on each `EvaluateRegistration` call
and each information-matrix call, even when the target geometry is unchanged.
This follows directly from the
[pinned registration implementation](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/pipelines/registration/Registration.cpp#L128).
The completed session-5 GPU audit recorded 105 original held-out checks with
both directions evaluated, hence 210 native geometry evaluations. The direct
CPU control recorded 9.234 seconds in fragment reconnection. Tree construction
alone has not been timed, so those observations establish repeated work,
not the size of a possible gain.

After the candidate benchmarks, measure target-tree construction separately
from transformation, search and reduction on a genuine verifier trajectory.
If material, test a bounded native evaluator that caches only immutable owned
target points and their index. Preserve original queries, strict radii,
correspondence ordering, reductions, thresholds and every proposal verdict.
Invalidate on point replacement or mutation; retain owners through completion.
Compare every actual gate result against the fresh original API before timing.
Keep pose-dependent answers fresh, including independent witnesses and graph
pose revalidation. The Python registration API has no injected-tree parameter;
per-row Python searches would add a new overhead.

The smaller related seam is `refinement._distance`: Open3D also constructs a
target tree on each point-cloud distance call in its
[pinned implementation](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/geometry/PointCloud.cpp#L123).
Preserve the original square-root, clamp, square and mean sequence. Session 5
recorded only six such calls, so held-out verifier evaluations are the first
measurement target. Descriptor matches are already cached, and fragment
unions/FPFH are prepared once per fragment.
