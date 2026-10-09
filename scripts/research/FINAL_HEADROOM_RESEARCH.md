# Original Final activation with reserve headroom

The v1 exact-unique allocation failed its physical C5 check: a 3328-block
initial capacity grew to 6656 during the unchanged Final integration even
though the unique block count was 3328. Its capacity guard rejected Finish
and retained the Live geometry/pose owners. The failed report and both v1
source files stay immutable. The earlier proposed attribute savings are not
established by that run.

The [official Open3D v0.19 HashMap activation implementation](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/core/hashmap/HashMap.cpp)
checks current unique size plus the entire incoming row count before it
inserts keys. When that estimate exceeds capacity it reserves at least twice
the old capacity. Thus repeated keys can trigger growth without increasing
the actual unique size. This published rule explains the observed v0.20
behavior; the selected installed binary still needs physical verification.

`final_allocation_headroom.py` declares a separate v2 policy. It observes the
exact frustum coordinate arrays already returned by the original one-block
planner and computes two counts: the final exact unique union and
`max(prefix_unique_before_frame + incoming_frustum_rows)`. The latter is the
physical reserve capacity, with an empty union allocated capacity one. It is
tighter than `final_unique + largest_frustum` because each prefix excludes
future frames. The observer returns the identical original coordinate arrays
to the unchanged planner, uses a private engine/container/calibration-cache
copy, and confirms that scratch never activates voxel blocks.

Logical and physical budgets are separate. The existing configured
`final_block_count` still limits unique accepted blocks in original fusion.
The new experiment requires an explicit `physical_budget` (at most 100000)
and reports it separately. If the union exceeds the logical limit, fail with
the full required unique count before candidate allocation. If its reserve
bound exceeds the physical budget, fail before allocation with both counts.
Do not clamp an insufficient reserve bound and hope automatic growth stays
within budget. A planned physical reserve can be greater than the configured
logical cap; that is visible additional attribute memory, not a saving.

`HeadroomFinalScope(engine, original_final=..., physical_budget=...)` retains
the original Final integration, confidence, activation, truncation, helpers,
backend counters, accepted poses and transactional surface commit. It changes
only the private candidate allocator's initial physical capacity. Initial
and final actual capacities must equal the planned reserve, and the actual
integrated unique count must equal the planned union. Any surprise growth or
key-count mismatch rejects the experiment before committing the Final
surface. `ObservedOriginalHeadroomScope` gives the matched control the same
planner overhead and delegates its original allocation/overflow behavior.
Both scopes are instance-only and restore the original hook.

Fourteen stdlib contracts reproduce the duplicate-key resize, show that the
tight reserve avoids it on identical synthetic keys/update order, distinguish
logical and physical early errors, verify private planning/unchanged 5 mm
Final and 80 mm truncation, and inject initial-capacity, post-write growth and
missing-key failures. They do not prove actual CUDA/Open3D capacity, geometry
quality or a memory/runtime gain. A new producer must pin this helper, its
immutable measured-v1 dependency and all checkpoint/observer sources under
a distinct report kind. It must use the checkpoint's original RANSAC policy,
run original CPU registration for both controls, charge planning/setup inside
Finish, and compare original accepted poses/gates and fixed-coordinate mesh
surface quality. Attribute memory is only a logical floor; native pools and
RSS/GPU measurements are additional evidence.

The long-term missing-keys-only activation idea is a different experiment:
it could avoid this pessimistic reserve but changes activation inputs and
requires new returned-index/mask, initialization and whole-mesh proofs. V2
does not implement that change or modify any production integration source.
