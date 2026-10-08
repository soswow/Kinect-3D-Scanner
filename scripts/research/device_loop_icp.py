"""Bounded complete-device ICP iterations, a separately declared experiment.

No scanner integration and no historical proof permit. Numerical imports are
lazy. The first constructor is exhaustive NN-audit by default; timing requires
a fresh experiment authorizer on the exact consumed arrays/seed/source/config.
GPU LDLT, scalar pose composition and convergence remain a NEW evaluator even
when guarded transform/equation/retrieval bytes recover their original sources.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
POLICY = "explicit-stream-chunked-device-icp-v1"
STAGES = ((.12, 40), (.06, 30), (.03, 20))
MAX_POINTS = 1_000_000
BLOCK = 128
PHASES = ("prepare", "solve", "query", "equations", "nn_block", "solve_block", "done", "fault")
RESIDENT = ROOT / "scripts/research/archive/research_resident_icp.py"
FLAT = ROOT / "scripts/research/archive/research_flat_grid_nn.cu"
CLASSIFIER = ROOT / "scripts/research/research_device_flat_grid_nn.cu"
LDLT = ROOT / "scripts/research/research_device_ldlt.cu"
CONTROL = Path(__file__).with_name("device_loop_control.cu")


class DeviceLoopError(RuntimeError):
    """Hard execution/provenance/audit error: the lane cannot be retried."""


class DeviceLoopBlocked(DeviceLoopError):
    """A rejected solve, with actual system evidence; never a successful fallback."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_math_source():
    tree = ast.parse(RESIDENT.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == "GPU_SOURCE" for t in n.targets)]
    if len(nodes) != 1:
        raise DeviceLoopError("Original resident mathematical literal is ambiguous")
    value = ast.literal_eval(nodes[0].value)
    if not isinstance(value, str) or not value:
        raise DeviceLoopError("Original resident mathematical literal is not text")
    return value


def edited(original, replacements):
    """Recover every original source byte after narrowly adding control seams."""
    value = original
    for old, new in replacements:
        if value.count(old) != 1 or old == new or (new in value and new not in old):
            raise DeviceLoopError("Kernel source differs from the reviewed exact seam")
        value = value.replace(old, new)
    recovered = value
    for old, new in reversed(replacements):
        if recovered.count(new) != 1:
            raise DeviceLoopError("Guarded kernel source has ambiguous ownership")
        recovered = recovered.replace(new, old)
    if recovered != original:
        raise DeviceLoopError("Control injection changed original mathematical source")
    return value


def generated_source():
    math_source = edited(original_math_source(), (
        ("void transform_points(", "void loop_transform_points("),
        ("const double* original, double* moving, const double* pose, int count) {",
         "const double* input, double* moving, const double* accumulated, int count, "
         "const double* update, const int* control) {\n"
         "    if (control[0]!=PREPARE && control[0]!=QUERY) return;\n"
         "    const double* original=control[0]==PREPARE?input:moving;\n"
         "    const double* pose=control[0]==PREPARE?accumulated:update;\n"
         "    if(control[0]==PREPARE) {\n"
         "        bool identity=true; for(int i=0;i<16;i++) identity=identity && pose[i]==(i%5==0?1.:0.);\n"
         "        if(identity) {int r=blockIdx.x*blockDim.x+threadIdx.x;\n"
         "            if(r<count) for(int j=0;j<3;j++) moving[3*r+j]=input[3*r+j]; return;}\n"
         "    }"),
        ("void normal_partials(", "void loop_normal_partials("),
        ("double radius_squared, double* partials) {",
         "double* partials, const int* control) {\n"
         "    if(control[0]!=EQUATIONS) return;\n"
         "    const double radius_squared=loop_radius_squared(control[1]);"),
        ("void collapse_partials(", "void loop_collapse_partials("),
        ("const double* partials, double* totals, int blocks) {",
         "const double* partials, double* totals, int blocks, const int* control) {\n"
         "    if(control[0]!=EQUATIONS) return;"),
    ))
    flat = edited(FLAT.read_text(encoding="utf-8"), (
        ("void flat_grid_nearest_two(", "void loop_flat_grid_nearest_two("),
        ("const double *points,const int *ids,const unsigned long long *keys,\n"
         "    const unsigned int *offsets,unsigned int cells,\n"
         "    const double *queries,unsigned int count,int shift,double *output) {",
         "const unsigned long long* tables,const int* shifts,\n"
         "    const double *queries,unsigned int count,double *output,const int* control) {\n"
         "    if(control[0]!=PREPARE && control[0]!=QUERY) return;\n"
         "    int stage=control[1],shift=shifts[stage];\n"
         "    const unsigned long long* table=tables+5*stage;\n"
         "    const double* points=(const double*)table[0];\n"
         "    const int* ids=(const int*)table[1];\n"
         "    const unsigned long long* keys=(const unsigned long long*)table[2];\n"
         "    const unsigned int* offsets=(const unsigned int*)table[3];\n"
         "    unsigned int cells=(unsigned int)table[4];\n"
         "    if(!points) {if(threadIdx.x==0 && blockIdx.x<count) {\n"
         "        unsigned int r=blockIdx.x; output[5*r]=-3.;output[5*r+1]=-1.;\n"
         "        output[5*r+2]=output[5*r+3]=__longlong_as_double(0x7ff0000000000000LL);\n"
         "        output[5*r+4]=0.;}return;}"),
    ))
    classifier = edited(CLASSIFIER.read_text(encoding="utf-8"), (
        ("void classify_flat_results(", "void loop_classify_flat_results("),
        ("const double *raw,unsigned int count,unsigned int target_count,double r2,",
         "const double *raw,unsigned int count,unsigned int target_count,"),
        ("unsigned int *reasons,unsigned int *flagged,unsigned long long *totals) {",
         "unsigned int *reasons,unsigned int *flagged,unsigned long long *totals,const int* control) {\n"
         "    if(control[0]!=PREPARE && control[0]!=QUERY) return;\n"
         "    const double r2=loop_radius_squared(control[1]);"),
    ))
    ldlt = edited(LDLT.read_text(encoding="utf-8"), (
        ("void device_ldlt_solve(", "void loop_device_ldlt_solve("),
        ("int* pivots_out,double* diagonal_out){",
         "int* pivots_out,double* diagonal_out,const int* control,const double* totals){\n"
         "    if(control[0]!=SOLVE || totals[28]==0.)return;"),
    ))
    return "\n".join((CONTROL.read_text(encoding="utf-8"), math_source, flat, classifier, ldlt))


def source_contract():
    source = generated_source()
    return {"policy": POLICY, "artifacts": {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path)
        for path in (Path(__file__), CONTROL, RESIDENT, FLAT, CLASSIFIER, LDLT)},
        "generated_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "original_math_guard_inverse": True, "stages": [list(s) for s in STAGES],
        "huber_m": .01, "relative_fitness": 1e-6, "relative_rmse": 1e-6,
        "options": ["--std=c++11", "--fmad=false"], "graph_option_supported": True,
        "arithmetic": "New device LDLT/libdevice Euler/FP64 scalar composition and convergence; original guarded transform, NN classifier, Huber partials and ordered collapse. Fresh original CPU trajectory evidence required."}


def scratch_forecast(count, target_count):
    if (type(count) is not int or type(target_count) is not int or
            not 0 <= count <= MAX_POINTS or not 0 <= target_count <= MAX_POINTS):
        raise ValueError("Unsupported bounded original point counts")
    # Source24N+moving24N; target/normals48M when not shared. NN raw40N,
    # IDs4N+metrics8N+reasons4N+flagged4N+filtered4N; partials240ceil(N/128).
    # Packet64N + corrected4N + row4N are allocated once and reused.
    lane = 184*count + 240*((count+127)//128) + 4096
    return {"lane_bytes": lane, "private_pair_bytes": 48*target_count,
            "combined_bytes": lane+48*target_count,
            "host_exception_packet_bytes": 64*count,
            "small_control_packet_bytes": 256,
            "excludes": "Existing retained grid/CPU-tree cache, allocator pools, native library workspace and Python objects"}


def checked_chunk(value):
    if type(value) is not int or value not in (1, 2, 4):
        raise ValueError("Use a bounded chunk of 1, 2 or 4 device steps")
    return value


def check_packet(values, count, target_count, *, audit_hits, audit_misses):
    if (len(values) != 10 or any(type(v) is not int or not 0 <= v < 2**64 for v in values)
            or type(count) is not int or type(target_count) is not int
            or not 0 < count <= MAX_POINTS or not 0 < target_count <= MAX_POINTS
            or type(audit_hits) is not bool or type(audit_misses) is not bool):
        raise DeviceLoopError("Malformed bounded NN counter packet")
    flag, hit, miss, fallback, unsupported, uncertain, visits, malformed, ah, am = values
    if (malformed or flag != fallback+ah+am or flag>count or hit+miss+fallback!=count
            or unsupported>fallback or uncertain>fallback or visits>count*target_count
            or ah!=(hit if audit_hits else 0) or am!=(miss if audit_misses else 0)):
        raise DeviceLoopError("NN counts cannot authorize equations or CPU recovery")
    return flag


def control_state(values):
    if (len(values) != 8 or any(type(v) is not int for v in values) or
            values[0] not in range(8) or not 0 <= values[1] <= 3 or
            not 0 <= values[2] <= 40 or not 0 <= values[3] <= 90 or
            not 0 <= values[5] <= 93 or values[6] not in (0, 2)):
        raise DeviceLoopError("Malformed device iteration control")
    phase, stage = values[:2]
    if ((phase == 6) != (stage == 3) or
            stage < 3 and values[2] > STAGES[stage][1] or
            phase == 5 and values[4] not in (2, 3, 4, 5, 6, 8, 9)):
        raise DeviceLoopError("Device iteration exceeded stage or solve boundaries")
    return {"phase": PHASES[phase], "stage": stage, "stage_updates": values[2],
            "updates": values[3], "error": values[4], "queries": values[5]}


def check_reason_counts(counts, counters):
    """Cross-bind the actual CPU packet masks to the device counter packet."""
    if (len(counts) != 8 or len(counters) != 10 or
            any(type(v) is not int or v < 0 for v in [*counts, *counters])):
        raise DeviceLoopError("Malformed NN reason-mask accounting")
    cpu, hits, misses, unsupported, uncertain, both_audits, cpu_audit, direct_miss_bit = counts
    if ((cpu, hits, misses, unsupported, uncertain) !=
            (counters[3], counters[8], counters[9], counters[4], counters[5]) or
            both_audits or cpu_audit or direct_miss_bit):
        raise DeviceLoopError("NN packet reason masks differ from actual classifier counts")


def preserved_cleanup(primary, cleanup):
    if primary is not None:
        note = getattr(primary, "add_note", None)
        if callable(note):
            note("Device-loop cleanup also failed: "+repr(cleanup))
        raise primary from cleanup
    raise cleanup


class DeviceLoopICP:
    """One private lane; target cache ownership remains with its caller.

    start/advance/result are available to a future batch scheduler. The caller
    must hold an exclusive cache lease throughout the lane; this class never
    mutates/evicts the cache during device steps. Failed lanes cannot restart.
    Timing authorizer is a NEW experiment callback, executed before allocation;
    it must close exact inputs/current source/runtime. No old token is accepted.
    """
    def __init__(self, retrieval, *, device=0, stream=None, max_points=MAX_POINTS,
                 max_scratch_bytes=256*1024**2, max_total_bytes=512*1024**2,
                 audit_nearest=True, audit_misses=True,
                 timing_authorizer=None, audit_observer=None, cuda_graph=False):
        if (type(device) is not int or device < 0 or type(max_points) is not int or
                not 1 <= max_points <= MAX_POINTS or type(max_scratch_bytes) is not int or
                not 1 <= max_scratch_bytes <= 512*1024**2 or
                type(max_total_bytes) is not int or not 1 <= max_total_bytes <= 1024*1024**2 or
                type(cuda_graph) is not bool or
                type(audit_nearest) is not bool or type(audit_misses) is not bool or
                audit_nearest != audit_misses or
                not audit_nearest and not callable(timing_authorizer)):
            raise ValueError("Require explicit device/budgets and exhaustive audit or fresh timing authorizer")
        if getattr(retrieval, "device_id", None) != device:
            raise ValueError("Retrieval and lane must own the same selected CUDA device")
        import cupy as cp
        import numpy as np
        self.cp, self.np, self.retrieval, self.device = cp, np, retrieval, device
        self.max_points, self.max_scratch_bytes = max_points, max_scratch_bytes
        self.max_total_bytes = max_total_bytes
        self.audit_nearest, self.audit_misses = audit_nearest, audit_misses
        self.cuda_graph = cuda_graph
        self.configuration = (device, max_points, max_scratch_bytes, max_total_bytes, audit_nearest, audit_misses, cuda_graph)
        self.authorizer, self.audit_observer = timing_authorizer, audit_observer
        self.buffers, self.lease, self.failure = {}, None, None
        self.grid_owners = []
        self.graph = None
        self.graph_chunk = None
        self.pending_host_packets = None
        self.started = self.closed = False
        self.provenance = source_contract()
        self.statistics = {k: 0 for k in ("chunks", "enqueued_steps", "control_copies", "queries",
            "query_rows", "updates", "direct_hits", "direct_misses", "audited_hits", "audited_misses",
            "cpu_ambiguity_rows", "flagged_rows", "packet_bytes", "solve_blocks", "peak_lane_bytes",
            "graph_captures", "graph_launches", "maximum_graph_nodes")}
        self.statistics.update({k: 0. for k in ("setup_s", "chunk_wall_s", "cpu_nn_s", "cleanup_s", "graph_capture_s")})
        begin = time.perf_counter()
        with cp.cuda.Device(device):
            self.stream = stream if stream is not None else cp.cuda.Stream(non_blocking=True)
            if not isinstance(self.stream, cp.cuda.Stream) or self.stream.device_id != device:
                raise ValueError("Require an explicit CuPy lane stream")
            with self.stream:
                self.module = cp.RawModule(code=generated_source(), options=("--std=c++11", "--fmad=false"))
                self.module.compile()
                names = ("loop_transform_points", "loop_normal_partials", "loop_collapse_partials",
                    "loop_flat_grid_nearest_two", "loop_classify_flat_results", "loop_device_ldlt_solve",
                    "loop_reset_counters", "loop_build_system", "loop_commit_update", "loop_classified",
                    "loop_resume_nn", "loop_metrics", "loop_small_packet", "pack_flat_cpu_rows",
                    "scatter_flat_cpu_results")
                self.kernels = {name: self.module.get_function(name) for name in names}
                self.stream.synchronize()
        self.provenance.update({"device": f"CUDA:{device}", "stream_ptr": int(self.stream.ptr),
            "cuda_graph": cuda_graph,
            "cupy": cp.__version__, "numpy": np.__version__,
            "driver_version": cp.cuda.runtime.driverGetVersion(),
            "runtime_version": cp.cuda.runtime.runtimeGetVersion()})
        self.statistics["setup_s"] = time.perf_counter()-begin

    def _launch(self, name, grid, block, args):
        self.kernels[name](grid, block, args)

    def _check_healthy(self, *, source_check=False):
        if self.closed or self.failure is not None:
            raise DeviceLoopError("Closed or damaged device lane cannot execute") from self.failure
        if self.configuration != (self.device, self.max_points, self.max_scratch_bytes,
                self.max_total_bytes, self.audit_nearest, self.audit_misses, self.cuda_graph):
            raise DeviceLoopError("Device-loop execution policy changed after construction")
        if source_check:
            for relative, digest in self.provenance["artifacts"].items():
                if file_hash(ROOT / relative) != digest:
                    raise DeviceLoopError("Device-loop source changed after construction")

    def _check_inputs(self):
        for key, array in zip(("source", "target", "normals", "seed"), self.host_inputs):
            descriptor = self.input_binding[key]
            if (array.dtype.str != descriptor["dtype"] or list(array.shape) != descriptor["shape"] or
                    hashlib.sha256(array.tobytes(order="C")).hexdigest() != descriptor["sha256"]):
                raise DeviceLoopError("Original device-loop input bytes changed during the trajectory")

    def _host_array(self, value, shape, label):
        np = self.np
        result = np.asarray(value)
        if result.dtype != np.float64 or result.shape != shape or not np.isfinite(result).all():
            raise ValueError("Require finite original float64 "+label)
        return np.ascontiguousarray(result)

    def start(self, source, target, initial, *, pair_lease=None):
        self._check_healthy(source_check=True)
        if self.started:
            raise DeviceLoopError("A lane owns exactly one trajectory; create another for the next seed")
        np, cp = self.np, self.cp
        n, m = len(source.points), len(target.points)
        if not 0 <= n <= self.max_points or not 0 <= m <= self.max_points:
            raise ValueError("Original cloud exceeds lane point cap")
        forecast = scratch_forecast(n, m)
        if forecast["combined_bytes"] > self.max_scratch_bytes:
            raise ValueError("Lane scratch budget rejects before allocation or iteration")
        host_source = self._host_array(source.points, (n, 3), "source points")
        host_target = self._host_array(target.points, (m, 3), "target points")
        normals = self._host_array(target.normals, (m, 3), "target normals")
        seed = self._host_array(initial, (4, 4), "unrounded seed")
        if (np.abs(host_source)>1e6).any() or (np.abs(host_target)>1e6).any():
            raise ValueError("Original coordinates exceed the declared finite domain")
        def desc(a):
            return {"sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest(),
                    "dtype": a.dtype.str, "shape": list(a.shape)}
        self.input_binding = {"source": desc(host_source), "target": desc(host_target),
            "normals": desc(normals), "seed": desc(seed), "configuration": {
                "device": f"CUDA:{self.device}", "max_points": self.max_points,
                "max_scratch_bytes": self.max_scratch_bytes, "stages": [list(s) for s in STAGES],
                "max_total_bytes": self.max_total_bytes,
                "cuda_graph": self.cuda_graph,
                "audit_nearest": self.audit_nearest, "audit_misses": self.audit_misses}}
        if not self.audit_nearest:
            permit = self.authorizer(self.input_binding, dict(self.provenance))
            if permit is None or permit is False:
                raise DeviceLoopError("Fresh device-loop timing authorization was not granted")
            self.timing_permit = permit
        self.source_owner, self.target_owner = source, target
        self.host_inputs = (host_source, host_target, normals, seed)
        self.n, self.m, self.blocks = n, m, (n+127)//128
        self.started = True
        self.statistics["peak_lane_bytes"] = forecast["combined_bytes"]
        if not n or not m:
            self.empty_result = SimpleNamespace(transformation=seed.copy(), fitness=0., inlier_rmse=0.,
                correspondence_set=np.empty((0, 2), np.int32))
            return self
        try:
            # Existing cache setup is synchronized before entering an explicit
            # lane stream. No cache methods are called again during iteration.
            if pair_lease is None:
                items = {r: self.retrieval._dataset(target, r) for r, _ in STAGES}
                with cp.cuda.Device(self.device), cp.cuda.Stream.null:
                    cp.cuda.Stream.null.synchronize()
                self.lease = {"target": target, "items": items}
            else:
                if (pair_lease.get("target") is not target or pair_lease.get("closed", False)
                        or pair_lease.get("device_id") != self.device):
                    raise ValueError("Shared pair lease has wrong target or is closed")
                self.lease = pair_lease
                items = pair_lease["items"]
                pair_lease["owner"].register_lease_stream(pair_lease, self.stream)
            retained = getattr(self.retrieval, "cache_bytes", 0)
            if pair_lease is not None:
                retained = pair_lease["owned_bytes"]
            if type(retained) is not int or retained < 0 or retained+forecast["combined_bytes"] > self.max_total_bytes:
                raise ValueError("Combined retained pair/index and lane byte cap rejects before lane allocation")
            table, shifts = [], []
            for radius, _ in STAGES:
                shift = -math.frexp(radius)[1]
                item = items[radius]
                if (item.get("target") is not target or item.get("digest") != bytes.fromhex(desc(host_target)["sha256"])):
                    raise DeviceLoopError("Cached target bytes differ from current original inputs")
                grid = item["grids"].get(shift)
                if grid is None:
                    table.append([0, 0, 0, 0, 0])
                else:
                    if any(a.device.id != self.device for a in grid):
                        raise DeviceLoopError("Grid storage belongs to another CUDA device")
                    # Independent references keep pointer-table storage alive
                    # even if a caller accidentally mutates the cache dict.
                    self.grid_owners.append(grid)
                    table.append([a.data.ptr for a in grid]+[len(grid[2])])
                shifts.append(shift)
            with cp.cuda.Device(self.device), self.stream:
                b = self.buffers
                for key, host in (("original", host_source), ("target", host_target), ("normals", normals), ("pose", seed)):
                    shared = None
                    if pair_lease is not None:
                        shared = {"original": pair_lease["source_points"],
                                  "target": items[.12].get("data"),
                                  "normals": pair_lease["normals"]}.get(key)
                    if shared is None:
                        b[key] = cp.asarray(host)
                    else:
                        if (not isinstance(shared, cp.ndarray) or shared.dtype != cp.float64
                                or shared.shape != host.shape or not shared.flags.c_contiguous
                                or shared.device.id != self.device):
                            raise DeviceLoopError("Shared lease has malformed original-order numeric storage")
                        shadow = cp.asnumpy(shared, stream=self.stream)
                        if shadow.tobytes() != host.tobytes():
                            raise DeviceLoopError("Shared immutable pair bytes differ from consumed original inputs")
                        b[key] = shared
                b.update({"moving": cp.empty((n, 3), cp.float64), "raw": cp.empty((n, 5), cp.float64),
                    "nearest": cp.empty(n, cp.int32), "squared": cp.empty(n, cp.float64),
                    "reasons": cp.empty(n, cp.uint32), "flagged": cp.empty(n, cp.uint32),
                    "filtered": cp.empty(n, cp.int32), "partials": cp.empty((self.blocks, 30), cp.float64),
                    "totals": cp.empty(30, cp.float64), "matrix": cp.empty((6, 6), cp.float64),
                    "gradient": cp.empty(6, cp.float64), "status": cp.empty(1, cp.int32),
                    "step": cp.empty(6, cp.float64), "update": cp.empty((4, 4), cp.float64),
                    "composed": cp.empty((4, 4), cp.float64), "pivots": cp.empty(6, cp.int32),
                    "diagonal": cp.empty(6, cp.float64), "control": cp.zeros(8, cp.int32),
                    "counters": cp.zeros(10, cp.uint64), "cumulative": cp.zeros(10, cp.uint64),
                    "metrics": cp.zeros(4, cp.float64),
                    "small_packet": cp.empty(32, cp.uint64), "packet": cp.empty((n, 8), cp.float64),
                    "corrected": cp.empty(n, cp.int32), "rows": cp.empty(n, cp.uint32),
                    "tables": cp.asarray(table, cp.uint64), "shifts": cp.asarray(shifts, cp.int32)})
                actual = sum(a.nbytes for a in b.values())
                if actual > forecast["combined_bytes"]:
                    raise DeviceLoopError("Declared lane forecast underestimated actual owned arrays")
            return self
        except BaseException as exc:
            self.failure = exc
            raise

    def _equations(self):
        b, np = self.buffers, self.np
        self._launch("loop_normal_partials", (self.blocks,), (128,), (
            b["moving"], b["target"], b["normals"], b["nearest"], b["filtered"],
            np.int32(self.n), np.int32(self.m), b["partials"], b["control"]))
        self._launch("loop_collapse_partials", (1,), (32,), (b["partials"], b["totals"],
            np.int32(self.blocks), b["control"]))
        self._launch("loop_metrics", (1,), (1,), (b["control"], b["totals"], np.uint32(self.n), b["metrics"]))

    def _enqueue_step(self):
        b, np = self.buffers, self.np
        self._launch("loop_build_system", (1,), (1,), (b["totals"], b["matrix"], b["gradient"],
            b["status"], b["update"], b["composed"], b["pose"], b["control"]))
        self._launch("loop_device_ldlt_solve", (1,), (1,), (b["matrix"], b["gradient"], b["pose"],
            np.int32(1), b["status"], b["step"], b["update"], b["composed"], b["pivots"],
            b["diagonal"], b["control"], b["totals"]))
        self._launch("loop_commit_update", (1,), (1,), (b["control"], b["status"], b["update"],
            b["composed"], b["pose"], b["metrics"]))
        self._launch("loop_transform_points", (self.blocks,), (128,), (b["original"], b["moving"],
            b["pose"], np.int32(self.n), b["update"], b["control"]))
        self._launch("loop_reset_counters", (1,), (1,), (b["control"], b["counters"]))
        self._launch("loop_flat_grid_nearest_two", (self.n,), (128,), (b["tables"], b["shifts"],
            b["moving"], np.uint32(self.n), b["raw"], b["control"]))
        self._launch("loop_classify_flat_results", (self.blocks,), (128,), (b["raw"],
            np.uint32(self.n), np.uint32(self.m), np.int32(1), np.int32(self.audit_nearest),
            np.int32(self.audit_misses), b["nearest"], b["squared"], b["reasons"], b["flagged"],
            b["counters"], b["control"]))
        self._launch("loop_classified", (1,), (1,), (b["control"], b["counters"], b["cumulative"],
            np.uint32(self.n), np.uint32(self.m), np.int32(self.audit_nearest), np.int32(self.audit_misses)))
        self._equations()

    def _read_control(self):
        cp, b = self.cp, self.buffers
        self._launch("loop_small_packet", (1,), (1,), (b["control"], b["counters"], b["cumulative"], b["metrics"], b["small_packet"]))
        words = cp.asnumpy(b["small_packet"], stream=self.stream)
        self.pending_host_packets = None  # This blocking copy proves previous H2D completion.
        self.statistics["control_copies"] += 1
        state = control_state([int(v) for v in words[:8]])
        state["counters"] = [int(v) for v in words[8:18]]
        state["cumulative_counters"] = [int(v) for v in words[18:28]]
        state["metrics"] = words[28:].view(self.np.float64).tolist()
        return state

    def _launch_graph(self, chunk_iterations):
        """One bounded graph/trajectory; fixed pointers and no capture-time copies.

        CuPy's graph owns CUDA graph/executable handles. Native handle bytes are
        not exposed by its public API; report this separately from owned arrays.
        No graph is retained across start() or across pair/seed allocations.
        """
        if self.graph is not None and self.graph_chunk != chunk_iterations:
            raise DeviceLoopError("One graph per lane: chunk size cannot change after capture")
        if self.graph is None:
            begin = time.perf_counter()
            self.stream.synchronize()
            self.stream.begin_capture()
            try:
                for _ in range(chunk_iterations):
                    self._enqueue_step()
            except BaseException as primary:
                try:
                    # End an invalidated capture before any later cleanup;
                    # failure is latched and no non-graph retry is attempted.
                    abandoned = self.stream.end_capture()
                    self.graph = abandoned
                except BaseException as cleanup:
                    preserved_cleanup(primary, cleanup)
                raise
            self.graph = self.stream.end_capture()
            self.graph_chunk = chunk_iterations
            self.statistics["graph_capture_s"] += time.perf_counter()-begin
            self.statistics["graph_captures"] += 1
            self.statistics["maximum_graph_nodes"] = 11*chunk_iterations
        self.graph.launch(stream=self.stream)
        self.statistics["graph_launches"] += 1

    def _resolve_nn(self, state):
        b, np, cp = self.buffers, self.np, self.cp
        count = check_packet(state["counters"], self.n, self.m,
            audit_hits=self.audit_nearest, audit_misses=self.audit_misses)
        if state["phase"] != "nn_block" or not count:
            raise DeviceLoopError("Only a flagged actual query can enter CPU resolution")
        self._launch("pack_flat_cpu_rows", ((count+127)//128,), (128,), (b["moving"], b["raw"],
            b["nearest"], b["reasons"], b["flagged"], np.uint32(count), b["packet"]))
        packet = cp.asnumpy(b["packet"][:count], stream=self.stream)
        if not np.isfinite(packet[:, :6]).all():
            raise DeviceLoopError("Nonfinite NN transport must never reach original CPU search")
        rows, ids, reasons = packet[:, 0], packet[:, 4], packet[:, 5]
        if (not np.equal(rows, np.floor(rows)).all() or (rows<0).any() or (rows>=self.n).any()
                or len(np.unique(rows))!=count or not np.equal(ids, np.floor(ids)).all()
                or ((ids < 0) & (ids != -1) & (ids != -3)).any() or (ids>=self.m).any()
                or not np.equal(reasons, np.floor(reasons)).all() or (reasons<1).any() or (reasons>31).any()):
            raise DeviceLoopError("Malformed exact NN row/ID/reason transport")
        rows = rows.astype(np.uint32); ids = ids.astype(np.int32); reasons = reasons.astype(np.uint32)
        check_reason_counts([int(np.count_nonzero(mask)) for mask in (
            reasons&7, reasons&8, reasons&16, reasons&1, reasons&2,
            ((reasons&8)!=0)&((reasons&16)!=0),
            ((reasons&7)!=0)&((reasons&24)!=0), reasons&4)], state["counters"])
        if (((reasons&8)!=0)&(ids<0)).any() or (((reasons&16)!=0)&(ids!=-1)).any():
            raise DeviceLoopError("Direct hit/miss packet has incompatible original ID")
        a, other = packet[:, 6], packet[:, 7]
        if (not ((np.isfinite(a)&(a>=0))|(a==np.inf)).all() or
                not ((np.isfinite(other)&(other>=0))|(other==np.inf)).all() or (a>other).any()):
            raise DeviceLoopError("Malformed original-distance NN transport")
        corrected = ids.copy()
        radius = STAGES[state["stage"]][0]
        item = self.lease["items"][radius]
        self._check_inputs()
        begin = time.perf_counter()
        for j in range(count):
            found, indices, _ = item["cpu"].search_hybrid_vector_3d(packet[j, 1:4], radius, 1)
            expected = int(indices[0]) if found else -1
            if expected < -1 or expected >= self.m:
                raise DeviceLoopError("Original CPU tree produced malformed nearest ID")
            if reasons[j]&24 and expected != ids[j]:
                raise DeviceLoopError("Actual direct hit/miss failed original CPU audit before correction")
            if reasons[j]&7:
                corrected[j] = expected
        self.statistics["cpu_nn_s"] += time.perf_counter()-begin
        self.statistics["flagged_rows"] += count
        self.statistics["packet_bytes"] += int(packet.nbytes)
        c = state["counters"]
        if self.audit_observer is not None:
            self.audit_observer({"stage": state["stage"], "query_index": state["queries"]-1,
                "radius": radius, "packet": packet, "corrected_ids": corrected,
                "counters": list(c), "target_sha256": self.input_binding["target"]["sha256"]})
        self.pending_host_packets = (rows, corrected, packet)
        b["rows"][:count].set(rows, stream=self.stream)
        b["corrected"][:count].set(corrected, stream=self.stream)
        self._launch("scatter_flat_cpu_results", ((count+127)//128,), (128,), (b["rows"], b["corrected"],
            np.uint32(count), b["moving"], b["target"], b["nearest"], b["squared"]))
        self._launch("loop_resume_nn", (1,), (1,), (b["control"],))
        self._equations()

    def advance(self, chunk_iterations=1):
        self._check_healthy()
        checked_chunk(chunk_iterations)
        if not self.started:
            raise DeviceLoopError("Start exact inputs before enqueuing device steps")
        if hasattr(self, "empty_result"):
            return {"phase": "done", "queries": 0, "updates": 0}
        begin = time.perf_counter()
        try:
            with self.cp.cuda.Device(self.device), self.stream:
                if self.cuda_graph:
                    self._launch_graph(chunk_iterations)
                else:
                    for _ in range(chunk_iterations):
                        self._enqueue_step()
                state = self._read_control()
                self.statistics["chunks"] += 1
                self.statistics["enqueued_steps"] += chunk_iterations
                if state["phase"] == "fault":
                    raise DeviceLoopError("Device lane hard fault: "+repr(state))
                if state["phase"] == "solve_block":
                    self.statistics["solve_blocks"] += 1
                    evidence = {"control": state, "matrix": self.cp.asnumpy(self.buffers["matrix"], stream=self.stream),
                        "gradient": self.cp.asnumpy(self.buffers["gradient"], stream=self.stream),
                        "previous_pose": self.cp.asnumpy(self.buffers["pose"], stream=self.stream)}
                    raise DeviceLoopBlocked("Rejected device LDLT; pose is unchanged and no complete-call retry is allowed", evidence)
                if state["phase"] == "nn_block":
                    self._resolve_nn(state)
                    state = self._read_control()
                    if state["phase"] == "fault":
                        raise DeviceLoopError("Actual NN resolution left invalid equations: "+repr(state))
                self.statistics["queries"] = state["queries"]
                self.statistics["query_rows"] = state["queries"]*self.n
                self.statistics["updates"] = state["updates"]
                c = state["cumulative_counters"]
                for name, value in (("direct_hits", c[1]), ("direct_misses", c[2]),
                        ("audited_hits", c[8]), ("audited_misses", c[9]), ("cpu_ambiguity_rows", c[3])):
                    self.statistics[name] = value
                self.last_state = state
                return state
        except BaseException as exc:
            self.failure = exc
            raise
        finally:
            self.statistics["chunk_wall_s"] += time.perf_counter()-begin

    def result(self):
        self._check_healthy(source_check=True)
        self._check_inputs()
        if hasattr(self, "empty_result"):
            return self.empty_result
        if not self.started or getattr(self, "last_state", {}).get("phase") != "done":
            raise DeviceLoopError("Only a fully completed three-stage trajectory has a result")
        cp, np = self.cp, self.np
        with cp.cuda.Device(self.device), self.stream:
            pose = cp.asnumpy(self.buffers["pose"], stream=self.stream)
            filtered = cp.asnumpy(self.buffers["filtered"], stream=self.stream)
        if not np.isfinite(pose).all() or (filtered < -1).any() or (filtered >= self.m).any():
            raise DeviceLoopError("Malformed terminal pose/correspondence packet")
        rows = np.flatnonzero(filtered >= 0).astype(np.int32)
        pairs = np.column_stack((rows, filtered[rows])).astype(np.int32, copy=False)
        return SimpleNamespace(transformation=pose, fitness=self.last_state["metrics"][0],
            inlier_rmse=self.last_state["metrics"][1], correspondence_set=pairs)

    def match(self, source, target, initial, *, chunk_iterations=1, pair_lease=None):
        checked_chunk(chunk_iterations)
        self.start(source, target, initial, pair_lease=pair_lease)
        # 93 evaluations at most; CPU audit resumes a blocked evaluation before
        # the next step, so each advance must make progress or hard-fail.
        for _ in range(94):
            if self.advance(chunk_iterations)["phase"] == "done":
                return self.result()
        self.failure = DeviceLoopError("Bounded trajectory exhausted without terminal completion")
        raise self.failure

    def close(self):
        if self.closed:
            return
        begin = time.perf_counter()
        try:
            with self.cp.cuda.Device(self.device), self.stream:
                self.stream.synchronize()
        except BaseException as cleanup:
            # Keep owners alive when completion is unproved. Caller must not
            # release the shared cache lease after this failure.
            self.failure = self.failure or cleanup
            preserved_cleanup(self.failure if self.failure is not cleanup else None, cleanup)
        else:
            with self.cp.cuda.Device(self.device):
                self.graph = None
                self.pending_host_packets = None
                self.buffers.clear()
                self.grid_owners = []
                self.lease = None
                self.closed = True
        finally:
            self.statistics["cleanup_s"] += time.perf_counter()-begin

    def report(self):
        return {"policy": POLICY, "provenance": self.provenance, "statistics": dict(self.statistics),
            "input_binding": getattr(self, "input_binding", None), "closed": self.closed,
            "failure": None if self.failure is None else repr(self.failure),
            "result_authority": False, "whole_finish_authority": False, "cuda_graph": self.cuda_graph,
            "graph_ownership": {"maximum_live_graphs": 1, "maximum_nodes": 44,
                "native_handle_bytes": "unobserved; CuPy/CUDA internal allocations separate from bounded numeric arrays",
                "release_after_selected_stream_completion": True}}
