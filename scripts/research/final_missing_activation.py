"""Research-only exact-size weighted Final allocation with missing-key activation.

Import is stdlib-only. The numerical runtime is supplied by an already loaded
engine when an explicitly allocated experiment constructs a scope. No existing
source, activation caller, fusion kernel or geometry authority is replaced.
"""

from __future__ import annotations

import array
import ast
import copy
import hashlib
from pathlib import Path
import sys
import time
from types import FunctionType, MethodType, ModuleType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.research import final_allocation_scope as original

KIND = "offline-weighted-final-missing-activation-v2"
MAX_KEYS = 50000
BASE_HELPER_SHA256 = "a6c6591a8d10352df81adb7e0177889b1e2ee94c316aad873c4d3be1b0f62153"
CALLER_AST = {
    "scanner_server/weighted_fusion.py": {
        "_integrate_cpu": "c461369a7edef7d08a2a6dc352f06794cade950a30d11ffc1b33ac8d43456593",
        "_integrate_tensor": "269d9fbc6d88aa6d56ab501a9e0a62642d98083387074b5c68344bff1cfb78c9"},
    "scanner_server/cuda_fusion.py": {
        "_integrate_validated": "58e6594564719729a7c1f9a1d923823f7377ecc3fdf7ce50b3dbd85b6a287785"}}
ENGINE_INTEGRATE_AST = "f5817978f509d94eb7bb267b1edff3141835e9e014bd34c5e277df8af2a03021"
ENGINE_PREPARE_AST = "02c81337572bafc5629be1a2cedde59cf7fbc02bd8428dc1e3b5a1c23cf8ec99"
PREPARATION_FIELDS = ("_input_preparation", "_confidence_preparation")


class MissingActivationContractError(RuntimeError):
    pass


def check(condition, message):
    if not condition:
        raise MissingActivationContractError(message)


def digest_keys(rows):
    """Canonical signed-int32 little-endian bytes; no float conversion."""
    values = array.array("i", (item for row in rows for item in row))
    check(values.itemsize == 4, "Host signed-int32 array layout is unsupported")
    if sys.byteorder != "little":
        values.byteswap()
    return hashlib.sha256(values.tobytes()).hexdigest()


class WeightedActivationSourceContract:
    """Pin the three return-unused callers, original dispatch, and base helper.

    Caller function code is checked at activation, including lazy CUDA imports.
    Source files are rehashed at Final boundaries, outside voxel update loops.
    """
    def __init__(self):
        self.caller_signatures = {}
        self.loaded_codes = {}
        self.paths = tuple(CALLER_AST) + (
            "scanner_server/engine.py", "scanner_server/weighted_fusion.cu",
            "scanner_server/cuda_input.py", "scanner_server/cuda_confidence.py",
            "scripts/research/final_allocation_scope.py")
        self.source_sha256 = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                              for path in self.paths}
        check(self.source_sha256["scripts/research/final_allocation_scope.py"] == BASE_HELPER_SHA256,
              "Original exact allocator helper changed")
        for path, names in CALLER_AST.items():
            text = (ROOT/path).read_text(encoding="utf-8")
            tree = ast.parse(text)
            compiled = compile(text, str(ROOT/path), "exec", dont_inherit=True)
            for name, expected in names.items():
                node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                check(hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == expected,
                      "Original weighted activation caller changed")
                calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "activate"]
                check(len(calls) == 1, "Expected one original activation per weighted caller")
                position = next((i for i, n in enumerate(node.body)
                                 if isinstance(n, ast.Expr) and n.value is calls[0]), None)
                check(position is not None and ast.unparse(calls[0]) == "hashmap.activate(blocks)"
                      and ast.unparse(node.body[position+1]) == "buffers, found = hashmap.find(blocks)",
                      "Activation output must be unused before the original full-frustum find")
                code = next(c for c in compiled.co_consts if hasattr(c, "co_name") and c.co_name == name)
                module = path[:-3].replace("/", ".")
                self.caller_signatures[module, name] = original.code_signature(code)
        engine_tree = ast.parse((ROOT/"scanner_server/engine.py").read_text(encoding="utf-8"))
        dispatch = next(n for n in ast.walk(engine_tree)
                        if isinstance(n, ast.FunctionDef) and n.name == "_integrate_vbg")
        check(hashlib.sha256(ast.dump(dispatch, include_attributes=False).encode()).hexdigest() == ENGINE_INTEGRATE_AST,
              "Original weighted/unweighted integration dispatch changed")
        preparation = next(n for n in ast.walk(engine_tree)
                           if isinstance(n, ast.FunctionDef) and n.name == "_prepare_input")
        check(hashlib.sha256(ast.dump(preparation, include_attributes=False).encode()).hexdigest() == ENGINE_PREPARE_AST,
              "Original input preparation dispatch changed")

    def caller(self, frame):
        code = frame.f_code
        key = frame.f_globals.get("__name__"), code.co_name
        expected = self.caller_signatures.get(key)
        function = frame.f_globals.get(code.co_name)
        check(expected is not None and getattr(function, "__code__", None) is code
              and getattr(function, "__closure__", None) is None,
              "Missing-only activation received an unreviewed or changed caller")
        if self.loaded_codes.get(key) is not code:
            check(original.code_signature(code) == expected,
                  "Missing-only activation caller code differs from pinned source")
            self.loaded_codes[key] = code

    def unchanged(self):
        current = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in self.paths}
        check(current == self.source_sha256, "Weighted activation source changed during the experiment")
        return current


class MissingOnlyHashMap:
    """Single private candidate's hashmap; every other operation is delegated.

    Only unique original int32 block coordinates are supported. Returned indices
    are the full post-activation find result, and masks mark previously absent
    keys. Existing-key index slots are therefore resolved rather than relying
    on the native activate output's unspecified failure slots. Reviewed callers
    discard the tuple. This is not general HashMap replacement authority.
    """
    def __init__(self, hashmap, *, core, device, physical_capacity, logical_limit,
                 source_contract, configuration):
        check(type(physical_capacity) is int and 1 <= physical_capacity <= MAX_KEYS
              and type(logical_limit) is int and physical_capacity <= logical_limit <= MAX_KEYS,
              "Invalid physical/logical candidate capacity")
        check(isinstance(source_contract, WeightedActivationSourceContract), "Missing source contract")
        self.native, self.core, self.device = hashmap, core, device
        self.physical_capacity, self.logical_limit = physical_capacity, logical_limit
        self.source_contract, self.configuration = source_contract, configuration
        self.initial_configuration = (True, str(device))
        self.calls, self.expected_keys, self.fault = [], set(), None
        check(str(device) in ("CPU:0", "CUDA:0"), "This bounded experiment supports CPU:0/CUDA:0 only")
        check(hashmap.device == device, "Candidate hashmap device differs from the engine")
        check(int(hashmap.capacity()) == physical_capacity and int(hashmap.size()) == 0,
              "Missing-only activation requires a new empty exact-size candidate")
        self._active()

    def _active(self):
        if self.fault is not None:
            raise self.fault
        check(self.configuration() == self.initial_configuration,
              "Weighted candidate policy/device changed during activation")

    def __getattr__(self, name):
        self._active()
        return getattr(self.native, name)

    def _tensor(self, value, *, shape, dtype, label):
        check(isinstance(value, self.core.Tensor) and tuple(value.shape) == tuple(shape)
              and value.dtype == dtype and value.device == self.device and value.is_contiguous(),
              f"Malformed/noncontiguous/wrong-device {label}")

    def _rows(self, keys):
        check(isinstance(keys, self.core.Tensor) and len(keys.shape) == 2
              and keys.shape[1] == 3 and 0 <= keys.shape[0] <= MAX_KEYS,
              "Original block keys must be bounded N x 3 tensors")
        self._tensor(keys, shape=(keys.shape[0], 3), dtype=self.core.int32, label="block keys")
        rows = tuple(tuple(row) for row in keys.cpu().numpy().tolist())
        check(len(rows) == keys.shape[0] and all(len(row) == 3 and all(type(x) is int and -2**31 <= x < 2**31 for x in row)
                  for row in rows), "Malformed signed-int32 block key values")
        check(len(set(rows)) == len(rows), "Original unique block coordinates contain duplicates")
        return rows

    def _find(self, keys, rows, *, all_found=False):
        indices, mask = self.native.find(keys)
        self._tensor(indices, shape=(len(rows),), dtype=self.core.int32, label="find indices")
        self._tensor(mask, shape=(len(rows),), dtype=self.core.bool, label="find masks")
        found = mask.cpu().numpy().tolist()
        ids = indices.cpu().numpy().tolist()
        check(len(found) == len(rows) and len(ids) == len(rows)
              and all(type(x) is bool for x in found), "Native find returned malformed masks")
        valid_ids = [item for item, present in zip(ids, found) if present]
        check(all(type(x) is int and 0 <= x < self.physical_capacity for x in valid_ids)
              and len(set(valid_ids)) == len(valid_ids), "Native find returned invalid/aliased buffer indices")
        check(not all_found or all(found), "Native activation omitted original frustum keys")
        return indices, mask, found

    def activate(self, blocks):
        started = time.perf_counter()
        row = {"complete": False}
        self.calls.append(row)
        try:
            self._active()
            self.source_contract.caller(sys._getframe(1))
            rows = self._rows(blocks)
            before_capacity, before_size = int(self.native.capacity()), int(self.native.size())
            check(before_capacity == self.physical_capacity and before_size == len(self.expected_keys),
                  "Candidate capacity/size changed outside controlled activation")
            check(0 <= before_size <= self.physical_capacity, "Malformed native candidate size")
            row.update(input_rows=len(rows), input_sha256=digest_keys(rows), size_before=before_size,
                       capacity_before=before_capacity, native_activate_called=False)
            if not rows:
                row.update(existing_rows=0, missing_rows=0, size_after=before_size,
                           capacity_after=before_capacity, complete=True)
                return (self.core.Tensor([], dtype=self.core.int32, device=self.device),
                        self.core.Tensor([], dtype=self.core.bool, device=self.device))
            _, _, found = self._find(blocks, rows)
            check(found == [key in self.expected_keys for key in rows],
                  "Native membership differs from the original requested key union")
            new = [not present for present in found]
            missing_count = sum(new)
            row.update(existing_rows=len(rows)-missing_count, missing_rows=missing_count)
            check(before_size+missing_count <= self.physical_capacity
                  and before_size+missing_count <= self.logical_limit,
                  "Missing block keys exceed the exact physical or configured logical budget")
            new_mask = self.core.Tensor(new, dtype=self.core.bool, device=self.device)
            if missing_count:
                missing = blocks[new_mask]
                self._tensor(missing, shape=(missing_count, 3), dtype=self.core.int32, label="missing keys")
                row["native_activate_called"] = True
                # No union preactivation: preserve original per-frame order and
                # initialize new native value storage through original Activate.
                # Retain both input and output tensors until the explicit
                # selected-device synchronization in finally, including faults.
                native_output = self.native.activate(missing)
            indices, _, _ = self._find(blocks, rows, all_found=True)
            after_capacity, after_size = int(self.native.capacity()), int(self.native.size())
            row.update(size_after=after_size, capacity_after=after_capacity)
            check(after_capacity == self.physical_capacity and after_size == before_size+missing_count,
                  "Native activation grew the physical capacity or changed the expected size")
            check(self._rows(blocks) == rows, "Original block key tensor changed during activation")
            self.expected_keys.update(rows)
            row["complete"] = True
            return indices, new_mask
        except BaseException as error:
            self.fault = error
            row["failure"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            primary = sys.exception()
            try:
                if str(self.device) == "CUDA:0":
                    try:
                        self.core.cuda.synchronize(self.device)
                        row["cuda_synchronized"] = True
                    except BaseException as secondary:
                        row["complete"] = False
                        row["cuda_synchronized"] = False
                        row["synchronization_failure"] = {"type": type(secondary).__name__, "message": str(secondary)}
                        self.fault = primary if primary is not None else secondary
                        if primary is not None:
                            primary.add_note(f"Candidate activation synchronization also failed: {secondary}")
                            raise primary from secondary
                        raise
            finally:
                row["wall_s"] = time.perf_counter()-started

    def verify_final(self, required_blocks):
        self._active()
        check(type(required_blocks) is int and required_blocks == len(self.expected_keys),
              "Final requested key union differs from the exact original planner")
        check(int(self.native.capacity()) == self.physical_capacity
              and int(self.native.size()) == required_blocks, "Final capacity/active count changed")
        if required_blocks:
            ids = self.native.active_buf_indices()
            self._tensor(ids, shape=(required_blocks,), dtype=self.core.int32, label="active indices")
            id_rows = ids.cpu().numpy().tolist()
            check(len(set(id_rows)) == len(id_rows)
                  and all(type(x) is int and 0 <= x < self.physical_capacity for x in id_rows),
                  "Malformed active buffer indices")
            keys = self.native.key_tensor()[ids.to(self.core.int64)]
            active = set(self._rows(keys))
        else:
            active = set()
        check(active == self.expected_keys, "Native active keys differ from the original ordered input union")
        return {"expected_union_sha256": digest_keys(sorted(self.expected_keys)),
                "actual_union_sha256": digest_keys(sorted(active)), "exact_key_union": True,
                "required_blocks": required_blocks, "initial_capacity": self.physical_capacity,
                "final_capacity": int(self.native.capacity()), "final_blocks": int(self.native.size())}

    def report(self):
        return {"calls": list(self.calls), "fault": None if self.fault is None else
                {"type": type(self.fault).__name__, "message": str(self.fault)},
                "input_rows": sum(row.get("input_rows", 0) for row in self.calls),
                "existing_rows": sum(row.get("existing_rows", 0) for row in self.calls),
                "missing_rows": sum(row.get("missing_rows", 0) for row in self.calls),
                "native_activate_calls": sum(row.get("native_activate_called", False) for row in self.calls),
                "activation_wall_s": sum(row["wall_s"] for row in self.calls),
                "physical_capacity": self.physical_capacity, "logical_limit": self.logical_limit,
                "retained_key_count": len(self.expected_keys),
                "memory_scope": "At most 50,000 Python key tuples plus bounded tensor/download temporaries; native attribute/hashmap memory and allocator reservations are separate. No RSS claim."}


class MissingOnlyGrid:
    """Private adapter; native attribute/geometry methods use the actual grid."""
    def __init__(self, volume, **kwargs):
        self.volume = volume
        self.controlled_hashmap = MissingOnlyHashMap(volume.hashmap(), **kwargs)

    def hashmap(self):
        self.controlled_hashmap._active()
        return self.controlled_hashmap

    def integrate(self, *args, **kwargs):
        error = MissingActivationContractError("Unweighted native VBG.integrate bypasses the activation proxy")
        self.controlled_hashmap.fault = error
        raise error

    def __getattr__(self, name):
        self.controlled_hashmap._active()
        return getattr(self.volume, name)


def private_prepare_input(owner):
    """Same engine method code, private original CPU calibration globals/LRUs."""
    method = owner._prepare_input.__func__
    text = (ROOT/"scanner_server/engine.py").read_text(encoding="utf-8")
    compiled = compile(text, str(ROOT/"scanner_server/engine.py"), "exec", dont_inherit=True)
    class_code = next(c for c in compiled.co_consts if hasattr(c, "co_name") and c.co_name == "ScanEngine")
    expected = next(c for c in class_code.co_consts if hasattr(c, "co_name") and c.co_name == "_prepare_input")
    check(method.__module__ == "scanner_server.engine" and method.__closure__ is None
          and original.code_signature(method.__code__) == original.code_signature(expected),
          "Loaded input preparation differs from the unchanged original dispatcher")
    namespace = dict(method.__globals__)
    namespace["prepare_rgbd"] = original.private_depth_preparation(namespace["prepare_rgbd"])
    local = FunctionType(method.__code__, namespace, method.__name__, method.__defaults__)
    local.__kwdefaults__ = method.__kwdefaults__
    return local


def private_final_owner(owner):
    """Detach Python containers and adapter dictionaries, retaining native owners.

    Existing warm native input/confidence operators are deliberately observed,
    never deep-copied. Their immutable lookup/kernel arrays remain shared. This
    does not promise native scratch/global allocator or kernel-cache rollback.
    """
    memo = {}
    proxy = copy.copy(owner)
    proxy.__dict__ = original.private_containers(owner.__dict__, memo)
    for field in PREPARATION_FIELDS:
        if field not in owner.__dict__:
            continue
        value = getattr(owner, field)
        check(hasattr(value, "__dict__"), "Preparation state cannot be privately copied")
        cloned = copy.copy(value)
        cloned.__dict__ = original.private_containers(value.__dict__, memo)
        setattr(proxy, field, cloned)
    proxy._prepare_input = MethodType(private_prepare_input(owner), proxy)
    return proxy


def preparation_owner_evidence(owner):
    """Owner/configuration observation only; shared native array bits unobserved."""
    def state(value):
        if value is None or type(value) in (str, int, bool):
            return {"scalar": value}
        if type(value) is float:
            return {"float": value.hex()}
        if isinstance(value, dict):
            check(len(value) <= 128, "Unbounded observed native configuration dictionary")
            return {"owner": id(value), "dict": {str(key): state(item) for key, item in value.items()}}
        if isinstance(value, (list, tuple)):
            check(len(value) <= 128, "Unbounded observed native configuration sequence")
            return {"owner": id(value), "items": [state(item) for item in value]}
        result = {"owner": id(value), "type": type(value).__module__+"."+type(value).__name__}
        # CuPy/NumPy modules expose a shape FUNCTION. They and opaque kernel
        # callables are configuration owners, not arrays. Do not traverse their
        # dictionaries, invoke functions or interpret callable names as shapes.
        if isinstance(value, ModuleType):
            result["module_name"] = value.__name__
            return result
        if callable(value):
            result["callable_name"] = str(getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__)))
            return result
        shape = getattr(value, "shape", None)
        # Actual retained NumPy/CuPy arrays expose a nonnegative integer shape
        # tuple. Generic objects with a shape method remain opaque owners. This
        # is metadata/owner evidence, never a claim about native array contents.
        if isinstance(shape, (list, tuple)) and all(type(item) is int and item >= 0 for item in shape):
            result["shape"] = list(shape)
            for name in ("dtype", "device", "nbytes"):
                selected = getattr(value, name, None)
                if selected is not None and not callable(selected):
                    result[name] = str(selected)
        return result
    result = {}
    for field in PREPARATION_FIELDS:
        if field in owner.__dict__:
            adapter = getattr(owner, field)
            gpu = getattr(adapter, "_gpu", None)
            result[field] = {"adapter_owner": id(adapter), "gpu_owner": None if gpu is None else id(gpu),
                             "gpu_state": None if gpu is None else state(vars(gpu))}
    return result


class MissingKeyFinalScope(original.RightSizedFinalScope):
    """Only candidate creation differs; original Final and surface commit remain."""
    def __init__(self, engine, *, original_final, on_plan=None):
        super().__init__(engine, original_final=original_final, on_plan=on_plan)
        self.contract = WeightedActivationSourceContract()
        self.grids, self.final_key_proofs = [], []
        self.used_final = False
        self.preparation_observations, self.private_telemetry = [], []

    def _weighted(self, owner):
        check(owner.settings.confidence_fusion is True,
              "Missing-key Final activation only supports original weighted fusion")
        check(str(owner.device) in ("CPU:0", "CUDA:0"), "Unsupported Final device")

    def __enter__(self):
        self._weighted(self.engine)
        self.contract.unchanged()
        return super().__enter__()

    def final(self, owner, progress_cb=None):
        check(owner is self.engine, "Missing-key scope received a different engine")
        self._weighted(owner)
        self.contract.unchanged()
        if owner.settings.final_voxel_m is None or owner._final_vbg is not None:
            return self.original_final(owner, progress_cb)
        check(not self.used_final, "A missing-key scope may allocate only one Final candidate")
        self.used_final = True
        before_preparation = preparation_owner_evidence(owner)
        plan = original.plan_final(owner, original_final=self.original_final)
        self.plans.append(plan)
        if self.on_plan is not None:
            self.on_plan(plan.report)
        if not plan.capacity_fits:
            raise original.FinalCapacityError(plan)
        self._weighted(owner)
        self.contract.unchanged()
        original_create = owner._create_vbg.__func__
        allocation_start = len(self.allocations)
        allocations, grids, contract = self.allocations, self.grids, self.contract
        core = sys.modules["open3d"].core

        class PrivateFinal(type(owner)):
            def _create_vbg(candidate, block_count=None):
                check(candidate.settings.confidence_fusion is True
                      and block_count == plan.configured_limit and len(allocations) == allocation_start,
                      "Original Final candidate allocation/policy changed")
                value = original_create(candidate, block_count=plan.allocated_blocks)
                actual = int(value.hashmap().capacity())
                allocations.append({"requested_blocks": block_count, "allocated_blocks": plan.allocated_blocks,
                                    "actual_capacity": actual})
                check(actual == plan.allocated_blocks, "Native candidate initial capacity differs from the plan")
                adapter = MissingOnlyGrid(value, core=core, device=candidate.device,
                    physical_capacity=plan.allocated_blocks, logical_limit=plan.configured_limit,
                    source_contract=contract,
                    configuration=lambda: (candidate.settings.confidence_fusion, str(candidate.device)))
                grids.append(adapter)
                return adapter

        proxy = private_final_owner(owner)
        proxy.__class__ = PrivateFinal
        try:
            result = self.original_final(proxy, progress_cb)
            check(len(allocations) == allocation_start+1 and result is grids[-1],
                  "Original Final omitted or replaced its controlled candidate")
            proof = result.hashmap().verify_final(plan.required_blocks)
            self.final_key_proofs.append(proof)
            allocations[-1]["final_capacity"] = proof["final_capacity"]
            self.contract.unchanged()
            check(preparation_owner_evidence(owner) == before_preparation,
                  "Original Live preparation/native data owners changed during private Final")
            if "final_reintegration" in proxy.stage_totals_ms:
                owner.stage_totals_ms["final_reintegration"] = proxy.stage_totals_ms["final_reintegration"]
            owner.final_reconstruction = proxy.final_reconstruction
            owner.final_reconstruction.update(required_blocks=plan.required_blocks,
                allocated_blocks=plan.allocated_blocks, configured_block_limit=plan.configured_limit,
                attribute_budget_mib=plan.allocated_blocks*original.ATTRIBUTE_BYTES_PER_BLOCK/2**20,
                configured_attribute_budget_mib=plan.configured_limit*original.ATTRIBUTE_BYTES_PER_BLOCK/2**20,
                planning_wall_s=plan.planning_wall_s, allocator_scope=KIND,
                missing_activation_key_proof=dict(proof))
            # No proxy commits into the engine: surface extraction and all later
            # operations receive the actual original Open3D VoxelBlockGrid.
            return result.volume
        except BaseException as error:
            if len(grids) > allocation_start and grids[-1].controlled_hashmap.fault is not None:
                fault = grids[-1].controlled_hashmap.fault
                if fault is not error:
                    fault.add_note(f"Original Final also failed after activation: {error}")
                    raise fault from error
            raise
        finally:
            primary = sys.exception()
            try:
                after_preparation = preparation_owner_evidence(owner)
                self.preparation_observations.append({"before": before_preparation, "after": after_preparation,
                    "data_owners_and_configuration_unchanged": before_preparation == after_preparation,
                    "native_array_bit_rollback_proven": False})
                self.private_telemetry.append({"backend": original.private_containers(proxy.backend),
                    "stage_totals_ms": original.private_containers(proxy.stage_totals_ms),
                    "preparation": {field: {"status": original.private_containers(getattr(getattr(proxy, field), "status", {})),
                        "private_adapter_owner": id(getattr(proxy, field)),
                        "private_gpu_owner": None if getattr(getattr(proxy, field), "_gpu", None) is None
                            else id(getattr(getattr(proxy, field), "_gpu"))}
                        for field in PREPARATION_FIELDS if field in proxy.__dict__}})
                check(after_preparation == before_preparation, "Shared Live preparation/native owner state changed")
            except BaseException as error:
                if primary is not None:
                    primary.add_note(f"Final preparation observation also failed: {error}")
                    raise primary from error
                raise

    def report(self):
        result = super().report()
        result.update(kind=KIND, weighted_only=True, original_final_body=True,
            native_grid_returned=bool(self.final_key_proofs), source_sha256=dict(self.contract.source_sha256),
            activation=[grid.controlled_hashmap.report() for grid in self.grids],
            final_key_proofs=list(self.final_key_proofs),
            preparation_observations=list(self.preparation_observations),
            private_final_telemetry=list(self.private_telemetry),
            allowed_shared_effects="Native allocator pools, compiled immutable kernel caches and any opaque native scratch contents are not copied or claimed to roll back. Existing native data owners/configuration are observed; usable input parity requires an allocated proof. Python calibration LRUs and adapter/container state are private, with actual success/failure work reported separately.",
            performance_measured=False, geometry_proven=False,
            note="Source-only weighted activation experiment. Native CPU/CUDA capacity, key-mapped voxel attributes and whole-session mesh evidence remain required. Existing Finish/NN proofs do not authorize it.")
        return result
