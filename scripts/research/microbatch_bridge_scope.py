"""Private original bridge verifier with a separately supplied ICP evaluator.

The original function code is cloned into a private namespace. Only _match is
rebound; output-only observers call each original helper once and return its
original object. No scanner/module hook, seed substitution, or gate relaxation.
Numerical imports are lazy. Serial jobs support fresh exhaustive audit and
separately authorized repetition of that audit's exact ordered GPU trajectory.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import CodeType, FunctionType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
POLICY = "private-original-complete-bridge-device-loop-audit-v1"
HELPERS = ("_rigid", "_disagrees", "_strong", "_heldout", "_pair", "_matches",
    "_cache_matches", "_visual_witness", "_verify_bridge", "_independent_pairs",
    "_verify_partial_bridge", "_verify_visual_bridge")


class BridgeFailure(BaseException):
    """Latched research failure cannot become an ordinary rejected proposal."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def code_identity(code):
    return (code.co_code, code.co_names, code.co_varnames, code.co_freevars,
        code.co_cellvars, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_flags,
        tuple(code_identity(v) if isinstance(v, CodeType) else v for v in code.co_consts))


class OriginalVerifierGuard:
    """Check actual loaded code/default/namespace ownership against source."""
    def __init__(self, module):
        self.module, self.path = module, Path(module.__file__).resolve()
        self.digest = sha(self.path)
        compiled = compile(self.path.read_text(encoding="utf-8"), str(self.path),
            "exec", dont_inherit=True)
        codes = {v.co_name: v for v in compiled.co_consts if isinstance(v, CodeType)}
        self.functions = {}
        for name in HELPERS:
            fn = getattr(module, name)
            if (not isinstance(fn, FunctionType) or fn.__globals__ is not module.__dict__
                    or name not in codes or code_identity(fn.__code__) != code_identity(codes[name])):
                raise BridgeFailure("Loaded original verifier differs from source: "+name)
            self.functions[name] = (fn, fn.__code__, repr(fn.__defaults__), repr(fn.__kwdefaults__))
        self.dependencies = {name: getattr(module, name) for name in
            ("REG", "np", "motion", "correspondences", "feature_agreement", "MAX_MATCH_CACHE", "_match")}
        self.cpu_match = module._match
        self.native_functions = {name: getattr(module.REG, name) for name in
            ("evaluate_registration", "get_information_matrix_from_point_clouds",
             "registration_icp", "TransformationEstimationPointToPlane", "HuberLoss", "ICPConvergenceCriteria")}
        self.imported_functions = []
        for name in ("motion", "correspondences", "feature_agreement", "_match"):
            fn = self.dependencies[name]
            owner = sys.modules.get(fn.__module__)
            if (owner is None or getattr(owner, fn.__name__, None) is not fn
                    or fn.__globals__ is not owner.__dict__):
                raise BridgeFailure("Imported original dependency owner differs: "+name)
            path = Path(owner.__file__).resolve()
            compiled = compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
            codes = {v.co_name: v for v in compiled.co_consts if isinstance(v, CodeType)}
            if fn.__name__ not in codes or code_identity(fn.__code__) != code_identity(codes[fn.__name__]):
                raise BridgeFailure("Loaded imported function differs from original source: "+name)
            self.imported_functions.append((owner, path, sha(path), fn, fn.__code__,
                repr(fn.__defaults__), repr(fn.__kwdefaults__)))

    def check(self, *, source=False):
        if (sys.modules.get(self.module.__name__) is not self.module
                or Path(self.module.__file__).resolve() != self.path
                or source and sha(self.path) != self.digest):
            raise BridgeFailure("Original verifier source/module changed")
        for name, (fn, code, defaults, kwdefaults) in self.functions.items():
            if (getattr(self.module, name) is not fn or fn.__code__ is not code
                    or repr(fn.__defaults__) != defaults or repr(fn.__kwdefaults__) != kwdefaults):
                raise BridgeFailure("Original helper code/default ownership changed: "+name)
        if any(getattr(self.module, k) is not v for k, v in self.dependencies.items()):
            raise BridgeFailure("Original verifier dependency ownership changed")
        if any(getattr(self.module.REG, k) is not v for k, v in self.native_functions.items()):
            raise BridgeFailure("Original native registration/gate callable changed")
        for owner, path, digest, fn, code, defaults, kwdefaults in self.imported_functions:
            if (sys.modules.get(owner.__name__) is not owner or getattr(owner, fn.__name__, None) is not fn
                    or fn.__globals__ is not owner.__dict__ or fn.__code__ is not code
                    or repr(fn.__defaults__) != defaults or repr(fn.__kwdefaults__) != kwdefaults
                    or Path(owner.__file__).resolve() != path or source and sha(path) != digest):
                raise BridgeFailure("Original imported code/default/module alias changed: "+fn.__name__)
        return True


def descriptor(np, value):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise BridgeFailure("Only finite numeric arrays can enter bridge evidence")
    return {"dtype": array.dtype.str, "shape": list(array.shape), "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def input_binding(np, source, target, initial):
    return {"source": {k: descriptor(np, getattr(source, k)) for k in ("points", "normals", "colors")},
        "target": {k: descriptor(np, getattr(target, k)) for k in ("points", "normals", "colors")},
        "seed": descriptor(np, initial)}


def result_evidence(np, result, n, m):
    pose, pairs = np.asarray(result.transformation), np.asarray(result.correspondence_set)
    if (pose.dtype != np.float64 or pose.shape != (4, 4) or not np.isfinite(pose).all()
            or pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu"
            or len(np.unique(pairs[:, 0])) != len(pairs) or (pairs[:, 0] < 0).any()
            or (pairs[:, 0] >= n).any() or (pairs[:, 1] < 0).any() or (pairs[:, 1] >= m).any()):
        raise BridgeFailure("Malformed original/candidate ICP result")
    mapping = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))].astype(np.int32)
    fitness, rmse = float(result.fitness), float(result.inlier_rmse)
    if not math.isfinite(fitness) or not 0 <= fitness <= 1 or not math.isfinite(rmse) or rmse < 0:
        raise BridgeFailure("Nonfinite/out-of-domain ICP metrics")
    return {"transformation": pose.tolist(), "pose": descriptor(np, pose),
        "fitness": fitness, "inlier_rmse": rmse,
        "correspondence_mapping": descriptor(np, mapping),
        "raw_correspondences": descriptor(np, pairs)}


def result_shadow(np, native, candidate, n, m):
    a, b = result_evidence(np, native, n, m), result_evidence(np, candidate, n, m)
    delta = float(np.max(np.abs(np.asarray(a["transformation"])-np.asarray(b["transformation"]))))
    value = {"native": a, "candidate": b, "transform_max_abs_delta": delta,
        "fitness_abs_delta": abs(a["fitness"]-b["fitness"]),
        "rmse_abs_delta": abs(a["inlier_rmse"]-b["inlier_rmse"]),
        "correspondence_ids_equal": a["correspondence_mapping"] == b["correspondence_mapping"]}
    value["passed"] = value["correspondence_ids_equal"] and all(value[k] <= 1e-8 for k in
        ("transform_max_abs_delta", "fitness_abs_delta", "rmse_abs_delta"))
    return value


def semantic(np, value):
    """Small numeric values plus exact larger arrays; raw result order diagnostic."""
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value): raise BridgeFailure("Nonfinite gate evidence")
        return value
    if isinstance(value, np.generic): return semantic(np, value.item())
    if isinstance(value, np.ndarray):
        out = {"array": descriptor(np, value)}
        if value.size <= 64: out["values"] = value.tolist()
        return out
    if isinstance(value, (list, tuple)): return [semantic(np, v) for v in value]
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value): raise BridgeFailure("Unknown gate dictionary key")
        return {k: semantic(np, v) for k, v in value.items()}
    if all(hasattr(value, k) for k in ("transformation", "fitness", "inlier_rmse", "correspondence_set")):
        pose = np.asarray(value.transformation)
        pairs = np.asarray(value.correspondence_set)
        # Shape/index bounds are independently checked at the actual _match seam.
        mapping = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))].astype(np.int32)
        return {"registration_result": {"transformation": semantic(np, pose),
            "fitness": semantic(np, float(value.fitness)), "inlier_rmse": semantic(np, float(value.inlier_rmse)),
            "correspondence_mapping": descriptor(np, mapping)}}
    raise BridgeFailure("Unknown original gate result type: "+type(value).__name__)


class RegistrationObserver:
    def __init__(self, original, scope): self.original, self.scope = original, scope
    def __getattr__(self, name):
        fn = getattr(self.original, name)
        if name in ("evaluate_registration", "get_information_matrix_from_point_clouds"):
            return self.scope.observe("REG."+name, fn)
        return fn


class PrivateBridgeScope:
    """No global patches; each proposal owns its private helper namespace/caches."""
    def __init__(self, module, matcher, *, trace=None, guard=None):
        self.module, self.matcher = module, matcher
        self.guard = guard or OriginalVerifierGuard(module)
        self.trace = [] if trace is None else trace
        self.failure = None
        self.namespace = dict(module.__dict__)
        self.originals = {}
        for name in HELPERS:
            old = self.guard.functions[name][0]
            fn = FunctionType(old.__code__, self.namespace, old.__name__, old.__defaults__, old.__closure__)
            fn.__kwdefaults__ = old.__kwdefaults__
            self.originals[name] = fn
            self.namespace[name] = fn
        for name in HELPERS:
            self.namespace[name] = self.observe(name, self.originals[name])
        self.namespace["REG"] = RegistrationObserver(module.REG, self)
        self.namespace["_match"] = self.match
        self.private_objects = dict((k, self.namespace[k]) for k in HELPERS+("REG", "_match"))
        self.private_functions = []
        for fn in list(self.originals.values())+[self.namespace[k] for k in HELPERS]:
            self.private_functions.append((fn, fn.__code__, fn.__globals__, repr(fn.__defaults__),
                repr(fn.__kwdefaults__), tuple(c.cell_contents for c in fn.__closure__ or ())))

    def fail(self, error):
        if self.failure is None:
            self.failure = error if isinstance(error, BridgeFailure) else BridgeFailure(
                "Complete bridge research failed: "+repr(error))
            if self.failure is not error: self.failure.__cause__ = error
        raise self.failure

    def healthy(self):
        if self.failure is not None: self.fail(self.failure)
        self.guard.check()
        if any(self.namespace[k] is not v for k, v in self.private_objects.items()):
            self.fail(BridgeFailure("Private verifier namespace ownership changed"))
        for fn, code, namespace, defaults, kwdefaults, cells in self.private_functions:
            actual_cells = tuple(c.cell_contents for c in fn.__closure__ or ())
            if (fn.__code__ is not code or fn.__globals__ is not namespace
                    or repr(fn.__defaults__) != defaults or repr(fn.__kwdefaults__) != kwdefaults
                    or len(actual_cells) != len(cells) or any(a is not b for a, b in zip(actual_cells, cells))):
                self.fail(BridgeFailure("Private original/observer code/default/closure ownership changed"))

    def observe(self, name, fn):
        def observed(*args, **kwargs):
            self.healthy()
            row = {"index": len(self.trace), "name": name,
                "caller": sys._getframe(1).f_code.co_name, "complete": False}
            self.trace.append(row)
            started = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                row["wall_s_inclusive"] = time.perf_counter()-started
            except BridgeFailure as error:
                self.fail(error)
            except BaseException as error:
                row["original_exception"] = {"type": type(error).__name__, "message": str(error)}
                raise
            try:
                row["result"] = semantic(self.module.np, result)
                row["complete"] = True
                return result
            except BaseException as error:
                self.fail(error)
        return observed

    def match(self, source, target, initial):
        self.healthy()
        try: return self.matcher(source, target, initial)
        except BaseException as error: self.fail(error)

    def verify(self, source, target, initial, camera):
        self.guard.check(source=True)
        self.healthy()
        result = self.namespace["_verify_bridge"](source, target, initial, camera)
        self.healthy()
        self.guard.check(source=True)
        if any(not row.get("complete") for row in self.trace):
            self.fail(BridgeFailure("Original gate suffix/evidence incomplete"))
        return result


def private_fragments(source, target):
    """Exact clouds/features/poses shared; each actual View cache copied privately."""
    from dataclasses import replace
    aliases = {}
    def view(old):
        if id(old) not in aliases:
            aliases[id(old)] = replace(old, match_cache=dict(old.match_cache))
        return aliases[id(old)]
    def fragment(old):
        return replace(old, keys=[view(v) for v in old.keys],
            views=[view(v) for v in old.views], context=[view(v) for v in old.context])
    return fragment(source), fragment(target)


def original_pair_verdict(module, results):
    """Same ordered post-proposal ambiguity test used by the original fixture."""
    verified = [v for v in results if v is not None]
    ambiguous = bool(verified) and any(module._disagrees(verified[0]["transform"], v["transform"])
        for v in verified[1:])
    return {"proposal_count": len(results), "verified_proposals": len(verified),
        "ambiguous": bool(ambiguous), "accepted": bool(verified) and not ambiguous,
        "chosen_proposal": next((i for i, v in enumerate(results) if v is not None), None)
            if verified and not ambiguous else None}


def compare_evidence(native, candidate, *, tolerance=1e-8, path="", differences=None):
    """Discrete gates exact; small matrices numeric; information separate 1e-5."""
    differences = [] if differences is None else differences
    if isinstance(native, dict) and isinstance(candidate, dict):
        if set(native) != set(candidate): differences.append({"path": path, "reason": "keys"})
        elif "array" in native and "values" in native:
            a, b = native["array"], candidate["array"]
            if (a["dtype"], a["shape"], a["nbytes"]) != (b["dtype"], b["shape"], b["nbytes"]):
                differences.append({"path": path, "reason": "matrix descriptor"})
            compare_evidence(native["values"], candidate["values"], tolerance=tolerance,
                path=path+".values", differences=differences)
        else:
            for key in native:
                compare_evidence(native[key], candidate[key],
                    tolerance=1e-5 if key == "information" or "get_information_matrix" in path else tolerance,
                    path=path+"."+key, differences=differences)
    elif isinstance(native, list) and isinstance(candidate, list):
        if len(native) != len(candidate): differences.append({"path": path, "reason": "length"})
        else:
            for i, (a, b) in enumerate(zip(native, candidate)):
                compare_evidence(a, b, tolerance=tolerance, path=f"{path}[{i}]", differences=differences)
    elif type(native) is float and type(candidate) is float:
        delta = abs(native-candidate)
        if not math.isfinite(delta) or delta > tolerance:
            differences.append({"path": path, "reason": "numeric", "absolute_delta": delta if math.isfinite(delta) else None})
    elif type(native) is not type(candidate) or native != candidate:
        differences.append({"path": path, "reason": "discrete or exact array"})
    return differences
