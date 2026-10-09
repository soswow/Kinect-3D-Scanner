"""Offline marker proposals through every original fragment bridge authority.

The only body insertion extends proposals after the original TOP4+two RANSAC
seeds and before original uniqueness/verification. Pair ranking, frontier,
budgets, witnesses, ambiguity handling and graph optimization stay original.
No numerical runtime is imported by this module.
"""

from __future__ import annotations
import ast
import copy
import hashlib
from pathlib import Path
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[2]
# v2 binds the unchanged marker insertion to the temporal-boundary production
# algorithm. Historical v1 measurements remain historical; new runs must emit
# this policy and fresh source fingerprints rather than reuse those reports.
POLICY = "native-marker-additional-global-fragment-proposals-v2"
ORIGINAL_AST_SHA256 = "bca7ccfddacd150328bd1df59bd6e8e4c93dad557fb626529f1a6861c13a998f"
PROVIDER_SYMBOL = "_offline_global_marker_seed_provider"
MAX_ADDITIONAL_SEEDS = 4
MAX_KEYS_PER_FRAGMENT = 18


class GlobalMarkerFailure(BaseException):
    """Research failure that original ordinary registration rejection cannot hide."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ast_hash(node):
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def code_signature(code):
    def value(item):
        if hasattr(item,"co_code"):
            return code_signature(item)
        if isinstance(item,tuple):
            return tuple(value(n) for n in item)
        if isinstance(item,float):
            return ("float",item.hex())
        return (type(item).__name__,repr(item))
    return (code.co_code,code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars,
            code.co_argcount,code.co_kwonlyargcount,code.co_posonlyargcount,code.co_flags,
            tuple(value(item) for item in code.co_consts))


def enhanced_function(original, source_path=None):
    """Compile one pinned insertion; deleting it must reconstruct the whole body."""
    source_path = ROOT/"scanner_server/fragments.py" if source_path is None else Path(source_path)
    source = source_path.read_bytes()
    tree = ast.parse(source.decode("utf-8"))
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                    and node.name == "propose_fragment_poses")
    require(ast_hash(function) == ORIGINAL_AST_SHA256, "Original entire fragment proposal body changed")
    original_code = compile(tree,str(source_path),"exec",dont_inherit=True)
    expected = next(item for item in original_code.co_consts
                    if hasattr(item,"co_name") and item.co_name == function.name)
    require(isinstance(original,FunctionType) and original.__module__ == "scanner_server.fragments"
            and original.__closure__ is None and code_signature(original.__code__) == code_signature(expected),
            "Require the loaded unwrapped original fragment proposal function")
    require(not any(isinstance(node,ast.Name) and node.id == PROVIDER_SYMBOL for node in ast.walk(tree)),
            "Research provider symbol already belongs to original source")
    parents = []
    for node in ast.walk(function):
        for field,value in ast.iter_fields(node):
            if not isinstance(value,list):
                continue
            for index,child in enumerate(value):
                if (isinstance(child,ast.For) and isinstance(child.target,ast.Name) and child.target.id == "seed"
                    and any(isinstance(call,ast.Call) and isinstance(call.func,ast.Name)
                            and call.func.id == "_global_seed" for call in ast.walk(child))):
                    parents.append((value,index,child))
    require(len(parents) == 1, "Original two-RANSAC proposal seam is not unique")
    body,index,loop = parents[0]
    require(index+1 < len(body) and isinstance(body[index+1],ast.Assign)
            and len(body[index+1].targets) == 1 and isinstance(body[index+1].targets[0],ast.Name)
            and body[index+1].targets[0].id == "verified", "Original verification must follow its RANSAC proposals")
    insertion = ast.parse("proposals.extend("+PROVIDER_SYMBOL+"(source,target,engine.settings.camera))").body[0]
    ast.copy_location(insertion,loop)
    body.insert(index+1,insertion)
    mutated_hash = ast_hash(function)
    removed = body.pop(index+1)
    require(ast_hash(function) == ORIGINAL_AST_SHA256, "Deleting the insertion did not reproduce the entire original body")
    body.insert(index+1,removed)
    compiled = compile(ast.fix_missing_locations(tree),str(source_path),"exec",dont_inherit=True)
    code = next(item for item in compiled.co_consts if hasattr(item,"co_name") and item.co_name == function.name)
    enhanced = FunctionType(code,original.__globals__,original.__name__,original.__defaults__)
    enhanced.__kwdefaults__ = original.__kwdefaults__
    enhanced.__annotations__ = dict(original.__annotations__)
    enhanced.__qualname__ = original.__qualname__
    return enhanced,{"source_path":str(source_path),"source_sha256":hashlib.sha256(source).hexdigest(),
        "original_ast_sha256":ORIGINAL_AST_SHA256,"enhanced_ast_sha256":mutated_hash,
        "removal_reconstructs_entire_original":True,
        "insertion":"Only proposals.extend(provider(source,target,engine.settings.camera)) after both original global RANSAC proposals"}


class OriginalMarkerGlobalProvider:
    """Rank additional camera-pair marker PnP seeds; original gates accept poses."""
    def __init__(self, native_provider, fragments):
        self.native,self.np = native_provider,native_provider.np
        self.rigid,self.disagrees = fragments._rigid,fragments._disagrees
        self.calls,self.rows = 0,[]

    def validate_seed(self, pose):
        require(getattr(pose,"shape",None) == (4,4) and self.np.isfinite(pose).all() and self.rigid(pose),
                "Marker fragment proposal is not an original finite rigid transform")

    def seed_hash(self, pose):
        return self.native.array_hash(pose)

    def global_seeds(self, source, target, camera):
        require(camera is self.native.engine.settings.camera, "Marker camera differs from original native measured calibration")
        require(0 < len(source.keys) <= MAX_KEYS_PER_FRAGMENT and 0 < len(target.keys) <= MAX_KEYS_PER_FRAGMENT,
                "Original fragment camera witnesses exceed marker preparation bound")
        started = time.perf_counter()
        self.calls += 1
        record = {"source_fragment":source.index,"target_fragment":target.index,"camera_pairs":[],"offered":[]}
        self.rows.append(record)
        # Features/IDs are private bounded marker buffers; original SIFT views
        # are untouched. Ranking never enters original candidate-pair ranking.
        candidates = []
        for a in source.keys:
            _,left = self.native._features(a.index)
            for b in target.keys:
                if a.index == b.index:
                    continue
                _,right = self.native._features(b.index)
                common = len(left.keys() & right.keys())
                if common >= 40:
                    candidates.append((-common,a.index,b.index,a,b))
        candidates.sort(key=lambda row: row[:3])
        poses = []
        for negative_count,_,_,a,b in candidates:
            before = [self.native.array_hash(a.pose),self.native.array_hash(b.pose),
                      self.native.features_hash(a.features),self.native.features_hash(b.features)]
            marker = self.native.proposal(a,b,camera)
            pair = {"source_index":a.index,"target_index":b.index,"common_identity_corners":-negative_count,
                    "original_marker_pnp_present":marker is not None}
            record["camera_pairs"].append(pair)
            if marker is not None:
                self.validate_seed(marker)
                # a maps source-camera -> source-fragment; b maps target-camera
                # -> target-fragment. No Live/world/archived pose is a seed.
                pose = b.pose @ marker @ self.np.linalg.inv(a.pose)
                self.validate_seed(pose)
                pair.update(marker_camera_seed=self.seed_hash(marker),fragment_seed=self.seed_hash(pose))
                if not any(not self.disagrees(pose,previous,.01,1) for previous in poses):
                    poses.append(pose)
                    record["offered"].append(dict(pair))
                else:
                    pair["duplicate_original_uniqueness_rule"] = True
            after = [self.native.array_hash(a.pose),self.native.array_hash(b.pose),
                     self.native.features_hash(a.features),self.native.features_hash(b.features)]
            require(before == after,"Marker proposal changed original local pose/SIFT witness bytes")
            if len(poses) == MAX_ADDITIONAL_SEEDS:
                break
        record.update(complete=True,wall_s=time.perf_counter()-started,offered_count=len(poses))
        return poses

    def report(self):
        return {"policy":POLICY,"calls":self.calls,"max_additional_seeds":MAX_ADDITIONAL_SEEDS,
            "ordering":"Descending shared decoded ID/canonical-corner count, then original raw view indices",
            "authority":"Proposal only; original full bridge, independent-camera support and ambiguity gates are unchanged",
            "rows":self.rows,"native_provider":self.native.report(),
            "timing_scope":"Nested wall_s includes lazy native decoder/projection/CPU-shadowed CUDA lookup and original PnP; do not add overlapping native provider timers"}


class GlobalMarkerProposalScope:
    """Enter before original output-only observers; restore every module seam."""
    def __init__(self, fragments, provider, *, trace=None):
        self.module,self.provider,self.trace = fragments,provider,trace
        self.original = fragments.propose_fragment_poses
        self.failure,self.entered,self.active,self.restored = None,False,False,True
        self.rows,self.cleanup_failures,self.binding = [],[],None

    def fail(self, message, cause=None):
        if self.failure is None:
            self.failure = GlobalMarkerFailure(message)
            if cause is not None:
                self.failure.__cause__ = cause
        raise self.failure

    def healthy(self):
        if self.failure is not None:
            raise self.failure

    def check_budgets(self):
        require(all(type(self.module.__dict__.get(name)) is int and self.module.__dict__[name] == value
                    for name,value in (("MAX_PAIRS",256),("MAX_FRAGMENTS",32),("MAX_FRAGMENT_VIEWS",16))),
                "Original fragment/ranking budgets changed")

    def call(self, source, target, camera):
        self.healthy()
        row = {"source_fragment":source.index,"target_fragment":target.index,"complete":False}
        self.rows.append(row)
        try:
            self.check_budgets()
            result = self.provider.global_seeds(source,target,camera)
            self.check_budgets()
            require(isinstance(result,list) and len(result) <= MAX_ADDITIONAL_SEEDS,
                    "Marker provider exceeded the bounded additional proposal domain")
            for pose in result:
                self.provider.validate_seed(pose)
            row.update(complete=True,offered_count=len(result),seeds=[self.provider.seed_hash(pose) for pose in result])
            if self.trace is not None:
                self.trace(row)
            return result
        except GlobalMarkerFailure:
            raise
        except BaseException as error:
            row["failure"] = {"type":type(error).__name__,"message":str(error)}
            self.fail("Global marker proposal/evidence failed",error)

    def restore(self):
        actions = [lambda: setattr(self.module,"propose_fragment_poses",self.entry_function)]
        if self.had_symbol:
            actions.append(lambda: self.module.__dict__.__setitem__(PROVIDER_SYMBOL,self.previous_symbol))
        else:
            actions.append(lambda: self.module.__dict__.pop(PROVIDER_SYMBOL,None))
        errors = []
        for action in actions:
            try:
                action()
            except BaseException as error:
                errors.append(error)
                self.cleanup_failures.append({"type":type(error).__name__,"message":str(error)})
        self.active,self.restored = False,not errors
        return errors

    def __enter__(self):
        if self.entered:
            self.fail("Global marker scope cannot be reused or nested")
        self.healthy()
        self.entered = True
        self.entry_function = self.module.propose_fragment_poses
        self.had_symbol = PROVIDER_SYMBOL in self.module.__dict__
        self.previous_symbol = self.module.__dict__.get(PROVIDER_SYMBOL)
        try:
            require(self.module.propose_fragment_poses is self.original and
                    self.original.__globals__ is self.module.__dict__,
                    "Enter global marker scope before original gate observers or other proposal patches")
            self.check_budgets()
            function,self.binding = enhanced_function(self.original)
            self.module.__dict__[PROVIDER_SYMBOL] = self.call
            self.module.propose_fragment_poses = function
            self.active,self.restored = True,False
        except BaseException as error:
            for secondary in self.restore():
                error.add_note(f"Global marker partial-entry cleanup also failed: {secondary}")
            self.fail("Global marker source guard or installation failed",error)
        return self

    def __exit__(self, kind, error, traceback):
        errors = self.restore()
        if self.failure is not None:
            for secondary in errors:
                self.failure.add_note(f"Global marker restore also failed: {secondary}")
            raise self.failure
        if errors:
            self.failure = GlobalMarkerFailure("Global marker restoration failed")
            self.failure.__cause__ = error if error is not None else errors[0]
            for secondary in errors:
                self.failure.add_note(f"Global marker restore also failed: {secondary}")
            raise self.failure
        return False

    def report(self):
        return {"policy":POLICY,"source_binding":self.binding,"hooks_restored":self.restored,
            "failure":None if self.failure is None else {"type":type(self.failure).__name__,"message":str(self.failure)},
            "cleanup_failures":self.cleanup_failures,"provider_calls":len(self.rows),"rows":self.rows,
            "scope":"Original TOP4+two RANSAC, pair ranking/frontier/budgets/unique/verification/witness/ambiguity/graph body unchanged; <=4 additional marker seeds only",
            "provider":self.provider.report(),"geometry_quality_proven":False,"performance_measured":False}
