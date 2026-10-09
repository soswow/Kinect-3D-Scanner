"""Complete-proposal sibling with an owned immutable audit reference.

Windows timing holds the closed proof against write/delete until final stream
and cache completion. Decoded references are immutable and reused. Original
ICP math, actual-input checks, native shadows and gate consumers are unchanged;
the performance hypothesis concerns Python proof ownership/I/O, not CUDA math.
Fresh v2 audit and separately registered v2 timing are required.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import microbatch_bridge_driver as original

ORIGINAL_PATH = ROOT/"scripts/research/microbatch_bridge_driver.py"
ORIGINAL_SHA256 = "d048525610ac41d4b591ed5779d3d01ad06876f786d662fe067fcea8e6d2c2ec"
KIND = "gpu-icp-complete-bridge-owned-proof-audit-v2"
TIMING_KIND = "gpu-icp-complete-bridge-owned-proof-timing-v2"
NEW_FILES = ("scripts/research/microbatch_bridge_owned_driver.py",
    "scripts/research/device_loop_owned_workspace.py", "scripts/research/microbatch_bridge_owned_protocol.py",
    "tests/test_microbatch_bridge_owned_driver.py", "tests/test_device_loop_owned_workspace.py",
    "tests/test_microbatch_bridge_owned_protocol.py")
OLD_FILES = ("scripts/research/microbatch_bridge_driver.py", "scripts/research/microbatch_bridge_scope.py",
    "scripts/research/device_loop_workspace.py", "scripts/research/device_loop_workspace_protocol.py",
    "scripts/research/microbatch_bridge_protocol.py", "tests/test_microbatch_bridge.py",
    "tests/test_device_loop_workspace.py", "tests/test_microbatch_bridge_protocol.py")


def require(value, message):
    if not value: raise RuntimeError(message)


def ast_text(value): return ast.dump(value, include_attributes=False)
def ast_sha(value): return hashlib.sha256(ast_text(value).encode()).hexdigest()


def derive():
    require(Path(original.__file__).resolve() == ORIGINAL_PATH
        and original.scope.sha(ORIGINAL_PATH) == ORIGINAL_SHA256,
        "Only the exact measured v1 complete-proposal producer may be derived")
    source = ast.parse(ORIGINAL_PATH.read_text(encoding="utf-8"))
    nodes = [copy.deepcopy(node) for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    tree = ast.Module(body=nodes, type_ignores=[])
    before = ast_text(tree)
    replacement = {"device_loop_workspace": "device_loop_owned_workspace",
        "microbatch_bridge_protocol": "microbatch_bridge_owned_protocol"}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "scripts.research":
            for item in node.names:
                if item.name in replacement:
                    imports.append(item.name); item.name = replacement[item.name]
    require(sorted(imports) == sorted(replacement), "Exactly the two owned authorization imports may change")
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    run.body.insert(0, ast.parse("permit = None").body[0])
    binding = next(node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == "binding")
    result = next(node.value for node in binding.body if isinstance(node, ast.Return))
    result.keys.extend([ast.Constant("owned_proof_source"), ast.Constant("owned_driver_source")])
    result.values.extend([ast.parse("protocol.source_contract()", mode="eval").body,
        ast.parse("source_contract()", mode="eval").body])
    boundary = next(node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == "boundary")
    boundary.body.insert(0, ast.parse("_check_owned_loaded()").body[0])
    outer = next(node for node in run.body if isinstance(node, ast.Try))
    position = next(i for i, node in enumerate(outer.finalbody) if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript) and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "report" and isinstance(node.targets[0].slice, ast.Constant)
        and node.targets[0].slice.value == "owner_cleanup_wall_s")
    close = ast.parse('''if permit is not None:
    clean("owned proof read-lock closure", lambda: protocol.close_permit(permit, primary=primary))
    report["owned_proof"] = clean("owned proof diagnostic closure", lambda: protocol.permit_report(permit))
else:
    report["owned_proof"] = {"required": False}
''').body[0]
    outer.finalbody.insert(position, close)
    status = next(node for node in outer.finalbody if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.IfExp) and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].slice, ast.Constant) and node.targets[0].slice.value == "status")
    owned_closed = ast.parse('permit is None or (report.get("owned_proof", {}).get("closed") is True and report.get("owned_proof", {}).get("failure") is None)', mode="eval").body
    require(isinstance(status.value.test, ast.BoolOp), "Original final success predicate changed")
    status.value.test.values.append(owned_closed)
    # Reverse every explicit metadata/ownership edit and prove that all original
    # numerical, gate, cleanup, ordering and timing statements remain present.
    recovered = copy.deepcopy(tree)
    for node in ast.walk(recovered):
        if isinstance(node, ast.ImportFrom) and node.module == "scripts.research":
            for item in node.names:
                for old, new in replacement.items():
                    if item.name == new: item.name = old
    restored_run = next(node for node in recovered.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    require(ast_text(restored_run.body.pop(0)) == ast_text(ast.parse("permit = None").body[0]), "Owned permit initialization changed")
    restored_binding = next(node for node in ast.walk(restored_run) if isinstance(node, ast.FunctionDef) and node.name == "binding")
    restored_result = next(node.value for node in restored_binding.body if isinstance(node, ast.Return))
    del restored_result.keys[-2:]; del restored_result.values[-2:]
    restored_boundary = next(node for node in ast.walk(restored_run) if isinstance(node, ast.FunctionDef) and node.name == "boundary")
    require(ast_text(restored_boundary.body.pop(0)) == ast_text(ast.parse("_check_owned_loaded()").body[0]), "Owned loaded-code boundary changed")
    restored_outer = next(node for node in restored_run.body if isinstance(node, ast.Try))
    require(ast_text(restored_outer.finalbody.pop(position)) == ast_text(close), "Owned proof finalization changed")
    restored_status = next(node for node in restored_outer.finalbody if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.IfExp) and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].slice, ast.Constant) and node.targets[0].slice.value == "status")
    require(ast_text(restored_status.value.test.values.pop()) == ast_text(owned_closed), "Owned success predicate changed")
    require(ast_text(recovered) == before, "Owned derivation changed an original numerical/control statement")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__dict__)
    namespace.update(__name__=__name__, __file__=__file__, __doc__=__doc__, ROOT=ROOT,
        KIND=KIND, TIMING_KIND=TIMING_KIND, OWN_FILES=NEW_FILES+OLD_FILES,
        source_contract=source_contract, _check_owned_loaded=_check_owned_loaded)
    exec(compile(tree, __file__, "exec", dont_inherit=True), namespace)
    return namespace, {"original_sha256": ORIGINAL_SHA256, "original_ast_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "derived_ast_sha256": ast_sha(tree), "inverse_ast_exact": True,
        "changes": "Two authorization imports, version/source pins, owned proof closure and loaded-code checks only",
        "original_math_and_gates_unchanged": True, "new_cuda_math": False}


def source_contract():
    return {"policy": "complete-original-proposals-owned-immutable-proof-v2",
        "derivation": dict(DERIVATION), "artifacts": {name: original.scope.sha(ROOT/name) for name in NEW_FILES+OLD_FILES},
        "performance_hypothesis": "Remove repeated proof-file hashing and JSON parsing; original CUDA math unchanged"}


def _check_owned_loaded():
    require(original.scope.sha(ORIGINAL_PATH) == ORIGINAL_SHA256, "Held producer source changed")
    for owner, name, function, code, defaults, keywords in _LOADED:
        require(getattr(owner, name) is function and function.__code__ is code
            and function.__globals__ is _NAMESPACE and repr(function.__defaults__) == defaults
            and repr(function.__kwdefaults__) == keywords, "Derived loaded helper/control owner changed")
        if owner is sys.modules[__name__]:
            require(_NAMESPACE.get(name) is function, "Derived private function alias changed")
        else:
            require(_NAMESPACE.get(owner.__name__) is owner and globals().get(owner.__name__) is owner,
                "Derived private class alias changed")


_NAMESPACE, DERIVATION = derive()
_LOADED = []
for _name, _value in tuple(_NAMESPACE.items()):
    if isinstance(_value, FunctionType) and _value.__globals__ is _NAMESPACE:
        _LOADED.append((sys.modules[__name__], _name, _value, _value.__code__, repr(_value.__defaults__), repr(_value.__kwdefaults__)))
        globals()[_name] = _value
    elif isinstance(_value, type) and _value.__module__ == __name__:
        globals()[_name] = _value
        for _method, _function in _value.__dict__.items():
            if isinstance(_function, FunctionType):
                _LOADED.append((_value, _method, _function, _function.__code__, repr(_function.__defaults__), repr(_function.__kwdefaults__)))
_LOADED = tuple(_LOADED)


if __name__ == "__main__": run(parse())
