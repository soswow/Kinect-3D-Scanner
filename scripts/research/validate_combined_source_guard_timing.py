"""New registry-owned authority for an equivalent source-check-only sibling.

The inherited current all-nine timing authority certifies unchanged numerical
work. Extra current sources bind only full-source guard memoization and nested
host bookkeeping measurements. No whole-Finish/production authority exists.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from scripts.research import validate_combined_sync_timing as parent
from scripts.research.archive.validate_uniform_grid_proof import require,file_hash

ROOT=Path(__file__).resolve().parents[2]
MODES=("original-ast","whole-file-sha")
POLICY="original-combined-equivalent-source-guard-component-v1"
ARTIFACTS=("scripts/research/combined_source_guard_adapter.py",
    "scripts/research/validate_combined_source_guard_timing.py",
    "scripts/research/benchmark_combined_source_guard_timing.py",
    "scripts/research/benchmark_combined_source_guard.py",
    "tests/test_combined_source_guard_contract.py")
SOURCE="scripts/research/archive/research_resident_icp.py"


@dataclass(frozen=True)
class SourceGuardTimingAuthority:
    original_timing_authority:parent.CombinedSyncTimingAuthority
    artifact_sha256:tuple
    source_check_mode:str
    original_source_sha256:str
    source_contract_json:str
    policy:str=POLICY

_ACTIVE={}


def validate_source_guard_authority(original,mode,expected_artifacts):
    require(type(original) is parent.CombinedSyncTimingAuthority,"Require original current timing authority")
    require(mode in MODES,"Unknown source-guard timing mode")
    runtime=original.runtime_binding
    parent.validate_activation(original,original.device_authority,runtime["resident_configuration"],
        {key:runtime["cache_policy"][key] for key in ("max_clouds","retained_gpu_bytes")})
    pins=parent.device.normalized_artifacts(expected_artifacts)
    require(set(pins)==set(ARTIFACTS) and all(file_hash(ROOT/name)==digest for name,digest in pins.items()),
        "Current separate source-guard sibling sources differ")
    expected=dict(original.combined_authority.artifact_sha256)[SOURCE]
    require(file_hash(ROOT/SOURCE)==expected,"Original whole source bytes differ")
    token=SourceGuardTimingAuthority(original,tuple(sorted(pins.items())),mode,expected,
        json.dumps(runtime["combined_sync_contract"],sort_keys=True,separators=(",",":"),allow_nan=False))
    while len(_ACTIVE)>=4:del _ACTIVE[next(iter(_ACTIVE))]
    _ACTIVE[id(token)]=token
    return token


def validate_source_guard_activation(token,original,mode):
    require(type(token) is SourceGuardTimingAuthority and _ACTIVE.get(id(token)) is token
            and token.original_timing_authority is original and token.source_check_mode==mode and token.policy==POLICY,
        "Only a newly validated exact source-guard sibling token can activate")
    require(all(file_hash(ROOT/name)==digest for name,digest in token.artifact_sha256)
            and file_hash(ROOT/SOURCE)==token.original_source_sha256,
        "Current source-guard/whole-original bytes changed before activation")
    return token
